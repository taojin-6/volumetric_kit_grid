// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file dct_transform.hpp
/// @brief The codec's GPU block transform: a grid's blocks to quantized 8^3
///        DCT coefficients plus an observed-voxel mask, and back.
///
/// Internal (under src/, never installed): the `Encoder` and `Decoder` share
/// it, and nothing outside the tier should build a frame from raw coefficients
/// -- that is the transform-only public API the 2026-09-26 decision declined
/// to repeat. It carries `VR_CODEC_API` anyway, so the tier's own test links
/// against it in a shared-library build too.

#include <cstdint>
#include <memory>

#include "volumetric_kit/core/base/result.hpp"
#include "volumetric_kit/core/vulkan/allocator.hpp"
#include "volumetric_kit/core/vulkan/buffer.hpp"
#include "volumetric_kit/core/vulkan/compute_kernel.hpp"
#include "volumetric_kit/core/vulkan/descriptor.hpp"
#include "volumetric_kit/core/vulkan/vulkan.hpp"
#include "volumetric_kit/recon/codec/codec_params.hpp"
#include "volumetric_kit/recon/codec/export.hpp"
#include "volumetric_kit/recon/core/dispatch.hpp"
#include "volumetric_kit/recon/core/fwd.hpp"
#include "volumetric_kit/recon/volume/hash_types.hpp"
#include "volumetric_kit/recon/volume/voxel_block_grid.hpp"

namespace volumetric_kit::recon::codec::detail {

/// The weight the inverse writes on an observed voxel: anything at or above
/// @ref volume::kObservedWeight meshes the same, and a decoded grid is not
/// fused into. Pushed to the kernel, so this is its one definition.
inline constexpr float kDecodedWeight = 1.0f;

/// @brief The words one entry's @p k coefficients take on the device: two
///        int16 a word, each entry starting a word of its own, so an odd K
///        pads each entry by a half-word (`coefficient_words` in
///        dct_common.glsl).
constexpr std::uint32_t coefficient_words(std::uint32_t k) {
  return k / 2 + k % 2;
}

/// @brief Blocks where the device holds them: @ref count entries of masks
///        and coefficients, and their block list, in the layout the kernels
///        write. Borrowed from what produced them -- the transform's
///        @ref DctTransform::record_forward (whose buffers are its caller's),
///        the device frame reader or the reference store -- and valid until
///        that object's next call.
///
/// A P-frame's entries are its stream positions (bitstream.hpp): its
/// @ref reference_count reference entries, each with its mode, then its new
/// blocks, which alone the list holds.
struct ResidentBlocks {
  /// `volume::BlockIndex` entries in frame order, one per block entry:
  /// `count - reference_count`.
  const core::Buffer* list = nullptr;
  const core::Buffer* masks =
      nullptr;  ///< `kMaskWordsPerBlock` words an entry.
  /// Two int16 a word, coefficient 2p low; @ref coefficient_words an entry.
  const core::Buffer* coefficients = nullptr;
  /// A P-frame's mode for each reference entry, a u32 each
  /// (`frame_models.glsl`'s `kMode*`); null without reference entries.
  const core::Buffer* modes = nullptr;
  /// The device frame reader's and the reference store's diff: for each
  /// reference entry e, and at e = M, how many of the entries before it
  /// survive, a u32 each; null elsewhere.
  const core::Buffer* ranks = nullptr;
  std::uint32_t count = 0;
  std::uint32_t coefficient_count = 0;  ///< K.
  /// M, the entries before the blocks: 0 for an intra frame.
  std::uint32_t reference_count = 0;
  /// Set only by @ref ReferenceStore::record_diff: @ref count bounds the
  /// entries at M plus every block of the frame, of which `ranks[M]`
  /// survive and are not new, so the frame's entries are the first
  /// `count - ranks[M]`.
  bool bounded = false;
};

/// @brief The forward and inverse block DCT, as two GLSL compute kernels, a
///        third that finds the blocks to code, and the true SDF passes before
///        it (`EncoderConfig::true_sdf`).
///
/// One workgroup of `kEdge^2` (64) invocations per listed block; each owns one
/// `kEdge`-voxel line per separable pass. The forward kernel reads `weight`,
/// and `tsdf` where the voxel is observed (`weight >=
/// volume::kObservedWeight`), divides that SDF by `trunc_dist` (so its input is
/// in [-1, 1]), and fills each unobserved voxel from the nearest observed one
/// along x, then y, then z (the lower index on a tie), so the transform sees a
/// smooth continuation of the field rather than a step to 0 at the mask edge.
/// It then transforms and keeps the first `coefficient_count` coefficients in
/// 3-D zigzag order, each rounded half-to-even after dividing by its step and
/// clamped to ±@ref kMaxQuantizedMagnitude, and packs the observed bits into
/// the mask.
///
/// The inverse undoes it: dequantizes, zero-fills the dropped coefficients,
/// inverse-transforms, and writes each observed voxel's `tsdf` (clamped back to
/// `±trunc_dist`, the integrator's invariant) with weight
/// @ref kDecodedWeight, and each unobserved voxel as a freshly allocated block
/// holds it -- zero `tsdf`, zero `weight` -- so the mesher uses exactly the
/// voxels the encoder saw observed.
///
/// Output is indexed by **list position**, not by `ptr`: entry `i`'s mask and
/// coefficients are the list's entry `i`'s block. The order the blocks travel
/// in is the caller's (the encoder sorts them on the device, @ref BlockLists).
///
/// Each workgroup first finds its entry's block by its `coord`, through the
/// grid's hash table, and never reads its `ptr`: so no ptr is trusted, and the
/// decoder can list blocks it has just allocated without reading their slots
/// back. A coord the table does not hold is neither read nor written, and the
/// call reports how many there were. That costs one probe per block, where a
/// host scan would be O(count) per call. Like the mesher's probe, it needs the
/// map quiescent: no allocate may run into the grid during a call.
///
/// Every buffer is device-local, and no block list, coefficient or mask
/// leaves the device: the forward reads its list where the caller holds it,
/// is recorded into the caller's batch and writes its output where the
/// caller says (the encoder's reference store, from which the device frame
/// writer codes it), and the inverse reads its input where the reference
/// store's apply wrote it, in the caller's batch too. Nothing is read back:
/// the forward and the inverse leave their reject count on the device for
/// the caller's results (@ref rejected), and the observed filter its count
/// and reject tally (@ref observed_counts), which size the encoder's later
/// passes there.
/// The observed list, the cut set and the keep flags grow when needed and
/// remain allocated for later calls. The coefficients travel two to a word,
/// 16 bits each.
///
/// @warning The `Device` and `Allocator` passed to @ref create must
///          outlive this object; it stores references to them.
class VR_CODEC_API DctTransform {
 public:
  /// @brief Build both kernels and upload the basis and zigzag tables.
  /// @param device     The compute device (must outlive this object).
  /// @param allocator  The allocator the scratch buffers come from (must
  ///                   outlive this object).
  /// @return The transform, or a non-OK `Status` if a pipeline, the pool or
  ///         the table upload fails.
  static core::Result<std::unique_ptr<DctTransform>> create(
      core::Device& device, core::Allocator& allocator);

  ~DctTransform() = default;
  DctTransform(const DctTransform&) = delete;
  DctTransform& operator=(const DctTransform&) = delete;

  /// @brief Record the transform and quantization of the first `n` blocks of
  ///        @p list into @p batch, `n` the word @p count names, its output
  ///        into @p masks and @p coefficients by list position.
  ///
  /// The groups are the count's, a block each, sized on the device; every
  /// buffer is bound for the bound. Once @p batch has run, @ref rejected
  /// holds the entries the kernel found no block for, which
  /// @ref check_rejected turns into a refusal.
  /// @param batch         The batch to record into; submitted by the caller.
  ///                      @ref record_steps for @p params must be recorded
  ///                      into it before.
  /// @param grid          The grid the list names; must carry 4-byte `tsdf`
  ///                      and `weight` attributes and have `block_size` 8.
  /// @param list          `volume::BlockIndex` entries on the device, found
  ///                      by `coord` (each `ptr` is ignored); must outlive
  ///                      the output.
  /// @param count         The entries to transform: a word a kernel wrote,
  ///                      at most its bound.
  /// @param params        The coefficient count and steps.
  /// @param masks         Where each entry's `kMaskWordsPerBlock` mask words
  ///                      go; must outlive the output.
  /// @param coefficients  Where each entry's @ref coefficient_words(K)
  ///                      words go, an odd K's pad half 0; must outlive the
  ///                      output.
  /// @param cut           Optional cut set, one bit per voxel of the grid's
  ///                      heap (slot s at bit s % 32 of word s / 32), whose
  ///                      set voxels are coded as unobserved
  ///                      (`EncoderConfig::true_sdf`); must outlive the
  ///                      submit. `nullptr` reads none.
  /// @param stage         Optional scope the caller's stage row is open
  ///                      under; the dispatches record their device spans
  ///                      into it. `nullptr` times nothing.
  /// @return The output, borrowing @p list, @p masks and @p coefficients,
  ///         its count the bound (no entries for a bound of 0, which records
  ///         the reject count's reset alone), or, before anything is
  ///         recorded, `Status::Code::InvalidArgument` for invalid @p params,
  ///         a grid that is moved-from, has another block size, a
  ///         non-positive `trunc_dist`, or lacks a float `tsdf` / `weight`; a
  ///         count word missing, misaligned or past its buffer; a bound past
  ///         the grid's block heap; a @p list `StorageInput::check` refuses
  ///         for it; output buffers smaller than it needs; a @p cut smaller
  ///         than the grid's heap; or a buffer that would exceed
  ///         `maxStorageBufferRange`. Otherwise a buffer failure.
  core::Result<ResidentBlocks> record_forward(
      core::CommandBatch& batch, const volume::VoxelBlockGrid& grid,
      const core::Buffer& list, const volume::DeviceCount& count,
      const CodecParams& params, const core::Buffer& masks,
      const core::Buffer& coefficients, const core::Buffer* cut = nullptr,
      core::GpuStageScope* stage = nullptr);

  /// @return The refusal for @p rejected entries of @p count the kernels
  ///         found no block for, or OK when there are none.
  static core::Status check_rejected(const char* op, std::uint32_t rejected,
                                     std::uint32_t count);

  /// @return OK for a grid the transform takes, as @ref record_observed
  ///         checks it; its refusal otherwise: what a caller checks before
  ///         it records anything else.
  core::Status check_grid(const volume::VoxelBlockGrid& grid) const;

  /// @brief Record the upload of @p params' steps, which the forward and the
  ///        inverse read, into @p batch: inline, so a caller records it ahead
  ///        of its batch's first dispatch, and MoltenVK opens no blit encoder
  ///        mid-batch.
  /// @return OK, or the batch's refusal.
  core::Status record_steps(core::CommandBatch& batch,
                            const CodecParams& params);

  /// @brief Record the reconstruction of every block of @p in from its
  ///        coefficients and mask, overwriting its `tsdf` and `weight`, into
  ///        @p batch after what it already holds -- the reference store's
  ///        apply, which wrote @p in, and @ref record_steps -- over
  ///        @p slices, a workgroup a block, sized and gated on the device.
  ///
  /// Advances the map's tick, which every block the inverse changes is
  /// stamped with. The list must be duplicate-free, and the coefficients
  /// made against the grid's `trunc_dist` (both unchecked: they come from a
  /// frame whose order and header the caller checked). Each entry's coord
  /// must be one @p grid holds, else it is counted in @ref rejected; its ptr
  /// is ignored.
  /// @param batch   The batch to record into; submitted by the caller.
  /// @param grid    As @ref record_forward, written in place.
  /// @param in      The blocks, bound where they are; a count of 0 records
  ///                the reject count's reset alone.
  /// @param params  Their coefficient count and steps.
  /// @param slices  The inverse's slices, a group a block, for @p in's
  ///                count.
  /// @param stage   As @ref record_forward.
  /// @return OK, or, before anything is recorded, @ref record_forward's
  ///         refusals of @p grid and @p params, or
  ///         `Status::Code::InvalidArgument` for blocks holding another K
  ///         than @p params, more blocks than the grid's heap, buffers
  ///         missing or smaller than @p in's count needs, or a P-frame's
  ///         entries, which are not yet its blocks. Otherwise a buffer
  ///         failure.
  core::Status record_inverse(core::CommandBatch& batch,
                              volume::VoxelBlockGrid& grid,
                              const ResidentBlocks& in,
                              const CodecParams& params,
                              const recon::detail::IndirectSlices& slices,
                              core::GpuStageScope* stage = nullptr);

  /// @return The entries the last forward or inverse found no block for --
  ///         which it neither read nor wrote -- a u32, once its batch ran,
  ///         for @ref check_rejected.
  const core::Buffer& rejected() const noexcept { return rejected_; }

  /// @brief Record the gathering of the first `n` entries of @p list whose
  ///        block holds an observed voxel (`weight >= volume::kObservedWeight`)
  ///        into @ref observed_list, `n` the word @p count names, in
  ///        unspecified order -- or, with @p true_sdf, those
  ///        `EncoderConfig::true_sdf` codes, its cut set left in
  ///        @ref cut_set.
  ///
  /// What the encoder keeps, found before the forward transform rather than
  /// after it. This reads each block's weights alone and appends only observed
  /// entries to a device list, a group an entry, sized on the device; their
  /// count and the reject tally stay there (@ref observed_counts). With
  /// @p true_sdf, two passes over the list come first in the same batch: the
  /// cut, then the flags of the blocks holding a corner of a cell kept, which
  /// the append then takes instead. The encoder sorts the entries by
  /// coordinate on the device before transforming them, so append order
  /// cannot affect bytes.
  /// @param batch     The batch to record into; submitted by the caller.
  /// @param grid      As @ref record_forward.
  /// @param list      The map's own active list, bound where it is: one
  ///                  `record_compaction` wrote into @p batch, or one
  ///                  @ref volume::VoxelHashMap::held_device_list holds.
  /// @param count     Its entries: a word a kernel wrote or the caller set, at
  ///                  most its bound, at most the grid's heap.
  /// @param true_sdf  Whether to keep the blocks `EncoderConfig::true_sdf`
  ///                  codes.
  /// @param stage     As @ref record_forward.
  /// @return OK, recording the counts' reset alone for a bound of 0; or,
  ///         before anything is recorded, `Status::Code::InvalidArgument` for
  ///         a grid @ref record_forward refuses, a count word missing,
  ///         misaligned or past its buffer, a bound past the grid's heap, a
  ///         @p list `StorageInput::check` refuses for it, or a buffer past
  ///         `maxStorageBufferRange`; otherwise a buffer failure.
  core::Status record_observed(core::CommandBatch& batch,
                               const volume::VoxelBlockGrid& grid,
                               const core::Buffer& list,
                               const volume::DeviceCount& count, bool true_sdf,
                               core::GpuStageScope* stage = nullptr);

  /// @return The last @ref record_observed's two words, once its batch ran:
  ///         the entries it found no block for, then how many entries
  ///         @ref observed_list holds.
  const core::Buffer& observed_counts() const noexcept {
    return observed_counts_;
  }

  /// @return How many entries @ref observed_list holds, once the last
  ///         @ref record_observed's batch has run: its word in
  ///         @ref observed_counts, bound by that call's list.
  volume::DeviceCount observed_count() const noexcept {
    return volume::DeviceCount{&observed_counts_, sizeof(std::uint32_t),
                               observed_bound_};
  }

  /// @return The last @ref record_observed's entries, `volume::BlockIndex`
  ///         each, valid until its next call; at least its bound's.
  const core::Buffer& observed_list() const noexcept {
    return observed_blocks_;
  }

  /// @return The cut set of the last @ref record_observed with `true_sdf`
  ///         and a non-empty list, laid out as @ref record_forward takes it,
  ///         valid until the next such call.
  const core::Buffer& cut_set() const noexcept { return cut_; }

 private:
  DctTransform() = default;

  /// The attributes both kernels bind, found by @ref check_inputs.
  struct GridViews {
    volume::AttributeView tsdf;
    volume::AttributeView weight;
  };

  /// What `EncoderConfig::true_sdf` adds to a call: the switch, the cut set
  /// bound, the transform's own unless the caller's, and the passes: the
  /// surface kernel's two (cut, then keep), each kernel's one otherwise.
  struct TrueSdf {
    bool on = false;
    const core::Buffer* cut = nullptr;
    std::uint32_t passes = 1;
  };

  /// The grid / params checks both directions share, all taken before
  /// anything is allocated, and for an empty list too.
  core::Result<GridViews> check_inputs(const char* op,
                                       const volume::VoxelBlockGrid& grid,
                                       const CodecParams& params) const;

  /// Bind the call's buffers and record @p kernel over the @p count entries
  /// of @p list into @p batch, a workgroup an entry, once per pass of
  /// @p true_sdf (a batch takes no set rewritten after a dispatch), its
  /// @p counts zeroed first; over @p slices when given, which the device
  /// sized. @p coefficients and @p masks may be null for a kernel that reads
  /// none.
  core::Status record(core::CommandBatch& batch, core::ComputeKernel& kernel,
                      const volume::VoxelBlockGrid& grid,
                      const GridViews& views, VkBuffer list,
                      std::uint32_t count, const CodecParams& params,
                      VkBuffer coefficients, VkBuffer masks,
                      VkDeviceSize masks_bytes, const TrueSdf& true_sdf,
                      const recon::detail::IndirectSlices* slices,
                      const core::Buffer& counts, core::GpuStageScope* stage);
  /// A count word's checks, and its bound against the grid's heap.
  core::Status check_count(const char* op, const volume::VoxelBlockGrid& grid,
                           const volume::DeviceCount& count) const;
  /// Grow @p buffer if needed, retaining it for later calls
  /// (`ensure_device_scratch`).
  core::Status ensure_scratch(core::Buffer& buffer, VkDeviceSize bytes,
                              const char* name);

  // Borrowed (must outlive this).
  core::Device* device_ = nullptr;
  core::Allocator* allocator_ = nullptr;

  // The device's maxComputeWorkGroupCount[0], cached at create().
  std::uint32_t max_workgroup_count_x_ = 0;
  // Cached maxStorageBufferRange; every per-call buffer is checked against it.
  VkDeviceSize max_storage_buffer_range_ = 0;

  // The kernels share one layout shape (twelve storage buffers + the push
  // range) and one pool. Declared before pool_ and tables_, so those are
  // destroyed first -- the kernels' sets are freed with the pool.
  core::ComputeKernel forward_kernel_;
  core::ComputeKernel inverse_kernel_;
  core::ComputeKernel observed_kernel_;
  core::ComputeKernel surface_kernel_;
  // The count-to-slices kernels (recon::detail::add_dispatch_args) of the
  // observed pass and the forward, one each, so both size in one batch.
  core::ComputeKernel observed_args_kernel_;
  core::ComputeKernel forward_args_kernel_;
  core::DescriptorPool pool_;
  // The slices those write.
  core::Buffer observed_slices_;
  core::Buffer forward_slices_;
  // The basis + zigzag tables (binding 3), uploaded once.
  core::Buffer tables_;
  // Canonical per-basis effective steps (binding 9), computed on the host
  // and uploaded in each transform batch.
  core::Buffer quantization_steps_;
  // The forward's and the inverse's rejected entries (binding 7, the first
  // word), and the observed pass's rejected entries and count (its binding
  // 7), each zeroed in its call's batch.
  core::Buffer rejected_;
  core::Buffer observed_counts_;
  // The last record_observed's bound.
  std::uint32_t observed_bound_ = 0;
  // Scratch stays alive between calls; each binding uses its logical range,
  // not the retained capacity. No allocation when a call fits it.
  core::Buffer observed_blocks_;
  // The true SDF passes' cut set (binding 10, a bit a voxel slot) and keep
  // flags (binding 11, a word a block slot), sized to the grid's heap and
  // zeroed in each record_observed batch that uses them.
  core::Buffer cut_;
  core::Buffer keep_;
};

}  // namespace volumetric_kit::recon::codec::detail
