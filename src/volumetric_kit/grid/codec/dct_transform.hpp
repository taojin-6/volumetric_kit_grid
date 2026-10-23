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

/// @brief The forward and inverse block DCT, as two GLSL compute kernels, and
///        a third that finds the blocks holding an observed voxel.
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
/// store's apply wrote it, in the caller's batch too; each reads back its
/// reject count alone, and the observed filter its count and reject tally.
/// The observed list grows when needed and remains allocated for later
/// calls. The coefficients travel two to a word, 16 bits each.
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

  /// @brief Record the transform and quantization of the first @p count
  ///        blocks of @p list into @p batch, its output into @p masks and
  ///        @p coefficients by list position.
  ///
  /// When @p batch is submitted, @p rejected receives the entries the
  /// kernel found no block for, which @ref check_rejected turns into a
  /// refusal.
  /// @param batch         The batch to record into; submitted by the caller.
  /// @param grid          The grid the list names; must carry 4-byte `tsdf`
  ///                      and `weight` attributes and have `block_size` 8.
  /// @param list          `volume::BlockIndex` entries on the device, found
  ///                      by `coord` (each `ptr` is ignored); must outlive
  ///                      the output.
  /// @param count         The entries to transform.
  /// @param params        The coefficient count and steps.
  /// @param masks         Where each entry's `kMaskWordsPerBlock` mask words
  ///                      go; must outlive the output.
  /// @param coefficients  Where each entry's @ref coefficient_words(K)
  ///                      words go, an odd K's pad half 0; must outlive the
  ///                      output.
  /// @param rejected      Written at submit; must outlive it.
  /// @param stage         Optional scope the caller's stage row is open
  ///                      under; the dispatches record their device spans
  ///                      into it. `nullptr` times nothing.
  /// @return The output, borrowing @p list, @p masks and @p coefficients
  ///         (no entries for @p count 0, which records nothing), or, before
  ///         anything is recorded, `Status::Code::InvalidArgument` for
  ///         invalid @p params, a grid that is moved-from, has another block
  ///         size, a non-positive `trunc_dist`, or lacks a float `tsdf` /
  ///         `weight`; more entries than the grid's block heap; a @p list
  ///         `StorageInput::check` refuses for them; output buffers smaller
  ///         than they need; or a buffer that would exceed
  ///         `maxStorageBufferRange`. Otherwise a buffer failure.
  core::Result<ResidentBlocks> record_forward(
      core::CommandBatch& batch, const volume::VoxelBlockGrid& grid,
      const core::Buffer& list, std::uint32_t count, const CodecParams& params,
      const core::Buffer& masks, const core::Buffer& coefficients,
      std::uint32_t& rejected, core::GpuStageScope* stage = nullptr);

  /// @return The refusal for @p rejected entries of @p count the kernels
  ///         found no block for, or OK when there are none.
  static core::Status check_rejected(const char* op, std::uint32_t rejected,
                                     std::uint32_t count);

  /// @brief Reconstruct every block of @p in from its coefficients and mask,
  ///        overwriting its `tsdf` and `weight`, from blocks the device holds
  ///        (the reference store's apply output), bound where they are and
  ///        recorded into the caller's @p batch after what it already holds
  ///        -- that apply -- and submitted with it, so the two share one
  ///        submit.
  ///
  /// The list must be duplicate-free, and the coefficients made against the
  /// grid's `trunc_dist` (both unchecked: they come from a frame whose order
  /// and header the caller checked). Each entry's coord must be one @p grid
  /// holds; its ptr is ignored.
  /// @param batch  Holding what the inverse reads; submitted here once the
  ///               checks pass (a refusal submits nothing).
  /// @param grid   As @ref record_forward, written in place.
  /// @param stage  As @ref record_forward.
  /// @return OK, @ref record_forward's refusals of @p grid and @p params, or
  ///         `Status::Code::InvalidArgument` for blocks holding another K
  ///         than @p params, more blocks than the grid's heap, buffers
  ///         missing or smaller than @p in's count needs, or a P-frame's
  ///         entries, which are not yet its blocks; a refusal for entries
  ///         whose coord the grid does not hold comes from the device, after
  ///         every other entry has been written. Otherwise a buffer or
  ///         dispatch failure.
  core::Status inverse(core::CommandBatch& batch, volume::VoxelBlockGrid& grid,
                       const ResidentBlocks& in, const CodecParams& params,
                       core::GpuStageScope* stage = nullptr);

  /// @brief Gather the entries of @p list whose block holds an observed
  ///        voxel (`weight >= volume::kObservedWeight`) into
  ///        @ref observed_list, in unspecified order.
  ///
  /// What the encoder keeps, found before the forward transform rather than
  /// after it. This reads each block's weights alone and appends only observed
  /// entries to a device list; only the count and the reject tally come back.
  /// The encoder sorts the entries by coordinate on the device before
  /// transforming them, so append order cannot affect bytes.
  /// @param grid   As @ref record_forward.
  /// @param list   The map's own active list, bound where it is
  ///               (@ref
  ///               volume::VoxelHashMap::compact_active_blocks_on_device).
  /// @param stage  As @ref record_forward.
  /// @return How many entries @ref observed_list holds;
  ///         `Status::Code::InvalidArgument` for a list
  ///         @ref volume::VoxelHashMap::check_device_block_list refuses, a
  ///         grid @ref record_forward refuses, or a buffer past
  ///         `maxStorageBufferRange`; otherwise a buffer or dispatch failure.
  core::Result<std::uint32_t> observed(const volume::VoxelBlockGrid& grid,
                                       const volume::DeviceBlockList& list,
                                       core::GpuStageScope* stage = nullptr);

  /// @return The last @ref observed call's entries, `volume::BlockIndex`
  ///         each, valid until its next call; empty until one keeps any.
  const core::Buffer& observed_list() const noexcept {
    return observed_blocks_;
  }

 private:
  DctTransform() = default;

  /// The attributes both kernels bind, found by @ref check_inputs.
  struct GridViews {
    volume::AttributeView tsdf;
    volume::AttributeView weight;
  };

  /// The grid / params checks both directions share, all taken before
  /// anything is allocated, and for an empty list too.
  core::Result<GridViews> check_inputs(const char* op,
                                       const volume::VoxelBlockGrid& grid,
                                       const CodecParams& params) const;

  /// Bind the call's buffers and record @p kernel over the @p count entries
  /// of @p list into @p batch, a workgroup an entry, the reject count zeroed
  /// first.
  /// @p coefficients may be null for the kernel that reads none.
  core::Status record(core::CommandBatch& batch, core::ComputeKernel& kernel,
                      const volume::VoxelBlockGrid& grid,
                      const GridViews& views, VkBuffer list,
                      std::uint32_t count, const CodecParams& params,
                      VkBuffer coefficients, VkBuffer masks,
                      VkDeviceSize masks_bytes, core::GpuStageScope* stage);
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

  // The kernels share one layout shape (ten storage buffers + the push range)
  // and one pool. Declared before pool_ and tables_, so those are destroyed
  // first -- the kernels' sets are freed with the pool.
  core::ComputeKernel forward_kernel_;
  core::ComputeKernel inverse_kernel_;
  core::ComputeKernel observed_kernel_;
  core::DescriptorPool pool_;
  // The basis + zigzag tables (binding 3), uploaded once.
  core::Buffer tables_;
  // Canonical per-basis effective steps (binding 9), computed on the host
  // and uploaded in each transform batch.
  core::Buffer quantization_steps_;
  // Rejected entries followed by the observed count (binding 7), zeroed
  // in each call's batch. Forward/inverse only read the first word back.
  core::Buffer rejected_;
  // Scratch stays alive between calls; each binding uses its logical range,
  // not the retained capacity. No allocation when a call fits it.
  core::Buffer observed_blocks_;
};

}  // namespace volumetric_kit::recon::codec::detail
