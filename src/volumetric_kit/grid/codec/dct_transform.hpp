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
#include <vector>

#include "dct_blocks.hpp"
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

/// @brief Construction-time options.
struct DctTransformConfig {
  /// Blocks per dispatch, capped at the device's `maxComputeWorkGroupCount[0]`
  /// either way; 0 means that cap. A list longer than this is run as several
  /// dispatches rather than refused. Lowering it only changes how the same
  /// work is batched, which is what lets a test reach the multi-dispatch path
  /// without a list larger than the device limit.
  std::uint32_t max_blocks_per_dispatch = 0;
};

/// @brief Blocks where the device holds them: @ref count entries of a block
///        list, their masks and their coefficients, in the layout the
///        kernels write. Borrowed from what produced them -- the transform's
///        @ref DctTransform::record_forward or the device frame reader -- and
///        valid until that object's next call.
struct ResidentBlocks {
  const core::Buffer* list =
      nullptr;  ///< `volume::BlockIndex` entries, frame order.
  const core::Buffer* masks =
      nullptr;  ///< `kMaskWordsPerBlock` words an entry.
  /// Two int16 a word, coefficient 2p low; `(K + 1) / 2` words an entry.
  const core::Buffer* coefficients = nullptr;
  std::uint32_t count = 0;
  std::uint32_t coefficient_count = 0;  ///< K.
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
/// Output is indexed by **list position**, not by `ptr` (see @ref DctBlocks).
/// The order the blocks travel in is the caller's (the encoder sorts them).
///
/// Each workgroup first finds its entry's block by its `coord`, through the
/// grid's hash table, and never reads its `ptr`: so no ptr is trusted, and the
/// decoder can list blocks it has just allocated without reading their slots
/// back. A coord the table does not hold is neither read nor written, and the
/// call reports how many there were. That costs one probe per block, where a
/// host scan would be O(count) per call. Like the mesher's probe, it needs the
/// map quiescent: no allocate may run into the grid during a call.
///
/// Every buffer is device-local. Forward and inverse each use one
/// @ref CommandBatch: the inputs staged up, the dispatches, and the results
/// and reject count read back. Scratch buffers grow when needed and remain
/// allocated for later calls. The observed filter compacts its result on the
/// device and reads back a predicted prefix with the count in one batch; only
/// a result that outgrows that prefix needs another, transfer-only batch.
/// The coefficients travel two to a word, 16 bits each.
///
/// @warning The @ref Device and @ref Allocator passed to @ref create must
///          outlive this object; it stores references to them.
///
/// @ref record_forward leaves the forward output on the device for the device
/// frame writer, which reads the coefficients where they were written.
class VR_CODEC_API DctTransform {
 public:
  /// @brief Build both kernels and upload the basis and zigzag tables.
  /// @param device     The compute device (must outlive this object).
  /// @param allocator  The allocator the scratch buffers come from (must
  ///                   outlive this object).
  /// @param config     Construction-time options.
  /// @return The transform, or a non-OK @ref Status if a pipeline, the pool or
  ///         the table upload fails.
  static core::Result<std::unique_ptr<DctTransform>> create(
      core::Device& device, core::Allocator& allocator,
      const DctTransformConfig& config = {});

  ~DctTransform() = default;
  DctTransform(const DctTransform&) = delete;
  DctTransform& operator=(const DctTransform&) = delete;

  /// @brief Transform and quantize every block in @p blocks.
  /// @param grid    The grid the list names; must carry 4-byte `tsdf` and
  ///                `weight` attributes and have `block_size` 8.
  /// @param blocks  The blocks to transform, anchored with
  ///                @ref volume::VoxelBlockGrid::block_list to a compaction the
  ///                caller holds -- `auto active =
  ///                grid.map().compact_active_blocks();` and then
  ///                `grid.block_list(active.value())`. An empty list is a
  ///                no-op success.
  /// @param params  The coefficient count and steps.
  /// @param out     Receives the coefficients and masks, and @p params and the
  ///                grid's `trunc_dist` beside them.
  /// @param stage   Optional scope the caller's stage row is open under; the
  ///                dispatches record their device spans into it. `nullptr`
  ///                times nothing.
  /// @return OK, or @ref Status::Code::InvalidArgument for invalid
  ///         @p params, a grid that is moved-from, has another block size, a
  ///         non-positive `trunc_dist`, or lacks a float `tsdf` / `weight`; a
  ///         list that
  ///         @ref volume::VoxelBlockGrid::check_block_list refuses, or one with
  ///         an entry whose coord @p grid does not hold; or a buffer that
  ///         would exceed `maxStorageBufferRange`. Otherwise a buffer or
  ///         dispatch failure.
  ///         On failure @p out is left empty.
  core::Status forward(const volume::VoxelBlockGrid& grid,
                       const volume::BlockList& blocks,
                       const CodecParams& params, DctBlocks& out,
                       core::GpuStageScope* stage = nullptr);

  /// @brief Record @ref forward into @p batch, leaving its output on the
  ///        device for the device frame writer.
  ///
  /// The list, masks and coefficients stay in this transform's buffers until
  /// its next call. When @p batch is submitted, @p rejected receives the
  /// entries the kernel found no block for, which @ref check_rejected turns
  /// into the refusal @ref forward makes.
  /// @param batch     The batch to record into; submitted by the caller.
  /// @param rejected  Written at submit; must outlive it.
  /// @return The resident output (no entries for an empty list, which records
  ///         nothing), or the refusals @ref forward makes before dispatching.
  core::Result<ResidentBlocks> record_forward(
      core::CommandBatch& batch, const volume::VoxelBlockGrid& grid,
      const volume::BlockList& blocks, const CodecParams& params,
      std::uint32_t& rejected, core::GpuStageScope* stage = nullptr);

  /// @return The refusal for @p rejected entries of @p count the kernels
  ///         found no block for, or OK when there are none.
  static core::Status check_rejected(const char* op, std::uint32_t rejected,
                                     std::uint32_t count);

  /// @brief Reconstruct every block in @p blocks from its coefficients and
  ///        mask, overwriting its `tsdf` and `weight`.
  ///
  /// The list must be duplicate-free (unchecked -- a repeat is two workgroups
  /// writing one block).
  /// @param grid    As @ref forward, written in place.
  /// @param blocks  The blocks to reconstruct, in the order @p in was produced
  ///                in; each coord must be one @p grid holds, and each ptr is
  ///                ignored.
  /// @param in      A @ref forward output, or one read back from a frame.
  /// @param stage   As @ref forward.
  /// @return OK, the same refusals as @ref forward, or
  ///         @ref Status::Code::InvalidArgument when @p in carries invalid
  ///         params, another `trunc_dist` than @p grid, or a size that does
  ///         not match the list. A refusal for entries whose coord the grid
  ///         does not hold comes from the device, after every other entry has
  ///         been written; the others refuse before anything is.
  core::Status inverse(volume::VoxelBlockGrid& grid,
                       const volume::BlockList& blocks, const DctBlocks& in,
                       core::GpuStageScope* stage = nullptr);

  /// @brief @ref inverse from blocks the device holds (the device frame
  ///        reader's), bound where they are rather than uploaded.
  ///
  /// The list must be duplicate-free, and the coefficients made against the
  /// grid's `trunc_dist` (both unchecked: the reader decoded them from a
  /// frame whose order and header the caller checked).
  /// @return As @ref inverse, and @ref Status::Code::InvalidArgument for
  ///         buffers missing or smaller than @p in's count needs.
  core::Status inverse(volume::VoxelBlockGrid& grid, const ResidentBlocks& in,
                       const CodecParams& params,
                       core::GpuStageScope* stage = nullptr);

  /// @brief The entries of @p list whose block holds an observed voxel
  ///        (`weight >= volume::kObservedWeight`), in unspecified order.
  ///
  /// What the encoder keeps, found before the forward transform rather than
  /// after it. This reads each block's weights alone and appends only observed
  /// entries to a device list. A predicted prefix (the last count plus 25%,
  /// bounded by the input count; the full input count when the last count is
  /// zero or there is none) is read back with the count and rejection tally. A
  /// growing result reads its remaining tail in a second batch. The encoder
  /// sorts the returned entries by coordinate before transforming, so append
  /// order cannot affect bytes.
  /// @param grid   As @ref forward.
  /// @param list   The map's own active list, bound where it is
  ///               (@ref
  ///               volume::VoxelHashMap::compact_active_blocks_on_device).
  /// @param stage  As @ref forward.
  /// @return The observed entries; @ref Status::Code::InvalidArgument for a
  ///         list @ref volume::VoxelHashMap::check_device_block_list refuses,
  ///         a grid @ref forward refuses, or a buffer past
  ///         `maxStorageBufferRange`; otherwise a buffer or dispatch failure.
  core::Result<std::vector<volume::BlockIndex>> observed(
      const volume::VoxelBlockGrid& grid, const volume::DeviceBlockList& list,
      core::GpuStageScope* stage = nullptr);

 private:
  DctTransform() = default;

  /// The attributes both kernels bind, found by @ref check_inputs.
  struct GridViews {
    volume::AttributeView tsdf;
    volume::AttributeView weight;
  };

  /// The grid / list / params checks both directions share, all taken before
  /// anything is allocated, and for an empty list too.
  core::Result<GridViews> check_inputs(const char* op,
                                       const volume::VoxelBlockGrid& grid,
                                       const volume::BlockList& blocks,
                                       const CodecParams& params) const;

  /// Bind the call's buffers and record @p kernel over the @p count entries
  /// of @p list into @p batch, in dispatches of at most
  /// @ref blocks_per_dispatch_ workgroups, the reject count zeroed first.
  /// @p coefficients may be null for the kernel that reads none.
  core::Status record(core::CommandBatch& batch, core::ComputeKernel& kernel,
                      const volume::VoxelBlockGrid& grid,
                      const GridViews& views, VkBuffer list,
                      std::uint32_t count, const CodecParams& params,
                      VkBuffer coefficients, VkBuffer masks,
                      VkDeviceSize masks_bytes, core::GpuStageScope* stage);
  /// Grow @p buffer if needed, retaining it for later calls
  /// (@ref ensure_device_scratch).
  core::Status ensure_scratch(core::Buffer& buffer, VkDeviceSize bytes,
                              const char* name);
  /// Stage @p blocks onto the retained block-list buffer, in @p batch.
  core::Status upload_list(core::CommandBatch& batch,
                           const volume::BlockList& blocks);
  /// Record the inverse over @p in into @p batch after what it holds, then
  /// submit it and take the reject count's refusal.
  core::Status run_inverse(core::CommandBatch& batch,
                           volume::VoxelBlockGrid& grid, const GridViews& views,
                           const ResidentBlocks& in, const CodecParams& params,
                           core::GpuStageScope* stage);

  // Borrowed (must outlive this).
  core::Device* device_ = nullptr;
  core::Allocator* allocator_ = nullptr;

  // The device's maxComputeWorkGroupCount[0], and the batch size run() uses:
  // the config's value capped at it. Cached at create().
  std::uint32_t max_workgroup_count_x_ = 0;
  std::uint32_t blocks_per_dispatch_ = 0;
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
  // not the retained capacity. No allocation when a call fits these buffers.
  core::Buffer observed_blocks_;
  core::Buffer block_list_;
  core::Buffer coefficients_;
  core::Buffer masks_;
  // The last observed count, the readback prediction only: every call still
  // obtains and checks its count. 0 predicts the whole input.
  std::uint32_t last_observed_ = 0;
};

}  // namespace volumetric_kit::recon::codec::detail
