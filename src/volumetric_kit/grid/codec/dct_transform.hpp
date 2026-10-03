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
#include <vector>

#include "dct_blocks.hpp"
#include "volumetric_kit/recon/codec/codec_params.hpp"
#include "volumetric_kit/recon/codec/export.hpp"
#include "volumetric_kit/recon/core/allocator.hpp"
#include "volumetric_kit/recon/core/buffer.hpp"
#include "volumetric_kit/recon/core/compute_kernel.hpp"
#include "volumetric_kit/recon/core/descriptor.hpp"
#include "volumetric_kit/recon/core/result.hpp"
#include "volumetric_kit/recon/core/vulkan.hpp"
#include "volumetric_kit/recon/volume/hash_types.hpp"
#include "volumetric_kit/recon/volume/voxel_block_grid.hpp"

namespace volumetric_kit::recon {
class CommandBatch;
class Device;
class GpuStageScope;
}  // namespace volumetric_kit::recon

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
/// TODO(codec): a device-resident forward output, so the GPU entropy coder
/// (the 2026-09-26 decision's fifth PR) reads the coefficients where they were
/// written rather than through this host round trip.
class VR_CODEC_API DctTransform {
 public:
  /// @brief Build both kernels and upload the basis and zigzag tables.
  /// @param device     The compute device (must outlive this object).
  /// @param allocator  The allocator the scratch buffers come from (must
  ///                   outlive this object).
  /// @param config     Construction-time options.
  /// @return The transform, or a non-OK @ref Status if a pipeline, the pool or
  ///         the table upload fails.
  static Result<DctTransform> create(Device& device, Allocator& allocator,
                                     const DctTransformConfig& config = {});

  ~DctTransform() = default;
  DctTransform(DctTransform&&) noexcept = default;
  DctTransform& operator=(DctTransform&&) noexcept = default;
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
  /// @return OK, or @ref Status::Code::InvalidArgument for a moved-from
  ///         transform, invalid @p params, a grid that is moved-from, has
  ///         another block size, a non-positive `trunc_dist`, or lacks a float
  ///         `tsdf` / `weight`; a list that
  ///         @ref volume::VoxelBlockGrid::check_block_list refuses, or one with
  ///         an entry whose coord @p grid does not hold; or a buffer that
  ///         would exceed `maxStorageBufferRange`. Otherwise a buffer or
  ///         dispatch failure.
  ///         On failure @p out is left empty.
  Status forward(const volume::VoxelBlockGrid& grid,
                 const volume::BlockList& blocks, const CodecParams& params,
                 DctBlocks& out, GpuStageScope* stage = nullptr);

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
  Status inverse(volume::VoxelBlockGrid& grid, const volume::BlockList& blocks,
                 const DctBlocks& in, GpuStageScope* stage = nullptr);

  /// @brief The entries of @p list whose block holds an observed voxel
  ///        (`weight >= volume::kObservedWeight`), in unspecified order.
  ///
  /// What the encoder keeps, found before the forward transform rather than
  /// after it. This reads each block's weights alone and appends only observed
  /// entries to a device list. A predicted prefix (the last count plus 25%,
  /// bounded by the input count; the full input count on the first call) is
  /// read back with the count and rejection tally. A growing result reads its
  /// remaining tail in a second batch. The encoder sorts the returned entries
  /// by coordinate before transforming, so append order cannot affect bytes.
  /// @param grid   As @ref forward.
  /// @param list   The map's own active list, bound where it is
  ///               (@ref
  ///               volume::VoxelHashMap::compact_active_blocks_on_device).
  /// @param stage  As @ref forward.
  /// @return The observed entries; @ref Status::Code::InvalidArgument for a
  ///         list @ref volume::VoxelHashMap::check_device_block_list refuses,
  ///         a grid @ref forward refuses, or a buffer past
  ///         `maxStorageBufferRange`; otherwise a buffer or dispatch failure.
  Result<std::vector<volume::BlockIndex>> observed(
      const volume::VoxelBlockGrid& grid, const volume::DeviceBlockList& list,
      GpuStageScope* stage = nullptr);

  /// @return `true` if this owns its live pipelines (`false` when moved-from).
  bool valid() const noexcept {
    return forward_kernel_.valid() && inverse_kernel_.valid() &&
           observed_kernel_.valid();
  }

 private:
  DctTransform() = default;

  /// The attributes both kernels bind, found by @ref check_inputs.
  struct GridViews {
    volume::AttributeView tsdf;
    volume::AttributeView weight;
  };

  /// The grid / list / params checks both directions share, all taken before
  /// anything is allocated, and for an empty list too.
  Result<GridViews> check_inputs(const char* op,
                                 const volume::VoxelBlockGrid& grid,
                                 const volume::BlockList& blocks,
                                 const CodecParams& params) const;

  /// Bind the call's buffers and record @p kernel over the @p count entries
  /// of @p list into @p batch, in dispatches of at most
  /// @ref blocks_per_dispatch_ workgroups, the reject count zeroed first.
  /// @p coefficients may be null for the kernel that reads none.
  Status record(CommandBatch& batch, ComputeKernel& kernel,
                const volume::VoxelBlockGrid& grid, const GridViews& views,
                VkBuffer list, std::uint32_t count, const CodecParams& params,
                VkBuffer coefficients, VkBuffer masks, VkDeviceSize masks_bytes,
                GpuStageScope* stage);
  /// Grow @p buffer if needed, retaining it for later calls.
  Status ensure_scratch(Buffer& buffer, VkDeviceSize bytes, const char* name);
  /// Stage @p blocks onto the retained block-list buffer, in @p batch.
  Status upload_list(CommandBatch& batch, const volume::BlockList& blocks);
  /// The refusal for @p rejected entries the kernels found no block for.
  static Status check_rejected(const char* op, std::uint32_t rejected,
                               std::uint32_t count);

  // Borrowed (must outlive this).
  Device* device_ = nullptr;
  Allocator* allocator_ = nullptr;

  // The device's maxComputeWorkGroupCount[0], and the batch size run() uses:
  // the config's value capped at it. Cached at create().
  std::uint32_t max_workgroup_count_x_ = 0;
  std::uint32_t blocks_per_dispatch_ = 0;
  // Cached maxStorageBufferRange; every per-call buffer is checked against it.
  VkDeviceSize max_storage_buffer_range_ = 0;

  // The kernels share one layout shape (ten storage buffers + the push range)
  // and one pool. Declared before pool_ and tables_, so those are destroyed
  // first -- the kernels' sets are freed with the pool.
  ComputeKernel forward_kernel_;
  ComputeKernel inverse_kernel_;
  ComputeKernel observed_kernel_;
  DescriptorPool pool_;
  // The basis + zigzag tables (binding 3), uploaded once.
  Buffer tables_;
  // Canonical per-basis effective steps (binding 9), computed on the host
  // and uploaded in each transform batch.
  Buffer quantization_steps_;
  // Rejected entries followed by the observed count (binding 7), zeroed
  // in each call's batch. Forward/inverse only read the first word back.
  Buffer rejected_;
  // Scratch stays alive between calls; each binding uses its logical range,
  // not the retained capacity. No allocation when a call fits these buffers.
  Buffer observed_blocks_;
  Buffer block_list_;
  Buffer coefficients_;
  Buffer masks_;
  // Readback prediction only: every call still obtains and checks its count.
  std::uint32_t last_observed_count_ = 0;
  bool has_observed_count_ = false;
};

}  // namespace volumetric_kit::recon::codec::detail
