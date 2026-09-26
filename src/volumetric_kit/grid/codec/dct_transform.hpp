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
class Device;
}

namespace volumetric_kit::recon::codec::detail {

/// @brief The weight at or above which a voxel counts as observed.
///
/// A **copy** of the mesh tier's private `kWeightThreshold`
/// (marching_cubes.cpp): the mesher skips a corner whose weight is below it, so
/// the mask must mark exactly the voxels the mesher would have used. `codec`
/// may not link `mesh`, so the value is repeated rather than shared -- change
/// the two together.
inline constexpr float kObservedWeight = 1e-6f;

/// The weight the inverse writes on an observed voxel: anything at or above
/// @ref kObservedWeight meshes the same, and a decoded grid is not fused into.
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

/// @brief The forward and inverse block DCT, as two GLSL compute kernels.
///
/// One workgroup of 64 invocations per listed block; each invocation owns one
/// 8-voxel line per separable pass. The forward kernel reads `tsdf` and
/// `weight`, divides the SDF by `trunc_dist` (so its input is in [-1, 1]),
/// transforms, and keeps the first `coefficient_count` coefficients in 3-D
/// zigzag order, each rounded half-to-even after dividing by its step and
/// clamped to ±@ref kMaxQuantizedMagnitude. It also packs one bit per voxel,
/// set where `weight >= kObservedWeight`, into @ref kMaskWordsPerBlock words:
/// voxel `v` (`x + 8y + 64z`) is bit `v % 32` of word `v / 32`.
///
/// The inverse undoes it: dequantizes, zero-fills the dropped coefficients,
/// inverse-transforms, and writes each observed voxel's `tsdf` (clamped back to
/// `±trunc_dist`, the integrator's invariant) with weight
/// @ref kDecodedWeight, and each unobserved voxel as a freshly allocated block
/// holds it -- zero `tsdf`, zero `weight` -- so the mesher uses exactly the
/// voxels the encoder saw observed.
///
/// Output is indexed by **list position**, not by `ptr`: coefficient `j` of
/// list entry `i` is `coefficients[i * coefficient_count + j]`, its mask words
/// `masks[i * 16 .. i * 16 + 15]`. The order the blocks travel in is the
/// caller's (the encoder sorts them).
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
  /// @param allocator  The allocator the per-call buffers come from (must
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
  /// @param grid          The grid the list names; must carry 4-byte `tsdf` and
  ///                      `weight` attributes and have `block_size` 8.
  /// @param blocks        The blocks to transform, typically
  ///                      `grid.block_list(grid.map().compact_active_blocks())`.
  ///                      An empty list is a no-op success.
  /// @param params        The coefficient count and steps.
  /// @param coefficients  Receives `blocks.count * coefficient_count` values.
  /// @param masks         Receives `blocks.count * 16` mask words.
  /// @return OK, or @ref Status::Code::InvalidArgument for a moved-from
  ///         transform, invalid @p params, a grid that is moved-from, has
  ///         another block size, a non-positive `trunc_dist`, or lacks a float
  ///         `tsdf` / `weight`; a list that is null with a count, holds more
  ///         blocks than the heap, carries another topology epoch, or names a
  ///         `ptr` outside the heap; or a buffer that would exceed
  ///         `maxStorageBufferRange`. Otherwise a buffer or dispatch failure.
  ///         On failure the outputs are left empty.
  Status forward(const volume::VoxelBlockGrid& grid,
                 const volume::BlockList& blocks, const CodecParams& params,
                 std::vector<std::int32_t>& coefficients,
                 std::vector<std::uint32_t>& masks);

  /// @brief Reconstruct every block in @p blocks from its coefficients and
  ///        mask, overwriting its `tsdf` and `weight`.
  ///
  /// The blocks must already be allocated in @p grid; the list must be
  /// duplicate-free (unchecked -- a repeat is two workgroups writing one
  /// block).
  /// @param grid          As @ref forward, written in place.
  /// @param blocks        The blocks to reconstruct, in the order
  ///                      @p coefficients and @p masks were produced in.
  /// @param params        The coefficient count and steps they were quantized
  ///                      with.
  /// @param coefficients  `blocks.count * coefficient_count` values.
  /// @param masks         `blocks.count * 16` mask words.
  /// @return OK, the same refusals as @ref forward, or
  ///         @ref Status::Code::InvalidArgument when either input's size does
  ///         not match the list.
  Status inverse(volume::VoxelBlockGrid& grid, const volume::BlockList& blocks,
                 const CodecParams& params,
                 const std::vector<std::int32_t>& coefficients,
                 const std::vector<std::uint32_t>& masks);

  /// @return `true` if this owns both live pipelines (`false` when moved-from).
  bool valid() const noexcept {
    return forward_kernel_.valid() && inverse_kernel_.valid();
  }

 private:
  DctTransform() = default;

  /// The grid / list / params checks both directions share.
  Status check_inputs(const char* op, const volume::VoxelBlockGrid& grid,
                      const volume::BlockList& blocks,
                      const CodecParams& params) const;

  /// Upload the list, bind the per-call buffers and run @p kernel over it in
  /// batches of at most @ref blocks_per_dispatch_ workgroups.
  Status run(const char* op, ComputeKernel& kernel,
             const volume::VoxelBlockGrid& grid,
             const volume::BlockList& blocks, const CodecParams& params,
             const Buffer& coefficients, const Buffer& masks);

  // Borrowed (must outlive this).
  Device* device_ = nullptr;
  Allocator* allocator_ = nullptr;

  // The device's maxComputeWorkGroupCount[0], and the batch size run() uses:
  // the config's value capped at it. Cached at create().
  std::uint32_t max_workgroup_count_x_ = 0;
  std::uint32_t blocks_per_dispatch_ = 0;
  // Cached maxStorageBufferRange; every per-call buffer is checked against it.
  VkDeviceSize max_storage_buffer_range_ = 0;

  // Both kernels share one layout shape (six storage buffers + the push
  // range) and one pool. Declared before pool_ and tables_, so those are
  // destroyed first -- the kernels' sets are freed with the pool.
  ComputeKernel forward_kernel_;
  ComputeKernel inverse_kernel_;
  DescriptorPool pool_;
  // The basis + zigzag tables (binding 3 of both kernels), uploaded once.
  Buffer tables_;
};

}  // namespace volumetric_kit::recon::codec::detail
