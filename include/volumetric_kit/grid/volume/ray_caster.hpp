// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file volume/ray_caster.hpp
/// @brief Posed pinhole cameras' rays cast into a sparse grid: per pixel,
///        the first observed surface's depth and normal, found through the
///        grid's own hash table, K views in one dispatch.

#include <array>
#include <cstdint>
#include <memory>
#include <string_view>

#include "volumetric_kit/core/base/result.hpp"
#include "volumetric_kit/core/vulkan/allocator.hpp"
#include "volumetric_kit/core/vulkan/buffer.hpp"
#include "volumetric_kit/core/vulkan/command_batch.hpp"
#include "volumetric_kit/core/vulkan/compute_kernel.hpp"
#include "volumetric_kit/core/vulkan/descriptor.hpp"
#include "volumetric_kit/core/vulkan/fwd.hpp"
#include "volumetric_kit/core/vulkan/gpu_timer.hpp"
#include "volumetric_kit/grid/volume/export.hpp"
#include "volumetric_kit/grid/volume/voxel_block_grid.hpp"
#include "volumetric_kit/grid/volume/voxel_hash_map.hpp"  // DepthPinhole

namespace volumetric_kit::grid::volume {

/// @brief How a @ref RayCaster's rays march, and how many casts one batch
///        may hold.
struct RayCasterConfig {
  /// The step through free space, as a share of the grid's truncation
  /// distance; in (0, 1].
  float free_step = 0.8f;
  /// The smallest step near a surface, in voxels; positive.
  float min_step_voxels = 0.5f;
  /// The most steps a ray takes before it reports a miss; positive. An
  /// unallocated block costs one step, however wide.
  std::uint32_t max_steps = 2048;
  /// Casts one batch may hold through this caster, at least 1. The caster
  /// owns this many descriptor sets and writes every binding of the next one
  /// at each @ref RayCaster::record_cast, so a batch may recast at poses it
  /// moved. One cast more rewrites the first one's set, and the batch's
  /// submit refuses it.
  std::uint32_t casts_per_batch = 3;
};

/// @brief Which voxels may answer a cast. grid gives the attribute no
///        meaning: its owner builds the filters.
///
/// A voxel answers when its `weight` is at least @ref min_weight and, with an
/// @ref attribute, the bit for its byte v is set in @ref allow. One that does
/// not answer reads as unobserved. A trilinear sample (the hit's refinement
/// and its normal) answers only when all 8 of its corners do.
struct CastFilter {
  /// The least weight that answers; positive and finite, so a voxel never
  /// integrated (weight 0) never answers. By default @ref kObservedWeight,
  /// the mesher's and the codec's rule.
  float min_weight = kObservedWeight;
  /// A 1-byte attribute of the grid, by its @ref AttributeSpec name, or
  /// empty for no attribute test. Its characters must outlive every
  /// @ref RayCaster::record_cast given this filter, as an `inline constexpr`
  /// name's do.
  std::string_view attribute{};
  /// The bytes that answer: v answers when `allow[v >> 5] >> (v & 31) & 1`.
  /// By default every byte answers.
  std::array<std::uint32_t, 8> allow{~0u, ~0u, ~0u, ~0u, ~0u, ~0u, ~0u, ~0u};
};

/// @brief K views of one size, stacked as a cast writes them: pixel (x, y) of
///        view i is element `(i * height + y) * width + x`.
///
/// Its buffers are device-only, with storage and transfer-source usage, so
/// a later kernel reads them in place and a batch can read them back. A view
/// keeps what the last cast into it wrote; a new one's values are undefined.
struct ModelView {
  std::uint32_t width = 0;   ///< Pixels a row.
  std::uint32_t height = 0;  ///< Rows a view.
  std::uint32_t views = 0;   ///< Views stacked.
  /// One `float` a pixel: the hit's depth along its camera's +Z, in metres;
  /// 0 for a miss.
  core::Buffer depth;
  /// Three `float`s a pixel (12 bytes, scalar layout): the world unit normal
  /// at the hit, toward free space; 0 for a miss.
  core::Buffer normal;
};

/// @brief Casts posed pinhole cameras into a grid's `tsdf` and `weight`: the
///        one walk of the hash table for a view.
///
/// Pixel (u, v)'s ray runs through the pixel's integer coordinates, the ray
/// @ref VoxelHashMap::allocate_from_depth unprojects. It runs from its
/// camera's `min_depth` to its `max_depth` along +Z.
/// - **Skipping.** An unallocated block is skipped whole.
/// - **Marching.** Free space is crossed in steps of
///   @ref RayCasterConfig::free_step truncation distances. The steps shrink
///   with the stored distance near a surface, down to
///   @ref RayCasterConfig::min_step_voxels.
/// - **Hits.** A crossing from + to - between two nearest voxels that answer
///   (@ref CastFilter) is refined by false position on the trilinear field,
///   every sample of which must answer. So a ray that first meets a negative
///   voxel with no answering positive one just before it misses, as one
///   starting behind a surface, or entering a back band from unobserved
///   space, does. A hit outside the camera's depth range misses.
/// - **Normals.** The normal is the trilinear field's gradient at the hit.
///
/// An aggregate of owners: copy and move are deleted, and it is made by
/// @ref create.
///
/// @code
/// VKC_ASSIGN(auto caster, volume::RayCaster::create(device, allocator));
/// VKC_ASSIGN(volume::ModelView view, caster->make_view(256, 192, 20));
/// VKC_TRY(caster->record_cast(batch, grid, cameras, 20, filter, view));
/// // later kernels in the batch read view.depth and view.normal in place
/// @endcode
///
/// @warning The `Device` and `Allocator` passed to @ref create must outlive
///          it. Not thread-safe: one thread records through it.
class VGRID_VOLUME_API RayCaster {
 public:
  /// @brief Build the cast kernel and its
  ///        @ref RayCasterConfig::casts_per_batch descriptor sets.
  /// @param device     The compute device (must outlive the caster).
  /// @param allocator  Backs every @ref make_view (must outlive the caster).
  /// @param config     How rays march, and the casts a batch may hold.
  /// @return The caster; `Status::Code::InvalidArgument` for a config out of
  ///         range; what `check_device_requirements` refuses; or a
  ///         pipeline, layout or pool failure.
  static core::Result<std::unique_ptr<RayCaster>> create(
      core::Device& device, core::Allocator& allocator,
      const RayCasterConfig& config = {});

  ~RayCaster();
  RayCaster(const RayCaster&) = delete;
  RayCaster& operator=(const RayCaster&) = delete;
  RayCaster(RayCaster&&) = delete;
  RayCaster& operator=(RayCaster&&) = delete;

  /// @brief Allocate @p views stacked views of @p width x @p height.
  /// @return The view; `Status::Code::InvalidArgument` for a zero size or
  ///         count, or more rays than one dispatch holds on this device
  ///         (`maxComputeWorkGroupCount[0]` x 64, and a normal binding within
  ///         `maxStorageBufferRange`); or an allocation failure.
  core::Result<ModelView> make_view(std::uint32_t width, std::uint32_t height,
                                    std::uint32_t views = 1);

  /// @brief Record one dispatch that casts camera i of @p cameras into view i
  ///        of @p out, for each i below @p count. Views from @p count up are
  ///        left as they are.
  ///
  /// Every binding is written at each call, into the next of the caster's
  /// sets in turn, so a grid that grew or moved since the last cast is the
  /// one bound (an attribute's buffer is never cached).
  /// @param batch    Records the dispatch; the caller submits it.
  /// @param grid     A grid with 4-byte `tsdf` and `weight`, and the filter's
  ///                 attribute at 1 byte; read only. Its table is quiescent
  ///                 but for what @p batch records first, and the grid is
  ///                 unchanged until @p batch resolves.
  /// @param cameras  A storage buffer holding @p count @ref DepthPinhole s,
  ///                 96 bytes each, from offset 0. They may be written
  ///                 earlier in @p batch, for a recast at poses the batch
  ///                 moved. A camera's own `width` and `height` are not read:
  ///                 the view's size governs.
  /// @param count    Cameras to cast; 1 to `out.views`.
  /// @param filter   Which voxels answer.
  /// @param out      From @ref make_view, or of its layout and usage.
  /// @param stage    Optional tag for the dispatch's device time; null is
  ///                 untimed.
  /// @return OK; `Status::Code::InvalidArgument` for any of:
  ///         - @p count 0 or above `out.views`;
  ///         - @p cameras under `count * 96` bytes or without storage usage;
  ///         - an @p out buffer smaller than its views, or more rays than
  ///           one dispatch holds;
  ///         - a grid without 4-byte `tsdf` and `weight`, or without the
  ///           filter's attribute at 1 byte;
  ///         - a filter whose `min_weight` is not positive and finite.
  ///         Otherwise a refusal @p batch makes.
  /// @pre Every batch this caster recorded into before, other than @p batch,
  ///      has resolved: its wait returned, or it is no longer in flight. So
  ///      two batches in flight at once take two casters.
  core::Status record_cast(core::CommandBatch& batch,
                           const VoxelBlockGrid& grid,
                           const core::Buffer& cameras, std::uint32_t count,
                           const CastFilter& filter, ModelView& out,
                           core::GpuStageScope* stage = nullptr);

 private:
  RayCaster(core::Device& device, core::Allocator& allocator,
            const RayCasterConfig& config) noexcept;

  // The most rays one dispatch holds on this device.
  std::uint64_t max_rays() const noexcept;

  core::Device& device_;        // borrowed, as an aggregate's are
  core::Allocator& allocator_;  // backs make_view
  const RayCasterConfig config_;
  core::DescriptorPool pool_;  // the kernel's own set; outlives it
  core::ComputeKernel kernel_;
  core::KernelSets sets_;         // casts_per_batch sets of kernel_'s layout
  std::uint32_t next_set_ = 0;    // the set the next record_cast writes
  std::uint32_t max_groups_ = 0;  // maxComputeWorkGroupCount[0]
  VkDeviceSize max_range_ = 0;    // maxStorageBufferRange
};

}  // namespace volumetric_kit::grid::volume
