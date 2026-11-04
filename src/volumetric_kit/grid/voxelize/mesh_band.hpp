// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file mesh_band.hpp
/// @brief @ref MeshBand: a triangle mesh on the device and its (triangle,
///        candidate block) decomposition, counted there.
///
/// Internal (under src/, never installed): `MeshIntegrator` owns one. It
/// carries `VGRID_VOXELIZE_API` so a test reaches it in a shared-library
/// build too.

#include <array>
#include <cstdint>
#include <memory>

#include "volumetric_kit/core/base/result.hpp"
#include "volumetric_kit/core/vulkan/buffer.hpp"
#include "volumetric_kit/core/vulkan/compute_kernel.hpp"
#include "volumetric_kit/core/vulkan/descriptor.hpp"
#include "volumetric_kit/core/vulkan/fwd.hpp"
#include "volumetric_kit/grid/base/math/vector_types.hpp"
#include "volumetric_kit/grid/volume/voxel_grid.hpp"
#include "volumetric_kit/grid/voxelize/export.hpp"

#include "mesh_scan.hpp"

namespace volumetric_kit::grid::voxelize::detail {

/// @brief A triangle mesh on the device, with its candidate offsets counted
///        there: what every triangle-driven pass of a conversion dispatches
///        over, one work item per (triangle, candidate block) pair.
///
/// A triangle's candidates are the blocks of its world bounding box grown by
/// `trunc_dist` (`volume/shaders/triangle_box.glsl`). A triangle with a
/// non-finite vertex, or of zero area, is skipped: it counts zero, so
/// `offsets[t] == offsets[t + 1]`, and every other triangle has at least one
/// candidate. A zero-area triangle is the one input that makes the
/// closest-point solve divide by zero, and a mesh file routinely carries a
/// few, so it is dropped rather than refused.
///
/// Use: @ref set_mesh (the host's checks), @ref record_count into a batch the
/// caller submits, then @ref settle_count once it has run. The mesh's
/// buffers are retained and grow to the largest mesh.
///
/// An aggregate of owners: copy and move are deleted, and @ref create returns
/// it behind a `unique_ptr`.
///
/// @warning The `Device` and `Allocator` passed to @ref create must outlive
///          it; it stores pointers to them.
class VGRID_VOXELIZE_API MeshBand {
 public:
  /// The candidates' block range, smallest and largest block per axis; empty
  /// when no triangle has a candidate.
  struct BlockRange {
    Vec3i min{0};
    Vec3i max{0};
    bool empty = true;
  };

  /// @brief Build the count, scan and offset kernels on @p device.
  /// @return The band, holding no mesh; or a pipeline or buffer failure.
  static core::Result<std::unique_ptr<MeshBand>> create(
      core::Device& device, core::Allocator& allocator);

  ~MeshBand();
  MeshBand(const MeshBand&) = delete;
  MeshBand& operator=(const MeshBand&) = delete;
  MeshBand(MeshBand&&) = delete;
  MeshBand& operator=(MeshBand&&) = delete;

  /// @brief Hold a mesh for @ref record_count: the host's checks, and the
  ///        retained buffers grown to it. The arrays are borrowed until
  ///        @ref record_count stages them.
  /// @return OK, or `Status::Code::InvalidArgument`, naming @p who, for a
  ///         null @p vertices or @p indices with triangles to read, an index
  ///         at or past @p vertex_count, or an array past one storage-buffer
  ///         binding; or a buffer failure.
  core::Status set_mesh(const volume::VoxelGridParams& grid,
                        const Vec3f* vertices, std::uint32_t vertex_count,
                        const std::uint32_t* indices,
                        std::uint32_t triangle_count, const char* who);

  /// @brief Record into @p batch the mesh's upload, the count, the scan, the
  ///        offsets, and the read of their result, which @ref settle_count
  ///        reads once @p batch has run.
  /// @param stage  Optional: the caller's scope, whose tile the work counts
  ///               to.
  /// @return OK, or the batch's refusal.
  core::Status record_count(core::CommandBatch& batch,
                            core::GpuStageScope* stage);

  /// @brief Once the batch @ref record_count recorded into has run: the
  ///        device's refusal.
  /// @return OK; or `Status::Code::InvalidArgument`, naming @p who, for a
  ///         candidate total past 2^32 (a mesh grossly mis-scaled against the
  ///         grid -- millimetres read as metres, say), whose total then reads
  ///         0 on the device and here.
  core::Status settle_count(const char* who) const;

  /// @return The held mesh's triangles.
  std::uint32_t triangle_count() const noexcept { return triangle_count_; }
  /// @return The candidate pairs, `offsets[triangle_count]`; settled only.
  std::uint32_t total() const noexcept { return result_.total; }
  /// @return The triangles with a candidate; settled only.
  std::uint32_t used_triangles() const noexcept { return result_.used; }
  /// @return The candidates' block range; settled only.
  BlockRange bounds() const noexcept;

  /// @return Device-local, the mesh's packed `Vec3f` vertices.
  const core::Buffer& vertex_buffer() const noexcept { return vertices_; }
  /// @return Device-local, its `3 * triangle_count` indices.
  const core::Buffer& index_buffer() const noexcept { return indices_; }
  /// @return Device-local, its `triangle_count + 1` candidate offsets.
  const core::Buffer& offset_buffer() const noexcept { return offsets_; }

 private:
  MeshBand() = default;

  core::Status grow(core::Buffer& buffer, VkDeviceSize bytes,
                    const char* name);

  // The count pass's 16-byte result, as triangle_offsets.comp writes it.
  struct CountResult {
    std::uint32_t total = 0;
    std::uint32_t used = 0;
    std::uint32_t refused = 0;
    std::uint32_t largest = 0;
  };

  core::Device* device_ = nullptr;
  core::Allocator* allocator_ = nullptr;
  std::uint32_t max_workgroup_count_x_ = 0;
  VkDeviceSize max_storage_buffer_range_ = 0;

  // Declared before the kernels, so it is destroyed after their sets.
  core::DescriptorPool pool_;
  core::ComputeKernel count_;
  std::unique_ptr<MeshScan> scan_;
  core::ComputeKernel finish_;

  core::Buffer vertices_;
  core::Buffer indices_;
  core::Buffer offsets_;
  core::Buffer flags_;
  core::Buffer result_buffer_;
  core::Buffer bounds_;

  volume::VoxelGridParams grid_{};
  const Vec3f* host_vertices_ = nullptr;
  const std::uint32_t* host_indices_ = nullptr;
  std::uint32_t vertex_count_ = 0;
  std::uint32_t triangle_count_ = 0;
  CountResult result_{};
  std::array<std::int32_t, 6> bounds_host_{};
};

}  // namespace volumetric_kit::grid::voxelize::detail
