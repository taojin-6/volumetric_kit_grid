// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file mesh_scan.hpp
/// @brief @ref MeshScan: voxelize's 256-way exclusive scan of `uint32` counts
///        (`mesh_scan_*.comp`), for the band's candidate offsets and the
///        write's bins. Internal (under src/, never installed).

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>

#include "volumetric_kit/core/base/result.hpp"
#include "volumetric_kit/core/vulkan/buffer.hpp"
#include "volumetric_kit/core/vulkan/compute_kernel.hpp"
#include "volumetric_kit/core/vulkan/descriptor.hpp"
#include "volumetric_kit/core/vulkan/fwd.hpp"

namespace volumetric_kit::grid::voxelize::detail {

/// @brief One scan's kernels and levels. @ref record leaves each count as its
///        exclusive prefix within its group of 256, @ref first the groups'
///        exclusive prefixes, and @ref root one summary of the whole: the sum,
///        the non-zero counts, the largest, and a sticky carry past 2^32
///        (`mesh_scan.glsl`).
///
/// An aggregate of owners: copy and move are deleted, and @ref create returns
/// it behind a `unique_ptr`.
///
/// @warning The `Device` and `Allocator` passed to @ref create must outlive
///          it.
class MeshScan {
 public:
  /// The scan's summaries are 16 bytes an element.
  static constexpr VkDeviceSize kSummaryBytes = 16;

  /// @brief Build the scan's kernels on @p device, named @p name; its levels
  ///        are named @p name too.
  static core::Result<std::unique_ptr<MeshScan>> create(
      core::Device& device, core::Allocator& allocator, const char* name);

  ~MeshScan();
  MeshScan(const MeshScan&) = delete;
  MeshScan& operator=(const MeshScan&) = delete;
  MeshScan(MeshScan&&) = delete;
  MeshScan& operator=(MeshScan&&) = delete;

  /// @brief Size the levels for @p size counts and bind the kernels to
  ///        @p values: buffers grown as core's `ensure_device_scratch` grows
  ///        them.
  core::Status prepare(const core::Buffer& values, std::uint32_t size);

  /// @brief Record the scan of the @p size counts @ref prepare bound.
  core::Status record(core::CommandBatch& batch, std::uint32_t size,
                      core::GpuStageScope* stage) const;

  /// @return The groups' exclusive prefixes.
  const core::Buffer& first() const noexcept { return levels_[0]; }
  /// @return The one summary of the whole.
  const core::Buffer& root() const noexcept { return levels_[root_level_]; }

 private:
  MeshScan() = default;

  core::Device* device_ = nullptr;
  core::Allocator* allocator_ = nullptr;
  const char* name_ = nullptr;
  std::uint32_t max_workgroup_count_x_ = 0;
  VkDeviceSize max_storage_buffer_range_ = 0;
  // Declared before the kernels, so it is destroyed after their sets.
  core::DescriptorPool pool_;
  core::ComputeKernel counts_;
  std::array<core::ComputeKernel, 3> sums_;
  std::array<core::ComputeKernel, 2> add_;
  std::array<core::Buffer, 4> levels_;
  std::array<std::uint32_t, 4> level_sizes_{};
  std::size_t root_level_ = 0;
};

}  // namespace volumetric_kit::grid::voxelize::detail
