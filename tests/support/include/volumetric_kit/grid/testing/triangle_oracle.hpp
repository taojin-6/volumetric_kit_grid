// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

// The host oracle of a mesh's (triangle, candidate block) decomposition, the
// rules voxelize's triangle_counts.comp counts by: each triangle's candidate
// blocks counted and summed into exclusive offsets, as
// `MeshIntegrator::convert` counts them on the device.

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#include "volumetric_kit/core/base/result.hpp"
#include "volumetric_kit/grid/base/math/vector_types.hpp"
#include "volumetric_kit/grid/volume/voxel_coords.hpp"
#include "volumetric_kit/grid/volume/voxel_grid.hpp"

namespace vgrid_test {

namespace detail {

// Every component finite. A mesh vertex is caller data, and one NaN turns a
// bounding box into a block range of garbage extent -- so it is tested before
// the box is formed rather than after.
inline bool all_finite(volumetric_kit::grid::Vec3f v) {
  return std::isfinite(v.x) && std::isfinite(v.y) && std::isfinite(v.z);
}

// Whether a triangle has zero area: |cross(v1 - v0, v2 - v0)|^2 <= 0, glm's
// cross and dot in their order, each product rounded on its own -- what
// triangle_counts.comp computes `precise`. No multiply-add may be fused here:
// arm64 compilers fuse by default, and a fused `a*b - b*a` keeps the first
// product's rounding error, so a triangle with two equal vertices would read
// as having area on the host and none on the device.
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC push_options
#pragma GCC optimize("fp-contract=off")
#endif
inline bool zero_area(volumetric_kit::grid::Vec3f v0,
                      volumetric_kit::grid::Vec3f v1,
                      volumetric_kit::grid::Vec3f v2) {
#if defined(__clang__)
#pragma clang fp contract(off)
#endif
  const volumetric_kit::grid::Vec3f a = v1 - v0;
  const volumetric_kit::grid::Vec3f b = v2 - v0;
  const float nx = a.y * b.z - b.y * a.z;
  const float ny = a.z * b.x - b.z * a.x;
  const float nz = a.x * b.y - b.x * a.y;
  return nx * nx + ny * ny + nz * nz <= 0.0f;
}
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC pop_options
#endif

}  // namespace detail

// Count each triangle's candidate blocks -- the blocks of its world bounding
// box grown by trunc_dist -- as an exclusive prefix sum: triangle_count + 1
// offsets, the last the total. A triangle with a non-finite vertex, or of zero
// area, counts zero. Refuses a null array with triangles to read, an index at
// or past vertex_count, and a total past 2^32, naming `who`.
inline volumetric_kit::core::Result<std::vector<std::uint32_t>>
triangle_candidate_offsets(
    const volumetric_kit::grid::volume::VoxelGridParams& grid,
    const volumetric_kit::grid::Vec3f* vertices, std::uint32_t vertex_count,
    const std::uint32_t* indices, std::uint32_t triangle_count,
    const char* who) {
  namespace vgrid = volumetric_kit::grid;
  namespace vkc = volumetric_kit::core;
  const std::string name = who != nullptr ? who : "triangle_candidate_offsets";
  if (triangle_count != 0 && (vertices == nullptr || indices == nullptr)) {
    return vkc::Status::invalid_argument(name +
                                         ": vertices or indices is null");
  }
  std::vector<std::uint32_t> offsets(std::size_t{triangle_count} + 1, 0u);
  std::uint64_t total = 0;
  for (std::uint32_t t = 0; t < triangle_count; ++t) {
    offsets[t] = static_cast<std::uint32_t>(total);
    const std::uint32_t i0 = indices[3 * std::size_t{t}];
    const std::uint32_t i1 = indices[3 * std::size_t{t} + 1];
    const std::uint32_t i2 = indices[3 * std::size_t{t} + 2];
    if (i0 >= vertex_count || i1 >= vertex_count || i2 >= vertex_count) {
      return vkc::Status::invalid_argument(
          name + ": a triangle index is at or past vertex_count");
    }
    const vgrid::Vec3f v0 = vertices[i0];
    const vgrid::Vec3f v1 = vertices[i1];
    const vgrid::Vec3f v2 = vertices[i2];
    if (!detail::all_finite(v0) || !detail::all_finite(v1) ||
        !detail::all_finite(v2) || detail::zero_area(v0, v1, v2)) {
      continue;
    }
    const vgrid::Vec3f lo =
        glm::min(v0, glm::min(v1, v2)) - vgrid::Vec3f(grid.trunc_dist);
    const vgrid::Vec3f hi =
        glm::max(v0, glm::max(v1, v2)) + vgrid::Vec3f(grid.trunc_dist);
    const vgrid::Vec3i bmin = vgrid::volume::world_to_block(lo, grid);
    const vgrid::Vec3i bmax = vgrid::volume::world_to_block(hi, grid);
    total += static_cast<std::uint64_t>(bmax.x - bmin.x + 1) *
             static_cast<std::uint64_t>(bmax.y - bmin.y + 1) *
             static_cast<std::uint64_t>(bmax.z - bmin.z + 1);
    if (total > std::numeric_limits<std::uint32_t>::max()) {
      return vkc::Status::invalid_argument(
          name +
          ": candidate blocks exceed 2^32 -- is the mesh scaled to the grid's "
          "units (metres)?");
    }
  }
  offsets[triangle_count] = static_cast<std::uint32_t>(total);
  return offsets;
}

}  // namespace vgrid_test
