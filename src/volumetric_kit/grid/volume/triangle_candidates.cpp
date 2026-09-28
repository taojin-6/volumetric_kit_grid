// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include "volumetric_kit/recon/volume/triangle_candidates.hpp"

#include <cmath>
#include <cstddef>
#include <limits>
#include <string>

#include "volumetric_kit/recon/volume/voxel_coords.hpp"

namespace volumetric_kit::recon::volume {
namespace {

// Every component finite. A mesh vertex is caller data, and one NaN turns a
// bounding box into a block range of garbage extent -- so it is tested before
// the box is formed rather than after.
bool all_finite(Vec3f v) {
  return std::isfinite(v.x) && std::isfinite(v.y) && std::isfinite(v.z);
}

}  // namespace

Result<std::vector<std::uint32_t>> triangle_candidate_offsets(
    const VoxelGridParams& grid, const Vec3f* vertices,
    std::uint32_t vertex_count, const std::uint32_t* indices,
    std::uint32_t triangle_count, const char* who) {
  std::vector<std::uint32_t> offsets(std::size_t{triangle_count} + 1, 0u);
  if (triangle_count == 0) {
    return offsets;
  }
  if (vertices == nullptr || indices == nullptr) {
    return Status::invalid_argument(std::string(who) +
                                    ": vertices or indices is null");
  }

  std::uint64_t total = 0;
  for (std::uint32_t t = 0; t < triangle_count; ++t) {
    offsets[t] = static_cast<std::uint32_t>(total);

    const std::uint32_t i0 = indices[3 * std::size_t{t}];
    const std::uint32_t i1 = indices[3 * std::size_t{t} + 1];
    const std::uint32_t i2 = indices[3 * std::size_t{t} + 2];
    if (i0 >= vertex_count || i1 >= vertex_count || i2 >= vertex_count) {
      return Status::invalid_argument(
          std::string(who) + ": a triangle index is at or past vertex_count");
    }
    const Vec3f v0 = vertices[i0];
    const Vec3f v1 = vertices[i1];
    const Vec3f v2 = vertices[i2];

    // Skipped triangles keep their offsets[t] and contribute no work item; the
    // kernel's search steps over the resulting zero-length run. A non-finite
    // vertex has no bounding box, and a zero-area one is what would divide by
    // zero in the closest-point solve.
    if (!all_finite(v0) || !all_finite(v1) || !all_finite(v2)) {
      continue;
    }
    const Vec3f normal = cross(v1 - v0, v2 - v0);
    if (dot(normal, normal) <= 0.0f) {
      continue;
    }

    // Mirrored bit-for-bit by vrCandidateBlock, which recomputes rather than
    // reads this back -- see triangle_candidates.glsl.
    const Vec3f lo = glm::min(v0, glm::min(v1, v2)) - Vec3f(grid.trunc_dist);
    const Vec3f hi = glm::max(v0, glm::max(v1, v2)) + Vec3f(grid.trunc_dist);
    const Vec3i bmin = world_to_block(lo, grid);
    const Vec3i bmax = world_to_block(hi, grid);
    const std::uint64_t n = static_cast<std::uint64_t>(bmax.x - bmin.x + 1) *
                            static_cast<std::uint64_t>(bmax.y - bmin.y + 1) *
                            static_cast<std::uint64_t>(bmax.z - bmin.z + 1);

    total += n;
    // One work item per candidate block, so the total is the dispatch width.
    // Reaching 2^32 of them means the mesh is enormously out of scale with the
    // grid -- a model in millimetres read as metres is the usual cause -- and
    // saying so beats truncating the count and covering a fraction of it.
    if (total > std::numeric_limits<std::uint32_t>::max()) {
      return Status::invalid_argument(
          std::string(who) +
          ": candidate blocks exceed 2^32 -- is the mesh scaled to the grid's "
          "units (metres)?");
    }
  }
  offsets[triangle_count] = static_cast<std::uint32_t>(total);
  return offsets;
}

}  // namespace volumetric_kit::recon::volume
