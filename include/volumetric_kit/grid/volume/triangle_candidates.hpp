// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file volume/triangle_candidates.hpp
/// @brief The work decomposition every triangle-driven kernel dispatches over:
///        one work item per (triangle, candidate block) pair.

#include <cstdint>
#include <vector>

#include "volumetric_kit/core/base/result.hpp"
#include "volumetric_kit/recon/core/math/vector_types.hpp"
#include "volumetric_kit/recon/volume/export.hpp"
#include "volumetric_kit/recon/volume/voxel_grid.hpp"

namespace volumetric_kit::recon::volume {

/// @brief Count each triangle's candidate blocks, as an exclusive prefix sum.
///
/// A triangle's candidates are the blocks of its world bounding box grown by
/// `trunc_dist`. A kernel handed these offsets runs one work item per
/// candidate, finds its triangle by binary search, and decodes the block with
/// `volume/shaders/triangle_candidates.glsl`, which recomputes the box from the
/// vertices bit-for-bit. That shader also holds the band test that prunes a
/// candidate to the blocks the band actually reaches.
/// @ref VoxelHashMap::allocate_from_triangles allocates over this
/// decomposition and `tsdf::MeshIntegrator` bins triangles over it. Sharing
/// one definition is what guarantees the integrator finds every block the
/// allocation made.
///
/// Work per item stays bounded however large a triangle is, which is why the
/// unit is the block and not the triangle (the 2026-08-31 decision).
///
/// A triangle with a non-finite vertex, or of zero area, is **skipped**: it
/// gets a zero count, and so `offsets[t] == offsets[t + 1]`. Every other
/// triangle has at least one candidate, so a non-empty range is exactly
/// "this triangle is used". A zero-area triangle is the one input that makes
/// the closest-point solve divide by zero, and a mesh file routinely carries a
/// few, so it is dropped rather than refused.
///
/// This is also the only place an index is bounds-checked. The kernels index
/// the vertex buffer with it directly, and `robustBufferAccess` is enabled
/// nowhere in this repo.
/// @param grid            The block lattice and `trunc_dist` to draw
///                        candidates on.
/// @param vertices        World-space vertex positions, metres.
/// @param vertex_count    How many @p vertices.
/// @param indices         `3 * triangle_count` indices into @p vertices.
/// @param triangle_count  How many triangles.
/// @param who             Names the caller, for the error message (null names
///                        this function instead).
/// @return `triangle_count + 1` offsets, the last being the total work-item
///         count (0 when every triangle was skipped); or
///         `Status::Code::InvalidArgument` for a null @p vertices or
///         @p indices with triangles to read, an index at or past
///         @p vertex_count, or a total past 2^32 (a mesh grossly mis-scaled
///         against the grid -- millimetres read as metres, say).
VR_VOLUME_API core::Result<std::vector<std::uint32_t>>
triangle_candidate_offsets(const VoxelGridParams& grid, const Vec3f* vertices,
                           std::uint32_t vertex_count,
                           const std::uint32_t* indices,
                           std::uint32_t triangle_count, const char* who);

}  // namespace volumetric_kit::recon::volume
