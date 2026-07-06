// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file volume/voxel_coords.hpp
/// @brief World <-> voxel <-> block coordinate transforms for the sparse grid.
///
/// The conversions every voxel-hash stage shares (allocation, integration,
/// meshing). Each takes the @ref VoxelGridParams explicitly and is
/// `VR_DEVICE_HOST`, so the same definition serves host code and the CUDA
/// accelerator; the GLSL compute mirror keeps identical arithmetic.

#include <cmath>

#include "volumetric_kit/recon/core/device_macros.hpp"
#include "volumetric_kit/recon/core/math/vector_types.hpp"
#include "volumetric_kit/recon/volume/voxel_grid.hpp"

namespace volumetric_kit::recon::volume {

namespace detail {

/// @brief Round to nearest integer, ties to even -- independent of the current
///        floating-point rounding mode.
///
/// `std::rint`/`std::nearbyint` honour the dynamic rounding mode
/// (`fesetround`), so a host caller that changed it would round differently
/// from the fixed round-half-to-even of the device path (`__float2int_rn`) and
/// the GLSL mirror
/// (`roundEven`) -- disagreeing on which voxel/block a sample owns. This
/// derives ties-to-even from `std::floor` (which ignores the mode), keeping
/// host, CUDA, and GLSL byte-identical in any FP environment.
/// @param x  The value to round.
/// @return The nearest integer value to @p x; an exact half rounds to even.
VR_DEVICE_HOST inline float round_half_even(float x) {
  const float lower = std::floor(x);
  const float frac = x - lower;
  if (frac < 0.5f) return lower;
  if (frac > 0.5f) return lower + 1.0f;
  return std::fmod(lower, 2.0f) == 0.0f ? lower : lower + 1.0f;  // tie -> even
}

}  // namespace detail

/// @brief World position -> continuous voxel coordinate (no rounding).
/// @param world  World-space position, metres.
/// @param grid   Grid parameters (uses @ref VoxelGridParams::voxel_size).
/// @return The fractional voxel coordinate of @p world.
VR_DEVICE_HOST inline Vec3f world_to_voxel_f(Vec3f world,
                                             const VoxelGridParams& grid) {
  return world / grid.voxel_size;
}

/// @brief World position -> nearest integer voxel coordinate.
///
/// Rounds half-to-even, matching the device path's `__float2int_rn` and the
/// GLSL `roundEven` the shader mirror uses. Uses a rounding-mode-independent
/// round-half-to-even (@ref detail::round_half_even) rather than `std::rint`,
/// so host, CUDA, and GLSL agree regardless of the caller's FP rounding mode.
/// @param world  World-space position, metres.
/// @param grid   Grid parameters (uses @ref VoxelGridParams::voxel_size).
/// @return The nearest voxel coordinate to @p world.
VR_DEVICE_HOST inline Vec3i world_to_voxel(Vec3f world,
                                           const VoxelGridParams& grid) {
  const Vec3f v = world / grid.voxel_size;
  return Vec3i(static_cast<int>(detail::round_half_even(v.x)),
               static_cast<int>(detail::round_half_even(v.y)),
               static_cast<int>(detail::round_half_even(v.z)));
}

/// @brief Voxel coordinate -> the block that contains it.
///
/// Block indexing floors toward negative infinity, but C++ integer division
/// truncates toward zero -- so negative coordinates are biased by
/// `block_size - 1` first. Without it, voxel -1 (which belongs to block -1,
/// spanning voxels -8..-1) would land in block 0.
/// @param voxel  Voxel coordinate.
/// @param grid   Grid parameters (uses @ref VoxelGridParams::block_size).
/// @return The block coordinate containing @p voxel.
VR_DEVICE_HOST inline Vec3i voxel_to_block(Vec3i voxel,
                                           const VoxelGridParams& grid) {
  const int bs = grid.block_size;
  if (voxel.x < 0) voxel.x -= bs - 1;
  if (voxel.y < 0) voxel.y -= bs - 1;
  if (voxel.z < 0) voxel.z -= bs - 1;
  return Vec3i(voxel.x / bs, voxel.y / bs, voxel.z / bs);
}

/// @brief Block coordinate -> its origin (minimum-index) voxel.
/// @param block  Block coordinate.
/// @param grid   Grid parameters (uses @ref VoxelGridParams::block_size).
/// @return The coordinate of the block's minimum-index (origin) voxel.
VR_DEVICE_HOST inline Vec3i block_to_voxel(Vec3i block,
                                           const VoxelGridParams& grid) {
  return block * grid.block_size;
}

/// @brief Voxel coordinate -> world position of the voxel centre.
///
/// Voxel-centre convention: voxel `i` is centred at `i * voxel_size`,
/// consistent with @ref world_to_voxel rounding to the nearest voxel.
/// @param voxel  Voxel coordinate.
/// @param grid   Grid parameters (uses @ref VoxelGridParams::voxel_size).
/// @return The world-space centre of @p voxel, metres.
VR_DEVICE_HOST inline Vec3f voxel_to_world(Vec3i voxel,
                                           const VoxelGridParams& grid) {
  return Vec3f(voxel) * grid.voxel_size;
}

/// @brief Block coordinate -> world position of its origin voxel's centre.
///
/// Composes @ref block_to_voxel and @ref voxel_to_world, so this is the centre
/// of the block's minimum-index voxel under the voxel-centre convention -- not
/// the block's geometric minimum corner, which sits half a voxel lower.
/// @param block  Block coordinate.
/// @param grid   Grid parameters.
/// @return The world-space centre of the block's origin voxel, metres.
VR_DEVICE_HOST inline Vec3f block_to_world(Vec3i block,
                                           const VoxelGridParams& grid) {
  return voxel_to_world(block_to_voxel(block, grid), grid);
}

/// @brief World position -> the block that contains it.
///
/// Composes @ref world_to_voxel (nearest voxel) and @ref voxel_to_block.
/// @param world  World-space position, metres.
/// @param grid   Grid parameters.
/// @return The block coordinate containing @p world.
VR_DEVICE_HOST inline Vec3i world_to_block(Vec3f world,
                                           const VoxelGridParams& grid) {
  return voxel_to_block(world_to_voxel(world, grid), grid);
}

/// @brief Truncation band half-width, in blocks (at least 1).
///
/// The TSDF truncation distance @ref VoxelGridParams::trunc_dist (metres)
/// expressed as a whole number of blocks, so allocation can expand a centre
/// block into the surrounding band it will integrate into.
/// @param grid  Grid parameters (uses block_size, voxel_size, trunc_dist).
/// @return The truncation half-width in blocks, clamped to at least 1.
VR_DEVICE_HOST inline int truncation_blocks(const VoxelGridParams& grid) {
  const float block_extent =
      static_cast<float>(grid.block_size) * grid.voxel_size;
  const int blocks =
      static_cast<int>(std::ceil(grid.trunc_dist / block_extent));
  return blocks > 1 ? blocks : 1;
}

}  // namespace volumetric_kit::recon::volume
