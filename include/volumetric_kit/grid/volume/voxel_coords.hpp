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

/// @brief World position -> continuous voxel coordinate (no rounding).
VR_DEVICE_HOST inline Vec3f world_to_voxel_f(Vec3f world,
                                             const VoxelGridParams& grid) {
  return world / grid.voxel_size;
}

/// @brief World position -> nearest integer voxel coordinate.
///
/// Rounds half-to-even (`std::rint` under the default rounding mode), matching
/// the prior engine's `__float2int_rn` / `rint` and the GLSL `roundEven` the
/// shader mirror will use.
VR_DEVICE_HOST inline Vec3i world_to_voxel(Vec3f world,
                                           const VoxelGridParams& grid) {
  const Vec3f v = world / grid.voxel_size;
  return Vec3i(static_cast<int>(std::rint(v.x)),
               static_cast<int>(std::rint(v.y)),
               static_cast<int>(std::rint(v.z)));
}

/// @brief Voxel coordinate -> the block that contains it.
///
/// Block indexing floors toward negative infinity, but C++ integer division
/// truncates toward zero -- so negative coordinates are biased by
/// `block_size - 1` first. Without it, voxel -1 (which belongs to block -1,
/// spanning voxels -8..-1) would land in block 0.
VR_DEVICE_HOST inline Vec3i voxel_to_block(Vec3i voxel,
                                           const VoxelGridParams& grid) {
  const int bs = grid.block_size;
  if (voxel.x < 0) voxel.x -= bs - 1;
  if (voxel.y < 0) voxel.y -= bs - 1;
  if (voxel.z < 0) voxel.z -= bs - 1;
  return Vec3i(voxel.x / bs, voxel.y / bs, voxel.z / bs);
}

/// @brief Block coordinate -> the voxel at its minimum corner.
VR_DEVICE_HOST inline Vec3i block_to_voxel(Vec3i block,
                                           const VoxelGridParams& grid) {
  return block * grid.block_size;
}

/// @brief Voxel coordinate -> world position (voxel-centre convention).
VR_DEVICE_HOST inline Vec3f voxel_to_world(Vec3i voxel,
                                           const VoxelGridParams& grid) {
  return Vec3f(voxel) * grid.voxel_size;
}

/// @brief Block coordinate -> world position of its minimum corner.
VR_DEVICE_HOST inline Vec3f block_to_world(Vec3i block,
                                           const VoxelGridParams& grid) {
  return voxel_to_world(block_to_voxel(block, grid), grid);
}

/// @brief World position -> the block that contains it.
VR_DEVICE_HOST inline Vec3i world_to_block(Vec3f world,
                                           const VoxelGridParams& grid) {
  return voxel_to_block(world_to_voxel(world, grid), grid);
}

/// @brief Truncation band half-width, in blocks (at least 1).
///
/// The TSDF truncation distance @ref VoxelGridParams::trunc_dist (metres)
/// expressed as a whole number of blocks, so allocation can expand a centre
/// block into the surrounding band it will integrate into.
VR_DEVICE_HOST inline int truncation_blocks(const VoxelGridParams& grid) {
  const float block_extent =
      static_cast<float>(grid.block_size) * grid.voxel_size;
  const int blocks =
      static_cast<int>(std::ceil(grid.trunc_dist / block_extent));
  return blocks > 1 ? blocks : 1;
}

}  // namespace volumetric_kit::recon::volume
