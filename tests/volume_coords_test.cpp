// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// Coordinate-math tests for the sparse voxel grid: world <-> voxel <-> block
// round-trips, the negative-coordinate block-indexing bias (the subtle part),
// and the truncation-band width. Pure host math -- no device -- so it always
// runs.

#include <cstdio>

#include "volumetric_kit/recon/core/math/vector_types.hpp"
#include "volumetric_kit/recon/volume/voxel_coords.hpp"
#include "volumetric_kit/recon/volume/voxel_grid.hpp"

namespace vr = volumetric_kit::recon;
namespace vol = volumetric_kit::recon::volume;

#define CHECK(cond)                                                        \
  do {                                                                     \
    if (!(cond)) {                                                         \
      std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
      return 1;                                                            \
    }                                                                      \
  } while (0)

namespace {

bool eq(const vr::Vec3i& a, int x, int y, int z) {
  return a.x == x && a.y == y && a.z == z;
}

}  // namespace

int main() {
  // Defaults: voxel_size 0.005 m, block_size 8, trunc_dist 0.04 m.
  const vol::VoxelGridParams grid = vol::VoxelGridParams::defaults();

  // --- voxel -> block, including the negative-coordinate bias. Block b spans
  // voxels [b*8, b*8+7], so voxel -1 is block -1, not block 0.
  CHECK(eq(vol::voxel_to_block(vr::Vec3i(0, 0, 0), grid), 0, 0, 0));
  CHECK(eq(vol::voxel_to_block(vr::Vec3i(7, 7, 7), grid), 0, 0, 0));
  CHECK(eq(vol::voxel_to_block(vr::Vec3i(8, 8, 8), grid), 1, 1, 1));
  CHECK(eq(vol::voxel_to_block(vr::Vec3i(-1, -1, -1), grid), -1, -1, -1));
  CHECK(eq(vol::voxel_to_block(vr::Vec3i(-8, -8, -8), grid), -1, -1, -1));
  CHECK(eq(vol::voxel_to_block(vr::Vec3i(-9, -9, -9), grid), -2, -2, -2));

  // --- block -> voxel corner.
  CHECK(eq(vol::block_to_voxel(vr::Vec3i(3, -2, 5), grid), 24, -16, 40));

  // --- world -> voxel rounds to nearest; exact voxel positions land on target.
  CHECK(
      eq(vol::world_to_voxel(vr::Vec3f(0.0f, 0.005f, 0.010f), grid), 0, 1, 2));
  // 0.52 voxels -> 1; -0.48 voxels -> 0 (nearest, not toward -inf).
  CHECK(eq(vol::world_to_voxel(vr::Vec3f(0.0026f, -0.0024f, 0.0f), grid), 1, 0,
           0));

  // --- world -> block composes the two, including the negative path e2e.
  CHECK(eq(vol::world_to_block(vr::Vec3f(0.041f, 0.0f, 0.0f), grid), 1, 0, 0));
  CHECK(
      eq(vol::world_to_block(vr::Vec3f(-0.005f, 0.0f, 0.0f), grid), -1, 0, 0));

  // --- block_to_world corner: block (1,0,0) -> voxel 8 -> 0.04 m.
  const vr::Vec3f corner = vol::block_to_world(vr::Vec3i(1, 0, 0), grid);
  CHECK(corner.x > 0.03999f && corner.x < 0.04001f);
  CHECK(corner.y == 0.0f && corner.z == 0.0f);

  // --- truncation band. Default: 0.04 / (8 * 0.005) = 1.0 -> 1 block.
  CHECK(vol::truncation_blocks(grid) == 1);
  vol::VoxelGridParams wide = grid;
  wide.trunc_dist = 0.10f;  // 0.10 / 0.04 = 2.5 -> ceil -> 3.
  CHECK(vol::truncation_blocks(wide) == 3);
  wide.trunc_dist = 0.0f;  // clamped to a minimum of 1.
  CHECK(vol::truncation_blocks(wide) == 1);

  std::printf("recon volume coords test passed\n");
  return 0;
}
