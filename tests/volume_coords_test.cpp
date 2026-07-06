// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// Coordinate-math tests for the sparse voxel grid: world <-> voxel <-> block
// round-trips, the negative-coordinate block-indexing bias (the subtle part),
// round-half-to-even tie-breaking, the truncation-band width, and params
// validation. Pure host math -- no device -- so it always runs.

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

  // --- block -> origin (minimum-index) voxel.
  CHECK(eq(vol::block_to_voxel(vr::Vec3i(3, -2, 5), grid), 24, -16, 40));

  // --- world -> continuous voxel coordinate (no rounding).
  const vr::Vec3f vf =
      vol::world_to_voxel_f(vr::Vec3f(0.0026f, -0.0060f, 0.0125f), grid);
  CHECK(vf.x > 0.51f && vf.x < 0.53f);    // 0.0026 / 0.005 = 0.52
  CHECK(vf.y > -1.21f && vf.y < -1.19f);  // -0.006 / 0.005 = -1.2
  CHECK(vf.z > 2.49f && vf.z < 2.51f);    // 0.0125 / 0.005 = 2.5

  // --- world -> voxel rounds to nearest; exact voxel positions land on target.
  CHECK(
      eq(vol::world_to_voxel(vr::Vec3f(0.0f, 0.005f, 0.010f), grid), 0, 1, 2));
  // 0.52 voxels -> 1; -0.48 voxels -> 0 (nearest, not toward -inf).
  CHECK(eq(vol::world_to_voxel(vr::Vec3f(0.0026f, -0.0024f, 0.0f), grid), 1, 0,
           0));

  // --- world -> voxel rounds half-to-even, independent of the FP rounding
  // mode. A power-of-two voxel size makes world/voxel_size an exact half (0.005
  // m is not exact in float, which would perturb the tie). Ties go to the even
  // neighbour: 0.5->0, 1.5->2, 2.5->2, -0.5->0, -1.5->-2.
  vol::VoxelGridParams po2 = grid;
  po2.voxel_size = 0.5f;
  CHECK(eq(vol::world_to_voxel(vr::Vec3f(0.25f, 0.75f, 1.25f), po2), 0, 2, 2));
  CHECK(
      eq(vol::world_to_voxel(vr::Vec3f(-0.25f, -0.75f, 0.0f), po2), 0, -2, 0));

  // --- world -> block composes the two, including the negative path e2e.
  CHECK(eq(vol::world_to_block(vr::Vec3f(0.041f, 0.0f, 0.0f), grid), 1, 0, 0));
  CHECK(
      eq(vol::world_to_block(vr::Vec3f(-0.005f, 0.0f, 0.0f), grid), -1, 0, 0));

  // --- voxel -> world centre, directly and for negative coords (voxel-centre
  // convention: voxel i sits at i * voxel_size).
  const vr::Vec3f w = vol::voxel_to_world(vr::Vec3i(2, -3, 5), grid);
  CHECK(w.x > 0.00999f && w.x < 0.01001f);    // 2 * 0.005 = 0.010
  CHECK(w.y > -0.01501f && w.y < -0.01499f);  // -3 * 0.005 = -0.015
  CHECK(w.z > 0.02499f && w.z < 0.02501f);    // 5 * 0.005 = 0.025

  // --- block -> world: block (1,0,0) -> origin voxel 8 -> 0.04 m (the origin
  // voxel's centre under the voxel-centre convention).
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

  // --- VoxelGridParams::validate: the defaults are valid; a zero bucket count
  // (which would make hash_bucket divide by zero) is rejected, as is a stale
  // precomputed field.
  CHECK(grid.validate().ok());
  vol::VoxelGridParams bad = grid;
  bad.num_buckets = 0;
  CHECK(!bad.validate().ok());
  bad = grid;
  bad.voxels_per_block = 511;  // != 8^3, a stale precomputed value.
  CHECK(!bad.validate().ok());

  std::printf("recon volume coords test passed\n");
  return 0;
}
