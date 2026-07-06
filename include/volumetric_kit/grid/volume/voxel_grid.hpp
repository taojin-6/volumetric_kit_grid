// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file volume/voxel_grid.hpp
/// @brief Voxel-grid resolution + hash-table capacity -- the parameters every
///        voxel-hash operation is measured against.

#include <cstddef>
#include <cstdint>

#include "volumetric_kit/recon/core/device_macros.hpp"

namespace volumetric_kit::recon::volume {

/// @brief Grid resolution and hash-table shape: metres-per-voxel, block size,
///        truncation band, and the hash-table capacity.
///
/// Passed *explicitly* to the coordinate and hash helpers rather than read from
/// a process-global singleton (the prior engine's `d_scene`/`h_scene`) -- the
/// god-object is dropped on port so each operation states its dependency. On
/// the device the compute shaders will read this as a uniform block through
/// scalar block layout (the 2026-07-05 ABI); it is an all-4-byte-scalar struct,
/// so the host packing here is byte-identical to the GLSL mirror (no `vec3` to
/// 16-byte-align). Ported from the prior engine's `SceneParams`, scoped to what
/// the `volume` tier owns -- the TSDF, meshing, and depth knobs live with their
/// tiers.
struct VoxelGridParams {
  float voxel_size;               ///< Metres per voxel edge.
  std::int32_t block_size;        ///< Voxels per block edge (e.g. 8).
  std::int32_t voxels_per_block;  ///< `block_size^3` (precomputed).
  float trunc_dist;               ///< TSDF truncation distance, metres.
  std::int32_t bucket_size;       ///< Hash entries per bucket.
  std::int32_t num_buckets;       ///< Number of hash buckets.
  std::int32_t num_blocks;        ///< `bucket_size * num_buckets` (heap size).
  std::int32_t max_chain;         ///< Max collision linked-list length.

  /// @brief The ported production defaults: 5 mm voxels, 8-voxel blocks, a 40
  /// mm
  ///        truncation band, and a 50 x 30000 hash table.
  static constexpr VoxelGridParams defaults();
};

// All-scalar, tightly packed: pin the size + offsets so the struct stays the
// byte-for-byte ABI the compute shaders will mirror (a drift is a compile
// error, not silent buffer corruption).
static_assert(sizeof(VoxelGridParams) == 32,
              "VoxelGridParams must be 32 bytes");
static_assert(offsetof(VoxelGridParams, voxel_size) == 0, "layout drift");
static_assert(offsetof(VoxelGridParams, block_size) == 4, "layout drift");
static_assert(offsetof(VoxelGridParams, voxels_per_block) == 8, "layout drift");
static_assert(offsetof(VoxelGridParams, trunc_dist) == 12, "layout drift");
static_assert(offsetof(VoxelGridParams, bucket_size) == 16, "layout drift");
static_assert(offsetof(VoxelGridParams, num_buckets) == 20, "layout drift");
static_assert(offsetof(VoxelGridParams, num_blocks) == 24, "layout drift");
static_assert(offsetof(VoxelGridParams, max_chain) == 28, "layout drift");

constexpr VoxelGridParams VoxelGridParams::defaults() {
  constexpr std::int32_t kBlockSize = 8;
  constexpr std::int32_t kBucketSize = 50;
  constexpr std::int32_t kNumBuckets = 30000;
  return VoxelGridParams{
      /*voxel_size=*/0.005f,
      /*block_size=*/kBlockSize,
      /*voxels_per_block=*/kBlockSize * kBlockSize * kBlockSize,
      /*trunc_dist=*/0.04f,
      /*bucket_size=*/kBucketSize,
      /*num_buckets=*/kNumBuckets,
      /*num_blocks=*/kBucketSize * kNumBuckets,
      /*max_chain=*/128,
  };
}

}  // namespace volumetric_kit::recon::volume
