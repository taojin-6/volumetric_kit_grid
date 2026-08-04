// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file volume/voxel_grid.hpp
/// @brief Voxel-grid resolution + hash-table capacity -- the parameters every
///        voxel-hash operation is measured against.

#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>

#include "volumetric_kit/recon/core/result.hpp"

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
  /// @return A fully-populated parameter set (passes @ref validate).
  static constexpr VoxelGridParams defaults();

  /// @brief Check that every field is a usable value the helpers can trust.
  ///
  /// The coordinate and hash helpers treat these parameters as preconditions --
  /// e.g. @ref hash_bucket divides by `num_buckets`, so a zero bucket count is
  /// undefined. Validate a user-built set once here at configuration time
  /// rather than per-call on the device hot path; @ref defaults always passes.
  /// The two precomputed fields (`voxels_per_block`, `num_blocks`) are checked
  /// against their defining products so a stale value cannot slip through.
  /// @return An OK @ref Status when every field is valid, otherwise
  ///         @ref Status::invalid_argument naming the offending field.
  Status validate() const;
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

// Uploaded to the GPU by value and inspected with offsetof, so it must stay a
// trivially-copyable, standard-layout POD even as fields (or methods) are
// added.
static_assert(std::is_trivially_copyable_v<VoxelGridParams>,
              "VoxelGridParams must be trivially copyable");
static_assert(std::is_standard_layout_v<VoxelGridParams>,
              "VoxelGridParams must be standard-layout");

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

inline Status VoxelGridParams::validate() const {
  if (voxel_size <= 0.0f) {
    return Status::invalid_argument("VoxelGridParams: voxel_size must be > 0");
  }
  if (block_size <= 0) {
    return Status::invalid_argument("VoxelGridParams: block_size must be > 0");
  }
  // Bound the edge *before* cubing it. `voxels_per_block` is a signed 32-bit
  // int, so the largest edge whose cube can be represented is 1290 (1291^3
  // exceeds INT32_MAX) -- and past it the check below would be signed-overflow
  // UB inside the very function whose job is rejecting bad input. It also lets
  // provably-broken values through: 2048^3 wraps to exactly 0, so
  // `voxels_per_block == 0` passes and every block then aliases pointer 0.
  constexpr std::int32_t kMaxBlockSize = 1290;
  if (block_size > kMaxBlockSize) {
    return Status::invalid_argument(
        "VoxelGridParams: block_size must be <= 1290 (block_size^3 must fit a "
        "signed 32-bit voxels_per_block)");
  }
  if (voxels_per_block != block_size * block_size * block_size) {
    return Status::invalid_argument(
        "VoxelGridParams: voxels_per_block must equal block_size^3");
  }
  if (trunc_dist <= 0.0f) {
    return Status::invalid_argument("VoxelGridParams: trunc_dist must be > 0");
  }
  // Two, not one. The last entry of each bucket is that bucket's chain anchor,
  // so at bucket_size == 1 *every* slot in the table is an anchor and there is
  // nowhere for an overflow-chain node to live: `allocate_in_overflow` skips
  // anchors, finds the whole table is anchors, and returns false for every
  // insert past the first collision. That is a degenerate shape rather than a
  // kernel bug, so it is rejected here -- otherwise the first two coords that
  // hash together fail permanently, and a caller reads that as capacity
  // pressure and grows the volume until it runs out of memory.
  if (bucket_size < 2) {
    return Status::invalid_argument(
        "VoxelGridParams: bucket_size must be >= 2 (the last entry of each "
        "bucket is its chain anchor, so a 1-entry bucket has no room for an "
        "overflow chain)");
  }
  if (num_buckets <= 0) {
    return Status::invalid_argument("VoxelGridParams: num_buckets must be > 0");
  }
  // Widened, because the narrow product wraps exactly as the uint32 multiply
  // that produced `num_blocks` in VoxelHashMap::resize does -- so both sides
  // agreed on the wrapped value and the guard could never fire, which also
  // dodged the block-pointer check below it. (bucket_size and num_buckets are
  // positive int32 by the checks above, so their int64 product cannot
  // overflow.)
  if (static_cast<std::int64_t>(bucket_size) * num_buckets != num_blocks) {
    return Status::invalid_argument(
        "VoxelGridParams: num_blocks must equal bucket_size * num_buckets");
  }
  // A block pointer is block_idx * voxels_per_block held as a signed 32-bit int
  // (BlockIndex::ptr); reject a pool whose largest pointer would overflow it,
  // so a pointer stays a valid non-negative offset into the SoA attribute
  // arrays. (resize validates the grown grid, so a grow that would overflow is
  // rejected before it mutates the live map.)
  if (static_cast<std::int64_t>(num_blocks) * voxels_per_block >
      std::numeric_limits<std::int32_t>::max()) {
    return Status::invalid_argument(
        "VoxelGridParams: num_blocks * voxels_per_block must fit a signed "
        "32-bit block pointer");
  }
  if (max_chain <= 0) {
    return Status::invalid_argument("VoxelGridParams: max_chain must be > 0");
  }
  return {};
}

}  // namespace volumetric_kit::recon::volume
