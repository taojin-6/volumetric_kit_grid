// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// Shared definitions for the voxel-hash compute kernels: the device struct
// layouts (scalar block layout, byte-identical to the host POD structs in
// volume/hash_types.hpp and volume/voxel_grid.hpp), the hash-table constants,
// the push-constant block, and the spatial hash. #included by hash_init.comp,
// hash_allocate_coords.comp, and hash_compact.comp.
//
// Buffer bindings and the SSBO-touching helpers are NOT here: GLSL buffer blocks
// are global and cannot be passed to functions, so each kernel declares only the
// buffers it uses, and the lock/heap/allocate helpers live in the allocate
// kernel that needs them.

#extension GL_EXT_scalar_block_layout : require

// --- Device struct layouts. Under scalar block layout these are byte-identical
// to the host POD structs (HashEntry: ptr@0, ivec3 pos@4, offset@16, 20 B),
// which is what lets one definition serve CPU / GLSL / CUDA (2026-07-05 ABI). ---

struct VoxelGridParams {
  float voxel_size;
  int block_size;
  int voxels_per_block;
  float trunc_dist;
  int bucket_size;
  int num_buckets;
  int num_blocks;
  int max_chain;
};

struct HashEntry {
  int ptr;
  ivec3 pos;
  int offset;
};

struct BlockIndex {
  ivec3 coord;
  int ptr;
};

// --- Hash-table constants (mirror volume/hash.hpp). ---
const int kFreeEntry = -1;
const int kLockEntry = -2;
const int kNoOffset = 0;

const uint kHashPrimeX = 73856093u;
const uint kHashPrimeY = 19349669u;
const uint kHashPrimeZ = 83492791u;

// Spin/retry caps (mirror hash_ops.metal): bounded so a contended dispatch
// cannot hang the GPU; the host retries any failed allocations across dispatches.
const int kMaxSpinRetries = 128;
const int kMaxHeapRetries = 256;

// --- Push constants: the grid shape + one kernel-specific argument. ---
layout(push_constant, scalar) uniform PushConstants {
  VoxelGridParams grid;
  uint arg;  // allocate: input coord count; compact: output capacity.
} pc;

// Spatial hash: block coordinate -> bucket index (mirrors volume/hash.hpp).
uint computeHashPos(ivec3 block, int num_buckets) {
  uint h = (uint(block.x) * kHashPrimeX) ^ (uint(block.y) * kHashPrimeY) ^
           (uint(block.z) * kHashPrimeZ);
  return h % uint(num_buckets);
}
