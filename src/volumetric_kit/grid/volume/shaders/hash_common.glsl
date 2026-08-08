// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// Shared definitions for the voxel-hash compute kernels: the device struct
// layouts (scalar block layout, byte-identical to the host POD structs in
// volume/hash_types.hpp and volume/voxel_grid.hpp), the hash-table constants,
// the push-constant block, and the spatial hash. #included by hash_init.comp,
// hash_allocate_coords.comp, hash_compact.comp, and hash_delete_coords.comp.
//
// Buffer bindings and the SSBO-touching helpers are NOT here: GLSL buffer blocks
// are global and cannot be passed to functions, so each kernel declares only the
// buffers it uses, and the lock/heap/allocate helpers live in the allocate
// kernel that needs them.
//
// A kernel OUTSIDE the volume tier reaches this through hash_lookup.glsl, which
// wants the struct layouts, the constants and computeHashPos but has push
// constants of its own -- so the `pc` block below is opt-out via
// VR_HASH_COMMON_NO_PUSH_CONSTANTS. Everything else is shared unconditionally,
// which is the point: a second copy of these constants would compile clean, pass
// spirv-val, and resolve neighbours from the wrong bucket the moment one of them
// changed. The include guard is what lets a translation unit reach this both
// directly and transitively.

#ifndef VR_HASH_COMMON_GLSL
#define VR_HASH_COMMON_GLSL

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

// --- Fail-count slots. A host/device ABI (VoxelHashMap::dispatch_with_retry
// reads them), so they live here beside the other shared constants rather than
// in one kernel's header.
//
// [0..3] mirror hash_ops.metal and are **retryable**: the host re-dispatches
// while the total keeps dropping, because the element that failed is still
// unprocessed and the next round can succeed. [4] is the one that is not, and
// the distinction is load-bearing: a delete whose heap append fails has already
// cleared the entry, so re-dispatching finds the coord absent and counts
// nothing -- the convergence heuristic would erase the report instead of
// resolving it, returning "0 failures" over a permanently leaked block index.
// The host therefore accumulates [4] across rounds and adds it to the final
// retryable count. Kernels that have no non-retryable failure never touch it.
const int kFailTotal = 0;     // retryable total; drives the retry loop
const int kFailLock = 1;      // lock contention
const int kFailChain = 2;     // collision chain full
const int kFailHeap = 3;      // block heap empty
const int kFailTerminal = 4;  // non-retryable; host accumulates across rounds
const int kFailTable = 5;     // no free entry anywhere in the table
const int kFailSlots = 6;

// --- Push constants: the grid shape + one kernel-specific argument. ---
//
// A GLSL translation unit may declare only ONE push-constant block, so a kernel
// with push constants of its own defines VR_HASH_COMMON_NO_PUSH_CONSTANTS before
// including this and passes the table shape as function arguments instead (see
// hash_lookup.glsl). Nothing above this point reads `pc` -- the coord transforms
// all take their VoxelGridParams as a parameter -- so opting out costs nothing.
#ifndef VR_HASH_COMMON_NO_PUSH_CONSTANTS
layout(push_constant, scalar) uniform PushConstants {
  VoxelGridParams grid;
  uint arg;  // allocate: input coord count; compact: output capacity.
} pc;
#endif

// Spatial hash: block coordinate -> bucket index (mirrors volume/hash.hpp).
uint computeHashPos(ivec3 block, int num_buckets) {
  uint h = (uint(block.x) * kHashPrimeX) ^ (uint(block.y) * kHashPrimeY) ^
           (uint(block.z) * kHashPrimeZ);
  return h % uint(num_buckets);
}

// --- World <-> voxel <-> block transforms (mirror volume/voxel_coords.hpp).
// Byte-identical arithmetic to the host so the block a sample owns agrees across
// CPU / GLSL / CUDA: roundEven is the host's round-half-to-even, and the
// negative bias reproduces its block indexing that floors toward -infinity. ---

// World position (metres) -> nearest integer voxel coordinate (ties to even).
ivec3 worldToVoxel(vec3 world, VoxelGridParams grid) {
  return ivec3(roundEven(world / grid.voxel_size));
}

// Voxel coordinate -> the block that contains it. Integer division truncates
// toward zero, so a negative axis is biased by block_size-1 first to floor
// toward -infinity (voxel -1 belongs to block -1, spanning voxels -8..-1).
ivec3 voxelToBlock(ivec3 voxel, VoxelGridParams grid) {
  int bs = grid.block_size;
  if (voxel.x < 0) voxel.x -= bs - 1;
  if (voxel.y < 0) voxel.y -= bs - 1;
  if (voxel.z < 0) voxel.z -= bs - 1;
  return voxel / bs;
}

// World position (metres) -> the block that contains it.
ivec3 worldToBlock(vec3 world, VoxelGridParams grid) {
  return voxelToBlock(worldToVoxel(world, grid), grid);
}

// Truncation band half-width in blocks (>= 1): trunc_dist as a whole number of
// blocks, so depth/point allocation can dilate a surface block into the band
// the TSDF integrates. Matches host truncation_blocks (same float op order).
int truncationBlocks(VoxelGridParams grid) {
  float block_extent = float(grid.block_size) * grid.voxel_size;
  int blocks = int(ceil(grid.trunc_dist / block_extent));
  return blocks > 1 ? blocks : 1;
}

#endif  // VR_HASH_COMMON_GLSL
