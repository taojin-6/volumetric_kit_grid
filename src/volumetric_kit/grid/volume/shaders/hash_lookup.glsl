// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// Read-only block lookup by coordinate, for kernels OUTSIDE the volume tier that
// need to resolve a neighbouring block's voxel-array base without the host
// building them a table.
//
// Separate from hash_common.glsl on purpose, and not includable alongside it:
// that header declares its own `pc` push-constant block and the VoxelGridParams
// struct, which collide with any kernel that has push constants of its own. So
// this takes the table shape as arguments instead of reading `pc`, and declares
// only what a lookup needs.
//
// Why this exists at all: the meshing dispatch is *quiescent* -- nothing is
// allocating while it runs -- so a probe here needs no bucket lock and no
// memoryBarrier ordering against a concurrent writer. That is the same property
// that let the host resolve the neighbourhood from a compacted snapshot, and it
// is what makes doing it on-device strictly better: same answer, no serial
// O(active_blocks * 8) host pass, no extra buffer to upload.
//
// The caller must #define VR_HASH_ENTRIES_BINDING to the set-0 binding it has
// reserved for the hash entries, before including this.

#ifndef VR_HASH_ENTRIES_BINDING
#error "define VR_HASH_ENTRIES_BINDING before including hash_lookup.glsl"
#endif

// Mirrors volume/hash_types.hpp HashEntry (ptr@0, ivec3 pos@4, offset@16, 20 B)
// under scalar block layout, exactly as hash_common.glsl does.
struct VrHashEntry {
  int ptr;
  ivec3 pos;
  int offset;
};

layout(set = 0, binding = VR_HASH_ENTRIES_BINDING, scalar) readonly buffer
    VrHashEntries {
  VrHashEntry vr_hash_entries[];
};

// Mirror the volume tier's constants. Named with the Vr prefix so a translation
// unit that somehow pulled in both headers fails to compile rather than
// silently binding to one set of values.
const int kVrFreeEntry = -1;
const int kVrNoOffset = 0;
const uint kVrHashPrimeX = 73856093u;
const uint kVrHashPrimeY = 19349669u;
const uint kVrHashPrimeZ = 83492791u;

/// Block coordinate -> bucket index. Byte-identical to computeHashPos in
/// hash_common.glsl and hash_bucket on the host; a divergence here would resolve
/// neighbours from the wrong bucket and silently drop surface at block seams.
uint vrHashBucket(ivec3 block, int num_buckets) {
  uint h = (uint(block.x) * kVrHashPrimeX) ^ (uint(block.y) * kVrHashPrimeY) ^
           (uint(block.z) * kVrHashPrimeZ);
  return h % uint(num_buckets);
}

/// @return The block's voxel-array base pointer, or -1 when @p coord is not
///         allocated. The traversal mirrors block_exists in
///         hash_allocate_common.glsl -- scan the primary bucket, then walk the
///         anchor's collision chain -- differing only in returning the pointer
///         and in taking no locks (see the quiescence note above).
int vrFindBlockPtr(ivec3 coord, int num_buckets, int bucket_size,
                   int max_chain) {
  uint bs = uint(bucket_size);
  uint total_entries = uint(num_buckets) * bs;
  uint bucket = vrHashBucket(coord, num_buckets);
  uint bucket_start = bucket * bs;

  for (uint i = 0u; i < bs; ++i) {
    uint slot = bucket_start + i;
    if (vr_hash_entries[slot].ptr != kVrFreeEntry &&
        vr_hash_entries[slot].pos == coord) {
      return vr_hash_entries[slot].ptr;
    }
  }
  // The last entry of each bucket anchors its chain; follow `offset` hops.
  uint idx_last = (bucket + 1u) * bs - 1u;
  uint idx = idx_last;
  for (int it = 0; it < max_chain; ++it) {
    if (vr_hash_entries[idx].ptr != kVrFreeEntry &&
        vr_hash_entries[idx].pos == coord) {
      return vr_hash_entries[idx].ptr;
    }
    int off = vr_hash_entries[idx].offset;
    if (off == kVrNoOffset) {
      break;
    }
    idx = (idx_last + uint(off)) % total_entries;
  }
  return -1;
}
