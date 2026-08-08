// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// Read-only block lookup by coordinate, for kernels OUTSIDE the volume tier that
// need to resolve a neighbouring block's voxel-array base without the host
// building them a table.
//
// The struct layouts, the hash constants and computeHashPos come from
// hash_common.glsl -- ONE definition, not a mirror of one. An earlier cut
// restated them here under a `Vr` prefix, on the theory that the prefix would
// force a compile error if the two ever drifted; it does the opposite. GLSL
// rejects a duplicate `const int kFreeEntry` outright, so *identical* names are
// the compile-time guard and a prefix is precisely what lets two divergent sets
// coexist in silence -- and a divergence here does not fail, it resolves
// neighbours from the wrong bucket and drops surface at block seams while
// returning Status::ok. The stated blocker to sharing (hash_common.glsl's own
// `pc` block colliding with the includer's push constants) is opt-out now; the
// table shape still arrives as arguments, which is the only part that had to.
//
// The caller must #define VR_HASH_ENTRIES_BINDING to the set-0 binding it has
// reserved for the hash entries, before including this.

#ifndef VR_HASH_LOOKUP_GLSL
#define VR_HASH_LOOKUP_GLSL

#ifndef VR_HASH_ENTRIES_BINDING
#error "define VR_HASH_ENTRIES_BINDING before including hash_lookup.glsl"
#endif

// The includer has its own push constants, so take the table shape as function
// arguments rather than reading hash_common's `pc`. Everything else -- HashEntry,
// kFreeEntry / kNoOffset, computeHashPos, and the scalar-block-layout
// `#extension` this file's buffer block depends on -- is shared from there.
#define VR_HASH_COMMON_NO_PUSH_CONSTANTS
#include "volumetric_kit/recon/volume/shaders/hash_common.glsl"

layout(set = 0, binding = VR_HASH_ENTRIES_BINDING, scalar) readonly buffer
    VrHashEntries {
  HashEntry vr_hash_entries[];
};

/// @return The block's voxel-array base pointer, or -1 when @p coord is not
///         allocated.
///
/// The traversal mirrors block_exists in hash_allocate_common.glsl -- scan the
/// primary bucket, then walk the anchor's collision chain -- with two
/// differences: it returns the pointer, and it takes neither the bucket lock nor
/// the acquire memoryBarrierBuffer() that guards block_exists's ptr/pos pair.
///
/// Dropping BOTH is sound for the same single reason, and only for that reason:
/// the caller's dispatch is **quiescent** -- see VoxelHashMap::entries_buffer's
/// @warning and the precondition on MarchingCubes::extract_device. block_exists
/// needs its acquire because it runs *inside* an allocating dispatch, where a
/// concurrent invocation's just-published ptr could be observed beside that
/// slot's stale init-sentinel pos and false-match coord (0,0,0). With no
/// concurrent writer there is nothing for an acquire to pair with, so one here
/// would be a per-slot cost buying nothing -- and, worse, would read as though
/// this traversal were safe against a concurrent insert. It is not, and no
/// barrier could make it so: a walk can also meet a chain mid-splice, which is a
/// structural race, not an ordering one.
///
/// The primary scan reads all @p bucket_size slots and MUST NOT exit early on
/// the first free one. Insertion takes the first empty slot, but delete_coords
/// clears an arbitrary slot in place (hash_delete_coords.comp::delete_primary),
/// so a bucket may hold a free slot ahead of an occupied one. block_exists scans
/// in full for exactly this reason.
int vrFindBlockPtr(ivec3 coord, int num_buckets, int bucket_size,
                   int max_chain) {
  uint bs = uint(bucket_size);
  uint total_entries = uint(num_buckets) * bs;
  uint bucket = computeHashPos(coord, num_buckets);
  uint bucket_start = bucket * bs;

  for (uint i = 0u; i < bs; ++i) {
    uint slot = bucket_start + i;
    if (vr_hash_entries[slot].ptr != kFreeEntry &&
        vr_hash_entries[slot].pos == coord) {
      return vr_hash_entries[slot].ptr;
    }
  }
  // The last entry of each bucket anchors its chain; follow `offset` hops.
  uint idx_last = (bucket + 1u) * bs - 1u;
  uint idx = idx_last;
  for (int it = 0; it < max_chain; ++it) {
    if (vr_hash_entries[idx].ptr != kFreeEntry &&
        vr_hash_entries[idx].pos == coord) {
      return vr_hash_entries[idx].ptr;
    }
    int off = vr_hash_entries[idx].offset;
    if (off == kNoOffset) {
      break;
    }
    idx = (idx_last + uint(off)) % total_entries;
  }
  return -1;
}

#endif  // VR_HASH_LOOKUP_GLSL
