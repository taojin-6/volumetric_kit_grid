// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// A block's 2x2x2 +neighbourhood, resolved on the device, and the one
// cross-block addressing of a cell's corners through it: what the sparse
// mesher (marching_cubes_sparse_common.glsl), the codec's surface passes
// (dct_surface.comp) and its deblock's line passes (deblock_line.comp) walk,
// so each reaches the same voxels.
//
// The includer must have included hash_lookup.glsl (for `BlockIndex`,
// `vgridFindBlockPtr` and grid_types.glsl's in-block index, vgridLocalIndex and
// vgridLocalBase) first, and runs one workgroup per block. The hash table's
// shape arrives as arguments, since each includer has its own push constants.

#ifndef VGRID_BLOCK_NEIGHBOURHOOD_GLSL
#define VGRID_BLOCK_NEIGHBOURHOOD_GLSL

#ifndef VGRID_HASH_LOOKUP_GLSL
#error "include hash_lookup.glsl before block_neighbourhood.glsl"
#endif

// This block's 2x2x2 +neighbourhood: s_neighbour[ox + 2*oy + 4*oz] is the voxel-
// array base of the neighbour that owns corners spilling in that direction, or -1
// when it is unallocated. Resolved once per workgroup and read by every voxel,
// which is the whole point of one-workgroup-per-block: 8 probes serve
// voxels_per_block cells.
shared int s_neighbour[8];

// s_neighbour's index of octant @p o, each component 0 or 1.
int vgridOctantIndex(ivec3 o) {
  return o.x + 2 * o.y + 4 * o.z;
}

// Fill s_neighbour's octants whose bit (1 << octant) is set in @p octants for
// the block at @p coord, whose voxel-array base is @p self_ptr; the others
// read -1. Threads 0..7 each resolve one octant; octant 0 is this block,
// already known, and only thread 0's @p self_ptr is read. A caller that reads
// two octants probes two blocks, not seven. The caller must `barrier()`
// before any cell reads it.
void vgridResolveOctants(ivec3 coord, int self_ptr, uint tid, uint octants,
                      int num_buckets, int bucket_size, int max_chain) {
  if (tid < 8u) {
    uint oi = tid;
    if (((octants >> oi) & 1u) == 0u) {
      s_neighbour[oi] = -1;
    } else if (oi == 0u) {
      s_neighbour[0] = self_ptr;
    } else {
      ivec3 off = ivec3(int(oi & 1u), int((oi >> 1) & 1u), int((oi >> 2) & 1u));
      s_neighbour[oi] =
          vgridFindBlockPtr(coord + off, num_buckets, bucket_size, max_chain);
    }
  }
}

// The whole neighbourhood: vgridResolveOctants with every octant.
//
// TODO(mesh): this prologue, not an occupancy change, is what the on-device
// probe costs -- 8 lanes each walking a bucket while the rest wait at the
// barrier, serialised ahead of every cell in the block. The dispatch measured
// +25% (25.9 -> 32.2 ms) on the M5 iPad Pro when the host table went away, and
// the flat grid it replaced was NOT more densely packed: at block_size 8,
// voxels_per_block is 512 = 2*256, so `gid / 512` was already constant across
// each 256-thread group and every workgroup belonged to one block. Only the
// group count changed (2 per block -> 1). So the levers are the probe --
// spreading the 8 across subgroups, or one lane per bucket slot with a
// reduction -- not the cell or block count. Left unmeasured rather than guessed
// at; the repo's own lesson is to profile the phase before optimising it. The
// mesh kernels then resolve the vertex normals' 3x3x3 neighbourhood in a second
// round (27 lanes, so lanes 1-7 walk two buckets); a single round in place of
// both, skipping the 7 blocks no sample reaches, saved at most 2% of the
// extract on an M5 Max (2026-10-29), so the two stay. Note
// that an early exit on the first free slot is NOT one of the levers: see the
// primary-scan note on vgridFindBlockPtr.
void vgridResolveNeighbourhood(ivec3 coord, int self_ptr, uint tid,
                            int num_buckets, int bucket_size, int max_chain) {
  vgridResolveOctants(coord, self_ptr, tid, 0xFFu, num_buckets, bucket_size,
                   max_chain);
}

// Resolve one corner voxel to its slot in the per-voxel attribute arrays.
//
// A corner at local coordinate bs on an axis spills into the +neighbour block on
// that axis: octant selects which of the 2x2x2 neighbourhood holds it, and the
// residual indexes within it. False when that neighbour is unallocated (ptr < 0)
// -- a corner with no block behind it invalidates the whole cell, no
// extrapolation -- and `si` is then 0 rather than undefined.
//
// `c` may be the +1 layer (a component == bs) when a cell reaches for an edge
// whose owner lies outside the block, so corners reach bs + 1. That still lands
// inside the 2x2x2 neighbourhood -- octant is `c / bs`, which is 1 for anything
// in [bs, 2*bs), and bs + 1 < 2*bs for every legal block size.
//
// THE cross-block addressing, and one copy of it: the mesher's gathers and the
// codec's surface passes resolve a corner through here, so the sign-only count
// and the full gather cannot land on different voxels -- they must not, because
// the count reserves the range the gather writes into -- and the codec
// resolves corners exactly as the mesher does.
bool vgridCornerStorage(int bs, ivec3 c, out uint si) {
  si = 0u;
  ivec3 octant = c / bs;
  ivec3 lc = c - octant * bs;
  int nptr = s_neighbour[vgridOctantIndex(octant)];
  if (nptr < 0) {
    return false;
  }
  si = uint(nptr) + uint(vgridLocalIndex(lc, bs));
  return true;
}

#endif  // VGRID_BLOCK_NEIGHBOURHOOD_GLSL
