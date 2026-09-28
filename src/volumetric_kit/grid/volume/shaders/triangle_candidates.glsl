// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// The (triangle, candidate block) work decomposition, device side: which
// triangle a work item belongs to, which block it names, and whether the
// truncation band reaches that block. The host side is
// volume::triangle_candidate_offsets, which counts each triangle's candidates
// and uploads the exclusive prefix sum this file searches.
//
// Shared rather than restated by every triangle-driven kernel.
// hash_allocate_triangles.comp allocates the blocks that pass vrBlockInBand,
// and tsdf's mesh_bin.comp bins triangles into exactly the blocks that pass it.
// A second copy of the test would compile clean and let the integrator look
// for a block the allocator never made -- or skip one it did.
//
// The includer must include hash_common.glsl first (directly, or through
// hash_lookup.glsl), for VoxelGridParams and worldToBlock, and must #define
// VR_TRI_OFFSETS_BINDING to the set-0 binding it reserved for the offsets.

#ifndef VR_TRIANGLE_CANDIDATES_GLSL
#define VR_TRIANGLE_CANDIDATES_GLSL

#ifndef VR_HASH_COMMON_GLSL
#error "include hash_common.glsl (or hash_lookup.glsl) before triangle_candidates.glsl"
#endif
#ifndef VR_TRI_OFFSETS_BINDING
#error "define VR_TRI_OFFSETS_BINDING before including triangle_candidates.glsl"
#endif

#include "volumetric_kit/recon/core/shaders/triangle_common.glsl"

// Exclusive prefix sum of per-triangle candidate-block counts, tri_count + 1
// entries. vr_tri_offsets[tri_count] is the total work-item count, which is
// how a kernel learns its own bound without a second push constant.
layout(set = 0, binding = VR_TRI_OFFSETS_BINDING, scalar) readonly buffer
    VrTriOffsets {
  uint vr_tri_offsets[];
};

// The triangle owning work item `gid`: the largest t with offsets[t] <= gid.
//
// "Largest", not "any", is what makes a skipped triangle free: the host records
// a zero count for a degenerate or non-finite one, leaving offsets[t] ==
// offsets[t+1], and taking the largest match steps over every such run to the
// triangle that actually owns the item. offsets[tri_count] is the total and
// gid is below it, so `hi` is never itself an answer and the invariant
// offsets[lo] <= gid < offsets[hi] holds throughout.
uint vrTriangleOf(uint gid, uint tri_count) {
  uint lo = 0u;  // offsets[0] == 0 <= gid, so lo always names a valid answer
  uint hi = tri_count;
  while (hi - lo > 1u) {
    uint mid = lo + (hi - lo) / 2u;
    if (vr_tri_offsets[mid] <= gid) {
      lo = mid;
    } else {
      hi = mid;
    }
  }
  return lo;
}

// Decode work item `local` of triangle (v0, v1, v2) into the block it names.
//
// Recomputes the candidate box the host counted rather than reading one back:
// the host mirrors worldToBlock exactly (round_half_even <-> roundEven, the
// same negative bias -- the invariant hash_allocate_depth already rests on),
// and min/max/subtract are exact in IEEE, so the two agree bit-for-bit. The
// `local >= n` refusal is what makes a future divergence fail SAFE: it drops a
// candidate, rather than decoding `local` onto some unrelated coordinate.
bool vrCandidateBlock(uint local, vec3 v0, vec3 v1, vec3 v2,
                      VoxelGridParams grid, out ivec3 coord) {
  float trunc = grid.trunc_dist;
  vec3 lo_world = min(v0, min(v1, v2)) - vec3(trunc);
  vec3 hi_world = max(v0, max(v1, v2)) + vec3(trunc);
  ivec3 bmin = worldToBlock(lo_world, grid);
  ivec3 bmax = worldToBlock(hi_world, grid);
  uvec3 ext = uvec3(bmax - bmin + ivec3(1));

  uint plane = ext.x * ext.y;
  coord = bmin;
  if (local >= plane * ext.z) {
    return false;
  }
  uint dz = local / plane;
  uint rem = local - dz * plane;
  uint dy = rem / ext.x;
  uint dx = rem - dy * ext.x;
  coord = bmin + ivec3(int(dx), int(dy), int(dz));
  return true;
}

// Whether the truncation band of triangle (v0, v1, v2) reaches block `coord`:
// the block's centre lies within trunc_dist + its half-diagonal of the
// triangle. Conservative -- a block holding any voxel within trunc_dist is
// never rejected -- and measured from the surface, so a large slanted triangle
// keeps a sheet of blocks rather than the solid interior of its bounding box.
//
// Voxels are node-centred at voxel * voxel_size, so block b holds voxels
// [b*bs, b*bs + bs-1] and its VOXELS span (bs-1)*voxel_size -- one voxel short
// of the block's nominal extent. Measuring the half-diagonal off the nominal
// bs*voxel_size instead would still be conservative, just needlessly looser by
// a voxel in each axis.
bool vrBlockInBand(ivec3 coord, vec3 v0, vec3 v1, vec3 v2,
                   VoxelGridParams grid) {
  float span = float(grid.block_size - 1) * grid.voxel_size;
  vec3 centre = vec3(coord * grid.block_size) * grid.voxel_size + vec3(0.5 * span);
  float half_diag = 0.5 * sqrt(3.0) * span;
  vec3 closest;
  float dist = vrPointTriangleDistance(centre, v0, v1, v2, closest);
  return dist <= grid.trunc_dist + half_diag;
}

#endif  // VR_TRIANGLE_CANDIDATES_GLSL
