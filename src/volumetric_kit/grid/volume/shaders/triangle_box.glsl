// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// A triangle's candidate blocks: the blocks of its world bounding box grown by
// trunc_dist. The one device definition both halves of the (triangle,
// candidate block) decomposition read: voxelize's triangle_counts.comp counts
// each triangle's candidates with it, and vgridCandidateBlock
// (triangle_candidates.glsl) decodes a work item onto one of them with it, so
// the count and the decode agree bit for bit on any device. A work item past
// a triangle's count, or a count past its box, would name some other block.
//
// The includer must include hash_common.glsl first, for VoxelGridParams and
// worldToBlock.

#ifndef VGRID_TRIANGLE_BOX_GLSL
#define VGRID_TRIANGLE_BOX_GLSL

#ifndef VGRID_HASH_COMMON_GLSL
#error "include hash_common.glsl (or hash_lookup.glsl) before triangle_box.glsl"
#endif

// The first block of triangle (v0, v1, v2)'s candidate box, and its extent in
// blocks along each axis (each at least 1).
void vgridCandidateBox(vec3 v0, vec3 v1, vec3 v2, VoxelGridParams grid,
                       out ivec3 bmin, out uvec3 ext) {
  float trunc = grid.trunc_dist;
  vec3 lo_world = min(v0, min(v1, v2)) - vec3(trunc);
  vec3 hi_world = max(v0, max(v1, v2)) + vec3(trunc);
  bmin = worldToBlock(lo_world, grid);
  ivec3 bmax = worldToBlock(hi_world, grid);
  ext = uvec3(bmax - bmin + ivec3(1));
}

#endif  // VGRID_TRIANGLE_BOX_GLSL
