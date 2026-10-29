// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// Shared by deblock_flags.comp, deblock_line.comp, deblock_gauss.comp and
// deblock_store.comp, the decoder-side deblock (deblock.hpp has the filters):
// the bindings and push constants of their one layout, the per-block
// fallback's flags, the normalized field, the shell index and the stamp.
//
// ONE WORKGROUP OF kLanes (64) INVOCATIONS PER FRAME BLOCK, the entry
// item_base + the group's id in the frame's list, found by its coord through
// the grid's hash; its ptr is never read. deblock_flags.comp alone runs an
// invocation a block.

#ifndef VGRID_DEBLOCK_COMMON_GLSL
#define VGRID_DEBLOCK_COMMON_GLSL

#extension GL_EXT_scalar_block_layout : require

// The block geometry, mirroring kBlockSize / kVoxelsPerBlock /
// kMaskWordsPerBlock (codec_params.hpp) and kShellVoxels (deblock.hpp).
const uint kEdge = 8u;
const uint kBlockVoxels = 512u;
const uint kLanes = 64u;
const uint kMaskWords = 16u;
const uint kShellVoxels = 296u;  // kBlockVoxels - 6^3

// The set-0 bindings, mirroring `Binding` in deblock.cpp.
#define VGRID_DEBLOCK_BINDING_BLOCKS 0
#define VGRID_DEBLOCK_BINDING_MASKS 1
#define VGRID_DEBLOCK_BINDING_ENTRIES 2
#define VGRID_DEBLOCK_BINDING_TSDF 3
#define VGRID_DEBLOCK_BINDING_WEIGHT 4
#define VGRID_DEBLOCK_BINDING_STAMPS 5
#define VGRID_DEBLOCK_BINDING_SHELL 6
#define VGRID_DEBLOCK_BINDING_FULL 7
#define VGRID_DEBLOCK_BINDING_EXACT 8

// BlockIndex, BlockStamp and the read-only hash lookup from the volume tier
// (one definition; see hash_lookup.glsl).
#define VGRID_HASH_ENTRIES_BINDING VGRID_DEBLOCK_BINDING_ENTRIES
#include "volumetric_kit/grid/volume/shaders/hash_lookup.glsl"

// Mirrors PushConstants in deblock.cpp; all 4-byte scalars, so scalar layout
// places each at its host offset (68 bytes).
layout(push_constant, scalar) uniform PushConstants {
  uint item_base;         // first list entry this dispatch covers
  uint num_blocks;        // entries in the whole list
  uint axis;              // the line pass's: 0 x, 1 y, 2 z
  uint tick;              // the map's tick, which a written block is stamped
  int num_buckets;        // the hash table's shape, for the probes
  int bucket_size;
  int max_chain;
  float trunc_dist;       // metres; v = tsdf / trunc_dist
  float inv_trunc;        // float(1 / double(trunc_dist))
  float observed_weight;  // weight >= this marks a neighbour's voxel observed
  float saturated;        // 1 - 1e-5: a fitted |v| at or above skips the line
  float c0;               // the 3-sample least-squares line's value at the
  float c1;               // face: sum_i c_i v_i, c = (13, 4, -5) / 12
  float c2;
  float line_clip;        // the line offset's clip, in v
  float gauss_g1;         // the Gaussian's weight on a neighbour (1 on self)
  float gauss_clip;       // a Gaussian pass's clip on a change, in v
}
pc;

layout(set = 0, binding = VGRID_DEBLOCK_BINDING_BLOCKS, scalar) readonly buffer
    Blocks {
  BlockIndex blocks[];
};
// The frame's masks by list position: voxel l of entry e is bit l % 32 of
// word e * kMaskWords + l / 32, as the inverse read them.
layout(set = 0, binding = VGRID_DEBLOCK_BINDING_MASKS, scalar) readonly buffer
    Masks {
  uint masks[];
};
layout(set = 0, binding = VGRID_DEBLOCK_BINDING_TSDF, scalar) buffer Tsdf {
  float tsdf[];
};
layout(set = 0, binding = VGRID_DEBLOCK_BINDING_WEIGHT, scalar) readonly buffer
    Weight {
  float weight[];
};
layout(set = 0, binding = VGRID_DEBLOCK_BINDING_STAMPS, scalar) buffer Stamps {
  BlockStamp stamps[];
};
// The Gaussian's shell snapshot: kShellVoxels metres an entry, by list
// position, kUnobserved where the store leaves tsdf as it is.
layout(set = 0, binding = VGRID_DEBLOCK_BINDING_SHELL, scalar) buffer Shell {
  float shell[];
};
// The per-block fallback's flags, a word a slot of the grid's heap, zeroed
// before deblock_flags.comp sets a frame block's to 1 when its mask is all
// ones: fully observed, which the passes may write. A 0 is a masked block --
// one holding an unobserved voxel, which keeps its exact decode -- or a slot
// the frame does not list. Read by the dispatches after it alone.
layout(set = 0, binding = VGRID_DEBLOCK_BINDING_FULL, scalar) buffer Full {
  uint full[];
};
// The frame's masked blocks, counted by deblock_flags.comp into a word zeroed
// before it: the blocks the deblock leaves as decoded.
layout(set = 0, binding = VGRID_DEBLOCK_BINDING_EXACT, scalar) buffer
    ExactBlocks {
  uint exact_blocks;
};

// An unobserved voxel's value in the kernels' working field, and a shell
// entry the store skips. Observed values lie in [-1, 1] (metres within the
// band), so the test below is exact; finite, so fast math keeps it.
const float kUnobserved = 3.0e38;
bool vgridObserved(float v) { return v < 1.0e38; }

// tsdf / trunc_dist correctly rounded, as the CPU's float division: the
// product with the correctly rounded reciprocal is within an ulp, and one
// fused residual step rounds it correctly (Markstein), whatever division
// instruction fast math would pick.
float vgridNormalized(float t) {
  precise float q = t * pc.inv_trunc;
  precise float r = fma(-q, pc.trunc_dist, t);
  precise float v = fma(r, pc.inv_trunc, q);
  return v;
}

// Whether the frame observed every voxel of its entry @p entry: its
// kMaskWords mask words all ones. Each kernel asks it of its own entry first
// thing, so a masked block's workgroup returns before it probes the hash:
// every lane reads the same words, so the return is uniform.
bool vgridEntryFull(uint entry) {
  uint all = ~0u;
  for (uint w = 0u; w < kMaskWords; ++w) {
    all &= masks[entry * kMaskWords + w];
  }
  return all == ~0u;
}

// Whether the block at voxel base @p ptr is fully observed, so the passes may
// write it: a neighbour's flag. Every lane of a workgroup that asks of one
// block reads the same word, which no dispatch after deblock_flags.comp
// writes, so a return on it is uniform.
bool vgridFull(int ptr) {
  return full[uint(ptr) / kBlockVoxels] != 0u;
}

// The shell index of local voxel @p p (x, y, z in 0..7): its rank among the
// block's kShellVoxels shell voxels in local-index order -- the z = 0 face,
// then each z = 1..6 slice's 28-voxel ring, then the z = 7 face -- or
// kShellVoxels for an interior voxel, which no neighbour's halo reaches.
uint vgridShellIndex(uvec3 p) {
  if (p.z == 0u) return p.x + kEdge * p.y;
  if (p.z == kEdge - 1u) return 232u + p.x + kEdge * p.y;
  uint ring = 64u + 28u * (p.z - 1u);
  if (p.y == 0u) return ring + p.x;
  if (p.y == kEdge - 1u) return ring + 20u + p.x;
  if (p.x == 0u) return ring + 8u + 2u * (p.y - 1u);
  if (p.x == kEdge - 1u) return ring + 9u + 2u * (p.y - 1u);
  return kShellVoxels;
}

// Stamp the block at voxel base @p ptr changed at the pass's tick. Atomic,
// since a line pass's two workgroups may stamp one block.
void vgridStampChanged(int ptr) {
  atomicExchange(stamps[uint(ptr) / kBlockVoxels].changed, pc.tick);
}

#endif  // VGRID_DEBLOCK_COMMON_GLSL
