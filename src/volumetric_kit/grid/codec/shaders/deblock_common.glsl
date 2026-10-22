// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// Shared by deblock_line.comp, deblock_gauss.comp and deblock_store.comp, the
// decoder-side deblock (deblock.hpp has the filters): the bindings and push
// constants of their one layout, the skip word, the normalized field, the
// mask bit, the shell index and the stamp.
//
// ONE WORKGROUP OF kLanes (64) INVOCATIONS PER FRAME BLOCK, the entry
// item_base + the group's id in the frame's list, found by its coord through
// the grid's hash; its ptr is never read.

#ifndef VR_DEBLOCK_COMMON_GLSL
#define VR_DEBLOCK_COMMON_GLSL

#extension GL_EXT_scalar_block_layout : require

// The block geometry, mirroring kBlockSize / kVoxelsPerBlock /
// kMaskWordsPerBlock (codec_params.hpp) and kShellVoxels (deblock.hpp).
const uint kEdge = 8u;
const uint kBlockVoxels = 512u;
const uint kLanes = 64u;
const uint kMaskWords = 16u;
const uint kShellVoxels = 296u;  // kBlockVoxels - 6^3

// The set-0 bindings, mirroring `Binding` in deblock.cpp.
#define VR_DEBLOCK_BINDING_BLOCKS 0
#define VR_DEBLOCK_BINDING_MASKS 1
#define VR_DEBLOCK_BINDING_ENTRIES 2
#define VR_DEBLOCK_BINDING_TSDF 3
#define VR_DEBLOCK_BINDING_WEIGHT 4
#define VR_DEBLOCK_BINDING_STAMPS 5
#define VR_DEBLOCK_BINDING_SHELL 6
#define VR_DEBLOCK_BINDING_SKIP 7

// BlockIndex, BlockStamp and the read-only hash lookup from the volume tier
// (one definition; see hash_lookup.glsl).
#define VR_HASH_ENTRIES_BINDING VR_DEBLOCK_BINDING_ENTRIES
#include "volumetric_kit/recon/volume/shaders/hash_lookup.glsl"

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

layout(set = 0, binding = VR_DEBLOCK_BINDING_BLOCKS, scalar) readonly buffer
    Blocks {
  BlockIndex blocks[];
};
// The frame's masks by list position: voxel l of entry e is bit l % 32 of
// word e * kMaskWords + l / 32, as the inverse read them.
layout(set = 0, binding = VR_DEBLOCK_BINDING_MASKS, scalar) readonly buffer
    Masks {
  uint masks[];
};
layout(set = 0, binding = VR_DEBLOCK_BINDING_TSDF, scalar) buffer Tsdf {
  float tsdf[];
};
layout(set = 0, binding = VR_DEBLOCK_BINDING_WEIGHT, scalar) readonly buffer
    Weight {
  float weight[];
};
layout(set = 0, binding = VR_DEBLOCK_BINDING_STAMPS, scalar) buffer Stamps {
  BlockStamp stamps[];
};
// The Gaussian's shell snapshot: kShellVoxels metres an entry, by list
// position, kUnobserved where the store leaves tsdf as it is.
layout(set = 0, binding = VR_DEBLOCK_BINDING_SHELL, scalar) buffer Shell {
  float shell[];
};
// The frame's skip word: non-zero, and every kernel returns before it reads
// or writes anything else. No deblock dispatch writes it, so each invocation
// reads the same value and the return is uniform.
layout(set = 0, binding = VR_DEBLOCK_BINDING_SKIP, scalar) readonly buffer
    Skip {
  uint skip;
};

// An unobserved voxel's value in the kernels' working field, and a shell
// entry the store skips. Observed values lie in [-1, 1] (metres within the
// band), so the test below is exact; finite, so fast math keeps it.
const float kUnobserved = 3.0e38;
bool vrObserved(float v) { return v < 1.0e38; }

// tsdf / trunc_dist correctly rounded, as the CPU's float division: the
// product with the correctly rounded reciprocal is within an ulp, and one
// fused residual step rounds it correctly (Markstein), whatever division
// instruction fast math would pick.
float vrNormalized(float t) {
  precise float q = t * pc.inv_trunc;
  precise float r = fma(-q, pc.trunc_dist, t);
  precise float v = fma(r, pc.inv_trunc, q);
  return v;
}

// Whether the frame observed voxel @p l of its entry @p entry.
bool vrMaskObserved(uint entry, uint l) {
  return ((masks[entry * kMaskWords + l / 32u] >> (l % 32u)) & 1u) != 0u;
}

// The shell index of local voxel @p p (x, y, z in 0..7): its rank among the
// block's kShellVoxels shell voxels in local-index order -- the z = 0 face,
// then each z = 1..6 slice's 28-voxel ring, then the z = 7 face -- or
// kShellVoxels for an interior voxel, which no neighbour's halo reaches.
uint vrShellIndex(uvec3 p) {
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
void vrStampChanged(int ptr) {
  atomicExchange(stamps[uint(ptr) / kBlockVoxels].changed, pc.tick);
}

#endif  // VR_DEBLOCK_COMMON_GLSL
