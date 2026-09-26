// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// Shared by dct_forward.comp and dct_inverse.comp: the push-constant block, the
// block list, the basis + zigzag table buffer, the live-block probe, the
// shared-memory work cube, and the one separable 8-point line transform both
// directions are made of.
//
// ONE WORKGROUP OF 64 INVOCATIONS PER BLOCK, each owning one 8-voxel line per
// pass. A separable 8^3 transform is three passes of 64 independent 8-point
// transforms (along x, then y, then z), so 64 is the natural width -- and it
// stays under Vulkan's guaranteed 128 invocations per workgroup, which the prior
// engine's 512 threads per block (one per voxel) does not.
//
// The block's 512 values live in `s_work`, two halves of 512 floats that the
// passes ping-pong between (4 KB of shared memory, against a guaranteed 16 KB).

#ifndef VR_DCT_COMMON_GLSL
#define VR_DCT_COMMON_GLSL

#extension GL_EXT_scalar_block_layout : require

// BlockIndex and the read-only hash lookup from the volume tier (one
// definition, not a mirror -- see hash_lookup.glsl), which brings hash_common
// without its push constants: this kernel has its own, below.
#define VR_HASH_ENTRIES_BINDING 6
#include "volumetric_kit/recon/volume/shaders/hash_lookup.glsl"

// Mirrors PushConstants in dct_transform.cpp; all 4-byte scalars, so scalar
// layout places each at its host offset (48 bytes).
layout(push_constant, scalar) uniform PushConstants {
  uint block_base;         // first list entry this dispatch covers
  uint num_blocks;         // entries in the whole list
  uint coefficient_count;  // K, in [1, 512]
  float trunc_dist;        // metres; the SDF is divided by it (forward) and
                           // multiplied back (inverse)
  float dc_step;           // quantization steps, fractions of trunc_dist
  float ac_step;
  float observed_weight;   // weight >= this marks a voxel observed
  float decoded_weight;    // what the inverse writes on an observed voxel
  int num_buckets;         // the hash table's shape, for the live-block probe
  int bucket_size;
  int max_chain;
  uint reserved;           // keeps the block at 48 bytes; unread
}
pc;

layout(set = 0, binding = 0, scalar) readonly buffer Blocks {
  BlockIndex blocks[];
};

// Built on the host (dct_tables.hpp) and uploaded once, so every device
// transforms against the same bits rather than its own cos().
//   basis[k * 8 + n]  orthonormal 1-D DCT-II: X[k] = sum_n basis[k][n] x[n]
//   zigzag[j]         voxel index (x + 8y + 64z) of kept coefficient j
layout(set = 0, binding = 3, scalar) readonly buffer Tables {
  float basis[64];
  uint zigzag[512];
};

// Entries block_is_live rejected, summed over every batch for the host to
// report.
layout(set = 0, binding = 7, scalar) buffer Rejected { uint rejected; };

const uint kLanes = 64u;
const uint kMaskWords = 16u;
const uint kHalf = 512u;  // offset of s_work's second half

shared float s_work[1024];
shared float s_basis[64];
shared uint s_live;

// Whether list entry `entry` names a live block of this grid: its coord must
// resolve through the hash table to exactly its ptr. That one probe subsumes a
// ptr range check -- every ptr the table holds is a heap slot -- and catches
// what a range check cannot: a free slot, or a coord paired with another
// block's ptr, either of which the inverse would write into. Lane 0 probes and
// the verdict is shared, so every lane returns on it or none does. Sound only
// while nothing allocates into the map, the same quiescence hash_lookup.glsl
// states.
bool block_is_live(uint entry, uint lane) {
  if (lane == 0u) {
    BlockIndex b = blocks[entry];
    int found =
        vrFindBlockPtr(b.coord, pc.num_buckets, pc.bucket_size, pc.max_chain);
    bool live = b.ptr >= 0 && found == b.ptr;
    s_live = live ? 1u : 0u;
    if (!live) {
      atomicAdd(rejected, 1u);
    }
  }
  barrier();
  return s_live != 0u;
}

// Transform one 8-point line of the cube held in s_work[src..src+511], writing
// it to the same line of s_work[dst..dst+511]. The line is voxels
// base, base + stride, ..., base + 7 * stride. Forward multiplies by the basis,
// inverse by its transpose -- the basis is orthonormal, so that is its inverse.
void transform_line(uint src, uint dst, uint base, uint stride, bool inverse) {
  float line[8];
  for (uint i = 0u; i < 8u; ++i) {
    line[i] = s_work[src + base + i * stride];
  }
  for (uint o = 0u; o < 8u; ++o) {
    float acc = 0.0;
    for (uint i = 0u; i < 8u; ++i) {
      acc += (inverse ? s_basis[i * 8u + o] : s_basis[o * 8u + i]) * line[i];
    }
    s_work[dst + base + o * stride] = acc;
  }
}

// The three separable passes, s_work[0..] -> [kHalf..] -> [0..] -> [kHalf..].
// Lane r owns, per pass: the x-line starting at 8r (r = y + 8z); the y-line at
// x + 64z (x = r % 8, z = r / 8); the z-line at x + 8y (x = r % 8, y = r / 8).
// Every pass reads lines other lanes wrote, hence a barrier after each. The
// caller must have filled s_work[0..511] and s_basis, and passed a barrier.
// The result is left in s_work[kHalf..kHalf+511].
void transform_cube(uint lane, bool inverse) {
  transform_line(0u, kHalf, lane * 8u, 1u, inverse);
  barrier();
  transform_line(kHalf, 0u, (lane & 7u) + (lane >> 3u) * 64u, 8u, inverse);
  barrier();
  transform_line(0u, kHalf, (lane & 7u) + (lane >> 3u) * 8u, 64u, inverse);
  barrier();
}

// The step coefficient j (zigzag index) was quantized with.
float step_for(uint j) { return j == 0u ? pc.dc_step : pc.ac_step; }

#endif  // VR_DCT_COMMON_GLSL
