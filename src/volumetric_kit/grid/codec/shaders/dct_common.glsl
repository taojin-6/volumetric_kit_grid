// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// Shared by dct_forward.comp and dct_inverse.comp: the push-constant block, the
// block list, the basis + zigzag table buffer, the live-block probe, the
// shared-memory work cube, and the one separable 8-point line transform both
// directions are made of.
//
// ONE WORKGROUP OF kLanes (64) INVOCATIONS PER BLOCK, each owning one kEdge-voxel
// line per pass. A separable 8^3 transform is three passes of 64 independent
// 8-point transforms (along x, then y, then z), so 64 is the natural width --
// and it stays under Vulkan's guaranteed 128 invocations per workgroup, which
// the prior engine's 512 threads per block (one per voxel) does not.
//
// The block's values live in `s_work`, two halves of kBlockVoxels floats that
// the passes ping-pong between (4 KB of shared memory, against a guaranteed
// 16 KB).

#ifndef VR_DCT_COMMON_GLSL
#define VR_DCT_COMMON_GLSL

#extension GL_EXT_scalar_block_layout : require

// The block geometry, mirroring kBlockSize / kVoxelsPerBlock / kMaskWordBits
// (codec_params.hpp) and kStrideY / kStrideZ / kBasisSize (dct_tables.hpp).
// Fixed rather than tunable: the lane-to-line mapping below is built on it, and
// the host refuses a grid with any other block size.
const uint kEdge = 8u;                   // voxels along one block edge
const uint kStrideX = 1u;                // voxel v = x + kStrideY y + kStrideZ z
const uint kStrideY = kEdge;
const uint kStrideZ = kEdge * kEdge;
const uint kBlockVoxels = kStrideZ * kEdge;
const uint kBasisSize = kEdge * kEdge;   // the kEdge x kEdge 1-D DCT matrix
const uint kLanes = kEdge * kEdge;       // lines per pass: one per invocation
const uint kMaskWordBits = 32u;
const uint kMaskWords = kBlockVoxels / kMaskWordBits;
const uint kLinesPerMaskWord = kMaskWordBits / kEdge;
const uint kLineBits = (1u << kEdge) - 1u;  // one line's bits of a mask word
const uint kHalf = kBlockVoxels;         // offset of s_work's second half

// The set-0 bindings, mirroring `Binding` in dct_transform.cpp -- the compute
// core is explicit, not reflected (2026-07-05).
#define VR_DCT_BINDING_BLOCKS 0
#define VR_DCT_BINDING_TSDF 1
#define VR_DCT_BINDING_WEIGHT 2
#define VR_DCT_BINDING_TABLES 3
#define VR_DCT_BINDING_COEFFICIENTS 4
#define VR_DCT_BINDING_MASKS 5
#define VR_DCT_BINDING_ENTRIES 6
#define VR_DCT_BINDING_REJECTED 7

// BlockIndex and the read-only hash lookup from the volume tier (one
// definition, not a mirror -- see hash_lookup.glsl), which brings hash_common
// without its push constants: this kernel has its own, below.
#define VR_HASH_ENTRIES_BINDING VR_DCT_BINDING_ENTRIES
#include "volumetric_kit/recon/volume/shaders/hash_lookup.glsl"

// Mirrors PushConstants in dct_transform.cpp; all 4-byte scalars, so scalar
// layout places each at its host offset (48 bytes).
layout(push_constant, scalar) uniform PushConstants {
  uint block_base;         // first list entry this dispatch covers
  uint num_blocks;         // entries in the whole list
  uint coefficient_count;  // K, in [1, kBlockVoxels]
  float trunc_dist;        // metres; the SDF is divided by it (forward) and
                           // multiplied back (inverse)
  float dc_step;           // quantization steps, fractions of trunc_dist
  float ac_step;
  float observed_weight;   // weight >= this marks a voxel observed
  float decoded_weight;    // what the inverse writes on an observed voxel
  int num_buckets;         // the hash table's shape, for the live-block probe
  int bucket_size;
  int max_chain;
  int max_quantized;       // the clamp on a quantized coefficient
}
pc;

layout(set = 0, binding = VR_DCT_BINDING_BLOCKS, scalar) readonly buffer
    Blocks {
  BlockIndex blocks[];
};

// Built on the host (dct_tables.hpp) and uploaded once, so every device
// transforms against the same bits rather than its own cos().
//   basis[k * kEdge + n]  orthonormal 1-D DCT-II: X[k] = sum_n basis[k][n] x[n]
//   zigzag[j]             voxel index of kept coefficient j
layout(set = 0, binding = VR_DCT_BINDING_TABLES, scalar) readonly buffer
    Tables {
  float basis[kBasisSize];
  uint zigzag[kBlockVoxels];
};

// Entries block_is_live rejected, summed over every batch for the host to
// report.
layout(set = 0, binding = VR_DCT_BINDING_REJECTED, scalar) buffer Rejected {
  uint rejected;
};

shared float s_work[2u * kBlockVoxels];
shared float s_basis[kBasisSize];
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

// Transform one kEdge-point line of the cube held in s_work from `src`,
// writing it to the same line of the cube at `dst`. The line is voxels base,
// base + stride, ..., base + (kEdge - 1) * stride. Forward multiplies by the basis,
// inverse by its transpose -- the basis is orthonormal, so that is its inverse.
void transform_line(uint src, uint dst, uint base, uint stride, bool inverse) {
  float line[kEdge];
  for (uint i = 0u; i < kEdge; ++i) {
    line[i] = s_work[src + base + i * stride];
  }
  for (uint o = 0u; o < kEdge; ++o) {
    float acc = 0.0;
    for (uint i = 0u; i < kEdge; ++i) {
      acc += (inverse ? s_basis[i * kEdge + o] : s_basis[o * kEdge + i]) *
             line[i];
    }
    s_work[dst + base + o * stride] = acc;
  }
}

// Lane r's line in each pass, by its first voxel: the x-line of r = y + kEdge z,
// the y-line of r = x + kEdge z, and the z-line of r = x + kEdge y. Every
// separable pass -- the fill's and the transform's -- runs on this mapping, so
// each pass covers the cube once.
uint x_line_base(uint lane) { return lane * kEdge; }
uint y_line_base(uint lane) {
  return lane % kEdge + (lane / kEdge) * kStrideZ;
}
uint z_line_base(uint lane) {
  return lane % kEdge + (lane / kEdge) * kStrideY;
}

// The three separable passes, s_work[0..] -> [kHalf..] -> [0..] -> [kHalf..].
// Every pass reads lines other lanes wrote, hence a barrier after each. The
// caller must have filled the first half of s_work and s_basis, and passed a
// barrier. The result is left in the second half, from kHalf.
void transform_cube(uint lane, bool inverse) {
  transform_line(0u, kHalf, x_line_base(lane), kStrideX, inverse);
  barrier();
  transform_line(kHalf, 0u, y_line_base(lane), kStrideY, inverse);
  barrier();
  transform_line(0u, kHalf, z_line_base(lane), kStrideZ, inverse);
  barrier();
}

// The step coefficient j (zigzag index) was quantized with.
float step_for(uint j) { return j == 0u ? pc.dc_step : pc.ac_step; }

#endif  // VR_DCT_COMMON_GLSL
