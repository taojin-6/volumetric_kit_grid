// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// A block's frame symbols on the device, shared by rans_count.comp and
// rans_ops.comp: frame_grammar.glsl's writer half over the forward output,
// in the order the host writer emits them. The includer defines the two
// sinks.

#ifndef VR_RANS_WALK_GLSL
#define VR_RANS_WALK_GLSL

#extension GL_EXT_scalar_block_layout : require

#define VR_HASH_COMMON_NO_PUSH_CONSTANTS
#include "volumetric_kit/recon/volume/shaders/hash_common.glsl"

#include "rans_push.glsl"

#define VR_FRAME_WRITER
#include "frame_grammar.glsl"

layout(set = 0, binding = 0, scalar) readonly buffer Blocks {
  BlockIndex blocks[];  // in frame order; only coord is read
};
layout(set = 0, binding = 1, scalar) readonly buffer Masks {
  uint masks[];  // kMaskWords a block
};
layout(set = 0, binding = 2, scalar) readonly buffer Coefficients {
  uint coefficients[];  // two int16 a word, (K + 1) / 2 words a block
};

uint g_coefficients = 0u;  // the block's first word

// Within the transform's +-32767, except in an entry the forward rejected and
// never wrote: clamped, so the count kernel, which runs before the rejection
// is refused, stays within the model's classes.
int coefficient(uint j) {
  const uint word = coefficients[g_coefficients + j / 2u];
  return max(bitfieldExtract(int(word), int(16u * (j & 1u)), 16), -32767);
}

void walk_block(uint i) {
  for (uint w = 0u; w < kMaskWords; ++w) {
    g_mask[w] = masks[i * kMaskWords + w];
  }
  g_coefficients = i * ((pc.coefficient_count + 1u) / 2u);
  const bool first = i % pc.segment_size == 0u;
  const ivec3 cur = blocks[i].coord;
  ivec3 prev = cur;
  if (!first) prev = blocks[i - 1u].coord;
  emit_block(first, prev, cur, pc.coefficient_count);
}

#endif  // VR_RANS_WALK_GLSL
