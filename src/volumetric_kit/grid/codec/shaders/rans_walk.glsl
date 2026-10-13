// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// An entry's frame symbols on the device, shared by rans_count.comp and
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

// By stream position (segment_range): a P-frame's M reference entries, then
// its new blocks; an intra frame's blocks.
layout(set = 0, binding = 0, scalar) readonly buffer Blocks {
  BlockIndex blocks[];  // the new blocks in frame order; only coord is read
};
layout(set = 0, binding = 1, scalar) readonly buffer Masks {
  uint masks[];  // kMaskWords an entry
};
layout(set = 0, binding = 2, scalar) readonly buffer Coefficients {
  uint coefficients[];  // two int16 a word, (K + 1) / 2 words an entry
};
layout(set = 0, binding = 3, std430) readonly buffer Modes {
  uint modes[];  // a P-frame's, one a reference entry
};

uint g_coefficients = 0u;  // the entry's first word
// What write_frame refuses, found on the way: the mode past the alphabet
// (0 for none), and an emitted -32768.
uint g_bad_mode = 0u;
bool g_out_of_range = false;

// Within +-32767 as the transform writes it, except a stale word in an entry
// the forward rejected, or a -32768 the caller wrote: clamped, so a count
// stays within the model's classes, and flagged, so the frame is refused.
int coefficient(uint j) {
  const uint word = coefficients[g_coefficients + j / 2u];
  const int v = bitfieldExtract(int(word), int(16u * (j & 1u)), 16);
  if (v < -32767) g_out_of_range = true;
  return max(v, -32767);
}

// Stream position p: reference entry p, or new block p - M, where every
// R-th starts a segment. A mode past the alphabet emits nothing.
void walk_entry(uint p) {
  for (uint w = 0u; w < kMaskWords; ++w) {
    g_mask[w] = masks[p * kMaskWords + w];
  }
  g_coefficients = p * ((pc.coefficient_count + 1u) / 2u);
  if (p < pc.reference_count) {
    const uint mode = modes[p];
    if (mode >= kModeSymbols) {
      g_bad_mode = mode;
      return;
    }
    emit_entry(mode, pc.coefficient_count);
    return;
  }
  const uint q = p - pc.reference_count;
  const bool first = q % pc.segment_size == 0u;
  const ivec3 cur = blocks[q].coord;
  ivec3 prev = cur;
  if (!first) prev = blocks[q - 1u].coord;
  emit_block(first, prev, cur, pc.coefficient_count);
}

#endif  // VR_RANS_WALK_GLSL
