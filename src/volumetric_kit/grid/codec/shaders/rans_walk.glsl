// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// A block's frame symbols on the device, shared by rans_count.comp and
// rans_encode.comp. The host writer (bitstream.cpp's emit_block) is the
// reference: these kernels must produce its frames byte for byte, which the
// device frame test checks.
//
// The walk runs in the order the encoder runs the coder -- each block's
// symbols last first, a raw field's chunks high first -- because rANS encodes
// in reverse. Counting is order-free, so both kernels share it. The includer
// defines the two sinks declared below.

#ifndef VR_RANS_WALK_GLSL
#define VR_RANS_WALK_GLSL

#extension GL_EXT_scalar_block_layout : require

#define VR_HASH_COMMON_NO_PUSH_CONSTANTS
#include "volumetric_kit/recon/volume/shaders/hash_common.glsl"

// Mirrors `Model` in bitstream.cpp.
const uint kDz = 0u;
const uint kDySame = 1u;
const uint kDyFree = 2u;
const uint kDxRun = 3u;
const uint kDxFree = 4u;
const uint kMaskClass = 5u;
const uint kPlane = 6u;
const uint kLine = 7u;
const uint kByte = 10u;
const uint kFirstCoef = 13u;

// Mask classes, and a plane's or line's symbol, as in bitstream.cpp.
const uint kMaskFull = 0u;
const uint kMaskEmpty = 1u;
const uint kMaskPartial = 2u;
const uint kSame = 0u;
const uint kAllEmpty = 1u;
const uint kAllFull = 2u;
const uint kOther = 3u;
const uint kMaskWords = 16u;

// Mirrors `Push` in device_frame_writer.cpp.
layout(push_constant, scalar) uniform Push {
  uint item_base;          // first block / segment / word this dispatch covers
  uint num_blocks;
  uint coefficient_count;  // K
  uint segment_size;       // R
  uint segment_count;
}
pc;

layout(set = 0, binding = 0, scalar) readonly buffer Blocks {
  BlockIndex blocks[];  // in frame order; only coord is read
};
layout(set = 0, binding = 1, scalar) readonly buffer Masks {
  uint masks[];  // kMaskWords a block
};
layout(set = 0, binding = 2, scalar) readonly buffer Coefficients {
  uint coefficients[];  // two int16 a word, (K + 1) / 2 words a block
};
// Each model's first entry in the per-symbol arrays (the counts, the
// tables): its alphabet's offset in TABLES order -- five coordinate models of
// 33 classes, the mask class's 3, the plane's and three lines' 4, three byte
// models' 256, then 16 classes per coefficient. Mirrors frame_model_alphabet
// in bitstream.cpp, which the device frame test holds it to.
uint model_base(uint model) {
  if (model <= kDxFree) return 33u * model;
  if (model == kMaskClass) return 165u;
  if (model < kByte) return 168u + 4u * (model - kPlane);
  if (model < kFirstCoef) return 184u + 256u * (model - kByte);
  return 952u + 16u * (model - kFirstCoef);
}

// What the includer does with each symbol and each raw field of 1-32 bits.
void sink_symbol(uint model, uint symbol);
void sink_bits(uint value, uint bits);

uint g_mask[kMaskWords];

uint bit_length(uint v) { return uint(findMSB(v) + 1); }

// emit_unsigned and emit_signed, reversed: the raw field, then the class.
void walk_unsigned(uint model, uint u) {
  const uint c = bit_length(u);
  if (c > 1u) sink_bits(u - (1u << (c - 1u)), c - 1u);
  sink_symbol(model, c);
}

void walk_signed(uint model, uint magnitude, bool negative) {
  const uint c = bit_length(magnitude);
  if (c > 0u) {
    const uint below = magnitude - (1u << (c - 1u));
    sink_bits((below << 1u) | (negative ? 1u : 0u), c);
  }
  sink_symbol(model, c);
}

// A signed step from a to b, which the sort keeps within 2^32 - 1.
void walk_step(uint model, int a, int b) {
  const bool negative = b < a;
  walk_signed(model, negative ? uint(a) - uint(b) : uint(b) - uint(a),
              negative);
}

uint mask_line(uint l) { return (g_mask[l / 4u] >> (8u * (l % 4u))) & 0xFFu; }

uint line_context(uint predictor) {
  return predictor == 0x00u ? 0u : predictor == 0xFFu ? 1u : 2u;
}

uint line_predictor(uint z, uint y) {
  if (y > 0u) return mask_line(8u * z + y - 1u);
  return z > 0u ? mask_line(8u * (z - 1u)) : 0x00u;
}

uint plane_symbol(uint z) {
  bool same = true;
  bool empty = true;
  bool full = true;
  for (uint y = 0u; y < 8u; ++y) {
    const uint line = mask_line(8u * z + y);
    same = same && line == (z > 0u ? mask_line(8u * (z - 1u) + y) : 0x00u);
    empty = empty && line == 0x00u;
    full = full && line == 0xFFu;
  }
  return same ? kSame : empty ? kAllEmpty : full ? kAllFull : kOther;
}

// emit_block, reversed: coefficients, then the mask, then the coordinate.
void walk_block(uint i, bool first) {
  // Two coefficients a word, the next word's load in flight while this
  // word's pair is walked.
  const uint k = pc.coefficient_count;
  const uint words = (k + 1u) / 2u;
  const uint base = i * words;
  uint next = coefficients[base + words - 1u];
  for (uint w = words; w-- > 0u;) {
    const uint word = next;
    if (w > 0u) next = coefficients[base + w - 1u];
    for (uint part = 2u; part-- > 0u;) {
      const uint j = 2u * w + part;
      if (j >= k) continue;  // an odd K's pad
      const int v = bitfieldExtract(int(word), int(16u * part), 16);
      walk_signed(kFirstCoef + j, uint(abs(v)), v < 0);
    }
  }

  bool full = true;
  bool none = true;
  for (uint w = 0u; w < kMaskWords; ++w) {
    g_mask[w] = masks[i * kMaskWords + w];
    full = full && g_mask[w] == ~0u;
    none = none && g_mask[w] == 0u;
  }
  const uint mask_class = full ? kMaskFull : none ? kMaskEmpty : kMaskPartial;
  if (mask_class == kMaskPartial) {
    for (uint z = 8u; z-- > 0u;) {
      const uint plane = plane_symbol(z);
      if (plane == kOther) {
        for (uint y = 8u; y-- > 0u;) {
          const uint line = mask_line(8u * z + y);
          const uint predictor = line_predictor(z, y);
          const uint context = line_context(predictor);
          const uint s = line == predictor ? kSame
                         : line == 0x00u   ? kAllEmpty
                         : line == 0xFFu   ? kAllFull
                                           : kOther;
          if (s == kOther) sink_symbol(kByte + context, line);
          sink_symbol(kLine + context, s);
        }
      }
      sink_symbol(kPlane, plane);
    }
  }
  sink_symbol(kMaskClass, mask_class);

  const ivec3 cur = blocks[i].coord;
  if (first) {
    // A segment's first block, in full, as x, y, z: reversed, z first.
    sink_bits(uint(cur.z), 32u);
    sink_bits(uint(cur.y), 32u);
    sink_bits(uint(cur.x), 32u);
    return;
  }
  const ivec3 prev = blocks[i - 1u].coord;
  const uint dz = uint(cur.z) - uint(prev.z);  // the sort keeps z rising
  if (dz == 0u) {
    const uint dy = uint(cur.y) - uint(prev.y);  // and y within a z slice
    if (dy == 0u) {
      walk_unsigned(kDxRun, uint(cur.x) - uint(prev.x) - 1u);
    } else {
      walk_step(kDxFree, prev.x, cur.x);
    }
    walk_unsigned(kDySame, dy);
  } else {
    walk_step(kDxFree, prev.x, cur.x);
    walk_step(kDyFree, prev.y, cur.y);
  }
  walk_unsigned(kDz, dz);
}

#endif  // VR_RANS_WALK_GLSL
