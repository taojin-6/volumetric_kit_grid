// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// A block's symbols in the v3 frame: what a writer emits, and how a reader
// takes them back. The one text the host and the device code a frame with:
// the kernels include it as GLSL, and bitstream.cpp includes each half inside
// a struct, after defining `uint`, `ivec3`, `uvec2` and `findMSB`, so on the
// host the same lines compile as C++ member functions.
//
// Inside a segment each block is, in order: its coordinate (the segment's
// first in full, the rest as deltas from the block before), its mask class
// (and a partial mask, a plane and a line at a time), then its K
// coefficients. Every integer is a class -- its bit length -- through its
// model's table, then one raw field of the bits below its leading one, and
// below those its sign.
//
// The includer picks a half with VR_FRAME_WRITER or VR_FRAME_READER, and
// defines what it calls: a writer, sink_symbol, sink_bits (a raw field of 1
// to 32 bits) and coefficient(j); a reader, get, get_bits, coder_failed and
// store_coefficient. Both work on the block's mask in g_mask, and a reader
// leaves the block's coordinate in g_coord, which holds the block before's
// when it starts.
//
// Written in what GLSL and C++ mean alike: unsigned arithmetic wherever a
// value can wrap (a signed overflow is undefined in C++), one coder call a
// statement (C++ leaves argument order unspecified), no out parameters, and
// no built-in but findMSB. No include guard: bitstream.cpp includes it once
// for each half.

#ifndef __cplusplus
#include "frame_models.glsl"

uint g_mask[kMaskWords];
#ifdef VR_FRAME_WRITER
void sink_symbol(uint model, uint symbol);
void sink_bits(uint value, uint bits);
int coefficient(uint j);
#endif
#ifdef VR_FRAME_READER
ivec3 g_coord;
uint get(uint model);
uint get_bits(uint bits);
bool coder_failed();
void store_coefficient(uint j, int value);
#endif
#endif

uint bit_length(uint v) { return uint(findMSB(v) + 1); }

uint mask_line(uint l) { return (g_mask[l / 4u] >> (8u * (l % 4u))) & 0xFFu; }

// Which of the three line models a predictor selects.
uint line_context(uint predictor) {
  return predictor == 0x00u ? 0u : predictor == 0xFFu ? 1u : 2u;
}

// A line's predictor: the line before it in its plane, else the same line of
// the plane before, else (the first line of the block) unobserved.
uint line_predictor(uint z, uint y) {
  if (y > 0u) return mask_line(kLinesPerPlane * z + y - 1u);
  return z > 0u ? mask_line(kLinesPerPlane * (z - 1u)) : 0x00u;
}

#ifdef VR_FRAME_WRITER

void emit_unsigned(uint model, uint u) {
  const uint c = bit_length(u);
  sink_symbol(model, c);
  if (c > 1u) sink_bits(u - (1u << (c - 1u)), c - 1u);
}

// The sign is the low bit of one raw field of c bits, above it the c - 1 below
// the leading one, so a value of class c <= 12 costs one coder step rather
// than two. c is at most 32 (a step across all of int32), so the field fits.
void emit_signed(uint model, uint magnitude, bool negative) {
  const uint c = bit_length(magnitude);
  sink_symbol(model, c);
  if (c > 0u) {
    const uint below = magnitude - (1u << (c - 1u));
    sink_bits((below << 1u) | (negative ? 1u : 0u), c);
  }
}

// The signed step from a to b, which the sort keeps within 2^32 - 1.
void emit_step(uint model, int a, int b) {
  const bool negative = b < a;
  emit_signed(model, negative ? uint(a) - uint(b) : uint(b) - uint(a),
              negative);
}

// What plane z is against the plane before it (unobserved before the first),
// and so whether its lines follow.
uint plane_symbol(uint z) {
  bool same = true;
  bool empty = true;
  bool full = true;
  for (uint y = 0u; y < kLinesPerPlane; ++y) {
    const uint line = mask_line(kLinesPerPlane * z + y);
    same = same && line == (z > 0u ? mask_line(kLinesPerPlane * (z - 1u) + y)
                                   : 0x00u);
    empty = empty && line == 0x00u;
    full = full && line == 0xFFu;
  }
  return same ? kSame : empty ? kAllEmpty : full ? kAllFull : kOther;
}

void emit_partial_mask() {
  for (uint z = 0u; z < kPlanes; ++z) {
    const uint plane = plane_symbol(z);
    sink_symbol(kPlane, plane);
    if (plane != kOther) continue;
    for (uint y = 0u; y < kLinesPerPlane; ++y) {
      const uint line = mask_line(kLinesPerPlane * z + y);
      const uint predictor = line_predictor(z, y);
      const uint context = line_context(predictor);
      const uint s = line == predictor ? kSame
                     : line == 0x00u   ? kAllEmpty
                     : line == 0xFFu   ? kAllFull
                                       : kOther;
      sink_symbol(kLine + context, s);
      if (s == kOther) sink_symbol(kByte + context, line);
    }
  }
}

// Block cur, after prev unless it is its segment's first, of mask g_mask and
// K coefficients.
void emit_block(bool first, ivec3 prev, ivec3 cur, uint k) {
  if (first) {
    // A segment's first block, in full: the segment decodes on its own.
    sink_bits(uint(cur.x), 32u);
    sink_bits(uint(cur.y), 32u);
    sink_bits(uint(cur.z), 32u);
  } else {
    const uint dz = uint(cur.z) - uint(prev.z);  // the sort keeps z rising
    emit_unsigned(kDz, dz);
    if (dz == 0u) {
      const uint dy = uint(cur.y) - uint(prev.y);  // and y within a z slice
      emit_unsigned(kDySame, dy);
      if (dy == 0u) {
        // And x along a row: a step of at least 1.
        emit_unsigned(kDxRun, uint(cur.x) - uint(prev.x) - 1u);
      } else {
        emit_step(kDxFree, prev.x, cur.x);
      }
    } else {
      emit_step(kDyFree, prev.y, cur.y);
      emit_step(kDxFree, prev.x, cur.x);
    }
  }

  bool full = true;
  bool none = true;
  for (uint w = 0u; w < kMaskWords; ++w) {
    full = full && g_mask[w] == ~0u;
    none = none && g_mask[w] == 0u;
  }
  const uint mask_class = full ? kMaskFull : none ? kMaskEmpty : kMaskPartial;
  sink_symbol(kMaskClass, mask_class);
  if (mask_class == kMaskPartial) emit_partial_mask();

  for (uint j = 0u; j < k; ++j) {
    const int v = coefficient(j);
    emit_signed(kFirstCoef + j, v < 0 ? 0u - uint(v) : uint(v), v < 0);
  }
}

#endif  // VR_FRAME_WRITER

#ifdef VR_FRAME_READER

uint get_unsigned(uint model) {
  const uint c = get(model);
  if (c <= 1u) return c;
  return (1u << (c - 1u)) + get_bits(c - 1u);
}

// A signed value as its magnitude, at most 2^32 - 1, and 1 if negative.
uvec2 get_signed(uint model) {
  const uint c = get(model);
  if (c == 0u) return uvec2(0u, 0u);
  const uint field = get_bits(c);
  return uvec2((1u << (c - 1u)) + (field >> 1u), field & 1u);
}

// Whether a coordinate stepped by a signed delta stays within int32, and where
// it lands.
bool step_fits(int from, uvec2 delta) {
  return delta.x <= (delta.y != 0u ? uint(from) - 0x80000000u
                                   : 0x7FFFFFFFu - uint(from));
}
int stepped(int from, uvec2 delta) {
  return int(delta.y != 0u ? uint(from) - delta.x : uint(from) + delta.x);
}

void read_partial_mask() {
  for (uint w = 0u; w < kMaskWords; ++w) g_mask[w] = 0u;
  for (uint z = 0u; z < kPlanes; ++z) {
    const uint plane = get(kPlane);
    for (uint y = 0u; y < kLinesPerPlane; ++y) {
      uint line = 0u;
      if (plane == kOther) {
        const uint predictor = line_predictor(z, y);
        const uint context = line_context(predictor);
        const uint s = get(kLine + context);
        line = s == kSame       ? predictor
               : s == kAllEmpty ? 0x00u
               : s == kAllFull  ? 0xFFu
                                : get(kByte + context);
      } else if (plane == kSame) {
        line = z > 0u ? mask_line(kLinesPerPlane * (z - 1u) + y) : 0x00u;
      } else {
        line = plane == kAllFull ? 0xFFu : 0x00u;
      }
      const uint l = kLinesPerPlane * z + y;
      g_mask[l / 4u] |= line << (8u * (l % 4u));
    }
  }
}

// The block after g_coord, unless it is its segment's first, into g_coord,
// g_mask and its K coefficients. False for a coordinate outside int32; the
// coder's own failure carries everything else, including a failure partway
// through the deltas, whose zeros would otherwise decode as a run step and
// could be misreported as leaving int32.
bool read_block(bool first, uint k) {
  if (first) {
    g_coord.x = int(get_bits(32u));
    g_coord.y = int(get_bits(32u));
    g_coord.z = int(get_bits(32u));
  } else {
    const uvec2 dz = uvec2(get_unsigned(kDz), 0u);
    uvec2 dy = uvec2(0u, 0u);
    uvec2 dx = uvec2(0u, 0u);
    bool run_past = false;  // a run step of 2^32
    if (dz.x == 0u) {
      dy.x = get_unsigned(kDySame);
      if (dy.x == 0u) {
        const uint run = get_unsigned(kDxRun);
        run_past = run == 0xFFFFFFFFu;
        dx.x = run + 1u;
      } else {
        dx = get_signed(kDxFree);
      }
    } else {
      dy = get_signed(kDyFree);
      dx = get_signed(kDxFree);
    }
    if (coder_failed()) return true;  // the segment's end check reports it
    if (run_past || !step_fits(g_coord.x, dx) || !step_fits(g_coord.y, dy) ||
        !step_fits(g_coord.z, dz)) {
      return false;
    }
    g_coord = ivec3(stepped(g_coord.x, dx), stepped(g_coord.y, dy),
                    stepped(g_coord.z, dz));
  }

  // TODO(codec): refuse a partial mask that decodes all-full or all-empty,
  // and the mask code's other second spellings: a plane coded line by line
  // that one plane symbol names, a line coded as the byte another line
  // symbol names. The writer makes none of them and each decodes correctly,
  // so each is a second spelling of one frame rather than a wrong one (the
  // 2026-09-27 and 2026-10-01 entries).
  const uint mask_class = get(kMaskClass);
  if (mask_class == kMaskPartial) {
    read_partial_mask();
  } else {
    for (uint w = 0u; w < kMaskWords; ++w) {
      g_mask[w] = mask_class == kMaskFull ? ~0u : 0u;
    }
  }

  for (uint j = 0u; j < k; ++j) {
    // Class < 16, so the magnitude is at most 32767: every coefficient is in
    // range.
    const uvec2 v = get_signed(kFirstCoef + j);
    store_coefficient(j, v.y != 0u ? -int(v.x) : int(v.x));
  }
  return true;
}

#endif  // VR_FRAME_READER
