// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// Every rANS kernel's push constants. Mirrors `RansPush` in
// rans_dispatch.hpp.

#ifndef VGRID_RANS_PUSH_GLSL
#define VGRID_RANS_PUSH_GLSL

layout(push_constant, std430) uniform Push {
  uint item_base;  // first entry or segment this dispatch covers
  // M + A: a P-frame's reference entries, then blocks; or for the count and
  // layout a bound whose words they bind (counted_entries)
  uint num_entries;
  uint coefficient_count;  // K
  uint segment_size;       // R
  uint segment_count;
  uint frame_type;       // header byte 6
  uint reference_count;  // M, 0 for an intra frame
  uint update_size;      // U
  // The decode's: the payload's first byte in its binding, which starts at
  // the record's offset aligned down to minStorageBufferOffsetAlignment.
  // The gather's: the first byte it writes, 4-aligned.
  uint payload_byte;
}
pc;

// The frame's entries, on the device: its M reference entries and @p blocks
// of the bound's blocks (num_entries less M), less @p surplus of those: the
// survivors a bounded P-frame's bound counts twice
// (ResidentBlocks::bounded), 0 otherwise. Each word is clamped, so every
// read stays inside the bound whatever it holds; the writer refuses a word
// that needed it.
uint counted_entries(uint surplus, uint blocks) {
  const uint found = min(blocks, pc.num_entries - pc.reference_count);
  return pc.reference_count + found - min(surplus, found);
}

#endif  // VGRID_RANS_PUSH_GLSL
