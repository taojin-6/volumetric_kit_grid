// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// Every rANS kernel's push constants. Mirrors `RansPush` in
// rans_dispatch.hpp.

#ifndef VR_RANS_PUSH_GLSL
#define VR_RANS_PUSH_GLSL

layout(push_constant, std430) uniform Push {
  uint item_base;  // first entry or segment this dispatch covers
  // M + A: a P-frame's reference entries, then blocks; or for the count and
  // layout a bound whose surplus they bind (counted_entries)
  uint num_entries;
  uint coefficient_count;  // K
  uint segment_size;       // R
  uint segment_count;
  uint frame_type;       // header byte 6
  uint reference_count;  // M, 0 for an intra frame
  uint update_size;      // U
}
pc;

// The frame's entries, of num_entries less @p surplus: the positions past
// the frame's, which a bounded P-frame's count and layout take off on the
// device (ResidentBlocks::bounded), 0 otherwise. Clamped to the new blocks,
// so every read stays inside the bound whatever the word holds; the writer
// refuses a surplus that needed it.
uint counted_entries(uint surplus) {
  return pc.num_entries - min(surplus, pc.num_entries - pc.reference_count);
}

#endif  // VR_RANS_PUSH_GLSL
