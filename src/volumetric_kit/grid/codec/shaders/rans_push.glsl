// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// Every rANS kernel's push constants. Mirrors `RansPush` in
// rans_dispatch.hpp.

#ifndef VR_RANS_PUSH_GLSL
#define VR_RANS_PUSH_GLSL

layout(push_constant, std430) uniform Push {
  uint item_base;          // first entry or segment this dispatch covers
  uint num_entries;        // M + A: a P-frame's reference entries, then blocks
  uint coefficient_count;  // K
  uint segment_size;       // R
  uint segment_count;
  uint frame_type;       // header byte 6
  uint reference_count;  // M, 0 for an intra frame
  uint update_size;      // U
}
pc;

#endif  // VR_RANS_PUSH_GLSL
