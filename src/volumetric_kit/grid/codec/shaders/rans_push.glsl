// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// Every rANS kernel's push constants. Mirrors `Push` in
// device_frame_writer.cpp.

#ifndef VR_RANS_PUSH_GLSL
#define VR_RANS_PUSH_GLSL

layout(push_constant, std430) uniform Push {
  uint item_base;          // first block or segment this dispatch covers
  uint num_blocks;
  uint coefficient_count;  // K
  uint segment_size;       // R
  uint segment_count;
}
pc;

#endif  // VR_RANS_PUSH_GLSL
