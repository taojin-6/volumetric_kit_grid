// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// The push constants of the block-list kernels (block_lists.hpp). Mirrors
// `ListsPush` in block_lists.cpp.

#ifndef VR_BLOCK_LISTS_PUSH_GLSL
#define VR_BLOCK_LISTS_PUSH_GLSL

layout(push_constant, std430) uniform Push {
  uint item_base;   // first item this dispatch covers
  uint count;       // the call's items
  uint run_length;  // block_sort_merge: the sorted runs it merges in pairs
}
pc;

#endif  // VR_BLOCK_LISTS_PUSH_GLSL
