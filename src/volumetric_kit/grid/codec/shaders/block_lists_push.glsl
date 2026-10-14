// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// The push constants of the block-list kernels (block_lists.hpp). Mirrors
// `ListsPush` in block_lists.cpp.

#ifndef VR_BLOCK_LISTS_PUSH_GLSL
#define VR_BLOCK_LISTS_PUSH_GLSL

layout(push_constant, std430) uniform Push {
  uint item_base;        // first item this dispatch covers
  uint count;            // the call's items
  uint run_length;       // block_sort_merge: the sorted runs it merges
  uint reference_count;  // frame_merge: M, the reference's blocks
  uint new_count;        // frame_merge: A, the frame's new blocks
  uint block_count;      // frame_merge, grid_diff: N, the frame's blocks
  uint pass;             // grid_diff: 0 the table's slots, 1 the frame's
  int num_buckets;       // grid_diff: the hash table's shape
  int bucket_size;
  int max_chain;
}
pc;

#endif  // VR_BLOCK_LISTS_PUSH_GLSL
