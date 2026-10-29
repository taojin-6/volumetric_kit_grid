// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// One workgroup's exclusive scan of u32, a chunk of kScanLanes values a call:
// the codec's one scan, which rans_layout.comp runs over the entries' coder
// steps and the segments' slots, rans_scan.comp over the segments' lengths
// and survivor_ranks.comp over a P-frame's survivors. The includer declares
// local_size_x = kScanLanes, and every lane makes each call, with 0 past the
// end of its values: a frame has tens of thousands of entries, not millions,
// so one workgroup walks them a chunk at a time.

#ifndef VGRID_SCAN_GLSL
#define VGRID_SCAN_GLSL

const uint kScanLanes = 256u;

shared uint s_scan[kScanLanes];

// @return This lane's @p value's exclusive prefix within the chunk, plus
//         @p carry, which then advances by the chunk's total.
uint vgridScanChunk(uint value, inout uint carry) {
  const uint lane = gl_LocalInvocationID.x;
  s_scan[lane] = value;
  barrier();
  for (uint step = 1u; step < kScanLanes; step <<= 1u) {
    const uint left = lane >= step ? s_scan[lane - step] : 0u;
    barrier();
    s_scan[lane] += left;
    barrier();
  }
  const uint prefix = carry + s_scan[lane] - value;
  carry += s_scan[kScanLanes - 1u];
  barrier();
  return prefix;
}

#endif  // VGRID_SCAN_GLSL
