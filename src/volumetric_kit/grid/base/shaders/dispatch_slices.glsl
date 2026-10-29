// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// The indirect slices of a pass whose size a kernel finds on the device, split
// as grid::detail::dispatch_items splits one whose size the host knows:
// slice d covers groups [d * per_dispatch, (d + 1) * per_dispatch), and a
// slice past the pass's end gets none. The host records the slices for the
// most the pass can hold (grid::detail::IndirectSlices), each telling its
// kernel its first item.
//
// The one home of that split on the device: the count-to-slices kernel
// (dispatch_args.comp), the hash map's rounds (hash_rounds.comp) and the
// extractor's sizing (mc_args.comp) all write their slices through it.
//
// Define VGRID_SLICES_BINDING, the binding of the VkDispatchIndirectCommand
// array, before including this.

#ifndef VGRID_DISPATCH_SLICES_GLSL
#define VGRID_DISPATCH_SLICES_GLSL

#extension GL_EXT_scalar_block_layout : require

// VkDispatchIndirectCommand each, 12 bytes apart under scalar layout.
layout(set = 0, binding = VGRID_SLICES_BINDING, scalar) writeonly buffer Slices {
  uvec3 slices[];
};

// Groups for `items` items of `lanes` a group, rounded up without wrapping.
uint vgridGroups(uint items, uint lanes) {
  return items / lanes + (items % lanes != 0u ? 1u : 0u);
}

// Write slices [first, first + dispatches) of a pass of `groups` groups.
void vgridWriteSlices(uint first, uint dispatches, uint per_dispatch,
                   uint groups) {
  for (uint d = 0u; d < dispatches; ++d) {
    const uint begin = d * per_dispatch;
    slices[first + d] =
        uvec3(groups > begin ? min(groups - begin, per_dispatch) : 0u, 1u, 1u);
  }
}

#endif  // VGRID_DISPATCH_SLICES_GLSL
