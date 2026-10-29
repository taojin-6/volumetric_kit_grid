#ifndef VR_MARCHING_CUBES_SPARSE_COMMON_GLSL
#define VR_MARCHING_CUBES_SPARSE_COMMON_GLSL

// What both sparse marching-cubes kernels do identically: address a block's
// voxels, resolve its 2x2x2 +neighbourhood on-device, and gather a cell's eight
// corners through it.
//
// TWO kernels, because vertex sharing is a pipeline choice rather than a
// runtime branch. `marching_cubes_sparse.comp` gives every triangle three
// private vertices (what this tier always did); `marching_cubes_sparse_shared.comp`
// shares a vertex between the cells that meet on an edge. Sharing needs ~8 KiB
// of `shared` arrays, and a `shared` array is reserved at pipeline creation
// whatever a push constant later says -- so folding the two into one kernel
// made the DEFAULT path pay sharing's threadgroup budget (44 B -> 8452 B, which
// bounds residency to 3 workgroups on Apple's ~32 KiB and to 1 at Vulkan's
// guaranteed 16 KiB floor) plus its per-corner vertex atomic, for a feature it
// does not use. MarchingCubes::create builds exactly one of the two.
//
// Both figures are SPIR-V-verified and both move as the kernels grow, so treat
// the gap rather than the endpoints as the decision: 44 B is s_neighbour plus
// the default kernel's three-word block reservation, and the default kernel
// keeps its per-cell triangle-count cache in a private register precisely so
// this side of the gap stays a rounding error.
//
// The cost of two kernels is two copies of the code that is genuinely common,
// which is what this header exists to prevent: the neighbour probe and the
// corner gather are the cross-block-correctness-critical parts, and a fix to
// either must not be appliable to only one path. Mirrors marching_cubes_common.glsl,
// which plays the same role for the per-cell numerics.
//
// The includer must, *before* the include, declare the `pc` push block (fields
// `block_size`, `voxels_per_block`, `iso`, `weight_threshold`, `has_color`,
// `num_buckets`, `bucket_size`, `max_chain`), the `active_blocks` / `tsdf` /
// `weight` / `color_attr` buffers, and have included hash_lookup.glsl (for
// `BlockIndex` + `vrFindBlockPtr`, which block_neighbourhood.glsl needs),
// voxel_color.glsl and marching_cubes_common.glsl (for `mcCornerShift`).

// vrLocalIndex / vrLocalBase / s_neighbour / vrResolveNeighbourhood /
// vrCornerStorage: the block's +neighbourhood and the one cross-block corner
// addressing, from the volume tier, which the codec's surface passes walk too.
#include "volumetric_kit/recon/volume/shaders/block_neighbourhood.glsl"

// Gather a cell's eight corners through the shared neighbourhood: sdf values,
// linear colours, and the cube index. False when a corner is unavailable --
// its block unallocated, or the voxel unintegrated (weight below threshold).
//
// On that failure `cube_index` comes back 0, NOT the partial mask the loop had
// accumulated over the corners it did reach. That is load-bearing rather than
// tidy: mcCellTriangleCount reads 0 as "no triangles", so a caller that ignores
// the bool still cannot count, reserve, or emit for a cell whose corners were
// never all read.
bool mcGather(int bs, ivec3 base, out float sdf[8], out vec3 corner_color[8],
              out int cube_index) {
  cube_index = 0;
  for (int i = 0; i < 8; ++i) {
    uint si;
    if (!vrCornerStorage(bs, base + mcCornerShift(i), si) ||
        weight[si] < pc.weight_threshold) {
      cube_index = 0;  // drop the partial mask -- see the note above
      return false;
    }
    sdf[i] = tsdf[si];
    if (pc.has_color != 0u) {
      // A depth-integrated but colour-unobserved corner reads opaque white
      // rather than dragging the interpolated vertex toward black, mirroring
      // the integrator's first-observation-assigns anti-darkening rule.
      // Decoded to LINEAR working values, because the edge interpolation is
      // an average (the 2026-08-02 decision).
      corner_color[i] = vrVoxelColorToLinear(color_attr[si]);
    }
    if (sdf[i] < pc.iso) {
      cube_index |= (1 << i);
    }
  }
  return true;
}

// The cube index ALONE -- what a caller that only needs a cell's TRIANGLE COUNT
// has to read. 0 for a cell whose corners are unavailable, exactly as mcGather
// returns on failure, so an invalid cell and an empty one need no separate
// encoding.
//
// Separate from mcGather rather than a flag on it because what it drops is the
// expensive part, and the sparse kernel's counting phase runs over 100% of
// cells: no `out float[8]` / `out vec3[8]` copy-out (GLSL passes arrays by
// value, so each is a real 128-byte copy at every call), and no colour decode --
// vrUnpackSrgbToLinear is a transfer curve per corner, up to 24 of them per
// cell, for values a count cannot use.
//
// The sign test below is the one line this shares with mcGather by convention
// rather than by construction, and the two must agree exactly: the count taken
// here reserves the range mcGather's caller then writes into. Both read
// `tsdf[si] < pc.iso` over the same quiescent buffer through the same
// vrCornerStorage, so they can only diverge if the two lines are edited apart --
// and the emitter bounds its writes to the reserved range, so even that drops
// triangles rather than writing over the next block's.
int mcCellSigns(int bs, ivec3 base) {
  int cube_index = 0;
  for (int i = 0; i < 8; ++i) {
    uint si;
    if (!vrCornerStorage(bs, base + mcCornerShift(i), si) ||
        weight[si] < pc.weight_threshold) {
      return 0;
    }
    if (tsdf[si] < pc.iso) {
      cube_index |= (1 << i);
    }
  }
  return cube_index;
}

#endif  // VR_MARCHING_CUBES_SPARSE_COMMON_GLSL
