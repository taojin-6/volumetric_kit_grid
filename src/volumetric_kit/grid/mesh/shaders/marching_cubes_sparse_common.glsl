#ifndef VGRID_MARCHING_CUBES_SPARSE_COMMON_GLSL
#define VGRID_MARCHING_CUBES_SPARSE_COMMON_GLSL

// What both sparse marching-cubes kernels do identically: address a block's
// voxels, resolve its 2x2x2 +neighbourhood on-device, gather a cell's eight
// corners through it, and take the field's gradient at those corners.
//
// TWO kernels, because vertex sharing is a pipeline choice rather than a
// runtime branch. `marching_cubes_sparse.comp` gives every triangle three
// private vertices (what this tier always did); `marching_cubes_sparse_shared.comp`
// shares a vertex between the cells that meet on an edge. Sharing needs ~8 KiB
// of `shared` arrays, and a `shared` array is reserved at pipeline creation
// whatever a push constant later says -- so folding the two into one kernel
// made the DEFAULT path pay sharing's threadgroup budget (152 B -> 8560 B, which
// bounds residency to 3 workgroups on Apple's ~32 KiB and to 1 at Vulkan's
// guaranteed 16 KiB floor) plus its per-corner vertex atomic, for a feature it
// does not use. MarchingCubes::create builds exactly one of the two.
//
// Both figures are SPIR-V-verified and both move as the kernels grow, so treat
// the gap rather than the endpoints as the decision: 152 B is s_neighbour,
// s_gradient_block and the default kernel's three-word block reservation, and
// the default kernel keeps its per-cell triangle-count cache in a private
// register precisely so this side of the gap stays a rounding error.
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
// `BlockIndex` + `vgridFindBlockPtr`, which block_neighbourhood.glsl needs),
// voxel_color.glsl and marching_cubes_common.glsl (for `mcCornerShift` and the
// `g_corner_grad` registers).

// vgridLocalIndex / vgridLocalBase / s_neighbour / vgridResolveNeighbourhood /
// vgridCornerStorage: the block's +neighbourhood and the one cross-block corner
// addressing, from the volume tier, which the codec's surface passes walk too.
#include "volumetric_kit/grid/volume/shaders/block_neighbourhood.glsl"

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
    if (!vgridCornerStorage(bs, base + mcCornerShift(i), si) ||
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
      corner_color[i] = vgridVoxelColorToLinear(color_attr[si]);
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
// vgridUnpackSrgbToLinear is a transfer curve per corner, up to 24 of them per
// cell, for values a count cannot use.
//
// The sign test below is the one line this shares with mcGather by convention
// rather than by construction, and the two must agree exactly: the count taken
// here reserves the range mcGather's caller then writes into. Both read
// `tsdf[si] < pc.iso` over the same quiescent buffer through the same
// vgridCornerStorage, so they can only diverge if the two lines are edited apart --
// and the emitter bounds its writes to the reserved range, so even that drops
// triangles rather than writing over the next block's.
int mcCellSigns(int bs, ivec3 base) {
  int cube_index = 0;
  for (int i = 0; i < 8; ++i) {
    uint si;
    if (!vgridCornerStorage(bs, base + mcCornerShift(i), si) ||
        weight[si] < pc.weight_threshold) {
      return 0;
    }
    if (tsdf[si] < pc.iso) {
      cube_index |= (1 << i);
    }
  }
  return cube_index;
}

// ---------------------------------------------------------------------------
// The corner gradients a vertex normal mixes (mcEdgeVertex).
//
// A corner's gradient is the central difference over its six neighbours. A
// neighbour that is unallocated or unobserved (weight below
// pc.weight_threshold, as mcGather) is never used: that axis takes the
// one-sided difference instead, whose other side is always there -- it is a
// corner of the same cell, which mcGather read. Every cell holding a corner
// reads the same two neighbours for it under the same test, so every cell
// computes the same gradient there: a corner's gradient is a function of its
// voxel alone.
//
// The neighbours reach one voxel past the cell, local -1 .. bs + 1 per axis,
// so through the block's whole 3x3x3 neighbourhood rather than the 2x2x2
// +neighbourhood the corners need: s_gradient_block[ox + 3 oy + 9 oz] is the
// voxel-array base of the block at offset (ox, oy, oz) - 1, or -1 when it is
// unallocated.
shared int s_gradient_block[27];

// Fill s_gradient_block for the block at @p coord, whose voxel-array base is
// @p self_ptr: threads 0..26 each probe one block. The caller must
// `barrier()` before any cell reads it.
void mcResolveGradientBlocks(ivec3 coord, int self_ptr, uint tid) {
  if (tid >= 27u) {
    return;
  }
  ivec3 off = ivec3(int(tid % 3u), int((tid / 3u) % 3u), int(tid / 9u)) - 1;
  s_gradient_block[tid] =
      off == ivec3(0) ? self_ptr
                      : vgridFindBlockPtr(coord + off, pc.num_buckets,
                                          pc.bucket_size, pc.max_chain);
}

// The distance at block-local voxel @p c (-1 .. bs + 1 per axis), or false
// when its block is unallocated or it is unobserved. At block_size 1 alone,
// bs + 1 lies two blocks away, past s_gradient_block, and reads as missing.
//
// Branch-free: a voxel with no block reads index 0 and is discarded, so a
// cell's 24 samples can all be in flight at once. Branching on the block and
// the weight made the extract 5 to 11% slower on a decoded live room capture
// (RTX 4090, M5 Max).
bool mcGradientSample(ivec3 c, out float s) {
  int bs = pc.block_size;
  ivec3 o = ivec3(greaterThanEqual(c, ivec3(bs))) - ivec3(lessThan(c, ivec3(0)));
  ivec3 lc = c - o * bs;
  int ptr = s_gradient_block[(o.x + 1) + 3 * (o.y + 1) + 9 * (o.z + 1)];
  bool held = ptr >= 0 && all(lessThan(lc, ivec3(bs)));
  uint si = held ? uint(ptr) + uint(vgridLocalIndex(lc, bs)) : 0u;
  float w = weight[si];
  s = tsdf[si];
  return held && !(w < pc.weight_threshold);
}

// d distance / d x along unit step @p step at the corner at block-local @p p,
// whose distance is @p s0 and whose in-cell neighbour along that axis is
// @p s_in; the neighbour past the cell is at p + dir * step.
float mcGradientAxis(ivec3 p, ivec3 step, int dir, float s0, float s_in) {
  float s_out;
  if (mcGradientSample(p + dir * step, s_out)) {
    return dir > 0 ? 0.5 * (s_out - s_in) : 0.5 * (s_in - s_out);
  }
  return dir > 0 ? s0 - s_in : s_in - s0;
}

// The gradient at the corner at shift @p shift from the cell at block-local
// @p base, whose distance is @p s0 and whose in-cell neighbours along x, y
// and z are @p sx, @p sy and @p sz.
vec3 mcGradientAt(ivec3 base, ivec3 shift, float s0, float sx, float sy,
                  float sz) {
  ivec3 p = base + shift;
  ivec3 dir = 2 * shift - 1;  // toward the neighbour past the cell
  return vec3(mcGradientAxis(p, ivec3(1, 0, 0), dir.x, s0, sx),
              mcGradientAxis(p, ivec3(0, 1, 0), dir.y, s0, sy),
              mcGradientAxis(p, ivec3(0, 0, 1), dir.z, s0, sz));
}

// Fill the g_corner_grad registers for the cell at block-local @p base, whose
// gathered distances are @p sdf. The corners are written out in
// kCornerOffset's numbering, (0,0,0) (1,0,0) (1,1,0) (0,1,0) (0,0,1) (1,0,1)
// (1,1,1) (0,1,1), so no array is indexed at run time. @p owned_only fills
// corners 0, 1, 3 and 4 alone: the ends of the cell's +x/+y/+z edges, all the
// sharing kernel's owned-edge pass reads.
void mcCornerGradients(float sdf[8], ivec3 base, bool owned_only) {
  float s0 = sdf[0], s1 = sdf[1], s2 = sdf[2], s3 = sdf[3];
  float s4 = sdf[4], s5 = sdf[5], s6 = sdf[6], s7 = sdf[7];
  g_corner_grad0 = mcGradientAt(base, ivec3(0, 0, 0), s0, s1, s3, s4);
  g_corner_grad1 = mcGradientAt(base, ivec3(1, 0, 0), s1, s0, s2, s5);
  g_corner_grad3 = mcGradientAt(base, ivec3(0, 1, 0), s3, s2, s0, s7);
  g_corner_grad4 = mcGradientAt(base, ivec3(0, 0, 1), s4, s5, s7, s0);
  if (owned_only) {
    return;
  }
  g_corner_grad2 = mcGradientAt(base, ivec3(1, 1, 0), s2, s3, s1, s6);
  g_corner_grad5 = mcGradientAt(base, ivec3(1, 0, 1), s5, s4, s6, s1);
  g_corner_grad6 = mcGradientAt(base, ivec3(1, 1, 1), s6, s7, s5, s2);
  g_corner_grad7 = mcGradientAt(base, ivec3(0, 1, 1), s7, s6, s4, s3);
}

#endif  // VGRID_MARCHING_CUBES_SPARSE_COMMON_GLSL
