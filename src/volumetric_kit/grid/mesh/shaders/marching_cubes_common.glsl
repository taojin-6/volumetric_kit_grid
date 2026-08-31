#ifndef VR_MARCHING_CUBES_COMMON_GLSL
#define VR_MARCHING_CUBES_COMMON_GLSL

// Shared marching-cubes per-cell body for the two sparse kernels:
// marching_cubes_sparse.comp (three private vertices per triangle) and
// marching_cubes_sparse_shared.comp (in-block sharing). What differs between
// them is how a vertex slot is CLAIMED, not how the vertex is computed, so
// that stays in each kernel's main(); everything downstream of the gathered
// `sdf[8]` / `corner_color[8]` is identical and lives here as one source, so a
// fix to the numerically load-bearing math (the near-tangent edge guard, the
// 8-corner gradient, the reversed winding) can never be applied to only one
// path. Mirrors the volume tier's hash_common.glsl / the tsdf tier's
// tsdf_common.glsl shared-include discipline.
//
// The includer must, *before* the include, declare the binding-0 `Tables`
// block (`int tri_table[4096]`, `corner_offset[24]`, `edge_to_vert[24]`), the
// `Vertex` struct, and the writeonly `vertices[]` buffer -- all by those
// names. They are identical in both kernels; only the binding indices differ,
// which is why the declarations stay in each shader.

// Indices one triangle contributes. Each kernel bumps `index_count` by exactly
// this per triangle, which is what makes the counter the draw command's
// indexCount rather than a triangle total a host would have to convert.
// Mirrors mesh::kIndicesPerTriangle on the host -- the two must not drift,
// since the host divides by it to recover the triangle count.
//
// And the count is of the FIELD, not of what fitted: past `capacity` a
// triangle is dropped but still counted, so `index_count` always ends as this
// times the true triangle total rather than a lower bound. That is what the
// host sizes its arena from when it fits the arena to the surface instead of
// the 5-tri/cell worst case, and what it refits against after an undersized
// guess -- a total that stopped at the capacity would make the refit undershoot
// and the retry overflow again. Both kernels take a whole block's count, and
// add it to `index_count`, before any slot in it is claimed, which is what
// makes that hold through a span the arena cannot hold.
const uint kIndicesPerTriangle = 3u;

// Corner c's step (0/1 per axis) from the cell base.
ivec3 mcCornerShift(int corner) {
  return ivec3(corner_offset[corner * 3 + 0], corner_offset[corner * 3 + 1],
               corner_offset[corner * 3 + 2]);
}

// Which cell owns edge `edge`, and along which axis it runs.
//
// Every marching-cubes edge is the +x/+y/+z edge of exactly one cell: the one
// at the componentwise MINIMUM of its two endpoint corners. That cell is `delta`
// steps from the cell asking, so a vertex-sharing kernel can name an edge by
// (owner cell, axis) and have every cell touching it agree -- which is the whole
// basis of sharing.
//
// DERIVED from the tables rather than tabulated beside them, deliberately. A
// fourth mirror of the edge/corner numbering is exactly the drift hazard the
// volume tier hit with hash_lookup.glsl: a divergence would silently resolve the
// wrong owner and split vertices that should share, under Status::ok. Here the
// answer is a pure function of `edge_to_vert` + `corner_offset`, so it cannot
// disagree with them.
void mcEdgeOwner(int edge, out ivec3 delta, out int axis) {
  ivec3 sa = mcCornerShift(edge_to_vert[edge * 2 + 0]);
  ivec3 sb = mcCornerShift(edge_to_vert[edge * 2 + 1]);
  delta = min(sa, sb);
  ivec3 d = abs(sa - sb);  // exactly one component is 1 -- edges are axis-aligned
  axis = d.x != 0 ? 0 : (d.y != 0 ? 1 : 2);
}

// Write one vertex at an already-claimed slot. The appearance fields are
// constants, kept in this one body so no emitter can drift from another --
// mcWriteTriangle reaches them for every private vertex, and the sharing kernel
// for every shared one:
// uv0 is the "use vertex colour" sentinel until projective texturing runs, and
// tangent is the placeholder the renderer's slot needs (meshing has no surface
// parameterisation to derive a real one from, and a slot left unwritten would
// carry whatever the previous, larger mesh put there).
void mcWriteVertex(uint slot, vec3 position, vec3 normal, vec3 color) {
  vertices[slot].position = position;
  vertices[slot].normal = normal;
  vertices[slot].tangent = vec4(1.0, 0.0, 0.0, 1.0);
  vertices[slot].uv0 = vec2(-1.0);
  vertices[slot].color = vec4(color, 1.0);
}

// Clamped iso-crossing ratio along the edge sa -> sb. Guards the near-tangent
// case so the ratio stays finite (mirrors the prior engine's copysign(kEpsilon)
// guard); clamp keeps the interpolant on the segment.
//
// NOT symmetric under swapping its endpoints, which is why mcEdgeVertex below
// orders them canonically before calling it. `1 - mcEdgeRatio(sb, sa, iso)`
// equals `mcEdgeRatio(sa, sb, iso)` only in exact arithmetic, and under the
// near-tangent guard not even approximately: at sa = -1e-9, sb = +1e-9, iso = 0
// the guard forces denom to +1e-6 one way and -1e-6 the other, and BOTH
// directions come out at ratio ~1e-3 -- i.e. hugging whichever endpoint was
// passed first, a full edge apart.
float mcEdgeRatio(float sa, float sb, float iso) {
  float denom = sb - sa;
  if (abs(denom) < 1e-6) {
    denom = denom < 0.0 ? -1e-6 : 1e-6;
  }
  return clamp((iso - sa) / denom, 0.0, 1.0);
}

// One normal per cell from the 8-corner central-difference gradient (points
// toward increasing distance, i.e. outward); shared by the cell's vertices. The
// 1/(4*voxel_size) scale is dropped -- it cancels under normalization. Falls
// back to +z when the gradient vanishes (a degenerate flat cell).
vec3 mcCellNormal(float sdf[8]) {
  vec3 grad =
      vec3((sdf[1] + sdf[2] + sdf[5] + sdf[6]) - (sdf[0] + sdf[3] + sdf[4] + sdf[7]),
           (sdf[2] + sdf[3] + sdf[6] + sdf[7]) - (sdf[0] + sdf[1] + sdf[4] + sdf[5]),
           (sdf[4] + sdf[5] + sdf[6] + sdf[7]) - (sdf[0] + sdf[1] + sdf[2] + sdf[3]));
  float glen = length(grad);
  return glen > 1e-6 ? grad / glen : vec3(0.0, 0.0, 1.0);
}

// The position + colour of the iso-crossing on edge `edge` of the cell whose
// eight corner samples are `sdf` / `corner_color` and whose base voxel is
// `base_voxel` (in GLOBAL voxel coordinates -- see mcWriteTriangle's
// preconditions). The ONE interpolator: mcWriteTriangle calls it, and so do
// both halves of the sharing kernel's vertex claim (the owned-edge pass and
// the +face duplicate path), so a shared vertex and a duplicated one land on
// bit-identical coordinates -- which is what makes sharing a pure vertex-count
// change and leaves every triangle exactly where it was.
//
// Bit-identical needs the endpoints in a CANONICAL ORDER, not merely resolved
// to the same global voxels, and that is what the swap below is for. kEdgeToVert
// lists edges 2/3/6/7 max-corner-first, so one cell's owned +y edge (edge 3 =
// corners {3,0}) is its neighbour's edge 1 (corners {1,2}) -- the same segment,
// traversed in opposite directions. `mix` and mcEdgeRatio are both
// direction-dependent (see mcEdgeRatio's note: under the near-tangent guard the
// two directions land at OPPOSITE ENDS of the edge), so without this the two
// emitters would disagree, by ulps normally and by a whole voxel on a tangent
// cell. Ordering by corner shift is exactly the rule mcEdgeOwner uses to name
// the edge, so the two cannot disagree about which end is which.
void mcEdgeVertex(int edge, float sdf[8], vec3 corner_color[8],
                  ivec3 base_voxel, float voxel_size, float iso, uint has_color,
                  out vec3 position, out vec3 color) {
  int a = edge_to_vert[edge * 2 + 0];
  int b = edge_to_vert[edge * 2 + 1];
  ivec3 shift_a = mcCornerShift(a);
  ivec3 shift_b = mcCornerShift(b);
  // Edges are axis-aligned, so the shifts differ in exactly one component (0 vs
  // 1): `any(greaterThan(...))` is true precisely when `a` is the high end.
  if (any(greaterThan(shift_a, shift_b))) {
    int swap_corner = a;
    a = b;
    b = swap_corner;
    ivec3 swap_shift = shift_a;
    shift_a = shift_b;
    shift_b = swap_shift;
  }
  float ratio = mcEdgeRatio(sdf[a], sdf[b], iso);
  vec3 pa = vec3(base_voxel + shift_a) * voxel_size;
  vec3 pb = vec3(base_voxel + shift_b) * voxel_size;
  position = mix(pa, pb, ratio);
  // LINEAR working values (each kernel decodes at the gather), because this is
  // an average -- the 2026-08-02 colour decision. White is 1.0 either way.
  color = has_color != 0u ? mix(corner_color[a], corner_color[b], ratio)
                          : vec3(1.0);
}

// How many triangles cell `cube_index` contributes.
//
// Separable from the emission because the count is a property of the cube index
// ALONE -- it needs no interpolation and no world position -- which is what lets
// the sparse kernel count a whole block, reserve one span for it, and only then
// write. `tri_table` rows are -1-terminated in groups of three.
//
// The all-inside / all-outside reject stays a comparison on two registers, and
// is not left to the table's own -1. Both are correct today, but only one of
// them stays correct independently of the SSBO the host uploads: `tri_table`
// is DATA, so a truncated or edited `McTables` sends cube index 0 or 255 into
// the walk below and hands back a count for a cell with no surface in it,
// while this comparison cannot be wrong about a cell with no sign change. It
// is the whole check now rather than one of two -- the dense kernel used to
// keep the same guard inline and be the oracle every equivalence test compared
// against, and neither exists any more. It is also the cheap half of the
// counting phase, which runs over 100% of cells while ~92% of them are exactly
// this case.
// The walk is bounded by the row as well as by its terminator, and the bound is
// not belt-and-braces. `tri_table` is an SSBO the host uploads, so the -1 that
// stops this loop is DATA; the 0..5 range is what the sparse kernel packs into
// an 8-bit cache field and, since it reserves per block, what sizes a whole
// block's arena span. A row that lost its terminator would not merely spin --
// it would walk into `corner_offset` (no robustBufferAccess anywhere in this
// tier) and hand out a span the block then overruns. Five is a property of
// marching cubes, not of the upload: 15 of a row's 16 entries, the sixteenth
// always -1.
const int kMaxTrianglesPerCell = 5;

int mcCellTriangleCount(int cube_index) {
  if (cube_index == 0 || cube_index == 255) {
    return 0;  // no sign change -> no surface in this cell
  }
  int n = 0;
  while (n < kMaxTrianglesPerCell && tri_table[cube_index * 16 + n * 3] != -1) {
    ++n;
  }
  return n;
}

// Write the cell's `t`-th triangle (counted in triangles, not table entries)
// into the already-claimed slot `tri`, as three private vertices at `tri * 3`.
//
// The DEFAULT sparse kernel's writer, and its only caller: the sharing kernel
// claims each vertex through its own counter and calls mcEdgeVertex directly,
// because a shared vertex is not at `tri * 3`. What the two do share is
// everything this reaches -- mcEdgeVertex and mcWriteVertex -- so a private
// vertex and a shared one land on bit-identical coordinates carrying identical
// appearance fields. That is why those are one function each and not two: a
// duplicated emitter is what let two paths drift once already, and here a
// drift would separate the two while every triangle count still matched.
//
// The caller reserves the run and passes the slot; this never claims one. That
// is what keeps a block's triangles CONTIGUOUS in the arena -- the property
// `block_spans` publishes and the incremental path rests on -- so a third
// emitter added later must reserve the same way and come through here, not
// append through `index_count` per triangle.
//
// THE PRECONDITIONS, stated here because this is the entry point a third
// caller would reach for and neither is checkable from inside:
//
//   * Corner c's world position is `(base_voxel + cornerShift(c)) *
//     voxel_size`, so the caller passes the cell's base voxel in GLOBAL voxel
//     coordinates -- the base block's own voxel plus the local cell, which is
//     what lets a boundary cell's far corners land on the neighbour's voxels
//     without a second coordinate lookup. A block-RELATIVE base voxel compiles
//     and produces a surface displaced by the block.
//   * `corner_color` is in LINEAR working values (each kernel decodes the
//     canonical 8-bit attribute at the gather, the 2026-08-02 colour decision),
//     because the edge interpolation below is an average. Passing encoded sRGB
//     compiles and darkens every interpolated vertex. It is ignored entirely
//     when has_color == 0, where vertices get opaque white.
//
// Neither shows up as a wrong triangle count, which is what the tests key on.
//
// Winding is reversed (0, 2, 1) so the CCW front face points along the outward
// gradient `normal` -- the orientation gfx expects; colour follows the same
// reversal so each vertex keeps its own edge's colour, and uv0 stays the "use
// vertex colour" sentinel until projective texturing runs.
void mcWriteTriangle(int cube_index, int t, float sdf[8], vec3 corner_color[8],
                     ivec3 base_voxel, float voxel_size, vec3 normal, float iso,
                     uint has_color, uint tri) {
  vec3 p[3];
  vec3 col[3];
  for (int k = 0; k < 3; ++k) {
    mcEdgeVertex(tri_table[cube_index * 16 + t * 3 + k], sdf, corner_color,
                 base_voxel, voxel_size, iso, has_color, p[k], col[k]);
  }
  uint vbase = tri * 3u;
  mcWriteVertex(vbase + 0u, p[0], normal, col[0]);
  mcWriteVertex(vbase + 1u, p[2], normal, col[2]);
  mcWriteVertex(vbase + 2u, p[1], normal, col[1]);
}

#endif  // VR_MARCHING_CUBES_COMMON_GLSL
