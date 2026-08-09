#ifndef VR_MARCHING_CUBES_COMMON_GLSL
#define VR_MARCHING_CUBES_COMMON_GLSL

// Shared marching-cubes per-cell body for the dense (marching_cubes.comp) and
// sparse (marching_cubes_sparse.comp) kernels. Only corner *sampling* differs
// between the two -- a dense linear index vs. a cross-block neighbour probe --
// so that stays in each kernel's main(); everything downstream of the gathered
// `sdf[8]` / `corner_color[8]` is identical and lives here as one source, so a
// fix to the numerically load-bearing math (the near-tangent edge guard, the
// 8-corner gradient, the reversed winding) can never be applied to only one
// path. Mirrors the volume tier's hash_common.glsl / the tsdf tier's
// tsdf_common.glsl shared-include discipline.
//
// The includer must, *before* the include, declare the binding-0 `Tables`
// block (`int tri_table[4096]`, `corner_offset[24]`, `edge_to_vert[24]`), the
// `Vertex` struct, the writeonly `vertices[]` buffer, and the `index_count`
// counter -- all by those names. They are identical in both kernels; only the
// binding indices differ, which is why the declarations stay in each shader.

// Indices one triangle contributes. The append atomic bumps `index_count` by
// exactly this, which is what makes the counter the draw command's indexCount
// rather than a triangle total a host would have to convert. Mirrors
// mesh::kIndicesPerTriangle on the host -- the two must not drift, since the
// host divides by it to recover the triangle count.
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

// Write one vertex at an already-claimed slot. The appearance fields are the
// same constants mcEmitCell writes, kept here so the two emitters cannot drift:
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
// `base_voxel`. The ONE interpolator: mcEmitCell calls it, and so do both
// halves of the sparse kernel's vertex sharing (the owned-edge pass and the
// +face duplicate path), so a shared vertex and a duplicated one land on
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
void mcEdgeVertex(int edge, float sdf[8], vec3 corner_color[8], vec3 origin,
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
  vec3 pa = origin + vec3(base_voxel + shift_a) * voxel_size;
  vec3 pb = origin + vec3(base_voxel + shift_b) * voxel_size;
  position = mix(pa, pb, ratio);
  // LINEAR working values (each kernel decodes at the gather), because this is
  // an average -- the 2026-08-02 colour decision. White is 1.0 either way.
  color = has_color != 0u ? mix(corner_color[a], corner_color[b], ratio)
                          : vec3(1.0);
}

// Emit the cell's triangles into `vertices[]` / `index_count`. Corner c's world
// position is `origin + (base_voxel + cornerShift(c)) * voxel_size`, so the
// caller passes the cell's integer base voxel and both kernels keep their exact
// original arithmetic (the dense kernel supplies its grid `origin`; the sparse
// kernel anchors on the base block's global voxel with a zero origin, so a
// boundary cell's far corners land on the neighbour's voxels without a second
// coordinate lookup). `sdf` / `corner_color` are the eight gathered corner
// values; `corner_color` is ignored when has_color == 0 (vertices get opaque
// white). `corner_color` is in LINEAR working values -- each kernel decodes the
// canonical 8-bit attribute at the gather -- because the edge interpolation
// below is an average. Triangles emit with reversed winding (0, 2, 1) so the CCW front face
// points along the outward gradient `normal` -- the orientation gfx expects;
// color follows the same reversal so each vertex keeps its own edge's color,
// and uv0 stays the "use vertex color" sentinel until projective texturing
// runs. Past `capacity` a triangle is dropped but still counted, so index_count
// always ends as kIndicesPerTriangle times the field's true TRIANGLE total --
// counted in indices, because it is the draw command's indexCount. That is the
// contract the host sizes its arena from when it fits the arena to the surface
// rather than to the 5-tri/cell worst case, and re-runs after an undersized
// guess; the host divides by kIndicesPerTriangle to recover the triangle count,
// and bounds the counter against uint32 in indices for the same reason.
void mcEmitCell(int cube_index, float sdf[8], vec3 corner_color[8], vec3 origin,
                ivec3 base_voxel, float voxel_size, vec3 normal, float iso,
                uint has_color, uint capacity) {
  for (int t = 0; tri_table[cube_index * 16 + t] != -1; t += 3) {
    // Claim the slot BEFORE interpolating: the claim decides whether this
    // triangle's three edge interpolations are worth doing at all, and on the
    // dispatch that discovers an undersized arena most of them are not. The
    // reservation depends on nothing the loop below computes, so hoisting it is
    // semantically identical.
    // Three at a time, so the counter *is* the draw command's indexCount and
    // the GPU never needs a host pass to turn a triangle count into one. The
    // quotient is this triangle's slot, exactly as before -- the units changed,
    // the meaning of `tri` did not.
    uint tri = atomicAdd(index_count, kIndicesPerTriangle) / kIndicesPerTriangle;
    if (tri >= capacity) {
      // Drop this triangle but keep counting: `index_count` must end up
      // kIndicesPerTriangle times the field's TRUE triangle total, because the
      // host sizes the arena from it and re-runs. A `return` here would abandon
      // this cell's remaining triangles uncounted, making the reported total a
      // lower bound -- so the host's refit would still be too small and the
      // retry would overflow again.
      continue;
    }
    vec3 p[3];
    vec3 col[3];
    for (int k = 0; k < 3; ++k) {
      // Through mcEdgeVertex, not open-coded here. It was a verbatim copy of
      // that function's body, and the copy is what let the two emitters drift:
      // the sparse kernel's sharing path calls mcEdgeVertex, so a vertex it
      // shares and one this loop duplicates have to agree BIT FOR BIT or the
      // "sharing moves only the vertex count" invariant is false. One call
      // site, one canonical endpoint order, one near-tangent guard.
      mcEdgeVertex(tri_table[cube_index * 16 + t + k], sdf, corner_color, origin,
                   base_voxel, voxel_size, iso, has_color, p[k], col[k]);
    }
    uint vbase = tri * 3u;
    vertices[vbase + 0u].position = p[0];
    vertices[vbase + 1u].position = p[2];
    vertices[vbase + 2u].position = p[1];
    vertices[vbase + 0u].normal = normal;
    vertices[vbase + 1u].normal = normal;
    vertices[vbase + 2u].normal = normal;
    vertices[vbase + 0u].color = vec4(col[0], 1.0);
    vertices[vbase + 1u].color = vec4(col[2], 1.0);
    vertices[vbase + 2u].color = vec4(col[1], 1.0);
    vertices[vbase + 0u].uv0 = vec2(-1.0);
    vertices[vbase + 1u].uv0 = vec2(-1.0);
    vertices[vbase + 2u].uv0 = vec2(-1.0);
    // The renderer's tangent slot. Marching cubes has no surface
    // parameterisation to derive a real tangent from, so write the same
    // placeholder the host converter used to synthesize -- the kernel fills it
    // because the buffer is handed to the renderer as-is, and a slot left
    // unwritten would carry whatever the previous, larger mesh put there.
    vertices[vbase + 0u].tangent = vec4(1.0, 0.0, 0.0, 1.0);
    vertices[vbase + 1u].tangent = vec4(1.0, 0.0, 0.0, 1.0);
    vertices[vbase + 2u].tangent = vec4(1.0, 0.0, 0.0, 1.0);
  }
}

#endif  // VR_MARCHING_CUBES_COMMON_GLSL
