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
// `Vertex` struct, the writeonly `vertices[]` buffer, and the `tri_count`
// counter -- all by those names. They are identical in both kernels; only the
// binding indices differ, which is why the declarations stay in each shader.

// Corner c's step (0/1 per axis) from the cell base.
ivec3 mcCornerShift(int corner) {
  return ivec3(corner_offset[corner * 3 + 0], corner_offset[corner * 3 + 1],
               corner_offset[corner * 3 + 2]);
}

// Clamped iso-crossing ratio along the edge sa -> sb. Guards the near-tangent
// case so the ratio stays finite (mirrors the prior engine's copysign(kEpsilon)
// guard); clamp keeps the interpolant on the segment.
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

// Emit the cell's triangles into `vertices[]` / `tri_count`. Corner c's world
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
// runs. Past `capacity` a triangle is dropped but still counted, so tri_count
// always ends as the field's true total -- that is the contract the host sizes
// its arena from when it fits the arena to the surface rather than to the
// 5-tri/cell worst case, and re-runs after an undersized guess.
void mcEmitCell(int cube_index, float sdf[8], vec3 corner_color[8], vec3 origin,
                ivec3 base_voxel, float voxel_size, vec3 normal, float iso,
                uint has_color, uint capacity) {
  for (int t = 0; tri_table[cube_index * 16 + t] != -1; t += 3) {
    // Claim the slot BEFORE interpolating: the claim decides whether this
    // triangle's three edge interpolations are worth doing at all, and on the
    // dispatch that discovers an undersized arena most of them are not. The
    // reservation depends on nothing the loop below computes, so hoisting it is
    // semantically identical.
    uint tri = atomicAdd(tri_count, 1u);
    if (tri >= capacity) {
      // Drop this triangle but keep counting: `tri_count` must end up the
      // field's TRUE total, because the host sizes the arena from it and
      // re-runs. A `return` here would abandon this cell's remaining triangles
      // uncounted, making the reported total a lower bound -- so the host's
      // refit would still be too small and the retry would overflow again.
      continue;
    }
    vec3 p[3];
    vec3 col[3];
    for (int k = 0; k < 3; ++k) {
      int edge = tri_table[cube_index * 16 + t + k];
      int a = edge_to_vert[edge * 2 + 0];
      int b = edge_to_vert[edge * 2 + 1];
      float ratio = mcEdgeRatio(sdf[a], sdf[b], iso);
      vec3 pa = origin + vec3(base_voxel + mcCornerShift(a)) * voxel_size;
      vec3 pb = origin + vec3(base_voxel + mcCornerShift(b)) * voxel_size;
      p[k] = mix(pa, pb, ratio);
      // Interpolate color at the same ratio; opaque white where no color input.
      // `corner_color` arrives LINEAR (each kernel decodes at the gather), so
      // this mix -- an average -- happens in linear working values and
      // Vertex::color leaves linear, which is what glTF COLOR_0 means and what
      // the renderer shades in. White is 1.0 in either space.
      col[k] = has_color != 0u ? mix(corner_color[a], corner_color[b], ratio)
                               : vec3(1.0);
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
