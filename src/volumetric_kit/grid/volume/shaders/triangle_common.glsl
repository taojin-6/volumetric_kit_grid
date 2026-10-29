// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// Closest-point-on-triangle, the one geometric primitive the mesh-driven
// passes share: `volume`'s triangle_candidates.glsl prunes candidate blocks
// with it, and voxelize's mesh_integrate.comp evaluates the distance field
// itself with it. It lives in `volume`, the lowest tier either is in, and a
// second copy would compile clean, pass spirv-val, and disagree about a
// voxel's sign only on the creases.
//
// Ported from pointTriangleDistance in the prior engine's core/math/math_ops.hpp
// (Ericson, Real-Time Collision Detection): the barycentric region test, which
// resolves to a vertex, an edge, or the interior without ever forming the plane
// projection explicitly.

#ifndef VR_TRIANGLE_COMMON_GLSL
#define VR_TRIANGLE_COMMON_GLSL

// The closest point to `p` on triangle (a, b, c).
//
// Each of the three edge branches divides by a squared edge length -- d1-d3 is
// |ab|^2, d2-d6 is |ac|^2, and (d4-d3)+(d5-d6) is |bc|^2 -- so every denominator
// is non-zero for a triangle of non-zero area, and a degenerate one is the
// caller's to exclude (volume::triangle_candidate_offsets drops zero-area
// triangles before they reach a work item). The interior branch guards its own
// denominator anyway, since that one is a sum of signed barycentric areas and
// can cancel.
//
// That guard is against zero, not against a size: the sum is |ab x ac|^2, in
// metres^4, so any fixed threshold is a triangle size below which the interior
// collapses onto vertex `a` -- 1e-12 was every triangle with edges under about
// a millimetre, which a dense scan has everywhere. Where a sliver's rounding
// lets the sum through with barycentrics that do not describe a point of the
// triangle, they are clamped onto it, so the closest point is always one the
// triangle actually has.
vec3 vrClosestPointOnTriangle(vec3 p, vec3 a, vec3 b, vec3 c) {
  vec3 ab = b - a;
  vec3 ac = c - a;
  vec3 ap = p - a;

  float d1 = dot(ab, ap);
  float d2 = dot(ac, ap);
  if (d1 <= 0.0 && d2 <= 0.0) {
    return a;  // vertex region A
  }

  vec3 bp = p - b;
  float d3 = dot(ab, bp);
  float d4 = dot(ac, bp);
  if (d3 >= 0.0 && d4 <= d3) {
    return b;  // vertex region B
  }

  float vc = d1 * d4 - d3 * d2;
  if (vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0) {
    return a + ab * (d1 / (d1 - d3));  // edge region AB
  }

  vec3 cp = p - c;
  float d5 = dot(ab, cp);
  float d6 = dot(ac, cp);
  if (d6 >= 0.0 && d5 <= d6) {
    return c;  // vertex region C
  }

  float vb = d5 * d2 - d1 * d6;
  if (vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0) {
    return a + ac * (d2 / (d2 - d6));  // edge region AC
  }

  float va = d3 * d6 - d5 * d4;
  if (va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0) {
    return b + (c - b) * ((d4 - d3) / ((d4 - d3) + (d5 - d6)));  // edge BC
  }

  float sum = va + vb + vc;
  float denom = (sum > 0.0) ? (1.0 / sum) : 0.0;
  float v = clamp(vb * denom, 0.0, 1.0);
  float w = clamp(vc * denom, 0.0, 1.0 - v);
  return a + ab * v + ac * w;  // face interior
}

// Distance from `p` to triangle (a, b, c), with the closest point itself in
// `closest` -- the mesh-to-SDF sign test needs the point, the allocation prune
// needs only the distance, and computing one without the other saves nothing.
float vrPointTriangleDistance(vec3 p, vec3 a, vec3 b, vec3 c, out vec3 closest) {
  closest = vrClosestPointOnTriangle(p, a, b, c);
  return length(p - closest);
}

#endif  // VR_TRIANGLE_COMMON_GLSL
