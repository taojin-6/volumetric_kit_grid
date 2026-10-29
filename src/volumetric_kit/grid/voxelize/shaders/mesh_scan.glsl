// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// Associative summary: entries, occupied slots, largest bin, overflow flag.
// Carry is sticky across every level, so a wrapped total is never consumed.
uvec4 meshCombine(uvec4 a, uvec4 b) {
  uint entries = a.x + b.x;
  uint occupied = a.y + b.y;
  return uvec4(entries, occupied, max(a.z, b.z),
               a.w | b.w | uint(entries < a.x) | uint(occupied < a.y));
}

shared uvec4 meshScanValues[256];
shared uvec4 meshScanTotal;

// Work-efficient exclusive scan. Every invocation, including padding, calls.
uvec4 meshScan(uvec4 value) {
  uint lane = gl_LocalInvocationID.x;
  meshScanValues[lane] = value;
  barrier();
  for (uint step = 1u; step < 256u; step <<= 1u) {
    uint right = (lane + 1u) * (step << 1u) - 1u;
    if (right < 256u)
      meshScanValues[right] = meshCombine(meshScanValues[right - step],
                                          meshScanValues[right]);
    barrier();
  }
  if (lane == 0u) {
    meshScanTotal = meshScanValues[255];
    meshScanValues[255] = uvec4(0u);
  }
  barrier();
  for (uint step = 128u; step > 0u; step >>= 1u) {
    uint right = (lane + 1u) * (step << 1u) - 1u;
    if (right < 256u) {
      uvec4 left = meshScanValues[right - step];
      meshScanValues[right - step] = meshScanValues[right];
      meshScanValues[right] = meshCombine(meshScanValues[right], left);
    }
    barrier();
  }
  return meshScanValues[lane];
}
