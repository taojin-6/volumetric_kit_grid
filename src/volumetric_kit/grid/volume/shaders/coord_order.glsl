// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// The order every sorted block list uses, for kernels: vgridCoordLess mirrors
// volume::coord_less (hash_types.hpp), by z, then y, then x, each a signed
// int32. VGRID_COORD_SEARCH(NAME, ARRAY) generates the binary searches over one
// array whose elements carry an ivec3 `coord` (BlockIndex, a buffer's or a
// shared array's), since GLSL passes no buffer or shared array to a function:
//
//   uint NAME_lower(ivec3 key, uint first, uint count)
//   uint NAME_upper(ivec3 key, uint first, uint count)
//
// How many of ARRAY[first, first + count) sort before @p key, and how many do
// not sort after it: std::lower_bound's and std::upper_bound's offsets on a
// sorted range. On any other range they still return within [0, count] after
// ceil(log2(count + 1)) probes, so a kernel over a corrupt list ends.

#ifndef VGRID_COORD_ORDER_GLSL
#define VGRID_COORD_ORDER_GLSL

bool vgridCoordLess(ivec3 a, ivec3 b) {
  if (a.z != b.z) return a.z < b.z;
  if (a.y != b.y) return a.y < b.y;
  return a.x < b.x;
}

#define VGRID_COORD_SEARCH(NAME, ARRAY)                                   \
  uint NAME##_lower(ivec3 key, uint first, uint count) {               \
    uint lo = 0u;                                                      \
    uint hi = count;                                                   \
    while (lo < hi) {                                                  \
      const uint mid = lo + (hi - lo) / 2u;                            \
      if (vgridCoordLess(ARRAY[first + mid].coord, key)) {                \
        lo = mid + 1u;                                                 \
      } else {                                                         \
        hi = mid;                                                      \
      }                                                                \
    }                                                                  \
    return lo;                                                         \
  }                                                                    \
  uint NAME##_upper(ivec3 key, uint first, uint count) {               \
    uint lo = 0u;                                                      \
    uint hi = count;                                                   \
    while (lo < hi) {                                                  \
      const uint mid = lo + (hi - lo) / 2u;                            \
      if (!vgridCoordLess(key, ARRAY[first + mid].coord)) {               \
        lo = mid + 1u;                                                 \
      } else {                                                         \
        hi = mid;                                                      \
      }                                                                \
    }                                                                  \
    return lo;                                                         \
  }

#endif  // VGRID_COORD_ORDER_GLSL
