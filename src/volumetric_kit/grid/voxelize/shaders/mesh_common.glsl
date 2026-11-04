// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// What mesh_bin.comp, mesh_full_band.comp, mesh_tri_records.comp and the write
// (mesh_integrate_body.glsl) share, in one place: the full band's mask bits,
// the rounding margin's rate, and a point's size. The constants are macros, so
// a kernel that does not use one does not carry it.

// A slot's write mask (MeshSdfParams::full_band_m): VGRID_FULL_BLOCK for a
// block written whole, which the count pass sets; else the bit of each full
// neighbour, which mesh_full_band.comp sets and the write reads.
#define VGRID_FULL_BLOCK 0x80000000u
// The bit of the neighbour at offset d, each component in -1..1, given
// e = d + 1: e.x + 3 e.y + 9 e.z. Bit 13, d = 0, is never set.
#define VGRID_APRON_BIT(ex, ey, ez) uint((ex) + 3 * (ey) + 9 * (ez))

// 2^-14 per metre of coordinate magnitude and of band: the rounding margin the
// write's bounds and the full band's radius take (mesh_integrate_body.glsl has
// the count), about 1,000 times float's unit roundoff.
#define VGRID_MARGIN (1.0 / 16384.0)

// A point's largest |coordinate|, which the rounding of anything computed from
// it grows with.
float sizeOf(vec3 v) { return max(abs(v.x), max(abs(v.y), abs(v.z))); }
