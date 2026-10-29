// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// GPU test for mesh -> TSDF. Every voxel of every block the band reaches is
// checked against a brute-force HOST reference that shares nothing with the
// kernel but the contract:
//
//   - distance: an independent closest point (barycentric projection with an
//     edge clamp, not the shader's region test), minimised over EVERY triangle
//     of the mesh -- so a bin that dropped a triangle diverges too;
//   - inside/outside: the generalized winding number (the solid angles the mesh
//     subtends, summed), not a face normal -- so the sign is checked against a
//     different definition of "inside" rather than against itself.
//
// Signed mode signs by the closest face's normal, which is ambiguous where the
// faces tied for nearest disagree -- past an edge or corner sharper than 90
// degrees -- so there only the distance is checked; the regular tetrahedron's
// 70.5-degree edges put many voxels there. A cube with one corner pushed in has
// concave edges and a concave vertex, and writes the same bytes with its
// binning split across dispatches. An open quad's field runs on past its
// rim, since there is no topology to find one. A soup must write the same bytes
// as its indexed mesh, and the same mesh twice the same bytes. A triangle under
// a millimetre across checks the closest point has no size threshold, and a
// finely divided sheet splits the write across dispatches. Then every refusal,
// each checked to leave the grid untouched. Skips where no device is
// present.

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <map>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "device_blocks.hpp"
#include "test_check.hpp"
#include "test_meshes.hpp"
#include "volumetric_kit/core/vulkan/allocator.hpp"
#include "volumetric_kit/core/vulkan/device.hpp"
#include "volumetric_kit/core/vulkan/vulkan.hpp"
#include "volumetric_kit/recon/core/dispatch.hpp"
#include "volumetric_kit/recon/core/math/vector_types.hpp"
#include "volumetric_kit/recon/volume/hash_types.hpp"
#include "volumetric_kit/recon/volume/voxel_block_grid.hpp"
#include "volumetric_kit/recon/volume/voxel_coords.hpp"
#include "volumetric_kit/recon/volume/voxel_grid.hpp"
#include "volumetric_kit/recon/volume/voxel_hash_map.hpp"
#include "volumetric_kit/recon/voxelize/mesh_integrator.hpp"

#include "gpu_test.hpp"
#include "grid_readback.hpp"

namespace vr = volumetric_kit::recon;
namespace vkc = volumetric_kit::core;
namespace vol = volumetric_kit::recon::volume;
namespace vx = volumetric_kit::recon::voxelize;

namespace {

using test_meshes::closest_point;
using test_meshes::corner;
using test_meshes::dented_cube;
using test_meshes::Mesh;
using test_meshes::orient_outward;

using Coord = std::tuple<int, int, int>;

// Kernel and reference compute the same float distance two different ways; they
// agree to well under a micrometre at these scales.
constexpr float kTol = 2e-6f;
// Voxels this close to a decision boundary (the surface, or the band's edge)
// are not compared: which side a rounding error lands on is not the contract.
constexpr float kEdge = 1e-5f;

// ---- The independent reference --------------------------------------------

float nearest(vr::Vec3f p, const Mesh& m) {
  float best = 1e30f;
  for (std::size_t t = 0; t < m.triangle_count(); ++t) {
    const vr::Vec3f q =
        closest_point(p, corner(m, t, 0), corner(m, t, 1), corner(m, t, 2));
    best = std::fmin(best, vr::length(p - q));
  }
  return best;
}

// The generalized winding number: 1 inside a closed outward-wound mesh, 0
// outside (van Oosterom & Strackee's solid angle per triangle, in double).
double winding(vr::Vec3f p, const Mesh& m) {
  double total = 0.0;
  for (std::size_t t = 0; t < m.triangle_count(); ++t) {
    double A[3], B[3], C[3];
    for (int k = 0; k < 3; ++k) {
      A[k] = double(corner(m, t, 0)[k]) - p[k];
      B[k] = double(corner(m, t, 1)[k]) - p[k];
      C[k] = double(corner(m, t, 2)[k]) - p[k];
    }
    auto len = [](const double* x) {
      return std::sqrt(x[0] * x[0] + x[1] * x[1] + x[2] * x[2]);
    };
    auto dot3 = [](const double* x, const double* y) {
      return x[0] * y[0] + x[1] * y[1] + x[2] * y[2];
    };
    const double bc[3] = {B[1] * C[2] - B[2] * C[1], B[2] * C[0] - B[0] * C[2],
                          B[0] * C[1] - B[1] * C[0]};
    const double la = len(A), lb = len(B), lc = len(C);
    const double num = dot3(A, bc);
    const double den =
        la * lb * lc + dot3(A, B) * lc + dot3(A, C) * lb + dot3(B, C) * la;
    total += 2.0 * std::atan2(num, den);
  }
  return total / (4.0 * 3.14159265358979323846);
}

float clampf(float x, float lo, float hi) {
  return x < lo ? lo : (x > hi ? hi : x);
}

// ---- Fixtures
// ----------------------------------------------------------------

// A regular tetrahedron, edge 2*sqrt(2)*s, centred at c.
Mesh tetrahedron(vr::Vec3f c, float s) {
  Mesh m;
  m.v = {c + s * vr::Vec3f(1, 1, 1), c + s * vr::Vec3f(1, -1, -1),
         c + s * vr::Vec3f(-1, 1, -1), c + s * vr::Vec3f(-1, -1, 1)};
  m.i = {0, 1, 2, 0, 1, 3, 0, 2, 3, 1, 2, 3};
  orient_outward(m, c);
  return m;
}

// A +z-facing quad at height z0, divided into n x n cells of two triangles.
Mesh divided_quad(float x0, float x1, float y0, float y1, float z0, int n) {
  Mesh m;
  for (int j = 0; j <= n; ++j) {
    for (int i = 0; i <= n; ++i) {
      m.v.emplace_back(x0 + (x1 - x0) * float(i) / float(n),
                       y0 + (y1 - y0) * float(j) / float(n), z0);
    }
  }
  const auto row = static_cast<std::uint32_t>(n + 1);
  for (std::uint32_t j = 0; j < std::uint32_t(n); ++j) {
    for (std::uint32_t i = 0; i < std::uint32_t(n); ++i) {
      const std::uint32_t c = j * row + i;
      m.i.insert(m.i.end(), {c, c + 1, c + row + 1, c, c + row + 1, c + row});
    }
  }
  return m;
}

// The same triangles with no shared vertex -- how an STL stores a mesh.
Mesh soup(const Mesh& m) {
  Mesh s;
  for (std::size_t t = 0; t < m.triangle_count(); ++t) {
    for (int k = 0; k < 3; ++k) {
      s.v.push_back(corner(m, t, k));
      s.i.push_back(static_cast<std::uint32_t>(s.v.size() - 1));
    }
  }
  return s;
}

// ---- Grid plumbing ---------------------------------------------------------

std::vector<vol::BlockIndex> active_blocks(vol::VoxelBlockGrid& grid) {
  vkc::Result<std::vector<vol::BlockIndex>> a =
      grid.map().compact_active_blocks();
  return a.ok() ? std::move(a).value() : std::vector<vol::BlockIndex>{};
}

// A host copy of one attribute.
std::vector<float> attr(const vr_test::Gpu& ctx, vol::VoxelBlockGrid& grid,
                        const char* name) {
  return vr_test::read_attribute<float>(ctx.device, ctx.allocator, grid, name)
      .value();
}

// Every tsdf + weight value, so a refusal can be shown to have written none.
std::vector<float> snapshot(const vr_test::Gpu& ctx,
                            vol::VoxelBlockGrid& grid) {
  std::vector<float> out;
  for (const char* name : {"tsdf", "weight"}) {
    const std::vector<float> p = attr(ctx, grid, name);
    out.insert(out.end(), p.begin(), p.end());
  }
  return out;
}

// The active blocks' voxels of one attribute, block i's from i * vpb.
std::vector<float> attr(const vr_test::Gpu& ctx, vol::VoxelBlockGrid& grid,
                        const char* name,
                        const std::vector<vol::BlockIndex>& blocks) {
  return vr_test::read_blocks<float>(ctx, grid, name, blocks).value();
}

// Each written block's voxels, keyed by coordinate: the heap hands a block a
// different slot in each grid, so bytes are compared block by block.
std::map<Coord, std::vector<float>> by_coord(const vr_test::Gpu& ctx,
                                             vol::VoxelBlockGrid& grid) {
  std::map<Coord, std::vector<float>> out;
  const std::vector<vol::BlockIndex> blocks = active_blocks(grid);
  const std::vector<float> tsdf = attr(ctx, grid, "tsdf", blocks);
  const std::vector<float> weight = attr(ctx, grid, "weight", blocks);
  const auto vpb = static_cast<std::size_t>(grid.grid().voxels_per_block);
  for (std::size_t i = 0; i < blocks.size(); ++i) {
    const vol::BlockIndex& b = blocks[i];
    std::vector<float>& vals = out[{b.coord.x, b.coord.y, b.coord.z}];
    const auto first = static_cast<std::ptrdiff_t>(i * vpb);
    const auto last = static_cast<std::ptrdiff_t>((i + 1) * vpb);
    vals.assign(tsdf.begin() + first, tsdf.begin() + last);
    vals.insert(vals.end(), weight.begin() + first, weight.begin() + last);
  }
  return out;
}

struct Expect {
  bool skip = false;
  bool observed = false;
  float value = 0.0f;
  // Observed, but only |tsdf| is the contract: see signed_closed.
  bool magnitude_only = false;
};

// Compare every voxel of every allocated block with `expect(p)`.
template <class F>
int verify(const vr_test::Gpu& ctx, vol::VoxelBlockGrid& grid, const char* what,
           F expect, int* observed_count = nullptr) {
  const vol::VoxelGridParams& g = grid.grid();
  const std::vector<vol::BlockIndex> blocks = active_blocks(grid);
  const std::vector<float> tsdf = attr(ctx, grid, "tsdf", blocks);
  const std::vector<float> weight = attr(ctx, grid, "weight", blocks);
  int observed = 0;
  for (std::size_t i = 0; i < blocks.size(); ++i) {
    const vr::Vec3i base = vol::block_to_voxel(blocks[i].coord, g);
    for (int local = 0; local < g.voxels_per_block; ++local) {
      const vr::Vec3i lc(local % g.block_size,
                         (local / g.block_size) % g.block_size,
                         local / (g.block_size * g.block_size));
      const vr::Vec3f p = vol::voxel_to_world(base + lc, g);
      const Expect e = expect(p);
      if (e.skip) continue;
      const std::size_t at = i * static_cast<std::size_t>(g.voxels_per_block) +
                             static_cast<std::size_t>(local);
      const float w = weight[at];
      const float t = tsdf[at];
      const float err = e.magnitude_only
                            ? std::fabs(std::fabs(t) - std::fabs(e.value))
                            : std::fabs(t - e.value);
      const bool ok =
          e.observed ? (w == 1.0f && err <= kTol) : (w == 0.0f && t == 0.0f);
      if (!ok) {
        std::fprintf(stderr,
                     "FAIL %s: voxel (%.4f, %.4f, %.4f): got tsdf %.7f weight "
                     "%.1f, want %s %.7f\n",
                     what, p.x, p.y, p.z, t, w,
                     e.observed ? "observed" : "unobserved", e.value);
        return 1;
      }
      observed += e.observed ? 1 : 0;
    }
  }
  if (observed_count != nullptr) *observed_count = observed;
  return 0;
}

// Could the closest-face rule get this voxel's sign wrong? It can wherever a
// face tied for nearest -- every face meeting at the closest edge or vertex --
// puts the voxel on the wrong side of its own plane, since which of the tied
// faces decides is the lowest index, not the right side.
bool face_rule_can_fail(vr::Vec3f p, const Mesh& m, bool inside) {
  const float d = nearest(p, m);
  for (std::size_t t = 0; t < m.triangle_count(); ++t) {
    const vr::Vec3f a = corner(m, t, 0), b = corner(m, t, 1),
                    c = corner(m, t, 2);
    const vr::Vec3f q = closest_point(p, a, b, c);
    if (vr::length(p - q) > d + 1e-6f) continue;
    const bool face_says_inside =
        vr::dot(p - q, vr::cross(b - a, c - a)) < 0.0f;
    if (face_says_inside != inside) return true;
  }
  return false;
}

// The signed contract for a closed mesh: +-distance within the band, the sign
// from the winding number wherever the faces tied for nearest agree about it.
// Where they do not, the sign is the lowest-indexed face's -- the documented
// cost of signing without topology -- so only the distance is checked.
Expect signed_closed(vr::Vec3f p, const Mesh& m, float trunc) {
  Expect e;
  const float d = nearest(p, m);
  if (d < kEdge || std::fabs(d - trunc) < kEdge) {
    e.skip = true;
    return e;
  }
  e.observed = d <= trunc;
  if (e.observed) {
    const bool inside = winding(p, m) > 0.5;
    e.value = inside ? -d : d;
    e.magnitude_only = face_rule_can_fail(p, m, inside);
  }
  return e;
}

// The signed contract for +z-facing sheets at height z0: +-distance to the
// nearest sheet, with the height's sign. No topology means no rim, so the
// field runs on past a sheet's edge, and its sign there is still the height's:
// the closest point is on the rim, and the face normal is +z.
struct Rect {
  float x0, x1, y0, y1;
};
Expect signed_sheets(vr::Vec3f p, const std::vector<Rect>& sheets, float z0,
                     float trunc) {
  Expect e;
  const float h = p.z - z0;
  float d = 1e30f;
  for (const Rect& r : sheets) {
    const float dx = std::fmax(std::fmax(r.x0 - p.x, 0.0f), p.x - r.x1);
    const float dy = std::fmax(std::fmax(r.y0 - p.y, 0.0f), p.y - r.y1);
    d = std::fmin(d, std::sqrt(h * h + dx * dx + dy * dy));
  }
  if (std::fabs(h) < kEdge || std::fabs(d - trunc) < kEdge) {
    e.skip = true;
    return e;
  }
  e.observed = d <= trunc;
  e.value = e.observed ? (h >= 0.0f ? d : -d) : 0.0f;
  return e;
}

int gpu_main(vr_test::GpuContext& gpu) {
  // 10 mm voxels in 8-voxel blocks, a 40 mm (4-voxel) band: coarse enough that
  // the brute-force reference stays quick, with a band several voxels deep.
  // 1,024 blocks hold the largest band here, the metre sheet's 392.
  vol::VoxelGridParams gp{};
  gp.voxel_size = 0.01f;
  gp.block_size = 8;
  gp.voxels_per_block = 512;
  gp.trunc_dist = 0.04f;
  gp.bucket_size = 8;
  gp.num_buckets = 128;
  gp.num_blocks = 1024;
  gp.max_chain = 128;
  const float trunc = gp.trunc_dist;

  const vol::AttributeSpec attrs[] = {{"tsdf", sizeof(float)},
                                      {"weight", sizeof(float)}};
  vkc::Result<vol::VoxelBlockGrid> grid_r =
      vol::VoxelBlockGrid::create(gpu.device, gpu.allocator, gp, attrs, 2);
  CHECK(grid_r.ok());
  vol::VoxelBlockGrid grid = std::move(grid_r).value();

  vkc::Result<vx::MeshIntegrator> integ_r =
      vx::MeshIntegrator::create(gpu.device, gpu.allocator);
  CHECK(integ_r.ok());
  vx::MeshIntegrator integ = std::move(integ_r).value();
  CHECK(integ.valid());

  // Allocate `m`'s band into a cleared grid, then write it.
  auto convert = [&](const Mesh& m, const vx::MeshSdfParams& params)
      -> vkc::Result<vx::MeshIntegrateStats> {
    VKC_TRY(grid.clear());
    VKC_ASSIGN(std::uint32_t failed, grid.map().allocate_from_triangles(
                                         m.v.data(), m.vertex_count(),
                                         m.i.data(), m.triangle_count()));
    if (failed != 0) {
      return vkc::Status::out_of_memory("allocate_from_triangles failed");
    }
    return integ.integrate(grid, m.v.data(), m.vertex_count(), m.i.data(),
                           m.triangle_count(), params);
  };
  const vx::MeshSdfParams kSigned{vx::MeshSdfMode::Signed, 1.5f};
  const vx::MeshSdfParams kShell{vx::MeshSdfMode::Shell, 1.5f};
  const float shell = 1.5f * gp.voxel_size;

  // ---- 1. Signed tetrahedron: the sharp-edge case -------------------------
  const Mesh tet = tetrahedron(vr::Vec3f(0.203f, 0.117f, -0.089f), 0.1f);
  vkc::Result<vx::MeshIntegrateStats> tet_stats = convert(tet, kSigned);
  if (!tet_stats.ok()) {
    std::fprintf(stderr, "tetrahedron: %s\n",
                 tet_stats.status().message().c_str());
    return 1;
  }
  CHECK(tet_stats.value().triangles == 4);
  // Every block the allocation made is written.
  CHECK(tet_stats.value().blocks == active_blocks(grid).size());
  CHECK(tet_stats.value().bin_entries >= tet_stats.value().blocks);

  int tet_observed = 0;
  int tet_sign_checked = 0;
  if (verify(
          gpu, grid, "signed tetrahedron",
          [&](vr::Vec3f p) {
            const Expect e = signed_closed(p, tet, trunc);
            if (!e.skip && e.observed && !e.magnitude_only) ++tet_sign_checked;
            return e;
          },
          &tet_observed) != 0) {
    return 1;
  }
  std::printf("tetrahedron: %d observed voxels, %d with an unambiguous sign\n",
              tet_observed, tet_sign_checked);
  CHECK(tet_sign_checked > 1000);
  CHECK(tet_sign_checked < tet_observed);  // the sharp edges are exercised

  // ---- 2. Same mesh, same bytes ------------------------------------------
  // And every block written stamped changed, at a tick of the call's own.
  const std::map<Coord, std::vector<float>> tet_bytes = by_coord(gpu, grid);
  const std::uint32_t tick_before = grid.map().tick();
  vkc::Result<vx::MeshIntegrateStats> again =
      integ.integrate(grid, tet.v.data(), tet.vertex_count(), tet.i.data(),
                      tet.triangle_count(), kSigned);
  CHECK(again.ok());
  CHECK(by_coord(gpu, grid) == tet_bytes);
  CHECK(grid.map().tick() == tick_before + 1);
  {
    vkc::Result<std::vector<vol::BlockStamp>> st =
        grid.map().read_block_stamps();
    CHECK(st.ok());
    std::uint32_t stamped = 0;
    for (const vol::BlockStamp& s : st.value()) {
      stamped += s.changed == grid.map().tick() ? 1u : 0u;
    }
    CHECK(stamped == again.value().blocks);
  }

  // ---- 3. A soup writes the same bytes as its indexed mesh -------------
  // The same triangles in the same order at the same positions: nothing
  // reads the indices but to fetch a position.
  const Mesh tet_soup = soup(tet);
  CHECK(convert(tet_soup, kSigned).ok());
  CHECK(by_coord(gpu, grid) == tet_bytes);

  // So do negated zeros, which an exporter that computes one corner as 0 - 0
  // and another as 0 * -1 writes: -0.0 is the same position as +0.0. A
  // tetrahedron with vertices on the coordinate planes, whose soup has one
  // triangle's zeros negated.
  const Mesh zeros = tetrahedron(vr::Vec3f(0.1f, 0.1f, 0.1f), 0.1f);
  CHECK(convert(zeros, kSigned).ok());
  const std::map<Coord, std::vector<float>> zeros_bytes = by_coord(gpu, grid);
  Mesh zeros_soup = soup(zeros);
  int negated = 0;
  for (int k = 0; k < 3; ++k) {
    for (int axis = 0; axis < 3; ++axis) {
      if (zeros_soup.v[k][axis] == 0.0f) {
        zeros_soup.v[k][axis] = -0.0f;
        ++negated;
      }
    }
  }
  CHECK(negated > 0);
  CHECK(convert(zeros_soup, kSigned).ok());
  CHECK(by_coord(gpu, grid) == zeros_bytes);

  // ---- 4. Signed dented cube: concave edges and a concave vertex ---------
  const Mesh cube = dented_cube(vr::Vec3f(0.013f, -0.021f, 0.017f), 0.3f);
  vkc::Result<vx::MeshIntegrateStats> cube_stats = convert(cube, kSigned);
  CHECK(cube_stats.ok());
  CHECK(cube_stats.value().triangles == 12);
  if (verify(gpu, grid, "signed dented cube",
             [&](vr::Vec3f p) { return signed_closed(p, cube, trunc); }) != 0) {
    return 1;
  }
  // Its ~600 candidates binned at one group a dispatch, three dispatches a
  // pass, write the same bytes: the path a mesh past 16.7 M candidates takes
  // on a device with Vulkan's 65,535-group floor.
  {
    const std::map<Coord, std::vector<float>> cube_bytes = by_coord(gpu, grid);
    vr::detail::set_test_groups_per_dispatch(1);
    const bool split = convert(cube, kSigned).ok();
    vr::detail::set_test_groups_per_dispatch(0);
    CHECK(split);
    CHECK(by_coord(gpu, grid) == cube_bytes);
  }

  // ---- 5. Signed open quad: no topology, so no rim ----------------------
  // A +z-facing quad at z = z0. Nothing finds its rim, so the field runs on
  // past it to trunc_dist, signed by the height. The bounds sit off the voxel
  // lattice, so no voxel projects onto the rim itself.
  const float z0 = 0.052f;
  const float x0 = 0.013f, x1 = 0.297f, y0 = 0.027f, y1 = 0.243f;
  Mesh quad;
  quad.v = {{x0, y0, z0}, {x1, y0, z0}, {x1, y1, z0}, {x0, y1, z0}};
  quad.i = {0, 1, 2, 0, 2, 3};
  CHECK(convert(quad, kSigned).ok());
  int quad_observed = 0;
  if (verify(
          gpu, grid, "signed open quad",
          [&](vr::Vec3f p) {
            return signed_sheets(p, {{x0, x1, y0, y1}}, z0, trunc);
          },
          &quad_observed) != 0) {
    return 1;
  }
  CHECK(quad_observed > 0);

  // ---- 6. Shell: any mesh, distance minus the half-thickness ------------
  // The same open quad, now observed past its rim too.
  CHECK(convert(quad, kShell).ok());
  auto shell_expect = [&](const Mesh& m) {
    return [&m, trunc, shell](vr::Vec3f p) {
      Expect e;
      const float d = nearest(p, m);
      if (std::fabs(d - trunc) < kEdge) {
        e.skip = true;
        return e;
      }
      e.observed = d <= trunc;
      e.value = e.observed ? clampf(d - shell, -trunc, trunc) : 0.0f;
      return e;
    };
  };
  if (verify(gpu, grid, "shell open quad", shell_expect(quad)) != 0) return 1;

  // A triangle under a millimetre across, centred under a voxel column, so
  // that column's closest points lie inside it. Its |ab x ac|^2 is ~4e-13
  // m^4: below any fixed threshold on it worth having, and the closest point
  // must not collapse onto a corner there -- which would be 0.5 mm out.
  {
    const float r = 0.0005f, zt = 0.0523f;
    Mesh tiny;
    tiny.v = {{0.03f, 0.04f + r, zt},
              {0.03f - 0.8660254f * r, 0.04f - 0.5f * r, zt},
              {0.03f + 0.8660254f * r, 0.04f - 0.5f * r, zt}};
    tiny.i = {0, 1, 2};
    CHECK(convert(tiny, kShell).ok());
    if (verify(gpu, grid, "shell sub-millimetre triangle",
               shell_expect(tiny)) != 0) {
      return 1;
    }
  }

  // Around a closed solid the shell has two walls: an outer one, and an inner
  // one the same distance inside, with positive voxels deeper still.
  CHECK(convert(tet, kShell).ok());
  if (verify(gpu, grid, "shell tetrahedron", shell_expect(tet)) != 0) return 1;
  {
    int deep_inside_positive = 0;
    if (verify(gpu, grid, "shell inner wall", [&](vr::Vec3f p) {
          Expect e = shell_expect(tet)(p);
          if (!e.skip && e.observed && e.value > 0.0f &&
              winding(p, tet) > 0.5) {
            ++deep_inside_positive;
          }
          return e;
        }) != 0) {
      return 1;
    }
    CHECK(deep_inside_positive > 0);
  }

  // ---- 7. A write past one dispatch's budget is split, and still exact ---
  // A metre-square sheet of 2.5 mm cells: bins of thousands, and more bin
  // entries in all than one dispatch may measure. `dispatches` counts the
  // submissions, one per dispatch, so at least as many as the budget needs.
  {
    const float sx0 = -0.487f, sx1 = 0.513f, sy0 = -0.493f, sy1 = 0.507f;
    const Mesh sheet = divided_quad(sx0, sx1, sy0, sy1, z0, 400);
    vkc::Result<vx::MeshIntegrateStats> sheet_stats = convert(sheet, kSigned);
    if (!sheet_stats.ok()) {
      std::fprintf(stderr, "divided sheet: %s\n",
                   sheet_stats.status().message().c_str());
      return 1;
    }
    CHECK(sheet_stats.value().bin_entries >
          vx::MeshIntegrator::kMaxDispatchBinEntries);
    CHECK(std::uint64_t{sheet_stats.value().dispatches} *
              vx::MeshIntegrator::kMaxDispatchBinEntries >=
          sheet_stats.value().bin_entries);
    CHECK(sheet_stats.value().dispatches > 1);
    CHECK(sheet_stats.value().blocks == active_blocks(grid).size());
    if (verify(gpu, grid, "signed divided sheet", [&](vr::Vec3f p) {
          return signed_sheets(p, {{sx0, sx1, sy0, sy1}}, z0, trunc);
        }) != 0) {
      return 1;
    }
  }

  // ---- 8. Blocks the band does not reach keep what they held ------------
  CHECK(grid.clear().ok());
  const vol::BlockIndex far_block{vr::Vec3i(40, 40, 40), 0};
  CHECK(vr_test::allocate_blocks(gpu, grid.map(), &far_block, 1).ok());
  std::int32_t far_ptr = -1;
  for (const vol::BlockIndex& b : active_blocks(grid)) far_ptr = b.ptr;
  CHECK(far_ptr >= 0);
  {
    std::vector<float> tsdf = attr(gpu, grid, "tsdf");
    std::vector<float> weight = attr(gpu, grid, "weight");
    for (int k = 0; k < gp.voxels_per_block; ++k) {
      tsdf[far_ptr + k] = 7.0f;
      weight[far_ptr + k] = 7.0f;
    }
    CHECK(
        vr_test::write_attribute(gpu.device, gpu.allocator, grid, "tsdf", tsdf)
            .ok());
    CHECK(vr_test::write_attribute(gpu.device, gpu.allocator, grid, "weight",
                                   weight)
              .ok());
  }
  CHECK(grid.map()
            .allocate_from_triangles(tet.v.data(), tet.vertex_count(),
                                     tet.i.data(), tet.triangle_count())
            .ok());
  vkc::Result<vx::MeshIntegrateStats> beside =
      integ.integrate(grid, tet.v.data(), tet.vertex_count(), tet.i.data(),
                      tet.triangle_count(), kSigned);
  CHECK(beside.ok());
  CHECK(beside.value().blocks == active_blocks(grid).size() - 1);
  {
    const std::vector<float> tsdf = attr(gpu, grid, "tsdf");
    const std::vector<float> weight = attr(gpu, grid, "weight");
    for (int k = 0; k < gp.voxels_per_block; ++k) {
      CHECK(tsdf[far_ptr + k] == 7.0f);
      CHECK(weight[far_ptr + k] == 7.0f);
    }
  }

  // ---- 9. Refusals, each before the grid is written ---------------------
  auto refused = [&](const Mesh& m, const vx::MeshSdfParams& params,
                     const char* needle) -> bool {
    const std::vector<float> before = snapshot(gpu, grid);
    vkc::Result<vx::MeshIntegrateStats> r =
        integ.integrate(grid, m.v.data(), m.vertex_count(), m.i.data(),
                        m.triangle_count(), params);
    if (r.ok()) {
      std::fprintf(stderr, "expected a refusal containing \"%s\"\n", needle);
      return false;
    }
    if (r.status().message().find(needle) == std::string::npos) {
      std::fprintf(stderr, "refusal \"%s\" does not mention \"%s\"\n",
                   r.status().message().c_str(), needle);
      return false;
    }
    return snapshot(gpu, grid) == before;
  };

  // A band that was never allocated.
  CHECK(grid.clear().ok());
  CHECK(refused(tet, kSigned, "not allocated"));
  // One block of it missing.
  CHECK(grid.map()
            .allocate_from_triangles(tet.v.data(), tet.vertex_count(),
                                     tet.i.data(), tet.triangle_count())
            .ok());
  {
    const std::vector<vol::BlockIndex> band = active_blocks(grid);
    CHECK(!band.empty());
    CHECK(vr_test::remove_blocks(gpu, grid, &band[band.size() / 2], 1).ok());
  }
  CHECK(refused(tet, kSigned, "not allocated"));

  // Signed mode checks nothing about the mesh: a face wound the other way,
  // and three triangles on one edge, are both written rather than refused.
  CHECK(convert(tet, kSigned).ok());
  Mesh flipped = tet;
  std::swap(flipped.i[1], flipped.i[2]);
  CHECK(integ
            .integrate(grid, flipped.v.data(), flipped.vertex_count(),
                       flipped.i.data(), flipped.triangle_count(), kSigned)
            .ok());
  Mesh book;
  book.v = {{0.0f, 0.0f, 0.0f},
            {0.2f, 0.0f, 0.0f},
            {0.1f, 0.15f, 0.0f},
            {0.1f, -0.1f, 0.1f},
            {0.1f, -0.1f, -0.1f}};
  book.i = {0, 1, 2, 1, 0, 3, 0, 1, 4};
  CHECK(convert(book, kSigned).ok());

  // A mesh far finer than the voxels: a 30 mm patch of 0.15 mm cells puts all
  // 80 000 of its triangles in the bins of the blocks around it.
  {
    const Mesh fine = divided_quad(0.013f, 0.043f, 0.027f, 0.057f, z0, 200);
    CHECK(grid.clear().ok());
    CHECK(grid.map()
              .allocate_from_triangles(fine.v.data(), fine.vertex_count(),
                                       fine.i.data(), fine.triangle_count())
              .ok());
    CHECK(refused(fine, kShell, "kMaxBinTriangles"));
  }

  // Parameters and inputs.
  CHECK(convert(tet, kSigned).ok());
  CHECK(refused(tet, {vx::MeshSdfMode::Shell, 0.5f}, "shell_voxels"));
  CHECK(refused(tet, {vx::MeshSdfMode::Shell, 4.0f}, "shell_voxels"));
  CHECK(refused(tet, {static_cast<vx::MeshSdfMode>(7), 1.5f}, "unknown mode"));
  // MeshSdfParams::validate is integrate's rule, refusal for refusal and
  // message for message, against the grid's own truncation: a shell of 4
  // voxels is refused in this 4-voxel band and taken in an 8-voxel one.
  for (const vx::MeshSdfParams& bad :
       {vx::MeshSdfParams{vx::MeshSdfMode::Shell, 0.5f},
        vx::MeshSdfParams{vx::MeshSdfMode::Shell, 0.866f},
        vx::MeshSdfParams{vx::MeshSdfMode::Shell, 4.0f},
        vx::MeshSdfParams{vx::MeshSdfMode::Shell,
                          std::numeric_limits<float>::quiet_NaN()},
        vx::MeshSdfParams{static_cast<vx::MeshSdfMode>(7), 1.5f}}) {
    const vkc::Status checked = bad.validate(gp);
    CHECK(checked.domain() == vkc::Status::Code::InvalidArgument);
    vkc::Result<vx::MeshIntegrateStats> r =
        integ.integrate(grid, tet.v.data(), tet.vertex_count(), tet.i.data(),
                        tet.triangle_count(), bad);
    CHECK(!r.ok() && r.status().message() == checked.message());
  }
  const auto takes = [](vx::MeshSdfMode mode, float shell,
                        const vol::VoxelGridParams& g) {
    return vx::MeshSdfParams{mode, shell}.validate(g).ok();
  };
  CHECK(takes(vx::MeshSdfMode::Shell, 0.8660254f, gp));
  CHECK(takes(vx::MeshSdfMode::Shell, 3.99f, gp));
  CHECK(takes(vx::MeshSdfMode::Signed, 0.0f, gp));
  {
    vol::VoxelGridParams wide = gp;
    wide.trunc_dist = 2.0f * gp.trunc_dist;
    CHECK(takes(vx::MeshSdfMode::Shell, 4.0f, wide));
  }
  {
    const std::vector<float> before = snapshot(gpu, grid);
    CHECK(!integ.integrate(grid, nullptr, 4, tet.i.data(), 4, kSigned).ok());
    CHECK(!integ.integrate(grid, tet.v.data(), 4, nullptr, 4, kSigned).ok());
    const std::uint32_t bad[3] = {0, 1, 4};
    CHECK(!integ.integrate(grid, tet.v.data(), 4, bad, 1, kSigned).ok());
    // No triangles: nothing to write, and not an error.
    vkc::Result<vx::MeshIntegrateStats> none =
        integ.integrate(grid, tet.v.data(), 4, tet.i.data(), 0, kSigned);
    CHECK(none.ok());
    CHECK(none.value().blocks == 0);
    CHECK(snapshot(gpu, grid) == before);
  }
  // A grid without a weight attribute.
  {
    const vol::AttributeSpec tsdf_only[] = {{"tsdf", sizeof(float)}};
    vkc::Result<vol::VoxelBlockGrid> bare = vol::VoxelBlockGrid::create(
        gpu.device, gpu.allocator, gp, tsdf_only, 1);
    CHECK(bare.ok());
    CHECK(!integ
               .integrate(bare.value(), tet.v.data(), tet.vertex_count(),
                          tet.i.data(), tet.triangle_count(), kSigned)
               .ok());
  }
  // Blocks past kMaxVoxelsPerBlock: 32^3 voxels.
  {
    vol::VoxelGridParams big = gp;
    big.block_size = 32;
    big.voxels_per_block = 32 * 32 * 32;
    big.num_buckets = 8;
    big.num_blocks = 64;
    vkc::Result<vol::VoxelBlockGrid> big_grid =
        vol::VoxelBlockGrid::create(gpu.device, gpu.allocator, big, attrs, 2);
    CHECK(big_grid.ok());
    const std::vector<float> before = snapshot(gpu, big_grid.value());
    vkc::Result<vx::MeshIntegrateStats> r =
        integ.integrate(big_grid.value(), tet.v.data(), tet.vertex_count(),
                        tet.i.data(), tet.triangle_count(), kSigned);
    CHECK(!r.ok() &&
          r.status().message().find("kMaxVoxelsPerBlock") != std::string::npos);
    CHECK(snapshot(gpu, big_grid.value()) == before);
  }

  // A capacity past 256^2 with a partial final group exercises a second
  // prefix level. A coarse sheet reaches >4096 blocks, growing the compact
  // output on the first call; all later calls reuse it. Move the same input
  // allocation and flip winding so neither geometry nor bins may be cached.
  {
    vol::VoxelGridParams wide_gp = gp;
    wide_gp.block_size = 2;
    wide_gp.voxels_per_block = 8;
    wide_gp.trunc_dist = 0.015f;
    wide_gp.num_buckets = 8193;
    wide_gp.num_blocks = 65544;
    auto wide_r = vol::VoxelBlockGrid::create(gpu.device, gpu.allocator,
                                              wide_gp, attrs, 2);
    CHECK(wide_r.ok());
    vol::VoxelBlockGrid wide = std::move(wide_r).value();
    Mesh moving = divided_quad(-0.487f, 0.513f, -0.493f, 0.507f, z0, 1);
    auto write = [&]() -> vkc::Result<vx::MeshIntegrateStats> {
      VKC_TRY(wide.clear());
      VKC_ASSIGN(auto failed, wide.map().allocate_from_triangles(
                                  moving.v.data(), moving.vertex_count(),
                                  moving.i.data(), moving.triangle_count()));
      if (failed != 0) return vkc::Status::out_of_memory("wide allocation");
      return integ.integrate(wide, moving.v.data(), moving.vertex_count(),
                             moving.i.data(), moving.triangle_count(), kSigned);
    };
    auto initial = write();
    CHECK(initial.ok());
    CHECK(initial.value().blocks > 4096);
    const auto first_bytes = by_coord(gpu, wide);
    CHECK(write().ok());
    CHECK(by_coord(gpu, wide) == first_bytes);
    for (vr::Vec3f& v : moving.v) v.z += 0.137f;
    for (std::size_t t = 0; t < moving.i.size(); t += 3) {
      std::swap(moving.i[t + 1], moving.i[t + 2]);
    }
    CHECK(write().ok());
    CHECK(by_coord(gpu, wide) != first_bytes);
    if (verify(gpu, wide, "reused scratch, moved/reversed sheet",
               [&](vr::Vec3f p) {
                 Expect e =
                     signed_sheets(p, {{-0.487f, 0.513f, -0.493f, 0.507f}},
                                   z0 + 0.137f, wide_gp.trunc_dist);
                 e.value = -e.value;
                 return e;
               }) != 0)
      return 1;
    // Return to the smaller original grid after the hierarchy grew.
    CHECK(convert(tet, kSigned).ok());
    CHECK(by_coord(gpu, grid) == tet_bytes);
  }

  // ---- 10. Move semantics ------------------------------------------------
  {
    vx::MeshIntegrator moved(std::move(integ));
    CHECK(moved.valid());
    CHECK(!integ.valid());  // NOLINT(bugprone-use-after-move): the point
    CHECK(!integ
               .integrate(grid, tet.v.data(), tet.vertex_count(), tet.i.data(),
                          tet.triangle_count(), kSigned)
               .ok());
    vkc::Result<vx::MeshIntegrator> other_r =
        vx::MeshIntegrator::create(gpu.device, gpu.allocator);
    CHECK(other_r.ok());
    vx::MeshIntegrator other = std::move(other_r).value();
    other = std::move(moved);  // over a live object
    CHECK(other.valid());
    CHECK(!moved.valid());  // NOLINT(bugprone-use-after-move)
    vx::MeshIntegrator* self = &other;
    other = std::move(*self);  // self-move, laundered past -Wself-move
    CHECK(other.valid());
    CHECK(other
              .integrate(grid, tet.v.data(), tet.vertex_count(), tet.i.data(),
                         tet.triangle_count(), kSigned)
              .ok());
  }

  std::printf("voxelize_mesh_integrate: OK\n");
  return 0;
}

}  // namespace

int main() { return vr_test::run_on_gpu(gpu_main); }
