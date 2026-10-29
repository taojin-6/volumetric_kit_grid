// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// GPU test: a marching-cubes vertex on a corner whose distance is exactly the
// iso-value is that corner, fl(k * voxel_size) on each axis, bit for bit, with
// that corner's attributes and normal, from both kernels on every device; and a
// vertex whose crossing is only near a corner is not moved onto it.
//
// The field is a set of lattice planes at 1 cm, each in a region of its own:
// normal to x, y and z, with the distance growing along the normal (the zero
// corner is a crossing edge's high end) or falling (its low end), at k = -248
// (-2.48 m, far from the origin), -1 and 0. Each column has its own slope, so
// the crossing edges carry many different distances. The high end is where the
// ratio is x / x, which NVIDIA's and Apple's division round below 1 for some
// x; near the origin that put the vertex an ulp off the corner, or at -6e-10
// for the corner at 0. One more region has slopes below the near-tangent
// guard's 1e-6, where the ratio came out at slope / 1e-6, 0.1 to 0.9, on
// every device. The "color" attribute is one value on each plane's voxels and
// another everywhere else, so a vertex must carry its own corner's. So must
// its normal: every distance on the plane is the iso-value, so a plane
// corner's gradient lies along the plane's normal, while the corners beside it
// tilt with their columns' slopes.
//
// Four more regions, at k = 0, put the plane's voxels 1e-8 above or below the
// iso-value: their crossings are near the corner but not on it, so along the
// normal no vertex there may be on the lattice (a vertex moved onto its
// corner would be exactly 0).
//
// The grid is meshed at iso 0 and again at iso 0.0025 with every distance
// shifted by it. Every triangle of every region lies in its plane, so every
// corner it has is checked. Skips where no device is present.

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <utility>
#include <vector>

#include "volumetric_kit/core/testing/test_check.hpp"
#include "volumetric_kit/core/vulkan/allocator.hpp"
#include "volumetric_kit/core/vulkan/device.hpp"
#include "volumetric_kit/grid/base/math/vector_types.hpp"
#include "volumetric_kit/grid/mesh/marching_cubes.hpp"
#include "volumetric_kit/grid/mesh/mesh.hpp"
#include "volumetric_kit/grid/testing/device_blocks.hpp"
#include "volumetric_kit/grid/testing/grid_readback.hpp"
#include "volumetric_kit/grid/volume/hash_types.hpp"
#include "volumetric_kit/grid/volume/voxel_block_grid.hpp"
#include "volumetric_kit/grid/volume/voxel_grid.hpp"
#include "volumetric_kit/grid/volume/voxel_hash_map.hpp"

#include "volumetric_kit/core/testing/gpu_test.hpp"
#include "volumetric_kit/grid/base/device_requirements.hpp"

namespace vgrid = volumetric_kit::grid;
namespace vkc = volumetric_kit::core;
namespace vol = volumetric_kit::grid::volume;
namespace mesh = volumetric_kit::grid::mesh;

namespace {

constexpr int kBlock = 8;
constexpr float kVoxel = 0.01f;
// A region is two blocks across its plane: 15 x 15 whole cells, two
// triangles each, all in the plane.
constexpr std::size_t kTrianglesPerRegion = 15 * 15 * 2;
// RGBA8, r in the low byte: the plane's voxels red, every other voxel blue.
constexpr std::uint32_t kPlaneColor = 0xFF0000FFu;
constexpr std::uint32_t kOtherColor = 0xFFFF0000u;

std::uint32_t bits(float f) {
  std::uint32_t u = 0;
  std::memcpy(&u, &f, sizeof(u));
  return u;
}

int floor_div(int a, int b) { return a >= 0 ? a / b : -((-a + b - 1) / b); }

// One lattice plane: distance s * (k[axis] - k0) * slope(column) + near.
struct Region {
  int axis = 0;
  int s = 1;
  int k0 = 0;
  bool tiny = false;  // slopes below the near-tangent guard's 1e-6
  float near = 0.0f;  // the plane's distance; 0: its corners are the crossing
  std::array<int, 3> lo{}, hi{};  // blocks, inclusive
};

float slope(const Region& r, int u, int w) {
  std::uint32_t h = static_cast<std::uint32_t>(u * 7919 + w * 104729 + 17);
  h ^= h >> 16;
  h *= 0x7feb352dU;
  h ^= h >> 15;
  h *= 0x846ca68bU;
  h ^= h >> 16;
  const float t = static_cast<float>(h % 100000u) * 1.0e-5f;  // [0, 1)
  return r.tiny ? 1.0e-7f + 8.0e-7f * t : 0.001f + 0.01f * t;
}

// The distance at @p k, shifted by @p iso: on the plane d is 0, so an exact
// region's plane is at iso itself.
float field(const Region& r, const std::array<int, 3>& k, float iso) {
  const float d = static_cast<float>(r.s * (k[r.axis] - r.k0));
  return d * slope(r, k[(r.axis + 1) % 3], k[(r.axis + 2) % 3]) + r.near + iso;
}

std::vector<Region> regions() {
  std::vector<Region> out;
  int n = 0;
  for (int k0 : {-248, -1, 0}) {
    for (int axis = 0; axis < 3; ++axis) {
      for (int s : {1, -1}) {
        Region r;
        r.axis = axis;
        r.s = s;
        r.k0 = k0;
        out.push_back(r);
      }
    }
  }
  Region tiny;
  tiny.axis = 1;
  tiny.k0 = -248;
  tiny.tiny = true;
  out.push_back(tiny);
  for (float near : {1.0e-8f, -1.0e-8f}) {
    for (int s : {1, -1}) {
      Region r;
      r.axis = 2;
      r.s = s;
      r.near = near;
      out.push_back(r);
    }
  }
  // Across the plane, two blocks at -32 + 4n: apart from every other region.
  // Along the normal, the blocks holding k0 - 1 through k0 + 1.
  for (Region& r : out) {
    r.lo.fill(-32 + 4 * n);
    r.hi.fill(-31 + 4 * n);
    r.lo[r.axis] = floor_div(r.k0 - 1, kBlock);
    r.hi[r.axis] = floor_div(r.k0 + 1, kBlock);
    ++n;
  }
  return out;
}

int region_of(const std::vector<Region>& rs, const std::array<int, 3>& k) {
  for (std::size_t i = 0; i < rs.size(); ++i) {
    bool in = true;
    for (int a = 0; a < 3; ++a) {
      const int b = floor_div(k[a], kBlock);
      in = in && b >= rs[i].lo[a] && b <= rs[i].hi[a];
    }
    if (in) return static_cast<int>(i);
  }
  return -1;
}

// Every triangle corner of @p m: in an exact region, its lattice point, that
// point's attribute and the plane's normal; in a near region, off the lattice
// along the normal.
// The count checked and the count wrong, the first few printed.
std::pair<std::size_t, std::size_t> check_corners(const char* kernel,
                                                  const mesh::Mesh& m,
                                                  const std::vector<Region>& rs,
                                                  float iso) {
  std::size_t checked = 0;
  std::size_t wrong = 0;
  // The plane voxels' attribute as the kernel unpacks it: the first one seen.
  bool have_red = false;
  std::uint32_t red = 0;
  for (const std::uint32_t index : m.indices) {
    const mesh::Vertex& v = m.vertices[index];
    const float c[3] = {v.position.x, v.position.y, v.position.z};
    std::array<int, 3> k{};
    for (int a = 0; a < 3; ++a) {
      k[a] = static_cast<int>(std::lround(double(c[a]) / double(kVoxel)));
    }
    const int r = region_of(rs, k);
    ++checked;
    bool ok = r >= 0;
    if (ok && rs[r].near != 0.0f) {
      const int a = rs[r].axis;
      ok = c[a] != static_cast<float>(k[a]) * kVoxel;  // by value: -0 is on it
    } else if (ok) {
      ok = field(rs[r], k, iso) == iso;
      for (int a = 0; a < 3; ++a) {
        ok = ok && bits(c[a]) == bits(static_cast<float>(k[a]) * kVoxel);
      }
      if (!have_red) {
        have_red = true;
        red = bits(v.color.x);
      }
      ok = ok && v.color.x > 0.5f && bits(v.color.x) == red &&
           v.color.y == 0.0f && v.color.z == 0.0f;
      const float n[3] = {v.normal.x, v.normal.y, v.normal.z};
      const int a = rs[r].axis;
      ok = ok && std::fabs(n[a] - static_cast<float>(rs[r].s)) < 1e-6f &&
           std::fabs(n[(a + 1) % 3]) < 1e-6f &&
           std::fabs(n[(a + 2) % 3]) < 1e-6f;
    }
    if (!ok && ++wrong <= 8) {
      std::fprintf(stderr,
                   "%s, iso %g: corner (%d, %d, %d) at (%08x, %08x, %08x), "
                   "color (%g, %g, %g), normal (%g, %g, %g); lattice (%08x, "
                   "%08x, %08x)\n",
                   kernel, double(iso), k[0], k[1], k[2], bits(c[0]),
                   bits(c[1]), bits(c[2]), double(v.color.x), double(v.color.y),
                   double(v.color.z), double(v.normal.x), double(v.normal.y),
                   double(v.normal.z), bits(static_cast<float>(k[0]) * kVoxel),
                   bits(static_cast<float>(k[1]) * kVoxel),
                   bits(static_cast<float>(k[2]) * kVoxel));
    }
  }
  return {checked, wrong};
}

int gpu_main(vkc::test::GpuContext& gpu) {
  const std::vector<Region> rs = regions();
  // No two regions touch, so no cell reads two fields.
  for (std::size_t i = 0; i < rs.size(); ++i) {
    for (std::size_t j = i + 1; j < rs.size(); ++j) {
      bool apart = false;
      for (int a = 0; a < 3; ++a) {
        apart = apart || rs[i].hi[a] + 1 < rs[j].lo[a] ||
                rs[j].hi[a] + 1 < rs[i].lo[a];
      }
      CHECK(apart);
    }
  }

  vol::VoxelGridParams gp{};
  gp.voxel_size = kVoxel;
  gp.block_size = kBlock;
  gp.voxels_per_block = kBlock * kBlock * kBlock;
  gp.trunc_dist = 0.04f;  // unused by meshing; must pass validate()
  gp.bucket_size = 8;
  gp.num_buckets = 128;
  gp.num_blocks = 1024;
  gp.max_chain = 128;
  const vol::AttributeSpec attrs[] = {
      {vol::kTsdfAttribute, sizeof(float)},
      {vol::kWeightAttribute, sizeof(float)},
      {vol::kColorAttribute, sizeof(std::uint32_t)}};
  vkc::Result<vol::VoxelBlockGrid> grid_result =
      vol::VoxelBlockGrid::create(gpu.device, gpu.allocator, gp, attrs, 3);
  CHECK(grid_result.ok());
  vol::VoxelBlockGrid grid = std::move(grid_result).value();

  std::vector<vol::BlockIndex> blocks;
  for (const Region& r : rs) {
    for (int z = r.lo[2]; z <= r.hi[2]; ++z) {
      for (int y = r.lo[1]; y <= r.hi[1]; ++y) {
        for (int x = r.lo[0]; x <= r.hi[0]; ++x) {
          vol::BlockIndex b{};
          b.coord = vgrid::Vec3i(x, y, z);
          blocks.push_back(b);
        }
      }
    }
  }
  vkc::Result<std::uint32_t> failed =
      vgrid_test::allocate_blocks(gpu, grid.map(), blocks);
  CHECK(failed.ok() && failed.value() == 0);
  vkc::Result<std::vector<vol::BlockIndex>> active =
      grid.map().compact_active_blocks();
  CHECK(active.ok() && active.value().size() == blocks.size());
  auto tsdf = vgrid_test::read_attribute<float>(gpu.device, gpu.allocator, grid,
                                                vol::kTsdfAttribute);
  auto weight = vgrid_test::read_attribute<float>(gpu.device, gpu.allocator,
                                                  grid, vol::kWeightAttribute);
  auto color = vgrid_test::read_attribute<std::uint32_t>(
      gpu.device, gpu.allocator, grid, vol::kColorAttribute);
  CHECK(tsdf.ok() && weight.ok() && color.ok());

  for (const float iso : {0.0f, 0.0025f}) {
    for (const vol::BlockIndex& b : active.value()) {
      for (int lz = 0; lz < kBlock; ++lz) {
        for (int ly = 0; ly < kBlock; ++ly) {
          for (int lx = 0; lx < kBlock; ++lx) {
            const std::array<int, 3> k = {b.coord.x * kBlock + lx,
                                          b.coord.y * kBlock + ly,
                                          b.coord.z * kBlock + lz};
            const int r = region_of(rs, k);
            CHECK(r >= 0);
            const auto i =
                static_cast<std::size_t>(b.ptr) +
                static_cast<std::size_t>(lx + kBlock * (ly + kBlock * lz));
            tsdf.value()[i] = field(rs[r], k, iso);
            weight.value()[i] = 1.0f;
            color.value()[i] =
                k[rs[r].axis] == rs[r].k0 ? kPlaneColor : kOtherColor;
          }
        }
      }
    }
    CHECK(vgrid_test::write_attribute(gpu.device, gpu.allocator, grid,
                                      vol::kTsdfAttribute, tsdf.value())
              .ok());
    CHECK(vgrid_test::write_attribute(gpu.device, gpu.allocator, grid,
                                      vol::kWeightAttribute, weight.value())
              .ok());
    CHECK(vgrid_test::write_attribute(gpu.device, gpu.allocator, grid,
                                      vol::kColorAttribute, color.value())
              .ok());

    for (const bool share : {false, true}) {
      const char* kernel = share ? "sharing kernel" : "default kernel";
      mesh::MarchingCubesConfig config;
      config.share_vertices = share;
      vkc::Result<mesh::MarchingCubes> mc =
          mesh::MarchingCubes::create(gpu.device, gpu.allocator, config);
      CHECK(mc.ok());
      vkc::Result<mesh::Mesh> m = mc.value().extract_host(grid, iso);
      CHECK(m.ok());
      CHECK(m.value().triangle_count() == rs.size() * kTrianglesPerRegion);
      const auto [checked, wrong] = check_corners(kernel, m.value(), rs, iso);
      std::printf("%s, iso %g: %zu corners, %zu off\n", kernel, double(iso),
                  checked, wrong);
      CHECK(checked == rs.size() * kTrianglesPerRegion * 3);
      CHECK(wrong == 0);
    }
  }
  return 0;
}

}  // namespace

int main() {
  return vkc::test::run_on_gpu(vgrid::device_requirements(), gpu_main);
}
