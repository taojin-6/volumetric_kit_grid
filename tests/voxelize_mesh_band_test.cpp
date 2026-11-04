// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// GPU test for the blocks MeshIntegrator::convert allocates: the band index
// (one band test a candidate pair, into a bounded set of distinct blocks) and
// the distinct insert of its list. After convert, the grid's blocks equal a
// host oracle of the band test -- worked in double with an independent closest
// point, blocks within 50 um plus 2e-5 of the longest edge of the threshold
// left to either side -- on a sphere at the codec's 80 mm band, a soup of
// slivers, large, tiny and degenerate triangles at 40 mm, two equal vertices,
// an index started at 16 slots (which overflows, grows and runs again, ending
// at most twice the power of two above the band's blocks),
// and small triangles over 20 x 10 m (four key windows). Each block is listed
// once, and converting again gives the same set. Containment: every block
// holding a voxel within trunc_dist is allocated, a 0.6 m triangle leaves no
// hole, and a slanted triangle's far box corner stays out. Growth: a 2-bucket
// grid grows once, to twice its buckets or the band's need at kGrowThreshold,
// whichever is more, placing old and new blocks in one insert -- once, since a
// 27-block band ends at 5 buckets, which two grows cannot reach -- and a grid
// the band fits below the insert's limits keeps its buckets. Skips where no
// device is present.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <memory>
#include <set>
#include <tuple>
#include <vector>

#include "test_meshes.hpp"
#include "volumetric_kit/core/testing/test_check.hpp"
#include "volumetric_kit/core/vulkan/allocator.hpp"
#include "volumetric_kit/core/vulkan/command_batch.hpp"
#include "volumetric_kit/core/vulkan/device.hpp"
#include "volumetric_kit/core/vulkan/vulkan.hpp"
#include "volumetric_kit/grid/base/math/vector_types.hpp"
#include "volumetric_kit/grid/testing/triangle_oracle.hpp"
#include "volumetric_kit/grid/volume/hash_types.hpp"
#include "volumetric_kit/grid/volume/voxel_block_grid.hpp"
#include "volumetric_kit/grid/volume/voxel_coords.hpp"
#include "volumetric_kit/grid/volume/voxel_grid.hpp"
#include "volumetric_kit/grid/volume/voxel_hash_map.hpp"
#include "volumetric_kit/grid/voxelize/mesh_band.hpp"
#include "volumetric_kit/grid/voxelize/mesh_integrator.hpp"

#include "volumetric_kit/core/testing/gpu_test.hpp"
#include "volumetric_kit/grid/base/device_requirements.hpp"

namespace vgrid = volumetric_kit::grid;
namespace vkc = volumetric_kit::core;
namespace vol = volumetric_kit::grid::volume;
namespace vx = volumetric_kit::grid::voxelize;

namespace {

using Coord = std::tuple<int, int, int>;
using test_meshes::Mesh;
using Vec3d = glm::dvec3;

// The closest point on a triangle in double: barycentric projection with a
// clamp to the edges, the shader's region test reformulated.
Vec3d closest_point(Vec3d p, Vec3d a, Vec3d b, Vec3d c) {
  const Vec3d ab = b - a;
  const Vec3d ac = c - a;
  const Vec3d n = glm::cross(ab, ac);
  const double nn = glm::dot(n, n);
  const Vec3d q = p - n * (glm::dot(n, p - a) / nn);
  const double u = glm::dot(n, glm::cross(c - b, q - b)) / nn;
  const double v = glm::dot(n, glm::cross(a - c, q - c)) / nn;
  const double w = glm::dot(n, glm::cross(ab, q - a)) / nn;
  if (nn > 0.0 && u >= 0.0 && v >= 0.0 && w >= 0.0) return q;
  const auto on_segment = [](Vec3d pt, Vec3d s0, Vec3d s1) {
    const Vec3d d = s1 - s0;
    const double dd = glm::dot(d, d);
    double t = dd > 0.0 ? glm::dot(pt - s0, d) / dd : 0.0;
    t = std::min(1.0, std::max(0.0, t));
    return s0 + d * t;
  };
  const Vec3d e[3] = {on_segment(p, a, b), on_segment(p, b, c),
                      on_segment(p, c, a)};
  Vec3d best = e[0];
  for (const Vec3d& x : e) {
    if (glm::dot(p - x, p - x) < glm::dot(p - best, p - best)) best = x;
  }
  return best;
}

struct Oracle {
  std::set<Coord> in;      // clearly inside the band
  std::set<Coord> border;  // within the tolerance of the threshold
};

// The band test (a block's centre within trunc_dist plus its half-diagonal of
// a triangle) on the host, over every used triangle's candidate box. Which
// triangles are used is the oracle's count's rule.
Oracle band_blocks(const vol::VoxelGridParams& grid, const Mesh& m) {
  Oracle out;
  const std::vector<std::uint32_t> offsets =
      vgrid_test::triangle_candidate_offsets(grid, m.v.data(), m.vertex_count(),
                                             m.i.data(), m.triangle_count(),
                                             "oracle")
          .value();
  const double span = double(grid.block_size - 1) * double(grid.voxel_size);
  const double limit = double(grid.trunc_dist) + 0.5 * std::sqrt(3.0) * span;
  for (std::uint32_t t = 0; t < m.triangle_count(); ++t) {
    if (offsets[t + 1] == offsets[t]) continue;
    const vgrid::Vec3f a = test_meshes::corner(m, t, 0);
    const vgrid::Vec3f b = test_meshes::corner(m, t, 1);
    const vgrid::Vec3f c = test_meshes::corner(m, t, 2);
    const vgrid::Vec3i lo = vol::world_to_block(
        glm::min(a, glm::min(b, c)) - vgrid::Vec3f(grid.trunc_dist), grid);
    const vgrid::Vec3i hi = vol::world_to_block(
        glm::max(a, glm::max(b, c)) + vgrid::Vec3f(grid.trunc_dist), grid);
    const double edge =
        std::max({double(vgrid::length(b - a)), double(vgrid::length(c - b)),
                  double(vgrid::length(a - c))});
    const double tol = 5e-5 + 2e-5 * edge;
    for (int z = lo.z; z <= hi.z; ++z) {
      for (int y = lo.y; y <= hi.y; ++y) {
        for (int x = lo.x; x <= hi.x; ++x) {
          const Vec3d centre = Vec3d(double(x), double(y), double(z)) *
                                   double(grid.block_size) *
                                   double(grid.voxel_size) +
                               Vec3d(0.5 * span);
          const double d = glm::length(
              centre - closest_point(centre, Vec3d(a), Vec3d(b), Vec3d(c)));
          if (std::fabs(d - limit) < tol) {
            out.border.insert({x, y, z});
          } else if (d < limit) {
            out.in.insert({x, y, z});
          }
        }
      }
    }
  }
  for (const Coord& k : out.in) out.border.erase(k);
  return out;
}

// The grid's blocks, each once.
int active(vol::VoxelBlockGrid& grid, std::set<Coord>& got) {
  got.clear();
  vkc::Result<std::vector<vol::BlockIndex>> list =
      grid.map().compact_active_blocks();
  CHECK(list.ok());
  for (const vol::BlockIndex& b : list.value()) {
    got.insert({b.coord.x, b.coord.y, b.coord.z});
  }
  CHECK(got.size() == list.value().size());
  return 0;
}

// Convert `m` into `grid`; its blocks into `got`, each written.
int convert(vx::MeshIntegrator& integ, vol::VoxelBlockGrid& grid, const Mesh& m,
            std::set<Coord>& got) {
  vkc::Result<vx::MeshIntegrateStats> stats = integ.convert(
      grid, m.v.data(), m.vertex_count(), m.i.data(), m.triangle_count());
  if (!stats.ok()) {
    std::fprintf(stderr, "convert: %s\n", stats.status().message().c_str());
    return 1;
  }
  if (active(grid, got) != 0) return 1;
  CHECK(stats.value().blocks == got.size());
  CHECK(grid.map().allocated_blocks() == got.size());
  return 0;
}

// The blocks of `m` equal the oracle's, and a second convert gives them again.
int check_mesh(const char* name, vx::MeshIntegrator& integ,
               vol::VoxelBlockGrid& grid, const Mesh& m) {
  std::set<Coord> got;
  if (convert(integ, grid, m, got) != 0) return 1;
  const Oracle want = band_blocks(grid.grid(), m);
  std::size_t border = 0;
  for (const Coord& k : got) {
    if (want.in.count(k) == 0 && want.border.count(k) == 0) {
      std::fprintf(stderr, "FAIL %s: block (%d,%d,%d) is not in the band\n",
                   name, std::get<0>(k), std::get<1>(k), std::get<2>(k));
      return 1;
    }
    border += want.border.count(k);
  }
  for (const Coord& k : want.in) {
    if (got.count(k) == 0) {
      std::fprintf(stderr, "FAIL %s: band block (%d,%d,%d) missing\n", name,
                   std::get<0>(k), std::get<1>(k), std::get<2>(k));
      return 1;
    }
  }
  std::set<Coord> again;
  if (convert(integ, grid, m, again) != 0) return 1;
  CHECK(again == got);
  std::printf("  %s: %zu blocks (%zu near the threshold)\n", name, got.size(),
              border);
  return 0;
}

vol::VoxelGridParams grid_params(float voxel, float trunc, int buckets) {
  vol::VoxelGridParams g{};
  g.voxel_size = voxel;
  g.block_size = 8;
  g.voxels_per_block = 512;
  g.trunc_dist = trunc;
  g.bucket_size = 8;
  g.num_buckets = buckets;
  g.num_blocks = 8 * buckets;
  g.max_chain = 128;
  return g;
}

vkc::Result<vol::VoxelBlockGrid> make_grid(vkc::test::GpuContext& gpu,
                                           const vol::VoxelGridParams& g) {
  const vol::AttributeSpec attrs[] = {{vol::kTsdfAttribute, sizeof(float)},
                                      {vol::kWeightAttribute, sizeof(float)}};
  return vol::VoxelBlockGrid::create(gpu.device, gpu.allocator, g, attrs, 2);
}

Mesh one_triangle(vgrid::Vec3f a, vgrid::Vec3f b, vgrid::Vec3f c) {
  Mesh m;
  m.v = {a, b, c};
  m.i = {0, 1, 2};
  return m;
}

int gpu_main(vkc::test::GpuContext& gpu) {
  vkc::Result<vx::MeshIntegrator> made =
      vx::MeshIntegrator::create(gpu.device, gpu.allocator);
  CHECK(made.ok());
  vx::MeshIntegrator& integ = made.value();

  std::printf("blocks against the oracle:\n");
  const vol::VoxelGridParams codec = grid_params(0.005f, 0.08f, 1024);
  vkc::Result<vol::VoxelBlockGrid> grid = make_grid(gpu, codec);
  CHECK(grid.ok());
  const Mesh ball =
      test_meshes::displaced_sphere({0.31f, 0.17f, -0.23f}, 0.25f, 32, 64, 1);
  if (check_mesh("sphere, 80 mm band", integ, *grid, ball) != 0) return 1;
  {
    // Random triangles of every size in a 1.2 m box: slivers (every eleventh
    // has two equal vertices), tiny, mid and large ones.
    Mesh soup;
    std::uint32_t s = 7u;
    const auto next = [&s]() {
      s = s * 1664525u + 1013904223u;
      return float(s >> 8) / float(1u << 24);
    };
    for (int t = 0; t < 400; ++t) {
      const vgrid::Vec3f a(1.2f * next() - 0.6f, 1.2f * next() - 0.2f,
                           1.2f * next() - 0.9f);
      const float size = t % 23 == 0 ? 0.25f : (t % 3 == 0 ? 0.002f : 0.03f);
      const vgrid::Vec3f b =
          a + size * vgrid::Vec3f(next() - 0.5f, next() - 0.5f, next() - 0.5f);
      const vgrid::Vec3f c =
          t % 11 == 0 ? b
                      : a + size * vgrid::Vec3f(next() - 0.5f, next() - 0.5f,
                                                0.01f * (next() - 0.5f));
      const auto base = std::uint32_t(soup.v.size());
      soup.v.insert(soup.v.end(), {a, b, c});
      soup.i.insert(soup.i.end(), {base, base + 1, base + 2});
    }
    vkc::Result<vol::VoxelBlockGrid> soup_grid =
        make_grid(gpu, grid_params(0.005f, 0.04f, 4096));
    CHECK(soup_grid.ok());
    if (check_mesh("soup, 40 mm band", integ, *soup_grid, soup) != 0) {
      return 1;
    }
  }
  {
    // Two equal vertices at coordinates that are not binary fractions, beside
    // a good triangle: the twins count zero on the device as on the host.
    Mesh twin;
    twin.v = {{-0.313463062f, 0.896191895f, -0.16501002f},
              {-0.2067177f, 0.783527195f, -0.0760641396f},
              {0.114287429f, 0.169952452f, 0.151224583f},
              {0.104702316f, 0.176071525f, 0.161035895f},
              {0.13f, 0.21f, 0.17f}};
    twin.i = {0, 1, 1, 2, 3, 3, 2, 3, 4};
    if (check_mesh("two equal vertices", integ, *grid, twin) != 0) return 1;
  }
  {
    // An index started at 16 slots overflows and grows until it holds the
    // band, the first batch running again each time: the same blocks.
    vx::detail::set_test_band_index_slots(16);
    vkc::Result<vx::MeshIntegrator> small =
        vx::MeshIntegrator::create(gpu.device, gpu.allocator);
    CHECK(small.ok());
    const int rc =
        check_mesh("sphere, index from 16 slots", *small, *grid, ball);
    // The band itself, in its own index: it ends at most twice the power of
    // two above the band's blocks.
    vkc::Result<std::unique_ptr<vx::detail::MeshBand>> band =
        vx::detail::MeshBand::create(gpu.device, gpu.allocator);
    if (rc != 0) return 1;
    CHECK(band.ok());
    vx::detail::MeshBand& b = **band;
    CHECK(grid->clear().ok());
    CHECK(b.set_mesh(codec, ball.v.data(), ball.vertex_count(), ball.i.data(),
                     ball.triangle_count(), "test")
              .ok());
    {
      vkc::CommandBatch batch(gpu.device, gpu.allocator);
      CHECK(b.record_count(batch, nullptr).ok());
      CHECK(batch.submit().ok());
    }
    CHECK(b.settle_count("test").ok());
    std::uint32_t growths = 0;
    for (;;) {
      CHECK(b.prepare_index(codec).ok());  // the first sized at the seam's
      vx::detail::set_test_band_index_slots(0);
      vkc::CommandBatch batch(gpu.device, gpu.allocator);
      vkc::Result<vol::RecordedRounds> placed =
          b.record_index(batch, grid->map(), nullptr, nullptr);
      CHECK(placed.ok());
      CHECK(batch.submit().ok());
      CHECK(grid->map().settle(placed.value()).ok());
      if (!b.index_overflowed()) break;
      CHECK(b.grow_index("test").ok());
      ++growths;
    }
    std::set<Coord> got;
    if (active(*grid, got) != 0) return 1;
    const auto n = static_cast<std::uint32_t>(got.size());
    std::uint32_t pow2 = 1;
    while (pow2 <= n) pow2 <<= 1;
    CHECK(growths > 0 && b.band_blocks() == n);
    CHECK(b.index_slots() / 2 >= n && b.index_slots() <= 2 * pow2);
    std::printf("  its index: 16 slots grew %u times to %u for %u blocks\n",
                growths, b.index_slots(), n);
  }
  {
    // 1 mm voxels (8 mm blocks): small triangles every 0.1 m along 20 m of x
    // (past 2048 blocks) and 10 m of y (past 1024), so the candidates span
    // four key windows; small triangles keep the float closest point precise
    // far from the origin.
    Mesh wide;
    for (int k = 0; k < 200; ++k) {
      const float x = 0.05f + 0.1f * float(k);
      const auto base = std::uint32_t(wide.v.size());
      wide.v.insert(wide.v.end(), {{x, 0.003f, 0.002f},
                                   {x + 0.03f, 0.007f, 0.004f},
                                   {x + 0.01f, 0.031f, 0.021f}});
      wide.i.insert(wide.i.end(), {base, base + 1, base + 2});
    }
    for (int k = 0; k < 100; ++k) {
      const float y = 0.05f + 0.1f * float(k);
      const auto base = std::uint32_t(wide.v.size());
      wide.v.insert(wide.v.end(), {{0.5f, y, 0.4f},
                                   {0.53f, y + 0.02f, 0.41f},
                                   {0.51f, y + 0.01f, 0.43f}});
      wide.i.insert(wide.i.end(), {base, base + 1, base + 2});
    }
    vkc::Result<vol::VoxelBlockGrid> wide_grid =
        make_grid(gpu, grid_params(0.001f, 0.004f, 2048));
    CHECK(wide_grid.ok());
    if (check_mesh("triangles over 20 x 10 m, 4 windows", integ, *wide_grid,
                   wide) != 0) {
      return 1;
    }
  }

  std::printf("containment:\n");
  {
    // Every block holding a voxel within trunc_dist of a tilted quad, brute
    // force over its voxels: the containment the band test promises.
    const vol::VoxelGridParams g = grid_params(0.005f, 0.04f, 1024);
    vkc::Result<vol::VoxelBlockGrid> cgrid = make_grid(gpu, g);
    CHECK(cgrid.ok());
    Mesh quad;
    quad.v = {{0.013f, 0.007f, 0.011f},
              {0.131f, 0.019f, 0.023f},
              {0.127f, 0.113f, 0.077f},
              {0.009f, 0.101f, 0.065f}};
    quad.i = {0, 1, 2, 0, 2, 3};
    std::set<Coord> got;
    if (convert(integ, *cgrid, quad, got) != 0) return 1;
    std::size_t within = 0;
    for (std::uint32_t t = 0; t < quad.triangle_count(); ++t) {
      const vgrid::Vec3f a = test_meshes::corner(quad, t, 0);
      const vgrid::Vec3f b = test_meshes::corner(quad, t, 1);
      const vgrid::Vec3f c = test_meshes::corner(quad, t, 2);
      const vgrid::Vec3i lo = vol::world_to_block(
          glm::min(a, glm::min(b, c)) - vgrid::Vec3f(g.trunc_dist), g);
      const vgrid::Vec3i hi = vol::world_to_block(
          glm::max(a, glm::max(b, c)) + vgrid::Vec3f(g.trunc_dist), g);
      for (int z = lo.z; z <= hi.z; ++z) {
        for (int y = lo.y; y <= hi.y; ++y) {
          for (int x = lo.x; x <= hi.x; ++x) {
            const vgrid::Vec3i base =
                vol::block_to_voxel(vgrid::Vec3i(x, y, z), g);
            bool hit = false;
            for (int v = 0; v < g.voxels_per_block && !hit; ++v) {
              const vgrid::Vec3f p = vol::voxel_to_world(
                  base + vgrid::Vec3i(v % 8, (v / 8) % 8, v / 64), g);
              hit = vgrid::length(p - test_meshes::closest_point(p, a, b, c)) <=
                    g.trunc_dist;
            }
            if (hit) {
              CHECK(got.count({x, y, z}) == 1);
              ++within;
            }
          }
        }
      }
    }
    CHECK(within > 0 && got.size() <= 2 * within);

    // A 0.6 m triangle spans ~15 blocks a side, far more than the cube a
    // vertex dilates into: the block at its centroid is allocated.
    const Mesh big = one_triangle({0.0f, 0.0f, 0.0f}, {0.6f, 0.0f, 0.0f},
                                  {0.0f, 0.6f, 0.0f});
    if (convert(integ, *cgrid, big, got) != 0) return 1;
    const vgrid::Vec3i centre =
        vol::world_to_block(vgrid::Vec3f(0.2f, 0.2f, 0.0f), g);
    CHECK(got.count({centre.x, centre.y, centre.z}) == 1);
    const int tb = vol::truncation_blocks(g);
    for (const vgrid::Vec3f& v : big.v) {
      const vgrid::Vec3i d = vol::world_to_block(v, g) - centre;
      CHECK(std::max({std::abs(d.x), std::abs(d.y), std::abs(d.z)}) > tb);
    }

    // A triangle in the x = y plane: its box's far corner, ~0.42 m from the
    // surface, stays out, and the surface is covered.
    const Mesh diag = one_triangle({0.0f, 0.0f, 0.0f}, {0.6f, 0.6f, 0.0f},
                                   {0.0f, 0.0f, 0.3f});
    if (convert(integ, *cgrid, diag, got) != 0) return 1;
    const vgrid::Vec3i far_block =
        vol::world_to_block(vgrid::Vec3f(0.58f, 0.02f, 0.15f), g);
    CHECK(got.count({far_block.x, far_block.y, far_block.z}) == 0);
    const vgrid::Vec3i on_block =
        vol::world_to_block(vgrid::Vec3f(0.30f, 0.30f, 0.10f), g);
    CHECK(got.count({on_block.x, on_block.y, on_block.z}) == 1);
    std::printf("  quad, 0.6 m triangle, far corner: contained\n");
  }

  std::printf("growth:\n");
  {
    // A 2-bucket grid (16 blocks) the band overflows: one grow, to the
    // band's need at kGrowThreshold, the blocks the first insert placed kept
    // and the rest placed beside them.
    vkc::Result<vol::VoxelBlockGrid> tiny =
        make_grid(gpu, grid_params(0.005f, 0.08f, 2));
    CHECK(tiny.ok());
    std::set<Coord> got;
    if (convert(integ, *tiny, ball, got) != 0) return 1;
    const double need = std::ceil(
        std::ceil(double(got.size()) / vol::VoxelHashMap::kGrowThreshold) /
        8.0);
    CHECK(tiny->grid().num_buckets ==
          std::max<std::int32_t>(4, static_cast<std::int32_t>(need)));
    std::set<Coord> want;
    if (convert(integ, *grid, ball, want) != 0) return 1;
    CHECK(got == want);
    // The band fits below the insert's limits: the grid keeps its buckets.
    const std::int32_t buckets = tiny->grid().num_buckets;
    if (convert(integ, *tiny, ball, got) != 0) return 1;
    CHECK(tiny->grid().num_buckets == buckets);
    CHECK(grid->grid().num_buckets == codec.num_buckets);
    std::printf("  2 buckets grew to %d for %zu blocks\n", buckets, got.size());

    // Once: each grow at least doubles, so from 2 buckets two grows reach 8
    // or more, and a band whose need lies between 4 and 8 buckets ends there
    // only by one grow to the need. A small triangle at a 40 mm band: 27
    // blocks, 5 buckets at kGrowThreshold.
    vkc::Result<vol::VoxelBlockGrid> small =
        make_grid(gpu, grid_params(0.005f, 0.04f, 2));
    CHECK(small.ok());
    const Mesh dot =
        one_triangle({0.011f, 0.007f, 0.013f}, {0.031f, 0.009f, 0.017f},
                     {0.019f, 0.027f, 0.021f});
    std::set<Coord> dot_got;
    if (convert(integ, *small, dot, dot_got) != 0) return 1;
    const Oracle dot_want = band_blocks(small->grid(), dot);
    CHECK(dot_want.border.empty() && dot_got == dot_want.in);
    const auto dot_need = static_cast<std::int32_t>(std::ceil(
        std::ceil(double(dot_got.size()) / vol::VoxelHashMap::kGrowThreshold) /
        8.0));
    CHECK(dot_need > 4 && dot_need < 8);
    CHECK(small->grid().num_buckets == dot_need);
    std::printf("  2 buckets grew once to %d for %zu blocks\n", dot_need,
                dot_got.size());
  }

  std::printf("voxelize_mesh_band: OK\n");
  return 0;
}

}  // namespace

int main() {
  return vkc::test::run_on_gpu(vgrid::device_requirements(), gpu_main);
}
