// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// GPU test for voxelize::detail::MeshBand's candidate count: the offsets
// counted and scanned on the device, and the block range the candidates fall
// in. The oracle is tests/support's triangle_candidate_offsets: every offset, the
// total included, and the range equal it word for word -- on a tilted quad,
// degenerate triangles (a repeated vertex, exactly collinear, one vertex three
// times, first and last, all of them), two equal vertices, non-finite
// vertices, body-like spheres at two placements (one straddling 0) and of
// 240k triangles, triangle soups of small and large triangles, and a mesh past
// two scan levels. An index at or past vertex_count is refused on the host; a
// total past 2^32, one triangle's or the sum's, at the wait, with a total of
// 0 left on the device. On half-voxel ties, where a device may divide within
// its ulps and round apart from the host, the device's count and decode agree:
// MeshIntegrator::convert allocates every band block, by host brute force.
// Skips where no device is present.

#include <cstdint>
#include <cstdio>
#include <limits>
#include <memory>
#include <set>
#include <tuple>
#include <vector>

#include "test_meshes.hpp"
#include "volumetric_kit/core/testing/buffer_readback.hpp"
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
#include "volumetric_kit/grid/voxelize/mesh_band.hpp"
#include "volumetric_kit/grid/voxelize/mesh_integrator.hpp"

#include "volumetric_kit/core/testing/gpu_test.hpp"
#include "volumetric_kit/grid/base/device_requirements.hpp"

namespace vgrid = volumetric_kit::grid;
namespace vkc = volumetric_kit::core;
namespace vol = volumetric_kit::grid::volume;

namespace {

using test_meshes::Mesh;
using Vec3f = vgrid::Vec3f;
using vgrid::voxelize::detail::MeshBand;

vol::VoxelGridParams test_grid() {
  // 5 mm voxels, 8-voxel blocks, the codec's 80 mm band.
  vol::VoxelGridParams grid{};
  grid.voxel_size = 0.005f;
  grid.block_size = 8;
  grid.voxels_per_block = 512;
  grid.trunc_dist = 0.08f;
  grid.bucket_size = 8;
  grid.num_buckets = 1024;
  grid.num_blocks = 8192;
  grid.max_chain = 128;
  return grid;
}

// The oracle's block range of the counted triangles' candidates.
MeshBand::BlockRange host_bounds(const vol::VoxelGridParams& grid,
                                 const Mesh& m,
                                 const std::vector<std::uint32_t>& offsets) {
  MeshBand::BlockRange r;
  for (std::uint32_t t = 0; t < m.triangle_count(); ++t) {
    if (offsets[t + 1] == offsets[t]) continue;
    const Vec3f a = test_meshes::corner(m, t, 0);
    const Vec3f b = test_meshes::corner(m, t, 1);
    const Vec3f c = test_meshes::corner(m, t, 2);
    const vgrid::Vec3i lo = vol::world_to_block(
        glm::min(a, glm::min(b, c)) - Vec3f(grid.trunc_dist), grid);
    const vgrid::Vec3i hi = vol::world_to_block(
        glm::max(a, glm::max(b, c)) + Vec3f(grid.trunc_dist), grid);
    r.min = r.empty ? lo : glm::min(r.min, lo);
    r.max = r.empty ? hi : glm::max(r.max, hi);
    r.empty = false;
  }
  return r;
}

// Count `m` on the device in a batch of its own; the refusal, if any.
vkc::Status count(vkc::test::GpuContext& gpu, MeshBand& band,
                  const vol::VoxelGridParams& grid, const Mesh& m) {
  VKC_TRY(band.set_mesh(grid, m.v.data(), m.vertex_count(), m.i.data(),
                        m.triangle_count(), "test"));
  vkc::CommandBatch batch(gpu.device, gpu.allocator);
  VKC_TRY(band.record_count(batch, nullptr));
  VKC_TRY(batch.submit());
  return band.settle_count("test");
}

// The device offsets and range equal the oracle's, word for word.
int same_as_host(vkc::test::GpuContext& gpu, MeshBand& band,
                 const vol::VoxelGridParams& grid, const Mesh& m,
                 const char* what) {
  vkc::Result<std::vector<std::uint32_t>> host =
      vgrid_test::triangle_candidate_offsets(grid, m.v.data(), m.vertex_count(),
                                      m.i.data(), m.triangle_count(), what);
  CHECK(host.ok());
  const vkc::Status counted = count(gpu, band, grid, m);
  if (!counted.ok()) {
    std::fprintf(stderr, "FAIL %s: device refused: %s\n", what,
                 counted.message().c_str());
    return 1;
  }
  vkc::Result<std::vector<std::uint32_t>> dev =
      vkc::test::read_back<std::uint32_t>(gpu.device, gpu.allocator,
                                          band.offset_buffer(),
                                          host->size());
  CHECK(dev.ok());
  std::size_t mismatches = 0;
  for (std::size_t t = 0; t < host->size(); ++t) {
    if ((*host)[t] != (*dev)[t] && mismatches++ < 5) {
      std::fprintf(stderr, "  %s: offsets[%zu] host %u device %u\n", what, t,
                   (*host)[t], (*dev)[t]);
    }
  }
  if (mismatches != 0) {
    std::fprintf(stderr, "FAIL %s: %zu of %zu offsets differ\n", what,
                 mismatches, host->size());
    return 1;
  }
  std::uint32_t used = 0;
  for (std::uint32_t t = 0; t < m.triangle_count(); ++t) {
    used += (*host)[t + 1] > (*host)[t] ? 1u : 0u;
  }
  CHECK(band.total() == host->back());
  CHECK(band.used_triangles() == used);
  const MeshBand::BlockRange want = host_bounds(grid, m, *host);
  const MeshBand::BlockRange got = band.bounds();
  CHECK(got.empty == want.empty);
  CHECK(want.empty || (got.min == want.min && got.max == want.max));
  std::printf("  %s: %u triangles, %u used, %u candidates: equal\n", what,
              m.triangle_count(), used, band.total());
  return 0;
}

// The device refuses `m` at the wait, as the oracle does, and leaves a total
// of 0 behind it, so nothing sized by it runs.
int device_refuses(vkc::test::GpuContext& gpu, MeshBand& band,
                   const vol::VoxelGridParams& grid, const Mesh& m,
                   const char* what) {
  CHECK(!vgrid_test::triangle_candidate_offsets(grid, m.v.data(), m.vertex_count(),
                                         m.i.data(), m.triangle_count(), what)
             .ok());
  CHECK(band.set_mesh(grid, m.v.data(), m.vertex_count(), m.i.data(),
                      m.triangle_count(), "test")
            .ok());
  const vkc::Status refused = count(gpu, band, grid, m);
  CHECK(refused.domain() == vkc::Status::Code::InvalidArgument);
  CHECK(band.total() == 0 && band.bounds().empty);
  vkc::Result<std::vector<std::uint32_t>> words =
      vkc::test::read_back<std::uint32_t>(
          gpu.device, gpu.allocator, band.offset_buffer(),
          std::size_t{m.triangle_count()} + 1);
  CHECK(words.ok() && words->back() == 0u);
  std::printf("  %s: refused at the wait (%s)\n", what,
              refused.message().c_str());
  return 0;
}

int gpu_main(vkc::test::GpuContext& gpu) {
  const vol::VoxelGridParams grid = test_grid();
  vkc::Result<std::unique_ptr<MeshBand>> made =
      MeshBand::create(gpu.device, gpu.allocator);
  CHECK(made.ok());
  MeshBand& band = **made;

  std::printf("offsets:\n");
  {
    Mesh quad;
    quad.v = {{0.013f, 0.007f, 0.011f},
              {0.131f, 0.019f, 0.023f},
              {0.127f, 0.113f, 0.077f},
              {0.009f, 0.101f, 0.065f}};
    quad.i = {0, 1, 2, 0, 2, 3};
    if (same_as_host(gpu, band, grid, quad, "tilted quad") != 0) return 1;
  }
  {
    // Degenerate triangles first, between and last: a repeated vertex,
    // exactly collinear ones (binary fractions, so the cross product is 0 in
    // floats), one vertex three times; the good ones between. Vertices 0, 3
    // and 4 are collinear in reals but not in floats: both keep that one.
    Mesh m;
    m.v = {{0.1f, 0.2f, 0.3f},     {0.15f, 0.2f, 0.3f},
           {0.1f, 0.26f, 0.3f},    {0.2f, 0.4f, 0.6f},
           {0.3f, 0.6f, 0.9f},     {-0.31f, -0.02f, 0.47f},
           {-0.27f, 0.05f, 0.41f}, {0.125f, 0.25f, 0.375f},
           {0.25f, 0.5f, 0.75f},   {0.375f, 0.75f, 1.125f}};
    m.i = {7, 7, 1,   // repeated vertex, first
           0, 1, 2,   // good
           7, 8, 9,   // exactly collinear
           2, 2, 2,   // one vertex
           5, 6, 1,   // good, negative coordinates
           0, 3, 4,   // collinear in reals only
           9, 7, 8};  // exactly collinear, last
    if (same_as_host(gpu, band, grid, m, "degenerate") != 0) return 1;
    Mesh all;
    all.v = m.v;
    all.i = {0, 0, 1, 7, 8, 9, 2, 2, 2, 8, 9, 7};
    if (same_as_host(gpu, band, grid, all, "all degenerate") != 0) return 1;
    CHECK(band.total() == 0);
    // Two equal vertices at coordinates that are not binary fractions: a
    // fused a*b - b*a keeps a rounding error, so a host that fuses (arm64
    // compilers do by default) would read area here. Zero on both.
    Mesh twin;
    twin.v = {{-0.313463062f, 0.896191895f, -0.16501002f},
              {-0.2067177f, 0.783527195f, -0.0760641396f},
              {0.114287429f, 0.169952452f, 0.151224583f},
              {0.104702316f, 0.176071525f, 0.161035895f}};
    twin.i = {0, 1, 1, 2, 3, 3, 3, 2, 3};
    if (same_as_host(gpu, band, grid, twin, "two equal vertices") != 0) {
      return 1;
    }
    CHECK(band.total() == 0);
  }
  {
    const float inf = std::numeric_limits<float>::infinity();
    const float nan = std::numeric_limits<float>::quiet_NaN();
    Mesh m;
    m.v = {{0.1f, 0.2f, 0.3f},   {0.15f, 0.2f, 0.3f},  {0.1f, 0.26f, 0.3f},
           {nan, 0.2f, 0.3f},    {0.1f, inf, 0.3f},    {0.1f, 0.2f, -inf},
           {0.2f, 0.25f, 0.33f}, {nan, nan, nan}};
    // Vertex 7 is NaN but no triangle uses it.
    m.i = {0, 1, 3, 0, 1, 2, 4, 1, 2, 1, 2, 6, 0, 5, 6, 3, 4, 5, 0, 2, 6};
    if (same_as_host(gpu, band, grid, m, "non-finite") != 0) return 1;
  }
  if (same_as_host(gpu, band, grid,
                   test_meshes::displaced_sphere({0.31f, 1.07f, -0.23f}, 0.25f,
                                                 125, 200, 1),
                   "body-like, 50k") != 0 ||
      same_as_host(
          gpu, band, grid,
          test_meshes::displaced_sphere({0.0f, 0.0f, 0.0f}, 0.25f, 125, 200, 2),
          "body-like across 0") != 0 ||
      same_as_host(gpu, band, grid,
                   test_meshes::displaced_sphere({-1.3f, 0.4f, 2.1f}, 0.6f, 300,
                                                 400, 3),
                   "body-like, 240k") != 0 ||
      same_as_host(gpu, band, grid,
                   test_meshes::soup({0.2f, 0.9f, -0.1f}, 50000, 0.05f, 4),
                   "soup, 50k") != 0 ||
      same_as_host(gpu, band, grid,
                   test_meshes::soup({0.0f, 0.0f, 0.0f}, 3000, 0.4f, 5),
                   "soup, large triangles") != 0) {
    return 1;
  }
  {
    // Past two scan levels (more than 65,536 triangles), on another lattice.
    vol::VoxelGridParams coarse = grid;
    coarse.voxel_size = 0.01f;
    coarse.trunc_dist = 0.03f;
    if (same_as_host(gpu, band, coarse,
                     test_meshes::soup({0.0f, 0.0f, 0.0f}, 70000, 0.02f, 6),
                     "70k, 10 mm voxels") != 0) {
      return 1;
    }
  }

  std::printf("refusals:\n");
  {
    // On the host, before anything is staged.
    Mesh m = test_meshes::displaced_sphere({0.0f, 0.0f, 0.0f}, 0.1f, 20, 30, 8);
    m.i[37] = m.vertex_count();  // one past the last
    CHECK(band.set_mesh(grid, m.v.data(), m.vertex_count(), m.i.data(),
                        m.triangle_count(), "test")
              .domain() == vkc::Status::Code::InvalidArgument);
    m.i[37] = std::numeric_limits<std::uint32_t>::max();
    CHECK(band.set_mesh(grid, m.v.data(), m.vertex_count(), m.i.data(),
                        m.triangle_count(), "test")
              .domain() == vkc::Status::Code::InvalidArgument);
    CHECK(band.set_mesh(grid, m.v.data(), 0, m.i.data(), 1, "test").domain() ==
          vkc::Status::Code::InvalidArgument);  // triangles, no vertices
    CHECK(band.set_mesh(grid, nullptr, 3, m.i.data(), 1, "test").domain() ==
          vkc::Status::Code::InvalidArgument);
    std::printf("  index at vertex_count, 2^32 - 1, null: refused on the host\n");
  }
  {
    // A model in millimetres read as metres: one triangle 1 km across has
    // ~25,000^3 candidates, past 2^32 in one triangle's product.
    Mesh m;
    m.v = {{0.0f, 0.0f, 0.0f}, {1000.0f, 0.0f, 0.0f}, {0.0f, 1000.0f, 1000.0f}};
    m.i = {0, 1, 2};
    if (device_refuses(gpu, band, grid, m, "one triangle past 2^32") != 0) {
      return 1;
    }
    // Five triangles of ~1.0e9 candidates each: each fits, the sum does not.
    Mesh five;
    for (std::uint32_t t = 0; t < 5; ++t) {
      const float o = 0.37f * float(t);
      five.v.push_back({o, o, o});
      five.v.push_back({o + 40.0f, o, o});
      five.v.push_back({o, o + 40.0f, o + 40.0f});
      five.i.insert(five.i.end(), {3 * t, 3 * t + 1, 3 * t + 2});
    }
    if (device_refuses(gpu, band, grid, five, "sum past 2^32") != 0) return 1;
  }

  std::printf("ties:\n");
  {
    // Every block holding a voxel within trunc_dist of a triangle, brute
    // force over each triangle's box and a block beyond it on every side.
    const Mesh tied = test_meshes::ties(300, 11, 40);  // within 0.1 m
    const vol::AttributeSpec attrs[] = {
        {vol::kTsdfAttribute, sizeof(float)},
        {vol::kWeightAttribute, sizeof(float)}};
    vkc::Result<vol::VoxelBlockGrid> block_grid =
        vol::VoxelBlockGrid::create(gpu.device, gpu.allocator, grid, attrs, 2);
    vkc::Result<vgrid::voxelize::MeshIntegrator> integ =
        vgrid::voxelize::MeshIntegrator::create(gpu.device, gpu.allocator);
    CHECK(block_grid.ok() && integ.ok());
    CHECK(integ
              ->convert(*block_grid, tied.v.data(), tied.vertex_count(),
                        tied.i.data(), tied.triangle_count())
              .ok());
    vkc::Result<std::vector<vol::BlockIndex>> active =
        block_grid->map().compact_active_blocks();
    CHECK(active.ok());
    std::set<std::tuple<int, int, int>> got;
    for (const vol::BlockIndex& b : *active) {
      got.insert({b.coord.x, b.coord.y, b.coord.z});
    }
    std::size_t want = 0;
    for (std::uint32_t t = 0; t < tied.triangle_count(); ++t) {
      const Vec3f a = test_meshes::corner(tied, t, 0);
      const Vec3f b = test_meshes::corner(tied, t, 1);
      const Vec3f c = test_meshes::corner(tied, t, 2);
      const vgrid::Vec3i lo = vol::world_to_block(
          glm::min(a, glm::min(b, c)) - Vec3f(grid.trunc_dist), grid);
      const vgrid::Vec3i hi = vol::world_to_block(
          glm::max(a, glm::max(b, c)) + Vec3f(grid.trunc_dist), grid);
      for (int z = lo.z - 1; z <= hi.z + 1; ++z) {
        for (int y = lo.y - 1; y <= hi.y + 1; ++y) {
          for (int x = lo.x - 1; x <= hi.x + 1; ++x) {
            const vgrid::Vec3i base =
                vol::block_to_voxel(vgrid::Vec3i(x, y, z), grid);
            bool hit = false;
            for (int v = 0; v < grid.voxels_per_block && !hit; ++v) {
              const Vec3f p = vol::voxel_to_world(
                  base + vgrid::Vec3i(v % 8, (v / 8) % 8, v / 64), grid);
              hit = vgrid::length(p - test_meshes::closest_point(p, a, b, c)) <=
                    grid.trunc_dist;
            }
            if (hit && got.count({x, y, z}) == 0) {
              std::fprintf(stderr, "FAIL ties: band block (%d,%d,%d) missing\n",
                           x, y, z);
              return 1;
            }
            want += hit ? 1 : 0;
          }
        }
      }
    }
    CHECK(want > 0);
    std::printf("  300 triangles on half-voxel ties: every band block of %zu "
                "allocated\n",
                got.size());
  }

  std::printf("voxelize_mesh_candidates: OK\n");
  return 0;
}

}  // namespace

int main() {
  return vkc::test::run_on_gpu(vgrid::device_requirements(), gpu_main);
}
