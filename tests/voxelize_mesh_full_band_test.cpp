// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// GPU test for MeshSdfParams::full_band_m: the band written whole only near
// the field's zero level, plus a one-voxel apron.
//
//   - oracle: against a host oracle of which blocks lie within the full band
//     (a block's centre within full_band_m + its half-diagonal of some
//     triangle) and of the apron (every voxel one step from such a block),
//     convert writes the oracle's full blocks whole and its apron voxels, each
//     the bytes the whole-band conversion writes, the rest of the band's
//     blocks unobserved, and counts both sets. On every write layout the
//     device takes;
//   - shell: in shell mode, every voxel whose field lies within full_band_m
//     of zero -- the shell's half-thickness out from the mesh -- and every
//     neighbour one step from it holds the whole band's bytes;
//   - curtain: a closest-face sign flip far from the mesh -- the plane
//     extended past an open sheet's rim -- that runs along the face between a
//     block the codec codes and its neighbour outside the full band. The
//     codec cuts the coded block's voxels on that face only through cells
//     that reach into the neighbour, so a conversion without the apron codes
//     them uncut. With it, a true-SDF frame is byte-identical to the whole
//     band's;
//   - refusals: a negative or non-finite full_band_m.
//
// Skips where no device is present.

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <map>
#include <tuple>
#include <utility>
#include <vector>

#include "test_meshes.hpp"
#include "volumetric_kit/core/testing/test_check.hpp"
#include "volumetric_kit/core/vulkan/allocator.hpp"
#include "volumetric_kit/core/vulkan/device.hpp"
#include "volumetric_kit/grid/base/math/vector_types.hpp"
#include "volumetric_kit/grid/codec/decode_grid.hpp"
#include "volumetric_kit/grid/codec/decoder.hpp"
#include "volumetric_kit/grid/codec/encoder.hpp"
#include "volumetric_kit/grid/volume/hash_types.hpp"
#include "volumetric_kit/grid/volume/voxel_block_grid.hpp"
#include "volumetric_kit/grid/volume/voxel_coords.hpp"
#include "volumetric_kit/grid/volume/voxel_grid.hpp"
#include "volumetric_kit/grid/volume/voxel_hash_map.hpp"
#include "volumetric_kit/grid/voxelize/mesh_band.hpp"
#include "volumetric_kit/grid/voxelize/mesh_integrator.hpp"

#include "volumetric_kit/core/testing/gpu_test.hpp"
#include "volumetric_kit/grid/base/device_requirements.hpp"
#include "volumetric_kit/grid/testing/grid_readback.hpp"

namespace vgrid = volumetric_kit::grid;
namespace vkc = volumetric_kit::core;
namespace vol = volumetric_kit::grid::volume;
namespace vx = volumetric_kit::grid::voxelize;
namespace codec = volumetric_kit::grid::codec;

namespace {

using test_meshes::closest_point;
using test_meshes::corner;
using test_meshes::Mesh;
using Coord = std::tuple<int, int, int>;
using Blocks = std::map<Coord, std::vector<float>>;  // tsdf then weight

constexpr float kVoxel = 0.005f;
constexpr int kBs = 8;
constexpr int kVpb = kBs * kBs * kBs;

// The codec_mesh template's grid: 5 mm voxels, a 16-voxel band.
vol::VoxelGridParams grid_params() {
  vol::VoxelGridParams gp{};
  gp.voxel_size = kVoxel;
  gp.block_size = kBs;
  gp.voxels_per_block = kVpb;
  gp.trunc_dist = 16 * kVoxel;
  gp.bucket_size = 8;
  gp.num_buckets = 512;
  gp.num_blocks = 8 * 512;
  gp.max_chain = 128;
  return gp;
}

// ---- Fixtures --------------------------------------------------------------

void append(Mesh& to, const Mesh& m) {
  const auto base = static_cast<std::uint32_t>(to.v.size());
  to.v.insert(to.v.end(), m.v.begin(), m.v.end());
  for (std::uint32_t i : m.i) to.i.push_back(base + i);
}

// A closed octahedron of radius r at c, wound outward.
Mesh octahedron(vgrid::Vec3f c, float r) {
  Mesh m;
  m.v = {c + vgrid::Vec3f(r, 0, 0), c + vgrid::Vec3f(-r, 0, 0),
         c + vgrid::Vec3f(0, r, 0), c + vgrid::Vec3f(0, -r, 0),
         c + vgrid::Vec3f(0, 0, r), c + vgrid::Vec3f(0, 0, -r)};
  m.i = {0, 2, 4, 2, 1, 4, 1, 3, 4, 3, 0, 4,
         2, 0, 5, 1, 2, 5, 3, 1, 5, 0, 3, 5};
  test_meshes::orient_outward(m, c);
  return m;
}

// An open +x-facing rectangle in the plane x = x0: y in [y0, y1], z in
// [z0, z1]. Past its rim the closest-face sign runs on: the sign of x - x0.
Mesh x_sheet(float x0, float y0, float y1, float z0, float z1) {
  Mesh m;
  m.v = {vgrid::Vec3f(x0, y0, z0), vgrid::Vec3f(x0, y1, z0),
         vgrid::Vec3f(x0, y1, z1), vgrid::Vec3f(x0, y0, z1)};
  m.i = {0, 1, 2, 0, 2, 3};
  return m;
}

// The curtain: K = block (0, 0, 0) holds a speck that the codec codes; the
// sheet's plane x = 37.5 mm, midway between K's last voxel layer (35 mm) and
// its +x neighbour N's first (40 mm), stands below a rim at z = -24 mm, so on
// K and N's low layers the plane runs on past the rim as a sign flip 24 mm or
// more from the mesh, between K and N alone. N's centre lies 46 mm from the
// rim and 51 mm from the speck: outside the full band's 44 mm.
constexpr float kPlaneX = 0.0375f;
constexpr float kRimZ = -0.024f;
Mesh curtain() {
  Mesh m = octahedron(vgrid::Vec3f(0.0025f, 0.0175f, 0.0175f), 0.004f);
  append(m, x_sheet(kPlaneX, -0.1f, 0.14f, -0.15f, kRimZ));
  return m;
}

// A closed solid, an open sheet beside it and a speck: full blocks, apron
// blocks on faces, edges and corners, and band blocks written not at all.
Mesh scene() {
  Mesh m =
      test_meshes::dented_cube(vgrid::Vec3f(-0.052f, -0.047f, -0.041f), 0.1f);
  append(m, x_sheet(0.1375f, -0.06f, 0.07f, -0.05f, 0.03f));
  append(m, octahedron(vgrid::Vec3f(-0.0625f, 0.1225f, 0.0825f), 0.003f));
  return m;
}

// ---- Plumbing
// ----------------------------------------------------------------

struct Converted {
  vx::MeshIntegrateStats stats;
  Blocks blocks;
};

vkc::Result<Blocks> by_coord(const vkc::test::Gpu& gpu,
                             vol::VoxelBlockGrid& grid) {
  VKC_ASSIGN(const std::vector<vol::BlockIndex> active,
             grid.map().compact_active_blocks());
  VKC_ASSIGN(
      const std::vector<float> tsdf,
      vgrid_test::read_blocks<float>(gpu, grid, vol::kTsdfAttribute, active));
  VKC_ASSIGN(
      const std::vector<float> weight,
      vgrid_test::read_blocks<float>(gpu, grid, vol::kWeightAttribute, active));
  Blocks out;
  for (std::size_t i = 0; i < active.size(); ++i) {
    const vol::BlockIndex& b = active[i];
    std::vector<float>& vals = out[{b.coord.x, b.coord.y, b.coord.z}];
    const auto first = static_cast<std::ptrdiff_t>(i * kVpb);
    vals.assign(tsdf.begin() + first, tsdf.begin() + first + kVpb);
    vals.insert(vals.end(), weight.begin() + first,
                weight.begin() + first + kVpb);
  }
  return out;
}

// Convert, and read every block back.
vkc::Result<Converted> convert(const vkc::test::Gpu& gpu,
                               vol::VoxelBlockGrid& grid,
                               vx::MeshIntegrator& integ, const Mesh& m,
                               vx::MeshSdfParams params) {
  Converted out;
  VKC_ASSIGN(out.stats, integ.convert(grid, m.v.data(), m.vertex_count(),
                                      m.i.data(), m.triangle_count(), params));
  VKC_ASSIGN(out.blocks, by_coord(gpu, grid));
  return out;
}

vx::MeshSdfParams with_full_band(float full_band_m) {
  vx::MeshSdfParams params;
  params.full_band_m = full_band_m;
  return params;
}

vgrid::Vec3i coord_of(const Coord& c) {
  return vgrid::Vec3i(std::get<0>(c), std::get<1>(c), std::get<2>(c));
}

vgrid::Vec3i local_of(int local) {
  return vgrid::Vec3i(local % kBs, (local / kBs) % kBs, local / (kBs * kBs));
}

vgrid::Vec3i voxel_of(const Coord& c, int local) {
  return vol::block_to_voxel(coord_of(c), grid_params()) + local_of(local);
}

// The host oracle's distance from block `c`'s centre to the mesh, in double.
double centre_distance(const Coord& c, const Mesh& m) {
  const float span = float(kBs - 1) * kVoxel;
  const vgrid::Vec3f centre =
      vgrid::Vec3f(float(std::get<0>(c) * kBs), float(std::get<1>(c) * kBs),
                   float(std::get<2>(c) * kBs)) *
          kVoxel +
      vgrid::Vec3f(0.5f * span);
  double best = std::numeric_limits<double>::infinity();
  for (std::size_t t = 0; t < m.triangle_count(); ++t) {
    const vgrid::Vec3f q = closest_point(centre, corner(m, t, 0),
                                         corner(m, t, 1), corner(m, t, 2));
    const vgrid::Vec3f d = centre - q;
    best = std::fmin(best, std::sqrt(double(d.x) * d.x + double(d.y) * d.y +
                                     double(d.z) * d.z));
  }
  return best;
}

// ---- Cases -----------------------------------------------------------------

// Every block of the whole-band conversion, graded by the host oracle: full
// blocks are written whole, every voxel one step from a full block is written,
// both with the whole band's bytes, and every other voxel is unobserved.
int oracle_case(const vkc::test::Gpu& gpu, vol::VoxelBlockGrid& grid,
                vx::MeshIntegrator& integ) {
  const Mesh m = scene();
  const float full = codec::true_sdf_keep_distance(kVoxel);
  vkc::Result<Converted> whole = convert(gpu, grid, integ, m, {});
  vkc::Result<Converted> part =
      convert(gpu, grid, integ, m, with_full_band(full));
  CHECK(whole.ok() && part.ok());
  const Blocks& a = whole.value().blocks;
  const Blocks& b = part.value().blocks;
  CHECK(a.size() == b.size());  // the same band allocated
  CHECK(whole.value().stats.blocks == a.size());
  CHECK(whole.value().stats.full_blocks == 0);

  // The oracle's full set, refusing a fixture with a block near the radius,
  // where rounding may decide.
  const double radius =
      double(full) + 0.5 * std::sqrt(3.0) * double((kBs - 1) * kVoxel);
  std::map<Coord, bool> is_full;
  std::size_t n_full = 0;
  for (const auto& [c, vals] : a) {
    const double d = centre_distance(c, m);
    CHECK(std::fabs(d - radius) > 1e-4);
    is_full[c] = d <= radius;
    n_full += d <= radius ? 1 : 0;
  }
  // Every voxel one step from a full block's, outside the full blocks: the
  // apron, by global voxel coordinate.
  std::map<Coord, bool> apron;
  for (const auto& [c, full_block] : is_full) {
    if (!full_block) continue;
    for (int local = 0; local < kVpb; ++local) {
      const vgrid::Vec3i v = voxel_of(c, local);
      for (int dz = -1; dz <= 1; ++dz) {
        for (int dy = -1; dy <= 1; ++dy) {
          for (int dx = -1; dx <= 1; ++dx) {
            const vgrid::Vec3i n = v + vgrid::Vec3i(dx, dy, dz);
            const vgrid::Vec3i nb = vol::voxel_to_block(n, grid_params());
            const auto it = is_full.find({nb.x, nb.y, nb.z});
            if (it != is_full.end() && !it->second) {
              apron[{n.x, n.y, n.z}] = true;
            }
          }
        }
      }
    }
  }
  std::size_t n_apron_blocks = 0, n_unwritten = 0, n_apron_voxels = 0,
              n_apron_observed = 0;
  for (const auto& [c, want] : a) {
    const std::vector<float>& got = b.at(c);
    bool any_apron = false;
    for (int local = 0; local < kVpb; ++local) {
      const vgrid::Vec3i v = voxel_of(c, local);
      const bool written = is_full.at(c) || apron.count({v.x, v.y, v.z}) != 0;
      any_apron = any_apron || (!is_full.at(c) && written);
      const float t = written ? want[std::size_t(local)] : 0.0f;
      const float w = written ? want[std::size_t(kVpb + local)] : 0.0f;
      if (std::memcmp(&t, &got[std::size_t(local)], 4) != 0 ||
          std::memcmp(&w, &got[std::size_t(kVpb + local)], 4) != 0) {
        std::fprintf(stderr,
                     "FAIL full band: block (%d, %d, %d) voxel %d (%s): got "
                     "tsdf %.7g weight %g, want %.7g %g\n",
                     std::get<0>(c), std::get<1>(c), std::get<2>(c), local,
                     is_full.at(c) ? "full" : (written ? "apron" : "unwritten"),
                     double(got[std::size_t(local)]),
                     double(got[std::size_t(kVpb + local)]), double(t),
                     double(w));
        return 1;
      }
      if (!is_full.at(c) && written) {
        ++n_apron_voxels;
        n_apron_observed += w == 1.0f ? 1 : 0;
      }
    }
    if (!is_full.at(c)) ++(any_apron ? n_apron_blocks : n_unwritten);
  }
  const vx::MeshIntegrateStats& s = part.value().stats;
  CHECK(s.full_blocks == n_full);
  CHECK(s.blocks == n_full + n_apron_blocks);
  // The premises: every kind of block, and an apron that is mostly observed,
  // so a missing apron voxel would show.
  CHECK(n_full > 0 && n_apron_blocks > 0 && n_unwritten > 0);
  CHECK(n_apron_observed > n_apron_voxels / 2);
  std::printf(
      "full band oracle: %zu band blocks, %zu written whole, %zu for %zu apron "
      "voxels (%zu observed), %zu not written\n",
      a.size(), n_full, n_apron_blocks, n_apron_voxels, n_apron_observed,
      n_unwritten);
  return 0;
}

// Shell mode: every voxel whose field lies within the full band of zero, and
// each neighbour one step from it, holds the whole band's bytes; what else is
// written holds them too, or is unobserved.
int shell_case(const vkc::test::Gpu& gpu, vol::VoxelBlockGrid& grid,
               vx::MeshIntegrator& integ) {
  Mesh m = x_sheet(0.0125f, -0.06f, 0.07f, -0.05f, 0.03f);
  append(m, octahedron(vgrid::Vec3f(-0.0625f, 0.1225f, 0.0825f), 0.003f));
  const float full = codec::true_sdf_keep_distance(kVoxel);
  vx::MeshSdfParams params;
  params.mode = vx::MeshSdfMode::Shell;
  vkc::Result<Converted> whole = convert(gpu, grid, integ, m, params);
  params.full_band_m = full;
  vkc::Result<Converted> part = convert(gpu, grid, integ, m, params);
  CHECK(whole.ok() && part.ok());
  const Blocks& a = whole.value().blocks;
  const Blocks& b = part.value().blocks;
  CHECK(a.size() == b.size());
  // Every voxel's bytes by global voxel coordinate, whole band and full band.
  std::map<Coord, std::pair<const float*, const float*>> at;
  for (const auto& [c, vals] : a) {
    for (int local = 0; local < kVpb; ++local) {
      const vgrid::Vec3i v = voxel_of(c, local);
      at[{v.x, v.y, v.z}] = {&vals[std::size_t(local)],
                             &b.at(c)[std::size_t(local)]};
    }
  }
  const auto same = [](const std::pair<const float*, const float*>& p) {
    return std::memcmp(p.first, p.second, 4) == 0 &&
           std::memcmp(p.first + kVpb, p.second + kVpb, 4) == 0;
  };
  std::size_t near = 0, unwritten = 0;
  for (const auto& [v, p] : at) {
    const bool observed = p.first[kVpb] == 1.0f;
    const bool written = p.second[kVpb] != 0.0f || p.second[0] != 0.0f;
    unwritten += observed && !written ? 1 : 0;
    // Whatever the full band wrote is the whole band's.
    if (written && !same(p)) {
      std::fprintf(stderr, "FAIL shell: voxel (%d, %d, %d) differs\n",
                   std::get<0>(v), std::get<1>(v), std::get<2>(v));
      return 1;
    }
    if (!observed || !(std::fabs(*p.first) <= full)) continue;
    ++near;
    for (int dz = -1; dz <= 1; ++dz) {
      for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
          const auto it = at.find(
              {std::get<0>(v) + dx, std::get<1>(v) + dy, std::get<2>(v) + dz});
          if (it != at.end() && !same(it->second)) {
            std::fprintf(stderr,
                         "FAIL shell: voxel (%d, %d, %d), one step from one "
                         "%g m from zero, differs\n",
                         std::get<0>(it->first), std::get<1>(it->first),
                         std::get<2>(it->first), double(*p.first));
            return 1;
          }
        }
      }
    }
  }
  // The premises: voxels near zero, and observed voxels the full band left.
  CHECK(near > 0 && unwritten > 0);
  CHECK(part.value().stats.full_blocks > 0 &&
        part.value().stats.blocks < whole.value().stats.blocks);
  std::printf(
      "full band shell: %zu voxels within %g m of zero, %zu observed voxels "
      "left unwritten\n",
      near, double(full), unwritten);
  return 0;
}

// The curtain codes, as a true-SDF intra frame, the same bytes from the full
// band as from the whole band.
int curtain_case(const vkc::test::Gpu& gpu, vol::VoxelBlockGrid& grid,
                 vx::MeshIntegrator& integ) {
  const Mesh m = curtain();
  const float full = codec::true_sdf_keep_distance(kVoxel);
  codec::EncoderConfig config;
  config.true_sdf = true;
  std::vector<std::uint8_t> frames[2];
  vx::MeshIntegrateStats stats[2];
  Blocks grids[2];
  for (int run = 0; run < 2; ++run) {
    vkc::Result<Converted> c =
        convert(gpu, grid, integ, m, with_full_band(run == 0 ? 0.0f : full));
    CHECK(c.ok());
    stats[run] = c.value().stats;
    grids[run] = std::move(c.value().blocks);
    vkc::Result<codec::Encoder> enc =
        codec::Encoder::create(gpu.device, gpu.allocator, config);
    CHECK(enc.ok());
    vkc::Result<std::vector<std::uint8_t>> f = enc.value().encode(grid);
    CHECK(f.ok());
    frames[run] = std::move(f).value();
  }

  // The premises, on the whole band. K and N's facing voxels on their low
  // layers are observed, past the cut distance (the keep distance less one
  // voxel) and of opposite signs; N is outside the full band, so only its
  // apron is written there.
  const Coord k{0, 0, 0}, n{1, 0, 0};
  const float cut = full - kVoxel;
  const std::vector<float>& wk = grids[0].at(k);
  const std::vector<float>& wn = grids[0].at(n);
  int flips = 0;
  for (int y = 0; y < kBs; ++y) {
    const int lk = 7 + kBs * y;  // (7, y, 0)
    const int ln = 0 + kBs * y;  // (0, y, 0)
    CHECK(wk[std::size_t(kVpb + lk)] == 1.0f &&
          wn[std::size_t(kVpb + ln)] == 1.0f);
    const float tk = wk[std::size_t(lk)], tn = wn[std::size_t(ln)];
    flips +=
        (tk < 0.0f) != (tn < 0.0f) && std::fabs(tk) > cut && std::fabs(tn) > cut
            ? 1
            : 0;
  }
  CHECK(flips == kBs);
  // The frame codes K and not N.
  vkc::Result<vol::VoxelBlockGrid> dg = codec::make_decode_grid(
      gpu.device, gpu.allocator, frames[0].data(), frames[0].size(), 1u << 16);
  CHECK(dg.ok());
  vol::VoxelBlockGrid decoded = std::move(dg).value();
  vkc::Result<codec::Decoder> dec =
      codec::Decoder::create(gpu.device, gpu.allocator);
  CHECK(dec.ok());
  CHECK(codec::decode_growing(dec.value(), frames[0].data(), frames[0].size(),
                              decoded, 1u << 16)
            .ok());
  vkc::Result<Blocks> coded = by_coord(gpu, decoded);
  CHECK(coded.ok());
  CHECK(coded.value().count(k) == 1 && coded.value().count(n) == 0);

  if (frames[0] != frames[1]) {
    std::fprintf(stderr,
                 "FAIL curtain: the full band's frame (%zu B) differs from the "
                 "whole band's (%zu B)\n",
                 frames[1].size(), frames[0].size());
    return 1;
  }
  // And the full band left N to its apron, and other blocks unwritten.
  const std::vector<float>& pn = grids[1].at(n);
  const int inner = 4 + kBs * (4 + kBs * 4);  // (4, 4, 4)
  CHECK(wn[std::size_t(kVpb + inner)] == 1.0f &&
        pn[std::size_t(kVpb + inner)] == 0.0f);
  CHECK(pn[std::size_t(kVpb)] == wn[std::size_t(kVpb)] &&
        pn[0] == wn[0]);  // (0, 0, 0): the apron
  CHECK(stats[1].blocks < stats[0].blocks);
  std::printf(
      "full band curtain: %u of %u band blocks written (%u whole); frame %zu "
      "B, byte-identical\n",
      stats[1].blocks, stats[0].blocks, stats[1].full_blocks, frames[0].size());
  return 0;
}

// A negative or non-finite full band is refused, by validate and by convert
// before any GPU work.
int validate_case(const vkc::test::Gpu& gpu, vol::VoxelBlockGrid& grid,
                  vx::MeshIntegrator& integ) {
  const vol::VoxelGridParams gp = grid_params();
  const Mesh m = curtain();
  const std::uint64_t epoch = grid.topology_epoch();
  for (float bad : {-0.001f, std::numeric_limits<float>::quiet_NaN(),
                    std::numeric_limits<float>::infinity()}) {
    CHECK(!with_full_band(bad).validate(gp).ok());
    vkc::Result<vx::MeshIntegrateStats> refused =
        integ.convert(grid, m.v.data(), m.vertex_count(), m.i.data(),
                      m.triangle_count(), with_full_band(bad));
    CHECK(!refused.ok() &&
          refused.status().domain() == vkc::Status::Code::InvalidArgument);
  }
  CHECK(grid.topology_epoch() == epoch);
  CHECK(with_full_band(0.01f).validate(gp).ok());
  (void)gpu;
  return 0;
}

int gpu_main(vkc::test::GpuContext& gpu) {
  const vol::AttributeSpec attrs[] = {{vol::kTsdfAttribute, sizeof(float)},
                                      {vol::kWeightAttribute, sizeof(float)}};
  vkc::Result<vol::VoxelBlockGrid> grid_r = vol::VoxelBlockGrid::create(
      gpu.device, gpu.allocator, grid_params(), attrs, 2);
  CHECK(grid_r.ok());
  vol::VoxelBlockGrid grid = std::move(grid_r).value();

  // The device's own write layout first, with every case; then the oracle on
  // every other layout it takes.
  const vx::detail::WriteLayout own =
      vx::detail::default_write_layout(gpu.device);
  std::vector<vx::detail::WriteLayout> layouts{own};
  for (const bool wide : {true, false}) {
    for (const bool subgroup_cull : {true, false}) {
      const vx::detail::WriteLayout l{wide, subgroup_cull};
      if ((wide != own.wide || subgroup_cull != own.subgroup_cull) &&
          vx::detail::write_layout_fits(gpu.device, l)) {
        layouts.push_back(l);
      }
    }
  }
  for (std::size_t li = 0; li < layouts.size(); ++li) {
    vx::detail::set_test_write_layout(&layouts[li]);
    vkc::Result<vx::MeshIntegrator> integ_r =
        vx::MeshIntegrator::create(gpu.device, gpu.allocator);
    vx::detail::set_test_write_layout(nullptr);
    CHECK(integ_r.ok());
    vx::MeshIntegrator integ = std::move(integ_r).value();
    std::printf("-- write layout: %s, %s\n",
                layouts[li].wide ? "512 x 1" : "256 x 2",
                layouts[li].subgroup_cull ? "subgroup cull" : "shared cull");
    if (oracle_case(gpu, grid, integ) != 0) return 1;
    if (li != 0) continue;
    if (shell_case(gpu, grid, integ) != 0) return 1;
    if (curtain_case(gpu, grid, integ) != 0) return 1;
    if (validate_case(gpu, grid, integ) != 0) return 1;
  }
  std::printf("voxelize_mesh_full_band: OK\n");
  return 0;
}

}  // namespace

int main() {
  return vkc::test::run_on_gpu(vgrid::device_requirements(), gpu_main);
}
