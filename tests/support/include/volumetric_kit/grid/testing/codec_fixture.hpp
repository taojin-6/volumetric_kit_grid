// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

// Shared by the codec's Encoder / Decoder tests: a device, grids of a chosen
// geometry, and an analytic sphere written straight into a grid -- a surface
// whose true position is known, so a decoded mesh can be judged against the
// truth rather than only against the mesh of the grid it was encoded from.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <vector>

#include "device_blocks.hpp"
#include "grid_readback.hpp"
#include "volumetric_kit/core/base/result.hpp"
#include "volumetric_kit/core/testing/test_check.hpp"
#include "volumetric_kit/core/vulkan/allocator.hpp"
#include "volumetric_kit/core/vulkan/device.hpp"
#include "volumetric_kit/core/vulkan/instance.hpp"
#include "volumetric_kit/recon/volume/hash_types.hpp"
#include "volumetric_kit/recon/volume/voxel_block_grid.hpp"
#include "volumetric_kit/recon/volume/voxel_grid.hpp"

namespace codec_fixture {

namespace vr = volumetric_kit::recon;
namespace vkc = volumetric_kit::core;
namespace vol = volumetric_kit::recon::volume;

constexpr float kVoxel = 0.005f;
constexpr float kTrunc = 0.04f;
constexpr std::uint32_t kVpb = 512;

// The device and allocator every case runs on, never moved: the allocator
// and every tier keep references to the device.
using Gpu = vkc::test::Gpu;

struct GridShape {
  float voxel_size = kVoxel;
  float trunc_dist = kTrunc;
  std::int32_t block_size = 8;
  std::int32_t bucket_size = 8;
  std::int32_t num_buckets = 512;
  std::int32_t max_chain = 128;
  bool weight = true;
  bool color = false;  // a third attribute, which a frame does not carry
};

inline vkc::Result<vol::VoxelBlockGrid> make_grid(Gpu& gpu,
                                                  const GridShape& s = {}) {
  vol::VoxelGridParams gp{};
  gp.voxel_size = s.voxel_size;
  gp.block_size = s.block_size;
  gp.voxels_per_block = s.block_size * s.block_size * s.block_size;
  gp.trunc_dist = s.trunc_dist;
  gp.bucket_size = s.bucket_size;
  gp.num_buckets = s.num_buckets;
  gp.num_blocks = s.bucket_size * s.num_buckets;
  gp.max_chain = s.max_chain;
  std::vector<vol::AttributeSpec> attrs = {
      {vol::kTsdfAttribute, sizeof(float)}};
  if (s.weight) attrs.push_back({vol::kWeightAttribute, sizeof(float)});
  if (s.color) attrs.push_back({vol::kColorAttribute, 3});
  return vol::VoxelBlockGrid::create(gpu.device, gpu.allocator, gp,
                                     attrs.data(), attrs.size());
}

using vol::coord_less;

// The grid's active blocks, in the codec's (z, y, x) order.
inline vkc::Result<std::vector<vol::BlockIndex>> active_sorted(
    vol::VoxelBlockGrid& grid) {
  VKC_ASSIGN(std::vector<vol::BlockIndex> active,
             grid.map().compact_active_blocks());
  std::sort(active.begin(), active.end(),
            [](const vol::BlockIndex& a, const vol::BlockIndex& b) {
              return coord_less(a.coord, b.coord);
            });
  return active;
}

inline vkc::Status allocate(Gpu& gpu, vol::VoxelBlockGrid& grid,
                            const std::vector<vr::Vec3i>& coords) {
  std::vector<vol::BlockIndex> blocks(coords.size());
  for (std::size_t i = 0; i < coords.size(); ++i) {
    blocks[i].coord = coords[i];
  }
  VKC_ASSIGN(const std::uint32_t failed,
             vr_test::allocate_blocks(gpu, grid.map(), blocks));
  if (failed != 0) {
    return vkc::Status::out_of_memory("fixture: allocation failed");
  }
  return {};
}

// A sphere: centre and radius in metres.
struct Sphere {
  vr::Vec3f centre;
  float radius;
  // Signed distance of a world point to the surface.
  float sdf(const vr::Vec3f& p) const {
    const vr::Vec3f d = p - centre;
    return std::sqrt(d.x * d.x + d.y * d.y + d.z * d.z) - radius;
  }
};

// Every block whose voxels come within the truncation band of the sphere.
inline std::vector<vr::Vec3i> band_blocks(const Sphere& s, float voxel = kVoxel,
                                          float trunc = kTrunc) {
  const float block = 8 * voxel;
  const float reach = s.radius + trunc + 2 * block;
  std::vector<vr::Vec3i> out;
  const int lo_x = int(std::floor((s.centre.x - reach) / block));
  const int hi_x = int(std::ceil((s.centre.x + reach) / block));
  const int lo_y = int(std::floor((s.centre.y - reach) / block));
  const int hi_y = int(std::ceil((s.centre.y + reach) / block));
  const int lo_z = int(std::floor((s.centre.z - reach) / block));
  const int hi_z = int(std::ceil((s.centre.z + reach) / block));
  for (int z = lo_z; z <= hi_z; ++z) {
    for (int y = lo_y; y <= hi_y; ++y) {
      for (int x = lo_x; x <= hi_x; ++x) {
        bool near = false;
        for (int v = 0; v < int(kVpb) && !near; ++v) {
          const vr::Vec3f p(float(x * 8 + v % 8) * voxel,
                            float(y * 8 + (v / 8) % 8) * voxel,
                            float(z * 8 + v / 64) * voxel);
          near = std::fabs(s.sdf(p)) <= trunc;
        }
        if (near) {
          out.push_back(vr::Vec3i(x, y, z));
        }
      }
    }
  }
  return out;
}

// Write the field @p sdf (signed distance of a world point, metres) into
// every active block: a voxel within the band is observed (weight 2) with its
// SDF; one outside it is unobserved (weight 0, tsdf 0) -- the shape a fused
// grid has, partially observed blocks included. With @p observe_all, every
// voxel is observed, its SDF clamped to +-trunc, so every mask is all ones.
template <typename Sdf>
vkc::Status write_field(Gpu& gpu, vol::VoxelBlockGrid& grid, const Sdf& sdf,
                        bool observe_all = false) {
  VKC_ASSIGN(const std::vector<vol::BlockIndex> active, active_sorted(grid));
  VKC_ASSIGN(vr_test::AttributeSpan<float> tsdf,
             vr_test::read_span<float>(gpu.device, gpu.allocator, grid,
                                       vol::kTsdfAttribute, active));
  VKC_ASSIGN(vr_test::AttributeSpan<float> weight,
             vr_test::read_span<float>(gpu.device, gpu.allocator, grid,
                                       vol::kWeightAttribute, active));
  const float voxel = grid.grid().voxel_size;
  const float trunc = grid.grid().trunc_dist;
  for (const vol::BlockIndex& b : active) {
    const std::size_t at = std::size_t(b.ptr) - tsdf.first;
    for (std::uint32_t v = 0; v < kVpb; ++v) {
      const vr::Vec3f p(float(b.coord.x * 8 + int(v % 8)) * voxel,
                        float(b.coord.y * 8 + int((v / 8) % 8)) * voxel,
                        float(b.coord.z * 8 + int(v / 64)) * voxel);
      const float d = sdf(p);
      const bool obs = observe_all || std::fabs(d) <= trunc;
      tsdf.values[at + v] = obs ? std::clamp(d, -trunc, trunc) : 0.0f;
      weight.values[at + v] = obs ? 2.0f : 0.0f;
    }
  }
  VKC_TRY(vr_test::write_span(gpu.device, gpu.allocator, grid,
                              vol::kTsdfAttribute, tsdf));
  return vr_test::write_span(gpu.device, gpu.allocator, grid,
                             vol::kWeightAttribute, weight);
}

// write_field of one sphere.
inline vkc::Status write_sphere(Gpu& gpu, vol::VoxelBlockGrid& grid,
                                const Sphere& s) {
  return write_field(gpu, grid, [&s](const vr::Vec3f& p) { return s.sdf(p); });
}

// The band blocks of every sphere in @p spheres, each coordinate once.
inline std::vector<vr::Vec3i> union_blocks(const std::vector<Sphere>& spheres,
                                           const GridShape& shape = {}) {
  std::vector<vr::Vec3i> coords;
  for (const Sphere& s : spheres) {
    for (const vr::Vec3i& c :
         band_blocks(s, shape.voxel_size, shape.trunc_dist)) {
      if (std::find(coords.begin(), coords.end(), c) == coords.end()) {
        coords.push_back(c);
      }
    }
  }
  return coords;
}

// A grid holding the union of two spheres, @p a and @p b, each's band blocks
// allocated and the nearer surface's distance written: one sphere that
// moves beside one that stays, as a chain of frames.
inline vkc::Result<vol::VoxelBlockGrid> two_sphere_grid(
    Gpu& gpu, const Sphere& a, const Sphere& b, const GridShape& shape = {}) {
  VKC_ASSIGN(vol::VoxelBlockGrid grid, make_grid(gpu, shape));
  VKC_TRY(allocate(gpu, grid, union_blocks({a, b}, shape)));
  VKC_TRY(write_field(gpu, grid, [&a, &b](const vr::Vec3f& p) {
    return std::min(a.sdf(p), b.sdf(p));
  }));
  return grid;
}

// A sphere that moves and grows beside one that stays, at step @p t.
inline vkc::Result<vol::VoxelBlockGrid> chain_grid(Gpu& gpu, int t) {
  const Sphere moving{
      vr::Vec3f(0.004f * float(t), -0.002f * float(t), 0.003f * float(t)),
      0.07f + 0.002f * float(t)};
  return two_sphere_grid(gpu, moving,
                         Sphere{vr::Vec3f(-0.5f, 0.1f, 0.0f), 0.05f});
}

// --- True signed distances (EncoderConfig::true_sdf). ----------------------

// A band of 16 voxels, past kTrueSdfBandVoxels, as the mesh templates code.
inline GridShape wide_band() {
  GridShape s;
  s.trunc_dist = 16 * kVoxel;
  return s;
}

// A speck: a sphere of 0.3 voxels centred in the cell at the shared corner of
// blocks (b - 1, b - 1, b - 1) to (b, b, b). No voxel is inside it, so no
// cell's sign changes, but that cell's 8 corners lie 0.57 voxels out: only
// the near rule keeps those 8 blocks.
inline Sphere speck(int b) {
  const float c = (8.0f * float(b) - 0.5f) * kVoxel;
  return Sphere{vr::Vec3f(c, c, c), 0.3f * kVoxel};
}

// A sphere's distance with a curtain: negated in a slab more than 20 mm out
// (x past the centre, y within 17.5 mm of the centre's plus slab_y), the
// sheet a mesh's closest-face sign leaves past a hole's rim. Every sign flip
// of the sheet lies 20 mm or more from zero, past the cut distance; the
// slab's faces lie inside blocks, so cells cut across block faces. With
// with_speck, a speck far from both.
struct Curtain {
  Sphere sphere;
  bool with_speck = true;
  float slab_y = 0.0f;
  float sdf(const vr::Vec3f& p) const {
    const float d = sphere.sdf(p);
    const vr::Vec3f q = p - sphere.centre;
    const float field =
        d > 0.02f && q.x > 0.0f && std::fabs(q.y - slab_y) < 0.0175f ? -d : d;
    return with_speck ? std::min(field, speck(10).sdf(p)) : field;
  }
  std::vector<vr::Vec3i> blocks(const GridShape& shape) const {
    return with_speck ? union_blocks({sphere, speck(10)}, shape)
                      : union_blocks({sphere}, shape);
  }
};

// The curtain's blocks allocated in @p order (reversed when false), in a
// grid of @p shape, and its field written.
inline vkc::Result<vol::VoxelBlockGrid> curtain_grid(
    Gpu& gpu, const Curtain& c, const GridShape& shape = wide_band(),
    bool in_order = true) {
  VKC_ASSIGN(vol::VoxelBlockGrid grid, make_grid(gpu, shape));
  std::vector<vr::Vec3i> coords = c.blocks(shape);
  if (!in_order) std::reverse(coords.begin(), coords.end());
  VKC_TRY(allocate(gpu, grid, coords));
  VKC_TRY(
      write_field(gpu, grid, [&c](const vr::Vec3f& p) { return c.sdf(p); }));
  return grid;
}

// A grid holding the sphere: its band blocks allocated and written, plus
// `extra` blocks far from it that are allocated and never observed.
inline vkc::Result<vol::VoxelBlockGrid> sphere_grid(
    Gpu& gpu, const Sphere& s, const GridShape& shape = {},
    const std::vector<vr::Vec3i>& extra = {}) {
  VKC_ASSIGN(vol::VoxelBlockGrid grid, make_grid(gpu, shape));
  std::vector<vr::Vec3i> coords =
      band_blocks(s, shape.voxel_size, shape.trunc_dist);
  coords.insert(coords.end(), extra.begin(), extra.end());
  VKC_TRY(allocate(gpu, grid, coords));
  VKC_TRY(write_sphere(gpu, grid, s));
  return grid;
}

// Per active block, its tsdf and weight -- a grid's full observable state,
// for "nothing changed" checks.
struct Snapshot {
  std::vector<vr::Vec3i> coords;
  std::vector<float> tsdf;
  std::vector<float> weight;
  bool operator==(const Snapshot& o) const {
    return coords == o.coords && tsdf == o.tsdf && weight == o.weight;
  }
};

inline vkc::Result<Snapshot> snapshot(Gpu& gpu, vol::VoxelBlockGrid& grid) {
  VKC_ASSIGN(const std::vector<vol::BlockIndex> active, active_sorted(grid));
  VKC_ASSIGN(const vr_test::AttributeSpan<float> tsdf,
             vr_test::read_span<float>(gpu.device, gpu.allocator, grid,
                                       vol::kTsdfAttribute, active));
  VKC_ASSIGN(const vr_test::AttributeSpan<float> weight,
             vr_test::read_span<float>(gpu.device, gpu.allocator, grid,
                                       vol::kWeightAttribute, active));
  Snapshot out;
  for (const vol::BlockIndex& b : active) {
    const auto at = [&](const vr_test::AttributeSpan<float>& a) {
      return a.values.begin() + std::ptrdiff_t(std::size_t(b.ptr) - a.first);
    };
    out.coords.push_back(b.coord);
    out.tsdf.insert(out.tsdf.end(), at(tsdf), at(tsdf) + kVpb);
    out.weight.insert(out.weight.end(), at(weight), at(weight) + kVpb);
  }
  return out;
}

}  // namespace codec_fixture
