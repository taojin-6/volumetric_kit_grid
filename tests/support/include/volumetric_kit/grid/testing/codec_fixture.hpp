// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

// Shared by the codec's Encoder / Decoder tests: a device, grids of a chosen
// geometry, and an analytic sphere written straight into a grid -- a surface
// whose true position is known, so a decoded mesh can be judged against the
// truth rather than only against the mesh of the grid it was encoded from.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

#include "volumetric_kit/recon/core/allocator.hpp"
#include "volumetric_kit/recon/core/device.hpp"
#include "volumetric_kit/recon/core/instance.hpp"
#include "volumetric_kit/recon/core/result.hpp"
#include "volumetric_kit/recon/volume/hash_types.hpp"
#include "volumetric_kit/recon/volume/voxel_block_grid.hpp"
#include "volumetric_kit/recon/volume/voxel_grid.hpp"

#define CHECK(cond)                                                        \
  do {                                                                     \
    if (!(cond)) {                                                         \
      std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
      return 1;                                                            \
    }                                                                      \
  } while (0)

namespace codec_fixture {

namespace vr = volumetric_kit::recon;
namespace vol = volumetric_kit::recon::volume;

constexpr float kVoxel = 0.005f;
constexpr float kTrunc = 0.04f;
constexpr std::uint32_t kVpb = 512;

// The device and allocator every case runs on. Held by reference, never moved:
// the allocator and every tier keep references to the device, so a test's
// main keeps all three where it created them.
struct Gpu {
  vr::Device& device;
  vr::Allocator& allocator;
};

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

inline vr::Result<vol::VoxelBlockGrid> make_grid(Gpu& gpu,
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
  std::vector<vol::AttributeSpec> attrs = {{"tsdf", sizeof(float)}};
  if (s.weight) attrs.push_back({"weight", sizeof(float)});
  if (s.color) attrs.push_back({"color", 3});
  return vol::VoxelBlockGrid::create(gpu.device, gpu.allocator, gp,
                                     attrs.data(), attrs.size());
}

inline bool coord_less(const vr::Vec3i& a, const vr::Vec3i& b) {
  if (a.z != b.z) return a.z < b.z;
  if (a.y != b.y) return a.y < b.y;
  return a.x < b.x;
}

// The grid's active blocks, in the codec's (z, y, x) order.
inline vr::Result<std::vector<vol::BlockIndex>> active_sorted(
    vol::VoxelBlockGrid& grid) {
  VR_ASSIGN(std::vector<vol::BlockIndex> active,
            grid.map().compact_active_blocks());
  std::sort(active.begin(), active.end(),
            [](const vol::BlockIndex& a, const vol::BlockIndex& b) {
              return coord_less(a.coord, b.coord);
            });
  return active;
}

inline vr::Status allocate(vol::VoxelBlockGrid& grid,
                           const std::vector<vr::Vec3i>& coords) {
  std::vector<vol::BlockIndex> blocks(coords.size());
  for (std::size_t i = 0; i < coords.size(); ++i) {
    blocks[i].coord = coords[i];
  }
  VR_ASSIGN(const std::uint32_t failed,
            grid.map().allocate(blocks.data(),
                                static_cast<std::uint32_t>(blocks.size())));
  if (failed != 0) {
    return vr::Status::out_of_memory("fixture: allocation failed");
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

// Write the sphere into every active block: a voxel within the band is
// observed (weight 2) with its true SDF clamped to +-trunc; one outside it is
// unobserved (weight 0, tsdf 0) -- the shape a fused grid has, partially
// observed blocks included.
inline vr::Status write_sphere(vol::VoxelBlockGrid& grid, const Sphere& s) {
  VR_ASSIGN(const std::vector<vol::BlockIndex> active, active_sorted(grid));
  VR_ASSIGN(const vol::AttributeView tsdf_view, grid.attribute("tsdf"));
  VR_ASSIGN(const vol::AttributeView weight_view, grid.attribute("weight"));
  float* tsdf = static_cast<float*>(tsdf_view.buffer->mapped());
  float* weight = static_cast<float*>(weight_view.buffer->mapped());
  const float voxel = grid.grid().voxel_size;
  const float trunc = grid.grid().trunc_dist;
  for (const vol::BlockIndex& b : active) {
    for (std::uint32_t v = 0; v < kVpb; ++v) {
      const vr::Vec3f p(float(b.coord.x * 8 + int(v % 8)) * voxel,
                        float(b.coord.y * 8 + int((v / 8) % 8)) * voxel,
                        float(b.coord.z * 8 + int(v / 64)) * voxel);
      const float d = s.sdf(p);
      const bool obs = std::fabs(d) <= trunc;
      tsdf[std::uint32_t(b.ptr) + v] = obs ? d : 0.0f;
      weight[std::uint32_t(b.ptr) + v] = obs ? 2.0f : 0.0f;
    }
  }
  return {};
}

// A grid holding the sphere: its band blocks allocated and written, plus
// `extra` blocks far from it that are allocated and never observed.
inline vr::Result<vol::VoxelBlockGrid> sphere_grid(
    Gpu& gpu, const Sphere& s, const GridShape& shape = {},
    const std::vector<vr::Vec3i>& extra = {}) {
  VR_ASSIGN(vol::VoxelBlockGrid grid, make_grid(gpu, shape));
  std::vector<vr::Vec3i> coords =
      band_blocks(s, shape.voxel_size, shape.trunc_dist);
  coords.insert(coords.end(), extra.begin(), extra.end());
  VR_TRY(allocate(grid, coords));
  VR_TRY(write_sphere(grid, s));
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

inline vr::Result<Snapshot> snapshot(vol::VoxelBlockGrid& grid) {
  VR_ASSIGN(const std::vector<vol::BlockIndex> active, active_sorted(grid));
  VR_ASSIGN(const vol::AttributeView tsdf_view, grid.attribute("tsdf"));
  VR_ASSIGN(const vol::AttributeView weight_view, grid.attribute("weight"));
  const float* tsdf = static_cast<const float*>(tsdf_view.buffer->mapped());
  const float* weight = static_cast<const float*>(weight_view.buffer->mapped());
  Snapshot out;
  for (const vol::BlockIndex& b : active) {
    out.coords.push_back(b.coord);
    out.tsdf.insert(out.tsdf.end(), tsdf + b.ptr, tsdf + b.ptr + kVpb);
    out.weight.insert(out.weight.end(), weight + b.ptr, weight + b.ptr + kVpb);
  }
  return out;
}

}  // namespace codec_fixture
