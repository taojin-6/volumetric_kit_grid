// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// GPU test for triangle-mesh block allocation. The kernel's contract is a
// containment one -- every block holding a voxel within trunc_dist of the mesh
// is allocated -- so the expectation is computed the only way that actually
// tests it: brute force on the HOST over every voxel of every candidate block,
// against an independent C++ closest-point-on-triangle. A shader that dilated
// the wrong band, decoded a work item onto the wrong coordinate, or mismatched
// the host's prefix sum would diverge.
//
// Also covers what motivates the kernel existing at all (a triangle wider than
// the truncation band leaves a hole under allocate_from_points, and does not
// here), that the distance prune actually prunes (a far corner of a slanted
// triangle's bounding box stays unallocated), idempotent re-run, and the
// null / degenerate / out-of-range guards. Exits 0 (skip) where no device is
// present.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <set>
#include <tuple>
#include <vector>

#include "volumetric_kit/recon/core/allocator.hpp"
#include "volumetric_kit/recon/core/device.hpp"
#include "volumetric_kit/recon/core/instance.hpp"
#include "volumetric_kit/recon/core/math/vector_types.hpp"
#include "volumetric_kit/recon/core/vulkan.hpp"
#include "volumetric_kit/recon/volume/hash_types.hpp"
#include "volumetric_kit/recon/volume/voxel_coords.hpp"
#include "volumetric_kit/recon/volume/voxel_grid.hpp"
#include "volumetric_kit/recon/volume/voxel_hash_map.hpp"

namespace vr = volumetric_kit::recon;
namespace vol = volumetric_kit::recon::volume;

#define CHECK(cond)                                                        \
  do {                                                                     \
    if (!(cond)) {                                                         \
      std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
      return 1;                                                            \
    }                                                                      \
  } while (0)

namespace {

using Coord = std::tuple<int, int, int>;

// Closest point on a triangle, written out independently of the GLSL the kernel
// uses so the two are not the same expression checked against itself.
// Barycentric projection with a clamp to the triangle's edges, which is a
// different formulation from the shader's region test and agrees with it.
vr::Vec3f closest_point(vr::Vec3f p, vr::Vec3f a, vr::Vec3f b, vr::Vec3f c) {
  const vr::Vec3f ab = b - a;
  const vr::Vec3f ac = c - a;
  const vr::Vec3f n = vr::cross(ab, ac);
  const float nn = vr::dot(n, n);
  // Project onto the plane, then read off barycentrics as signed sub-areas.
  const vr::Vec3f q = p - n * (vr::dot(n, p - a) / nn);
  const float u = vr::dot(n, vr::cross(c - b, q - b)) / nn;
  const float v = vr::dot(n, vr::cross(a - c, q - c)) / nn;
  const float w = vr::dot(n, vr::cross(ab, q - a)) / nn;
  if (u >= 0.0f && v >= 0.0f && w >= 0.0f) {
    return q;  // inside
  }
  // Outside: the nearest point on the nearest edge segment.
  auto on_segment = [](vr::Vec3f pt, vr::Vec3f s0, vr::Vec3f s1) {
    const vr::Vec3f d = s1 - s0;
    const float dd = vr::dot(d, d);
    float t = (dd > 0.0f) ? vr::dot(pt - s0, d) / dd : 0.0f;
    t = t < 0.0f ? 0.0f : (t > 1.0f ? 1.0f : t);
    return s0 + d * t;
  };
  const vr::Vec3f e0 = on_segment(p, a, b);
  const vr::Vec3f e1 = on_segment(p, b, c);
  const vr::Vec3f e2 = on_segment(p, c, a);
  const float d0 = vr::dot(p - e0, p - e0);
  const float d1 = vr::dot(p - e1, p - e1);
  const float d2 = vr::dot(p - e2, p - e2);
  if (d0 <= d1 && d0 <= d2) return e0;
  return (d1 <= d2) ? e1 : e2;
}

float distance_to_triangle(vr::Vec3f p, vr::Vec3f a, vr::Vec3f b, vr::Vec3f c) {
  const vr::Vec3f q = closest_point(p, a, b, c);
  return vr::length(p - q);
}

// Every block holding at least one voxel within trunc_dist of some triangle --
// the set the kernel promises to cover. Brute force over each triangle's
// bounding box in blocks, and over all 512 voxels of each candidate.
std::set<Coord> blocks_within_band(const vol::VoxelGridParams& grid,
                                   const std::vector<vr::Vec3f>& verts,
                                   const std::vector<std::uint32_t>& idx) {
  std::set<Coord> want;
  for (std::size_t t = 0; t + 2 < idx.size(); t += 3) {
    const vr::Vec3f a = verts[idx[t]];
    const vr::Vec3f b = verts[idx[t + 1]];
    const vr::Vec3f c = verts[idx[t + 2]];
    const vr::Vec3f lo =
        glm::min(a, glm::min(b, c)) - vr::Vec3f(grid.trunc_dist);
    const vr::Vec3f hi =
        glm::max(a, glm::max(b, c)) + vr::Vec3f(grid.trunc_dist);
    const vr::Vec3i bmin = vol::world_to_block(lo, grid);
    const vr::Vec3i bmax = vol::world_to_block(hi, grid);
    for (int bz = bmin.z; bz <= bmax.z; ++bz) {
      for (int by = bmin.y; by <= bmax.y; ++by) {
        for (int bx = bmin.x; bx <= bmax.x; ++bx) {
          const vr::Vec3i base =
              vol::block_to_voxel(vr::Vec3i(bx, by, bz), grid);
          bool hit = false;
          for (int vz = 0; vz < grid.block_size && !hit; ++vz) {
            for (int vy = 0; vy < grid.block_size && !hit; ++vy) {
              for (int vx = 0; vx < grid.block_size && !hit; ++vx) {
                const vr::Vec3f p =
                    vol::voxel_to_world(base + vr::Vec3i(vx, vy, vz), grid);
                hit = distance_to_triangle(p, a, b, c) <= grid.trunc_dist;
              }
            }
          }
          if (hit) {
            want.insert({bx, by, bz});
          }
        }
      }
    }
  }
  return want;
}

int collect_active(vol::VoxelHashMap& map, std::set<Coord>& got) {
  vr::Result<std::vector<vol::BlockIndex>> active = map.compact_active_blocks();
  CHECK(active.ok());
  for (const vol::BlockIndex& block : active.value()) {
    CHECK(block.ptr >= 0);
    got.insert({block.coord.x, block.coord.y, block.coord.z});
  }
  return 0;
}

}  // namespace

int main() {
  vr::Result<vr::Instance> instance = vr::Instance::create({});
  if (!instance) {
    std::fprintf(stderr, "no Vulkan instance (%s); skipping\n",
                 instance.status().message().c_str());
    return 0;
  }
  vr::Result<VkPhysicalDevice> gpu = instance.value().select_physical_device();
  if (!gpu) {
    std::fprintf(stderr, "no compute-capable device (%s); skipping\n",
                 gpu.status().message().c_str());
    return 0;
  }
  vr::Result<vr::Device> device =
      vr::Device::create(instance.value(), gpu.value(), {});
  if (!device) {
    std::fprintf(stderr, "device create failed: %s\n",
                 device.status().message().c_str());
    return 1;
  }
  vr::Result<vr::Allocator> allocator =
      vr::Allocator::create(instance.value().handle(), device.value());
  if (!allocator) {
    std::fprintf(stderr, "allocator create failed: %s\n",
                 allocator.status().message().c_str());
    return 1;
  }

  // 5 mm voxels, 8-voxel (40 mm) blocks, a 40 mm band -- the production
  // defaults -- over a small 8192-block heap, which is ample for the meshes
  // below and cheap to init.
  vol::VoxelGridParams grid{};
  grid.voxel_size = 0.005f;
  grid.block_size = 8;
  grid.voxels_per_block = 512;
  grid.trunc_dist = 0.04f;
  grid.bucket_size = 8;
  grid.num_buckets = 1024;
  grid.num_blocks = 8192;
  grid.max_chain = 128;

  vr::Result<vol::VoxelHashMap> map_result =
      vol::VoxelHashMap::create(device.value(), allocator.value(), grid);
  CHECK(map_result.ok());
  vol::VoxelHashMap& map = map_result.value();

  // ---- 1. Coverage: every block with a voxel in the band is allocated ------
  // A tilted quad (two triangles) about 0.12 m across, deliberately not axis-
  // aligned and not on a block boundary, so the band is not a shape the block
  // grid could reproduce by accident.
  const std::vector<vr::Vec3f> quad_verts = {
      {0.013f, 0.007f, 0.011f},
      {0.131f, 0.019f, 0.023f},
      {0.127f, 0.113f, 0.077f},
      {0.009f, 0.101f, 0.065f},
  };
  const std::vector<std::uint32_t> quad_idx = {0, 1, 2, 0, 2, 3};

  vr::Result<std::uint32_t> failed = map.allocate_from_triangles(
      quad_verts.data(), static_cast<std::uint32_t>(quad_verts.size()),
      quad_idx.data(), static_cast<std::uint32_t>(quad_idx.size() / 3));
  CHECK(failed.ok());
  CHECK(failed.value() == 0);

  std::set<Coord> got;
  if (collect_active(map, got) != 0) return 1;
  const std::set<Coord> want = blocks_within_band(grid, quad_verts, quad_idx);
  CHECK(!want.empty());
  for (const Coord& c : want) {
    if (got.count(c) == 0) {
      std::fprintf(stderr, "FAIL: band block (%d,%d,%d) was not allocated\n",
                   std::get<0>(c), std::get<1>(c), std::get<2>(c));
      return 1;
    }
  }
  // The prune is conservative, so a few blocks outside the exact band are
  // expected -- but not the whole bounding box. The quad's box is ~6x5x4
  // blocks (~120); a kernel that skipped the distance test would allocate all
  // of it, so bound the slack well under that.
  CHECK(got.size() <= want.size() * 2);

  // ---- 2. Idempotent re-run ------------------------------------------------
  vr::Result<std::uint32_t> again = map.allocate_from_triangles(
      quad_verts.data(), static_cast<std::uint32_t>(quad_verts.size()),
      quad_idx.data(), static_cast<std::uint32_t>(quad_idx.size() / 3));
  CHECK(again.ok());
  CHECK(again.value() == 0);
  std::set<Coord> got_again;
  if (collect_active(map, got_again) != 0) return 1;
  CHECK(got_again == got);

  // ---- 3. A triangle wider than the band has no hole ----------------------
  // The motivating case. One 0.6 m triangle spans ~15 blocks a side, far more
  // than the (2*tb+1)^3 = 3-block cube a vertex dilates into, so a block near
  // its centroid is reachable only by a triangle-aware pass.
  CHECK(map.clear().ok());
  const std::vector<vr::Vec3f> big_verts = {
      {0.0f, 0.0f, 0.0f}, {0.6f, 0.0f, 0.0f}, {0.0f, 0.6f, 0.0f}};
  const std::vector<std::uint32_t> big_idx = {0, 1, 2};
  const vr::Vec3f centroid =
      (big_verts[0] + big_verts[1] + big_verts[2]) / 3.0f;
  const vr::Vec3i centre_block = vol::world_to_block(centroid, grid);
  const Coord centre_coord = {centre_block.x, centre_block.y, centre_block.z};

  vr::Result<std::uint32_t> big =
      map.allocate_from_triangles(big_verts.data(), 3, big_idx.data(), 1);
  CHECK(big.ok());
  CHECK(big.value() == 0);
  std::set<Coord> big_got;
  if (collect_active(map, big_got) != 0) return 1;
  CHECK(big_got.count(centre_coord) == 1);

  // The contrast that makes the assertion above fail if this kernel is deleted
  // and callers fall back to the point path: the same three vertices, dilated
  // as points, leave the centroid's block unallocated.
  CHECK(map.clear().ok());
  vr::Result<std::uint32_t> pts = map.allocate_from_points(big_verts.data(), 3);
  CHECK(pts.ok());
  std::set<Coord> pts_got;
  if (collect_active(map, pts_got) != 0) return 1;
  CHECK(pts_got.count(centre_coord) == 0);

  // ---- 4. The distance prune prunes ---------------------------------------
  // A triangle in the x=y diagonal plane: its bounding box corners are ~0.42 m
  // from the surface, so a kernel that allocated the box rather than the band
  // would allocate them.
  CHECK(map.clear().ok());
  const std::vector<vr::Vec3f> diag_verts = {
      {0.0f, 0.0f, 0.0f}, {0.6f, 0.6f, 0.0f}, {0.0f, 0.0f, 0.3f}};
  const std::vector<std::uint32_t> diag_idx = {0, 1, 2};
  vr::Result<std::uint32_t> diag =
      map.allocate_from_triangles(diag_verts.data(), 3, diag_idx.data(), 1);
  CHECK(diag.ok());
  CHECK(diag.value() == 0);
  std::set<Coord> diag_got;
  if (collect_active(map, diag_got) != 0) return 1;
  // A corner of the box, maximally far from the x=y plane the triangle lies in.
  const vr::Vec3i far_block =
      vol::world_to_block(vr::Vec3f(0.58f, 0.02f, 0.15f), grid);
  CHECK(diag_got.count({far_block.x, far_block.y, far_block.z}) == 0);
  // ...while the surface itself is still covered.
  const vr::Vec3i on_block =
      vol::world_to_block(vr::Vec3f(0.30f, 0.30f, 0.10f), grid);
  CHECK(diag_got.count({on_block.x, on_block.y, on_block.z}) == 1);

  // ---- 5. Guards -----------------------------------------------------------
  CHECK(map.clear().ok());
  // Zero triangles: nothing to do, not an error.
  vr::Result<std::uint32_t> none =
      map.allocate_from_triangles(quad_verts.data(), 4, quad_idx.data(), 0);
  CHECK(none.ok());
  CHECK(none.value() == 0);

  CHECK(!map.allocate_from_triangles(nullptr, 4, quad_idx.data(), 2).ok());
  CHECK(!map.allocate_from_triangles(quad_verts.data(), 4, nullptr, 2).ok());
  // An index at vertex_count is out of range -- the kernel would read past the
  // vertex buffer, so the host must refuse rather than dispatch.
  const std::vector<std::uint32_t> bad_idx = {0, 1, 4};
  CHECK(!map.allocate_from_triangles(quad_verts.data(), 4, bad_idx.data(), 1)
             .ok());

  // A degenerate (zero-area) triangle and a non-finite one are skipped, not
  // refused: a mesh file routinely carries a few, and one bad face should not
  // fail the whole allocation.
  const float inf = std::numeric_limits<float>::infinity();
  const std::vector<vr::Vec3f> odd_verts = {
      {0.1f, 0.1f, 0.1f},  // 0: degenerate triangle uses 0,0,1
      {0.2f, 0.1f, 0.1f},  // 1
      {inf, 0.0f, 0.0f},   // 2: non-finite
  };
  const std::vector<std::uint32_t> odd_idx = {0, 0, 1, 0, 1, 2};
  vr::Result<std::uint32_t> odd =
      map.allocate_from_triangles(odd_verts.data(), 3, odd_idx.data(), 2);
  CHECK(odd.ok());
  CHECK(odd.value() == 0);
  std::set<Coord> odd_got;
  if (collect_active(map, odd_got) != 0) return 1;
  CHECK(odd_got.empty());

  std::printf("volume_allocate_triangles: OK\n");
  return 0;
}
