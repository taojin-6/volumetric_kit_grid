// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

// Cameras round a sphere at the world origin, what each sees, and a check
// that two grids fused from them hold the same blocks with the same bits.

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <map>
#include <tuple>
#include <vector>

#include "grid_readback.hpp"
#include "volumetric_kit/core/base/result.hpp"
#include "volumetric_kit/recon/core/camera_params.hpp"
#include "volumetric_kit/recon/core/math/vector_types.hpp"
#include "volumetric_kit/recon/volume/hash_types.hpp"
#include "volumetric_kit/recon/volume/voxel_block_grid.hpp"

namespace vr_test {

inline constexpr std::uint32_t kSphereWidth = 160;
inline constexpr std::uint32_t kSphereHeight = 120;
inline constexpr float kSphereRadius = 0.25f;

// One camera on a circle round the sphere, looking at it, and what it sees:
// depth in metres (0 where the ray misses) and a colour per pixel.
struct SphereView {
  volumetric_kit::recon::DepthCameraParams cam{};
  std::vector<float> depth;
  std::vector<std::uint32_t> color;
};

inline SphereView sphere_view(int index) {
  namespace vr = volumetric_kit::recon;
  const float angle = 2.1f * static_cast<float>(index);
  const vr::Vec3f eye(0.9f * std::sin(angle), 0.1f * static_cast<float>(index),
                      -0.9f * std::cos(angle));
  // OpenCV axes: z forward, y down, x = y cross z.
  const vr::Vec3f z = glm::normalize(-eye);
  const vr::Vec3f x = glm::normalize(glm::cross(vr::Vec3f(0, 1, 0), z));
  const vr::Vec3f y = glm::cross(z, x);
  SphereView v;
  v.cam.fx = 140.0f;
  v.cam.fy = 140.0f;
  v.cam.cx = 79.5f;
  v.cam.cy = 59.5f;
  v.cam.min_depth = 0.1f;
  v.cam.max_depth = 5.0f;
  v.cam.width = kSphereWidth;
  v.cam.height = kSphereHeight;
  v.cam.cam_to_world = vr::Mat4f(vr::Vec4f(x, 0), vr::Vec4f(y, 0),
                                 vr::Vec4f(z, 0), vr::Vec4f(eye, 1));
  v.depth.assign(kSphereWidth * kSphereHeight, 0.0f);
  v.color.assign(kSphereWidth * kSphereHeight, 0u);
  for (std::uint32_t row = 0; row < kSphereHeight; ++row) {
    for (std::uint32_t col = 0; col < kSphereWidth; ++col) {
      const vr::Vec3f ray = glm::normalize(
          (static_cast<float>(col) + 0.5f - v.cam.cx) / v.cam.fx * x +
          (static_cast<float>(row) + 0.5f - v.cam.cy) / v.cam.fy * y + z);
      // |eye + t ray| = r, the nearer root.
      const float b = glm::dot(eye, ray);
      const float disc =
          b * b - glm::dot(eye, eye) + kSphereRadius * kSphereRadius;
      const std::size_t i = row * kSphereWidth + col;
      v.color[i] = (col * 255 / kSphereWidth) |
                   ((row * 255 / kSphereHeight) << 8) |
                   (static_cast<std::uint32_t>(60 * index) << 16) | 0xFF000000u;
      if (disc <= 0.0f) continue;  // misses: no return
      const float t = -b - std::sqrt(disc);
      v.depth[i] = t * glm::dot(ray, z);
    }
  }
  return v;
}

using BlockCoord = std::tuple<int, int, int>;

// Every active block's coordinate and first voxel (`ptr` is a voxel offset).
inline vkc::Result<std::map<BlockCoord, std::int32_t>> blocks_of(
    volumetric_kit::recon::volume::VoxelBlockGrid& grid) {
  VKC_ASSIGN(std::vector<volumetric_kit::recon::volume::BlockIndex> active,
             grid.map().compact_active_blocks());
  std::map<BlockCoord, std::int32_t> out;
  for (const auto& b : active) {
    out[BlockCoord{b.coord.x, b.coord.y, b.coord.z}] = b.ptr;
  }
  return out;
}

// Whether `a` and `b` hold the same blocks, and in each the same bits of
// weight, tsdf and colour in every voxel; a block's slot may differ, since
// allocation order is the GPU's. Prints the first difference. `observed`, if
// given, receives the voxels with weight.
inline bool same_grids(const Gpu& gpu,
                       volumetric_kit::recon::volume::VoxelBlockGrid& a,
                       volumetric_kit::recon::volume::VoxelBlockGrid& b,
                       std::size_t* observed = nullptr) {
  auto ba = blocks_of(a);
  auto bb = blocks_of(b);
  if (!ba.ok() || !bb.ok()) {
    std::fprintf(stderr, "same_grids: compaction failed\n");
    return false;
  }
  if (ba.value().size() != bb.value().size()) {
    std::fprintf(stderr, "same_grids: %zu blocks against %zu\n",
                 ba.value().size(), bb.value().size());
    return false;
  }
  using Words = std::vector<std::uint32_t>;
  const char* names[] = {"weight", "tsdf", "color"};
  Words wa[3], wb[3];
  for (int n = 0; n < 3; ++n) {
    auto ra =
        read_attribute<std::uint32_t>(gpu.device, gpu.allocator, a, names[n]);
    auto rb =
        read_attribute<std::uint32_t>(gpu.device, gpu.allocator, b, names[n]);
    if (!ra.ok() || !rb.ok()) {
      std::fprintf(stderr, "same_grids: no %s attribute\n", names[n]);
      return false;
    }
    wa[n] = std::move(ra).value();
    wb[n] = std::move(rb).value();
  }
  const auto voxels = std::size_t(a.grid().voxels_per_block);
  std::size_t weighted = 0;
  for (const auto& [coord, ptr] : ba.value()) {
    const auto other = bb.value().find(coord);
    if (other == bb.value().end()) {
      std::fprintf(stderr, "same_grids: block (%d, %d, %d) only in one\n",
                   std::get<0>(coord), std::get<1>(coord), std::get<2>(coord));
      return false;
    }
    for (std::size_t k = 0; k < voxels; ++k) {
      const std::size_t ia = std::size_t(ptr) + k;
      const std::size_t ib = std::size_t(other->second) + k;
      for (int n = 0; n < 3; ++n) {
        if (wa[n][ia] != wb[n][ib]) {
          std::fprintf(stderr, "same_grids: %s differs in (%d, %d, %d)\n",
                       names[n], std::get<0>(coord), std::get<1>(coord),
                       std::get<2>(coord));
          return false;
        }
      }
      weighted += wa[0][ia] != 0 ? 1 : 0;
    }
  }
  if (observed != nullptr) *observed = weighted;
  return true;
}

}  // namespace vr_test
