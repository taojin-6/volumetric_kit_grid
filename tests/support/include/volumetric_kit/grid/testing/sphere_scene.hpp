// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

// Cameras round a sphere at the world origin and what each sees.

#include <cmath>
#include <cstddef>
#include <cstdint>
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

}  // namespace vr_test
