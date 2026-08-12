// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file volume/frustum.hpp
/// @brief Camera view-frustum planes for culling the active voxel-block set.

#include <array>
#include <cstddef>
#include <cstdint>

#include <glm/gtc/matrix_inverse.hpp>

#include "volumetric_kit/recon/core/math/vector_types.hpp"

namespace volumetric_kit::recon::volume {

/// @brief The six planes of a camera view frustum, in world space.
///
/// Each plane is `(nx, ny, nz, d)` in **Hessian normal form** with an
/// inward-pointing unit normal: a world point `p` is inside when
/// `dot(n, p) + d >= 0` for all six. Order is left, right, top, bottom, near,
/// far. This is exactly what the `hash_compact_frustum` kernel reads (a
/// `vec4[6]` under scalar block layout), so the array uploads verbatim.
using FrustumPlanes = std::array<Vec4f, 6>;
// Pin the upload size the way every other shader-fed POD does (2026-07-05 ABI):
// the `vr::` backing type is swappable, so a Vec4f layout change is a compile
// error here, not a silent frustum misprojection.
static_assert(sizeof(FrustumPlanes) == 96,
              "FrustumPlanes must be 96 bytes (6 x vec4) to match the shader's "
              "scalar-layout vec4[6]");

/// @brief Build the world-space frustum planes for a pinhole camera.
///
/// The four side planes come from the image borders (a pixel `(u, v)` is inside
/// when `0 <= u <= width` and `0 <= v <= height`, the camera looking down +Z as
/// in @ref VoxelHashMap::allocate_from_depth); near/far clip in depth. The side
/// planes are widened ~10% (focal lengths scaled by 0.9) so a block straddling
/// the image edge is kept rather than clipped -- the cull is deliberately
/// conservative. Each camera-space plane is carried to world space by the
/// inverse-transpose of @p cam_to_world and renormalized to unit normal.
/// @param fx,fy          Focal lengths (pixels).
/// @param cx,cy          Principal point (pixels).
/// @param width,height   Image dimensions (pixels).
/// @param near_z,far_z   Near/far clip distances (metres, along camera +Z).
/// @param cam_to_world   Camera->world rigid transform (column-major).
/// @return The six inward-normal, unit-length frustum planes.
inline FrustumPlanes make_frustum_planes(float fx, float fy, float cx, float cy,
                                         std::uint32_t width,
                                         std::uint32_t height, float near_z,
                                         float far_z,
                                         const Mat4f& cam_to_world) {
  const float w = static_cast<float>(width);
  const float h = static_cast<float>(height);
  // Widen the lateral planes ~10% so an edge-straddling block survives the
  // cull.
  constexpr float kSideMargin = 0.9f;
  const float sx = fx * kSideMargin;
  const float sy = fy * kSideMargin;

  // Camera-space planes (inward normals; camera looks down +Z): left/right from
  // 0 <= u <= w, top/bottom from 0 <= v <= h, near/far from near_z <= z <=
  // far_z.
  const std::array<Vec4f, 6> camera_planes = {
      Vec4f(sx, 0.0f, cx, 0.0f),         // left:   sx*x + cx*z >= 0
      Vec4f(-sx, 0.0f, w - cx, 0.0f),    // right:  -sx*x + (w-cx)*z >= 0
      Vec4f(0.0f, sy, cy, 0.0f),         // top:    sy*y + cy*z >= 0
      Vec4f(0.0f, -sy, h - cy, 0.0f),    // bottom: -sy*y + (h-cy)*z >= 0
      Vec4f(0.0f, 0.0f, 1.0f, -near_z),  // near:   z - near_z >= 0
      Vec4f(0.0f, 0.0f, -1.0f, far_z),   // far:    far_z - z >= 0
  };

  // A plane transforms by the inverse-transpose of the point transform.
  const Mat4f to_world = glm::inverseTranspose(cam_to_world);
  FrustumPlanes planes;
  for (std::size_t i = 0; i < planes.size(); ++i) {
    const Vec4f p = to_world * camera_planes[i];
    const float len = glm::length(Vec3f(p));
    planes[i] = len > 0.0f ? p / len : p;
  }
  return planes;
}

/// @brief Build the world-space frustum planes from a camera's combined
///        view-projection matrix -- the *render* camera's form of the overload
///        above.
///
/// That one takes intrinsics in pixels, in the OpenCV convention the fusion
/// tiers pose a depth camera in (+Z forward, y down). A viewer's camera has
/// neither: it carries a projection built for clip space, and its handedness is
/// the renderer's. This reads the planes off the matrix itself
/// (Gribb-Hartmann), so it holds for whatever view and projection the caller
/// combined -- which is what culling the *meshed* set to what is on screen
/// needs, a question the depth camera's frustum answers only in follow-camera
/// mode.
///
/// @warning @p view_proj must map depth to **`[0, 1]`** -- Vulkan clip space,
///          which is what `volumetric_kit_gfx`'s `Camera::view_proj` emits (it
///          forces `GLM_FORCE_DEPTH_ZERO_TO_ONE` before including glm). The
///          near plane is read as row 2 alone; under OpenGL's `[-1, 1]` it
///          would be row 3 + row 2, and a GL-convention matrix passed here
///          clips at the eye rather than at `z_near`. That fails as mild
///          over-culling near the camera, not as anything obviously wrong.
///
/// @param view_proj  `proj * view`, column-major, mapping world -> clip.
/// @param margin_m   Push every plane outward by this many metres, keeping what
///                   is *nearly* visible rather than only what is. 0 (the
///                   default) is the exact frustum. Non-zero is what a consumer
///                   whose cull runs on a different thread from its draw wants:
///                   the mesh a frame draws was culled against a pose several
///                   frames old, and an exact frustum pops at the screen edge
///                   while the camera turns. A metre and not a focal scale --
///                   the planes are normalized below, so the offset is a real
///                   world-space distance and stays comparable across fields of
///                   view.
/// @return The six inward-normal, unit-length planes, in the left, right, top,
///         bottom, near, far order the pinhole overload also returns.
inline FrustumPlanes make_frustum_planes(const Mat4f& view_proj,
                                         float margin_m = 0.0f) {
  // Row i of a column-major glm matrix, which indexes m[column][row].
  const auto row = [&view_proj](int i) {
    return Vec4f(view_proj[0][i], view_proj[1][i], view_proj[2][i],
                 view_proj[3][i]);
  };
  const Vec4f r0 = row(0);
  const Vec4f r1 = row(1);
  const Vec4f r2 = row(2);
  const Vec4f r3 = row(3);
  // Each plane is the half-space one clip-space bound describes: -w <= x <= w,
  // -w <= y <= w, and 0 <= z <= w for Vulkan's [0, 1] depth. Vulkan's y is
  // down, so `r3 + r1` (the y >= -w bound) is the TOP of the image -- the same
  // edge the pinhole overload calls top, where v = 0 is the first row.
  const std::array<Vec4f, 6> clip_planes = {
      r3 + r0,  // left
      r3 - r0,  // right
      r3 + r1,  // top
      r3 - r1,  // bottom
      r2,       // near
      r3 - r2,  // far
  };
  FrustumPlanes planes;
  for (std::size_t i = 0; i < planes.size(); ++i) {
    const float len = glm::length(Vec3f(clip_planes[i]));
    planes[i] = len > 0.0f ? clip_planes[i] / len : clip_planes[i];
    // Applied AFTER normalizing, which is the whole reason the margin can be
    // stated in metres: on a unit normal, `dot(n, p) + d` is a signed distance,
    // so raising d moves the plane outward by exactly that far.
    planes[i].w += margin_m;
  }
  return planes;
}

}  // namespace volumetric_kit::recon::volume
