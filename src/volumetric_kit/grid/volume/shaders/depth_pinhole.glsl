// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// Volume's depth camera on the device: the mirror of volume::DepthPinhole
// (voxel_hash_map.hpp), byte-identical under scalar block layout (floats and
// uints at their 4-byte offsets, the mat4 at offset 32), and the one line of
// pinhole arithmetic depth allocation needs. The unprojection is the
// published pinhole's (recon's core/shaders/pinhole.glsl,
// vr_unproject_pinhole), operation for operation, copied so the voxel layer
// needs no camera vocabulary; recon's tests/volume_depth_unproject_test.cpp
// holds the two to the same bits.

#ifndef VGRID_DEPTH_PINHOLE_GLSL
#define VGRID_DEPTH_PINHOLE_GLSL

struct DepthPinhole {
  float fx;
  float fy;
  float cx;
  float cy;
  float min_depth;
  float max_depth;
  uint width;
  uint height;
  mat4 cam_to_world;  // column-major camera -> world rigid transform
};

// The camera-space point at pixel `px` and depth `z` along +Z, with `k` =
// (fx, fy, cx, cy): x = (u - cx) z / fx, y = (v - cy) z / fy.
vec3 vgridUnprojectDepth(vec2 px, float z, vec4 k) {
  return vec3((px - k.zw) * z / k.xy, z);
}

#endif  // VGRID_DEPTH_PINHOLE_GLSL
