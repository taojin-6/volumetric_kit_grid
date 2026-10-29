// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// The voxel colour attribute's format, which recon_volume exports for any
// kernel's #include (vkc_export_glsl, volume/CMakeLists.txt): a uint a voxel,
// R, G, B in the low three bytes in the canonical encoded form, alpha 0xFF
// once observed, and 0 for a voxel no colour was fused into. Fusion writes it
// (tsdf_integrate.glsl) and the mesher reads it
// (marching_cubes_sparse_common.glsl). volume/voxel_color.hpp is its host
// half, and recon_volume_voxel_color holds the two together on the device.
//
// The rule (DESIGN.md, "Color space"): 8-bit color is encoded, float color is
// linear. Averaging is linear and a display encoding deliberately is not, so
// every blend decodes first. The canonical encoded form is the working space
// (linear BT.709/D65) through the EXACT piecewise sRGB transfer, full range --
// exact, not pow(x, 2.2), because hardware _SRGB sampling decodes the atlas
// with the exact curve and the renderer's hybrid pipeline picks between the
// atlas and the vertex color per triangle across one surface: an
// approximation here seams exactly where texturing stops.

#ifndef VR_VOXEL_COLOR_GLSL
#define VR_VOXEL_COLOR_GLSL

// Exact piecewise sRGB decode, one channel. Values outside [0,1] take the same
// branches rather than clamping, so the function stays monotone.
float vrSrgbToLinear(float e) {
  return e <= 0.04045 ? e * (1.0 / 12.92)
                      : pow((e + 0.055) * (1.0 / 1.055), 2.4);
}

// Exact piecewise sRGB encode, one channel. Inverse of vrSrgbToLinear.
float vrLinearToSrgb(float l) {
  return l <= 0.0031308 ? l * 12.92 : 1.055 * pow(l, 1.0 / 2.4) - 0.055;
}

vec3 vrSrgbToLinear(vec3 e) {
  return vec3(vrSrgbToLinear(e.r), vrSrgbToLinear(e.g), vrSrgbToLinear(e.b));
}

vec3 vrLinearToSrgb(vec3 l) {
  return vec3(vrLinearToSrgb(l.r), vrLinearToSrgb(l.g), vrLinearToSrgb(l.b));
}

// Decode a packed canonical-encoded RGB word to linear working values. Alpha
// is ignored; see vrPackLinearToSrgb for why it is nonetheless always written.
vec3 vrUnpackSrgbToLinear(uint packed) {
  return vrSrgbToLinear(unpackUnorm4x8(packed).rgb);
}

// Encode linear working values back into a packed canonical word: what a
// fused colour's running mean stores. Clamps and rounds to nearest
// (packUnorm4x8's contract, matching the host's +0.5 round), and always writes
// alpha 0xFF, so a written color is never kVoxelColorUnobserved, not even
// pure black.
uint vrPackLinearToSrgb(vec3 linear) {
  return packUnorm4x8(vec4(vrLinearToSrgb(clamp(linear, 0.0, 1.0)), 1.0));
}

// A voxel no colour has been fused into: the zeroed attribute, and what a
// clear writes.
const uint kVoxelColorUnobserved = 0u;

// What a voxel's first colour observation stores: the observed word's own R,
// G, B, with no decode/encode round trip (an assignment is not an average),
// and alpha 0xFF, since an observation's high byte is undefined.
uint vrVoxelColorFromObserved(uint observed) {
  return (observed & 0x00FFFFFFu) | 0xFF000000u;
}

// A voxel's colour as linear working values, for a reader that averages it:
// an unobserved voxel reads opaque white, so it does not drag an
// interpolation toward black. White is 1.0 in either space.
vec3 vrVoxelColorToLinear(uint packed) {
  return packed != kVoxelColorUnobserved ? vrUnpackSrgbToLinear(packed)
                                         : vec3(1.0);
}

#endif  // VR_VOXEL_COLOR_GLSL
