// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file volume/voxel_color.hpp
/// @brief The voxel colour attribute's format (@ref kColorAttribute), the host
///        half of `volume/shaders/voxel_color.glsl`, which the kernels that
///        write and read it include.
///
/// A `uint32_t` a voxel: R, G, B in the low three bytes in the canonical
/// encoded form -- the linear working space (BT.709 primaries, D65) through the
/// exact piecewise sRGB curve, full range -- alpha `0xFF` once observed, and
/// @ref kVoxelColorUnobserved for a voxel no colour was fused into. The curve
/// is the published one, held to `core/color_space.hpp`'s and to the GLSL by
/// `recon_volume_voxel_color`.

#include <cmath>
#include <cstdint>

#include "volumetric_kit/recon/core/math/vector_types.hpp"

namespace volumetric_kit::recon::volume {

/// @brief A voxel no colour has been fused into: the zeroed attribute, and
///        what a clear writes. A fused colour carries alpha `0xFF`, so it is
///        never this, not even black.
inline constexpr std::uint32_t kVoxelColorUnobserved = 0;

/// @brief Decode one exact-piecewise-sRGB-encoded channel to linear.
/// @param e Encoded value, nominally in [0, 1]; values outside take the same
///          branches rather than clamping, so the function stays monotone.
/// @return The linear-light value.
inline float srgb_to_linear(float e) noexcept {
  return e <= 0.04045f ? e * (1.0f / 12.92f)
                       : std::pow((e + 0.055f) * (1.0f / 1.055f), 2.4f);
}

/// @brief Encode one linear channel to exact piecewise sRGB.
/// @param l Linear-light value, nominally in [0, 1].
/// @return The encoded value. Inverse of @ref srgb_to_linear.
inline float linear_to_srgb(float l) noexcept {
  return l <= 0.0031308f ? l * 12.92f
                         : 1.055f * std::pow(l, 1.0f / 2.4f) - 0.055f;
}

/// @brief What a voxel's first colour observation stores: the observed
///        word's own R, G, B, with no decode or encode, and alpha `0xFF`.
/// @param observed A canonical-encoded RGB word; its high byte is ignored.
constexpr std::uint32_t voxel_color_from_observed(
    std::uint32_t observed) noexcept {
  return (observed & 0x00FFFFFFu) | 0xFF000000u;
}

/// @brief A voxel's colour as linear working values, for a reader that
///        averages it, as the mesher's vertex colours do.
/// @param packed The voxel's word.
/// @return Its R, G, B decoded; opaque white for @ref kVoxelColorUnobserved,
///         so an unobserved voxel does not drag an average toward black.
inline Vec3f voxel_color_to_linear(std::uint32_t packed) noexcept {
  if (packed == kVoxelColorUnobserved) return Vec3f(1.0f);
  const float inv = 1.0f / 255.0f;
  return Vec3f{
      srgb_to_linear(static_cast<float>(packed & 0xFFu) * inv),
      srgb_to_linear(static_cast<float>((packed >> 8) & 0xFFu) * inv),
      srgb_to_linear(static_cast<float>((packed >> 16) & 0xFFu) * inv)};
}

/// @brief The word a fused colour's running mean stores: each channel
///        encoded, clamped to [0, 1] and rounded to nearest, as GLSL's
///        `packUnorm4x8`, with alpha `0xFF`. NaN encodes as 0.
/// @param linear Linear working values.
inline std::uint32_t voxel_color_from_linear(const Vec3f& linear) noexcept {
  auto to_u8 = [](float v) -> std::uint32_t {
    const float e = linear_to_srgb(v);
    const float c = (e > 0.0f) ? (e < 1.0f ? e : 1.0f) : 0.0f;
    return static_cast<std::uint32_t>(c * 255.0f + 0.5f);
  };
  return to_u8(linear.x) | (to_u8(linear.y) << 8) | (to_u8(linear.z) << 16) |
         0xFF000000u;
}

}  // namespace volumetric_kit::recon::volume
