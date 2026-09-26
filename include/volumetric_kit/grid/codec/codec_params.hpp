// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file codec/codec_params.hpp
/// @brief The knobs of the per-frame TSDF geometry codec: how many DCT
///        coefficients each block keeps and how coarsely they are quantized.

#include <cstdint>

#include "volumetric_kit/recon/codec/export.hpp"
#include "volumetric_kit/recon/core/result.hpp"

namespace volumetric_kit::recon::codec {

/// The only block edge the codec transforms. Its zigzag order and cosine table
/// are 8-specific, so a grid with another `block_size` is refused, never
/// transformed wrong.
inline constexpr std::int32_t kBlockSize = 8;

/// Voxels (and so DCT coefficients) in one block: `kBlockSize^3`.
inline constexpr std::uint32_t kVoxelsPerBlock = 512;

/// 32-bit words in one block's observed mask: one bit per voxel.
inline constexpr std::uint32_t kMaskWordsPerBlock = kVoxelsPerBlock / 32;

/// The magnitude a quantized coefficient is clamped to, and the bound the
/// entropy coder's alphabet is sized by.
inline constexpr std::int32_t kMaxQuantizedMagnitude = 32767;

/// @brief The smallest quantization step the codec accepts, as a fraction of
///        `trunc_dist`.
///
/// The transform is orthonormal and runs on SDF normalized into [-1, 1], so
/// every coefficient is a dot product of a unit basis vector with a vector of
/// norm at most `sqrt(512)` -- bounded by `sqrt(512) ~= 22.63`. A step at or
/// above `sqrt(512) / 32767` therefore keeps every quantized value inside
/// @ref kMaxQuantizedMagnitude, so the clamp is a guard against float drift and
/// never a silent loss. @ref CodecParams::validate refuses anything finer.
inline constexpr float kMinStep = 22.627417f / 32767.0f;

/// @brief How a frame's blocks are transformed and quantized.
///
/// Both steps are **fractions of the grid's `trunc_dist`**, not metres: the
/// transform runs on SDF divided by `trunc_dist`, so one setting holds across
/// scenes whatever their truncation band. The defaults are the prior engine's
/// 0.01 m DC / 0.002 m AC at its 40 mm band, carried over until a room0
/// measurement tunes them.
struct VR_CODEC_API CodecParams {
  /// Coefficients kept per block, taken in 3-D zigzag order (lowest `x+y+z`
  /// first). In [1, @ref kVoxelsPerBlock]; 512 keeps the whole transform.
  std::uint32_t coefficient_count = 32;
  /// Quantization step of the DC coefficient (zigzag index 0).
  float dc_step = 0.25f;
  /// Quantization step of every AC coefficient.
  float ac_step = 0.05f;

  /// @brief Check that every field is one the transform can honour.
  /// @return OK, or @ref Status::invalid_argument naming the field: a
  ///         @ref coefficient_count outside [1, 512], or a step that is not
  ///         finite or is below @ref kMinStep.
  Status validate() const;
};

}  // namespace volumetric_kit::recon::codec
