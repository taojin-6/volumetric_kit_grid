// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file codec/codec_params.hpp
/// @brief The knobs of the per-frame TSDF geometry codec: how many DCT
///        coefficients each block keeps and how coarsely they are quantized.

#include <cstdint>
#include <limits>

#include "volumetric_kit/recon/codec/export.hpp"
#include "volumetric_kit/recon/core/result.hpp"

namespace volumetric_kit::recon::codec {

namespace detail {

/// `sqrt(x)` for `x >= 0` at compile time, which `std::sqrt` is not in C++17,
/// to within an ulp: Newton's iteration from above, which decreases to the
/// root and stops once a step no longer moves it down.
constexpr double constexpr_sqrt(double x) {
  double r = x > 1.0 ? x : 1.0;
  for (double next = 0.5 * (r + x / r); next < r; next = 0.5 * (r + x / r)) {
    r = next;
  }
  return r;
}

}  // namespace detail

/// The only block edge the codec transforms. Its zigzag order and cosine table
/// are 8-specific, so a grid with another `block_size` is refused, never
/// transformed wrong.
inline constexpr std::int32_t kBlockSize = 8;

/// Voxels (and so DCT coefficients) in one block: `kBlockSize^3`.
inline constexpr std::uint32_t kVoxelsPerBlock =
    static_cast<std::uint32_t>(kBlockSize * kBlockSize * kBlockSize);

/// Bits in one word of the observed mask.
inline constexpr std::uint32_t kMaskWordBits =
    std::numeric_limits<std::uint32_t>::digits;

/// Words in one block's observed mask: one bit per voxel.
inline constexpr std::uint32_t kMaskWordsPerBlock =
    kVoxelsPerBlock / kMaskWordBits;
static_assert(kVoxelsPerBlock % kMaskWordBits == 0,
              "a block's mask must fill whole words");

/// The magnitude a quantized coefficient is clamped to: `INT16_MAX`, so a
/// coefficient and its negation both fit in 16 bits, the range the entropy
/// coder's alphabet is sized by.
inline constexpr std::int32_t kMaxQuantizedMagnitude =
    std::numeric_limits<std::int16_t>::max();

/// @brief The largest magnitude one of a block's DCT coefficients can take,
///        `sqrt(kVoxelsPerBlock)` (~22.63).
///
/// The transform is orthonormal and runs on SDF normalized into [-1, 1], so
/// every coefficient is a dot product of a unit basis vector with a vector of
/// norm at most `sqrt(kVoxelsPerBlock)`.
inline constexpr double kMaxCoefficientMagnitude =
    detail::constexpr_sqrt(kVoxelsPerBlock);

/// @brief The smallest quantization step the codec accepts, as a fraction of
///        `trunc_dist`: @ref kMaxCoefficientMagnitude over
///        @ref kMaxQuantizedMagnitude.
///
/// A step at or above it keeps every quantized value inside
/// @ref kMaxQuantizedMagnitude, so the clamp is a guard against float drift and
/// never a silent loss. @ref CodecParams::validate refuses anything finer.
inline constexpr float kMinStep =
    static_cast<float>(kMaxCoefficientMagnitude / kMaxQuantizedMagnitude);

/// @brief How a frame's blocks are transformed and quantized.
///
/// Both steps are **fractions of the grid's `trunc_dist`**, not metres: the
/// transform runs on SDF divided by `trunc_dist`, so one setting holds across
/// scenes whatever their truncation band. The defaults are the prior engine's
/// 0.01 m DC / 0.002 m AC at its 40 mm band, carried over until a room0
/// measurement tunes them.
struct VR_CODEC_API CodecParams {
  /// Coefficients kept per block, taken in 3-D zigzag order (lowest `x+y+z`
  /// first). In [1, @ref kVoxelsPerBlock]; @ref kVoxelsPerBlock keeps the
  /// whole transform.
  std::uint32_t coefficient_count = 32;
  /// Quantization step of the DC coefficient (zigzag index 0).
  float dc_step = 0.25f;
  /// Quantization step of every AC coefficient.
  float ac_step = 0.05f;

  /// @brief Check that every field is one the transform can honour.
  /// @return OK, or @ref Status::invalid_argument naming the field: a
  ///         @ref coefficient_count outside [1, @ref kVoxelsPerBlock], or a
  ///         step that is not finite or is below @ref kMinStep.
  Status validate() const;
};

}  // namespace volumetric_kit::recon::codec
