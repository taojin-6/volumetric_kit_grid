// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file codec/codec_params.hpp
/// @brief The knobs of the per-frame TSDF geometry codec: how many DCT
///        coefficients each block keeps and how coarsely they are quantized.

#include <cmath>
#include <cstdint>
#include <limits>

#include "volumetric_kit/recon/codec/export.hpp"
#include "volumetric_kit/recon/core/result.hpp"

namespace volumetric_kit::recon::codec {

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

/// @brief The smallest quantization step the codec accepts, as a fraction of
///        `trunc_dist`: `sqrt(kVoxelsPerBlock) / kMaxQuantizedMagnitude`.
///
/// The transform is orthonormal and runs on SDF normalized into [-1, 1], so
/// every coefficient is a dot product of a unit basis vector with a vector of
/// norm at most `sqrt(kVoxelsPerBlock)` (~22.63). A step at or above this
/// therefore keeps every quantized value inside @ref kMaxQuantizedMagnitude, so
/// the clamp is a guard against float drift and never a silent loss.
/// @ref CodecParams::validate refuses anything finer.
inline const float kMinStep = static_cast<float>(
    std::sqrt(double(kVoxelsPerBlock)) / kMaxQuantizedMagnitude);

/// @brief How a frame's blocks are transformed and quantized.
///
/// Both steps are **fractions of the grid's `trunc_dist`**, not metres: the
/// transform runs on SDF divided by `trunc_dist`, so one setting holds across
/// scenes whatever their truncation band.
///
/// The defaults -- K = 64, one step of 0.2 for DC and AC alike -- are
/// room0's (the 2026-09-27 measurement), and **provisional** until the
/// per-band quantization study. Against the prior engine's K = 32 with a DC
/// step five times its AC step, which they replaced, they are 7% smaller and
/// 28% more accurate at 1 cm, and fit a frame interval on the host. A DC step
/// coarser than the AC one bought nothing: the transform is orthonormal, so a
/// unit of error costs the same in any coefficient.
///
/// TODO(codec): re-choose the defaults with the per-band quantization study
/// -- a step per `x + y + z` band, judged on room0 and on the decoder test's
/// sphere, where these lose (the 2026-09-27 decision).
struct VR_CODEC_API CodecParams {
  /// Coefficients kept per block, taken in 3-D zigzag order (lowest `x+y+z`
  /// first). In [1, @ref kVoxelsPerBlock]; @ref kVoxelsPerBlock keeps the
  /// whole transform.
  std::uint32_t coefficient_count = 64;
  /// Quantization step of the DC coefficient (zigzag index 0).
  float dc_step = 0.2f;
  /// Quantization step of every AC coefficient.
  float ac_step = 0.2f;

  /// @brief Check that every field is one the transform can honour.
  /// @return OK, or @ref Status::invalid_argument naming the field: a
  ///         @ref coefficient_count outside [1, @ref kVoxelsPerBlock], or a
  ///         step that is not finite or is below @ref kMinStep.
  Status validate() const;
};

}  // namespace volumetric_kit::recon::codec
