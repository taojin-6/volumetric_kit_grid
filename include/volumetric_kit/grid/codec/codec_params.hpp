// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file codec/codec_params.hpp
/// @brief The knobs of the per-frame TSDF geometry codec: how many DCT
///        coefficients each block keeps and how coarsely they are quantized.

#include <array>
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
///
/// A literal, the formula's float, rather than the formula: `std::sqrt` is not
/// `constexpr`, so the formula was initialized at run time, and a validate()
/// in another file's static initializer could read it as 0. The params test
/// holds the two equal.
inline constexpr float kMinStep = 6.90555025e-4f;

/// @brief The coarsest quantization step the codec accepts, as a fraction of
///        `trunc_dist`.
///
/// Past `2 sqrt(kVoxelsPerBlock)` (~45.3) every coefficient quantizes to 0, so
/// a coarser step codes nothing more, and this is the power of two past it.
/// It is what keeps a decoded frame finite: a step from a corrupt header
/// times a coefficient of up to @ref kMaxQuantizedMagnitude stays far inside
/// a float, where 1e38 would make it infinite and the inverse transform's
/// sums of opposite infinities NaN. @ref CodecParams::validate refuses
/// anything coarser.
inline constexpr float kMaxStep = 64.0f;
static_assert(kMaxStep * kMaxStep >= 4.0f * kVoxelsPerBlock,
              "kMaxStep must quantize every coefficient to 0");

/// @brief How a frame's blocks are transformed and quantized.
///
/// A coefficient at frequency `(x, y, z)` is quantized with step
/// `quantization_scale * quantization_weights[x + 8*y + 64*z]`, a fraction of
/// the grid's `trunc_dist`. The transform runs on SDF divided by `trunc_dist`,
/// so one setting holds across scenes whatever their truncation band. DC is
/// the ordinary entry at `(0, 0, 0)`.
///
/// The table controls relative precision; the scale changes overall quality.
/// Multiplying the table and dividing the scale by the same positive number
/// describes the same steps. Presets should use weight 1 at DC to make their
/// scales comparable, but this is a convention, not a validation requirement.
///
/// Defaults keep K = 64 and a uniform step of 0.2, the provisional room0
/// settings. TODO(codec): tune per-basis tables at matched rates across room0,
/// normalized Rafa2 and the analytic sphere before changing the defaults;
/// the first band/radial candidates did not establish a consistent win.
struct VR_CODEC_API CodecParams {
  /// Coefficients kept per block, taken in 3-D zigzag order (lowest `x+y+z`
  /// first). In [1, @ref kVoxelsPerBlock]; @ref kVoxelsPerBlock keeps the
  /// whole transform. This cutoff is independent of the quantization table.
  std::uint32_t coefficient_count = 64;
  /// Positive finite scale shared by every basis; effective steps are fractions
  /// of `trunc_dist` and must each lie in [@ref kMinStep, @ref kMaxStep].
  float quantization_scale = 0.2f;
  /// Positive finite relative steps in canonical frequency order
  /// `x + 8*y + 64*z`, with each frequency in [0, 7], not in zigzag order.
  /// Every entry is carried in a frame, including entries beyond the K cutoff.
  std::array<float, kVoxelsPerBlock> quantization_weights = [] {
    std::array<float, kVoxelsPerBlock> weights{};
    for (float& weight : weights) weight = 1.0f;
    return weights;
  }();

  /// @brief Check that every field is one the transform can honour.
  /// @return OK, or @ref Status::invalid_argument naming the field: a
  ///         @ref coefficient_count outside [1, @ref kVoxelsPerBlock],
  ///         a non-positive or non-finite scale or weight, or any effective
  ///         step outside [@ref kMinStep, @ref kMaxStep], even beyond K.
  Status validate() const;
};

}  // namespace volumetric_kit::recon::codec
