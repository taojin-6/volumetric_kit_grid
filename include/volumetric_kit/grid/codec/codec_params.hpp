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

/// @brief Where an @ref Encoder or a @ref Decoder runs a frame's rANS
///        coding. Every frame both sides can code gives the same result --
///        the same bytes written, the same blocks decoded, a corrupt frame
///        refused -- so the choice changes only the time it takes.
///
/// The device cannot code a frame past `maxStorageBufferRange` or free
/// memory, nor decode one whose segments are longer than
/// @ref kMaxDeviceDecodeSegmentSize. @ref kDevice refuses such a frame
/// (@ref Status::Code::InvalidArgument or @ref Status::Code::OutOfMemory),
/// which the host would code, and may refuse a corrupt one with another
/// message than the host's.
enum class EntropyCoding {
  /// The device for a frame of at least @ref kMinDeviceEncodeSegments
  /// segments when encoding, or @ref kMinDeviceDecodeSegments when decoding;
  /// the host for a smaller one, which has too few segments for the device
  /// to run in parallel. A frame the device cannot code is coded on the
  /// host, and so is every frame once the device's rANS kernels fail to
  /// build; any other device failure is reported as @ref kDevice reports it.
  kAuto,
  /// On the host: the coefficients and masks cross the bus, read back to
  /// encode or uploaded once decoded.
  kHost,
  /// On the device, one invocation per segment: the coefficients and masks
  /// stay in device memory, and only the frame, the symbol counts (encoding)
  /// and the coordinates (decoding) cross the bus.
  kDevice,
};

/// The fewest segments @ref EntropyCoding::kAuto encodes on the device:
/// about where the device stopped losing to the host on Apple M5 Max (the
/// RTX 5090 broke even near 20; the 2026-10-03 encoding decision).
inline constexpr std::uint32_t kMinDeviceEncodeSegments = 48;

/// The fewest segments @ref EntropyCoding::kAuto decodes on the device: at
/// the default 64-block segments, about where the device stopped losing to
/// the host on both Apple M5 Max and RTX 5090, for content as cheap to decode
/// on the host as room0 (the 2026-10-03 decoding decision). Decoding takes
/// more than encoding: its symbol walk is inside the serial chain.
inline constexpr std::uint32_t kMinDeviceDecodeSegments = 80;

/// The longest segment, in blocks, the device decodes. One invocation decodes
/// a whole segment, at 30 to 45 us a block on Apple M5 Max and RTX 5090, and
/// the frame sets its length; this keeps one invocation near 50 ms there,
/// well inside a GPU watchdog's limit on a slower device.
inline constexpr std::uint32_t kMaxDeviceDecodeSegmentSize = 1024;

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
  /// Positive normal scale shared by every basis; effective steps are
  /// fractions of `trunc_dist` and must each lie in
  /// [@ref kMinStep, @ref kMaxStep].
  float quantization_scale = 0.2f;
  /// Positive normal relative steps in canonical frequency order
  /// `x + 8*y + 64*z`, with each frequency in [0, 7], not in zigzag order.
  /// Only the K frequencies the zigzag cutoff keeps are used, checked and
  /// carried in a frame; the rest are ignored and decode as 1.
  std::array<float, kVoxelsPerBlock> quantization_weights = [] {
    std::array<float, kVoxelsPerBlock> weights{};
    for (float& weight : weights) weight = 1.0f;
    return weights;
  }();

  /// @brief Check that every field is one the transform can honour.
  /// @return OK, or @ref Status::invalid_argument naming the field: a
  ///         @ref coefficient_count outside [1, @ref kVoxelsPerBlock], a
  ///         scale or kept weight that is not positive and normal (zero,
  ///         subnormal, infinite or NaN), or a kept frequency's effective
  ///         step outside [@ref kMinStep, @ref kMaxStep].
  Status validate() const;
};

}  // namespace volumetric_kit::recon::codec
