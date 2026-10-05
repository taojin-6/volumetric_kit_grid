// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include "volumetric_kit/recon/codec/codec_params.hpp"

#include <cmath>
#include <string>

#include "dct_tables.hpp"

namespace volumetric_kit::recon::codec {
namespace {

core::Status check_step(std::size_t index, float step) {
  // Negated, so a NaN is refused too.
  if (!(step >= kMinStep && step <= kMaxStep)) {
    return core::Status::invalid_argument(
        "CodecParams: effective step at frequency index " +
        std::to_string(index) + " must be in [" + std::to_string(kMinStep) +
        ", " + std::to_string(kMaxStep) +
        "] (a fraction of trunc_dist): finer, a quantized coefficient could "
        "overflow its clamp; coarser codes nothing more");
  }
  return {};
}

}  // namespace

core::Status CodecParams::validate() const {
  if (coefficient_count < 1 || coefficient_count > kVoxelsPerBlock) {
    return core::Status::invalid_argument(
        "CodecParams: coefficient_count must be in [1, " +
        std::to_string(kVoxelsPerBlock) + "]");
  }
  // Normal, not merely positive: a host or device that flushes subnormals to
  // zero would otherwise disagree about whether the frame is valid.
  if (!(quantization_scale > 0.0f) || !std::isnormal(quantization_scale)) {
    return core::Status::invalid_argument(
        "CodecParams: quantization_scale must be normal and positive");
  }
  // Only the K kept frequencies are quantized, so only their weights count.
  static const auto zigzag = detail::zigzag_order();
  for (std::uint32_t j = 0; j < coefficient_count; ++j) {
    const std::size_t i = zigzag[j];
    const float weight = quantization_weights[i];
    if (!(weight > 0.0f) || !std::isnormal(weight)) {
      return core::Status::invalid_argument(
          "CodecParams: quantization_weights[" + std::to_string(i) +
          "] must be normal and positive");
    }
    VKC_TRY(check_step(i, quantization_scale * weight));
  }
  return {};
}

}  // namespace volumetric_kit::recon::codec
