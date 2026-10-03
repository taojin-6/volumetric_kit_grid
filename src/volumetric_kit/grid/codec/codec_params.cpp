// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include "volumetric_kit/recon/codec/codec_params.hpp"

#include <cmath>
#include <string>

namespace volumetric_kit::recon::codec {
namespace {

Status check_step(std::size_t index, float step) {
  // Negated, so a NaN is refused too.
  if (!(step >= kMinStep && step <= kMaxStep)) {
    return Status::invalid_argument(
        "CodecParams: effective step at frequency index " +
        std::to_string(index) + " must be in [" + std::to_string(kMinStep) +
        ", " + std::to_string(kMaxStep) +
        "] (a fraction of trunc_dist): finer, a quantized coefficient could "
        "overflow its clamp; coarser codes nothing more");
  }
  return {};
}

}  // namespace

Status CodecParams::validate() const {
  if (coefficient_count < 1 || coefficient_count > kVoxelsPerBlock) {
    return Status::invalid_argument(
        "CodecParams: coefficient_count must be in [1, " +
        std::to_string(kVoxelsPerBlock) + "]");
  }
  if (!(quantization_scale > 0.0f) || !std::isfinite(quantization_scale)) {
    return Status::invalid_argument(
        "CodecParams: quantization_scale must be finite and positive");
  }
  for (std::size_t i = 0; i < quantization_weights.size(); ++i) {
    const float weight = quantization_weights[i];
    if (!(weight > 0.0f) || !std::isfinite(weight)) {
      return Status::invalid_argument("CodecParams: quantization_weights[" +
                                      std::to_string(i) +
                                      "] must be finite and positive");
    }
    VR_TRY(check_step(i, quantization_scale * weight));
  }
  return {};
}

}  // namespace volumetric_kit::recon::codec
