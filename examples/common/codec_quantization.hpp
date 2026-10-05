// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

// Candidate tables for the codec examples' rate-distortion study. These are
// hypotheses to measure, not tuned production defaults. Every table treats
// axis permutations alike and has a DC weight of one, so --step has the same
// meaning across the families.

#include <cstdint>
#include <string>

#include "volumetric_kit/recon/codec/codec_params.hpp"

namespace vr_example {

/// @brief Set a candidate table, keeping the coefficient count and scale.
/// @param name "uniform", "band" (linear total frequency), or "radial"
///             (squared frequency magnitude, different within a total band).
/// @return OK, or InvalidArgument for an unknown table; on error @p params
///         is unchanged.
inline volumetric_kit::core::Status apply_quantization_table(
    volumetric_kit::recon::codec::CodecParams& params,
    const std::string& name) {
  namespace vkc = volumetric_kit::core;
  if (name != "uniform" && name != "band" && name != "radial") {
    return vkc::Status::invalid_argument(
        "unknown quantization table: " + name +
        " (expected uniform, band, or radial)");
  }
  for (std::uint32_t w = 0; w < 8; ++w) {
    for (std::uint32_t v = 0; v < 8; ++v) {
      for (std::uint32_t u = 0; u < 8; ++u) {
        float weight = 1.0f;
        if (name == "band") {
          weight += 0.25f * float(u + v + w);
        } else if (name == "radial") {
          weight += 0.125f * float(u * u + v * v + w * w);
        }
        params.quantization_weights[u + 8 * v + 64 * w] = weight;
      }
    }
  }
  return {};
}

}  // namespace vr_example
