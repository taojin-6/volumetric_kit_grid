// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include "volumetric_kit/recon/codec/codec_params.hpp"

#include <string>

namespace volumetric_kit::recon::codec {
namespace {

Status check_step(const char* field, float step) {
  // Negated, so a NaN is refused too.
  if (!(step >= kMinStep && step <= kMaxStep)) {
    return Status::invalid_argument(
        std::string("CodecParams: ") + field + " must be in [" +
        std::to_string(kMinStep) + ", " + std::to_string(kMaxStep) +
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
  VR_TRY(check_step("dc_step", dc_step));
  VR_TRY(check_step("ac_step", ac_step));
  return {};
}

}  // namespace volumetric_kit::recon::codec
