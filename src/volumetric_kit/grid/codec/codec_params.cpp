// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include "volumetric_kit/recon/codec/codec_params.hpp"

#include <cmath>
#include <string>

namespace volumetric_kit::recon::codec {
namespace {

Status check_step(const char* field, float step) {
  // !(step >= kMinStep) rather than step < kMinStep, so a NaN is refused too.
  if (!std::isfinite(step) || !(step >= kMinStep)) {
    return Status::invalid_argument(
        std::string("CodecParams: ") + field +
        " must be finite and >= sqrt(512)/32767 (a fraction of trunc_dist), "
        "or a quantized coefficient could overflow its clamp");
  }
  return {};
}

}  // namespace

Status CodecParams::validate() const {
  if (coefficient_count < 1 || coefficient_count > kVoxelsPerBlock) {
    return Status::invalid_argument(
        "CodecParams: coefficient_count must be in [1, 512]");
  }
  VR_TRY(check_step("dc_step", dc_step));
  VR_TRY(check_step("ac_step", ac_step));
  return {};
}

}  // namespace volumetric_kit::recon::codec
