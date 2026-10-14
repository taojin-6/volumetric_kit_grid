// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file examples/common/codec_config.hpp
/// @brief Map the core's validated codec and mesh settings to runtime types.

#include "codec_quantization.hpp"
#include "codec_stream.hpp"
#include "config.hpp"
#include "volumetric_kit/core/base/result.hpp"
#include "volumetric_kit/recon/codec/encoder.hpp"
#include "volumetric_kit/recon/tsdf/mesh_integrator.hpp"

namespace vr_example {

/// @return The selected table's name, as @ref apply_quantization_table and
///         the measurement report take it.
inline const char* quantization_name(config::QuantizationTable table) {
  switch (table) {
    case config::QuantizationTable::Uniform:
      return "uniform";
    case config::QuantizationTable::Band:
      return "band";
    case config::QuantizationTable::Radial:
      return "radial";
  }
  return "unknown";
}

/// @return Runtime encoder settings: core's values, and the weights of its
///         `quantization_table` (@ref apply_quantization_table); or
///         InvalidArgument for `entropy` `host`, since the codec codes on the
///         device (`auto` and `device` are the same), or from
///         `CodecParams::validate` for a kept coefficient's step outside the
///         codec's bounds, which core leaves to the codec.
inline volumetric_kit::core::Result<volumetric_kit::recon::codec::EncoderConfig>
encoder_config(const config::Codec& settings) {
  namespace codec = volumetric_kit::recon::codec;
  // TODO(codec): core drops codec.entropy with its next config change; until
  // then recon, which owns the codec's rules, refuses the host.
  if (settings.entropy == config::Entropy::Host) {
    return volumetric_kit::core::Status::invalid_argument(
        "codec.entropy: host is not a choice; the codec codes on the device");
  }
  codec::EncoderConfig result;
  result.params.coefficient_count = settings.coefficient_count;
  result.params.quantization_scale = settings.quantization_scale;
  VKC_TRY(apply_quantization_table(
      result.params, quantization_name(settings.quantization_table)));
  result.segment_size = settings.segment_size;
  VKC_TRY(result.params.validate());
  return result;
}

/// @return Where a codec stream writes besides its report: core's
///         `output.csv` and `output.stream`, already relative to the
///         scenario's directory; an empty path writes nothing.
inline CodecOutputs codec_outputs(const config::Output& output) {
  return {output.csv, output.stream};
}

/// @return Runtime SDF parameters copied from core's validated mesh source.
inline volumetric_kit::recon::tsdf::MeshSdfParams mesh_sdf_config(
    const config::Source& source) {
  namespace tsdf = volumetric_kit::recon::tsdf;
  tsdf::MeshSdfParams result;
  result.mode = source.sdf_mode == config::SdfMode::Signed
                    ? tsdf::MeshSdfMode::Signed
                    : tsdf::MeshSdfMode::Shell;
  result.shell_voxels = source.shell_voxels;
  return result;
}

}  // namespace vr_example
