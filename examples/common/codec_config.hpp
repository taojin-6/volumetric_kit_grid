// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file examples/common/codec_config.hpp
/// @brief Map the core's validated codec and mesh settings to runtime types.

#include "config.hpp"
#include "volumetric_kit/recon/codec/encoder.hpp"
#include "volumetric_kit/recon/tsdf/mesh_integrator.hpp"

namespace vr_example {

/// @return Runtime encoder settings; all values and weights come from core.
inline volumetric_kit::recon::codec::EncoderConfig encoder_config(
    const config::Codec& settings) {
  namespace codec = volumetric_kit::recon::codec;
  codec::EncoderConfig result;
  result.params.coefficient_count = settings.coefficient_count;
  result.params.quantization_scale = settings.quantization_scale;
  result.params.quantization_weights = settings.quantization_weights;
  result.segment_size = settings.segment_size;
  switch (settings.entropy) {
    case config::Entropy::Auto:
      result.entropy = codec::EntropyCoding::kAuto;
      break;
    case config::Entropy::Host:
      result.entropy = codec::EntropyCoding::kHost;
      break;
    case config::Entropy::Device:
      result.entropy = codec::EntropyCoding::kDevice;
      break;
  }
  return result;
}

// TODO(examples): CodecOutputs from core 0.5.0's output.csv and
// output.frame_directory, for codec_replica's and codec_mesh's
// CodecStream::create; until then they write no rows or frames.

/// @return The selected table's name for the measurement report.
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
