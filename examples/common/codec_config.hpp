// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file examples/common/codec_config.hpp
/// @brief Map the core's validated codec and mesh settings to runtime types.

#include <cstdio>
#include <string>

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
///         InvalidArgument from `EncoderConfig::validate` for what core
///         leaves to the codec: a kept coefficient's step outside its
///         bounds, or a segment or update size past its longest.
inline volumetric_kit::core::Result<volumetric_kit::recon::codec::EncoderConfig>
encoder_config(const config::Codec& settings) {
  namespace codec = volumetric_kit::recon::codec;
  codec::EncoderConfig result;
  result.params.coefficient_count = settings.coefficient_count;
  result.params.quantization_scale = settings.quantization_scale;
  VKC_TRY(apply_quantization_table(
      result.params, quantization_name(settings.quantization_table)));
  result.segment_size = settings.segment_size;
  result.update_size = settings.update_size;
  result.intra_period = settings.intra_period;
  VKC_TRY(result.validate());
  return result;
}

/// @return How @p config codes, as the codec examples print it: K, the
///         quantization @p table and scale, the segment size, and the intra
///         period, above 1 with the update size its P-frames use.
inline std::string codec_settings(
    const volumetric_kit::recon::codec::EncoderConfig& config,
    config::QuantizationTable table) {
  char text[128];
  std::snprintf(text, sizeof text,
                "K %u, table %s, scale %.6g; segments of %u blocks",
                config.params.coefficient_count, quantization_name(table),
                double(config.params.quantization_scale), config.segment_size);
  const std::string settings = text;
  if (config.intra_period == 1) return settings + "; every frame intra";
  std::snprintf(text, sizeof text,
                ", update segments of %u entries; an I-frame every %u frames",
                config.resolved_update_size(), config.intra_period);
  return settings + text;
}

/// @return Where a codec stream writes besides its report: core's
///         `output.csv` and `output.stream`, already relative to the
///         scenario's directory, an empty path writing nothing; and
///         `output.serve`, every interface on that port, 0 serving nothing.
inline CodecOutputs codec_outputs(const config::Output& output) {
  CodecOutputs outputs{output.csv, output.stream};
  if (output.serve != 0) {
    outputs.serve.emplace();  // every interface
    outputs.serve->port = output.serve;
  }
  return outputs;
}

/// @return Runtime SDF parameters from core's validated mesh source, or
///         InvalidArgument from `MeshSdfParams::validate` against @p grid:
///         the shell's bound, which core leaves to the mesh conversion.
inline volumetric_kit::core::Result<volumetric_kit::recon::tsdf::MeshSdfParams>
mesh_sdf_config(const config::Source& source,
                const volumetric_kit::recon::volume::VoxelGridParams& grid) {
  namespace tsdf = volumetric_kit::recon::tsdf;
  tsdf::MeshSdfParams result;
  result.mode = source.sdf_mode == config::SdfMode::Signed
                    ? tsdf::MeshSdfMode::Signed
                    : tsdf::MeshSdfMode::Shell;
  result.shell_voxels = source.shell_voxels;
  VKC_TRY(result.validate(grid));
  return result;
}

}  // namespace vr_example
