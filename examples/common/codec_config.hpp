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
///         InvalidArgument from `CodecParams::validate` for a kept
///         coefficient's step outside the codec's bounds, which core leaves
///         to the codec, or for `codec.intra_period` other than 1 until the
///         examples support P-frame records.
inline volumetric_kit::core::Result<volumetric_kit::recon::codec::EncoderConfig>
encoder_config(const config::Codec& settings) {
  namespace codec = volumetric_kit::recon::codec;
  // TODO(examples): C5b maps the period and writes record keys from FrameInfo.
  if (settings.intra_period != 1) {
    return volumetric_kit::core::Status::invalid_argument(
        "codec.intra_period: only 1 is supported until the examples support "
        "P-frame records");
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
