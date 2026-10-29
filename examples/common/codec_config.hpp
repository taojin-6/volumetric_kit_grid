// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file examples/common/codec_config.hpp
/// @brief Map the core's validated codec and mesh settings to runtime types.

#include <cstdio>
#include <string>
#include <utility>

#include "codec_quantization.hpp"
#include "config.hpp"
#include "volumetric_kit/core/base/result.hpp"
#include "volumetric_kit/recon/codec/encoder.hpp"
#include "volumetric_kit/recon/sender/codec_stream.hpp"
#include "volumetric_kit/recon/stream/session.hpp"
#include "volumetric_kit/recon/voxelize/mesh_integrator.hpp"

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
///         quantization @p table and scale, the segment size, the intra
///         period, above 1 with the update size its P-frames use, and
///         whether it codes a true SDF.
inline std::string codec_settings(
    const volumetric_kit::recon::codec::EncoderConfig& config,
    config::QuantizationTable table) {
  char text[128];
  std::snprintf(text, sizeof text,
                "K %u, table %s, scale %.6g; segments of %u blocks",
                config.params.coefficient_count, quantization_name(table),
                double(config.params.quantization_scale), config.segment_size);
  std::string settings = text;
  if (config.intra_period == 1) {
    settings += "; every frame intra";
  } else {
    std::snprintf(text, sizeof text,
                  ", update segments of %u entries; an I-frame every %u frames",
                  config.resolved_update_size(), config.intra_period);
    settings += text;
  }
  return config.true_sdf ? settings + "; true SDF" : settings;
}

/// @return Where a codec stream's frames go (`sender::CodecOutputs`): core's
///         `output.stream`, already relative to the scenario's directory, an
///         empty path writing nothing; and `output.serve`, every interface on
///         that port, 0 serving nothing, without `replay_gop`: the stream
///         codes live, so a viewer that joins or falls behind gets a key at
///         the next frame. `output.csv` is the report's (`CodecReport`).
inline volumetric_kit::recon::sender::CodecOutputs codec_outputs(
    const config::Output& output) {
  volumetric_kit::recon::sender::CodecOutputs outputs;
  outputs.stream = output.stream;
  if (output.serve != 0) {
    outputs.serve.emplace();  // every interface
    outputs.serve->port = output.serve;
    outputs.serve->replay_gop = false;
  }
  return outputs;
}

/// @return The outputs of @p opt's sender's stream (@ref codec_outputs),
///         decoding each frame only with `output.csv` or `codec.sweep`,
///         which read the decode: a recording or a server alone needs none.
inline volumetric_kit::recon::sender::CodecOutputs set_codec_outputs(
    const config::Config& opt) {
  volumetric_kit::recon::sender::CodecOutputs outputs =
      codec_outputs(opt.output);
  outputs.decode = !opt.output.csv.empty() || opt.codec.sweep;
  return outputs;
}

/// @return The session a codec example's stream writes and serves: its
///         writer @p program and one Geometry track at @p rate, the source's,
///         whose params are @p geometry_params, empty or a mesh source's
///         `io::encode_transform` bytes (stream/session.hpp). No session id
///         or start time, so a rerun writes the same bytes.
inline volumetric_kit::recon::stream::Session codec_session(
    const std::string& program, volumetric_kit::recon::stream::Rate rate,
    std::string geometry_params = {}) {
  namespace stream = volumetric_kit::recon::stream;
  stream::Track geometry;
  geometry.kind = stream::TrackKind::Geometry;
  geometry.rate = rate;
  geometry.params = std::move(geometry_params);
  stream::Session session;
  session.writer = program;
  session.tracks.push_back(std::move(geometry));
  return session;
}

/// @return OK when @p grid's band observes every voxel a true-SDF frame of a
///         mesh codes: `codec::kTrueSdfBandVoxels` voxels from the zero
///         level, which in shell mode lies @p sdf's half-thickness further
///         out; or InvalidArgument naming the minimum, for a caller that
///         refuses before any GPU work.
inline volumetric_kit::core::Status check_true_sdf_band(
    const volumetric_kit::recon::voxelize::MeshSdfParams& sdf,
    const volumetric_kit::recon::volume::VoxelGridParams& grid) {
  namespace codec = volumetric_kit::recon::codec;
  namespace voxelize = volumetric_kit::recon::voxelize;
  const bool shell = sdf.mode == voxelize::MeshSdfMode::Shell;
  const double least = double(codec::kTrueSdfBandVoxels) +
                       (shell ? double(sdf.shell_voxels) : 0.0);
  // A relative margin: two floats' ratio can fall just short of a whole band
  // (0.06 m of 4 mm voxels is 14.999999).
  const double band = double(grid.trunc_dist) / double(grid.voxel_size);
  if (band >= least * (1.0 - 1e-4)) return {};
  char text[256];
  std::snprintf(text, sizeof text,
                "fusion.truncation_m %.6g m is %.6g voxels; a mesh codes as a "
                "true SDF in at least %.6g voxels (%.6g m)%s",
                double(grid.trunc_dist), band, least,
                least * double(grid.voxel_size),
                shell ? ", the shell's half-thickness included" : "");
  return volumetric_kit::core::Status::invalid_argument(text);
}

/// @return Runtime SDF parameters from core's validated mesh source, or
///         InvalidArgument from `MeshSdfParams::validate` against @p grid:
///         the shell's bound, which core leaves to the mesh conversion.
inline volumetric_kit::core::Result<
    volumetric_kit::recon::voxelize::MeshSdfParams>
mesh_sdf_config(const config::Source& source,
                const volumetric_kit::recon::volume::VoxelGridParams& grid) {
  namespace voxelize = volumetric_kit::recon::voxelize;
  voxelize::MeshSdfParams result;
  result.mode = source.sdf_mode == config::SdfMode::Signed
                    ? voxelize::MeshSdfMode::Signed
                    : voxelize::MeshSdfMode::Shell;
  result.shell_voxels = source.shell_voxels;
  VKC_TRY(result.validate(grid));
  return result;
}

}  // namespace vr_example
