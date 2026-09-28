// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

// The codec's rate-distortion sweep: encode one grid under a list of
// configurations, decode each, mesh it, and print bytes against the decoded
// surface's distance from a reference mesh. The table the codec's defaults
// are chosen from (the 2026-09-27 decision).

#include <cstdint>
#include <cstdio>
#include <vector>

#include "codec_stream.hpp"
#include "volumetric_kit/recon/codec/codec_params.hpp"
#include "volumetric_kit/recon/codec/decoder.hpp"
#include "volumetric_kit/recon/codec/encoder.hpp"
#include "volumetric_kit/recon/eval/mesh_distance.hpp"
#include "volumetric_kit/recon/mesh/marching_cubes.hpp"
#include "volumetric_kit/recon/mesh/mesh.hpp"

namespace vr_example {

/// One configuration of the sweep.
struct SweepConfig {
  const char* family;
  std::uint32_t k;
  float dc;
  float ac;
};

/// @brief The sweep's configurations, in three families:
///  - "split": the prior engine's DC 5x AC, across K and across step scale;
///  - "uniform": DC = AC, across step at K from 32 to all 512;
///  - "ref": K = 512 at a fine step, near-lossless -- whatever error it still
///    shows is not the quantizer's.
inline std::vector<SweepConfig> sweep_configs() {
  std::vector<SweepConfig> c;
  for (std::uint32_t k : {8u, 16u, 32u, 64u, 128u}) {
    c.push_back({"split", k, 0.25f, 0.05f});
  }
  for (float s : {0.5f, 2.0f, 4.0f}) {
    c.push_back({"split", 32, 0.25f * s, 0.05f * s});
  }
  for (float step : {0.025f, 0.05f, 0.1f, 0.2f, 0.4f}) {
    c.push_back({"uniform", 32, step, step});
  }
  for (float step : {0.1f, 0.2f, 0.4f}) {
    c.push_back({"uniform", 64, step, step});
  }
  for (float step : {0.2f, 0.4f, 0.8f}) {
    c.push_back({"uniform", 128, step, step});
  }
  for (float step : {0.4f, 0.8f}) {
    c.push_back({"uniform", 256, step, step});
  }
  for (float step : {0.8f, 1.6f}) {
    c.push_back({"uniform", vr::codec::kVoxelsPerBlock, step, step});
  }
  c.push_back({"ref", vr::codec::kVoxelsPerBlock, 0.002f, 0.002f});
  return c;
}

/// @brief Run the sweep over @p source against @p reference (its mesh,
///        indexed once), and print the table.
///
/// Each configuration runs one untimed round first: a fresh encoder's first
/// dispatches pay one-off pipeline costs that would otherwise be read as the
/// configuration's. The timed decode is of a frame the player already holds,
/// a static scene's steady state.
///
/// Accuracy's RMS, p95 and max are over the decoded vertices within `reach`
/// of the source, so the max is under `reach` by construction; `acc>r`
/// counts the vertices past it -- floaters -- and `cov>r` the source
/// vertices with no decoded surface within it -- holes.
///
/// @param player  The grid to decode into: a stream's player, already built
///                for this geometry (it grows if a frame needs it).
inline vr::Status run_codec_sweep(vr::Device& device, vr::Allocator& allocator,
                                  vr::volume::VoxelBlockGrid& source,
                                  const vr::eval::ReferenceMesh& reference,
                                  vr::mesh::MarchingCubes& extractor,
                                  vr::volume::VoxelBlockGrid& player) {
  VR_ASSIGN(vr::codec::Decoder dec,
            vr::codec::Decoder::create(device, allocator));
  int grows = 0;
  const float tau = reference.options().fscore_threshold;
  if (tau > 0.0f) {
    std::printf(
        "sweep (steps are fractions of trunc_dist; distances in mm; "
        "F at %.1f mm):\n",
        double(tau) * 1e3);
  } else {
    std::printf(
        "sweep (steps are fractions of trunc_dist; distances in mm; "
        "no F-score asked for):\n");
  }
  std::printf(
      "  %-8s %4s %6s %6s | %8s %7s %6s | %7s %7s %7s %6s | %7s %6s | %6s | "
      "%7s %7s\n",
      "family", "K", "dc", "ac", "bytes", "B/block", "ratio", "acc rms",
      "acc p95", "acc max", "acc>r", "cov rms", "cov>r", "F", "enc ms",
      "dec ms");
  for (const SweepConfig& cfg : sweep_configs()) {
    vr::codec::EncoderConfig ec;
    ec.params.coefficient_count = cfg.k;
    ec.params.dc_step = cfg.dc;
    ec.params.ac_step = cfg.ac;
    VR_ASSIGN(vr::codec::Encoder enc,
              vr::codec::Encoder::create(device, allocator, ec));
    VR_ASSIGN(const Bytes warm, enc.encode(source));
    VR_TRY(decode_growing(dec, warm, player, nullptr, &grows));

    vr::StageMetrics enc_rows;
    vr::StageMetrics dec_rows;
    VR_ASSIGN(const Bytes frame, enc.encode(source, &enc_rows));
    VR_TRY(decode_growing(dec, frame, player, &dec_rows, &grows));
    VR_ASSIGN(const vr::codec::FrameInfo info,
              vr::codec::read_frame_info(frame.data(), frame.size()));
    VR_ASSIGN(const vr::mesh::Mesh decoded, extractor.extract_host(player));
    VR_ASSIGN(const vr::eval::MeshComparison c, reference.compare(decoded));
    const double per_block =
        info.block_count > 0 ? double(frame.size()) / info.block_count : 0.0;
    char f[16] = "-";  // not measured, which a 0 would misread as
    if (tau > 0.0f) {
      std::snprintf(f, sizeof f, "%.4f", c.fscore.f);
    }
    std::printf(
        "  %-8s %4u %6.3f %6.3f | %8zu %7.1f %5.0fx | %7.3f %7.3f %7.3f %6zu | "
        "%7.3f %6zu | %6s | %7.2f %7.2f\n",
        cfg.family, cfg.k, double(cfg.dc), double(cfg.ac), frame.size(),
        per_block, per_block > 0 ? kRawBytesPerBlock / per_block : 0.0,
        c.accuracy.rms * 1e3, c.accuracy.p95 * 1e3, c.accuracy.max * 1e3,
        c.accuracy.beyond_reach, c.coverage.rms * 1e3, c.coverage.beyond_reach,
        f,
        row_ms(enc_rows, "codec encode"), row_ms(dec_rows, "codec decode"));
  }
  return {};
}

}  // namespace vr_example
