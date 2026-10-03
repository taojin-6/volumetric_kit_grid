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

#include "codec_quantization.hpp"
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
  float scale;
};

/// @brief Compare uniform, total-frequency-band and per-basis radial tables.
///
/// K = 64 is the existing cutoff. 84 and 120 retain complete total-frequency
/// shells (through degree 6 and 7); 512 isolates quantization from truncation.
/// Every size includes the frame's complete quantization table. The final
/// uniform fine-step row is a near-lossless control, not an error-free field.
inline std::vector<SweepConfig> sweep_configs() {
  std::vector<SweepConfig> c;
  for (std::uint32_t k : {64u, 84u, 120u, 512u}) {
    for (const char* family : {"uniform", "band", "radial"}) {
      for (float scale : {0.05f, 0.1f, 0.2f, 0.4f, 0.8f}) {
        c.push_back({family, k, scale});
      }
    }
  }
  c.push_back({"uniform", vr::codec::kVoxelsPerBlock, 0.002f});
  return c;
}

/// @brief Run the sweep over @p source against @p reference (its mesh,
///        indexed once), and print the table.
///
/// Each configuration runs one untimed round first: a fresh encoder's first
/// dispatches pay one-off pipeline costs that would otherwise be read as the
/// configuration's. Three timed rounds are averaged. The timed decode is of
/// a frame the player already holds, a static scene's steady state. Mesh
/// extraction and quality evaluation are outside the codec timings.
///
/// Accuracy's RMS, p95 and max are over the decoded vertices within `reach`
/// of the source, so the max is under `reach` by construction; `acc>r`
/// counts the vertices past it -- floaters -- and `cov>r` the source
/// vertices with no decoded surface within it -- holes.
///
/// @param base    The command line's configuration: each row keeps its
///                entropy coding and segment size, and sets the params.
/// @param player  The grid to decode into: a stream's player, already built
///                for this geometry (it grows if a frame needs it).
inline vr::Status run_codec_sweep(vr::Device& device, vr::Allocator& allocator,
                                  vr::volume::VoxelBlockGrid& source,
                                  const vr::eval::ReferenceMesh& reference,
                                  vr::mesh::MarchingCubes& extractor,
                                  const vr::codec::EncoderConfig& base,
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
      "  %-8s %4s %6s | %8s %7s %6s | %7s %7s %7s %6s | %7s %6s | %6s | "
      "%7s %7s %7s %7s\n",
      "table", "K", "scale", "bytes", "B/block", "ratio", "acc rms", "acc p95",
      "acc max", "acc>r", "cov rms", "cov>r", "F", "enc ms", "dec ms",
      "enc gpu", "dec gpu");
  for (const SweepConfig& cfg : sweep_configs()) {
    vr::codec::EncoderConfig ec = base;
    ec.params = vr::codec::CodecParams{};
    ec.params.coefficient_count = cfg.k;
    ec.params.quantization_scale = cfg.scale;
    VR_TRY(apply_quantization_table(ec.params, cfg.family));
    VR_ASSIGN(vr::codec::Encoder enc,
              vr::codec::Encoder::create(device, allocator, ec));
    VR_ASSIGN(const Bytes warm, enc.encode(source));
    VR_TRY(decode_growing(dec, warm, player, nullptr, &grows));

    vr::StageMetrics enc_rows;
    vr::StageMetrics dec_rows;
    constexpr int kTimedRounds = 3;
    Bytes frame;
    for (int repeat = 0; repeat < kTimedRounds; ++repeat) {
      VR_ASSIGN(frame, enc.encode(source, &enc_rows));
      VR_TRY(decode_growing(dec, frame, player, &dec_rows, &grows));
    }
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
    const auto gpu_ms = [](const vr::StageMetrics& rows, const char* name) {
      for (const vr::StageRow& r : rows.rows()) {
        if (std::strcmp(r.name, name) == 0 && r.has_gpu) return r.gpu_ms;
      }
      return -1.0;
    };
    char enc_gpu[16] = "-";
    char dec_gpu[16] = "-";
    const double enc_device = gpu_ms(enc_rows, "codec encode");
    const double dec_device = gpu_ms(dec_rows, "codec decode");
    if (enc_device >= 0.0) {
      std::snprintf(enc_gpu, sizeof enc_gpu, "%.2f", enc_device / kTimedRounds);
    }
    if (dec_device >= 0.0) {
      std::snprintf(dec_gpu, sizeof dec_gpu, "%.2f", dec_device / kTimedRounds);
    }
    std::printf(
        "  %-8s %4u %6.3f | %8zu %7.2f %5.0fx | %7.3f %7.3f %7.3f %6zu | "
        "%7.3f %6zu | %6s | %7.2f %7.2f %7s %7s\n",
        cfg.family, cfg.k, double(cfg.scale), frame.size(), per_block,
        per_block > 0 ? kRawBytesPerBlock / per_block : 0.0,
        c.accuracy.rms * 1e3, c.accuracy.p95 * 1e3, c.accuracy.max * 1e3,
        c.accuracy.beyond_reach, c.coverage.rms * 1e3, c.coverage.beyond_reach,
        f, row_ms(enc_rows, "codec encode") / kTimedRounds,
        row_ms(dec_rows, "codec decode") / kTimedRounds, enc_gpu, dec_gpu);
    std::fflush(stdout);
  }
  return {};
}

}  // namespace vr_example
