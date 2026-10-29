// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// The integration boundary: core-validated settings reach the runtime options
// unchanged -- the stream's outputs, decoding for a sender only with a CSV or
// a sweep, and its session among them -- and the codec checks the steps and
// sizes core leaves to it; the codec sweep codes every row intra. The fusing
// examples' templates are recon's (its example_fuse_config_test.cpp). No file
// assets, device or camera are opened.
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <string>
#include <tuple>
#include <utility>

#include "codec_config.hpp"
#include "codec_sweep.hpp"
#include "volumetric_kit/grid/networking/stream_server.hpp"
#include "volumetric_kit/grid/volume/voxel_grid.hpp"

namespace cfg = volumetric_kit::core::config;
namespace vgrid = volumetric_kit::grid;
namespace vkc = volumetric_kit::core;

#define CHECK(condition)                                           \
  do {                                                             \
    if (!(condition)) {                                            \
      std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition); \
      return 1;                                                    \
    }                                                              \
  } while (false)

int main() {
  struct Example {
    const char* file;
    cfg::Application application;
  };
  const Example examples[] = {
      {"codec_mesh", cfg::Application::CodecMesh},
      {"codec_mesh_sequence", cfg::Application::CodecMesh},
  };
  for (const auto& example : examples) {
    const auto loaded = cfg::load(
        std::string(VGRID_CONFIG_FIXTURES) + "/" + example.file + ".json",
        example.application);
    if (!loaded)
      std::fprintf(stderr, "%s: %s\n", example.file,
                   loaded.status().message().c_str());
    CHECK(loaded);
    // A mesh fuses Classic, writes no codec rows or frames and serves nothing:
    // codec_mesh's templates write the PLYs alone.
    CHECK(!loaded->fusion.dynamic);
    CHECK(loaded->output.csv.empty() && loaded->output.stream.empty());
    CHECK(loaded->output.serve == 0);
    const vgrid::codec_stream::CodecOutputs outputs =
        vgrid_example::codec_outputs(loaded.value().output);
    CHECK(!outputs.serve.has_value());
    // No scenario codes a true SDF: codec_mesh alone sets it, in code.
    const auto encoder = vgrid_example::encoder_config(loaded->codec);
    CHECK(encoder.ok() && !encoder->true_sdf);
  }

  // The single-mesh template stays one frame, every frame intra; the
  // sequence's codes Rafa2's frames 1-150, holes admitted. Both convert in a
  // 16-voxel band, past kTrueSdfBandVoxels, and code at K 56 with the
  // study's step (DECISIONS 2026-10-21).
  // The sequence's template codes at period 30 with U = R, as the freeze
  // measured it (DECISIONS, the 2026-10-17 freeze), as recon's codec_replica
  // does.
  const auto single =
      cfg::load(std::string(VGRID_CONFIG_FIXTURES) + "/codec_mesh.json",
                cfg::Application::CodecMesh);
  CHECK(single && single->run.first == 0 && single->run.frames == 1);
  CHECK(single->codec.intra_period == 1);
  const auto sequence = cfg::load(
      std::string(VGRID_CONFIG_FIXTURES) + "/codec_mesh_sequence.json",
      cfg::Application::CodecMesh);
  CHECK(sequence);
  CHECK(sequence->source.path ==
        (std::filesystem::path(VGRID_CONFIG_FIXTURES) / "data" / "Rafa2_obj" /
         "Frame_%05d_textured.obj")
            .lexically_normal()
            .string());
  CHECK(sequence->run.first == 1 && sequence->run.frames == 150);
  for (const cfg::Config* mesh : {&single.value(), &sequence.value()}) {
    CHECK(mesh->fusion.voxel_size_m == 0.005F &&
          mesh->fusion.truncation_m == 0.08F);
    CHECK(mesh->codec.coefficient_count == 56 &&
          mesh->codec.quantization_scale == 0.040877F);
  }
  CHECK(sequence->source.height_m == 1.7 &&
        sequence->source.metres_per_unit == 0.0);
  CHECK(sequence->source.allow_holes && !sequence->codec.sweep &&
        sequence->output.prefix.empty());
  CHECK(sequence->codec.intra_period == 30 && sequence->codec.update_size == 0);
  const auto sequence_encoder = vgrid_example::encoder_config(sequence->codec);
  CHECK(sequence_encoder.ok() && sequence_encoder->intra_period == 30);
  CHECK(sequence_encoder->segment_size == 64 &&
        sequence_encoder->resolved_update_size() ==
            sequence_encoder->segment_size);

  // Any port maps to every interface on it, keys on demand (fuse_orbbec codes
  // live), the scenario's default viewer count the server's own, the server's
  // other options at their defaults; serving alone needs no file.
  const auto on_port = cfg::parse(R"({
    "version": 1, "operation": "fuse",
    "source": {"kind": "live", "sync_config": "sync.json"},
    "output": {"path": "rig.ply", "serve": 5000}
  })",
                                  cfg::Application::FuseOrbbec);
  CHECK(on_port);
  const vgrid::codec_stream::CodecOutputs to_port =
      vgrid_example::codec_outputs(on_port.value().output);
  const vgrid::networking::ServerOptions defaults_served;
  CHECK(on_port->output.csv.empty() && to_port.stream.empty());
  CHECK(to_port.serve && to_port.serve->host.empty() &&
        to_port.serve->port == 5000);
  CHECK(!to_port.serve->replay_gop && defaults_served.replay_gop);
  CHECK(on_port->output.max_viewers == defaults_served.max_viewers &&
        to_port.serve->max_viewers == defaults_served.max_viewers &&
        to_port.serve->max_gop_bytes == defaults_served.max_gop_bytes &&
        to_port.serve->stall_limit == defaults_served.stall_limit);
  // The scenario's viewer count goes to the server as it is, below the
  // default or above it.
  for (const unsigned viewers : {1U, 16U}) {
    const auto counted = cfg::parse(
        R"({"version": 1, "operation": "fuse",
            "source": {"kind": "live", "sync_config": "sync.json"},
            "output": {"path": "rig.ply", "serve": 5000, "max_viewers": )" +
            std::to_string(viewers) + "}}",
        cfg::Application::FuseOrbbec);
    CHECK(counted);
    const vgrid::codec_stream::CodecOutputs to_counted =
        vgrid_example::codec_outputs(counted.value().output);
    CHECK(to_counted.serve && to_counted.serve->port == 5000 &&
          to_counted.serve->max_viewers == viewers);
  }

  const auto encoded =
      cfg::parse(R"({
    "version": 1, "operation": "codec", "source": {"kind": "dataset", "path": "room0"},
    "codec": {"coefficient_count": 128, "quantization_scale": 0.3,
              "quantization_table": "radial", "segment_size": 32,
              "update_size": 128, "intra_period": 30, "encode_every": 0,
              "sweep": true},
    "output": {"csv": "results/rows.csv", "stream": "../streams/room0.vrst"}
  })",
                 cfg::Application::CodecReplica, "/tmp/scenarios");
  CHECK(encoded);
  const auto encoder = vgrid_example::encoder_config(encoded.value().codec);
  CHECK(encoder.ok());
  CHECK(encoder->params.coefficient_count == 128);
  CHECK(encoder->params.quantization_scale == 0.3F);
  // The radial table's weights (codec_quantization.hpp), 1 at DC.
  CHECK(encoder->params.quantization_weights[0] == 1.F);
  CHECK(encoder->params.quantization_weights[1] == 1.125F);
  CHECK(encoder->params.quantization_weights[2] == 1.5F);
  CHECK(encoder->params.quantization_weights[1 + 8 * 2 + 64 * 3] == 2.75F);
  CHECK(encoder->segment_size == 32);
  CHECK(encoder->update_size == 128 && encoder->intra_period == 30);
  CHECK(vgrid_example::codec_settings(*encoder,
                                      encoded->codec.quantization_table) ==
        "K 128, table radial, scale 0.3; segments of 32 blocks, update "
        "segments of 128 entries; an I-frame every 30 frames");
  // The sweep times repeated encodes of one grid: every row intra, with the
  // scenario's sizes and the row's params.
  const auto row =
      vgrid_example::sweep_encoder_config(*encoder, {"band", 84, 0.4F});
  CHECK(row.ok() && row->intra_period == 1);
  CHECK(row->segment_size == 32 && row->update_size == 128);
  CHECK(row->params.coefficient_count == 84 &&
        row->params.quantization_scale == 0.4F);
  CHECK(row->params.quantization_weights[1] == 1.25F);  // band's, (1, 0, 0)
  CHECK(
      !vgrid_example::sweep_encoder_config(*encoder, {"none", 84, 0.4F}).ok());
  // The outputs, relative to the scenario's directory; the CSV is the
  // report's.
  const vgrid::codec_stream::CodecOutputs outputs =
      vgrid_example::codec_outputs(encoded.value().output);
  CHECK(encoded->output.csv == "/tmp/scenarios/results/rows.csv");
  CHECK(outputs.stream == "/tmp/streams/room0.vrst");
  CHECK(!outputs.serve && outputs.decode);

  // A sender's stream decodes each frame only with output.csv or
  // codec.sweep, which read the decode, not for a recording or a server
  // alone; served live, without replay_gop.
  cfg::Config live_out;
  live_out.output.stream = "rig.vrst";
  live_out.output.serve = 47801;
  const vgrid::codec_stream::CodecOutputs live_sent =
      vgrid_example::set_codec_outputs(live_out);
  CHECK(!live_sent.decode && live_sent.stream == "rig.vrst");
  CHECK(live_sent.serve && live_sent.serve->port == 47801 &&
        !live_sent.serve->replay_gop);
  cfg::Config listed = live_out;
  listed.output.csv = "rig.csv";
  CHECK(vgrid_example::set_codec_outputs(listed).decode);
  cfg::Config swept = live_out;
  swept.codec.sweep = true;
  CHECK(vgrid_example::set_codec_outputs(swept).decode);
  // The stream's session: the program, and one Geometry track at the
  // source's rate whose params are the given bytes; no id or start time.
  const vgrid::stream::Session session =
      vgrid_example::codec_session("codec_mesh", {30, 1}, "placed");
  CHECK(session.writer == "codec_mesh" && session.tracks.size() == 1);
  CHECK(session.created_ns == 0 &&
        session.id == (std::array<std::uint8_t, 16>{}));
  CHECK(session.tracks[0].kind == vgrid::stream::TrackKind::Geometry);
  CHECK(session.tracks[0].rate.num == 30 && session.tracks[0].rate.den == 1);
  CHECK(session.tracks[0].params == "placed");

  // Both codec profiles: the period as given, and with P-frames the update
  // size as given, core's 0 included, up to the format's longest segment;
  // left out, the P-frames' update segments are the segment size, as printed.
  const vgrid::codec::EncoderConfig defaults;
  for (const auto& [source_json, application] :
       {std::pair{R"({"kind": "dataset", "path": "room0"})",
                  cfg::Application::CodecReplica},
        std::pair{R"({"kind": "mesh", "path": "shape.obj",
                      "metres_per_unit": 1, "up_axis": "y"})",
                  cfg::Application::CodecMesh}}) {
    const auto with_codec = [&, source = source_json,
                             app = application](const std::string& codec_json) {
      return cfg::parse(std::string(R"({"version": 1, "operation": "codec",
                                        "source": )") +
                            source + R"(, "codec": )" + codec_json + "}",
                        app);
    };
    const auto unset = with_codec("{}");
    CHECK(unset);
    const auto plain = vgrid_example::encoder_config(unset->codec);
    CHECK(plain.ok() && plain->intra_period == 1 &&
          plain->update_size == defaults.update_size);
    CHECK(vgrid_example::codec_settings(*plain,
                                        unset->codec.quantization_table) ==
          "K 64, table uniform, scale 0.2; segments of 64 blocks; every frame "
          "intra");
    for (const std::uint32_t period : {1u, 8u, 4294967295u}) {
      const auto settings =
          with_codec(R"({"intra_period": )" + std::to_string(period) + "}");
      CHECK(settings);
      const auto configured = vgrid_example::encoder_config(settings->codec);
      CHECK(configured.ok() && configured->intra_period == period);
    }
    for (const std::uint32_t size :
         {0u, 1u, 5u, vgrid::codec::kMaxDeviceDecodeSegmentSize,
          vgrid::codec::kMaxDeviceDecodeSegmentSize + 1}) {
      const auto settings =
          with_codec(R"({"intra_period": 2, "update_size": )" +
                     std::to_string(size) + "}");
      CHECK(settings);
      const auto configured = vgrid_example::encoder_config(settings->codec);
      if (size > vgrid::codec::kMaxDeviceDecodeSegmentSize) {
        CHECK(!configured.ok() && configured.status().domain() ==
                                      vkc::Status::Code::InvalidArgument);
        CHECK(configured.status().message().find(
                  "update_size " + std::to_string(size)) != std::string::npos);
      } else {
        CHECK(configured.ok() && configured->update_size == size);
      }
    }
    const auto segments =
        with_codec(R"({"segment_size": 16, "intra_period": 30})");
    CHECK(segments);
    const auto by_segment = vgrid_example::encoder_config(segments->codec);
    CHECK(by_segment.ok() && by_segment->resolved_update_size() == 16);
    CHECK(vgrid_example::codec_settings(*by_segment,
                                        segments->codec.quantization_table) ==
          "K 64, table uniform, scale 0.2; segments of 16 blocks, update "
          "segments of 16 entries; an I-frame every 30 frames");
    // No P-frame uses an update size at a period of 1, so core refuses one
    // given there, even 0: such a scenario maps only to the encoder's default.
    for (const char* codec_json :
         {R"({"update_size": 16})",
          R"({"intra_period": 1, "update_size": 0})"}) {
      const auto refused = with_codec(codec_json);
      CHECK(!refused && refused.status().message().find("codec.update_size") !=
                            std::string::npos);
    }
  }

  // Steps out of the codec's bounds load, and the codec refuses them: too
  // fine everywhere, or too coarse at radial's highest frequency.
  for (const char* codec_section :
       {R"({"quantization_scale": 0.0001})",
        R"({"quantization_table": "radial", "coefficient_count": 512,
            "quantization_scale": 4})"}) {
    const auto out_of_bounds =
        cfg::parse(std::string(R"({"version": 1, "operation": "codec",
                        "source": {"kind": "dataset", "path": "room0"},
                        "codec": )") +
                       codec_section + "}",
                   cfg::Application::CodecReplica);
    CHECK(out_of_bounds);
    const auto refused = vgrid_example::encoder_config(out_of_bounds->codec);
    CHECK(!refused.ok() &&
          refused.status().domain() == vkc::Status::Code::InvalidArgument);
  }
  const auto mesh = cfg::parse(R"({
    "version": 1, "operation": "codec",
    "source": {"kind": "mesh", "path": "person.obj", "height_m": 1.7,
               "up_vector": [-0.9120591159, 0.0661250017, -0.4046920474],
               "sdf_mode": "shell", "shell_voxels": 2.0}
  })",
                               cfg::Application::CodecMesh);
  CHECK(mesh);
  const auto grid = [](const cfg::Fusion& fusion) {
    return vgrid::volume::block_grid_params(fusion.voxel_size_m,
                                            fusion.truncation_m, 1024);
  };
  const auto sdf =
      vgrid_example::mesh_sdf_config(mesh.value().source, grid(mesh->fusion));
  CHECK(sdf.ok());
  CHECK(sdf->mode == vgrid::voxelize::MeshSdfMode::Shell &&
        sdf->shell_voxels == 2.F);
  CHECK(mesh.value().source.up_vector.has_value());
  CHECK((*mesh.value().source.up_vector)[0] < -0.9);
  // Core loads any positive shell; the conversion's bound is grid's, with
  // the truncation the scenario resolves: 5 voxels is refused within the
  // default 4-voxel band and taken within an 8-voxel one.
  for (const auto& [truncation, accepted] :
       {std::pair{"", false}, std::pair{R"(, "truncation_m": 0.04)", true}}) {
    const auto thick =
        cfg::parse(std::string(R"({"version": 1, "operation": "codec",
            "source": {"kind": "mesh", "path": "person.obj", "height_m": 1.7,
                       "up_axis": "y", "sdf_mode": "shell",
                       "shell_voxels": 5.0},
            "fusion": {"voxel_size_m": 0.005)") +
                       truncation + "}}",
                   cfg::Application::CodecMesh);
    CHECK(thick);
    const auto shell =
        vgrid_example::mesh_sdf_config(thick->source, grid(thick->fusion));
    CHECK(shell.ok() == accepted);
    CHECK(accepted ||
          (shell.status().domain() == vkc::Status::Code::InvalidArgument &&
           shell.status().message().find("shell_voxels") != std::string::npos));
  }

  // The true-SDF band: a whole 15 voxels is taken at every voxel size from 1
  // to 50 mm, though two floats' ratio can fall short of it (0.06 m of 4 mm
  // voxels), and 14 refused; shell mode adds the half-thickness. Both mesh
  // templates pass.
  vgrid::voxelize::MeshSdfParams solid;
  vgrid::voxelize::MeshSdfParams shelled;
  shelled.mode = vgrid::voxelize::MeshSdfMode::Shell;
  shelled.shell_voxels = 2.F;
  for (int mm = 1; mm <= 50; ++mm) {
    const auto band = [mm](int voxels) {
      return vgrid::volume::block_grid_params(
          float(mm / 1000.0), float(voxels * mm / 1000.0), 1024);
    };
    CHECK(vgrid_example::check_true_sdf_band(solid, band(15)).ok());
    CHECK(vgrid_example::check_true_sdf_band(shelled, band(17)).ok());
    for (const auto& [sdf, voxels, least] :
         {std::tuple{&solid, 14, "at least 15 voxels"},
          std::tuple{&shelled, 16, "at least 17 voxels"}}) {
      const vkc::Status refused =
          vgrid_example::check_true_sdf_band(*sdf, band(voxels));
      CHECK(refused.domain() == vkc::Status::Code::InvalidArgument &&
            refused.message().find(least) != std::string::npos);
    }
  }
  for (const cfg::Config* mesh : {&single.value(), &sequence.value()}) {
    const auto sdf =
        vgrid_example::mesh_sdf_config(mesh->source, grid(mesh->fusion));
    CHECK(sdf.ok() &&
          vgrid_example::check_true_sdf_band(*sdf, grid(mesh->fusion)).ok());
  }
  return 0;
}
