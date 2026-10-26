// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// The integration boundary: core-validated settings reach the runtime options
// unchanged, and the codec checks the steps and sizes core leaves to it; the
// codec sweep codes every row intra. No file assets, device or camera are
// opened.
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <string>
#include <tuple>
#include <utility>

#include "codec_config.hpp"
#include "codec_sweep.hpp"
#include "replica_config.hpp"
#include "volumetric_kit/recon/networking/stream_server.hpp"
#include "volumetric_kit/recon/volume/voxel_grid.hpp"
#ifdef VR_TEST_ORBBEC_CONFIG
#include "orbbec_config.hpp"
#endif

namespace cfg = volumetric_kit::core::config;
namespace vr = volumetric_kit::recon;
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
      {"fuse_replica", cfg::Application::FuseReplica},
      {"codec_replica", cfg::Application::CodecReplica},
      {"codec_mesh", cfg::Application::CodecMesh},
      {"codec_mesh_sequence", cfg::Application::CodecMesh},
      {"fuse_orbbec", cfg::Application::FuseOrbbec},
      {"codec_orbbec", cfg::Application::FuseOrbbec},
      {"serve_orbbec", cfg::Application::FuseOrbbec},
      {"stream_orbbec", cfg::Application::FuseOrbbec},
      {"readback_orbbec", cfg::Application::FuseOrbbec},
      {"dump_orbbec", cfg::Application::FuseOrbbec},
      {"fuse_render", cfg::Application::FuseRender},
      {"fuse_viewer", cfg::Application::FuseViewer},
      {"rig_viewer", cfg::Application::RigViewer},
  };
  for (const auto& example : examples) {
    const auto loaded = cfg::load(
        std::string(VR_CONFIG_FIXTURES) + "/" + example.file + ".json",
        example.application);
    if (!loaded)
      std::fprintf(stderr, "%s: %s\n", example.file,
                   loaded.status().message().c_str());
    CHECK(loaded);
    // codec_orbbec alone writes codec rows, and it and serve_orbbec frames;
    // serve_orbbec alone serves them, on the default port to every interface,
    // a joiner waiting for the key the stream makes for it.
    // Every live example fuses Dynamic, by core's default, the rest Classic.
    const bool coded = std::string(example.file) == "codec_orbbec";
    const bool served = std::string(example.file) == "serve_orbbec";
    const bool live = example.application == cfg::Application::FuseOrbbec ||
                      example.application == cfg::Application::RigViewer;
    CHECK(loaded->fusion.dynamic == live);
    CHECK(loaded->output.csv.empty() != coded);
    CHECK(loaded->output.stream.empty() != (coded || served));
    CHECK(loaded->output.serve == (served ? 47801 : 0));
    const vr_example::CodecOutputs outputs =
        vr_example::codec_outputs(loaded.value().output);
    CHECK(outputs.serve.has_value() == served);
    CHECK(!served || (outputs.serve->host.empty() &&
                      outputs.serve->port == vr::networking::kDefaultPort &&
                      !outputs.serve->replay_gop));
    // No scenario codes a true SDF: codec_mesh alone sets it, in code.
    const auto encoder = vr_example::encoder_config(loaded->codec);
    CHECK(encoder.ok() && !encoder->true_sdf);
  }

  // The single-mesh template stays one frame, every frame intra; the
  // sequence's codes Rafa2's frames 1-150, holes admitted. Both convert in a
  // 16-voxel band, past kTrueSdfBandVoxels, and code at K 56 with the
  // study's step (DECISIONS 2026-10-21).
  // The two sequences' templates code at period 30 with U = R, as the freeze
  // measured them (DECISIONS, the 2026-10-17 freeze): codec_replica every
  // frame, then its sweep.
  const auto single =
      cfg::load(std::string(VR_CONFIG_FIXTURES) + "/codec_mesh.json",
                cfg::Application::CodecMesh);
  CHECK(single && single->run.first == 0 && single->run.frames == 1);
  CHECK(single->codec.intra_period == 1);
  const auto sequence =
      cfg::load(std::string(VR_CONFIG_FIXTURES) + "/codec_mesh_sequence.json",
                cfg::Application::CodecMesh);
  CHECK(sequence);
  CHECK(sequence->source.path ==
        (std::filesystem::path(VR_CONFIG_FIXTURES) / "data" / "Rafa2_obj" /
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
  const auto replica =
      cfg::load(std::string(VR_CONFIG_FIXTURES) + "/codec_replica.json",
                cfg::Application::CodecReplica);
  CHECK(replica && replica->codec.encode_every == 1 && replica->codec.sweep);
  for (const cfg::Config* coded : {&sequence.value(), &replica.value()}) {
    CHECK(coded->codec.intra_period == 30 && coded->codec.update_size == 0);
    const auto encoder = vr_example::encoder_config(coded->codec);
    CHECK(encoder.ok() && encoder->intra_period == 30);
    CHECK(encoder->segment_size == 64 &&
          encoder->resolved_update_size() == encoder->segment_size);
  }

  const auto readback =
      cfg::load(std::string(VR_CONFIG_FIXTURES) + "/readback_orbbec.json",
                cfg::Application::FuseOrbbec);
  CHECK(readback && readback->operation == cfg::Operation::Stream);
  CHECK(readback->diagnostics.readback);
  CHECK(readback->diagnostics.dump_sets == 0);
  CHECK(readback->diagnostics.dump_directory.empty());
  // The first sets' JPEGs and luma, into a directory beside the scenario.
  const auto dump =
      cfg::load(std::string(VR_CONFIG_FIXTURES) + "/dump_orbbec.json",
                cfg::Application::FuseOrbbec);
  CHECK(dump && dump->operation == cfg::Operation::Stream);
  CHECK(dump->source.color_codec == cfg::ColorCodec::Mjpeg);
  CHECK(dump->diagnostics.readback && dump->diagnostics.dump_sets == 20);
  CHECK(dump->diagnostics.dump_directory ==
        (std::filesystem::path(VR_CONFIG_FIXTURES) / "results" / "dump")
            .lexically_normal()
            .string());
  const auto fusion =
      cfg::load(std::string(VR_CONFIG_FIXTURES) + "/fuse_orbbec.json",
                cfg::Application::FuseOrbbec);
  CHECK(fusion && !fusion->diagnostics.readback);
  // The rig coded set by set at 1 cm, fused Dynamic and freeing stale blocks
  // every 30 sets as rig_viewer does, at the codec templates' period; its
  // outputs beside the scenario.
  const auto coding =
      cfg::load(std::string(VR_CONFIG_FIXTURES) + "/codec_orbbec.json",
                cfg::Application::FuseOrbbec);
  CHECK(coding && coding->operation == cfg::Operation::Fuse);
  CHECK(coding->fusion.voxel_size_m == 0.01F &&
        coding->fusion.free_after == 30 && coding->fusion.max_weight == 20.F);
  CHECK(coding->codec.intra_period == 30 && !coding->codec.sweep);
  const auto results = std::filesystem::path(VR_CONFIG_FIXTURES) / "results";
  CHECK(coding->output.stream ==
        (results / "rig.vrst").lexically_normal().string());
  CHECK(coding->output.csv ==
        (results / "rig_frames.csv").lexically_normal().string());
  CHECK(coding->output.path ==
        (results / "rig_end.ply").lexically_normal().string());
  // The rig served live as it is coded and recorded, at the same period, for
  // 216000 camera frames: 30 minutes of four cameras at 30 fps.
  const auto serving =
      cfg::load(std::string(VR_CONFIG_FIXTURES) + "/serve_orbbec.json",
                cfg::Application::FuseOrbbec);
  CHECK(serving && serving->operation == cfg::Operation::Fuse);
  CHECK(serving->run.frames == 216000 && serving->fusion.voxel_size_m == 0.01F);
  CHECK(serving->codec.intra_period == 30 && !serving->codec.sweep);
  CHECK(serving->output.stream ==
        (results / "rig.vrst").lexically_normal().string());
  CHECK(serving->output.path ==
        (results / "rig_end.ply").lexically_normal().string());
  // Any port maps to every interface on it, keys on demand (fuse_orbbec codes
  // live), the server's other options at their defaults; serving alone needs
  // no file.
  const auto on_port = cfg::parse(R"({
    "version": 1, "operation": "fuse",
    "source": {"kind": "live", "sync_config": "sync.json"},
    "output": {"path": "rig.ply", "serve": 5000}
  })",
                                  cfg::Application::FuseOrbbec);
  CHECK(on_port);
  const vr_example::CodecOutputs to_port =
      vr_example::codec_outputs(on_port.value().output);
  const vr::networking::ServerOptions defaults_served;
  CHECK(to_port.csv.empty() && to_port.stream.empty());
  CHECK(to_port.serve && to_port.serve->host.empty() &&
        to_port.serve->port == 5000);
  CHECK(!to_port.serve->replay_gop && defaults_served.replay_gop);
  CHECK(to_port.serve->max_viewers == defaults_served.max_viewers &&
        to_port.serve->max_gop_bytes == defaults_served.max_gop_bytes &&
        to_port.serve->stall_limit == defaults_served.stall_limit);

  const auto dataset =
      cfg::parse(R"({
    "version": 1, "operation": "fuse",
    "source": {"kind": "dataset", "path": "room0",
               "camera_parameters": "intrinsics.json", "preload": true},
    "fusion": {"voxel_size_m": 0.01, "stride": 3,
               "min_depth_m": 0.3, "max_depth_m": 4.0},
    "run": {"frames": 17}
  })",
                 cfg::Application::FuseReplica, "/tmp/scenarios");
  CHECK(dataset);
  const auto source = vr_example::replica_options(dataset.value());
  CHECK(source.frame_limit == 17 && source.frame_stride == 3);
  CHECK(source.min_depth == 0.3F && source.max_depth == 4.F);
  CHECK(dataset.value().source.path == "/tmp/scenarios/room0");
  CHECK(dataset.value().source.camera_parameters ==
        "/tmp/scenarios/intrinsics.json");
  CHECK(dataset.value().fusion.truncation_m == 0.04F);

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
  const auto encoder = vr_example::encoder_config(encoded.value().codec);
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
  CHECK(
      vr_example::codec_settings(*encoder, encoded->codec.quantization_table) ==
      "K 128, table radial, scale 0.3; segments of 32 blocks, update "
      "segments of 128 entries; an I-frame every 30 frames");
  // The sweep times repeated encodes of one grid: every row intra, with the
  // scenario's sizes and the row's params.
  const auto row =
      vr_example::sweep_encoder_config(*encoder, {"band", 84, 0.4F});
  CHECK(row.ok() && row->intra_period == 1);
  CHECK(row->segment_size == 32 && row->update_size == 128);
  CHECK(row->params.coefficient_count == 84 &&
        row->params.quantization_scale == 0.4F);
  CHECK(row->params.quantization_weights[1] == 1.25F);  // band's, (1, 0, 0)
  CHECK(!vr_example::sweep_encoder_config(*encoder, {"none", 84, 0.4F}).ok());
  // The outputs, relative to the scenario's directory.
  const vr_example::CodecOutputs outputs =
      vr_example::codec_outputs(encoded.value().output);
  CHECK(outputs.csv == "/tmp/scenarios/results/rows.csv");
  CHECK(outputs.stream == "/tmp/streams/room0.vrst");
  CHECK(!outputs.serve);

  // Both codec profiles: the period as given, and with P-frames the update
  // size as given, core's 0 included, up to the format's longest segment;
  // left out, the P-frames' update segments are the segment size, as printed.
  const vr::codec::EncoderConfig defaults;
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
    const auto plain = vr_example::encoder_config(unset->codec);
    CHECK(plain.ok() && plain->intra_period == 1 &&
          plain->update_size == defaults.update_size);
    CHECK(vr_example::codec_settings(*plain, unset->codec.quantization_table) ==
          "K 64, table uniform, scale 0.2; segments of 64 blocks; every frame "
          "intra");
    for (const std::uint32_t period : {1u, 8u, 4294967295u}) {
      const auto settings =
          with_codec(R"({"intra_period": )" + std::to_string(period) + "}");
      CHECK(settings);
      const auto configured = vr_example::encoder_config(settings->codec);
      CHECK(configured.ok() && configured->intra_period == period);
    }
    for (const std::uint32_t size :
         {0u, 1u, 5u, vr::codec::kMaxDeviceDecodeSegmentSize,
          vr::codec::kMaxDeviceDecodeSegmentSize + 1}) {
      const auto settings =
          with_codec(R"({"intra_period": 2, "update_size": )" +
                     std::to_string(size) + "}");
      CHECK(settings);
      const auto configured = vr_example::encoder_config(settings->codec);
      if (size > vr::codec::kMaxDeviceDecodeSegmentSize) {
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
    const auto by_segment = vr_example::encoder_config(segments->codec);
    CHECK(by_segment.ok() && by_segment->resolved_update_size() == 16);
    CHECK(vr_example::codec_settings(*by_segment,
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
    const auto refused = vr_example::encoder_config(out_of_bounds->codec);
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
    return vr::volume::block_grid_params(fusion.voxel_size_m,
                                         fusion.truncation_m, 1024);
  };
  const auto sdf =
      vr_example::mesh_sdf_config(mesh.value().source, grid(mesh->fusion));
  CHECK(sdf.ok());
  CHECK(sdf->mode == vr::tsdf::MeshSdfMode::Shell && sdf->shell_voxels == 2.F);
  CHECK(mesh.value().source.up_vector.has_value());
  CHECK((*mesh.value().source.up_vector)[0] < -0.9);
  // Core loads any positive shell; the conversion's bound is recon's, with
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
        vr_example::mesh_sdf_config(thick->source, grid(thick->fusion));
    CHECK(shell.ok() == accepted);
    CHECK(accepted ||
          (shell.status().domain() == vkc::Status::Code::InvalidArgument &&
           shell.status().message().find("shell_voxels") != std::string::npos));
  }

  // The true-SDF band: a whole 15 voxels is taken at every voxel size from 1
  // to 50 mm, though two floats' ratio can fall short of it (0.06 m of 4 mm
  // voxels), and 14 refused; shell mode adds the half-thickness. Both mesh
  // templates pass.
  vr::tsdf::MeshSdfParams solid;
  vr::tsdf::MeshSdfParams shelled;
  shelled.mode = vr::tsdf::MeshSdfMode::Shell;
  shelled.shell_voxels = 2.F;
  for (int mm = 1; mm <= 50; ++mm) {
    const auto band = [mm](int voxels) {
      return vr::volume::block_grid_params(float(mm / 1000.0),
                                           float(voxels * mm / 1000.0), 1024);
    };
    CHECK(vr_example::check_true_sdf_band(solid, band(15)).ok());
    CHECK(vr_example::check_true_sdf_band(shelled, band(17)).ok());
    for (const auto& [sdf, voxels, least] :
         {std::tuple{&solid, 14, "at least 15 voxels"},
          std::tuple{&shelled, 16, "at least 17 voxels"}}) {
      const vkc::Status refused =
          vr_example::check_true_sdf_band(*sdf, band(voxels));
      CHECK(refused.domain() == vkc::Status::Code::InvalidArgument &&
            refused.message().find(least) != std::string::npos);
    }
  }
  for (const cfg::Config* mesh : {&single.value(), &sequence.value()}) {
    const auto sdf =
        vr_example::mesh_sdf_config(mesh->source, grid(mesh->fusion));
    CHECK(sdf.ok() &&
          vr_example::check_true_sdf_band(*sdf, grid(mesh->fusion)).ok());
  }

#ifdef VR_TEST_ORBBEC_CONFIG
  const auto live = cfg::parse(R"({
    "version": 1, "operation": "stream",
    "source": {"kind": "live", "sync_config": "sync.json",
               "calibration": "poses.json", "color_codec": "mjpeg",
               "color_width": 3840, "color_height": 2160, "fps": 25},
    "run": {"frames": 60000}
  })",
                               cfg::Application::FuseOrbbec);
  CHECK(live);
  const auto streams = vr_example::orbbec_stream_options(live.value());
  CHECK(streams.color_codec == vr::sensor::OrbbecColorCodec::Mjpeg);
  CHECK(streams.color_width == 3840 && streams.color_height == 2160);
  CHECK(streams.depth_width == 640 && streams.depth_height == 576);
  CHECK(streams.fps == 25);
  CHECK(streams.min_depth == 0.25F && streams.max_depth == 5.F);
  CHECK(streams.device == nullptr && streams.allocator == nullptr);
#endif
  return 0;
}
