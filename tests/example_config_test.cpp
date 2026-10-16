// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// The integration boundary: core-validated settings reach the runtime options
// unchanged, and the codec checks the steps core leaves to it. No file
// assets, device or camera are opened.
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <string>
#include <utility>

#include "codec_config.hpp"
#include "replica_config.hpp"
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
    // Classic unless the rig viewer's; no template writes codec rows or
    // frames.
    CHECK(loaded->fusion.dynamic ==
          (example.application == cfg::Application::RigViewer));
    CHECK(loaded->output.csv.empty() && loaded->output.stream.empty());
  }

  // The single-mesh template stays one frame; the sequence's codes Rafa2's
  // frames 1-150 in an 8-voxel band, holes admitted.
  const auto single =
      cfg::load(std::string(VR_CONFIG_FIXTURES) + "/codec_mesh.json",
                cfg::Application::CodecMesh);
  CHECK(single && single->run.first == 0 && single->run.frames == 1);
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
  CHECK(sequence->fusion.voxel_size_m == 0.005F &&
        sequence->fusion.truncation_m == 0.04F);
  CHECK(sequence->source.height_m == 1.7 &&
        sequence->source.metres_per_unit == 0.0);
  CHECK(sequence->source.allow_holes && !sequence->codec.sweep &&
        sequence->output.prefix.empty());

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
              "encode_every": 0, "sweep": true},
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
  CHECK(encoder->intra_period == 1);
  // The outputs, relative to the scenario's directory.
  const vr_example::CodecOutputs outputs =
      vr_example::codec_outputs(encoded.value().output);
  CHECK(outputs.csv == "/tmp/scenarios/results/rows.csv");
  CHECK(outputs.stream == "/tmp/streams/room0.vrst");

  // Core accepts a period for both codec profiles. Until the examples write
  // P-frame records, they must refuse a requested period instead of silently
  // keeping the encoder's all-intra default.
  for (const auto& [source_json, application] :
       {std::pair{R"({"kind": "dataset", "path": "room0"})",
                  cfg::Application::CodecReplica},
        std::pair{R"({"kind": "mesh", "path": "shape.obj",
                      "metres_per_unit": 1, "up_axis": "y"})",
                  cfg::Application::CodecMesh}}) {
    for (const char* period : {"1", "8", "4294967295"}) {
      const auto settings = cfg::parse(
          std::string(R"({"version": 1, "operation": "codec",
                                    "source": )") +
              source_json + R"(, "codec": {"intra_period": )" + period + "}}",
          application);
      CHECK(settings);
      const auto configured = vr_example::encoder_config(settings->codec);
      if (std::string(period) == "1") {
        CHECK(configured.ok() && configured->intra_period == 1);
      } else {
        CHECK(!configured.ok() && configured.status().domain() ==
                                      vkc::Status::Code::InvalidArgument);
        CHECK(configured.status().message().find("codec.intra_period") !=
              std::string::npos);
      }
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
    return vr_example::example_grid_params(fusion.voxel_size_m,
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
