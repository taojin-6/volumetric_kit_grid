// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// The integration boundary: core-validated settings reach the runtime options
// unchanged. No file assets, device or camera are opened.
#include <cmath>
#include <cstdio>
#include <string>

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
      {"fuse_orbbec", cfg::Application::FuseOrbbec},
      {"stream_orbbec", cfg::Application::FuseOrbbec},
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
  }

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

  const auto encoded = cfg::parse(R"({
    "version": 1, "operation": "codec", "source": {"kind": "dataset", "path": "room0"},
    "codec": {"coefficient_count": 128, "quantization_scale": 0.3,
              "quantization_table": "radial", "entropy": "device",
              "segment_size": 32, "encode_every": 0, "sweep": true}
  })",
                                  cfg::Application::CodecReplica);
  CHECK(encoded);
  const auto encoder = vr_example::encoder_config(encoded.value().codec);
  CHECK(encoder.params.coefficient_count == 128);
  CHECK(encoder.params.quantization_scale == 0.3F);
  CHECK(encoder.params.quantization_weights[0] == 1.F);
  CHECK(encoder.params.quantization_weights[1] == 1.125F);
  CHECK(encoder.params.quantization_weights[2] == 1.5F);
  CHECK(encoder.entropy == vr::codec::EntropyCoding::kDevice);
  CHECK(encoder.segment_size == 32);
  // The runtime library's own contract agrees with the central input policy.
  CHECK(encoder.params.validate().ok());

  const auto mesh = cfg::parse(R"({
    "version": 1, "operation": "codec",
    "source": {"kind": "mesh", "path": "person.obj", "height_m": 1.7,
               "up_vector": [-0.9120591159, 0.0661250017, -0.4046920474],
               "sdf_mode": "shell", "shell_voxels": 2.0}
  })",
                               cfg::Application::CodecMesh);
  CHECK(mesh);
  const auto sdf = vr_example::mesh_sdf_config(mesh.value().source);
  CHECK(sdf.mode == vr::tsdf::MeshSdfMode::Shell && sdf.shell_voxels == 2.F);
  CHECK(mesh.value().source.up_vector.has_value());
  CHECK((*mesh.value().source.up_vector)[0] < -0.9);

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
