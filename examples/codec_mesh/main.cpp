// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// Geometry-only mesh -> TSDF -> codec -> marching cubes evaluation. The input
// is never changed. Height and head-up direction are explicit normalization
// choices applied after the IO tier imports geometry in source coordinates.

#include <algorithm>
#include <cstdio>
#include <sstream>
#include <string>

#include "cli.hpp"
#include "codec_flags.hpp"
#include "codec_stream.hpp"
#include "codec_sweep.hpp"
#include "grid_layout.hpp"
#include "mesh_normalization.hpp"
#include "stage_table.hpp"
#include "volumetric_kit/core/vulkan/allocator.hpp"
#include "volumetric_kit/core/vulkan/device.hpp"
#include "volumetric_kit/core/vulkan/instance.hpp"
#include "volumetric_kit/recon/core/device_requirements.hpp"
#include "volumetric_kit/recon/eval/mesh_distance.hpp"
#include "volumetric_kit/recon/io/mesh_io.hpp"
#include "volumetric_kit/recon/io/ply_writer.hpp"
#include "volumetric_kit/recon/mesh/marching_cubes.hpp"
#include "volumetric_kit/recon/tsdf/mesh_integrator.hpp"
#include "volumetric_kit/recon/volume/grid_growth.hpp"

namespace vr = volumetric_kit::recon;
namespace vkc = volumetric_kit::core;
namespace mesh = volumetric_kit::recon::mesh;
namespace eval = volumetric_kit::recon::eval;
namespace {

struct Options {
  std::string input;
  std::string out_prefix;
  double height = 0.0;
  vr_example::Point3d up{};
  float voxel = 0.01f;
  vr::tsdf::MeshSdfParams sdf;
  vr_example::CodecFlags codec;
  bool sweep = false;
  bool inspect_only = false;
};

// --up-axis: x, y, z, -x, -y or -z.
vkc::Status parse_up_axis(const std::string& flag, std::string value,
                          vr_example::Point3d& up) {
  double sign = 1.0;
  if (!value.empty() && value[0] == '-') {
    sign = -1.0;
    value.erase(0, 1);
  }
  if (value == "x") {
    up = {sign, 0.0, 0.0};
  } else if (value == "y") {
    up = {0.0, sign, 0.0};
  } else if (value == "z") {
    up = {0.0, 0.0, sign};
  } else {
    return vkc::Status::invalid_argument(flag + " needs x, y, z, -x, -y or -z");
  }
  return {};
}

// --up-vector: x,y,z.
vkc::Status parse_up_vector(const std::string& flag, std::string value,
                            vr_example::Point3d& up) {
  std::replace(value.begin(), value.end(), ',', ' ');
  std::istringstream values(value);
  std::string extra;
  if (!(values >> up[0] >> up[1] >> up[2]) || values >> extra) {
    return vkc::Status::invalid_argument(flag + " needs x,y,z");
  }
  return {};
}

vkc::Result<Options> parse_args(int argc, char** argv) {
  Options o;
  vr_example::Cli cli("codec_mesh");
  cli.positional("mesh-file", o.input)
      .option("--height", "metres", o.height)
      .require("--height")
      .on("--up-axis", "[-]x|y|z",
          [&o](const std::string& flag, const char* value) {
            return parse_up_axis(flag, value, o.up);
          })
      .on("--up-vector", "x,y,z",
          [&o](const std::string& flag, const char* value) {
            return parse_up_vector(flag, value, o.up);
          })
      .one_of({"--up-axis", "--up-vector"})  // head-up, by axis or vector
      .require("--up-axis")
      .option("--voxel", "metres", o.voxel)
      .on("--mode", "signed|shell",
          [&o](const std::string& flag, const char* value) {
            const std::string mode = value;
            if (mode == "signed") {
              o.sdf.mode = vr::tsdf::MeshSdfMode::Signed;
            } else if (mode == "shell") {
              o.sdf.mode = vr::tsdf::MeshSdfMode::Shell;
            } else {
              return vkc::Status::invalid_argument(flag +
                                                   " needs signed or shell");
            }
            return vkc::Status{};
          })
      .option("--shell-voxels", "1.5", o.sdf.shell_voxels);
  o.codec.add_to(cli);
  cli.flag("--sweep", o.sweep)
      .flag("--inspect-only", o.inspect_only)
      .option("-o", "prefix", o.out_prefix)
      .check([&o]() -> vkc::Status {
        if (!(o.height > 0.0)) {
          return vkc::Status::invalid_argument("--height must be > 0");
        }
        VKC_TRY(vr_example::check_voxel(o.voxel));
        if (o.sdf.shell_voxels < 0.8660254f || o.sdf.shell_voxels >= 4.0f) {
          return vkc::Status::invalid_argument(
              "--shell-voxels must be in [sqrt(3)/2, 4)");
        }
        return {};
      });
  VKC_TRY(cli.parse(argc, argv));
  return o;
}

void print_bounds(const char* label, const vr_example::MeshBounds& b) {
  std::printf("%s: x [%.9g, %.9g], y [%.9g, %.9g], z [%.9g, %.9g]\n", label,
              b.min[0], b.max[0], b.min[1], b.max[1], b.min[2], b.max[2]);
}

vkc::Status run(const Options& opt) {
  VKC_ASSIGN(vr::io::TriangleMesh geometry, vr::io::load_mesh(opt.input));
  VKC_ASSIGN(const auto normalization,
             vr_example::normalize_mesh_height(geometry, opt.height, opt.up));
  // Audited after normalization, on the float metres the conversion reads,
  // so a triangle that collapses in the conversion counts as degenerate.
  const auto topology = vr_example::audit_mesh_topology(geometry);
  std::printf(
      "mesh: %zu vertices, %zu triangles; geometry only, physical scale "
      "set by --height\n",
      geometry.positions.size(), geometry.indices.size() / 3);
  print_bounds("input bounds (source units)", normalization.original);
  std::printf(
      "head-up (input coordinates): (%.9g, %.9g, %.9g); projected height %.12g "
      "source units\n",
      normalization.up[0], normalization.up[1], normalization.up[2],
      normalization.original_height);
  std::printf(
      "normalization: %.12g metres/source-unit; rotate head-up to +Y, centre "
      "X/Z, feet at Y=0\n",
      normalization.metres_per_unit);
  print_bounds("normalized bounds (metres)", normalization.normalized);
  std::printf(
      "indexed topology: %zu edges, %zu boundary, %zu nonmanifold, %zu "
      "inconsistently oriented; "
      "%zu degenerate triangles, %zu components (%zu nonpositive signed "
      "volumes); "
      "signed volume %.12g cubic metres; self-intersections not checked\n",
      topology.edges, topology.boundary_edges, topology.nonmanifold_edges,
      topology.inconsistent_edges, topology.degenerate_triangles,
      topology.components, topology.nonpositive_components,
      topology.signed_volume);
  const mesh::Mesh input = vr_example::geometry_mesh(geometry);
  if (!opt.out_prefix.empty()) {
    VKC_TRY(vr::io::write_ply(opt.out_prefix + "_input.ply", input));
  }
  if (opt.inspect_only) return {};
  if (opt.sdf.mode == vr::tsdf::MeshSdfMode::Signed &&
      !topology.supports_signed()) {
    return vkc::Status::invalid_argument(
        "indexed topology does not support signed mode; repair the mesh or "
        "choose --mode shell");
  }

  VKC_ASSIGN(vkc::Instance instance, vkc::Instance::create({}));
  VKC_ASSIGN(vkc::PhysicalDeviceInfo gpu,
             instance.select_physical_device(vr::device_requirements()));
  VKC_ASSIGN(vkc::Device device,
             vkc::Device::create(instance, gpu, vr::device_requirements()));
  VKC_ASSIGN(vkc::Allocator allocator,
             vkc::Allocator::create(instance.handle(), device));
  const float trunc = vr_example::default_trunc(opt.voxel);
  const vr::volume::AttributeSpec attrs[] = {{"tsdf", sizeof(float)},
                                             {"weight", sizeof(float)}};
  VKC_ASSIGN(
      vr::volume::VoxelBlockGrid volume,
      vr::volume::VoxelBlockGrid::create(
          device, allocator,
          vr_example::example_grid_params(opt.voxel, trunc, 1024), attrs, 2));
  VKC_TRY(volume.clear());
  const auto vertices = std::uint32_t(geometry.positions.size());
  const auto triangles = std::uint32_t(geometry.indices.size() / 3);
  // Grow only for a capacity limit. Lost bucket-lock races leave a residue
  // over a table with room, which a retry places: the mesh integrator refuses
  // a band with a block missing.
  for (int contended = 0;;) {
    vr::volume::AllocFailures failures;
    VKC_ASSIGN(const auto failed,
               volume.map().allocate_from_triangles(
                   geometry.positions.data(), vertices, geometry.indices.data(),
                   triangles, &failures));
    if (failed == 0) break;
    if (!failures.capacity_limited()) {
      if (++contended == 5) {
        return vkc::Status::io_error(
            "mesh allocation kept losing bucket-lock races");
      }
      continue;
    }
    VKC_TRY(vr::volume::grow_grid(volume));
  }
  VKC_ASSIGN(vr::tsdf::MeshIntegrator integrator,
             vr::tsdf::MeshIntegrator::create(device, allocator));
  vkc::StageMetrics conversion;
  VKC_ASSIGN(const auto stats,
             integrator.integrate(volume, geometry.positions.data(), vertices,
                                  geometry.indices.data(), triangles, opt.sdf,
                                  &conversion));
  std::printf(
      "mesh -> TSDF: %s mode, %.6g m voxels, %.6g m truncation; "
      "%u triangles, %u blocks, %u bin entries, %u dispatches\n",
      opt.sdf.mode == vr::tsdf::MeshSdfMode::Signed ? "signed closest-face"
                                                    : "shell",
      double(opt.voxel), double(trunc), stats.triangles, stats.blocks,
      stats.bin_entries, stats.dispatches);
  if (opt.sdf.mode == vr::tsdf::MeshSdfMode::Shell) {
    std::printf(
        "shell half-thickness: %.6g m; input comparison includes intentional "
        "shell offset\n",
        double(opt.sdf.shell_voxels * opt.voxel));
  }
  vr_example::print_stage_rows("mesh conversion", conversion, 1, 2);
  VKC_ASSIGN(mesh::MarchingCubes extractor,
             mesh::MarchingCubes::create(device, allocator));
  const vr::codec::EncoderConfig& config = opt.codec.config;
  VKC_ASSIGN(vr_example::CodecStream stream,
             vr_example::CodecStream::create(device, allocator, config));
  std::printf("codec: K %u, table %s, quantization scale %.6g\n",
              config.params.coefficient_count, opt.codec.quant_table.c_str(),
              double(config.params.quantization_scale));
  VKC_TRY(stream.code(volume, opt.out_prefix.empty()
                                  ? std::string{}
                                  : opt.out_prefix + ".vrtc"));
  stream.report(30.0, 0);  // One mesh frame, not a measured sequence bitrate.
  VKC_ASSIGN(const mesh::Mesh source, extractor.extract_host(volume));
  VKC_ASSIGN(const mesh::Mesh decoded, extractor.extract_host(stream.player()));
  if (!opt.out_prefix.empty()) {
    VKC_TRY(vr::io::write_ply(opt.out_prefix + "_source.ply", source));
    VKC_TRY(vr::io::write_ply(opt.out_prefix + "_decoded.ply", decoded));
  }
  eval::CompareOptions compare;
  compare.reach = std::max(trunc, 0.02f);
  compare.stride = 4;
  compare.fscore_threshold = 0.5f * opt.voxel;
  VKC_ASSIGN(const eval::ReferenceMesh input_reference,
             eval::ReferenceMesh::create(input, compare));
  VKC_ASSIGN(const eval::ReferenceMesh source_reference,
             eval::ReferenceMesh::create(source, compare));
  VKC_ASSIGN(const auto conversion_error, input_reference.compare(source));
  VKC_ASSIGN(const auto codec_error, source_reference.compare(decoded));
  VKC_ASSIGN(const auto total_error, input_reference.compare(decoded));
  std::printf(
      "surface triangles: %zu normalized input, %zu uncompressed TSDF, %zu "
      "decoded\n",
      input.triangle_count(), source.triangle_count(),
      decoded.triangle_count());
  std::printf(
      "conversion: uncompressed TSDF surface against normalized input:\n");
  vr_example::print_comparison(conversion_error, opt.voxel);
  std::printf(
      "codec only: decoded surface against uncompressed TSDF surface:\n");
  vr_example::print_comparison(codec_error, opt.voxel);
  std::printf("total: decoded surface against normalized input:\n");
  vr_example::print_comparison(total_error, opt.voxel);
  if (opt.sweep) {
    VKC_TRY(vr_example::run_codec_sweep(device, allocator, volume,
                                        source_reference, extractor, config,
                                        stream.player()));
  }
  return {};
}
}  // namespace

int main(int argc, char** argv) {
  const auto options = parse_args(argc, argv);
  if (!options.ok()) {
    std::fprintf(stderr, "%s\n", options.status().message().c_str());
    return 2;
  }
  const auto status = run(options.value());
  if (!status.ok()) {
    std::fprintf(stderr, "codec_mesh failed: %s\n", status.message().c_str());
    return 1;
  }
  return 0;
}
