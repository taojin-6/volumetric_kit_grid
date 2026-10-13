// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// Geometry-only mesh -> TSDF -> codec -> marching cubes evaluation. The input
// is never changed. Height and head-up direction are explicit normalization
// choices applied after the IO tier imports geometry in source coordinates.

#include <algorithm>
#include <cstdio>
#include <string>

#include "codec_config.hpp"
#include "codec_stream.hpp"
#include "codec_sweep.hpp"
#include "config.hpp"
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

using Options = vr_example::config::Config;

void print_bounds(const char* label, const vr_example::MeshBounds& b) {
  std::printf("%s: x [%.9g, %.9g], y [%.9g, %.9g], z [%.9g, %.9g]\n", label,
              b.min[0], b.max[0], b.min[1], b.max[1], b.min[2], b.max[2]);
}

vkc::Status run(const Options& opt) {
  VKC_ASSIGN(vr::io::TriangleMesh geometry, vr::io::load_mesh(opt.source.path));
  VKC_ASSIGN(const auto normalization,
             vr_example::normalize_mesh_height(geometry, opt.source.height_m,
                                               *opt.source.up_vector));
  // Audited after normalization, on the float metres the conversion reads,
  // so a triangle that collapses in the conversion counts as degenerate.
  const auto topology = vr_example::audit_mesh_topology(geometry);
  std::printf(
      "mesh: %zu vertices, %zu triangles; geometry only, physical scale "
      "set by source.height_m\n",
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
  if (!opt.output.prefix.empty()) {
    VKC_TRY(vr::io::write_ply(opt.output.prefix + "_input.ply", input));
  }
  if (opt.codec.inspect_only) return {};
  // Before any GPU work: a step the codec refuses costs none.
  VKC_ASSIGN(const vr::codec::EncoderConfig config,
             vr_example::encoder_config(opt.codec));
  if (opt.source.sdf_mode == vr_example::config::SdfMode::Signed &&
      !topology.supports_signed()) {
    return vkc::Status::invalid_argument(
        "indexed topology does not support signed mode; repair the mesh or "
        "choose source.sdf_mode=shell");
  }

  VKC_ASSIGN(vkc::Instance instance, vkc::Instance::create({}));
  VKC_ASSIGN(vkc::PhysicalDeviceInfo gpu,
             instance.select_physical_device(vr::device_requirements()));
  VKC_ASSIGN(vkc::Device device,
             vkc::Device::create(instance, gpu, vr::device_requirements()));
  VKC_ASSIGN(vkc::Allocator allocator,
             vkc::Allocator::create(instance.handle(), device));
  const float trunc = opt.fusion.truncation_m;
  const vr::volume::AttributeSpec attrs[] = {{"tsdf", sizeof(float)},
                                             {"weight", sizeof(float)}};
  VKC_ASSIGN(
      vr::volume::VoxelBlockGrid volume,
      vr::volume::VoxelBlockGrid::create(
          device, allocator,
          vr_example::example_grid_params(opt.fusion.voxel_size_m, trunc, 1024),
          attrs, 2));
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
                                  geometry.indices.data(), triangles,
                                  vr_example::mesh_sdf_config(opt.source),
                                  &conversion));
  std::printf(
      "mesh -> TSDF: %s mode, %.6g m voxels, %.6g m truncation; "
      "%u triangles, %u blocks, %u bin entries, %u dispatches\n",
      opt.source.sdf_mode == vr_example::config::SdfMode::Signed
          ? "signed closest-face"
          : "shell",
      double(opt.fusion.voxel_size_m), double(trunc), stats.triangles,
      stats.blocks, stats.bin_entries, stats.dispatches);
  if (opt.source.sdf_mode == vr_example::config::SdfMode::Shell) {
    std::printf(
        "shell half-thickness: %.6g m; input comparison includes intentional "
        "shell offset\n",
        double(opt.source.shell_voxels * opt.fusion.voxel_size_m));
  }
  vr_example::print_stage_rows("mesh conversion", conversion, 1, 2);
  VKC_ASSIGN(mesh::MarchingCubes extractor,
             mesh::MarchingCubes::create(device, allocator));
  VKC_ASSIGN(vr_example::CodecStream stream,
             vr_example::CodecStream::create(device, allocator, config));
  std::printf("codec: K %u, table %s, quantization scale %.6g\n",
              config.params.coefficient_count,
              vr_example::quantization_name(opt.codec.quantization_table),
              double(config.params.quantization_scale));
  VKC_TRY(stream.code(
      volume, 0,
      opt.output.prefix.empty() ? std::string{} : opt.output.prefix + ".vrtc"));
  stream.report(30.0, 0);  // One mesh frame, not a measured sequence bitrate.
  VKC_ASSIGN(const mesh::Mesh source, extractor.extract_host(volume));
  VKC_ASSIGN(const mesh::Mesh decoded, extractor.extract_host(stream.player()));
  if (!opt.output.prefix.empty()) {
    VKC_TRY(vr::io::write_ply(opt.output.prefix + "_source.ply", source));
    VKC_TRY(vr::io::write_ply(opt.output.prefix + "_decoded.ply", decoded));
  }
  eval::CompareOptions compare;
  compare.reach = std::max(trunc, 0.02f);
  compare.stride = 4;
  compare.fscore_threshold = 0.5f * opt.fusion.voxel_size_m;
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
  vr_example::print_comparison(conversion_error, opt.fusion.voxel_size_m);
  std::printf(
      "codec only: decoded surface against uncompressed TSDF surface:\n");
  vr_example::print_comparison(codec_error, opt.fusion.voxel_size_m);
  std::printf("total: decoded surface against normalized input:\n");
  vr_example::print_comparison(total_error, opt.fusion.voxel_size_m);
  if (opt.codec.sweep) {
    VKC_TRY(vr_example::run_codec_sweep(device, allocator, volume,
                                        source_reference, extractor, config,
                                        stream.player()));
  }
  return {};
}
}  // namespace

int run_main(const Options& opt) {
  const auto status = run(opt);
  if (!status.ok()) {
    std::fprintf(stderr, "codec_mesh failed: %s\n", status.message().c_str());
    return 1;
  }
  return 0;
}

int main(int argc, char** argv) {
  return vr_example::configured_main(argc, argv,
                                     vr_example::config::Application::CodecMesh,
                                     "codec_mesh", run_main);
}
