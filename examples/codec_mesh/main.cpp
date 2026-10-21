// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// Geometry-only mesh sequence -> TSDF -> codec -> marching cubes evaluation.
// The source files are never changed. The scenario's head-up direction and
// height or unit place every frame in metres through one transform fitted to
// the whole range, with io's opt-in normalization helpers.

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <optional>
#include <string>

#include "codec_config.hpp"
#include "codec_stream.hpp"
#include "codec_sweep.hpp"
#include "config.hpp"
#include "geometry_mesh.hpp"
#include "grid_layout.hpp"
#include "mesh_sequence.hpp"
#include "stage_table.hpp"
#include "volumetric_kit/core/vulkan/allocator.hpp"
#include "volumetric_kit/core/vulkan/device.hpp"
#include "volumetric_kit/core/vulkan/instance.hpp"
#include "volumetric_kit/recon/core/device_requirements.hpp"
#include "volumetric_kit/recon/eval/mesh_distance.hpp"
#include "volumetric_kit/recon/io/mesh_normalization.hpp"
#include "volumetric_kit/recon/io/ply_writer.hpp"
#include "volumetric_kit/recon/mesh/marching_cubes.hpp"
#include "volumetric_kit/recon/tsdf/mesh_integrator.hpp"

namespace vr = volumetric_kit::recon;
namespace vkc = volumetric_kit::core;
namespace mesh = volumetric_kit::recon::mesh;
namespace eval = volumetric_kit::recon::eval;
namespace {

using Options = vr_example::config::Config;

vkc::Status run(const Options& opt) {
  const bool inspect = opt.codec.inspect_only;
  const bool is_signed =
      opt.source.sdf_mode == vr_example::config::SdfMode::Signed;
  const float trunc = opt.fusion.truncation_m;
  const vr::volume::VoxelGridParams grid =
      vr_example::example_grid_params(opt.fusion.voxel_size_m, trunc, 1024);
  // Before any GPU work, and before the first pass loads every frame: a
  // step the codec or the conversion refuses costs neither.
  vr::codec::EncoderConfig config;
  vr::tsdf::MeshSdfParams sdf;
  if (!inspect) {
    VKC_ASSIGN(config, vr_example::encoder_config(opt.codec));
    VKC_ASSIGN(sdf, vr_example::mesh_sdf_config(opt.source, grid));
  }
  VKC_ASSIGN(const vr_example::MeshSequence sequence,
             vr_example::MeshSequence::open(opt));
  sequence.print();
  VKC_ASSIGN(const std::string placement,
             vr::io::encode_transform(sequence.transform()));
  if (inspect) {
    VKC_ASSIGN(const vr_example::Inspection inspected,
               vr_example::inspect_frames(sequence));
    vr_example::print_frame(inspected.last);
    if (opt.output.prefix.empty()) return {};
    return vr::io::write_ply(opt.output.prefix + "_input.ply",
                             vr_example::geometry_mesh(inspected.last.mesh));
  }

  VKC_ASSIGN(vkc::Instance instance, vkc::Instance::create({}));
  VKC_ASSIGN(vkc::PhysicalDeviceInfo gpu,
             instance.select_physical_device(vr::device_requirements()));
  VKC_ASSIGN(vkc::Device device,
             vkc::Device::create(instance, gpu, vr::device_requirements()));
  VKC_ASSIGN(vkc::Allocator allocator,
             vkc::Allocator::create(instance.handle(), device));
  const vr::volume::AttributeSpec attrs[] = {{"tsdf", sizeof(float)},
                                             {"weight", sizeof(float)}};
  VKC_ASSIGN(
      vr::volume::VoxelBlockGrid volume,
      vr::volume::VoxelBlockGrid::create(device, allocator, grid, attrs, 2));
  VKC_ASSIGN(vr::tsdf::MeshIntegrator integrator,
             vr::tsdf::MeshIntegrator::create(device, allocator));
  VKC_ASSIGN(mesh::MarchingCubes extractor,
             mesh::MarchingCubes::create(device, allocator));
  // Made before any frame is converted or written, so a rerun onto an
  // existing stream is refused before any conversion or output.
  VKC_ASSIGN(
      vr_example::CodecStream stream,
      vr_example::CodecStream::create(device, allocator, config,
                                      vr_example::codec_outputs(opt.output),
                                      "codec_mesh", placement));
  std::printf(
      "codec: %s\n",
      vr_example::codec_settings(config, opt.codec.quantization_table).c_str());

  vkc::StageMetrics conversion;
  std::size_t defective = 0;
  std::optional<vr_example::CodedFrame> last;
  for (std::uint32_t n = sequence.first(), i = 0; i < sequence.count();
       ++i, ++n) {
    last.reset();  // one frame's geometry held at a time
    VKC_ASSIGN(last, vr_example::code_frame(sequence, n, volume, integrator,
                                            sdf, stream, &conversion));
    const std::string defects =
        vr_example::topology_defects(last->frame.topology);
    if (defects.empty()) continue;
    ++defective;
    std::printf("frame %u: %s; coded %s\n", n, defects.c_str(),
                is_signed ? "signed (source.allow_holes)" : "as a shell");
  }
  VKC_TRY(stream.finish());
  const auto& stats = last->stats;
  std::printf(
      "mesh -> TSDF: %s mode, %.6g m voxels, %.6g m truncation; %u "
      "frame(s), %zu with topology defects; last: %u triangles, %u blocks, "
      "%u bin entries, %u dispatches\n",
      is_signed ? "signed closest-face" : "shell",
      double(opt.fusion.voxel_size_m), double(trunc), sequence.count(),
      defective, stats.triangles, stats.blocks, stats.bin_entries,
      stats.dispatches);
  if (!is_signed) {
    std::printf(
        "shell half-thickness: %.6g m; input comparison includes intentional "
        "shell offset\n",
        double(opt.source.shell_voxels * opt.fusion.voxel_size_m));
  }
  vr_example::print_stage_rows("mesh conversion, per frame", conversion,
                               sequence.count(), 2);
  // A single mesh is no measured sequence: it has no bitrate.
  stream.report(vr_example::kNominalFps, sequence.count() > 1 ? 1 : 0);

  // The last frame's surfaces, written first so a comparison that refuses a
  // mesh still leaves them to look at.
  vr_example::print_frame(last->frame);
  const mesh::Mesh input = vr_example::geometry_mesh(last->frame.mesh);
  VKC_ASSIGN(const mesh::Mesh source, extractor.extract_host(volume));
  VKC_ASSIGN(const mesh::Mesh decoded, extractor.extract_host(stream.player()));
  if (!opt.output.prefix.empty()) {
    VKC_TRY(vr::io::write_ply(opt.output.prefix + "_input.ply", input));
    VKC_TRY(vr::io::write_ply(opt.output.prefix + "_source.ply", source));
    VKC_TRY(vr::io::write_ply(opt.output.prefix + "_decoded.ply", decoded));
  }
  const eval::CompareOptions compare =
      vr_example::codec_compare_options(opt.fusion.voxel_size_m, trunc);
  VKC_ASSIGN(const eval::ReferenceMesh input_reference,
             eval::ReferenceMesh::create(input, compare));
  VKC_ASSIGN(const eval::ReferenceMesh source_reference,
             eval::ReferenceMesh::create(source, compare));
  VKC_ASSIGN(const auto conversion_error, input_reference.compare(source));
  VKC_ASSIGN(const auto codec_error, source_reference.compare(decoded));
  VKC_ASSIGN(const auto total_error, input_reference.compare(decoded));
  std::printf(
      "surface triangles, frame %u: %zu normalized input, %zu uncompressed "
      "TSDF, %zu decoded\n",
      last->frame.number, input.triangle_count(), source.triangle_count(),
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
