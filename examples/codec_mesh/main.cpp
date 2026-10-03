// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// Geometry-only mesh -> TSDF -> codec -> marching cubes evaluation. The input
// is never changed. Height and head-up direction are explicit normalization
// choices applied after the IO tier imports geometry in source coordinates.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <sstream>
#include <string>

#include "codec_quantization.hpp"
#include "codec_stream.hpp"
#include "codec_sweep.hpp"
#include "mesh_normalization.hpp"
#include "parse_number.hpp"
#include "volumetric_kit/recon/core/allocator.hpp"
#include "volumetric_kit/recon/core/device.hpp"
#include "volumetric_kit/recon/core/instance.hpp"
#include "volumetric_kit/recon/eval/mesh_distance.hpp"
#include "volumetric_kit/recon/io/mesh_io.hpp"
#include "volumetric_kit/recon/io/ply_writer.hpp"
#include "volumetric_kit/recon/mesh/marching_cubes.hpp"
#include "volumetric_kit/recon/tsdf/mesh_integrator.hpp"

namespace vr = volumetric_kit::recon;
namespace mesh = volumetric_kit::recon::mesh;
namespace eval = volumetric_kit::recon::eval;
namespace {

struct Options {
  std::string input;
  std::string out_prefix;
  std::string quant_table = "uniform";
  double height = 0.0;
  vr_example::Point3d up{};
  float voxel = 0.01f;
  vr::tsdf::MeshSdfParams sdf;
  vr::codec::EncoderConfig codec;
  bool sweep = false;
  bool inspect_only = false;
};

vr::Result<Options> parse_args(int argc, char** argv) {
  Options o;
  bool have_up = false;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    const bool takes_value =
        a == "--height" || a == "--up-axis" || a == "--up-vector" ||
        a == "--voxel" || a == "--mode" || a == "--shell-voxels" ||
        a == "--step" || a == "--k" || a == "-o" || a == "--quant-table" ||
        a == "--entropy" || a == "--segment-size";
    if (takes_value && i + 1 == argc) {
      return vr::Status::invalid_argument(a + " needs a value");
    }
    const char* v = takes_value ? argv[++i] : nullptr;
    if (a == "--height") {
      VR_TRY(vr_example::parse_number(a, v, o.height));
    } else if (a == "--up-axis" || a == "--up-vector") {
      if (have_up)
        return vr::Status::invalid_argument("specify head-up only once");
      have_up = true;
      std::string value = v;
      if (a == "--up-axis") {
        double sign = 1.0;
        if (!value.empty() && value[0] == '-') {
          sign = -1.0;
          value.erase(0, 1);
        }
        if (value == "x")
          o.up = {sign, 0.0, 0.0};
        else if (value == "y")
          o.up = {0.0, sign, 0.0};
        else if (value == "z")
          o.up = {0.0, 0.0, sign};
        else
          return vr::Status::invalid_argument(
              "--up-axis needs x, y, z, -x, -y or -z");
      } else {
        std::replace(value.begin(), value.end(), ',', ' ');
        std::istringstream values(value);
        std::string extra;
        if (!(values >> o.up[0] >> o.up[1] >> o.up[2]) || values >> extra) {
          return vr::Status::invalid_argument("--up-vector needs x,y,z");
        }
      }
    } else if (a == "--voxel") {
      VR_TRY(vr_example::parse_number(a, v, o.voxel));
    } else if (a == "--shell-voxels") {
      VR_TRY(vr_example::parse_number(a, v, o.sdf.shell_voxels));
    } else if (a == "--step") {
      VR_TRY(vr_example::parse_number(a, v, o.codec.params.quantization_scale));
    } else if (a == "--quant-table") {
      o.quant_table = v;
    } else if (a == "--entropy") {
      VR_TRY(vr_example::parse_entropy(a, v, o.codec.entropy));
    } else if (a == "--segment-size") {
      int r = 0;
      VR_TRY(vr_example::parse_number(a, v, r));
      if (r < 1) return vr::Status::invalid_argument(a + " must be >= 1");
      o.codec.segment_size = std::uint32_t(r);
    } else if (a == "--k") {
      int k = 0;
      VR_TRY(vr_example::parse_number(a, v, k));
      if (k < 1 || k > 512) {
        return vr::Status::invalid_argument("--k must be in [1, 512]");
      }
      o.codec.params.coefficient_count = std::uint32_t(k);
    } else if (a == "--mode") {
      const std::string mode = v;
      if (mode == "signed")
        o.sdf.mode = vr::tsdf::MeshSdfMode::Signed;
      else if (mode == "shell")
        o.sdf.mode = vr::tsdf::MeshSdfMode::Shell;
      else
        return vr::Status::invalid_argument("--mode needs signed or shell");
    } else if (a == "-o") {
      o.out_prefix = v;
    } else if (a == "--sweep") {
      o.sweep = true;
    } else if (a == "--inspect-only") {
      o.inspect_only = true;
    } else if (a.empty() || a[0] == '-' || !o.input.empty()) {
      return vr::Status::invalid_argument("unexpected argument: " + a);
    } else {
      o.input = a;
    }
  }
  if (o.input.empty() || !(o.height > 0.0) || !have_up) {
    return vr::Status::invalid_argument(
        "usage: codec_mesh mesh-file --height metres "
        "(--up-axis y | --up-vector x,y,z) [--voxel metres] "
        "[--mode signed|shell] [--shell-voxels 1.5] [--k 64] [--step 0.2] "
        "[--quant-table uniform|band|radial] [--entropy auto|host|device] "
        "[--segment-size n] [--sweep] [--inspect-only] "
        "[-o prefix]");
  }
  if (!(o.voxel > 0.0f) || !std::isfinite(4.0f * o.voxel)) {
    return vr::Status::invalid_argument("--voxel must be finite and positive");
  }
  if (!std::isfinite(o.sdf.shell_voxels) || o.sdf.shell_voxels < 0.8660254f ||
      o.sdf.shell_voxels >= 4.0f) {
    return vr::Status::invalid_argument(
        "--shell-voxels must be in [sqrt(3)/2, 4)");
  }
  VR_TRY(vr_example::apply_quantization_table(o.codec.params, o.quant_table));
  VR_TRY(o.codec.params.validate());
  return o;
}

void print_bounds(const char* label, const vr_example::MeshBounds& b) {
  std::printf("%s: x [%.9g, %.9g], y [%.9g, %.9g], z [%.9g, %.9g]\n", label,
              b.min[0], b.max[0], b.min[1], b.max[1], b.min[2], b.max[2]);
}

vr::Status run(const Options& opt) {
  VR_ASSIGN(vr::io::TriangleMesh geometry, vr::io::load_mesh(opt.input));
  VR_ASSIGN(const auto normalization,
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
    VR_TRY(vr::io::write_ply(opt.out_prefix + "_input.ply", input));
  }
  if (opt.inspect_only) return {};
  if (opt.sdf.mode == vr::tsdf::MeshSdfMode::Signed &&
      !topology.supports_signed()) {
    return vr::Status::invalid_argument(
        "indexed topology does not support signed mode; repair the mesh or "
        "choose --mode shell");
  }

  VR_ASSIGN(vr::Instance instance, vr::Instance::create({}));
  VR_ASSIGN(VkPhysicalDevice gpu, instance.select_physical_device());
  VR_ASSIGN(vr::Device device, vr::Device::create(instance, gpu, {}));
  VR_ASSIGN(vr::Allocator allocator,
            vr::Allocator::create(instance.handle(), device));
  const float trunc = 4.0f * opt.voxel;
  const vr::volume::AttributeSpec attrs[] = {{"tsdf", sizeof(float)},
                                             {"weight", sizeof(float)}};
  VR_ASSIGN(
      vr::volume::VoxelBlockGrid volume,
      vr::volume::VoxelBlockGrid::create(
          device, allocator,
          vr_example::example_grid_params(opt.voxel, trunc, 1024), attrs, 2));
  VR_TRY(volume.clear());
  const auto vertices = std::uint32_t(geometry.positions.size());
  const auto triangles = std::uint32_t(geometry.indices.size() / 3);
  // Grow only for a capacity limit. Lost bucket-lock races leave a residue
  // over a table with room, which a retry places (fuse_frame.hpp's
  // allocate_band_with does the same).
  for (int contended = 0;;) {
    vr::volume::AllocFailures failures;
    VR_ASSIGN(const auto failed,
              volume.map().allocate_from_triangles(
                  geometry.positions.data(), vertices, geometry.indices.data(),
                  triangles, &failures));
    if (failed == 0) break;
    if (!failures.capacity_limited()) {
      if (++contended == 5) {
        return vr::Status::io_error(
            "mesh allocation kept losing bucket-lock races");
      }
      continue;
    }
    const std::int64_t buckets = 2 * std::int64_t(volume.grid().num_buckets);
    if (buckets * vr_example::kExampleBucketSize >
        std::numeric_limits<std::int32_t>::max()) {
      return vr::Status::out_of_memory("mesh allocation exceeds grid capacity");
    }
    VR_TRY(volume.resize(std::int32_t(buckets)));
  }
  VR_ASSIGN(vr::tsdf::MeshIntegrator integrator,
            vr::tsdf::MeshIntegrator::create(device, allocator));
  vr::StageMetrics conversion;
  VR_ASSIGN(const auto stats,
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
  vr_example::print_stage_rows("mesh conversion", conversion, 1);
  VR_ASSIGN(mesh::MarchingCubes extractor,
            mesh::MarchingCubes::create(device, allocator));
  VR_ASSIGN(vr_example::CodecStream stream,
            vr_example::CodecStream::create(device, allocator, opt.codec));
  std::printf("codec: K %u, table %s, quantization scale %.6g\n",
              opt.codec.params.coefficient_count, opt.quant_table.c_str(),
              double(opt.codec.params.quantization_scale));
  VR_TRY(stream.code(volume, opt.out_prefix.empty()
                                 ? std::string{}
                                 : opt.out_prefix + ".vrtc"));
  stream.report(30.0, 0);  // One mesh frame, not a measured sequence bitrate.
  VR_ASSIGN(const mesh::Mesh source, extractor.extract_host(volume));
  VR_ASSIGN(const mesh::Mesh decoded, extractor.extract_host(stream.player()));
  if (!opt.out_prefix.empty()) {
    VR_TRY(vr::io::write_ply(opt.out_prefix + "_source.ply", source));
    VR_TRY(vr::io::write_ply(opt.out_prefix + "_decoded.ply", decoded));
  }
  eval::CompareOptions compare;
  compare.reach = std::max(trunc, 0.02f);
  compare.stride = 4;
  compare.fscore_threshold = 0.5f * opt.voxel;
  VR_ASSIGN(const eval::ReferenceMesh input_reference,
            eval::ReferenceMesh::create(input, compare));
  VR_ASSIGN(const eval::ReferenceMesh source_reference,
            eval::ReferenceMesh::create(source, compare));
  VR_ASSIGN(const auto conversion_error, input_reference.compare(source));
  VR_ASSIGN(const auto codec_error, source_reference.compare(decoded));
  VR_ASSIGN(const auto total_error, input_reference.compare(decoded));
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
    VR_TRY(vr_example::run_codec_sweep(device, allocator, volume,
                                       source_reference, extractor,
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
