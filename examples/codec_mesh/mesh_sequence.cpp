// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include "mesh_sequence.hpp"

#include <cstdio>
#include <string>
#include <utility>

#include "volumetric_kit/core/base/check.hpp"
#include "volumetric_kit/recon/io/mesh_io.hpp"
#include "volumetric_kit/recon/volume/grid_growth.hpp"

namespace vr_example {
namespace {

namespace io = vr::io;

// A pattern split at its number, each `%%` already made `%`.
struct Pattern {
  std::string head;
  std::string tail;
  int width = -1;  // the digits the number pads to; -1 with no number
};

vkc::Result<Pattern> parse_pattern(const std::string& text) {
  Pattern pattern;
  std::string* out = &pattern.head;
  for (std::size_t i = 0; i < text.size(); ++i) {
    if (text[i] != '%') {
      out->push_back(text[i]);
      continue;
    }
    const std::size_t rest = text.size() - i - 1;
    if (rest >= 1 && text[i + 1] == '%') {
      out->push_back('%');
      ++i;
      continue;
    }
    int width = 0;
    if (rest >= 1 && text[i + 1] == 'd') {
      i += 1;
    } else if (rest >= 3 && text[i + 1] == '0' && text[i + 2] >= '1' &&
               text[i + 2] <= '9' && text[i + 3] == 'd') {
      width = text[i + 2] - '0';
      i += 3;
    } else {
      return vkc::Status::invalid_argument(
          "source.path: a '%' must begin %d, %0Nd (N from 1 to 9) or %%: " +
          text);
    }
    if (pattern.width >= 0) {
      return vkc::Status::invalid_argument(
          "source.path: more than one frame number: " + text);
    }
    pattern.width = width;
    out = &pattern.tail;
  }
  return pattern;
}

std::string expand(const Pattern& pattern, std::uint32_t number) {
  if (pattern.width < 0) return pattern.head;
  std::string digits = std::to_string(number);
  if (digits.size() < std::size_t(pattern.width)) {
    digits.insert(0, std::size_t(pattern.width) - digits.size(), '0');
  }
  return pattern.head + digits + pattern.tail;
}

std::string frame_name(std::uint32_t number, const std::string& path) {
  return "frame " + std::to_string(number) + " (" + path + ")";
}

std::string range_text(const MeshSequence& sequence) {
  return sequence.count() == 1 ? "frame " + std::to_string(sequence.first())
                               : "frames " + std::to_string(sequence.first()) +
                                     ".." + std::to_string(sequence.last());
}

void print_bounds(const char* label, const io::MeshBounds& b) {
  std::printf("%s: x [%.9g, %.9g], y [%.9g, %.9g], z [%.9g, %.9g]\n", label,
              b.min[0], b.max[0], b.min[1], b.max[1], b.min[2], b.max[2]);
}

void print_vector(const char* label, const vr::Vec3d& v) {
  std::printf("%s (%.17g, %.17g, %.17g)", label, v.x, v.y, v.z);
}

}  // namespace

vkc::Result<std::string> numbered_path(const std::string& pattern,
                                       std::uint32_t number) {
  VKC_ASSIGN(const Pattern parsed, parse_pattern(pattern));
  return expand(parsed, number);
}

vkc::Result<MeshSequence> MeshSequence::open(const config::Config& settings) {
  const config::Source& source = settings.source;
  const config::Run& run = settings.run;
  VKC_ASSIGN(const Pattern pattern, parse_pattern(source.path));
  if (run.first < 0 || run.frames < 1) {
    return vkc::Status::invalid_argument(
        "run: first must be at least 0 and frames at least 1");
  }
  if (pattern.width < 0 && run.first != 0) {
    return vkc::Status::invalid_argument(
        "run.first: source.path has no frame number (%d or %0Nd) to select");
  }
  if (pattern.width < 0 && run.frames > 1) {
    return vkc::Status::invalid_argument(
        "run.frames: source.path has no frame number (%d or %0Nd) to count");
  }
  MeshSequence sequence;
  sequence.pattern_ = source.path;
  sequence.first_ = std::uint32_t(run.first);
  sequence.signed_ = source.sdf_mode == config::SdfMode::Signed;
  sequence.allow_holes_ = source.allow_holes;
  sequence.by_height_ = !(source.metres_per_unit > 0.0);
  for (int i = 0; i < run.frames; ++i) {
    std::string path = expand(pattern, sequence.first_ + std::uint32_t(i));
    const vkc::Status file = io::check_mesh_file(path);
    if (!file.ok()) return file.with_context("source.path");
    sequence.paths_.push_back(std::move(path));
  }

  const auto& up = *source.up_vector;
  VKC_ASSIGN(const io::MeshAxes axes, io::up_axes({up[0], up[1], up[2]}));
  sequence.upright_.axes = axes;
  for (std::uint32_t n = sequence.first(), i = 0; i < sequence.count();
       ++i, ++n) {
    const std::string& path = sequence.path(n);
    vkc::Result<io::TriangleMesh> mesh = io::load_mesh(path);
    if (!mesh.ok()) return mesh.status().with_context(frame_name(n, path));
    io::extend_bounds(sequence.input_, *mesh);
    io::extend_bounds(sequence.upright_, *mesh);
    const vkc::Result<io::MeshTopology> topology =
        io::audit_mesh_topology(*mesh);
    if (!topology.ok()) {
      return topology.status().with_context(frame_name(n, path));
    }
    if (!settings.codec.inspect_only) VKC_TRY(sequence.admit(n, *topology));
  }
  vkc::Result<io::MeshTransform> transform =
      sequence.by_height_
          ? io::fit_transform(sequence.upright_, source.height_m)
          : io::scale_transform(sequence.upright_, source.metres_per_unit);
  if (!transform.ok()) {
    return transform.status().with_context(range_text(sequence));
  }
  sequence.transform_ = *transform;
  return sequence;
}

const std::string& MeshSequence::path(std::uint32_t number) const {
  VKC_CHECK(number >= first_ && number - first_ < paths_.size(),
            "MeshSequence::path: a number outside the range");
  return paths_[number - first_];
}

void MeshSequence::print() const {
  std::printf("mesh: %s of %s; geometry only, physical scale set by %s\n",
              range_text(*this).c_str(), pattern_.c_str(),
              by_height_ ? "source.height_m" : "source.metres_per_unit");
  print_bounds("input bounds (source units)", input_);
  const vr::Vec3d& up = transform_.axes.up;
  std::printf(
      "head-up (input coordinates): (%.9g, %.9g, %.9g); projected height "
      "%.12g source units\n",
      up.x, up.y, up.z, upright_.max.y - upright_.min.y);
  std::printf(
      "normalization: %.12g metres/source-unit; rotate head-up to +Y, centre "
      "X/Z, feet at Y=0; one transform for every frame\n",
      transform_.metres_per_unit);
  print_vector("transform: x", transform_.axes.x);
  print_vector(", up", transform_.axes.up);
  print_vector(", z", transform_.axes.z);
  print_vector("; origin", transform_.origin);
  std::printf(" source units; %.17g metres/source-unit\n",
              transform_.metres_per_unit);
  io::MeshBounds placed;
  for (int a = 0; a < 3; ++a) {
    placed.min[a] =
        (upright_.min[a] - transform_.origin[a]) * transform_.metres_per_unit;
    placed.max[a] =
        (upright_.max[a] - transform_.origin[a]) * transform_.metres_per_unit;
  }
  print_bounds("normalized bounds (metres)", placed);
}

vkc::Result<MeshFrame> MeshSequence::load(std::uint32_t number) const {
  MeshFrame frame;
  frame.number = number;
  frame.path = path(number);
  const std::string name = frame_name(number, frame.path);
  vkc::Result<io::TriangleMesh> mesh = io::load_mesh(frame.path);
  if (!mesh.ok()) return mesh.status().with_context(name);
  frame.mesh = std::move(mesh).value();
  const vkc::Status placed = io::apply_transform(frame.mesh, transform_);
  if (!placed.ok()) return placed.with_context(name);
  vkc::Result<io::MeshTopology> topology = io::audit_mesh_topology(frame.mesh);
  if (!topology.ok()) return topology.status().with_context(name);
  frame.topology = *topology;
  return frame;
}

vkc::Status MeshSequence::admit(std::uint32_t number,
                                const io::MeshTopology& topology) const {
  if (!signed_ || topology.supports_signed(allow_holes_)) return {};
  const std::string what = frame_name(number, path(number)) + ": " +
                           topology_defects(topology) + "; ";
  if (topology.supports_signed(true)) {
    return vkc::Status::invalid_argument(
        what +
        "signed mode refuses holes unless source.allow_holes is true; or "
        "choose source.sdf_mode=shell");
  }
  return vkc::Status::invalid_argument(
      what +
      "indexed topology does not support signed mode; repair the mesh or "
      "choose source.sdf_mode=shell");
}

std::string topology_defects(const io::MeshTopology& t) {
  if (t.components == 0) return "no triangles";
  std::string out;
  const auto add = [&out](std::size_t count, const char* what) {
    if (count == 0) return;
    if (!out.empty()) out += ", ";
    out += std::to_string(count) + " " + what;
  };
  add(t.boundary_edges, "boundary edges");
  add(t.nonmanifold_edges, "nonmanifold edges");
  add(t.inconsistent_edges, "inconsistently oriented edges");
  add(t.degenerate_triangles, "degenerate triangles");
  add(t.nonpositive_components, "components with nonpositive volume");
  return out;
}

void print_frame(const MeshFrame& frame) {
  const io::MeshTopology& t = frame.topology;
  std::printf("frame %u: %zu vertices, %zu triangles\n", frame.number,
              frame.mesh.positions.size(), frame.mesh.indices.size() / 3);
  std::printf(
      "indexed topology: %zu edges, %zu boundary, %zu nonmanifold, %zu "
      "inconsistently oriented; %zu degenerate triangles, %zu components (%zu "
      "nonpositive signed volumes); signed volume %.12g cubic metres; "
      "self-intersections not checked\n",
      t.edges, t.boundary_edges, t.nonmanifold_edges, t.inconsistent_edges,
      t.degenerate_triangles, t.components, t.nonpositive_components,
      t.signed_volume);
}

vkc::Result<Inspection> inspect_frames(const MeshSequence& sequence) {
  Inspection result;
  for (std::uint32_t n = sequence.first(), i = 0; i < sequence.count();
       ++i, ++n) {
    VKC_ASSIGN(result.last, sequence.load(n));
    const std::string defects = topology_defects(result.last.topology);
    if (defects.empty()) continue;
    const bool holes = result.last.topology.supports_signed(true);
    ++(holes ? result.open : result.refused);
    std::printf("frame %u: %s; signed mode %s\n", n, defects.c_str(),
                holes ? "needs source.allow_holes" : "refuses it");
  }
  std::printf(
      "inspection: %u frame(s); %zu open, which signed mode takes with "
      "source.allow_holes; %zu it refuses\n",
      sequence.count(), result.open, result.refused);
  return result;
}

vkc::Result<vr::tsdf::MeshIntegrateStats> convert_frame(
    vr::volume::VoxelBlockGrid& grid, vr::tsdf::MeshIntegrator& integrator,
    const io::TriangleMesh& mesh, const vr::tsdf::MeshSdfParams& params,
    vkc::StageMetrics* metrics) {
  VKC_TRY(grid.clear());
  const auto vertices = std::uint32_t(mesh.positions.size());
  const auto triangles = std::uint32_t(mesh.indices.size() / 3);
  // Grow only for a capacity limit. Lost bucket-lock races leave a residue
  // over a table with room, which a retry places: the mesh integrator refuses
  // a band with a block missing.
  for (int contended = 0;;) {
    vr::volume::AllocFailures failures;
    VKC_ASSIGN(const auto failed,
               grid.map().allocate_from_triangles(mesh.positions.data(),
                                                  vertices, mesh.indices.data(),
                                                  triangles, &failures));
    if (failed == 0) break;
    if (!failures.capacity_limited()) {
      if (++contended == 5) {
        return vkc::Status::io_error(
            "mesh allocation kept losing bucket-lock races");
      }
      continue;
    }
    VKC_TRY(vr::volume::grow_grid(grid));
  }
  return integrator.integrate(grid, mesh.positions.data(), vertices,
                              mesh.indices.data(), triangles, params, metrics);
}

vkc::Result<CodedFrame> code_frame(const MeshSequence& sequence,
                                   std::uint32_t number,
                                   vr::volume::VoxelBlockGrid& grid,
                                   vr::tsdf::MeshIntegrator& integrator,
                                   const vr::tsdf::MeshSdfParams& params,
                                   CodecStream& stream,
                                   vkc::StageMetrics* conversion) {
  CodedFrame coded;
  VKC_ASSIGN(coded.frame, sequence.load(number));
  VKC_TRY(sequence.admit(number, coded.frame.topology));
  const std::string name = frame_name(number, coded.frame.path);
  vkc::Result<vr::tsdf::MeshIntegrateStats> stats =
      convert_frame(grid, integrator, coded.frame.mesh, params, conversion);
  if (!stats.ok()) return stats.status().with_context(name);
  coded.stats = *stats;
  const vkc::Status recorded =
      stream.code(grid, number, frame_pts_ns(number, kNominalFps));
  if (!recorded.ok()) return recorded.with_context(name);
  return coded;
}

}  // namespace vr_example
