// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// codec_mesh's sequence (examples/codec_mesh/mesh_sequence.hpp), on synthetic
// OBJ frames. On the host: the frame-number pattern and its refusals; every
// file checked before any load; one transform fitted to the whole range, so
// a vertex that does not move keeps its place, where frame 1 alone would fit
// wrong; a plain path placed as one mesh was before; a % in the scenario's
// directory, resolved through config::load, read as written; signed mode's
// rule and source.allow_holes, refused before any GPU work; inspection over a
// range and its counts. On a device: each frame converted into a cleared grid,
// its allocation in the conversion's metrics, and code_frame's records -- set
// the frame's number, pts that number at 30 fps, seq from 0, each payload what
// a fresh encoder makes of the frame alone, the transform in the track's params
// and 30 a second its rate -- and its own audit and allow_holes. The device
// part skips with no device.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "codec_fixture.hpp"
#include "gpu_test.hpp"
#include "grid_layout.hpp"
#include "mesh_sequence.hpp"
#include "scratch_dir.hpp"
#include "volumetric_kit/recon/codec/decode_grid.hpp"
#include "volumetric_kit/recon/codec/decoder.hpp"
#include "volumetric_kit/recon/codec/encoder.hpp"
#include "volumetric_kit/recon/io/mesh_io.hpp"
#include "volumetric_kit/recon/stream/stream_reader.hpp"

namespace vkc = volumetric_kit::core;
namespace cfg = volumetric_kit::core::config;
namespace io = volumetric_kit::recon::io;
namespace vol = volumetric_kit::recon::volume;
namespace codec = volumetric_kit::recon::codec;
namespace stream = volumetric_kit::recon::stream;
namespace ts = volumetric_kit::recon::tsdf;
namespace fs = std::filesystem;
using codec_fixture::Snapshot;
using Code = vkc::Status::Code;
using volumetric_kit::recon::Vec3d;
using volumetric_kit::recon::Vec3f;

namespace {

// An axis-aligned cube, wound outward: vertex i at centre + half * (bit 0, bit
// 1, bit 2 of i, each as -1 or +1).
io::TriangleMesh cube(const Vec3f& centre, float half) {
  io::TriangleMesh mesh;
  for (int i = 0; i < 8; ++i) {
    mesh.positions.push_back({centre.x + ((i & 1) ? half : -half),
                              centre.y + ((i & 2) ? half : -half),
                              centre.z + ((i & 4) ? half : -half)});
  }
  mesh.indices = {0, 2, 3, 0, 3, 1, 4, 5, 7, 4, 7, 6, 0, 1, 5, 0, 5, 4,
                  2, 6, 7, 2, 7, 3, 0, 4, 6, 0, 6, 2, 1, 3, 7, 1, 7, 5};
  return mesh;
}

// A tetrahedron wound outward with apex 3 at `apex`; the base stays put.
io::TriangleMesh tetrahedron(const Vec3f& apex) {
  return {{{0, 0, 0}, {1, 0, 0}, {0, 0, 1}, apex},
          {0, 1, 2, 0, 3, 1, 0, 2, 3, 1, 3, 2}};
}

bool write_obj(const fs::path& path, const io::TriangleMesh& mesh) {
  std::ofstream out(path);
  out << std::setprecision(9);  // every float back as written
  for (const Vec3f& p : mesh.positions) {
    out << "v " << p.x << ' ' << p.y << ' ' << p.z << '\n';
  }
  for (std::size_t t = 0; t < mesh.indices.size(); t += 3) {
    out << "f " << mesh.indices[t] + 1 << ' ' << mesh.indices[t + 1] + 1 << ' '
        << mesh.indices[t + 2] + 1 << '\n';
  }
  return bool(out);
}

// A CodecMesh scenario over `path`, with `extra` source keys and `rest`
// after the source, resolved against `dir`.
vkc::Result<cfg::Config> scenario(const fs::path& dir, const std::string& path,
                                  const std::string& extra = {},
                                  const std::string& rest = {}) {
  return cfg::parse(R"({"version": 1, "operation": "codec",
      "source": {"kind": "mesh", "path": ")" +
                        path + R"(", "up_axis": "y")" + extra + "}" + rest +
                        "}",
                    cfg::Application::CodecMesh, dir.string());
}

bool refused(const vkc::Status& status, Code code, const std::string& named) {
  if (status.domain() == code &&
      status.message().find(named) != std::string::npos) {
    return true;
  }
  std::fprintf(stderr, "expected %d naming \"%s\", got %d: %s\n", int(code),
               named.c_str(), int(status.domain()), status.message().c_str());
  return false;
}

bool same_bits(const io::TriangleMesh& a, const io::TriangleMesh& b) {
  return a.positions.size() == b.positions.size() && a.indices == b.indices &&
         std::memcmp(a.positions.data(), b.positions.data(),
                     a.positions.size() * sizeof(Vec3f)) == 0;
}

bool same(const io::MeshTransform& a, const io::MeshTransform& b) {
  return a.axes.x == b.axes.x && a.axes.up == b.axes.up &&
         a.axes.z == b.axes.z && a.origin == b.origin &&
         a.metres_per_unit == b.metres_per_unit;
}

int patterns() {
  const struct {
    const char* pattern;
    std::uint32_t number;
    const char* path;
  } expanded[] = {
      {"Frame_%05d_textured.obj", 1, "Frame_00001_textured.obj"},
      {"%d", 150, "150"},
      {"%01d.obj", 0, "0.obj"},
      {"%03d", 1234, "1234"},
      {"%09d", 4294967295u, "4294967295"},
      {"100%%_%02d.obj", 7, "100%_07.obj"},
      {"seq_%02d/mesh.obj", 3, "seq_03/mesh.obj"},
      {"plain.obj", 9, "plain.obj"},
      {"50%%.obj", 9, "50%.obj"},
      // Core escapes the scenario's directory, so a % in it reads as itself.
      {"pct/50%%off/cube.obj", 9, "pct/50%off/cube.obj"},
      {"pct/take%%d/mesh.obj", 3, "pct/take%d/mesh.obj"},
      {"pct/take%%d/m_%02d.obj", 3, "pct/take%d/m_03.obj"},
  };
  for (const auto& e : expanded) {
    const vkc::Result<std::string> path =
        vr_example::numbered_path(e.pattern, e.number);
    CHECK(path.ok() && *path == e.path);
  }
  for (const char* bad :
       {"a_%d_%d.obj", "%02d_%d", "%s", "%5d", "%-5d", "%0d", "%00d", "%010d",
        "%05", "%", "x%", "%%%", "%ld", "%u", "pct/50%off/cube.obj"}) {
    CHECK(refused(vr_example::numbered_path(bad, 1).status(),
                  Code::InvalidArgument, "source.path"));
  }
  return 0;
}

// Files are checked, every one, before the first is loaded; a plain path
// takes no range.
int refusals(const vr_test::Scratch& scratch) {
  const fs::path dir = scratch.path / "refusals";
  CHECK(fs::create_directory(dir));
  {
    std::ofstream(dir / "shape_01.obj") << "not an obj\n";
  }
  CHECK(write_obj(dir / "plain.obj", cube({0, 0, 0}, 0.5f)));
  const auto open = [&dir](const std::string& path, const std::string& run) {
    const vkc::Result<cfg::Config> settings =
        scenario(dir, path, R"(, "height_m": 1.7)", run);
    if (!settings.ok()) return settings.status();
    return vr_example::MeshSequence::open(*settings).status();
  };
  // Frame 2 is missing, and frame 1 is not a mesh: NotFound for frame 2
  // shows no frame was loaded first.
  CHECK(refused(open("shape_%02d.obj", R"(, "run": {"first": 1, "frames": 2})"),
                Code::NotFound, "shape_02.obj"));
  CHECK(fs::create_directory(dir / "shape_02.obj"));
  CHECK(refused(open("shape_%02d.obj", R"(, "run": {"first": 1, "frames": 2})"),
                Code::InvalidArgument, "shape_02.obj"));
  // Frame 1 alone is checked, then loaded, and named when it fails.
  const vkc::Status unreadable =
      open("shape_%02d.obj", R"(, "run": {"first": 1})");
  CHECK(!unreadable.ok() &&
        unreadable.message().find("frame 1") != std::string::npos);
  CHECK(
      refused(open("shape_%s.obj", ""), Code::InvalidArgument, "source.path"));
  CHECK(refused(open("plain.obj", R"(, "run": {"frames": 2})"),
                Code::InvalidArgument, "run.frames"));
  CHECK(refused(open("plain.obj", R"(, "run": {"first": 1})"),
                Code::InvalidArgument, "run.first"));
  CHECK(open("plain.obj", R"(, "run": {"frames": 1, "first": 0})").ok());
  return 0;
}

// A tetrahedron whose apex climbs: its base never moves, and frame 3 is the
// tallest. One transform fits all three, so the base keeps its place, bit
// for bit, and the range is height_m tall; frame 1's own fit would scale it
// to height_m alone and frame 3 past it.
int one_transform(const vr_test::Scratch& scratch) {
  const fs::path dir = scratch.path / "growing";
  CHECK(fs::create_directory(dir));
  const Vec3f apexes[] = {
      {0.2f, 1.0f, 0.3f}, {0.6f, 2.0f, 0.1f}, {0.4f, 3.0f, 0.5f}};
  for (int i = 0; i < 3; ++i) {
    char name[32];
    std::snprintf(name, sizeof name, "frame_%03d.obj", 4 + i);
    CHECK(write_obj(dir / name, tetrahedron(apexes[i])));
  }
  vkc::Result<cfg::Config> settings =
      scenario(dir, "frame_%03d.obj", R"(, "height_m": 1.5)",
               R"(, "run": {"first": 4, "frames": 3})");
  CHECK(settings.ok());
  vkc::Result<vr_example::MeshSequence> opened =
      vr_example::MeshSequence::open(*settings);
  CHECK(opened.ok());
  const vr_example::MeshSequence& sequence = *opened;
  CHECK(sequence.first() == 4 && sequence.count() == 3 && sequence.last() == 6);
  CHECK(sequence.path(5) == (dir / "frame_005.obj").string());
  sequence.print();

  // The transform is the fit to the union of every frame's bounds.
  io::MeshBounds all = io::mesh_bounds({}, io::up_axes({0, 1, 0}).value());
  for (int i = 0; i < 3; ++i) io::extend_bounds(all, tetrahedron(apexes[i]));
  const vkc::Result<io::MeshTransform> fit = io::fit_transform(all, 1.5);
  CHECK(fit.ok() && same(sequence.transform(), *fit));
  const vkc::Result<io::MeshTransform> alone =
      io::fit_transform(io::mesh_bounds(tetrahedron(apexes[0]), all.axes), 1.5);
  CHECK(alone.ok() && !same(sequence.transform(), *alone));

  std::vector<io::TriangleMesh> placed;
  float top = 0.0f;
  for (std::uint32_t n = 4; n <= 6; ++n) {
    vkc::Result<vr_example::MeshFrame> frame = sequence.load(n);
    CHECK(frame.ok() && frame->number == n);
    CHECK(frame->topology.supports_signed());
    for (const Vec3f& p : frame->mesh.positions) top = std::max(top, p.y);
    placed.push_back(std::move(frame->mesh));
  }
  // The base, vertices 0-2, keeps its place; the apex moves.
  for (const io::TriangleMesh& mesh : placed) {
    CHECK(std::memcmp(mesh.positions.data(), placed[0].positions.data(),
                      3 * sizeof(Vec3f)) == 0);
  }
  CHECK(!(placed[1].positions[3] == placed[0].positions[3]));
  // Frame 3 reaches the range's height, frame 1 only a third of it.
  CHECK(std::abs(top - 1.5f) < 1e-6f);
  CHECK(std::abs(placed[0].positions[3].y - 0.5f) < 1e-6f);
  // The same frame loads to the same bits again.
  vkc::Result<vr_example::MeshFrame> again = sequence.load(4);
  CHECK(again.ok() && same_bits(again->mesh, placed[0]));

  // A unit instead of a height: the scale as given, placed the same way.
  vkc::Result<cfg::Config> unit =
      scenario(dir, "frame_%03d.obj", R"(, "metres_per_unit": 0.25)",
               R"(, "run": {"first": 4, "frames": 3})");
  CHECK(unit.ok());
  vkc::Result<vr_example::MeshSequence> scaled =
      vr_example::MeshSequence::open(*unit);
  CHECK(scaled.ok());
  const vkc::Result<io::MeshTransform> by_unit = io::scale_transform(all, 0.25);
  CHECK(by_unit.ok() && same(scaled->transform(), *by_unit));

  // Inspection makes the second pass on the host and returns the last frame.
  vkc::Result<vr_example::Inspection> inspected =
      vr_example::inspect_frames(sequence);
  CHECK(inspected.ok() && inspected->last.number == 6 &&
        same_bits(inspected->last.mesh, placed[2]));
  CHECK(inspected->open == 0 && inspected->refused == 0);
  return 0;
}

// A plain path is one frame, number 0, placed exactly as one mesh: the fit to
// its own bounds, applied to it.
int single_mesh(const vr_test::Scratch& scratch) {
  const fs::path dir = scratch.path / "single";
  CHECK(fs::create_directory(dir));
  const io::TriangleMesh mesh = tetrahedron({0.3f, 1.25f, 0.2f});
  CHECK(write_obj(dir / "person.obj", mesh));
  vkc::Result<cfg::Config> settings = cfg::parse(
      R"({"version": 1, "operation": "codec",
          "source": {"kind": "mesh", "path": "person.obj", "height_m": 1.7,
                     "up_vector": [-0.9120591159, 0.0661250017,
                                   -0.4046920474]}})",
      cfg::Application::CodecMesh, dir.string());
  CHECK(settings.ok());
  vkc::Result<vr_example::MeshSequence> sequence =
      vr_example::MeshSequence::open(*settings);
  CHECK(sequence.ok());
  CHECK(sequence->first() == 0 && sequence->count() == 1);

  // What codec_mesh did with one mesh before it took a range.
  vkc::Result<io::TriangleMesh> loaded = io::load_mesh(sequence->path(0));
  CHECK(loaded.ok());
  const auto& up = *settings->source.up_vector;
  const vkc::Result<io::MeshAxes> axes = io::up_axes({up[0], up[1], up[2]});
  CHECK(axes.ok());
  const vkc::Result<io::MeshTransform> fit =
      io::fit_transform(io::mesh_bounds(*loaded, *axes), 1.7);
  CHECK(fit.ok() && same(sequence->transform(), *fit));
  CHECK(io::apply_transform(*loaded, *fit).ok());
  vkc::Result<vr_example::MeshFrame> frame = sequence->load(0);
  CHECK(frame.ok() && frame->number == 0 && same_bits(frame->mesh, *loaded));
  return 0;
}

// Signed mode needs a closed, consistently wound mesh; source.allow_holes
// admits boundary edges alone. Refused before any GPU work, by the first
// pass, unless inspecting; shell mode takes anything.
int signed_rule(const vr_test::Scratch& scratch) {
  const fs::path dir = scratch.path / "signed";
  CHECK(fs::create_directory(dir));
  io::TriangleMesh open = cube({0, 0, 0}, 0.5f);
  open.indices.resize(open.indices.size() - 6);  // the +x face
  io::TriangleMesh book = cube({0, 0, 0}, 0.5f);
  book.positions.push_back({0.0f, 2.0f, 0.0f});
  book.indices.insert(book.indices.end(), {0, 2, 8});  // a third on edge 0-2
  CHECK(write_obj(dir / "m_1.obj", cube({0, 0, 0}, 0.5f)));
  CHECK(write_obj(dir / "m_2.obj", open));
  CHECK(write_obj(dir / "m_3.obj", book));
  const auto open_range = [&dir](int first, int frames,
                                 const std::string& extra,
                                 const std::string& codec = {}) {
    const vkc::Result<cfg::Config> settings =
        scenario(dir, "m_%d.obj", R"(, "height_m": 1.7)" + extra,
                 R"(, "run": {"first": )" + std::to_string(first) +
                     R"(, "frames": )" + std::to_string(frames) + "}" + codec);
    if (!settings.ok()) return settings.status();
    return vr_example::MeshSequence::open(*settings).status();
  };
  CHECK(open_range(1, 1, "").ok());
  const vkc::Status holes = open_range(1, 2, "");
  CHECK(refused(holes, Code::InvalidArgument, "source.allow_holes"));
  CHECK(refused(holes, Code::InvalidArgument, "frame 2"));
  CHECK(refused(holes, Code::InvalidArgument, "4 boundary edges"));
  CHECK(open_range(1, 2, R"(, "allow_holes": true)").ok());
  CHECK(open_range(1, 2, R"(, "sdf_mode": "shell")").ok());
  // Non-manifold is refused even with holes allowed, and named.
  const vkc::Status nonmanifold = open_range(2, 2, R"(, "allow_holes": true)");
  CHECK(refused(nonmanifold, Code::InvalidArgument, "frame 3"));
  CHECK(refused(nonmanifold, Code::InvalidArgument, "nonmanifold"));
  CHECK(nonmanifold.message().find("allow_holes") == std::string::npos);
  CHECK(open_range(2, 2, R"(, "sdf_mode": "shell")").ok());
  // Inspection reports instead.
  CHECK(open_range(1, 3, "", R"(, "codec": {"inspect_only": true})").ok());

  // admit is the rule the second pass applies too, on placed coordinates.
  vkc::Result<cfg::Config> settings =
      scenario(dir, "m_%d.obj", R"(, "height_m": 1.7)",
               R"(, "run": {"first": 1, "frames": 3},
                  "codec": {"inspect_only": true})");
  CHECK(settings.ok());
  vkc::Result<vr_example::MeshSequence> sequence =
      vr_example::MeshSequence::open(*settings);
  CHECK(sequence.ok());
  for (std::uint32_t n = 1; n <= 3; ++n) {
    vkc::Result<vr_example::MeshFrame> frame = sequence->load(n);
    CHECK(frame.ok());
    CHECK(sequence->admit(n, frame->topology).ok() == (n == 1));
  }
  CHECK(vr_example::topology_defects(sequence->load(2)->topology) ==
        "4 boundary edges");
  // One open frame signed mode takes with allow_holes, one it refuses.
  vkc::Result<vr_example::Inspection> inspected =
      vr_example::inspect_frames(*sequence);
  CHECK(inspected.ok() && inspected->last.number == 3);
  CHECK(inspected->open == 1 && inspected->refused == 1);
  return 0;
}

// Load real scenarios beneath literal % directories. Plain and numbered
// paths must select their own meshes, including when take0 holds a decoy.
int escaped_directory(const vr_test::Scratch& scratch) {
  const fs::path base = scratch.path / "pct";
  CHECK(fs::create_directories(base / "take0"));
  for (const char* file : {"shape.obj", "m_01.obj", "m_02.obj"}) {
    CHECK(write_obj(base / "take0" / file, cube({0, 0, 0}, 2.0f)));
  }
  for (const char* name : {"50%off", "take%d"}) {
    const fs::path dir = base / name;
    CHECK(fs::create_directory(dir));
    for (const char* file : {"shape.obj", "m_01.obj", "m_02.obj"}) {
      CHECK(write_obj(dir / file, cube({0, 0, 0}, 0.5f)));
    }
    for (const bool numbered : {false, true}) {
      const fs::path config_path = dir / "inspect.json";
      {
        std::ofstream out(config_path);
        out << R"({"version": 1, "operation": "codec",
            "source": {"kind": "mesh", "path": ")"
            << (numbered ? "m_%02d.obj" : "shape.obj")
            << R"(", "metres_per_unit": 1, "up_axis": "y"},
            "codec": {"inspect_only": true})"
            << (numbered ? R"(, "run": {"first": 1, "frames": 2})" : "")
            << "}\n";
        out.close();
        CHECK(bool(out));
      }
      const vkc::Result<cfg::Config> settings =
          cfg::load(config_path.string(), cfg::Application::CodecMesh);
      CHECK(settings.ok());
      const vkc::Result<vr_example::MeshSequence> sequence =
          vr_example::MeshSequence::open(*settings);
      if (!sequence.ok())
        std::fprintf(stderr, "%s: %s\n", config_path.string().c_str(),
                     sequence.status().message().c_str());
      CHECK(sequence.ok());
      CHECK(sequence->first() == (numbered ? 1u : 0u));
      CHECK(sequence->count() == (numbered ? 2u : 1u));
      for (std::uint32_t n = sequence->first(); n <= sequence->last(); ++n) {
        const char* file =
            numbered ? (n == 1 ? "m_01.obj" : "m_02.obj") : "shape.obj";
        CHECK(sequence->path(n) == (dir / file).string());
        const vkc::Result<vr_example::MeshFrame> frame = sequence->load(n);
        CHECK(frame.ok() && frame->number == n);
        float top = 0.0f;
        for (const Vec3f& p : frame->mesh.positions) top = std::max(top, p.y);
        // The unit cube, never take0's four-unit decoy.
        CHECK(std::abs(top - 1.0f) < 1e-6f);
      }
    }
  }
  return 0;
}

// A unit tetrahedron beside one 1e-15 across. At 1e-31 metres a unit the
// small one's corners round to one float, so its triangles are degenerate
// only once placed: the first pass, on source coordinates, takes the frame,
// and the second pass's audit, which decides, refuses it.
vkc::Result<vr_example::MeshSequence> collapsing(const fs::path& dir,
                                                 const std::string& codec) {
  std::error_code ignored;
  fs::create_directory(dir, ignored);
  io::TriangleMesh mesh = tetrahedron({0.25f, 1.0f, 0.25f});
  const io::TriangleMesh unit = tetrahedron({0.2f, 1.0f, 0.2f});
  const auto base = std::uint32_t(mesh.positions.size());
  for (const Vec3f& p : unit.positions) {
    mesh.positions.push_back(Vec3f(1e-12f) + 1e-15f * p);
  }
  for (const std::uint32_t i : unit.indices) mesh.indices.push_back(base + i);
  if (!write_obj(dir / "f_7.obj", mesh)) {
    return vkc::Status::io_error("cannot write f_7.obj");
  }
  VKC_ASSIGN(const cfg::Config settings,
             scenario(dir, "f_%d.obj", R"(, "metres_per_unit": 1e-31)",
                      R"(, "run": {"first": 7})" + codec));
  return vr_example::MeshSequence::open(settings);
}

int placed_audit_decides(const vr_test::Scratch& scratch) {
  vkc::Result<vr_example::MeshSequence> sequence =
      collapsing(scratch.path / "collapsing", "");
  CHECK(sequence.ok());
  vkc::Result<vr_example::MeshFrame> frame = sequence->load(7);
  CHECK(frame.ok() && frame->topology.degenerate_triangles == 4);
  CHECK(refused(sequence->admit(7, frame->topology), Code::InvalidArgument,
                "4 degenerate triangles"));
  return 0;
}

// The scenario over `count` cubes on disk, each `half` across, frame n's
// centre at `centre(n)`, numbered from `first`, in metres as given.
template <typename Centre>
vkc::Result<vr_example::MeshSequence> cubes(const fs::path& dir,
                                            std::uint32_t first, int count,
                                            float half, Centre centre) {
  std::error_code ignored;
  fs::create_directory(dir, ignored);
  for (int i = 0; i < count; ++i) {
    const std::uint32_t n = first + std::uint32_t(i);
    char name[32];
    std::snprintf(name, sizeof name, "cube_%02u.obj", n);
    if (!write_obj(dir / name, cube(centre(n), half))) {
      return vkc::Status::io_error("cannot write " + std::string(name));
    }
  }
  VKC_ASSIGN(const cfg::Config settings,
             scenario(dir, "cube_%02d.obj", R"(, "metres_per_unit": 1)",
                      R"(, "run": {"first": )" + std::to_string(first) +
                          R"(, "frames": )" + std::to_string(count) + "}"));
  return vr_example::MeshSequence::open(settings);
}

// A small grid in the examples' layout: too small for the cubes, so the
// first conversion grows it.
vkc::Result<vol::VoxelBlockGrid> small_grid(vr_test::GpuContext& gpu) {
  const vol::AttributeSpec attrs[] = {{"tsdf", sizeof(float)},
                                      {"weight", sizeof(float)}};
  return vol::VoxelBlockGrid::create(
      gpu.device, gpu.allocator,
      vr_example::example_grid_params(codec_fixture::kVoxel,
                                      codec_fixture::kTrunc, 2),
      attrs, 2);
}

// Each frame is converted into a cleared grid: no block of the frame before
// stays, and the grid holds what a fresh one would.
int clean_grids(vr_test::GpuContext& gpu, const vr_test::Scratch& scratch) {
  vkc::Result<vr_example::MeshSequence> sequence =
      cubes(scratch.path / "clean", 1, 2, 0.06f,
            [](std::uint32_t n) { return Vec3f(0.5f * float(n), 0.0f, 0.0f); });
  CHECK(sequence.ok());
  vkc::Result<vr_example::MeshFrame> a = sequence->load(1);
  vkc::Result<vr_example::MeshFrame> b = sequence->load(2);
  CHECK(a.ok() && b.ok());
  vkc::Result<ts::MeshIntegrator> integrator =
      ts::MeshIntegrator::create(gpu.device, gpu.allocator);
  CHECK(integrator.ok());
  const auto alone = [&](const io::TriangleMesh& mesh) -> Snapshot {
    vkc::Result<vol::VoxelBlockGrid> fresh = small_grid(gpu);
    if (!fresh.ok() ||
        !vr_example::convert_frame(*fresh, *integrator, mesh, {}, nullptr)
             .ok()) {
      return {};
    }
    vkc::Result<Snapshot> shot = codec_fixture::snapshot(gpu, *fresh);
    return shot.ok() ? *shot : Snapshot{};
  };
  const Snapshot only_a = alone(a->mesh);
  const Snapshot only_b = alone(b->mesh);
  CHECK(!only_a.coords.empty() && !only_b.coords.empty());
  CHECK(!(only_a == only_b));

  vkc::Result<vol::VoxelBlockGrid> grid = small_grid(gpu);
  CHECK(grid.ok());
  const std::int32_t slots = grid->grid().num_blocks;
  for (const io::TriangleMesh* mesh : {&a->mesh, &b->mesh, &a->mesh}) {
    vkc::Result<ts::MeshIntegrateStats> stats =
        vr_example::convert_frame(*grid, *integrator, *mesh, {}, nullptr);
    CHECK(stats.ok() && stats->blocks > 0);
    vkc::Result<Snapshot> shot = codec_fixture::snapshot(gpu, *grid);
    CHECK(shot.ok() && *shot == (mesh == &a->mesh ? only_a : only_b));
  }
  CHECK(grid->grid().num_blocks > slots);  // the first frame grew it
  return 0;
}

// code_frame over frames 5-7 of a cube moving a block a frame: a record per
// frame, set its number, pts that number at 30 fps, seq from 0, each payload
// what a fresh encoder makes of the frame converted alone, and the track's
// params the sequence's transform. The player ends holding frame 7's blocks.
int records(vr_test::GpuContext& gpu, const vr_test::Scratch& scratch) {
  const float block = 8.0f * codec_fixture::kVoxel;
  vkc::Result<vr_example::MeshSequence> sequence = cubes(
      scratch.path / "moving", 5, 3, 0.06f,
      [block](std::uint32_t n) { return Vec3f(block * float(n), 0.0f, 0.0f); });
  CHECK(sequence.ok());
  vkc::Result<std::string> params = io::encode_transform(sequence->transform());
  CHECK(params.ok());
  vr_example::CodecOutputs outputs;
  outputs.stream = (scratch.path / "moving" / "frames.vrst").string();
  vkc::Result<vr_example::CodecStream> made = vr_example::CodecStream::create(
      gpu.device, gpu.allocator, {}, outputs, "example_mesh_sequence",
      {vr_example::kNominalFps, 1}, *params);
  CHECK(made.ok());
  vr_example::CodecStream coder = std::move(made).value();
  vkc::Result<ts::MeshIntegrator> integrator =
      ts::MeshIntegrator::create(gpu.device, gpu.allocator);
  CHECK(integrator.ok());
  vkc::Result<vol::VoxelBlockGrid> grid = small_grid(gpu);
  CHECK(grid.ok());
  vkc::StageMetrics conversion;
  std::vector<std::vector<vol::BlockIndex>> blocks;
  for (std::uint32_t n = 5; n <= 7; ++n) {
    vkc::Result<vr_example::CodedFrame> coded = vr_example::code_frame(
        *sequence, n, *grid, *integrator, {}, coder, &conversion);
    CHECK(coded.ok() && coded->frame.number == n && coded->stats.blocks > 0);
    vkc::Result<std::vector<vol::BlockIndex>> active =
        codec_fixture::active_sorted(*grid);
    CHECK(active.ok());
    blocks.push_back(*active);
  }
  // The conversion reports both of its calls, the allocation first.
  CHECK(conversion.rows().size() >= 2);
  CHECK(std::string(conversion.rows()[0].name) == "allocate");
  CHECK(coder.finish().ok());
  CHECK(coder.rows().size() == 3);
  // The player holds frame 7's blocks, no more.
  vkc::Result<std::vector<vol::BlockIndex>> played =
      codec_fixture::active_sorted(*coder.player());
  CHECK(played.ok() && played->size() == coder.rows()[2].blocks);

  vkc::Result<std::unique_ptr<stream::StreamReader>> opened =
      stream::StreamReader::open(outputs.stream);
  CHECK(opened.ok());
  const stream::StreamReader& reader = **opened;
  CHECK(reader.recovery().finished);
  CHECK(reader.session().tracks.size() == 1);
  const stream::Track& track = reader.session().tracks[0];
  CHECK(track.kind == stream::TrackKind::Geometry && track.params == *params);
  CHECK(track.rate.num == 30 && track.rate.den == 1);
  vkc::Result<io::MeshTransform> stored = io::decode_transform(track.params);
  CHECK(stored.ok() && same(*stored, sequence->transform()));
  vkc::Result<codec::Encoder> enc =
      codec::Encoder::create(gpu.device, gpu.allocator, {});
  CHECK(enc.ok());
  stream::Position pos = reader.begin();
  stream::Record record;
  std::vector<std::uint8_t> payload;
  for (std::uint32_t i = 0; i < 3; ++i) {
    const std::uint32_t n = 5 + i;
    vkc::Result<bool> more = reader.next(&pos, &record);
    CHECK(more.ok() && *more);
    CHECK(record.info.track == 0 && record.info.key);
    CHECK(record.info.set == n && record.info.seq == i);
    // 30 fps, from frame 0 at 0, to the nearest ns.
    const std::uint64_t pts[] = {166'666'667, 200'000'000, 233'333'333};
    CHECK(record.info.pts_ns == pts[i]);
    CHECK(reader.read(record, &payload).ok());
    // The frame alone, in a fresh grid, through a fresh encoder.
    vkc::Result<vr_example::MeshFrame> frame = sequence->load(n);
    vkc::Result<vol::VoxelBlockGrid> fresh = small_grid(gpu);
    CHECK(frame.ok() && fresh.ok());
    CHECK(
        vr_example::convert_frame(*fresh, *integrator, frame->mesh, {}, nullptr)
            .ok());
    vkc::Result<codec::Encoder> once =
        codec::Encoder::create(gpu.device, gpu.allocator, {});
    CHECK(once.ok());
    vkc::Result<std::vector<std::uint8_t>> expected = once->encode(*fresh);
    CHECK(expected.ok() && payload == *expected);
    // It decodes alone, with a fresh decoder, to the blocks the run's
    // player held after it.
    vkc::Result<codec::Decoder> dec =
        codec::Decoder::create(gpu.device, gpu.allocator);
    vkc::Result<vol::VoxelBlockGrid> alone = codec::make_decode_grid(
        gpu.device, gpu.allocator, payload.data(), payload.size(),
        static_cast<std::uint32_t>(fresh->grid().num_blocks));
    CHECK(dec.ok() && alone.ok());
    CHECK(dec->decode(payload.data(), payload.size(), *alone).ok());
    vkc::Result<std::vector<vol::BlockIndex>> decoded =
        codec_fixture::active_sorted(*alone);
    CHECK(decoded.ok() && decoded->size() == coder.rows()[i].blocks);
    if (n == 7) {
      for (std::size_t b = 0; b < decoded->size(); ++b) {
        CHECK((*decoded)[b].coord == (*played)[b].coord);
      }
    }
  }
  vkc::Result<bool> end = reader.next(&pos, &record);
  CHECK(end.ok() && !*end);
  // The frames moved: each holds blocks the one before did not.
  CHECK(!(blocks[0].size() == blocks[1].size() &&
          blocks[0][0].coord == blocks[1][0].coord));
  return 0;
}

// code_frame applies the second pass's audit: a frame that only its placed
// coordinates make degenerate is refused before anything is coded.
int refused_once_placed(vr_test::GpuContext& gpu,
                        const vr_test::Scratch& scratch) {
  vkc::Result<vr_example::MeshSequence> sequence =
      collapsing(scratch.path / "collapsing_gpu", "");
  CHECK(sequence.ok());
  vkc::Result<vr_example::CodecStream> coder =
      vr_example::CodecStream::create(gpu.device, gpu.allocator, {});
  vkc::Result<ts::MeshIntegrator> integrator =
      ts::MeshIntegrator::create(gpu.device, gpu.allocator);
  vkc::Result<vol::VoxelBlockGrid> grid = small_grid(gpu);
  CHECK(coder.ok() && integrator.ok() && grid.ok());
  CHECK(refused(vr_example::code_frame(*sequence, 7, *grid, *integrator, {},
                                       *coder, nullptr)
                    .status(),
                Code::InvalidArgument, "frame 7"));
  CHECK(coder->rows().empty());
  return 0;
}

// The second pass applies allow_holes too: code_frame codes an open cube when
// the scenario allows holes, and refuses it, coding nothing, when it does not
// (opened for inspection, so the first pass let it through).
int holes_in_code_frame(vr_test::GpuContext& gpu,
                        const vr_test::Scratch& scratch) {
  const fs::path dir = scratch.path / "holes_gpu";
  CHECK(fs::create_directory(dir));
  io::TriangleMesh open = cube({0.05f, 0.0f, 0.0f}, 0.06f);
  open.indices.resize(open.indices.size() - 6);  // the +x face
  CHECK(write_obj(dir / "h_1.obj", open));
  vkc::Status coded[2];
  std::size_t rows[2] = {};
  for (int allowed = 0; allowed < 2; ++allowed) {
    vkc::Result<cfg::Config> settings =
        scenario(dir, "h_%d.obj",
                 std::string(R"(, "metres_per_unit": 1)") +
                     (allowed ? R"(, "allow_holes": true)" : ""),
                 std::string(R"(, "run": {"first": 1})") +
                     (allowed ? "" : R"(, "codec": {"inspect_only": true})"));
    CHECK(settings.ok());
    vkc::Result<vr_example::MeshSequence> sequence =
        vr_example::MeshSequence::open(*settings);
    vkc::Result<vr_example::CodecStream> coder =
        vr_example::CodecStream::create(gpu.device, gpu.allocator, {});
    vkc::Result<ts::MeshIntegrator> integrator =
        ts::MeshIntegrator::create(gpu.device, gpu.allocator);
    vkc::Result<vol::VoxelBlockGrid> grid = small_grid(gpu);
    CHECK(sequence.ok() && coder.ok() && integrator.ok() && grid.ok());
    coded[allowed] = vr_example::code_frame(*sequence, 1, *grid, *integrator,
                                            {}, *coder, nullptr)
                         .status();
    rows[allowed] = coder->rows().size();
  }
  CHECK(refused(coded[0], Code::InvalidArgument, "source.allow_holes"));
  CHECK(rows[0] == 0);
  CHECK(coded[1].ok() && rows[1] == 1);
  return 0;
}

// A plain path codes as one frame: set 0 at pts 0, seq 0.
int single_record(vr_test::GpuContext& gpu, const vr_test::Scratch& scratch) {
  const fs::path dir = scratch.path / "one";
  CHECK(fs::create_directory(dir));
  CHECK(write_obj(dir / "cube.obj", cube({0.1f, 0.0f, 0.0f}, 0.06f)));
  vkc::Result<cfg::Config> settings =
      scenario(dir, "cube.obj", R"(, "metres_per_unit": 1)");
  CHECK(settings.ok());
  vkc::Result<vr_example::MeshSequence> sequence =
      vr_example::MeshSequence::open(*settings);
  CHECK(sequence.ok());
  vr_example::CodecOutputs outputs;
  outputs.stream = (dir / "one.vrst").string();
  vkc::Result<vr_example::CodecStream> coder =
      vr_example::CodecStream::create(gpu.device, gpu.allocator, {}, outputs);
  vkc::Result<ts::MeshIntegrator> integrator =
      ts::MeshIntegrator::create(gpu.device, gpu.allocator);
  vkc::Result<vol::VoxelBlockGrid> grid = small_grid(gpu);
  CHECK(coder.ok() && integrator.ok() && grid.ok());
  CHECK(vr_example::code_frame(*sequence, 0, *grid, *integrator, {}, *coder,
                               nullptr)
            .ok());
  CHECK(coder->finish().ok());
  vkc::Result<std::unique_ptr<stream::StreamReader>> opened =
      stream::StreamReader::open(outputs.stream);
  CHECK(opened.ok());
  stream::Position pos = (*opened)->begin();
  stream::Record record;
  vkc::Result<bool> more = (*opened)->next(&pos, &record);
  CHECK(more.ok() && *more);
  CHECK(record.info.set == 0 && record.info.pts_ns == 0 &&
        record.info.seq == 0 && record.info.key);
  return 0;
}

int gpu_main(vr_test::GpuContext& gpu) {
  vr_test::Scratch scratch("recon_example_mesh_sequence_gpu_");
  CHECK(clean_grids(gpu, scratch) == 0);
  CHECK(records(gpu, scratch) == 0);
  CHECK(single_record(gpu, scratch) == 0);
  CHECK(refused_once_placed(gpu, scratch) == 0);
  CHECK(holes_in_code_frame(gpu, scratch) == 0);
  std::puts("example_mesh_sequence: OK");
  return 0;
}

}  // namespace

int main() {
  vr_test::Scratch scratch("recon_example_mesh_sequence_");
  CHECK(patterns() == 0);
  CHECK(refusals(scratch) == 0);
  CHECK(one_transform(scratch) == 0);
  CHECK(single_mesh(scratch) == 0);
  CHECK(signed_rule(scratch) == 0);
  CHECK(placed_audit_decides(scratch) == 0);
  CHECK(escaped_directory(scratch) == 0);
  return vr_test::run_on_gpu(gpu_main);
}
