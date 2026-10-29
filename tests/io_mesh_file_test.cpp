// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// mesh_io::check_mesh_file, the one home of a mesh path's checks: a missing
// file is NotFound, a directory or an empty path InvalidArgument, a regular
// file OK; and mesh_io::load_mesh answers each the same way before it reads
// anything.

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>

#include "volumetric_kit/core/testing/scratch_dir.hpp"
#include "volumetric_kit/core/testing/test_check.hpp"
#include "volumetric_kit/grid/mesh_io/mesh_io.hpp"

namespace vkc = volumetric_kit::core;
namespace mesh_io = volumetric_kit::grid::mesh_io;
using Code = vkc::Status::Code;

namespace {

bool refused(const vkc::Status& status, Code code, const std::string& named) {
  return status.domain() == code &&
         status.message().find(named) != std::string::npos;
}

}  // namespace

int main() {
  vkc::test::Scratch scratch("grid_io_mesh_file_");
  const std::string tetra = (scratch.path / "tetra.obj").string();
  {
    std::ofstream(tetra) << "v 0 0 0\nv 1 0 0\nv 0 0 1\nv 0 1 0\n"
                            "f 1 2 3\nf 1 4 2\nf 1 3 4\nf 2 4 3\n";
  }
  const std::string missing = (scratch.path / "missing.obj").string();
  const std::string directory = (scratch.path / "frames.obj").string();
  CHECK(std::filesystem::create_directory(directory));

  CHECK(mesh_io::check_mesh_file(tetra).ok());
  CHECK(mesh_io::load_mesh(tetra).ok());
  const struct {
    const std::string& path;
    Code code;
    const char* named;
  } refusals[] = {{missing, Code::NotFound, "missing.obj"},
                  {directory, Code::InvalidArgument, "frames.obj"}};
  for (const auto& r : refusals) {
    CHECK(refused(mesh_io::check_mesh_file(r.path), r.code, r.named));
    CHECK(refused(mesh_io::load_mesh(r.path).status(), r.code, r.named));
  }
  CHECK(mesh_io::check_mesh_file("").domain() == Code::InvalidArgument);
  CHECK(mesh_io::load_mesh("").status().domain() == Code::InvalidArgument);
  std::puts("io_mesh_file: OK");
  return 0;
}
