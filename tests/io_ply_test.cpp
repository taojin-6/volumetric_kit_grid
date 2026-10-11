// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// Host-only PLY export checks. Read bytes directly to pin the public layout
// and color encoding independently of the writer's backend.

#include <array>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <system_error>
#include <vector>

#include "test_check.hpp"
#include "volumetric_kit/recon/io/ply_writer.hpp"

namespace vr = volumetric_kit::recon;
namespace fs = std::filesystem;

namespace {

struct Scratch {
  fs::path path =
      fs::temp_directory_path() /
      ("recon_io_ply_" +
       std::to_string(
           std::chrono::steady_clock::now().time_since_epoch().count()));
  Scratch() { fs::create_directory(path); }
  ~Scratch() {
    std::error_code ignored;
    fs::remove_all(path, ignored);
  }
};

std::vector<std::uint8_t> read_bytes(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}

std::uint32_t read_u32(const std::uint8_t* p) {
  return static_cast<std::uint32_t>(p[0]) |
         (static_cast<std::uint32_t>(p[1]) << 8) |
         (static_cast<std::uint32_t>(p[2]) << 16) |
         (static_cast<std::uint32_t>(p[3]) << 24);
}

float read_float(const std::uint8_t* p) {
  const std::uint32_t bits = read_u32(p);
  float result = 0.0f;
  static_assert(sizeof(result) == sizeof(bits), "PLY requires float32");
  std::memcpy(&result, &bits, sizeof(result));
  return result;
}

vr::mesh::Mesh triangle() {
  vr::mesh::Mesh mesh;
  mesh.vertices.resize(3);
  mesh.vertices[0].position = {1.25f, -2.0f, 3.5f};
  mesh.vertices[1].position = {4.0f, 5.25f, -6.0f};
  mesh.vertices[2].position = {-7.0f, 8.0f, 9.0f};
  mesh.vertices[0].normal = {1.0f, 0.0f, 0.0f};
  mesh.vertices[1].normal = {0.0f, -1.0f, 0.0f};
  mesh.vertices[2].normal = {0.0f, 0.0f, 1.0f};
  mesh.vertices[0].color = {0.0f, 0.5f, 1.0f, 0.0f};
  mesh.vertices[1].color = {0.0031308f, 0.18f, -0.25f, 1.0f};
  mesh.vertices[2].color = {2.0f, 0.25f, 0.75f, 0.5f};
  // A nontrivial order makes indexing vs. consecutive vertex triples visible.
  mesh.indices = {2, 0, 1};
  return mesh;
}

}  // namespace

int main() {
  Scratch scratch;
  const auto path = (scratch.path / "mesh.ply").string();
  const auto mesh = triangle();
  CHECK(vr::io::write_ply(path, mesh));
  const auto bytes = read_bytes(path);
  const std::string text(bytes.begin(), bytes.end());
  const auto end = text.find("end_header\n");
  CHECK(end != std::string::npos);
  const std::size_t payload = end + std::strlen("end_header\n");
  CHECK(text.substr(0, payload) ==
        "ply\nformat binary_little_endian 1.0\n"
        "comment volumetric_kit_recon\n"
        "element vertex 3\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property float nx\nproperty float ny\nproperty float nz\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "element face 1\nproperty list uchar int vertex_indices\nend_header\n");
  CHECK(bytes.size() == payload + 3 * 27 + 13);
  const std::array<std::array<std::uint8_t, 3>, 3> expected_rgb{
      {{0, 188, 255}, {10, 118, 0}, {255, 137, 225}}};
  for (std::size_t i = 0; i < mesh.vertices.size(); ++i) {
    const auto* record = bytes.data() + payload + i * 27;
    const auto& vertex = mesh.vertices[i];
    CHECK(read_float(record) == vertex.position.x);
    CHECK(read_float(record + 4) == vertex.position.y);
    CHECK(read_float(record + 8) == vertex.position.z);
    CHECK(read_float(record + 12) == vertex.normal.x);
    CHECK(read_float(record + 16) == vertex.normal.y);
    CHECK(read_float(record + 20) == vertex.normal.z);
    for (std::size_t channel = 0; channel < 3; ++channel) {
      CHECK(record[24 + channel] == expected_rgb[i][channel]);
    }
  }
  const auto* face = bytes.data() + payload + 3 * 27;
  CHECK(face[0] == 3);
  CHECK(read_u32(face + 1) == 2 && read_u32(face + 5) == 0 &&
        read_u32(face + 9) == 1);

  // A refused mesh leaves the existing file intact.
  auto invalid = mesh;
  invalid.indices.back() = 3;
  CHECK(!vr::io::write_ply(path, invalid));
  CHECK(read_bytes(path) == bytes);
  std::puts("io PLY tests passed");
  return 0;
}
