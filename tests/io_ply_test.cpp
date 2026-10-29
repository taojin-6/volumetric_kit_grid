// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// Host-only PLY export checks. Read bytes directly to pin the public layout
// and color encoding independently of the writer's backend. The writer's
// sRGB encode is a copy of its own, held here to the voxel colour format's
// curve (volume/voxel_color.hpp, which grid_volume_voxel_color pins to the
// published curve), code for code.

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <limits>
#include <string>
#include <vector>

#include "volumetric_kit/core/testing/scratch_dir.hpp"
#include "volumetric_kit/core/testing/test_check.hpp"
#include "volumetric_kit/grid/mesh_io/ply_writer.hpp"
#include "volumetric_kit/grid/volume/voxel_color.hpp"

namespace vkc = volumetric_kit::core;
namespace vgrid = volumetric_kit::grid;
namespace vol = volumetric_kit::grid::volume;

namespace {

using vkc::test::read_bytes;

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

vgrid::mesh::Mesh triangle() {
  vgrid::mesh::Mesh mesh;
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

// The code a linear channel writes, by the voxel colour format's curve: 0 for
// NaN and below, 255 from 1 up, else the encoded value rounded.
std::uint8_t srgb_code(float linear) {
  if (!(linear > 0.0f)) return 0;
  if (linear >= 1.0f) return 255;
  return static_cast<std::uint8_t>(vol::linear_to_srgb(linear) * 255.0f + 0.5f);
}

// Linear channels where a code changes: each code's rounding threshold and
// the floats beside it, the curve's linear toe, a sweep of [0, 1], and the
// values outside it.
std::vector<float> code_edges() {
  std::vector<float> values;
  for (int c = 0; c < 255; ++c) {
    float t = vol::srgb_to_linear((static_cast<float>(c) + 0.5f) / 255.0f);
    for (int i = 0; i < 3; ++i) t = std::nextafter(t, 0.0f);
    for (int i = 0; i < 7; ++i) {
      values.push_back(t);
      t = std::nextafter(t, 1.0f);
    }
  }
  for (int i = 0; i <= 4096; ++i) {
    values.push_back(static_cast<float>(i) / 4096.0f);
  }
  const float toe = 0.0031308f;
  values.insert(values.end(), {std::nextafter(toe, 0.0f), toe,
                               std::nextafter(toe, 1.0f), -0.0f, -1.0f, 1.0f,
                               2.0f, std::numeric_limits<float>::quiet_NaN(),
                               std::numeric_limits<float>::infinity(),
                               -std::numeric_limits<float>::infinity()});
  while (values.size() % 3 != 0) values.push_back(0.5f);
  return values;
}

// Every channel of code_edges(), written as vertex colours, reads back as
// srgb_code's.
int check_srgb_parity(const std::string& path) {
  const std::vector<float> values = code_edges();
  vgrid::mesh::Mesh mesh;
  mesh.vertices.resize(values.size() / 3);
  for (std::size_t i = 0; i < mesh.vertices.size(); ++i) {
    mesh.vertices[i].color = {values[3 * i], values[3 * i + 1],
                              values[3 * i + 2], 1.0f};
  }
  CHECK(vgrid::mesh_io::write_ply(path, mesh));
  const auto bytes = read_bytes(path);
  const std::string text(bytes.begin(), bytes.end());
  const auto end = text.find("end_header\n");
  CHECK(end != std::string::npos);
  const std::size_t payload = end + std::strlen("end_header\n");
  CHECK(bytes.size() == payload + mesh.vertices.size() * 27);
  int mismatches = 0;
  for (std::size_t i = 0; i < values.size(); ++i) {
    const std::uint8_t got = bytes[payload + (i / 3) * 27 + 24 + i % 3];
    const std::uint8_t want = srgb_code(values[i]);
    if (got != want && ++mismatches <= 10) {
      std::fprintf(stderr, "linear %.9g: PLY code %u, color_space %u\n",
                   static_cast<double>(values[i]), got, want);
    }
  }
  CHECK(mismatches == 0);
  std::printf("PLY sRGB matched color_space on %zu channels\n", values.size());
  return 0;
}

}  // namespace

int main() {
  vkc::test::Scratch scratch("grid_io_ply_");
  const auto path = (scratch.path / "mesh.ply").string();
  const auto mesh = triangle();
  CHECK(vgrid::mesh_io::write_ply(path, mesh));
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
  CHECK(!vgrid::mesh_io::write_ply(path, invalid));
  CHECK(read_bytes(path) == bytes);

  CHECK(check_srgb_parity((scratch.path / "codes.ply").string()) == 0);
  std::puts("io PLY tests passed");
  return 0;
}
