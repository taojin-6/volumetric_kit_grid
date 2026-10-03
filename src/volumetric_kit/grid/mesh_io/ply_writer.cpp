// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include "volumetric_kit/recon/io/ply_writer.hpp"

#include <cmath>
#include <cstdint>
#include <fstream>
#include <limits>
#include <ostream>
#include <vector>

#include "errors.hpp"
#include "tinyply_backend.hpp"
#include "volumetric_kit/recon/core/color_space.hpp"

namespace volumetric_kit::recon::io {
namespace {

namespace tinyply = detail::tinyply;

// Encode a LINEAR vertex-color channel to an 8-bit canonical-encoded code.
//
// PLY is a presentation site. `mesh::Vertex::color` is linear working values
// (the 2026-08-02 color-space decision -- what glTF calls COLOR_0), but a PLY's
// `uchar` red/green/blue properties carry no color-space tag and every external
// viewer (MeshLab, Blender, CloudCompare) reads them as sRGB. Writing the
// linear value straight out would therefore render visibly dark, so the encode
// happens here. glTF export does the opposite and passes COLOR_0 through
// unchanged, which is why the two exporters cannot share one path.
//
// Out-of-range values clamp, and NaN writes 0, so one bad color never costs
// the export.
std::uint8_t to_u8(float linear_channel) {
  if (!(linear_channel > 0.0f)) return 0;
  if (linear_channel >= 1.0f) return 255;
  return static_cast<std::uint8_t>(linear_to_srgb(linear_channel) * 255.0f +
                                   0.5f);
}

bool finite(Vec3f v) {
  return std::isfinite(v.x) && std::isfinite(v.y) && std::isfinite(v.z);
}

}  // namespace

Status write_ply(const std::string& path, const mesh::Mesh& mesh) try {
  VR_TRY(detail::check_path("write_ply", path));
  if (mesh.indices.size() % 3 != 0) {
    return Status::invalid_argument("write_ply: incomplete triangle indices");
  }
  const std::size_t vertex_count = mesh.vertices.size();
  const std::size_t face_count = mesh.indices.size() / 3;
  if (vertex_count > std::vector<float>().max_size() / 3 ||
      mesh.indices.size() > std::vector<std::int32_t>().max_size()) {
    return Status::invalid_argument(
        "write_ply: mesh exceeds host array limits");
  }
  for (std::size_t i = 0; i < vertex_count; ++i) {
    const mesh::Vertex& v = mesh.vertices[i];
    if (!finite(v.position) || !finite(v.normal)) {
      return Status::invalid_argument(
          "write_ply: nonfinite position or normal at vertex " +
          std::to_string(i));
    }
  }
  for (std::uint32_t index : mesh.indices) {
    if (index >= vertex_count ||
        index > static_cast<std::uint32_t>(
                    std::numeric_limits<std::int32_t>::max())) {
      return Status::invalid_argument("write_ply: invalid signed int32 index");
    }
  }
  // tinyply labels newly written files little-endian but writes native values.
  const std::uint16_t endian = 1;
  if (*reinterpret_cast<const std::uint8_t*>(&endian) != 1) {
    return Status::unsupported("write_ply: requires a little-endian host");
  }

  // tinyply writes each property group from a tightly-packed array, so
  // de-interleave the Mesh's array-of-Vertex-struct into per-attribute buffers
  // (position, normal, and the u8-quantized colour). One-shot final export, so
  // the extra copies are inconsequential.
  std::vector<float> positions(vertex_count * 3);
  std::vector<float> normals(vertex_count * 3);
  std::vector<std::uint8_t> colors(vertex_count * 3);
  for (std::size_t i = 0; i < vertex_count; ++i) {
    const mesh::Vertex& v = mesh.vertices[i];
    positions[i * 3 + 0] = v.position.x;
    positions[i * 3 + 1] = v.position.y;
    positions[i * 3 + 2] = v.position.z;
    normals[i * 3 + 0] = v.normal.x;
    normals[i * 3 + 1] = v.normal.y;
    normals[i * 3 + 2] = v.normal.z;
    colors[i * 3 + 0] = to_u8(v.color.x);
    colors[i * 3 + 1] = to_u8(v.color.y);
    colors[i * 3 + 2] = to_u8(v.color.z);
  }
  // PLY's conventional face-index type is signed int32. Mesh indices are
  // uint32, so the explicit range check above is required before conversion.
  std::vector<std::int32_t> faces(mesh.indices.begin(), mesh.indices.end());

  tinyply::PlyFile ply;
  ply.get_comments().push_back("volumetric_kit_recon");
  ply.add_properties_to_element(
      "vertex", {"x", "y", "z"}, tinyply::Type::FLOAT32, vertex_count,
      reinterpret_cast<const std::uint8_t*>(positions.data()),
      tinyply::Type::INVALID, 0);
  ply.add_properties_to_element(
      "vertex", {"nx", "ny", "nz"}, tinyply::Type::FLOAT32, vertex_count,
      reinterpret_cast<const std::uint8_t*>(normals.data()),
      tinyply::Type::INVALID, 0);
  ply.add_properties_to_element("vertex", {"red", "green", "blue"},
                                tinyply::Type::UINT8, vertex_count,
                                colors.data(), tinyply::Type::INVALID, 0);
  ply.add_properties_to_element(
      "face", {"vertex_indices"}, tinyply::Type::INT32, face_count,
      reinterpret_cast<const std::uint8_t*>(faces.data()), tinyply::Type::UINT8,
      3);

  std::ofstream out(path, std::ios::binary);
  if (!out) {
    return Status::io_error("write_ply: cannot open " + path);
  }
  // tinyply's writer is noexcept: only its small per-property tables allocate
  // there, so the large buffers above carry the allocation failure contract.
  ply.write(out, /*isBinary=*/true);
  // Explicit close includes the final flush and reports a delayed write error.
  out.close();
  if (!out) {
    return Status::io_error("write_ply: write failed for " + path);
  }
  return {};
} catch (...) {
  return detail::exception_status("write_ply");
}

}  // namespace volumetric_kit::recon::io
