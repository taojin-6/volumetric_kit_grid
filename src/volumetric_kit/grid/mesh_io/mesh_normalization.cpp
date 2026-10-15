// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include "volumetric_kit/recon/io/mesh_normalization.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <map>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include <glm/glm.hpp>

#include "volumetric_kit/core/base/exception_status.hpp"

// CMake compiles this file with -ffp-contract=off: no multiply-add is fused,
// so a transform's doubles do not depend on the compiler.

namespace volumetric_kit::recon::io {
namespace {

bool finite(const Vec3d& v) {
  return std::isfinite(v.x) && std::isfinite(v.y) && std::isfinite(v.z);
}
// Within this, the axes are a rotation: camera/geometry.cpp's tolerance.
constexpr double kOrthonormalTolerance = 1e-6;

// Orthonormal, with a positive determinant; false for a non-finite axis.
bool proper(const MeshAxes& axes) {
  const Vec3d* const rows[] = {&axes.x, &axes.up, &axes.z};
  for (int a = 0; a < 3; ++a) {
    for (int b = 0; b < 3; ++b) {
      const double d = glm::dot(*rows[a], *rows[b]);
      if (!(std::abs(d - (a == b ? 1.0 : 0.0)) <= kOrthonormalTolerance)) {
        return false;
      }
    }
  }
  return glm::dot(axes.x, glm::cross(axes.up, axes.z)) > 0.0;
}
// A position in the rotated frame.
Vec3d measure(const MeshAxes& axes, const Vec3d& p) {
  return {glm::dot(axes.x, p), glm::dot(axes.up, p), glm::dot(axes.z, p)};
}

core::Status invalid(const char* api, const char* reason) {
  return core::Status::invalid_argument(std::string(api) + ": " + reason);
}

// What apply_transform, and so a stored transform, needs.
core::Status check_transform(const char* api, const MeshTransform& transform) {
  if (!proper(transform.axes)) {
    return invalid(api, "the axes are not a proper rotation");
  }
  if (!finite(transform.origin) || !std::isfinite(transform.metres_per_unit) ||
      !(transform.metres_per_unit > 0.0)) {
    return invalid(api,
                   "the transform needs a finite origin and a positive scale");
  }
  return {};
}

constexpr char kTransformTag[4] = {'V', 'R', 'M', 'T'};
constexpr std::uint32_t kTransformVersion = 1;

// Little-endian words, whatever the host's order.
void put_word(std::string& out, std::uint64_t word, int bytes) {
  for (int i = 0; i < bytes; ++i) out.push_back(char((word >> (8 * i)) & 0xff));
}
std::uint64_t get_word(const std::string& in, std::size_t at, int bytes) {
  std::uint64_t word = 0;
  for (int i = 0; i < bytes; ++i) {
    word |= std::uint64_t(std::uint8_t(in[at + std::size_t(i)])) << (8 * i);
  }
  return word;
}

// Rotate to the bounds' axes, centre X/Z, floor Y, then scale.
core::Result<MeshTransform> place(const char* api, const MeshBounds& bounds,
                                  double metres_per_unit) {
  if (!proper(bounds.axes)) {
    return invalid(api, "the axes are not a proper rotation");
  }
  for (int a = 0; a < 3; ++a) {
    if (std::isnan(bounds.min[a]) || std::isnan(bounds.max[a])) {
      return invalid(api, "a measured position is not finite");
    }
    if (!(bounds.min[a] <= bounds.max[a])) {
      return invalid(api, "the bounds hold no positions");
    }
  }
  MeshTransform transform;
  transform.axes = bounds.axes;
  transform.origin = {0.5 * (bounds.min.x + bounds.max.x), bounds.min.y,
                      0.5 * (bounds.min.z + bounds.max.z)};
  transform.metres_per_unit = metres_per_unit;
  if (!finite(transform.origin)) {
    return invalid(api, "the bounds are not finite");
  }
  return transform;
}

}  // namespace

core::Result<MeshAxes> up_axes(const Vec3d& up) {
  const double largest =
      std::max({std::abs(up.x), std::abs(up.y), std::abs(up.z)});
  if (!finite(up) || !(largest > 0.0)) {
    return invalid("up_axes", "up must be finite and nonzero");
  }
  // Scaling by a power of two is exact, and it keeps the squared length from
  // overflowing or underflowing, so every finite, nonzero up normalizes.
  const int exponent = std::ilogb(largest);
  const Vec3d scaled{std::ldexp(up.x, -exponent), std::ldexp(up.y, -exponent),
                     std::ldexp(up.z, -exponent)};
  MeshAxes axes;
  axes.up = glm::normalize(scaled);
  const Vec3d reference =
      std::abs(axes.up.z) < 0.9 ? Vec3d{0.0, 0.0, 1.0} : Vec3d{1.0, 0.0, 0.0};
  axes.x = glm::normalize(glm::cross(axes.up, reference));
  axes.z = glm::cross(axes.x, axes.up);
  return axes;
}

MeshBounds mesh_bounds(const TriangleMesh& mesh, const MeshAxes& axes) {
  MeshBounds bounds;
  bounds.axes = axes;
  extend_bounds(bounds, mesh);
  return bounds;
}

void extend_bounds(MeshBounds& bounds, const TriangleMesh& mesh) {
  constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
  for (const auto& position : mesh.positions) {
    const Vec3d r = measure(bounds.axes, Vec3d(position));
    for (int a = 0; a < 3; ++a) {
      // NaN stays: std::min and std::max return their first argument when
      // a comparison with it is false.
      if (!std::isfinite(r[a])) {
        bounds.min[a] = bounds.max[a] = kNaN;
      } else {
        bounds.min[a] = std::min(bounds.min[a], r[a]);
        bounds.max[a] = std::max(bounds.max[a], r[a]);
      }
    }
  }
}

core::Result<MeshTransform> fit_transform(const MeshBounds& bounds,
                                          double height_m) {
  if (!std::isfinite(height_m) || !(height_m > 0.0)) {
    return invalid("fit_transform", "height_m must be finite and positive");
  }
  VKC_ASSIGN(MeshTransform transform, place("fit_transform", bounds, 0.0));
  const double extent = bounds.max.y - bounds.min.y;
  if (!(extent > 0.0)) {
    return invalid("fit_transform", "zero extent along up");
  }
  transform.metres_per_unit = height_m / extent;
  if (!std::isfinite(transform.metres_per_unit) ||
      !(transform.metres_per_unit > 0.0)) {
    return invalid("fit_transform",
                   "height_m over the extent along up is not a finite, "
                   "positive scale");
  }
  return transform;
}

core::Result<MeshTransform> scale_transform(const MeshBounds& bounds,
                                            double metres_per_unit) {
  if (!std::isfinite(metres_per_unit) || !(metres_per_unit > 0.0)) {
    return invalid("scale_transform",
                   "metres_per_unit must be finite and positive");
  }
  return place("scale_transform", bounds, metres_per_unit);
}

core::Status apply_transform(TriangleMesh& mesh,
                             const MeshTransform& transform) try {
  VKC_TRY(check_transform("apply_transform", transform));
  std::vector<Vec3f> positions;
  positions.reserve(mesh.positions.size());
  // The extent along up before and after, for the collapse check.
  double low = std::numeric_limits<double>::infinity();
  double high = -low;
  float placed_low = std::numeric_limits<float>::infinity();
  float placed_high = -placed_low;
  for (const auto& position : mesh.positions) {
    const Vec3d r = measure(transform.axes, Vec3d(position));
    Vec3d placed{0.0};
    for (int a = 0; a < 3; ++a) {
      if (!std::isfinite(r[a])) {
        return invalid("apply_transform", "a position is not finite");
      }
      placed[a] = (r[a] - transform.origin[a]) * transform.metres_per_unit;
      if (!std::isfinite(placed[a]) ||
          std::abs(placed[a]) > std::numeric_limits<float>::max()) {
        return invalid("apply_transform", "output overflows float");
      }
    }
    positions.push_back(Vec3f(placed));
    low = std::min(low, r.y);
    high = std::max(high, r.y);
    placed_low = std::min(placed_low, positions.back().y);
    placed_high = std::max(placed_high, positions.back().y);
  }
  if (high > low && !(placed_high > placed_low)) {
    return invalid("apply_transform",
                   "the height along up collapses at float precision");
  }
  mesh.positions = std::move(positions);
  return {};
} catch (...) {
  return core::exception_status("apply_transform",
                                core::Status::Code::InvalidArgument);
}

core::Result<std::string> encode_transform(const MeshTransform& transform) try {
  VKC_TRY(check_transform("encode_transform", transform));
  std::string out(kTransformTag, sizeof kTransformTag);
  out.reserve(kTransformBytes);
  put_word(out, kTransformVersion, 4);
  const double values[13] = {
      transform.axes.x.x,       transform.axes.x.y,  transform.axes.x.z,
      transform.axes.up.x,      transform.axes.up.y, transform.axes.up.z,
      transform.axes.z.x,       transform.axes.z.y,  transform.axes.z.z,
      transform.origin.x,       transform.origin.y,  transform.origin.z,
      transform.metres_per_unit};
  for (const double value : values) {
    std::uint64_t bits = 0;
    std::memcpy(&bits, &value, sizeof bits);
    put_word(out, bits, 8);
  }
  return out;
} catch (...) {
  return core::exception_status("encode_transform",
                                core::Status::Code::InvalidArgument);
}

core::Result<MeshTransform> decode_transform(const std::string& bytes) try {
  if (bytes.size() != kTransformBytes ||
      bytes.compare(0, sizeof kTransformTag, kTransformTag,
                    sizeof kTransformTag) != 0) {
    return invalid("decode_transform",
                   "expected the 112 bytes of a VRMT transform");
  }
  const auto version = std::uint32_t(get_word(bytes, 4, 4));
  if (version != kTransformVersion) {
    return core::Status::unsupported("decode_transform: version " +
                                     std::to_string(version));
  }
  double values[13];
  for (std::size_t i = 0; i < 13; ++i) {
    const std::uint64_t bits = get_word(bytes, 8 + 8 * i, 8);
    std::memcpy(&values[i], &bits, sizeof bits);
  }
  MeshTransform transform;
  transform.axes.x = {values[0], values[1], values[2]};
  transform.axes.up = {values[3], values[4], values[5]};
  transform.axes.z = {values[6], values[7], values[8]};
  transform.origin = {values[9], values[10], values[11]};
  transform.metres_per_unit = values[12];
  VKC_TRY(check_transform("decode_transform", transform));
  return transform;
} catch (...) {
  return core::exception_status("decode_transform",
                                core::Status::Code::InvalidArgument);
}

core::Result<MeshTopology> audit_mesh_topology(const TriangleMesh& mesh) try {
  if (mesh.indices.size() % 3 != 0) {
    return invalid("audit_mesh_topology", "incomplete triangle indices");
  }
  for (const std::uint32_t index : mesh.indices) {
    if (index >= mesh.positions.size()) {
      return invalid("audit_mesh_topology",
                     "an index names a missing position");
    }
  }
  struct Edge {
    std::size_t count = 0;
    int direction = 0;
  };
  std::map<std::pair<std::uint32_t, std::uint32_t>, Edge> edges;
  std::vector<std::uint32_t> parents(mesh.positions.size());
  std::iota(parents.begin(), parents.end(), 0u);
  auto root = [&parents](std::uint32_t v) {
    while (parents[v] != v) {
      parents[v] = parents[parents[v]];
      v = parents[v];
    }
    return v;
  };
  MeshTopology result;
  for (std::size_t t = 0; t < mesh.indices.size(); t += 3) {
    const auto a = mesh.indices[t];
    const auto b = mesh.indices[t + 1];
    const auto c = mesh.indices[t + 2];
    parents[root(b)] = root(a);
    parents[root(c)] = root(a);
    const Vec3d pa(mesh.positions[a]);
    const Vec3d area = glm::cross(Vec3d(mesh.positions[b]) - pa,
                                  Vec3d(mesh.positions[c]) - pa);
    const double twice_area_squared = glm::dot(area, area);
    if (!(std::isfinite(twice_area_squared) && twice_area_squared > 0.0)) {
      ++result.degenerate_triangles;
    }
    for (std::size_t e = 0; e < 3; ++e) {
      const auto from = mesh.indices[t + e];
      const auto to = mesh.indices[t + (e + 1) % 3];
      auto& edge = edges[std::minmax(from, to)];
      ++edge.count;
      edge.direction += from < to ? 1 : -1;
    }
  }
  result.edges = edges.size();
  for (const auto& item : edges) {
    const auto& edge = item.second;
    if (edge.count == 1) ++result.boundary_edges;
    if (edge.count > 2) ++result.nonmanifold_edges;
    if (edge.count == 2 && edge.direction != 0) ++result.inconsistent_edges;
  }
  std::map<std::uint32_t, double> volumes;
  for (std::size_t t = 0; t < mesh.indices.size(); t += 3) {
    const auto a = mesh.indices[t];
    const auto b = mesh.indices[t + 1];
    const auto c = mesh.indices[t + 2];
    // Translate to one component vertex to avoid cancellation far from origin.
    const auto component = root(a);
    const Vec3d origin(mesh.positions[component]);
    const Vec3d p = Vec3d(mesh.positions[a]) - origin;
    const Vec3d q = Vec3d(mesh.positions[b]) - origin;
    const Vec3d r = Vec3d(mesh.positions[c]) - origin;
    volumes[component] += glm::dot(p, glm::cross(q, r)) / 6.0;
  }
  if (result.boundary_edges > 0) {
    // Close each hole before taking the volume: every group of boundary
    // edges that share vertices gets a fan to the centroid of its vertices,
    // run against the edges' own direction. A closed surface's volume does
    // not depend on the point it is taken about, so neither the root vertex
    // nor the index order changes it. The edges and the centroid sums go in
    // index order, so the caps do not depend on triangle order either.
    std::vector<std::uint32_t> groups(mesh.positions.size());
    std::iota(groups.begin(), groups.end(), 0u);
    auto group = [&groups](std::uint32_t v) {
      while (groups[v] != v) {
        groups[v] = groups[groups[v]];
        v = groups[v];
      }
      return v;
    };
    std::vector<bool> on_rim(mesh.positions.size());
    for (const auto& item : edges) {
      if (item.second.count != 1) continue;
      groups[group(item.first.second)] = group(item.first.first);
      on_rim[item.first.first] = on_rim[item.first.second] = true;
    }
    struct Rim {
      Vec3d sum{0.0};
      std::size_t count = 0;
    };
    std::map<std::uint32_t, Rim> rims;
    for (std::uint32_t v = 0; v < mesh.positions.size(); ++v) {
      if (!on_rim[v]) continue;
      auto& rim = rims[group(v)];
      rim.sum += Vec3d(mesh.positions[v]);
      ++rim.count;
    }
    for (const auto& item : edges) {
      if (item.second.count != 1) continue;
      // Its one triangle runs the edge low to high when the direction is +1;
      // the cap runs it the other way.
      const auto low = item.first.first;
      const auto high = item.first.second;
      const bool forward = item.second.direction > 0;
      const auto& rim = rims[group(low)];
      const auto component = root(low);
      const Vec3d origin(mesh.positions[component]);
      const Vec3d p = Vec3d(mesh.positions[forward ? high : low]) - origin;
      const Vec3d q = Vec3d(mesh.positions[forward ? low : high]) - origin;
      const Vec3d r = rim.sum / double(rim.count) - origin;
      volumes[component] += glm::dot(p, glm::cross(q, r)) / 6.0;
    }
  }
  result.components = volumes.size();
  for (const auto& volume : volumes) {
    result.signed_volume += volume.second;
    if (!(volume.second > 0.0)) ++result.nonpositive_components;
  }
  return result;
} catch (...) {
  return core::exception_status("audit_mesh_topology",
                                core::Status::Code::InvalidArgument);
}

}  // namespace volumetric_kit::recon::io
