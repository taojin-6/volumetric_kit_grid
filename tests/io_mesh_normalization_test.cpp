// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// Host-only checks of the opt-in mesh normalization and topology helpers, on
// synthetic meshes: union bounds, the two fits, closed-form placement of a
// tilted tetrahedron, one transform over a moving sequence, every refusal
// leaving the geometry bit for bit, and every topology audit case.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include <glm/glm.hpp>

#include "test_allocation_failure.hpp"
#include "test_check.hpp"
#include "volumetric_kit/recon/io/mesh_normalization.hpp"

namespace vkc = volumetric_kit::core;
namespace io = volumetric_kit::recon::io;

namespace {

using volumetric_kit::recon::Vec3d;
using Code = vkc::Status::Code;
constexpr double kInf = std::numeric_limits<double>::infinity();
constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

// Wound outward: volume 8/6, 4 vertices, 6 edges.
io::TriangleMesh tetrahedron() {
  return {{{10, 20, 30}, {12, 20, 30}, {10, 22, 30}, {10, 20, 32}},
          {0, 2, 1, 0, 1, 3, 0, 3, 2, 1, 2, 3}};
}

io::TriangleMesh shifted(io::TriangleMesh mesh, const Vec3d& d) {
  for (auto& p : mesh.positions) {
    p.x = float(p.x + d[0]);
    p.y = float(p.y + d[1]);
    p.z = float(p.z + d[2]);
  }
  return mesh;
}

bool same_bits(const io::TriangleMesh& a, const io::TriangleMesh& b) {
  return a.positions.size() == b.positions.size() && a.indices == b.indices &&
         std::memcmp(a.positions.data(), b.positions.data(),
                     a.positions.size() * sizeof(a.positions[0])) == 0;
}

bool near(const Vec3d& a, const Vec3d& b, double tolerance) {
  for (int i = 0; i < 3; ++i) {
    if (!(std::abs(a[i] - b[i]) <= tolerance)) return false;
  }
  return true;
}

bool near(const volumetric_kit::recon::Vec3f& a, const Vec3d& b,
          double tolerance) {
  return near(Vec3d(a), b, tolerance);
}

bool same(const io::MeshAxes& a, const io::MeshAxes& b) {
  return a.x == b.x && a.up == b.up && a.z == b.z;
}

bool same(const io::MeshTransform& a, const io::MeshTransform& b) {
  return same(a.axes, b.axes) && a.origin == b.origin &&
         a.metres_per_unit == b.metres_per_unit;
}

io::MeshAxes oblique() { return io::up_axes({-2, 1, -3}).value(); }

// A square pyramid without its base, wound outward: apex 0 over the square
// 1-4, volume 4/3 once the base is capped. Each face is written from its base
// vertex, or from the apex, which every face shares.
io::TriangleMesh open_pyramid(bool from_apex) {
  io::TriangleMesh mesh{
      {{0, 1, 0}, {-1, 0, -1}, {1, 0, -1}, {1, 0, 1}, {-1, 0, 1}}, {}};
  mesh.indices =
      from_apex
          ? std::vector<std::uint32_t>{0, 2, 1, 0, 3, 2, 0, 4, 3, 0, 1, 4}
          : std::vector<std::uint32_t>{2, 1, 0, 3, 2, 0, 4, 3, 0, 1, 4, 0};
  return mesh;
}

// Refused as InvalidArgument, for the reason given.
template <typename T>
bool refused_for(const vkc::Result<T>& result, const std::string& reason) {
  return !result && result.status().domain() == Code::InvalidArgument &&
         result.status().message().find(reason) != std::string::npos;
}

// The union over frames is the bounds of their concatenation, exactly, in any
// order; an empty frame changes nothing.
int union_bounds() {
  const auto axes = oblique();
  const auto a = tetrahedron();
  const auto b = shifted(a, {5, -7, 1});
  const io::TriangleMesh c{{{-40, 3, 9}, {-38, 4, 9}, {-40, 3, 12}}, {0, 1, 2}};
  auto sequence = io::mesh_bounds(io::TriangleMesh{}, axes);
  CHECK(refused_for(io::fit_transform(sequence, 1.7), "hold no positions"));
  CHECK(refused_for(io::scale_transform(sequence, 1.0), "hold no positions"));
  for (const auto* frame : {&a, &b, &c}) io::extend_bounds(sequence, *frame);
  io::TriangleMesh all;
  for (const auto* frame : {&a, &b, &c}) {
    all.positions.insert(all.positions.end(), frame->positions.begin(),
                         frame->positions.end());
  }
  const auto whole = io::mesh_bounds(all, axes);
  CHECK(same(sequence.axes, axes));
  CHECK(sequence.min == whole.min && sequence.max == whole.max);
  auto reversed = io::mesh_bounds(io::TriangleMesh{}, axes);
  for (const auto* frame : {&c, &b, &a}) io::extend_bounds(reversed, *frame);
  CHECK(reversed.min == whole.min && reversed.max == whole.max);
  for (int axis = 0; axis < 3; ++axis) {
    double low = kInf;
    double high = -kInf;
    for (const auto* frame : {&a, &b, &c}) {
      const auto own = io::mesh_bounds(*frame, axes);
      low = std::min(low, own.min[axis]);
      high = std::max(high, own.max[axis]);
    }
    CHECK(sequence.min[axis] == low && sequence.max[axis] == high);
  }
  io::extend_bounds(sequence, io::TriangleMesh{});
  CHECK(sequence.min == whole.min && sequence.max == whole.max);
  // The default axes measure input coordinates as given.
  const auto given = io::mesh_bounds(a);
  CHECK(given.min == (Vec3d{10, 20, 30}) && given.max == (Vec3d{12, 22, 32}));
  const auto none = io::mesh_bounds(io::TriangleMesh{}, axes);
  CHECK(none.min == Vec3d(kInf) && none.max == Vec3d(-kInf));
  return 0;
}

// A fit is a unit transform at the scale it chose; units given place the mesh
// the same way, and a change of input unit does not change the fit's output.
int fit_against_unit() {
  const auto axes = oblique();
  const auto mesh = tetrahedron();
  const auto bounds = io::mesh_bounds(mesh, axes);
  const auto fit = io::fit_transform(bounds, 1.7);
  CHECK(fit);
  CHECK(fit.value().metres_per_unit == 1.7 / (bounds.max[1] - bounds.min[1]));
  const auto unit = io::scale_transform(bounds, fit.value().metres_per_unit);
  CHECK(unit && same(unit.value(), fit.value()));
  auto by_fit = mesh;
  auto by_unit = mesh;
  CHECK(io::apply_transform(by_fit, fit.value()));
  CHECK(io::apply_transform(by_unit, unit.value()));
  CHECK(same_bits(by_fit, by_unit));
  const auto millimetres = io::scale_transform(bounds, 0.001);
  CHECK(millimetres && same(millimetres.value().axes, axes) &&
        millimetres.value().origin == fit.value().origin &&
        millimetres.value().metres_per_unit == 0.001);

  auto scaled = mesh;
  for (auto& p : scaled.positions) p *= 1000.0f;
  const auto scaled_fit = io::fit_transform(io::mesh_bounds(scaled, axes), 1.7);
  CHECK(scaled_fit);
  CHECK(std::abs(scaled_fit.value().metres_per_unit * 1000.0 /
                     fit.value().metres_per_unit -
                 1.0) < 1e-12);
  CHECK(io::apply_transform(scaled, scaled_fit.value()));
  for (std::size_t i = 0; i < mesh.positions.size(); ++i) {
    const auto& p = by_fit.positions[i];
    CHECK(near(scaled.positions[i], {p.x, p.y, p.z}, 1e-6));
  }
  return 0;
}

// An oblique, unnormalized up against closed forms: up/√14, x = (1, 2, 0)/√5,
// z = (−6, 3, 5)/√70. The tetrahedron stands upright along those axes at
// c = (10, 20, 30), so it lands on known metres.
int tilted_tetrahedron() {
  const auto axes = io::up_axes({-2, 1, -3});
  CHECK(axes);
  const double r14 = std::sqrt(14.0);
  const double r5 = std::sqrt(5.0);
  const double r70 = std::sqrt(70.0);
  const Vec3d up{-2 / r14, 1 / r14, -3 / r14};
  const Vec3d x{1 / r5, 2 / r5, 0};
  const Vec3d z{-6 / r70, 3 / r70, 5 / r70};
  CHECK(near(axes.value().up, up, 1e-15));
  CHECK(near(axes.value().x, x, 1e-15));
  CHECK(near(axes.value().z, z, 1e-15));

  // Upright, with a distinct extent on each axis: a base at height 0, X in
  // [−0.5, 0.5], Z in [−0.75, 0.75], the apex at height 2. Volume 1/2.
  const Vec3d upright[] = {
      {-0.5, 0, -0.75}, {0.5, 0, -0.75}, {0, 0, 0.75}, {0, 2, 0}};
  const Vec3d c{10, 20, 30};
  io::TriangleMesh mesh;
  for (const auto& q : upright) {
    mesh.positions.push_back(
        volumetric_kit::recon::Vec3f(c + q.x * x + q.y * up + q.z * z));
  }
  mesh.indices = {0, 1, 2, 0, 3, 1, 1, 3, 2, 2, 3, 0};
  const auto before = io::audit_mesh_topology(mesh);
  CHECK(before && before.value().supports_signed());
  CHECK(std::abs(before.value().signed_volume - 0.5) < 1e-5);

  // c measured along the axes: (50/√5, −90/√14, 150/√70).
  const Vec3d rc{50 / r5, -90 / r14, 150 / r70};
  const auto bounds = io::mesh_bounds(mesh, axes.value());
  CHECK(near(bounds.min, {rc[0] - 0.5, rc[1], rc[2] - 0.75}, 1e-5));
  CHECK(near(bounds.max, {rc[0] + 0.5, rc[1] + 2, rc[2] + 0.75}, 1e-5));
  const auto fit = io::fit_transform(bounds, 1.7);
  CHECK(fit);
  CHECK(same(fit.value().axes, axes.value()));
  CHECK(near(fit.value().origin, rc, 1e-5));
  CHECK(std::abs(fit.value().metres_per_unit - 0.85) < 1e-5);
  CHECK(io::apply_transform(mesh, fit.value()));
  for (std::size_t i = 0; i < 4; ++i) {
    const auto& q = upright[i];
    CHECK(
        near(mesh.positions[i], {0.85 * q[0], 0.85 * q[1], 0.85 * q[2]}, 1e-5));
  }
  // The lowest Y is exactly the floor; the height is 1.7 m to rounding.
  const auto placed = io::mesh_bounds(mesh);
  CHECK(placed.min[1] == 0.0);
  CHECK(std::abs(placed.max[1] - 1.7) < 1e-6);
  CHECK(std::abs(placed.min[0] + placed.max[0]) < 1e-6);
  CHECK(std::abs(placed.min[2] + placed.max[2]) < 1e-6);
  // A proper rotation and a positive scale keep the winding and the volume's
  // sign.
  const auto after = io::audit_mesh_topology(mesh);
  CHECK(after && after.value().supports_signed());
  CHECK(std::abs(after.value().signed_volume - 0.5 * 0.85 * 0.85 * 0.85) <
        1e-5);

  // Z-up data takes +X as the reference: x = (0, −1, 0), z = (1, 0, 0).
  const auto down = io::up_axes({0, 0, -1});
  CHECK(down && down.value().up == (Vec3d{0, 0, -1}) &&
        down.value().x == (Vec3d{0, -1, 0}) &&
        down.value().z == (Vec3d{1, 0, 0}));
  // The reference switches at |up.z| = 0.9: +Z gives x.z = 0, +X x.x = 0.
  const auto below = io::up_axes({0, 0.45, 0.88});  // up.z = 0.8903
  const auto above = io::up_axes({0, 0.4, 0.92});   // up.z = 0.9170
  CHECK(below && below.value().x[2] == 0.0 && below.value().x[0] != 0.0);
  CHECK(above && above.value().x[0] == 0.0 && above.value().x[2] != 0.0);
  // At |up.z| = 0.9 exactly the reference is already +X: x = +Y, not −Y.
  bool at_boundary = false;
  double a = std::sqrt(0.19);
  for (int step = 0; step < 64 && !at_boundary; ++step) {
    const auto edge = io::up_axes({a, 0, 0.9});
    at_boundary = edge && edge.value().up.z == 0.9;
    if (at_boundary) CHECK(edge.value().x == Vec3d(0, 1, 0));
    a = std::nextafter(a, 1.0);
  }
  CHECK(at_boundary);
  // Any finite, nonzero length: the squared length of these over- or
  // underflows, yet each is +Y.
  for (const double length : {1e160, 1e-170, 4.9e-324}) {
    const auto far = io::up_axes({0, length, 0});
    CHECK(far && far.value().up == Vec3d(0, 1, 0));
  }
  return 0;
}

// One transform over a sequence: a frame moved by d lands at the first
// frame's result plus scale·R·d, whether the fit saw the first frame alone or
// the union.
int shifted_sequence() {
  const auto axes = oblique();
  const Vec3d d{0.5, -0.25, 1.0};  // Exact in float at these magnitudes.
  const auto first = tetrahedron();
  const auto moved = shifted(first, d);
  auto sequence = io::mesh_bounds(io::TriangleMesh{}, axes);
  io::extend_bounds(sequence, first);
  io::extend_bounds(sequence, moved);
  const auto own = io::fit_transform(io::mesh_bounds(first, axes), 1.7);
  const auto shared = io::fit_transform(sequence, 1.7);
  CHECK(own && shared);
  CHECK(same(own.value().axes, shared.value().axes));
  for (const auto& transform : {own.value(), shared.value()}) {
    auto a = first;
    auto b = moved;
    CHECK(io::apply_transform(a, transform));
    CHECK(io::apply_transform(b, transform));
    const double s = transform.metres_per_unit;
    const Vec3d offset{s * glm::dot(axes.x, d), s * glm::dot(axes.up, d),
                       s * glm::dot(axes.z, d)};
    for (std::size_t i = 0; i < a.positions.size(); ++i) {
      const auto& p = a.positions[i];
      CHECK(near(b.positions[i],
                 {p.x + offset[0], p.y + offset[1], p.z + offset[2]}, 1e-6));
    }
    // The same frame again maps to the same bits.
    auto again = first;
    CHECK(io::apply_transform(again, transform));
    CHECK(same_bits(again, a));
  }
  return 0;
}

// Every refusal; the geometry is unchanged, bit for bit.
int refusals() {
  for (const Vec3d& up : {Vec3d{0, 0, 0}, Vec3d{0, kInf, 0}, Vec3d{kNaN, 1, 0},
                          Vec3d{0, 1, -kInf}}) {
    const auto axes = io::up_axes(up);
    CHECK(!axes && axes.status().domain() == Code::InvalidArgument);
  }
  const auto axes = oblique();
  auto mesh = tetrahedron();
  const auto before = mesh;
  const auto bounds = io::mesh_bounds(mesh, axes);
  for (const double bad : {0.0, -1.0, kNaN, kInf, -kInf}) {
    CHECK(refused_for(io::fit_transform(bounds, bad), "height_m must be"));
    CHECK(refused_for(io::scale_transform(bounds, bad),
                      "metres_per_unit must be"));
  }
  const auto fit = io::fit_transform(bounds, 1.7).value();
  auto refused = [&](io::TriangleMesh& geometry,
                     const io::MeshTransform& transform) {
    const auto kept = geometry;
    const auto status = io::apply_transform(geometry, transform);
    return !status && status.domain() == Code::InvalidArgument &&
           same_bits(geometry, kept);
  };
  for (const double bad : {0.0, -1.0, kNaN, kInf}) {
    auto transform = fit;
    transform.metres_per_unit = bad;
    CHECK(refused(mesh, transform));
  }
  auto bad_origin = fit;
  bad_origin.origin[1] = kNaN;
  CHECK(refused(mesh, bad_origin));
  auto bad_axis = fit;
  bad_axis.axes.z[0] = kInf;
  CHECK(refused(mesh, bad_axis));

  // Axes that are not a proper rotation: a reflection, a repeated axis, a
  // scaled one. Both fits and apply refuse them; within 1e-6 is a rotation.
  io::MeshAxes reflection;
  reflection.z = {0, 0, -1};
  io::MeshAxes repeated;
  repeated.x = repeated.z = repeated.up;
  io::MeshAxes stretched;
  stretched.x = {2, 0, 0};
  for (const auto& wrong : {reflection, repeated, stretched}) {
    const auto measured = io::mesh_bounds(mesh, wrong);
    CHECK(refused_for(io::fit_transform(measured, 1.7), "proper rotation"));
    CHECK(refused_for(io::scale_transform(measured, 1.0), "proper rotation"));
    auto transform = fit;
    transform.axes = wrong;
    CHECK(refused(mesh, transform));
  }
  auto nearly = axes;
  nearly.x[0] += 1e-9;
  CHECK(io::fit_transform(io::mesh_bounds(mesh, nearly), 1.7));

  // Non-finite input: the bounds turn NaN, so the fits refuse, and so does
  // apply; one such frame spoils a sequence's union for good.
  for (const float bad : {std::numeric_limits<float>::quiet_NaN(),
                          std::numeric_limits<float>::infinity()}) {
    auto poisoned = tetrahedron();
    poisoned.positions[2].y = bad;
    const auto poisoned_bounds = io::mesh_bounds(poisoned, axes);
    for (int a = 0; a < 3; ++a) {
      CHECK(std::isnan(poisoned_bounds.min[a]) &&
            std::isnan(poisoned_bounds.max[a]));
    }
    CHECK(refused_for(io::fit_transform(poisoned_bounds, 1.7), "not finite"));
    CHECK(!io::scale_transform(poisoned_bounds, 1.0));
    CHECK(refused(poisoned, fit));
    auto sequence = io::mesh_bounds(io::TriangleMesh{}, axes);
    io::extend_bounds(sequence, poisoned);
    io::extend_bounds(sequence, mesh);
    CHECK(!io::fit_transform(sequence, 1.7));
  }

  // No extent along up: a fit has nothing to scale; units given still place
  // the mesh, as nothing collapses.
  auto flat = tetrahedron();
  for (auto& p : flat.positions) p.y = 0.0f;
  const auto level = io::mesh_bounds(flat, io::MeshAxes{});
  CHECK(!io::fit_transform(level, 1.7));
  const auto level_unit = io::scale_transform(level, 2.0);
  CHECK(level_unit && io::apply_transform(flat, level_unit.value()));
  CHECK(flat.positions[1].x == 2.0f && flat.positions[1].y == 0.0f);

  // A scale past double range, either way.
  io::MeshBounds thin;
  thin.min = {0, 0, 0};
  thin.max = {1, 1e-300, 1};
  CHECK(!io::fit_transform(thin, 1e10));
  io::MeshBounds tall;
  tall.min = {0, 0, 0};
  tall.max = {1, 1e300, 1};
  CHECK(!io::fit_transform(tall, 1e-30));
  // Bounds whose centre is not finite.
  io::MeshBounds wide;
  wide.min = {0, 0, 0};
  wide.max = {kInf, 1, 1};
  CHECK(!io::fit_transform(wide, 1.7) && !io::scale_transform(wide, 1.0));

  // Float overflow.
  CHECK(refused(mesh, io::scale_transform(bounds, 1e300).value()));
  CHECK(refused(mesh, io::fit_transform(bounds, 1e300).value()));
  // Nothing to place is not a refusal: an empty frame stays empty.
  io::TriangleMesh nothing;
  CHECK(io::apply_transform(nothing, fit) && nothing.positions.empty());
  // An output of exactly the largest float is in range.
  io::TriangleMesh corner{{{1, 0, 0}, {0, 1, 0}}, {}};
  io::MeshTransform largest;
  largest.metres_per_unit = double(std::numeric_limits<float>::max());
  CHECK(io::apply_transform(corner, largest));
  CHECK(corner.positions[0].x == std::numeric_limits<float>::max());
  // Height collapse: a scale so small that every Y rounds to zero.
  CHECK(refused(mesh, io::fit_transform(bounds, 1e-100).value()));
  CHECK(refused(mesh, io::scale_transform(bounds, 1e-60).value()));
  CHECK(same_bits(mesh, before));
  return 0;
}

// Each defect alone, and holes admitted only on request.
int topology() {
  const auto audit = [](const io::TriangleMesh& mesh) {
    return io::audit_mesh_topology(mesh).value();
  };
  const auto closed = audit(tetrahedron());
  CHECK(closed.edges == 6 && closed.boundary_edges == 0 &&
        closed.nonmanifold_edges == 0 && closed.inconsistent_edges == 0 &&
        closed.degenerate_triangles == 0 && closed.components == 1 &&
        closed.nonpositive_components == 0);
  CHECK(std::abs(closed.signed_volume - 8.0 / 6.0) < 1e-12);
  CHECK(closed.supports_signed() && closed.supports_signed(true));

  // A hole: one face gone, the rest still enclosing positive volume.
  auto open_mesh = tetrahedron();
  open_mesh.indices.erase(open_mesh.indices.begin(),
                          open_mesh.indices.begin() + 3);
  const auto open = audit(open_mesh);
  CHECK(open.boundary_edges == 3 && open.nonmanifold_edges == 0 &&
        open.inconsistent_edges == 0 && open.degenerate_triangles == 0 &&
        open.components == 1 && open.nonpositive_components == 0);
  CHECK(!open.supports_signed() && open.supports_signed(true));
  CHECK(std::abs(open.signed_volume - 8.0 / 6.0) < 1e-12);

  // The hole is capped, so the volume does not depend on index order: the
  // same faces, each written from vertex 3, give the same volume and verdict.
  auto turned_mesh = open_mesh;
  turned_mesh.indices = {3, 0, 1, 3, 2, 0, 3, 1, 2};
  const auto turned = audit(turned_mesh);
  CHECK(turned.boundary_edges == 3 && turned.nonpositive_components == 0);
  CHECK(std::abs(turned.signed_volume - 8.0 / 6.0) < 1e-12);
  CHECK(!turned.supports_signed() && turned.supports_signed(true));

  // Nor on position or rotation: an open pyramid, either order, as given,
  // shifted, then rotated.
  const auto rotation =
      io::scale_transform(io::mesh_bounds(open_pyramid(true), oblique()), 1.0)
          .value();
  for (const bool from_apex : {false, true}) {
    auto pyramid = open_pyramid(from_apex);
    for (int copy = 0; copy < 3; ++copy) {
      if (copy == 1) pyramid = shifted(pyramid, {100, -50, 30});
      if (copy == 2) CHECK(io::apply_transform(pyramid, rotation));
      const auto capped = audit(pyramid);
      CHECK(capped.boundary_edges == 4 && capped.components == 1 &&
            capped.inconsistent_edges == 0 && capped.nonmanifold_edges == 0 &&
            capped.nonpositive_components == 0);
      CHECK(std::abs(capped.signed_volume - 4.0 / 3.0) < 1e-4);
      CHECK(!capped.supports_signed() && capped.supports_signed(true));
    }
    // Inside out, holes allowed or not, it is refused.
    for (std::size_t t = 0; t < pyramid.indices.size(); t += 3) {
      std::swap(pyramid.indices[t], pyramid.indices[t + 1]);
    }
    const auto inverted = audit(pyramid);
    CHECK(std::abs(inverted.signed_volume + 4.0 / 3.0) < 1e-4);
    CHECK(!inverted.supports_signed(true));
  }

  // A skew hole is capped at the centroid of its vertices: the unit
  // octahedron (vertices ±X 0-1, ±Y 2-3, ±Z 4-5) without its two faces on edge
  // +X+Y leaves the loop +Z, +Y, −Z, +X, capped at (0.25, 0.25, 0) to 7/6. That
  // is the volume of the same mesh with its cap written out.
  io::TriangleMesh skew_mesh{
      {{1, 0, 0}, {-1, 0, 0}, {0, 1, 0}, {0, -1, 0}, {0, 0, 1}, {0, 0, -1}},
      {}};
  for (const std::uint32_t x : {0u, 1u}) {
    for (const std::uint32_t y : {2u, 3u}) {
      for (const std::uint32_t z : {4u, 5u}) {
        if (x == 0 && y == 2) continue;
        // Wound outward: an odd count of negative axes swaps two vertices.
        const bool odd = ((x == 1) != (y == 3)) != (z == 5);
        skew_mesh.indices.insert(skew_mesh.indices.end(),
                                 {x, odd ? z : y, odd ? y : z});
      }
    }
  }
  const auto skew = audit(skew_mesh);
  CHECK(skew.boundary_edges == 4 && skew.nonpositive_components == 0);
  CHECK(std::abs(skew.signed_volume - 7.0 / 6.0) < 1e-12);
  auto written = skew_mesh;
  written.positions.push_back({0.25f, 0.25f, 0.0f});
  written.indices.insert(written.indices.end(),
                         {2, 4, 6, 4, 0, 6, 0, 5, 6, 5, 2, 6});
  const auto closed_cap = audit(written);
  CHECK(closed_cap.boundary_edges == 0 && closed_cap.inconsistent_edges == 0);
  CHECK(std::abs(closed_cap.signed_volume - skew.signed_volume) < 1e-12);

  // One face flipped.
  auto flipped_mesh = tetrahedron();
  std::swap(flipped_mesh.indices[0], flipped_mesh.indices[1]);
  const auto flipped = audit(flipped_mesh);
  CHECK(flipped.inconsistent_edges == 3 && flipped.boundary_edges == 0 &&
        flipped.nonmanifold_edges == 0 && flipped.nonpositive_components == 0);
  CHECK(!flipped.supports_signed() && !flipped.supports_signed(true));

  // Every face flipped: consistent, but inside out.
  auto inward_mesh = tetrahedron();
  for (std::size_t t = 0; t < inward_mesh.indices.size(); t += 3) {
    std::swap(inward_mesh.indices[t], inward_mesh.indices[t + 1]);
  }
  const auto inward = audit(inward_mesh);
  CHECK(inward.nonpositive_components == 1 && inward.inconsistent_edges == 0 &&
        inward.boundary_edges == 0 && inward.signed_volume < 0.0);
  CHECK(!inward.supports_signed() && !inward.supports_signed(true));

  // A face twice: three triangles on each of its edges.
  auto doubled_mesh = tetrahedron();
  doubled_mesh.indices.insert(doubled_mesh.indices.end(), {0, 2, 1});
  const auto doubled = audit(doubled_mesh);
  CHECK(doubled.nonmanifold_edges == 3 && doubled.boundary_edges == 0 &&
        doubled.degenerate_triangles == 0 &&
        doubled.nonpositive_components == 0);
  CHECK(!doubled.supports_signed() && !doubled.supports_signed(true));

  // A sliver: vertex 4 on edge 1-2 splits face {1, 2, 3}, and the zero-area
  // triangle {1, 2, 4} closes the seam, so the mesh stays closed.
  io::TriangleMesh sliver_mesh = tetrahedron();
  sliver_mesh.positions.push_back({11, 21, 30});
  sliver_mesh.indices = {0, 2, 1, 0, 1, 3, 0, 3, 2, 1, 4, 3, 4, 2, 3, 1, 2, 4};
  const auto sliver = audit(sliver_mesh);
  CHECK(sliver.degenerate_triangles == 1 && sliver.boundary_edges == 0 &&
        sliver.nonmanifold_edges == 0 && sliver.inconsistent_edges == 0 &&
        sliver.nonpositive_components == 0);
  CHECK(!sliver.supports_signed() && !sliver.supports_signed(true));

  // No triangles: no component.
  const auto empty = audit(io::TriangleMesh{});
  CHECK(empty.components == 0 && empty.edges == 0);
  CHECK(!empty.supports_signed() && !empty.supports_signed(true));

  // A closed, double-sided sheet encloses nothing: zero volume is refused.
  const auto sheet =
      audit({{{0, 0, 0}, {1, 0, 0}, {0, 1, 0}}, {0, 1, 2, 0, 2, 1}});
  CHECK(sheet.boundary_edges == 0 && sheet.inconsistent_edges == 0 &&
        sheet.nonmanifold_edges == 0 && sheet.degenerate_triangles == 0 &&
        sheet.components == 1 && sheet.nonpositive_components == 1 &&
        sheet.signed_volume == 0.0);
  CHECK(!sheet.supports_signed(true));

  // Two triangles that share only a vertex, each holding it last or first,
  // are one component.
  const auto bowtie =
      audit({{{0, 0, 0}, {1, 0, 0}, {0, 0, 1}, {0, 1, 0}, {1, 1, 0}},
             {0, 1, 2, 2, 3, 4}});
  CHECK(bowtie.components == 1);

  // A non-finite position makes every triangle on it degenerate.
  for (const float bad : {std::numeric_limits<float>::quiet_NaN(),
                          std::numeric_limits<float>::infinity()}) {
    auto broken = tetrahedron();
    broken.positions[3] = {bad, bad, bad};
    const auto damaged = audit(broken);
    CHECK(damaged.degenerate_triangles == 3);
    CHECK(!damaged.supports_signed(true));
  }

  // Two components, one inside out.
  auto pair_mesh = tetrahedron();
  auto far_side = shifted(inward_mesh, {100, 0, 0});
  for (auto& index : far_side.indices) index += 4;
  pair_mesh.positions.insert(pair_mesh.positions.end(),
                             far_side.positions.begin(),
                             far_side.positions.end());
  pair_mesh.indices.insert(pair_mesh.indices.end(), far_side.indices.begin(),
                           far_side.indices.end());
  const auto pair = audit(pair_mesh);
  CHECK(pair.components == 2 && pair.nonpositive_components == 1 &&
        pair.edges == 12 && std::abs(pair.signed_volume) < 1e-9);
  CHECK(!pair.supports_signed(true));

  // Coincident positions under distinct indices are distinct vertices.
  auto seam_mesh = tetrahedron();
  const auto apex = seam_mesh.positions[3];
  seam_mesh.positions.push_back(apex);
  seam_mesh.indices.back() = 4;
  CHECK(audit(seam_mesh).boundary_edges == 4);

  // Indices that are not whole triangles, or name a missing position.
  auto partial = tetrahedron();
  partial.indices.pop_back();
  auto missing = tetrahedron();
  missing.indices.back() = 4;
  for (const auto* bad : {&partial, &missing}) {
    const auto result = io::audit_mesh_topology(*bad);
    CHECK(!result && result.status().domain() == Code::InvalidArgument);
  }
  return 0;
}

// An allocation failure is OutOfMemory, and apply leaves the geometry alone.
int out_of_memory() {
  auto mesh = tetrahedron();
  // 37 positions give allocation sizes nothing else in these calls asks for.
  const auto apex = mesh.positions.back();
  mesh.positions.resize(37, apex);
  const auto before = mesh;
  const auto transform =
      io::fit_transform(io::mesh_bounds(mesh, oblique()), 1.7).value();
  auto& failure = vr_test::allocation_failure;
  const auto fail = [&failure](std::size_t bytes, const auto& call) {
    failure = {};
    failure.bytes = bytes;
    failure.armed = true;
    const vkc::Status status = call();
    const bool injected = failure.injected;
    failure = {};
    return injected && status.domain() == Code::OutOfMemory;
  };
  CHECK(fail(37 * sizeof(mesh.positions[0]),
             [&] { return io::apply_transform(mesh, transform); }));
  CHECK(same_bits(mesh, before));
  CHECK(fail(37 * sizeof(std::uint32_t),
             [&] { return io::audit_mesh_topology(mesh).status(); }));
  CHECK(io::apply_transform(mesh, transform));
  return 0;
}

}  // namespace

int main() {
  for (const auto test :
       {union_bounds, fit_against_unit, tilted_tetrahedron, shifted_sequence,
        refusals, topology, out_of_memory}) {
    if (test() != 0) return 1;
  }
  std::puts("mesh normalization tests passed");
  return 0;
}
