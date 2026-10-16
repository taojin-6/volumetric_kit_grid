// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file io/mesh_normalization.hpp
/// @brief Opt-in host helpers that place a triangle mesh, or every frame of a
///        mesh sequence, in metres with its up direction along +Y, and audit
///        its topology before a signed conversion.
///
/// Nothing in recon calls these: a caller that knows its data's unit or up
/// direction opts in. For a sequence, measure every frame along one set of
/// axes (@ref extend_bounds), fit one @ref MeshTransform to the union, and
/// apply it to every frame, so a point that does not move keeps its place.
/// Host-only, with no GPU work and no Assimp. The source compiles without
/// floating-point contraction, so a transform's doubles are the same on every
/// compiler.

#include <cstddef>
#include <limits>
#include <string>

#include "volumetric_kit/core/base/result.hpp"
#include "volumetric_kit/recon/core/math/vector_types.hpp"
#include "volumetric_kit/recon/io/export.hpp"
#include "volumetric_kit/recon/io/triangle_mesh.hpp"

namespace volumetric_kit::recon::io {

/// @brief A right-handed orthonormal frame, its axes in input coordinates.
///
/// A position p measured along it is `(x·p, up·p, z·p)`. Default-constructed,
/// it is the input's own frame. Take any other from @ref up_axes: the fits and
/// @ref apply_transform refuse axes that are not orthonormal to 1e-6 or whose
/// determinant is not positive.
struct MeshAxes {
  Vec3d x{1.0, 0.0, 0.0};   ///< Measures the rotated X.
  Vec3d up{0.0, 1.0, 0.0};  ///< Measures the rotated Y.
  Vec3d z{0.0, 0.0, 1.0};   ///< Measures the rotated Z; `x × up`.
};

/// @brief The proper rotation that takes a head-up direction to +Y.
///
/// `up` is normalized after an exact power-of-two rescale, so any finite,
/// nonzero length works; `x` is the unit `up × r`, where r is +Z unless
/// `|up.z| >= 0.9`, then +X; and `z = x × up`. A proper rotation keeps
/// triangle winding and the sign of volumes.
/// @param up Head-up direction in input coordinates.
/// @return The axes, or InvalidArgument for a zero or non-finite @p up.
VR_IO_API core::Result<MeshAxes> up_axes(const Vec3d& up);

/// @brief Position bounds measured along a frame's axes, in source units.
///
/// Over no positions the bounds are empty, +inf above -inf; a non-finite
/// measurement makes its axis NaN, which the fits refuse.
struct MeshBounds {
  MeshAxes axes;  ///< The frame every measurement uses.
  /// Per axis, the lowest measurement so far.
  Vec3d min{std::numeric_limits<double>::infinity()};
  /// Per axis, the highest measurement so far.
  Vec3d max{-std::numeric_limits<double>::infinity()};
};

/// @brief Measure every position of a mesh, referenced or not, along @p axes.
/// @param mesh Positions in source units; indices are not read.
/// @param axes The frame; the default measures input coordinates as given.
/// @return The bounds, empty for a mesh with no positions.
VR_IO_API MeshBounds mesh_bounds(const TriangleMesh& mesh,
                                 const MeshAxes& axes = {});

/// @brief Grow @p bounds by every position of @p mesh, measured along
///        `bounds.axes`: the union over a sequence's frames.
///
/// The union is exact and independent of order, so a caller can load, measure
/// and drop one frame at a time.
VR_IO_API void extend_bounds(MeshBounds& bounds, const TriangleMesh& mesh);

/// @brief A similarity that places a mesh in metres.
///
/// A position p maps to `((x·p, up·p, z·p) − origin) · metres_per_unit`,
/// computed in double precision and rounded to float. Default-constructed,
/// it is the identity.
struct MeshTransform {
  MeshAxes axes;      ///< The rotation, its axes in input coordinates.
  Vec3d origin{0.0};  ///< Sent to zero: rotated frame, source units.
  double metres_per_unit = 1.0;  ///< Scale; finite and positive.
};

/// @brief Fit a transform whose scale makes the bounds' extent along up
///        @p height_m metres.
///
/// The transform rotates `bounds.axes.up` to +Y, centres X and Z on the
/// bounds, and puts their lowest Y at zero.
/// @param bounds Non-empty, finite bounds of every position the transform
///        will place, such as the union over a sequence.
/// @param height_m Target extent along up, in metres; finite and positive.
/// @return The transform, or InvalidArgument for empty or non-finite bounds,
///         axes that are not a proper rotation, a zero extent along up, a
///         @p height_m that is not finite and positive, a scale that is not,
///         or bounds it would place past float range, which
///         @ref apply_transform would refuse.
VR_IO_API core::Result<MeshTransform> fit_transform(const MeshBounds& bounds,
                                                    double height_m);

/// @brief The same placement as @ref fit_transform, with the scale given: for
///        data whose unit is known.
/// @param bounds Non-empty, finite bounds of every position the transform
///        will place.
/// @param metres_per_unit Scale; finite and positive (0.001 for millimetres).
/// @return The transform, or InvalidArgument for empty or non-finite bounds,
///         axes that are not a proper rotation, a @p metres_per_unit that
///         is not finite and positive, or bounds it would place past float
///         range.
VR_IO_API core::Result<MeshTransform> scale_transform(const MeshBounds& bounds,
                                                      double metres_per_unit);

/// @brief Map every position of @p mesh through @p transform, in place.
///
/// Indices are kept: the rotation is proper and the scale positive, so
/// winding and orientation hold. A refusal leaves the positions unchanged,
/// bit for bit.
/// @return OK, an empty mesh included; InvalidArgument for axes that are not
///         a proper rotation, a non-finite origin, a scale that is not finite
///         and positive, a non-finite position, an output past float range,
///         or a height collapse: a mesh with extent along up whose outputs
///         round to one Y; OutOfMemory for allocation failure.
VR_IO_API core::Status apply_transform(TriangleMesh& mesh,
                                       const MeshTransform& transform);

/// @brief The length of @ref encode_transform's bytes.
inline constexpr std::size_t kTransformBytes = 112;

/// @brief A @ref MeshTransform as bytes: what a `.vrst` Geometry track's
///        params hold (stream/session.hpp), so a player can place decoded
///        geometry back in the source's coordinates.
///
/// Little-endian, @ref kTransformBytes long; each binary64 is stored bit for
/// bit:
///
/// @code
///   offset  size  field
///        0     4  tag "VRMT"
///        4     4  version (1)
///        8    72  axes x, up, z: 9 binary64, each axis's x, y, z
///       80    24  origin: 3 binary64
///      104     8  metres_per_unit: binary64
/// @endcode
/// @return The bytes; InvalidArgument for a transform @ref apply_transform
///         refuses; OutOfMemory for allocation failure.
VR_IO_API core::Result<std::string> encode_transform(
    const MeshTransform& transform);

/// @brief Read @ref encode_transform's bytes back, bit for bit.
/// @return The transform; InvalidArgument for bytes that are not
///         @ref kTransformBytes long or do not start with "VRMT", or a
///         transform @ref apply_transform refuses; Unsupported for a version
///         other than 1.
VR_IO_API core::Result<MeshTransform> decode_transform(
    const std::string& bytes);

/// @brief Indexed edge and connected-component audit; not a self-intersection
///        test.
///
/// An edge is an unordered index pair, so positions that coincide but have
/// distinct indices are distinct vertices. Each component's signed volume is
/// taken about one of its vertices, so it does not cancel far from the origin.
/// A component with holes is closed first: each group of boundary edges that
/// share vertices is capped by a fan to the centroid of its vertices. Its
/// volume therefore does not depend on index order or on that vertex.
struct MeshTopology {
  std::size_t edges = 0;              ///< Distinct edges.
  std::size_t boundary_edges = 0;     ///< Edges of one triangle.
  std::size_t nonmanifold_edges = 0;  ///< Edges of three or more triangles.
  /// Edges of two triangles that run them the same way.
  std::size_t inconsistent_edges = 0;
  /// Triangles of zero or non-finite area, in double precision.
  std::size_t degenerate_triangles = 0;
  /// Components of triangles joined through shared vertex indices.
  std::size_t components = 0;
  /// Components whose signed volume is zero, negative or non-finite.
  std::size_t nonpositive_components = 0;
  /// Sum over components, holes capped, in source units cubed.
  double signed_volume = 0.0;

  /// @brief True when closest-face signed distance can use the mesh.
  ///
  /// That needs a component, no non-manifold or inconsistently oriented
  /// edge, no degenerate triangle, and every component's signed volume
  /// positive. Boundary edges are refused unless @p allow_holes: the signed
  /// conversion (`tsdf::MeshSdfMode::Signed`) grows an open mesh a skirt
  /// past its rim.
  bool supports_signed(bool allow_holes = false) const noexcept {
    return components > 0 && (allow_holes || boundary_edges == 0) &&
           nonmanifold_edges == 0 && inconsistent_edges == 0 &&
           degenerate_triangles == 0 && nonpositive_components == 0;
  }
};

/// @brief Audit a mesh's indexed topology and component volumes.
///
/// The audit reads the coordinates it is given. A proper similarity, such as
/// a @ref MeshTransform, scales the volumes by the cube of its scale and
/// changes no count, except where rounding makes a triangle's area zero.
/// @return The audit; InvalidArgument for indices that are not whole
///         triangles or name a missing position; OutOfMemory for allocation
///         failure.
VR_IO_API core::Result<MeshTopology> audit_mesh_topology(
    const TriangleMesh& mesh);

}  // namespace volumetric_kit::recon::io
