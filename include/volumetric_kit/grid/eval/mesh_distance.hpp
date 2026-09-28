// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file eval/mesh_distance.hpp
/// @brief How far one triangle mesh is from another: point-to-surface
///        distance, and the accuracy / coverage / F-score a reconstruction or
///        a codec is judged by.
///
/// The two directions:
///  - **accuracy**: each vertex of the mesh under test, its distance to the
///    reference surface -- is what was produced where the surface is?
///  - **coverage**: each vertex of the reference, its distance to the surface
///    under test -- is all of the surface still there?
///
/// Distances are to the nearest point on any triangle, not to the nearest
/// vertex, and are measured up to a `reach`: a point with nothing within it
/// reads as `reach` and is counted apart, so a hole shows up as a count rather
/// than as an unbounded mean. The **F-score** at a threshold is the standard
/// summary of both directions: precision is the fraction of test points
/// within the threshold of the reference, recall the fraction of reference
/// points within it of the test, and F their harmonic mean.
///
/// **What counts as surface:** every triangle but one whose three corners are
/// a single point. That has no extent, and it is what
/// `mesh::MarchingCubes::extract_device_incremental` retires a triangle to.
/// The points measured are the vertices the surface's triangles use, so a
/// vertex no triangle references -- one an incremental extract left
/// unreachable -- is not measured either.
///
/// Host-side and deterministic: the figures depend on the meshes' geometry,
/// never on the order of their vertices or triangles, a `stride` subsample
/// included. Not a tier the pipeline runs: the codec's tests and example
/// measure with it (the 2026-09-27 `eval` decision).

#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <vector>

#include "volumetric_kit/recon/core/math/vector_types.hpp"
#include "volumetric_kit/recon/core/result.hpp"
#include "volumetric_kit/recon/eval/export.hpp"
#include "volumetric_kit/recon/mesh/mesh.hpp"

namespace volumetric_kit::recon::eval {

// TODO(eval): the codec's ground-truth test (#94) carries its own copy of this
// closest-point code; move it onto eval, and drop the copy, once both land.

/// @brief The point of triangle `abc` nearest to @p p (Ericson, "Real-Time
///        Collision Detection", 5.1.5): a corner, a point on an edge, or one
///        inside the face.
/// @param p  The query point.
/// @param a  The triangle's first corner.
/// @param b  Its second corner.
/// @param c  Its third corner.
/// @return The nearest point. A degenerate (zero-area) triangle -- two corners
///         coinciding, whichever two, or all three on one line -- is handled as
///         the segment or point it collapses to.
VR_EVAL_API Vec3f closest_point_on_triangle(Vec3f p, Vec3f a, Vec3f b, Vec3f c);

/// @brief Distance from points to one triangle mesh's surface, up to a reach.
///
/// Built once over a mesh and queried many times. Each triangle is filed
/// under every `reach`-sized cell its bounding box touches, so a query needs
/// only the 27 cells around its own to see every triangle within `reach`. The
/// triangles are **copied** at @ref create, so the source mesh need not
/// outlive this.
class VR_EVAL_API MeshDistance {
 public:
  /// The most cells a triangle may be filed under, on average over the mesh.
  /// A triangle no larger than the reach touches at most 8. Past this the
  /// reach is too small for the triangles, and the index would grow as
  /// `(triangle size / reach)^3`.
  static constexpr std::size_t kMaxCellsPerTriangle = 64;

  /// @brief Build the index over @p mesh's triangles.
  /// @param mesh   The surface: `indices` taken three at a time.
  /// @param reach  The farthest distance measured, metres. It is also the cell
  ///               edge, so it should exceed the mesh's triangle size.
  /// @return The index, or @ref Status::Code::InvalidArgument for:
  ///         - a `reach` that is not finite and positive;
  ///         - an index count that is not a multiple of 3, or an index past
  ///           the vertices;
  ///         - a triangle corner that is not finite, or more than about a
  ///           million reaches from the origin, past the cell keys' range;
  ///         - triangles that would be filed under more than
  ///           @ref kMaxCellsPerTriangle cells each on average.
  static Result<MeshDistance> create(const mesh::Mesh& mesh, float reach);

  /// @brief How far @p p is from the surface.
  /// @param p  The query point.
  /// @return The distance from @p p to the nearest point of the surface, or
  ///         exactly @ref reach when nothing lies within it: an empty mesh,
  ///         and a @p p that is not finite, included.
  float distance(Vec3f p) const;

  /// @return The reach this was built with.
  float reach() const noexcept { return reach_; }

 private:
  MeshDistance() = default;

  float reach_ = 0.0f;
  std::vector<Vec3f> corners_;  // three per triangle
  std::unordered_map<std::uint64_t, std::vector<std::uint32_t>> cells_;
};

/// @brief A distance distribution: over the points within reach, and a count
///        of those beyond it.
struct DistanceStats {
  std::size_t count = 0;         ///< Points measured.
  std::size_t beyond_reach = 0;  ///< Of those, nothing within reach.
  double mean = 0.0;             ///< Metres, over the points within reach.
  double rms = 0.0;              ///< Metres, over the points within reach.
  double p95 = 0.0;  ///< Metres, the nearest-rank 95th percentile of those.
  double max = 0.0;  ///< Metres, over the points within reach.
};

/// @brief Summarize distances measured with @p reach.
/// @param distances  Consumed (sorted in place).
/// @param reach      The reach they were measured with; a distance at it is
///                   counted in @ref DistanceStats::beyond_reach.
/// @return The distribution. Every field but the two counts is 0 when no
///         distance is within reach.
VR_EVAL_API DistanceStats summarize(std::vector<float> distances, float reach);

/// @brief Precision, recall and their harmonic mean at one threshold.
struct FScore {
  float threshold = 0.0f;  ///< Metres.
  double precision = 0.0;  ///< Test points within the threshold, in [0, 1].
  double recall = 0.0;     ///< Reference points within it, in [0, 1].
  double f = 0.0;          ///< `2PR / (P + R)`; 0 when both are 0.
};

/// @brief How @ref compare_meshes measures.
struct CompareOptions {
  /// Farthest distance measured, metres (finite, positive).
  float reach = 0.02f;
  /// Measure about one vertex in `stride` in each direction (at least 1): a
  /// room-sized mesh has a million vertices, and a subsample estimates the
  /// same distribution. A vertex is chosen by a hash of its position, not by
  /// its place in the array, so the same meshes give the same figures
  /// whatever order marching cubes emitted them in.
  std::size_t stride = 1;
  /// The F-score's threshold, metres; 0 for none. At most @ref reach, since
  /// every distance past the reach reads as the reach.
  float fscore_threshold = 0.0f;
};

/// @brief Accuracy, coverage and (optionally) the F-score of a mesh under
///        test against a reference.
struct MeshComparison {
  DistanceStats accuracy;  ///< Test vertices to the reference surface.
  DistanceStats coverage;  ///< Reference vertices to the test surface.
  FScore fscore;           ///< Zeros unless a threshold was asked for.
};

/// @brief Compare two meshes both ways.
///
/// Both meshes are checked before either is indexed, and each index is
/// dropped before the next is built, so one comparison holds one index at a
/// time. To judge many meshes against one reference, index it once with
/// @ref ReferenceMesh.
/// @param reference  The mesh taken as the truth.
/// @param test       The mesh being judged.
/// @param options    Reach, stride and F-score threshold.
/// @return The comparison, or @ref Status::Code::InvalidArgument for a mesh
///         @ref MeshDistance::create refuses, a reach that is not finite and
///         positive, a stride of 0, or an F-score threshold that is negative,
///         not finite, or past the reach.
VR_EVAL_API Result<MeshComparison> compare_meshes(
    const mesh::Mesh& reference, const mesh::Mesh& test,
    const CompareOptions& options = {});

/// @brief A reference mesh indexed once, to judge many meshes against: a
///        sweep's configurations, say, each against the same source.
///
/// Holds a copy of the reference's index and of the points measured from it,
/// so the reference mesh need not outlive this. @ref compare then builds only
/// the index of the mesh under test, and gives exactly what
/// @ref compare_meshes gives for the same pair.
class VR_EVAL_API ReferenceMesh {
 public:
  /// @brief Index @p reference for comparisons measured as @p options says.
  /// @param reference  The mesh taken as the truth.
  /// @param options    Reach, stride and F-score threshold, for every
  ///                   comparison made against it.
  /// @return The reference, or @ref Status::Code::InvalidArgument for
  ///         anything @ref compare_meshes refuses.
  static Result<ReferenceMesh> create(const mesh::Mesh& reference,
                                      const CompareOptions& options = {});

  /// @brief Compare @p test against the reference.
  /// @param test  The mesh being judged.
  /// @return The comparison, or @ref Status::Code::InvalidArgument for a mesh
  ///         @ref MeshDistance::create refuses.
  Result<MeshComparison> compare(const mesh::Mesh& test) const;

  /// @return The options every comparison is measured with.
  const CompareOptions& options() const noexcept { return options_; }

 private:
  ReferenceMesh(const CompareOptions& options, MeshDistance surface,
                std::vector<Vec3f> points);

  CompareOptions options_;
  MeshDistance surface_;
  std::vector<Vec3f> points_;  // the reference's vertices, as measured
};

}  // namespace volumetric_kit::recon::eval
