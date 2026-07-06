// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file mesh/marching_cubes.hpp
/// @brief GPU marching-cubes iso-surface extraction: owns the compute pipeline
///        and drives a GLSL kernel that turns a dense SDF grid into a triangle
///        @ref Mesh.

#include <cstddef>
#include <cstdint>

#include "volumetric_kit/recon/core/buffer.hpp"
#include "volumetric_kit/recon/core/compute_kernel.hpp"
#include "volumetric_kit/recon/core/descriptor.hpp"
#include "volumetric_kit/recon/core/math/vector_types.hpp"
#include "volumetric_kit/recon/core/result.hpp"
#include "volumetric_kit/recon/mesh/export.hpp"
#include "volumetric_kit/recon/mesh/mesh.hpp"
#include "volumetric_kit/recon/volume/hash_types.hpp"

namespace volumetric_kit::recon {
class Device;
class Allocator;
}  // namespace volumetric_kit::recon

namespace volumetric_kit::recon::mesh {

/// @brief A dense grid of SDF samples -- the input to the analytic extraction
///        path this first slice proves.
///
/// Samples are @ref volume::Voxel (signed distance + integration weight),
/// stored x-fastest: sample `(x, y, z)` lives at linear index `x + dims.x * (y
/// + dims.y * z)`. The grid spans `dims` samples per axis, so marching cubes
/// walks `(dims - 1)` cells per axis, each cell's eight corners being adjacent
/// samples. Using @ref volume::Voxel (not a bare `float`) keeps this the same
/// payload the `tsdf` tier will fill on the sparse hash map -- the weight gates
/// out cells that touch an unintegrated voxel.
///
/// @note This dense entry point is the path-proving building block. Extraction
///       straight off the sparse @ref volume::VoxelHashMap (with cross-block
///       neighbour sampling at block boundaries) is the next slice; the
///       per-cell kernel is identical -- only the corner-sampling differs.
struct DenseGrid {
  Vec3i dims{};             ///< Sample count per axis (cells = `dims - 1`).
  float voxel_size = 0.0f;  ///< Metres between adjacent samples.
  Vec3f origin{};           ///< World position of sample index `(0, 0, 0)`.
};

/// @brief Owns the marching-cubes compute pipeline and extracts an iso-surface
///        from a @ref DenseGrid into a host @ref Mesh.
///
/// Built on the `core` compute foundation (@ref Allocator, @ref Buffer,
/// @ref ComputeKernel, @ref Device::submit_single_time), mirroring the volume
/// tier's @ref volume::VoxelHashMap. The kernel runs one invocation per cell,
/// builds the cube index from the eight corner signs, interpolates a vertex on
/// each crossed edge, and appends independent triangles through an atomic bump
/// counter -- the simple, correct form; shared-edge dedup and an incremental
/// block-mesh pool are later slices. Normals come from the SDF gradient -- one
/// central difference over the cell's eight corners, shared by that cell's
/// vertices -- so they point outward (increasing distance). Each vertex also
/// carries the hybrid appearance the renderer consumes: a @ref Vertex::color
/// interpolated from the optional per-sample color input (opaque white when
/// absent), and a @ref Vertex::uv0 left at the `(-1, -1)` sentinel -- the
/// projective-texturing pass (a later slice) fills real atlas coordinates.
///
/// @warning The @ref Device and @ref Allocator passed to @ref create must
///          outlive this object; it stores references to them.
class VR_MESH_API MarchingCubes {
 public:
  /// @brief Create the extractor on @p device, building its pipeline and
  ///        binding it to @p allocator for the per-extract scratch buffers.
  /// @param device     The compute device (must outlive this object).
  /// @param allocator  The allocator its buffers come from (must outlive this).
  /// @return The extractor, or a non-OK @ref Status if a pipeline, layout, or
  ///         descriptor allocation fails.
  static Result<MarchingCubes> create(Device& device, Allocator& allocator);

  // Rule of zero: every owned member (Buffer / ComputeKernel / pool) self-frees
  // and self-resets on move, so the defaulted moves are correct. device_ /
  // allocator_ are borrowed pointers, so a defaulted move leaving the
  // moved-from extractor pointing at them is harmless -- it reports valid() ==
  // false and is only destroyed.
  ~MarchingCubes() = default;
  MarchingCubes(MarchingCubes&&) noexcept = default;
  MarchingCubes& operator=(MarchingCubes&&) noexcept = default;
  MarchingCubes(const MarchingCubes&) = delete;
  MarchingCubes& operator=(const MarchingCubes&) = delete;

  /// @brief Extract the @p iso iso-surface from @p samples over @p grid.
  /// @param samples  The dense SDF grid, x-fastest (see @ref DenseGrid).
  /// @param count    Number of samples; must equal `dims.x * dims.y * dims.z`.
  /// @param grid     The grid dimensions, spacing, and world origin.
  /// @param iso      The iso-value to extract (0 for a raw signed-distance
  ///                 field).
  /// @param colors   Optional per-sample RGB, parallel to @p samples (same
  /// count
  ///                 and x-fastest layout) -- the color the `tsdf` tier fuses
  ///                 into the volume. When non-null, each vertex's
  ///                 @ref Vertex::color is interpolated from it at the edge
  ///                 crossing; when null, vertices are opaque white.
  ///                 @ref Vertex::uv0 is always the `(-1, -1)` sentinel
  ///                 (projective texturing fills it in a later slice).
  /// @return The extracted mesh (empty when the surface misses the grid), or a
  ///         non-OK @ref Status: @ref Status::Code::InvalidArgument for a
  ///         moved-from extractor, a null/mis-sized sample array, or a grid
  ///         that is not at least `2x2x2` samples; a backend error if a buffer
  ///         or the dispatch fails.
  Result<Mesh> extract(const volume::Voxel* samples, std::size_t count,
                       const DenseGrid& grid, float iso = 0.0f,
                       const Vec3u8* colors = nullptr);

  /// @return `true` if this owns a live kernel (`false` when moved-from).
  bool valid() const noexcept { return kernel_.valid(); }

 private:
  MarchingCubes() = default;

  // Borrowed (must outlive this). Pointers, not references, so a moved-from
  // extractor is left in a defined (empty) state.
  Device* device_ = nullptr;
  Allocator* allocator_ = nullptr;

  // Cached maxComputeWorkGroupCount[0]: the ceiling on a 1-D dispatch's
  // groupCountX (Vulkan guarantees only >= 65535), so extract() can reject an
  // over-large grid cleanly instead of risking a device-lost.
  std::uint32_t max_workgroup_count_x_ = 0;

  // The marching-cubes lookup tables, uploaded once and bound at set binding 0
  // for every extract (the counterpart to the volume tier's persistent
  // bindings). The per-extract sample / color / vertex / counter buffers track
  // the grid and are (re)written into bindings 1-4 before each dispatch.
  Buffer tables_;

  // The single marching-cubes kernel -- its descriptor-set layout, pipeline,
  // and the set allocated from the shared pool_ (see @ref ComputeKernel).
  ComputeKernel kernel_;
  DescriptorPool pool_;
};

}  // namespace volumetric_kit::recon::mesh
