// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file voxelize/mesh_integrator.hpp
/// @brief Write a triangle mesh's truncated distance field into a
///        @ref VoxelBlockGrid's `tsdf` + `weight` attributes, signed or as a
///        shell.

#include <array>
#include <cstdint>

#include "volumetric_kit/core/base/result.hpp"
#include "volumetric_kit/core/base/stage_metrics.hpp"
#include "volumetric_kit/core/vulkan/buffer.hpp"
#include "volumetric_kit/core/vulkan/compute_kernel.hpp"
#include "volumetric_kit/core/vulkan/descriptor.hpp"
#include "volumetric_kit/core/vulkan/fwd.hpp"
#include "volumetric_kit/core/vulkan/gpu_timer.hpp"
#include "volumetric_kit/grid/base/math/vector_types.hpp"
#include "volumetric_kit/grid/volume/voxel_block_grid.hpp"
#include "volumetric_kit/grid/voxelize/export.hpp"

namespace volumetric_kit::grid::voxelize {

/// @brief How @ref MeshIntegrator::integrate turns distance to the mesh into a
///        signed field.
enum class MeshSdfMode : std::uint32_t {
  /// Signed distance to the mesh: negative inside, positive outside, with the
  /// sign read off the closest triangle's face normal -- the rule the research
  /// codec shipped. For a closed mesh whose faces wind counter-clockwise seen
  /// from outside (`cross(v1 - v0, v2 - v0)` points out). The surface extracts
  /// where the mesh is.
  ///
  /// Nothing about the mesh is checked, and no topology is built: the field is
  /// the codec's input, not a reconstruction of the mesh to be measured
  /// against it (the 2026-09-27 decision). What that gives up: an open mesh
  /// grows a skirt up to `trunc_dist` past its rim, a mesh wound inside out
  /// comes back inside out, and past an edge or corner whose faces meet at
  /// under 90 degrees part of its region can take the wrong side, since the
  /// faces tied for nearest there disagree and the lowest-indexed one decides.
  /// A 90-degree edge off the grid's axes does too: past a corner, on the line
  /// that continues an edge, a tied face's normal is perpendicular to the
  /// offset and rounding picks the side.
  /// Use @ref Shell for a mesh that is not closed.
  Signed = 0,
  /// Unsigned distance minus a shell half-thickness: negative within
  /// @ref MeshSdfParams::shell_voxels of any triangle, positive beyond. Needs
  /// nothing from the mesh -- open, non-manifold, self-intersecting and
  /// inconsistently wound meshes are all fine -- and extracts as a thin solid
  /// around every triangle: two walls around an open sheet, and around a closed
  /// solid an outer surface plus an inner one the same distance inside.
  ///
  /// This is the unsigned offset field of Xu & Barbic, "Signed Distance Fields
  /// for Polygon Soup Meshes" (GI 2014), before their interior-component
  /// removal.
  Shell = 1,
};

/// @brief The per-call parameters of @ref MeshIntegrator::integrate.
struct MeshSdfParams {
  /// How the field is signed.
  MeshSdfMode mode = MeshSdfMode::Signed;
  /// @ref MeshSdfMode::Shell only: the shell half-thickness, in voxels. The
  /// prior engine's 1.5 by default.
  ///
  /// Refused below `sqrt(3) / 2` (0.866): a point can be that far from every
  /// voxel, so a thinner shell can fall between voxels and extract with holes.
  /// Refused at or above `trunc_dist` too, where no observed voxel would ever
  /// read as outside.
  float shell_voxels = 1.5f;

  /// @brief Check these parameters against the grid they will write: the
  ///        rule @ref MeshIntegrator::integrate applies, for a caller that
  ///        refuses before any GPU work.
  /// @param grid The grid's geometry: its voxel size and `trunc_dist`.
  /// @return OK, or `Status::invalid_argument` for an unknown mode, or in
  ///         shell mode a @ref shell_voxels below `sqrt(3) / 2` or a shell
  ///         not thinner than `trunc_dist`.
  VGRID_VOXELIZE_API core::Status validate(
      const volume::VoxelGridParams& grid) const;
};

/// @brief What one @ref MeshIntegrator::integrate wrote.
struct MeshIntegrateStats {
  /// Triangles that took part. A zero-area or non-finite triangle is skipped.
  std::uint32_t triangles = 0;
  /// Blocks written: every block the mesh's truncation band reaches.
  std::uint32_t blocks = 0;
  /// Triangle-in-block incidences: what the binning passes scale with, and
  /// the write's worst case -- a voxel measures at most every triangle binned
  /// into its block, and in practice the few no lower bound rules out.
  std::uint32_t bin_entries = 0;
  /// The dispatches the write was split into, each submitted on its own, so
  /// that no submission measures more than
  /// @ref MeshIntegrator::kMaxDispatchBinEntries bin entries.
  std::uint32_t dispatches = 0;
};

/// @brief Writes a triangle mesh's truncated distance field into a
///        @ref VoxelBlockGrid's `tsdf` + `weight` attributes.
///
/// Every voxel of every block the mesh's truncation band reaches is
/// **overwritten**, not blended: a voxel within `trunc_dist` of the mesh gets
/// its distance, clamped to `+-trunc_dist`, and `weight = 1`; every other voxel
/// of those blocks gets `tsdf = 0, weight = 0`, which is what the mesher and
/// the codec read as unobserved. That makes the result look exactly like a
/// frame the codec decoded, and it is what a mesh sequence -- one mesh per
/// frame -- converts through. Blocks the band does not reach keep whatever
/// they held, so to convert one mesh on its own, clear the grid first. The
/// grid's `color` attribute, if it has one, is not touched.
///
/// The blocks must already be allocated:
/// @ref VoxelHashMap::allocate_from_triangles with the same mesh allocates
/// exactly them, because both calls enumerate the same candidates through
/// `volume::triangle_candidate_offsets` and keep them with the same band test.
/// A band block that is missing is refused, with the count, before anything is
/// written.
///
/// Two dispatches bin every triangle into each block its band reaches (a
/// count, then a fill), each over one work item per (triangle, candidate block)
/// pair, so a large triangle costs more items and never a longer one. Then one
/// workgroup per block writes its voxels, in as many submissions as it takes to
/// keep each under @ref kMaxDispatchBinEntries bin entries: it sorts the bin
/// nearest its centre first, and each voxel measures only the triangles no
/// lower bound rules out, stopping once every later one is (mesh_integrate.comp
/// has the bounds, and why their rounding cannot matter). The minimum is taken
/// with the triangle index as tie-break, and a triangle is ruled out only when
/// it is strictly farther than the voxel's best so far, or than `trunc_dist`
/// where that is nearer, so the result does not depend on the order the
/// atomics filled the bin in, nor on which triangles were ruled out -- every
/// voxel writes what measuring its whole bin writes, and the same mesh writes
/// the same bytes.
/// Counts, prefix sums, fill cursors and occupied bins stay on the device.
/// Only validation totals and bounded dispatch ranges are read back. Scratch
/// grows as needed and is retained until this integrator is destroyed: about
/// 16 bytes per grid block slot, plus the largest mesh's inputs and bins.
/// Each call refreshes the mesh and bins.
///
/// A voxel's cost is at most its bin, which a mesh near the grid's resolution
/// keeps in the hundreds. A mesh far finer than the voxels is where it grows,
/// so a bin past @ref kMaxBinTriangles is refused rather than measured:
/// decimate the mesh toward the voxel size first.
///
/// @warning The `Device` and `Allocator` passed to @ref create must
///          outlive this object; it stores references to them. The grid must be
///          quiescent across the call: the binning kernel probes its hash table
///          without a lock, as the mesh tier does.
///
/// @note Every block this writes is stamped changed (@ref integrate), as a
///       fuse stamps one.
class VGRID_VOXELIZE_API MeshIntegrator {
 public:
  /// The most triangles one block's bin may hold. A voxel of the block may
  /// measure all of them, so this and @ref kMaxVoxelsPerBlock bound how long a
  /// workgroup runs. About 230 times the average bin of the 2026-09-27 entry's
  /// sphere, whose triangles are 1.6 voxels on a side; a bin reaches it with
  /// triangles near a tenth of a voxel across, detail the grid cannot hold.
  static constexpr std::uint32_t kMaxBinTriangles = 1u << 16;
  /// The most voxels a block may hold (`VoxelGridParams::voxels_per_block`),
  /// 16^3. One workgroup writes a block, 256 voxels at a time, each 256
  /// scanning the bin again, so this bounds a workgroup's rounds at 16.
  static constexpr std::uint32_t kMaxVoxelsPerBlock = 1u << 12;
  /// The most bin entries one integrate dispatch measures: each is at most
  /// one closest-point test per voxel of its block. A write with more is split
  /// across dispatches, so no one submission grows with the mesh -- the shape
  /// the 2026-08-08 entry records hanging an M5 iPad.
  static constexpr std::uint32_t kMaxDispatchBinEntries = 1u << 20;

  /// @brief Build the binning and integrate pipelines on @p device.
  /// @param device     The compute device (must outlive this object).
  /// @param allocator  The allocator its retained buffers come from (must
  ///                   outlive this).
  /// @return The integrator, or a non-OK `Status` if a pipeline or
  ///         descriptor object fails to build.
  static core::Result<MeshIntegrator> create(core::Device& device,
                                             core::Allocator& allocator);

  // Rule of zero: every owned pipeline / pool / buffer self-frees and
  // self-resets on move; device_ / allocator_ are borrowed, so the defaulted
  // moves leave a moved-from integrator empty (valid() == false).
  ~MeshIntegrator() = default;
  MeshIntegrator(MeshIntegrator&&) noexcept = default;
  MeshIntegrator& operator=(MeshIntegrator&&) noexcept = default;
  MeshIntegrator(const MeshIntegrator&) = delete;
  MeshIntegrator& operator=(const MeshIntegrator&) = delete;

  /// @brief Write the mesh's truncated distance field into @p grid.
  ///
  /// Advances the map's tick (@ref volume::VoxelHashMap::tick) and stamps
  /// `changed` with it on every block it writes, all of them, since each is
  /// overwritten whole (@ref volume::BlockStamp).
  /// @param grid            A grid carrying float `tsdf` and `weight`
  ///                        attributes, with the mesh's band allocated (see
  ///                        the class notes).
  /// @param vertices        World-space vertex positions, metres.
  /// @param vertex_count    How many @p vertices.
  /// @param indices         `3 * triangle_count` indices into @p vertices.
  /// @param triangle_count  How many triangles. 0 writes nothing.
  /// @param params          The mode, and the shell's thickness.
  /// @param metrics         Optional: receives a `"mesh integrate"` row with
  ///                        both halves.
  /// @return What was written, or a non-OK `Status`:
  ///         `Status::Code::InvalidArgument` for a moved-from integrator; a
  ///         grid without float `tsdf` / `weight`, or whose
  ///         `voxels_per_block` is past @ref kMaxVoxelsPerBlock; @p params that
  ///         @ref MeshSdfParams::validate refuses; a null @p vertices /
  ///         @p indices with triangles to read, an index at or past
  ///         @p vertex_count, or a mesh too large for the grid (see
  ///         `volume::triangle_candidate_offsets`); a band block that is not
  ///         allocated; a bin past
  ///         @ref kMaxBinTriangles; or a buffer past what one storage-buffer
  ///         binding can cover. Every refusal comes before the grid is written.
  ///         Otherwise whatever a buffer or a dispatch returns; a dispatch
  ///         that fails after the first has written leaves the blocks before
  ///         it written.
  core::Result<MeshIntegrateStats> integrate(
      volume::VoxelBlockGrid& grid, const Vec3f* vertices,
      std::uint32_t vertex_count, const std::uint32_t* indices,
      std::uint32_t triangle_count, const MeshSdfParams& params = {},
      core::StageMetrics* metrics = nullptr);

  /// @return `true` if this owns live pipelines (`false` when moved-from).
  bool valid() const noexcept { return bin_.valid() && integrate_.valid(); }

 private:
  MeshIntegrator() = default;

  core::Status ensure_scratch(core::Buffer& buffer, VkDeviceSize bytes,
                              const char* name);

  // Borrowed (must outlive this).
  core::Device* device_ = nullptr;
  core::Allocator* allocator_ = nullptr;

  // Device limits read once at create(): the cap on a 1-D dispatch's
  // groupCountX, and on one storage-buffer binding's range. Every input here is
  // per-call and bound whole, so each is checked against the second.
  std::uint32_t max_workgroup_count_x_ = 0;
  VkDeviceSize max_storage_buffer_range_ = 0;

  // The binning kernel (dispatched twice: count, then fill) and the integrate
  // kernel, their sets allocated from pool_ (which must outlive them).
  core::ComputeKernel bin_;
  core::ComputeKernel integrate_;
  core::ComputeKernel scan_counts_;
  std::array<core::ComputeKernel, 3> scan_sums_;
  std::array<core::ComputeKernel, 2> scan_add_;
  core::ComputeKernel compact_bins_;
  core::ComputeKernel bin_ranges_;
  core::DescriptorPool pool_;
  // The device-span collector, idle until a caller passes a StageMetrics.
  core::GpuTimer gpu_timer_;
  // The count pass's one output the host must read before anything is
  // written: how many band blocks the hash table could not find.
  core::Buffer missing_;
  // A 1-element stand-in for the fill pass's output during the count pass, so
  // every declared descriptor stays bound.
  core::Buffer dummy_;
  // Grow-only bulk storage; descriptors are rebound before recording each
  // call so mesh changes, grid replacement and allocation growth are safe.
  core::Buffer vertices_;
  core::Buffer indices_;
  core::Buffer offsets_;
  core::Buffer counts_;
  core::Buffer coordinates_;
  core::Buffer item_slots_;
  core::Buffer bins_;
  core::Buffer blocks_;
  core::Buffer ranges_;
  // 256-way scan levels, including the final 16-byte validation summary.
  std::array<core::Buffer, 4> scan_levels_;
};

}  // namespace volumetric_kit::grid::voxelize
