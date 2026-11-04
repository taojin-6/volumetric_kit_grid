// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file voxelize/mesh_integrator.hpp
/// @brief Make a @ref VoxelBlockGrid hold a triangle mesh's truncated
///        distance field in its `tsdf` + `weight` attributes, signed or as a
///        shell.

#include <cstdint>
#include <memory>

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

namespace detail {
class MeshBand;  // src/volumetric_kit/grid/voxelize/mesh_band.hpp
class MeshScan;  // src/volumetric_kit/grid/voxelize/mesh_scan.hpp
}  // namespace detail

/// @brief How @ref MeshIntegrator::convert turns distance to the mesh into a
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

/// @brief The per-call parameters of @ref MeshIntegrator::convert.
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
  /// How far from the field's zero level, in metres, the band is written
  /// whole. 0, the default, writes every band block whole.
  ///
  /// Above 0, a band block is written whole where its centre lies within
  /// this distance, plus its voxels' half-diagonal, of the mesh -- plus the
  /// shell's half-thickness in shell mode -- so every voxel whose field lies
  /// within this distance of zero is written. Of every other band block, only
  /// the voxels one step -- across a face, an edge or a corner -- from a block
  /// written whole are written; its other voxels, and every voxel of a band
  /// block with none, stay unobserved. Every voxel written holds the bytes a
  /// whole-band write gives it, so a reader that looks only at voxels whose
  /// field lies within this distance of zero, and at their neighbours one
  /// step away, sees the whole band's field. Refused when negative or not
  /// finite.
  float full_band_m = 0.0f;

  /// @brief Check these parameters against the grid they will write: the
  ///        rule @ref MeshIntegrator::convert applies, for a caller that
  ///        refuses before any GPU work.
  /// @param grid The grid's geometry: its voxel size and `trunc_dist`.
  /// @return OK, or `Status::invalid_argument` for an unknown mode, a
  ///         negative or non-finite @ref full_band_m, or in shell mode a
  ///         @ref shell_voxels below `sqrt(3) / 2` or a shell not thinner
  ///         than `trunc_dist`.
  VGRID_VOXELIZE_API core::Status validate(
      const volume::VoxelGridParams& grid) const;
};

/// @brief What one @ref MeshIntegrator::convert wrote.
struct MeshIntegrateStats {
  /// Triangles that took part. A zero-area or non-finite triangle is skipped.
  std::uint32_t triangles = 0;
  /// Blocks written, whole or in part, each allocated by the call: every
  /// block the mesh's truncation band reaches. With
  /// @ref MeshSdfParams::full_band_m, the band's other blocks stay allocated
  /// with no observed voxel.
  std::uint32_t blocks = 0;
  /// With @ref MeshSdfParams::full_band_m: of @ref blocks, those written
  /// whole. 0 without it.
  std::uint32_t full_blocks = 0;
  /// Triangle-in-block incidences: what the binning passes scale with, and
  /// the write's worst case -- a voxel measures at most every triangle binned
  /// into its block, and in practice the few no lower bound rules out.
  std::uint32_t bin_entries = 0;
  /// The ranges the write was split into, each a submission of its own, so
  /// that no submission measures more than
  /// @ref MeshIntegrator::kMaxDispatchBinEntries bin entries.
  std::uint32_t dispatches = 0;
};

/// @brief Converts a triangle mesh into a @ref VoxelBlockGrid's truncated
///        distance field, its `tsdf` + `weight` attributes.
///
/// @ref convert makes the grid hold that mesh's field and nothing else. It
/// empties the grid, every attribute zeroed as
/// @ref volume::VoxelBlockGrid::clear zeroes them; allocates every block the
/// mesh's truncation band reaches -- a block whose centre lies within
/// `trunc_dist` plus the block's half-diagonal of some triangle; and
/// overwrites each of their voxels: within `trunc_dist` of the mesh with its
/// distance, clamped to `+-trunc_dist`, and `weight = 1`; elsewhere with
/// `tsdf = 0, weight = 0`, which the mesher and the codec read as
/// unobserved. The grid then reads exactly like a frame the codec decoded,
/// and a mesh sequence converts one mesh a frame. With
/// @ref MeshSdfParams::full_band_m, only the voxels near the field's zero
/// level, and a one-voxel apron around them, are written.
///
/// The work runs one item per (triangle, candidate block) pair, a
/// triangle's candidates being the blocks of its box grown by `trunc_dist`,
/// so a large triangle costs more items and never a longer one (the
/// 2026-08-31 decision). The pairs are counted on the device. Each pair's
/// band test feeds a bounded index of the band's distinct blocks, and each
/// distinct block meets the hash table once. The write bins the band's
/// triangles by block, and one workgroup per block writes its voxels: it
/// sorts its bin nearest its centre first, and each voxel measures only the
/// triangles no lower bound rules out (mesh_integrate_body.glsl has the
/// bounds, and why their rounding cannot matter). The minimum is taken with
/// the triangle index as tie-break, and a triangle is ruled out only when it
/// is strictly farther than the voxel's best so far, or than `trunc_dist`
/// where that is nearer, so every voxel writes what measuring its whole bin
/// writes, and the same mesh writes the same bytes.
///
/// **Submits and waits.** A call stages the mesh and submits one batch --
/// the count, the clear, the band index, the insert and the write's plan --
/// and waits for it, since the plan's ranges size the write. It then submits
/// each range on its own, back to back with no host wait between them, and
/// waits once, for the last: two blocking waits. Each of these adds a submit
/// and a wait, and runs the first batch's passes again, without the count
/// and the clear: candidate pairs past the per-pair scratch this integrator
/// retains (so always on its first call), an index that outgrows its bound,
/// and a mesh whose candidate blocks span more than 2048 x 1024 x 1024
/// blocks. A grid that must grow adds the resize's own submits and one more
/// run of the first batch.
///
/// **Growth.** A band of more blocks than the grid holds grows the grid once
/// (@ref volume::grow_grid), to twice its buckets or to the buckets that
/// hold the band at @ref volume::VoxelHashMap::kGrowThreshold, whichever is
/// more, keeping the blocks already placed; it doubles again only if a
/// collision chain still overflows. A grid the band fits is never grown,
/// whatever its load. A band past @ref volume::VoxelBlockGrid::max_num_buckets
/// is refused before any resize. A grow replaces the grid's buffers, as
/// @ref volume::VoxelBlockGrid::resize says.
///
/// Scratch grows with the largest mesh and grid, and is retained until this
/// integrator is destroyed: the mesh's arrays, at most 8 bytes a candidate
/// pair, and a few tens of bytes a grid block slot.
///
/// A voxel's cost is at most its bin, which a mesh near the grid's resolution
/// keeps in the hundreds. A mesh far finer than the voxels is where it grows,
/// so a bin past @ref kMaxBinTriangles is refused rather than measured:
/// decimate the mesh toward the voxel size first.
///
/// @warning The `Device` and `Allocator` passed to @ref create must outlive
///          this object; it stores pointers to them.
/// @note Not thread-safe: one @ref convert at a time on an integrator, and
///       nothing else records work on the grid while one runs. Other threads
///       may submit other work to the same device meanwhile, since the core's
///       `Device` serializes queue submission; that work is not ordered
///       against the call's.
/// @note A call is one tick (@ref volume::VoxelHashMap::tick): it advances
///       the tick once, before its first batch, and its allocation stamps
///       each band block `requested`, and its write each block it writes
///       `changed`, with that tick (@ref volume::BlockStamp).
class VGRID_VOXELIZE_API MeshIntegrator {
 public:
  /// The most triangles one block's bin may hold. A voxel of the block may
  /// measure all of them, so this and @ref kMaxVoxelsPerBlock bound how long a
  /// workgroup runs. About 230 times the average bin of the 2026-09-27 entry's
  /// sphere, whose triangles are 1.6 voxels on a side; a bin reaches it with
  /// triangles near a tenth of a voxel across, detail the grid cannot hold.
  static constexpr std::uint32_t kMaxBinTriangles = 1u << 16;
  /// The most voxels a block may hold (`VoxelGridParams::voxels_per_block`),
  /// 16^3. One workgroup writes a block, 512 voxels at a time, each 512
  /// scanning the bin again, so this bounds a workgroup's rounds at 8.
  static constexpr std::uint32_t kMaxVoxelsPerBlock = 1u << 12;
  /// The most bin entries one write submission measures: each is at most one
  /// closest-point test per voxel of its block. A write with more is split
  /// into submissions, so no one submission grows with the mesh -- the shape
  /// the 2026-08-08 entry records hanging an M5 iPad.
  static constexpr std::uint32_t kMaxDispatchBinEntries = 1u << 20;

  /// @brief Build the count, band-index, binning and write pipelines on
  ///        @p device.
  /// @param device     The compute device (must outlive this object).
  /// @param allocator  The allocator its retained buffers come from (must
  ///                   outlive this).
  /// @return The integrator, or a non-OK `Status` if a pipeline or
  ///         descriptor object fails to build.
  static core::Result<MeshIntegrator> create(core::Device& device,
                                             core::Allocator& allocator);

  // Move-only. A moved-from integrator is empty (valid() == false). Defined
  // in the source, which sees detail::MeshBand whole.
  ~MeshIntegrator();
  MeshIntegrator(MeshIntegrator&&) noexcept;
  MeshIntegrator& operator=(MeshIntegrator&&) noexcept;
  MeshIntegrator(const MeshIntegrator&) = delete;
  MeshIntegrator& operator=(const MeshIntegrator&) = delete;

  /// @brief Make @p grid hold the mesh's truncated distance field and
  ///        nothing else: clear it, allocate the mesh's band, write the band.
  ///
  /// Blocks until the field is written (the class notes give its submits and
  /// waits), so the grid can be encoded or meshed as soon as it returns.
  /// @param grid            A grid carrying float `tsdf` and `weight`
  ///                        attributes; grown when the band does not fit.
  /// @param vertices        World-space vertex positions, metres; read
  ///                        during the call only.
  /// @param vertex_count    How many @p vertices.
  /// @param indices         `3 * triangle_count` indices into @p vertices;
  ///                        read during the call only.
  /// @param triangle_count  How many triangles. 0, or none both finite and
  ///                        of non-zero area, leaves the grid empty.
  /// @param params          The mode, the shell's thickness, and how much of
  ///                        the band is written whole.
  /// @param metrics         Optional: receives a `"mesh convert"` row with
  ///                        both halves, and its `"  ..allocate"` and
  ///                        `"  ..write"` parts.
  /// @return What was written. Or, with no GPU work done and the grid as it
  ///         was, `Status::Code::InvalidArgument` for a moved-from integrator;
  ///         a grid without float `tsdf` / `weight`, or whose
  ///         `voxels_per_block` is past @ref kMaxVoxelsPerBlock; @p params
  ///         that @ref MeshSdfParams::validate refuses; a null @p vertices or
  ///         @p indices with triangles to read; an index at or past
  ///         @p vertex_count; or an array past one storage-buffer binding.
  ///         Or, found on the device, with the grid then holding no observed
  ///         voxel: `Status::Code::InvalidArgument` for a candidate total
  ///         past 2^32 or past what one binding of scratch covers (a mesh
  ///         grossly mis-scaled against the grid -- millimetres read as
  ///         metres, say), a band index past one binding, or a bin past
  ///         @ref kMaxBinTriangles; `Status::Code::IoError` for an index that
  ///         kept overflowing as it grew; or `Status::Code::OutOfMemory` for
  ///         a band past the largest grid, refused before any resize.
  ///         Otherwise whatever a buffer, a resize or a submit returns. A
  ///         first batch that fails to record or to run leaves the grid
  ///         undefined until the next @ref convert or
  ///         @ref volume::VoxelBlockGrid::clear; a write range that fails
  ///         leaves those before it written.
  core::Result<MeshIntegrateStats> convert(
      volume::VoxelBlockGrid& grid, const Vec3f* vertices,
      std::uint32_t vertex_count, const std::uint32_t* indices,
      std::uint32_t triangle_count, const MeshSdfParams& params = {},
      core::StageMetrics* metrics = nullptr);

  /// @return `true` if this owns live pipelines (`false` when moved-from).
  bool valid() const noexcept;

 private:
  MeshIntegrator();

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

  // The binning kernel (dispatched twice: count, then fill), the write
  // kernel, the full band's apron and prune passes, the compaction and range
  // kernels, and the kernel that sizes the first batch's passes from the
  // device's pair total; their sets allocated from pool_ (which must outlive
  // them).
  core::ComputeKernel bin_;
  core::ComputeKernel integrate_;
  core::ComputeKernel full_band_;
  core::ComputeKernel compact_bins_;
  core::ComputeKernel bin_ranges_;
  core::ComputeKernel pair_args_;
  core::DescriptorPool pool_;
  // The device-span collector, idle until a caller passes a StageMetrics.
  core::GpuTimer gpu_timer_;
  // The count pass's one output the host must read before anything is
  // written: how many band blocks the hash table could not find.
  core::Buffer missing_;
  // A 1-element stand-in for the fill pass's output during the count pass, so
  // every declared descriptor stays bound.
  core::Buffer dummy_;
  // The full band's tallies: entries and blocks pruned, blocks written whole
  // and for their apron.
  core::Buffer full_band_stats_;
  // Grow-only bulk storage; descriptors are rebound before recording each
  // call so mesh changes, grid replacement and allocation growth are safe.
  // item_slots_ is the per-pair scratch, its size the first batch's bound.
  core::Buffer counts_;
  core::Buffer slot_masks_;
  core::Buffer coordinates_;
  core::Buffer item_slots_;
  core::Buffer bins_;
  core::Buffer blocks_;
  core::Buffer ranges_;
  core::Buffer pair_slices_;
  // The bins' 256-way scan, its last level the 16-byte validation summary.
  std::unique_ptr<detail::MeshScan> scan_;
  // The mesh on the device, its candidate count and offsets, and the index
  // of the band's distinct blocks.
  std::unique_ptr<detail::MeshBand> band_;
};

}  // namespace volumetric_kit::grid::voxelize
