// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file deblock.hpp
/// @brief The decoder-side deblock: a decoded frame smoothed across its block
///        faces on the device, by a filter its header's steps choose.
///
/// Internal (under src/, never installed): the `Decoder` runs it, inside the
/// decode's last submit, when `Decoder::set_deblock` is on. It carries
/// `VR_CODEC_API` so the tier's tests reach it in a shared-library build.
///
/// The filters are the 2026-10-21 quantization study's, on the field
/// `v = tsdf / trunc_dist`, under the per-block fallback (that decision's
/// 2026-10-23 amendment): a **masked** block, one holding an unobserved
/// voxel, keeps its exact decode -- no pass writes it -- and only a **fully
/// observed** block, its mask all ones, is filtered. A missing block is no
/// neighbour.
///
/// - **The line fit** (linfit, n 2, m 3). For each axis, x then y then z, and
///   each block B whose +axis neighbour N exists, B and N both fully
///   observed, each of the 64 lines across their face: p_i is B's voxel at
///   7 - i along the axis and q_i N's at i. When none of p_0..2 and q_0..2
///   has |v| at or above 1 - 1e-5 (the clamp's kink), the offset between the
///   two sides' least-squares lines at the face,
///   `delta = sum_i c_i (q_i - p_i)` with c = (13, 4, -5) / 12, clipped to
///   +-`line_clip`, is spread as a tent: p_0 and q_0 move 0.375 delta toward
///   each other, p_1 and q_1 0.125, each clamped to [-1, 1]. A face with a
///   masked side is not fitted, so the line passes leave both its sides as
///   they were.
/// - **The Gaussian** (r 1). Along x, then y, then z, each voxel of a fully
///   observed block becomes the mean of itself and its observed axis
///   neighbours, weighted 1 and `gauss_g1`, its change clipped to
///   +-`gauss_clip`, then clamped to [-1, 1]. A masked block's voxels keep
///   their values through every pass, so a neighbour's taps read its observed
///   voxels as decoded; its unobserved ones drop out.
///
/// Each pass reads the field as the previous pass left it. A voxel whose
/// value comes out with the bits it went in with is not written, so it keeps
/// its metres exactly.
///
/// On the device: a first kernel, an invocation a frame block, sets the flag
/// by slot of each fully observed block, the rest zeroed before it, and
/// counts the masked blocks into @ref Deblock::exact, for the decoder's
/// results. Then one 64-lane workgroup per frame block in every kernel, which
/// returns
/// at once for a block its own mask words mark masked. The line passes run in
/// place on `tsdf`, a dispatch an axis: a voxel lies on at most one line of a
/// pass, read before it is written. The Gaussian is one fused kernel per
/// block over the block and the 1-voxel halo of its 26 neighbours, x, y and z
/// in shared memory; it writes the block's interior in place, which no other
/// workgroup reads, and its shell (the voxels with a coordinate 0 or 7, which
/// neighbours' halos read) into a snapshot by list position, which a store
/// kernel then copies into `tsdf`. A neighbour is fully observed by its flag,
/// and a masked neighbour's observed voxels are found by their weights;
/// blocks are found by coordinate through the grid's hash, so no list `ptr`
/// is read. A block a pass writes is stamped `changed` with the map's
/// current tick.

#include <cstdint>
#include <memory>

#include "dct_transform.hpp"
#include "volumetric_kit/core/base/result.hpp"
#include "volumetric_kit/core/vulkan/allocator.hpp"
#include "volumetric_kit/core/vulkan/buffer.hpp"
#include "volumetric_kit/core/vulkan/compute_kernel.hpp"
#include "volumetric_kit/core/vulkan/descriptor.hpp"
#include "volumetric_kit/recon/codec/codec_params.hpp"
#include "volumetric_kit/recon/codec/export.hpp"
#include "volumetric_kit/recon/core/dispatch.hpp"
#include "volumetric_kit/recon/core/fwd.hpp"
#include "volumetric_kit/recon/volume/voxel_block_grid.hpp"

namespace volumetric_kit::recon::codec::detail {

/// A block's shell: its voxels with a coordinate 0 or 7 on some axis, which a
/// neighbour's 1-voxel halo reaches.
inline constexpr std::uint32_t kShellVoxels = kVoxelsPerBlock - 6 * 6 * 6;

/// @brief One frame's filter, as the kernels take it, in units of the
///        normalized field.
struct DeblockPasses {
  /// The line passes run, each line's offset clipped to +-@ref line_clip
  /// (positive).
  bool linfit = false;
  float line_clip = 0.0f;
  /// The Gaussian runs after them, a neighbour weighted @ref gauss_g1 against
  /// the voxel's 1, each pass's change clipped to +-@ref gauss_clip
  /// (positive).
  bool gauss = false;
  float gauss_g1 = 0.0f;
  float gauss_clip = 0.0f;
};

/// @brief One row of the step-keyed rule: a frame of K <= @ref max_k whose
///        T_mm is at least @ref min_t_mm runs the row's filter.
struct DeblockRow {
  const char* filter;  ///< Its name: "LF", "LG" or "R".
  std::uint32_t max_k;
  double min_t_mm;
  float line_clip;   ///< In units of T.
  float sigma;       ///< Voxels; 0 runs no Gaussian.
  float gauss_clip;  ///< In units of T.
};

/// The rule the 2026-10-21 study fitted on its rounding streams, first match
/// wins; a frame no row matches runs no filter. The study fitted K 35 and
/// below, and 56 and above; K 36-55 takes the latter's rows.
inline constexpr DeblockRow kDeblockRule[] = {
    {"R", 35, 0.0, 4.0f, 0.75f, 0.75f},
    {"R", kVoxelsPerBlock, 0.6, 4.0f, 0.75f, 0.75f},
    {"LG", kVoxelsPerBlock, 0.3, 4.0f, 0.5f, 0.375f},
    {"LF", kVoxelsPerBlock, 0.075, 4.0f, 0.0f, 0.0f},
};

/// @return T, in units of the normalized field: the largest change one
///         quantum of a kept lowest-frequency coefficient -- DC, and the
///         zigzag's next three, (0,0,1), (0,1,0), (1,0,0) -- makes to a
///         voxel, its step times its basis function's peak, in the study's
///         arithmetic (DC's step the transform's float product, the others'
///         in double) so the rule's thresholds fall where the study's did.
VR_CODEC_API float step_scale(const CodecParams& params);

/// @return The first row of @ref kDeblockRule that K @p k and @p t_mm
///         (T in millimetres) match, or null for none.
VR_CODEC_API const DeblockRow* deblock_rule(std::uint32_t k, double t_mm);

/// @return @p row's filter at T @p t: its clips times @p t in double, and
///         `exp(-1 / (2 sigma^2))`, each rounded to float.
VR_CODEC_API DeblockPasses deblock_passes(const DeblockRow& row, float t);

/// @brief The deblock's kernels, its flags and its shell snapshot.
///
/// The flags take a word a slot of the largest grid heap a filter ran on, and
/// the snapshot @ref kShellVoxels floats a block of the largest frame a
/// Gaussian ran on; each grows with 1.5x headroom (`ensure_device_scratch`),
/// and a frame without a filter takes neither.
///
/// @warning The `Device` and `Allocator` passed to @ref create must outlive
///          this object. Not thread-safe.
class VR_CODEC_API Deblock {
 public:
  /// @brief Build the four kernels, a one-slot flags buffer and a one-block
  ///        snapshot.
  /// @return The deblock, or a pipeline, pool or allocation failure.
  static core::Result<std::unique_ptr<Deblock>> create(
      core::Device& device, core::Allocator& allocator);

  ~Deblock() = default;
  Deblock(const Deblock&) = delete;
  Deblock& operator=(const Deblock&) = delete;
  Deblock(Deblock&&) = delete;
  Deblock& operator=(Deblock&&) = delete;

  /// @brief Make room for a frame of @p n blocks in a grid of @p slots block
  ///        slots under @p passes -- the flags, and the snapshot when a
  ///        Gaussian runs -- before anything is recorded.
  /// @return OK; `Status::Code::InvalidArgument` for a buffer past
  ///         `maxStorageBufferRange`, which grows nothing; or an
  ///         allocation's failure.
  core::Status reserve(std::uint32_t n, std::uint32_t slots,
                       const DeblockPasses& passes);

  /// @brief Record @p passes over @p frame's blocks into @p batch, after the
  ///        writes they read (the inverse transform's), each kernel over its
  ///        slices, which the device sized and gated; the caller submits.
  ///
  /// @p frame must name every block @p grid holds, each once, as a decode
  /// leaves them: a block it lacks reads as masked, and one it names twice
  /// races. Its order and `ptr`s are ignored. Records nothing for passes
  /// that run nothing or no blocks; @ref exact then holds nothing the frame
  /// needs, since every block is left as decoded.
  /// @param batch   The batch to record into; submitted by the caller.
  /// @param grid    The grid, its `tsdf` filtered in place; 4-byte `tsdf` and
  ///                `weight`, `block_size` 8.
  /// @param frame   The frame's list and masks, by list position, as the
  ///                reference store holds them; no reference entries.
  /// @param passes  The filter.
  /// @param blocks  The passes' slices, a workgroup a block, for @p frame's
  ///                count.
  /// @param flags   The flags kernel's, a workgroup 64 blocks.
  /// @param stage   Optional scope the dispatches record their device spans
  ///                into.
  /// @return OK, or, before anything is recorded,
  ///         `Status::Code::InvalidArgument` for a grid of another block
  ///         size or without float `tsdf` and `weight`, a pass without a
  ///         positive clip, a P-frame's entries, more blocks than the grid's
  ///         heap, or more blocks or slots than @ref reserve made room for,
  ///         buffers smaller than the count, or a hash table past
  ///         `maxStorageBufferRange`.
  core::Status record(core::CommandBatch& batch, volume::VoxelBlockGrid& grid,
                      const ResidentBlocks& frame, const DeblockPasses& passes,
                      const recon::detail::IndirectSlices& blocks,
                      const recon::detail::IndirectSlices& flags,
                      core::GpuStageScope* stage = nullptr);

  /// @return The blocks the last @ref record's passes left as decoded, the
  ///         frame's masked blocks, a u32, once its batch ran.
  const core::Buffer& exact() const noexcept { return exact_; }

  /// The flags kernel's blocks a workgroup (`kLanes` in deblock_common.glsl).
  static constexpr std::uint32_t kFlagLanes = 64;

 private:
  Deblock() = default;

  // Borrowed (must outlive this).
  core::Device* device_ = nullptr;
  core::Allocator* allocator_ = nullptr;
  VkDeviceSize max_storage_buffer_range_ = 0;

  // One layout for the four (deblock_common.glsl); declared before pool_,
  // so the pool is destroyed first.
  core::ComputeKernel flags_kernel_;
  core::ComputeKernel line_kernel_;
  core::ComputeKernel gauss_kernel_;
  core::ComputeKernel store_kernel_;
  core::DescriptorPool pool_;
  // The per-block fallback's flags, a word a slot of the grid's heap.
  core::Buffer full_;
  // The frame's masked blocks, one word, zeroed in each record's batch.
  core::Buffer exact_;
  // The Gaussian's shell snapshot, kShellVoxels floats an entry.
  core::Buffer shell_;
};

}  // namespace volumetric_kit::recon::codec::detail
