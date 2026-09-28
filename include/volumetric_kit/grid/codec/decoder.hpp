// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file codec/decoder.hpp
/// @brief Decode an intra frame into a @ref volume::VoxelBlockGrid, and read
///        a frame's header without decoding it.

#include <cstddef>
#include <cstdint>
#include <memory>

#include "volumetric_kit/recon/codec/codec_params.hpp"
#include "volumetric_kit/recon/codec/export.hpp"
#include "volumetric_kit/recon/core/gpu_timer.hpp"
#include "volumetric_kit/recon/core/result.hpp"
#include "volumetric_kit/recon/core/stage_metrics.hpp"
#include "volumetric_kit/recon/volume/voxel_block_grid.hpp"

namespace volumetric_kit::recon {
class Allocator;
class Device;
}  // namespace volumetric_kit::recon

namespace volumetric_kit::recon::codec {

namespace detail {
class DctTransform;
}

/// @brief What a frame's header says: enough to build a grid it decodes into.
struct FrameInfo {
  /// Metres per voxel edge. A grid decoding the frame must match it exactly.
  float voxel_size = 0.0f;
  /// Metres. A grid decoding the frame must match it exactly: the
  /// coefficients are fractions of it.
  float trunc_dist = 0.0f;
  /// Blocks the frame holds; a grid decoding it needs at least this many
  /// (`num_blocks`), plus the hash-table room to place them.
  std::uint32_t block_count = 0;
  /// The coefficient count and steps it was coded with.
  CodecParams params;
};

/// @brief Read a frame's header, and nothing after it.
///
/// What a player calls on its first frame to build a grid of the geometry the
/// stream carries (`voxel_size`, `trunc_dist`, `block_size` 8, and room for
/// `block_count` blocks), before it can @ref Decoder::decode anything.
/// @param data  The frame (at least its header).
/// @param size  Bytes available at @p data.
/// @return The header, or @ref Status::Code::Unsupported for another version,
///         frame type or block size, and @ref Status::Code::InvalidArgument
///         for anything else malformed. It does not check the rest of the
///         frame, which @ref Decoder::decode does.
VR_CODEC_API Result<FrameInfo> read_frame_info(const std::uint8_t* data,
                                               std::size_t size);

/// @brief Decodes intra frames into a caller's @ref volume::VoxelBlockGrid --
///        the codec's playback side (the 2026-09-26 decision).
///
/// After a successful @ref decode the grid holds **exactly** the frame's
/// blocks, whatever it held before: blocks the frame lacks are removed (and
/// their attributes zeroed), missing ones are allocated, and every frame
/// block's `tsdf` and `weight` are rewritten in full -- an observed voxel with
/// its decoded SDF and weight 1.0, an unobserved one as a fresh block holds
/// it. So a player decodes frame after frame into one grid and meshes it after
/// each; nothing is cleared between frames, and a block present in both costs
/// no allocation.
///
/// Only `tsdf` and `weight` are written. Any other attribute of a block the
/// grid keeps across frames (a `color`, say) is left as it was, so a player's
/// grid should declare those two alone.
///
/// @warning The @ref Device and @ref Allocator passed to @ref create must
///          outlive this object. Not thread-safe: one call at a time, and
///          nothing else may modify the grid during one.
class VR_CODEC_API Decoder {
 public:
  /// @brief Build the inverse transform's kernels.
  /// @param device     The compute device (must outlive this object).
  /// @param allocator  The allocator per-call buffers come from (must outlive
  ///                   this object).
  /// @return The decoder, or a pipeline or allocation failure.
  static Result<Decoder> create(Device& device, Allocator& allocator);

  ~Decoder();
  Decoder(Decoder&& other) noexcept;
  Decoder& operator=(Decoder&& other) noexcept;
  Decoder(const Decoder&) = delete;
  Decoder& operator=(const Decoder&) = delete;

  /// @brief Decode one intra frame into @p grid.
  ///
  /// Everything that can be checked before the grid is touched is: the whole
  /// frame is parsed and entropy-decoded, and the grid's geometry and heap
  /// are checked against it. A refusal there leaves @p grid exactly as it
  /// was. A failure **after** the grid starts to change -- the hash table
  /// cannot place every block, or a dispatch fails -- leaves it valid but
  /// holding an unspecified part of the frame; the next successful decode
  /// restores it exactly, since each decode starts from whatever the grid
  /// holds.
  /// @param data     The frame.
  /// @param size     Its exact length.
  /// @param grid     The grid to decode into: 4-byte float `tsdf` and
  ///                 `weight`, `block_size` 8, and the frame's `voxel_size`
  ///                 and `trunc_dist` exactly (see @ref read_frame_info).
  /// @param metrics  Optional @ref StageMetrics collecting a `"codec decode"`
  ///                 row with both halves -- its device half is the inverse
  ///                 transform -- over the breakdown rows `"  ..entropy"`
  ///                 (parsing and decoding the frame), `"  ..active set"`
  ///                 (the compactions), `"  ..apply"` (removing and
  ///                 allocating blocks) and `"  ..transform"`. `nullptr`
  ///                 measures nothing.
  /// @return OK, or: whatever the frame reader refuses
  ///         (@ref Status::Code::Unsupported, @ref
  ///         Status::Code::InvalidArgument for a malformed or corrupt frame);
  ///         @ref Status::Code::InvalidArgument for a moved-from decoder, a
  ///         grid that is moved-from, has another block size or geometry, lacks
  ///         a float `tsdf` / `weight`, or has fewer block slots than the frame
  ///         has blocks; @ref Status::Code::OutOfMemory when the hash table
  ///         cannot place every block (grow the grid with
  ///         @ref volume::VoxelBlockGrid::resize and decode again); otherwise
  ///         a compaction, buffer or dispatch failure.
  Status decode(const std::uint8_t* data, std::size_t size,
                volume::VoxelBlockGrid& grid, StageMetrics* metrics = nullptr);

  /// @return `true` if this owns a live transform (`false` when moved-from).
  bool valid() const noexcept;

 private:
  Decoder();

  std::unique_ptr<detail::DctTransform> transform_;
  // Device spans for the transform; idle until a caller asks for metrics.
  GpuTimer gpu_timer_;
};

}  // namespace volumetric_kit::recon::codec
