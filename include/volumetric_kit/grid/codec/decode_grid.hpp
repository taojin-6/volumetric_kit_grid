// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file codec/decode_grid.hpp
/// @brief A player's grid: built from a frame's header, and grown when a
///        frame outgrows it, which the @ref Decoder itself refuses to do.
///
/// A frame's size does not bound its block count, so both helpers take the
/// caller's ceiling, `max_blocks`: the most blocks a frame may hold. Nothing
/// past it is allocated: a frame of more blocks is refused before it is
/// decoded, and a grow to more slots than @ref make_decode_grid builds for a
/// frame of `max_blocks` before it is made. Each refusal is
/// `Status::Code::InvalidArgument`, a frame to skip. A player of frames it
/// did not code (a file, a socket) sets the ceiling from its memory budget.

#include <cstddef>
#include <cstdint>

#include "volumetric_kit/core/base/result.hpp"
#include "volumetric_kit/core/base/stage_metrics.hpp"
#include "volumetric_kit/core/vulkan/fwd.hpp"
#include "volumetric_kit/grid/codec/decoder.hpp"
#include "volumetric_kit/grid/codec/export.hpp"
#include "volumetric_kit/grid/volume/voxel_block_grid.hpp"

namespace volumetric_kit::grid::codec {

/// @brief Build a grid to decode a stream into, from one of its frames.
///
/// The grid is what @ref Decoder::decode takes: the frame's `voxel_size` and
/// `trunc_dist` (@ref read_frame_info) in @ref volume::block_grid_params'
/// layout, with 4-byte float `tsdf` and `weight` alone, and
/// `ceil(2 * block_count / kDefaultBucketSize)` buckets, at least 1024: the
/// frame fills at most half its blocks. The block count is
/// checked against the frame's segment table first, as @ref Decoder::decode
/// checks it, so a header cannot claim blocks its frame does not carry.
/// @param device      The compute device (must outlive the grid).
/// @param allocator   The allocator its buffers come from (must outlive the
///                    grid).
/// @param frame       A frame of the stream, usually its first, intra or a
///                    P-frame; it is parsed, not decoded.
/// @param size        Its exact length.
/// @param max_blocks  The ceiling: the most blocks a frame may hold.
/// @return The grid, empty; @ref read_frame_info's refusal of the header;
///         `Status::Code::InvalidArgument` for more than @p max_blocks
///         blocks; the refusals @ref Decoder::decode makes of the frame's
///         sections and segment table; or
///         @ref volume::VoxelBlockGrid::create's error, an allocation failure
///         included. Host allocation failures, including header diagnostics,
///         return `Status::Code::OutOfMemory`.
VGRID_CODEC_API core::Result<volume::VoxelBlockGrid> make_decode_grid(
    core::Device& device, core::Allocator& allocator, const std::uint8_t* frame,
    std::size_t size, std::uint32_t max_blocks);

/// @brief What @ref finish_growing did besides finishing the frame.
///
/// @code
/// VKC_ASSIGN(const codec::Finished done,
///            codec::finish_growing(decoder, frame, grid, max_blocks));
/// if (done.redone) remesh();  // its extract ran gated off
/// @endcode
struct Finished {
  /// The frame failed on the device (the grid too small, or contention past
  /// the rounds) and was decoded again from its own slot, blocking. Whatever
  /// its batch recorded after it, such as an extract, ran gated off.
  bool redone = false;
};

/// @brief @ref Decoder::record @p slot's record into @p batch, first growing
///        @p grid, blocking, when the frame holds more blocks than its heap.
///
/// The grow is @ref decode_growing's: to twice the grid's buckets or the
/// frame's @ref make_decode_grid buckets, whichever is more.
/// @param decoder     The decoder.
/// @param batch       Records the frame; the caller submits it.
/// @param slot        From @ref Decoder::claim_input, the record at its start.
/// @param size        The record's exact length.
/// @param grid        As @ref Decoder::record takes it.
/// @param max_blocks  The most blocks a frame may hold.
/// @param metrics     Optional: @ref Decoder::record's rows, and a grow's
///                    `"  ..grow"`, which their `"codec decode"` holds.
/// @return As @ref Decoder::record, out of device memory included, which no
///         grow cures; `Status::Code::InvalidArgument` for a frame or a grow
///         past @p max_blocks; or `volume::grow_grid`'s failure.
/// @pre As @ref Decoder::record. A grow frees the grid's buffers at once.
VGRID_CODEC_API core::Result<PendingDecode> record_growing(
    Decoder& decoder, core::CommandBatch& batch, const core::HostSlot& slot,
    std::size_t size, volume::VoxelBlockGrid& grid, std::uint32_t max_blocks,
    core::StageMetrics* metrics = nullptr);

/// @brief @ref Decoder::finish @p pending, recovering as @ref decode_growing
///        does: decoding the frame again from @p pending's own slot, after a
///        grow for `Status::Code::OutOfMemory` (once), or as it is for
///        `Status::Code::IoError` (twice at most).
///
/// Each recovery blocks, in batches of its own, and is timed whole, its
/// decode again included, as `"  ..grow"` or `"  ..retry"` inside a
/// `"codec decode"` row. It needs no other input slot.
/// @param decoder     The decoder that recorded @p pending.
/// @param pending     Its batch resolved.
/// @param grid        The grid it recorded into.
/// @param max_blocks  The most blocks a frame may hold.
/// @param metrics     Optional: the recoveries' rows.
/// @return What it did; or the last finish's error as @ref decode_growing
///         returns it, with the reference unchanged; or `volume::grow_grid`'s
///         failure.
VGRID_CODEC_API core::Result<Finished> finish_growing(
    Decoder& decoder, const PendingDecode& pending,
    volume::VoxelBlockGrid& grid, std::uint32_t max_blocks,
    core::StageMetrics* metrics = nullptr);

/// @brief @ref Decoder::decode @p frame into @p grid, growing the grid once
///        if the frame outgrows it.
///
/// It recovers as @ref record_growing and @ref finish_growing do, around one
/// submit of its own.
///
/// `Status::Code::OutOfMemory`, the decoder's word for a grid too small,
/// grows @p grid once (@ref volume::grow_grid) to twice its buckets or the
/// frame's @ref make_decode_grid buckets, whichever is more, and decodes
/// again; it is returned the second time. The first can also be the
/// device's own, out of memory for the decode's buffers, which the grow does
/// not cure. Growing a grid @ref make_decode_grid built leaves the frame at
/// most half its heap, so there the second is always the device's.
/// `Status::Code::IoError`, bucket-lock contention that outlasted the
/// decoder's own rounds, decodes again, twice at most. Any other error is
/// returned as the decoder gave it, `Status::Code::NotFound` for a P-frame
/// without its reference included. A grow, a
/// @ref volume::VoxelBlockGrid::resize, replaces the grid's buffers and shows
/// as a larger `grid.grid().num_buckets`.
/// @param decoder     The decoder.
/// @param frame       The frame.
/// @param size        Its exact length.
/// @param grid        The grid to decode into, as @ref Decoder::decode takes
///                    it.
/// @param max_blocks  The ceiling: the most blocks a frame may hold.
/// @param metrics     Optional `StageMetrics` collecting, once the decode
///                    succeeds, the rows @ref Decoder::decode gives for the
///                    first attempt and, after a recovery, its row as
///                    @ref finish_growing times it. `nullptr` measures
///                    nothing.
/// @return OK; @ref read_frame_info's refusal of the header;
///         `Status::Code::InvalidArgument` for a frame or a grow past
///         @p max_blocks; the last decode's
///         error; @ref volume::grow_grid's, such as
///         `Status::Code::OutOfMemory` at the device's largest grid; or
///         `Status::Code::OutOfMemory` for a host allocation failure before
///         the decode succeeds.
VGRID_CODEC_API core::Status decode_growing(
    Decoder& decoder, const std::uint8_t* frame, std::size_t size,
    volume::VoxelBlockGrid& grid, std::uint32_t max_blocks,
    core::StageMetrics* metrics = nullptr);

/// @brief @ref decode_growing a record already in an input slot: one an
///        encoder's frame was copied into, head then payload, with no other
///        copy (the other form copies its bytes into one).
/// @param slot  From @ref Decoder::claim_input, the record at its start.
/// @param size  The record's exact length.
/// @return As the other form.
VGRID_CODEC_API core::Status decode_growing(
    Decoder& decoder, const core::HostSlot& slot, std::size_t size,
    volume::VoxelBlockGrid& grid, std::uint32_t max_blocks,
    core::StageMetrics* metrics = nullptr);

}  // namespace volumetric_kit::grid::codec
