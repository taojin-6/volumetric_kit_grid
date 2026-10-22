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
#include "volumetric_kit/recon/codec/decoder.hpp"
#include "volumetric_kit/recon/codec/export.hpp"
#include "volumetric_kit/recon/core/fwd.hpp"
#include "volumetric_kit/recon/volume/voxel_block_grid.hpp"

namespace volumetric_kit::recon::codec {

/// @brief Build a grid to decode a stream into, from one of its frames.
///
/// The grid is what @ref Decoder::decode takes: the frame's `voxel_size` and
/// `trunc_dist` (@ref read_frame_info), 8^3-voxel blocks, and 4-byte float
/// `tsdf` and `weight` alone. Its hash table has the default layout
/// (@ref volume::kDefaultBucketSize, @ref volume::kDefaultMaxChain), a heap
/// slot per entry, and `ceil(2 * block_count / kDefaultBucketSize)` buckets,
/// at least 1024: the frame fills at most half its heap. The block count is
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
VR_CODEC_API core::Result<volume::VoxelBlockGrid> make_decode_grid(
    core::Device& device, core::Allocator& allocator, const std::uint8_t* frame,
    std::size_t size, std::uint32_t max_blocks);

/// @brief @ref Decoder::decode @p frame into @p grid, growing the grid once
///        if the frame outgrows it.
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
/// @param metrics     Optional `StageMetrics` collecting the rows
///                    @ref Decoder::decode gives, of the attempt that
///                    succeeds alone, so a frame that grew the grid is timed
///                    as one decode; the grow is not timed. `nullptr`
///                    measures nothing. If aggregating the successful attempt
///                    runs out of host memory, the prior metrics stay
///                    unchanged and the decode still returns OK: its
///                    reference has already committed.
/// @return OK; @ref read_frame_info's refusal of the header;
///         `Status::Code::InvalidArgument` for a frame or a grow past
///         @p max_blocks; the last decode's
///         error; @ref volume::grow_grid's, such as
///         `Status::Code::OutOfMemory` at the device's largest grid; or
///         `Status::Code::OutOfMemory` for a host allocation failure before
///         the decode succeeds.
VR_CODEC_API core::Status decode_growing(Decoder& decoder,
                                         const std::uint8_t* frame,
                                         std::size_t size,
                                         volume::VoxelBlockGrid& grid,
                                         std::uint32_t max_blocks,
                                         core::StageMetrics* metrics = nullptr);

}  // namespace volumetric_kit::recon::codec
