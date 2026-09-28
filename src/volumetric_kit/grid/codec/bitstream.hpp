// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file bitstream.hpp
/// @brief The v1 intra frame: what a frame holds, and its byte layout.
///
/// Internal (under src/, never installed): the `Encoder` and `Decoder` build
/// and consume it; nothing outside the tier should see the layout. The layout
/// is the 2026-09-27 decision's, all little-endian:
///
/// @code
///   offset  size  field
///        0     4  magic "VRTC"
///        4     2  version (1)
///        6     1  frame type (0 = intra)
///        7     1  reserved, 0
///        8     4  voxel_size          (f32, metres)
///       12     4  trunc_dist          (f32, metres)
///       16     4  block_size          (8)
///       20     4  coefficient_count K
///       24     4  dc_step             (f32, fraction of trunc_dist)
///       28     4  ac_step             (f32)
///       32     4  block_count N
///       36     4  segment_size R
///       40     4  section_count S
///       44  8 * S  section table: {u16 id, u16 flags, u32 length} each
///       ...        the section bodies, in table order, back to back
/// @endcode
///
/// Sections: TABLES (the frame's frequency tables), SEGMENTS (each segment's
/// stream length, u32), PAYLOAD (the segment streams). A segment is R
/// consecutive blocks coded as one independent rANS stream (@ref rans.hpp);
/// the last may be shorter. A decoder refuses an unknown section flagged
/// @ref kSectionRequired and skips an unknown optional one. Every other flag
/// bit is reserved: zero in v1, and a decoder refuses a known section that
/// sets one, since a later version may define a flag that changes how the
/// body reads.
///
/// Inside a segment each block is, in order: its coordinate (the segment's
/// first in full, the rest as deltas from the block before), its mask class
/// (and, for a partial mask, its 64 bytes), then its K coefficients. Every
/// integer is a *class* -- its bit length -- through a table, plus the bits
/// below its leading one (and a sign) raw.

#include <cstddef>
#include <cstdint>
#include <vector>

#include "dct_blocks.hpp"
#include "volumetric_kit/recon/codec/export.hpp"
#include "volumetric_kit/recon/core/math/vector_types.hpp"
#include "volumetric_kit/recon/core/result.hpp"

namespace volumetric_kit::recon::codec::detail {

/// The four bytes every frame starts with.
inline constexpr std::uint8_t kFrameMagic[4] = {'V', 'R', 'T', 'C'};
/// The layout this file writes and reads.
inline constexpr std::uint16_t kFrameVersion = 1;
/// Bytes before the section table.
inline constexpr std::size_t kFrameHeaderBytes = 44;
/// Bytes per section-table entry.
inline constexpr std::size_t kSectionEntryBytes = 8;
/// Section flag: a decoder that does not know the section must refuse the
/// frame rather than skip it.
inline constexpr std::uint16_t kSectionRequired = 1;
/// Every section flag v1 defines. A known section with any other bit set is
/// refused as @ref Status::Code::Unsupported.
inline constexpr std::uint16_t kSectionKnownFlags = kSectionRequired;
/// Blocks per segment unless the writer is told otherwise: about 1% of a
/// default frame in per-segment overhead, and room0's ~107 k blocks in ~1.7 k
/// independently decodable segments.
inline constexpr std::uint32_t kDefaultSegmentSize = 64;

/// What a frame's header says it is.
enum class FrameType : std::uint8_t {
  kIntra = 0,  ///< Self-contained: decodes with no reference frame.
};

/// The sections v1 defines, numbered `1..kSectionCount`.
enum class SectionId : std::uint16_t {
  kTables = 1,    ///< The frequency tables, one per model.
  kSegments = 2,  ///< Each segment's stream length in bytes, u32.
  kPayload = 3,   ///< The segment streams, back to back.
};
/// How many sections v1 defines.
inline constexpr std::uint32_t kSectionCount = 3;
static_assert(static_cast<std::uint32_t>(SectionId::kPayload) == kSectionCount,
              "section ids run 1..kSectionCount");

/// @brief One intra frame's content, as the writer takes it and the reader
///        returns it.
struct IntraFrame {
  /// The grid's voxel edge, metres: with the block coordinates it places the
  /// frame in the world, so a decoder builds a grid of the same geometry.
  float voxel_size = 0.0f;
  /// Block coordinates, **strictly increasing** in (z, y, x) order (see
  /// @ref coord_less). A format rule, not a convention: it is what makes the
  /// deltas small, and it guarantees the decoder hands the inverse transform
  /// the duplicate-free list it requires. Within a segment the delta code
  /// cannot step backwards; across segments the reader checks it, since a
  /// segment's first coordinate is raw bits.
  std::vector<Vec3i> coords;
  /// The transform output for `coords`, entry for entry.
  DctBlocks blocks;
};

/// @return `true` if @p a sorts before @p b: by z, then y, then x -- so a run
///         of blocks along x is a run of consecutive entries.
inline bool coord_less(const Vec3i& a, const Vec3i& b) noexcept {
  if (a.z != b.z) return a.z < b.z;
  if (a.y != b.y) return a.y < b.y;
  return a.x < b.x;
}

/// Writer options.
struct FrameWriteOptions {
  /// Blocks per segment, at least 1.
  std::uint32_t segment_size = kDefaultSegmentSize;
};

/// @brief Serialize an intra frame.
/// @return The frame's bytes, or @ref Status::Code::InvalidArgument for a
///         non-positive or non-finite `voxel_size` / `trunc_dist`, invalid
///         params, arrays whose sizes disagree with `coords`, coordinates not
///         strictly increasing, a coefficient outside ±32767, a segment size
///         of 0, or a frame whose SEGMENTS or PAYLOAD would outgrow its u32
///         length (more than 2^30 - 1 segments, or 4 GiB of payload).
///         @ref Status::Code::IoError only if the coder refuses a symbol its
///         own tables were counted from, which is a bug here, never the
///         input's.
VR_CODEC_API Result<std::vector<std::uint8_t>> write_intra_frame(
    const IntraFrame& frame, const FrameWriteOptions& options = {});

/// @brief Parse and decode an intra frame.
///
/// Reads nothing outside `[data, data + size)`, whatever the bytes say.
/// @param data        The frame.
/// @param size        Its length; must be exact (trailing bytes are refused).
/// @param max_blocks  The most blocks the caller can hold. A frame's size does
///                    not bound its block count -- a block whose every symbol
///                    has probability one costs no bits -- so this is what
///                    bounds the work and the allocation: `12 + 64 + 4 K`
///                    bytes per block, with K from the header (at most 512),
///                    so at most ~2.1 KB -- about half of the 4 KB of
///                    `tsdf` + `weight` the grid it decodes into holds per
///                    block.
/// @return The frame, or: @ref Status::Code::Unsupported for another version,
///         frame type, block size, an unknown required section, or a known
///         section with a flag v1 does not define;
///         @ref Status::Code::InvalidArgument for anything malformed,
///         truncated or inconsistent (coordinates out of order across
///         segments included), and for more than @p max_blocks blocks;
///         @ref Status::Code::OutOfMemory for a frame whose arrays would not
///         fit this platform's address space (reachable on a 32-bit build).
VR_CODEC_API Result<IntraFrame> read_intra_frame(const std::uint8_t* data,
                                                 std::size_t size,
                                                 std::uint32_t max_blocks);

}  // namespace volumetric_kit::recon::codec::detail
