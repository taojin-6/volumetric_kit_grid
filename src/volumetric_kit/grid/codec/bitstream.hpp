// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file bitstream.hpp
/// @brief The v3 frame: what a frame holds, and its byte layout.
///
/// Internal (under src/, never installed): the `Encoder` and `Decoder` build
/// and consume it; nothing outside the tier should see the layout. The layout
/// is the 2026-09-27 decision's, all little-endian:
///
/// @code
///   offset  size  field
///        0     4  magic "VRTC"
///        4     2  version (3)
///        6     1  frame type (0 = intra, 1 = P-frame)
///        7     1  reserved, 0
///        8     4  voxel_size          (f32, metres)
///       12     4  trunc_dist          (f32, metres)
///       16     4  block_size          (8)
///       20     4  coefficient_count K
///       24     4  quantization_scale  (f32, fraction of trunc_dist)
///       28     4  quantization weight count (K)
///       32     4  block_count N
///       36     4  segment_size R
///       40     4  section_count S
///       44   4 K  quantization weights (K f32, kept bases, zigzag order)
///   44+4K  8 * S  section table: {u16 id, u16 flags, u32 length} each
///       ...        the section bodies, in table order, back to back
/// @endcode
///
/// Sections: TABLES (the frame's frequency tables), SEGMENTS (each segment's
/// stream length, u32), PAYLOAD (the segment streams). A segment is R
/// consecutive blocks coded as one independent rANS stream (@ref rans.hpp);
/// the last may be shorter. A decoder refuses an unknown section flagged
/// @ref kSectionRequired and skips an unknown optional one. Every other flag
/// bit is reserved: zero in v3, and a decoder refuses a known section that
/// sets one, since a later version may define a flag that changes how the
/// body reads.
///
/// Inside a segment the blocks follow the block grammar of
/// `shaders/frame_grammar.glsl`, through the models of
/// `shaders/frame_models.glsl`: the one text both the host and the device
/// code a frame with.
///
/// Version 3 replaces DC/AC steps with a global scale and a weight for each of
/// the K kept bases, 256 bytes at the default K = 64; weights beyond K never
/// apply and are not carried. Older versions are refused. The fixed header
/// includes the weights, so `read_frame_info` needs no entropy decode and
/// returns every parameter the transform uses.
///
/// Version 2 (2026-10-01) is version 1 with the grammar's mask code and sign;
/// version 1 sent a partial mask as 64 bytes through one table and a sign as
/// a field of its own.
///
/// **P-frames** (type 1, experimental: its bytes may change until golden
/// P-frames freeze it) code a frame against its reference, the frame decoded
/// before it, named by @ref frame_hash of the reference's exact bytes
/// (@ref FrameReference). A P-frame adds a REFERENCE section (id 4), required
/// on type 1 and refused on type 0, whose 20-byte body is {u64 hash, u32 M,
/// the reference's block count, u32 U, reference entries per update segment
/// (at least 1), u32 A, new blocks}. N stays the decoded frame's block count:
/// the reference blocks that survive plus the A new ones, so
/// A <= N <= M + A, and M + A < 2^32, since stream positions are u32. R
/// counts new blocks per segment.
///
/// `frame_segment_count(M, U)` update segments come first, then
/// `frame_segment_count(A, R)` new-block segments, their lengths in SEGMENTS
/// and their streams in PAYLOAD in that order. Reference entry e sits at
/// stream position e and new block q at M + q (`segment_range` in
/// `frame_models.glsl`); an intra frame is the case M = 0. An update segment
/// holds its entries in reference order, each a mode symbol, then a payload by
/// mode:
///
/// @code
///   mode    survives  payload
///   keep    yes       the mask XOR the reference's, then K residuals,
///                     int16((q - q_ref) mod 2^16)
///   skip    yes       none: the reference block as it is
///   zero    yes       the mask and K coefficients, as an intra block's
///   remove  no        none
///   motion  reserved  none defined: a frame that uses it is Unsupported
/// @endcode
///
/// A new-block segment is an intra segment. The new coordinates strictly
/// increase across all A, and none is a reference coordinate (a reference
/// block is recoded with zero, never removed and added), so @ref merge_frame
/// merges survivors and new blocks into the frame's order with no ties. Keep,
/// zero and new blocks code through the intra models, and the modes through
/// the mode model (`frame_models.glsl`). A P-frame's `voxel_size`,
/// `trunc_dist`, K, scale and kept weights equal its reference's bit for bit.
///
/// A reader refuses a P-frame in this order: the header; the section table
/// (an unknown required section or an undefined flag `Unsupported`, a
/// duplicate, an overrun, trailing bytes or REFERENCE's presence
/// `InvalidArgument`); REFERENCE's body, then TABLES (`InvalidArgument`), a
/// nonzero motion frequency (`Unsupported`, before anything after the mode
/// table is read), then SEGMENTS (`InvalidArgument`); then against the held
/// reference: none held, or another hash or M, `NotFound` (the frame is sound
/// and decodes once its reference is held), other coding parameters
/// `InvalidArgument`; N past the caller's room `OutOfMemory`; then each
/// segment (@ref check_segment), the survivor count (@ref check_survivors)
/// and the new coordinates against the reference's (@ref merge_frame), each
/// `InvalidArgument`.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

#include "dct_blocks.hpp"
#include "rans.hpp"
#include "volumetric_kit/core/base/result.hpp"
#include "volumetric_kit/recon/codec/export.hpp"
#include "volumetric_kit/recon/core/math/vector_types.hpp"
#include "volumetric_kit/recon/volume/hash_types.hpp"

namespace volumetric_kit::recon::codec::detail {

/// The four bytes every frame starts with.
inline constexpr std::uint8_t kFrameMagic[4] = {'V', 'R', 'T', 'C'};
/// The layout this file writes and reads.
inline constexpr std::uint16_t kFrameVersion = 3;
/// Bytes before the quantization weights (the metadata prefix).
inline constexpr std::size_t kFramePrefixBytes = 44;
/// Bytes before the section table of a frame keeping @p k coefficients:
/// the prefix and one weight per kept basis.
constexpr std::size_t frame_header_bytes(std::uint32_t k) {
  return kFramePrefixBytes + std::size_t{k} * sizeof(float);
}
/// Bytes per section-table entry.
inline constexpr std::size_t kSectionEntryBytes = 8;
/// Section flag: a decoder that does not know the section must refuse the
/// frame rather than skip it.
inline constexpr std::uint16_t kSectionRequired = 1;
/// Every section flag v3 defines. A known section with any other bit set is
/// refused as `Status::Code::Unsupported`.
inline constexpr std::uint16_t kSectionKnownFlags = kSectionRequired;
/// Blocks per segment unless the writer is told otherwise:
/// @ref EncoderConfig::segment_size's default, whose doc gives its cost.
inline constexpr std::uint32_t kDefaultSegmentSize = 64;
/// A P-frame's reference entries per update segment unless the writer is
/// told otherwise.
inline constexpr std::uint32_t kDefaultUpdateSize = 64;
/// The most segments a frame holds: SEGMENTS' u32 length bounds it.
inline constexpr std::uint64_t kMaxFrameSegments =
    std::numeric_limits<std::uint32_t>::max() / 4;
/// Bytes in a REFERENCE section's body.
inline constexpr std::uint32_t kReferenceBodyBytes = 20;

/// @return How many segments a frame of @p blocks blocks has at
///         @p segment_size (at least 1) blocks a segment.
constexpr std::uint64_t frame_segment_count(std::uint64_t blocks,
                                            std::uint32_t segment_size) {
  return blocks == 0 ? 0 : (blocks - 1) / segment_size + 1;
}

/// @return The stream positions `[x, y)` segment @p s covers in a frame of
///         @p m reference entries, @p u to an update segment, and @p a new
///         blocks, @p r to a segment: `frame_models.glsl`'s `segment_range`,
///         for the device coders' host halves. An intra frame is `m = 0`.
VR_CODEC_API glm::uvec2 frame_segment_range(std::uint32_t s, std::uint32_t m,
                                            std::uint32_t u, std::uint32_t a,
                                            std::uint32_t r);

/// What a frame is: byte 6 of its header. @ref write_frame and the readers
/// refuse a type they do not code (@ref check_frame_type).
enum class FrameType : std::uint8_t {
  kIntra = 0,  ///< Self-contained: decodes with no reference frame.
  kInter = 1,  ///< A P-frame: decodes against its reference frame.
};

/// @brief The type check @ref write_frame and every reader make.
/// @param newest  The last type the caller codes: every type up to it is.
/// @param who     The caller, which a refusal names.
/// @return OK for a type up to @p newest; otherwise
///         `Status::Code::Unsupported`.
VR_CODEC_API core::Status check_frame_type(FrameType type, FrameType newest,
                                           const char* who);

/// The sections v3 defines, numbered `1..kSectionCount`.
enum class SectionId : std::uint16_t {
  kTables = 1,     ///< The frequency tables, one per model.
  kSegments = 2,   ///< Each segment's stream length in bytes, u32.
  kPayload = 3,    ///< The segment streams, back to back.
  kReference = 4,  ///< A P-frame's reference and its stream counts.
};
/// How many sections v3 defines.
inline constexpr std::uint32_t kSectionCount = 4;
static_assert(static_cast<std::uint32_t>(SectionId::kReference) ==
                  kSectionCount,
              "section ids run 1..kSectionCount");

/// A P-frame's mode for one entry of its reference: one symbol of the mode
/// model, `frame_models.glsl`'s `kMode*`.
enum class BlockMode : std::uint8_t {
  kKeep = 0,    ///< Survives: an XOR mask and K residuals.
  kSkip = 1,    ///< Survives as it is: no payload.
  kZero = 2,    ///< Survives, recoded: a mask and K coefficients.
  kRemove = 3,  ///< Gone: no payload.
  kMotion = 4,  ///< Reserved for motion compensation; refused everywhere.
};

/// The frame's order: @ref volume::coord_less, by z, then y, then x -- so a
/// run of blocks along x is a run of consecutive entries.
using volume::coord_less;

/// @brief Sort @p blocks into the order a frame holds them in (@ref
///        coord_less): the order the encoder writes and the decoder's merge
///        walks, so the two cannot drift apart.
inline void sort_by_coord(std::vector<volume::BlockIndex>& blocks) {
  std::sort(blocks.begin(), blocks.end(),
            [](const volume::BlockIndex& a, const volume::BlockIndex& b) {
              return coord_less(a.coord, b.coord);
            });
}

/// The index @ref merge_by_coord passes for the side with no entry at a
/// coordinate.
inline constexpr std::size_t kAbsent = std::numeric_limits<std::size_t>::max();

/// @brief Walk two lists, each strictly increasing by @ref coord_less, in
///        that order: the one merge the decoder's grid update, a P-frame's
///        block order and the encoder's split share.
/// @param left, right  Callables giving entry `i` of each list's coordinate.
/// @param visit  Called once per distinct coordinate, in order, as
///               `visit(i, j)`, with @ref kAbsent for the side that lacks it;
///               it returns a `core::Status`.
/// @return OK, or the first non-OK status @p visit returned, where the walk
///         stopped.
template <typename Left, typename Right, typename Visit>
core::Status merge_by_coord(std::size_t nl, Left left, std::size_t nr,
                            Right right, Visit visit) {
  std::size_t i = 0;
  std::size_t j = 0;
  while (i < nl || j < nr) {
    core::Status s;
    if (j == nr || (i < nl && coord_less(left(i), right(j)))) {
      s = visit(i++, kAbsent);
    } else if (i == nl || coord_less(right(j), left(i))) {
      s = visit(kAbsent, j++);
    } else {
      s = visit(i++, j++);
    }
    if (!s.ok()) return s;
  }
  return {};
}

/// @brief A frame's header, parsed and checked without decoding anything.
struct FrameHeader {
  /// What the frame is: a type this reader decodes.
  FrameType type = FrameType::kIntra;
  float voxel_size = 0.0f;  ///< Metres per voxel edge.
  float trunc_dist = 0.0f;  ///< Metres; the steps are fractions of it.
  CodecParams params;       ///< K, global scale, and the kept bases' weights.
  std::uint32_t block_count = 0;    ///< Blocks the frame holds.
  std::uint32_t segment_size = 0;   ///< Blocks per segment, at least 1.
  std::uint32_t section_count = 0;  ///< Entries in the section table.
};

/// @brief The frame a P-frame codes against: what a writer and a reader hold
///        of it.
struct FrameReference {
  /// @ref frame_hash of the reference frame's exact bytes.
  std::uint64_t hash = 0;
  /// Its header: M is `header.block_count`, and the coding parameters a
  /// P-frame must repeat.
  FrameHeader header;
};

/// @brief One frame's content, as the writer takes it and the reader
///        returns it.
struct Frame {
  FrameType type = FrameType::kIntra;  ///< What the frame is.
  /// The grid's voxel edge, metres: with the block coordinates it places the
  /// frame in the world, so a decoder builds a grid of the same geometry.
  float voxel_size = 0.0f;
  /// A P-frame's reference. An intra frame has none: hash 0 and
  /// `header.block_count` 0.
  FrameReference reference;
  /// A P-frame's mode for each of its reference's M entries, in reference
  /// order; empty for an intra frame.
  std::vector<BlockMode> modes;
  /// Block coordinates, **strictly increasing** in (z, y, x) order (see
  /// @ref coord_less): a P-frame's A new blocks, an intra frame's every
  /// block. A format rule, not a convention: it is what makes the deltas
  /// small, and it guarantees the decoder hands the inverse transform the
  /// duplicate-free list it requires. Within a segment the delta code cannot
  /// step backwards; across segments the reader checks it, since a segment's
  /// first coordinate is raw bits.
  std::vector<Vec3i> coords;
  /// The payload by stream position: a P-frame's M reference entries, then
  /// its new blocks, entry for entry with `coords`; an intra frame's blocks.
  /// A keep entry holds its XOR mask and residuals. Skip and remove entries
  /// are zero when read and ignored when written.
  DctBlocks blocks;
};

/// @brief Parse and check a frame's fixed header -- the first
///        @ref frame_header_bytes -- and nothing after it.
/// @return The header, or the same header refusals as @ref read_frame.
VR_CODEC_API core::Result<FrameHeader> read_frame_header(
    const std::uint8_t* data, std::size_t size);

/// Writer options.
struct FrameWriteOptions {
  /// Blocks per segment, at least 1: a P-frame's new blocks per segment.
  std::uint32_t segment_size = kDefaultSegmentSize;
  /// A P-frame's reference entries per update segment, at least 1.
  std::uint32_t update_size = kDefaultUpdateSize;
};

/// @brief Serialize a frame.
/// @return The frame's bytes; `Status::Code::Unsupported` for a type past
///         @ref FrameType::kInter or a @ref BlockMode::kMotion mode;
///         `Status::Code::InvalidArgument` for a non-positive or non-finite
///         `voxel_size` / `trunc_dist`, invalid params, a segment or update
///         size of 0, a mode count other than the reference's block count, a
///         mode past @ref BlockMode::kMotion, M + A past 2^32 - 1, arrays
///         whose sizes disagree with the entries, coordinates not strictly
///         increasing, a written coefficient or residual outside ±32767, a
///         P-frame whose coding parameters are not its reference's, an intra
///         frame with a reference, or a frame whose SEGMENTS or PAYLOAD would
///         outgrow its u32 length (more than 2^30 - 1 segments, or 4 GiB of
///         payload). `Status::Code::IoError` only if the coder refuses a
///         symbol its own tables were counted from, which is a bug here,
///         never the input's.
VR_CODEC_API core::Result<std::vector<std::uint8_t>> write_frame(
    const Frame& frame, const FrameWriteOptions& options = {});

/// @brief The refusals @ref write_frame makes after the type's, of the
///        coding parameters, which @ref assemble_frame and the device
///        writer make first too: one place for their codes and words.
/// @return OK, or `Status::Code::InvalidArgument` for a non-positive or
///         non-finite @p voxel_size or @p trunc_dist, or invalid @p params.
VR_CODEC_API core::Status check_coding(float voxel_size, float trunc_dist,
                                       const CodecParams& params);

/// @brief The last refusals @ref write_frame makes, of what its entries
///        hold, which the device writer's count pass finds: one place for
///        both writers' codes and words.
/// @param largest_mode  The largest mode past @ref BlockMode::kMotion; 0 for
///                      none.
/// @param motion        Whether a mode is @ref BlockMode::kMotion.
/// @param out_of_range  Whether a written coefficient or residual (a keep,
///                      zero or new entry's) is outside ±32767.
/// @return OK, or in this order: `Status::Code::InvalidArgument` for a mode
///         past the alphabet, `Status::Code::Unsupported` for motion, and
///         `Status::Code::InvalidArgument` for a value out of range.
VR_CODEC_API core::Status check_entries(std::uint32_t largest_mode, bool motion,
                                        bool out_of_range);

/// @return How many frequency tables a frame of @p type keeping @p k
///         coefficients carries, in TABLES order.
VR_CODEC_API std::uint32_t frame_model_count(FrameType type, std::uint32_t k);
/// @return The alphabet size of frequency table @p model at @p k.
VR_CODEC_API std::uint32_t frame_model_alphabet(std::uint32_t model,
                                                std::uint32_t k);
/// @return The first entry of table @p model at @p k in the kernels'
///         per-symbol arrays, every table's symbols back to back in TABLES
///         order; for @ref frame_model_count, the arrays' size.
VR_CODEC_API std::uint32_t frame_model_base(std::uint32_t model,
                                            std::uint32_t k);

/// @brief Normalize a frame's per-model symbol counts into its tables, the
///        one way every writer builds them.
VR_CODEC_API std::vector<FrequencyTable> frame_tables(
    const std::vector<std::vector<std::uint64_t>>& counts);

/// @brief A frame whose segments are already coded, which
///        @ref assemble_frame lays out.
struct CodedFrame {
  float voxel_size = 0.0f;
  float trunc_dist = 0.0f;
  CodecParams params;
  std::uint32_t block_count = 0;
  std::uint32_t segment_size = 0;
  /// A P-frame's reference, and its update size and new block count; for an
  /// intra frame, hash, block counts and sizes 0.
  FrameReference reference;
  std::uint32_t update_size = 0;
  std::uint32_t new_count = 0;
  /// @ref frame_model_count tables, in TABLES order.
  std::vector<FrequencyTable> tables;
  /// Each segment's stream length in bytes.
  std::vector<std::uint32_t> segment_lengths;
  /// The segment streams, back to back.
  std::vector<std::uint8_t> payload;
};

/// @brief Lay out a coded frame: header, section table, then REFERENCE for a
///        P-frame, TABLES, SEGMENTS and PAYLOAD. Shared by the host and
///        device writers, so their containers cannot differ.
/// @param type  Written to byte 6 as given: the caller coded the segments
///              for it.
/// @return The frame's bytes; `Status::Code::Unsupported` for a type past
///         @ref FrameType::kInter; or `Status::Code::InvalidArgument` for a
///         non-positive or non-finite `voxel_size` / `trunc_dist`, invalid
///         params, a segment size of 0, a P-frame's update size of 0, M + A
///         past 2^32 - 1, a block count outside [A, M + A], coding
///         parameters other than the reference's, reference fields set on an
///         intra frame, tables or segment lengths whose count the header
///         disagrees with, lengths that do not sum to the payload, a payload
///         past 4 GiB or more than 2^30 - 1 segments.
VR_CODEC_API core::Result<std::vector<std::uint8_t>> assemble_frame(
    FrameType type, const CodedFrame& frame);

/// @brief Parse and decode a frame.
///
/// Reads nothing outside `[data, data + size)`, whatever the bytes say.
/// @param data        The frame.
/// @param size        Its length; must be exact (trailing bytes are refused).
/// @param max_blocks  The most blocks the caller can hold. A frame's size does
///                    not bound its block count -- a block whose every symbol
///                    has probability one costs no bits -- so this is what
///                    bounds the work and the allocation: `12 + 64 + 2 K`
///                    bytes per block, with K from the header (at most 512),
///                    so at most ~1.1 KB -- about a quarter of the 4 KB of
///                    `tsdf` + `weight` the grid it decodes into holds per
///                    block. Checked once the block count agrees with the
///                    segment table, so a corrupt count is refused as corrupt
///                    and only a sound frame is refused for its size. A
///                    P-frame's M reference entries are bounded by the
///                    reference the caller holds.
/// @param reference   The reference the caller holds, for a P-frame; null
///                    for none.
/// @return The frame, or: `Status::Code::Unsupported` for another version,
///         frame type, block size, an unknown required section, a known
///         section with a flag v3 does not define, or a P-frame that uses
///         the motion mode; `Status::Code::InvalidArgument` for anything
///         malformed, truncated or inconsistent (coordinates out of order
///         across segments, and a P-frame whose coding parameters are not
///         its reference's, included); `Status::Code::NotFound` for a
///         P-frame whose reference is not @p reference (none held, or
///         another hash or block count); `Status::Code::OutOfMemory` for a
///         well-formed frame of more than @p max_blocks blocks, or one whose
///         arrays would not fit this platform's address space (reachable on a
///         32-bit build). The order the refusals are made in is this file's.
VR_CODEC_API core::Result<Frame> read_frame(
    const std::uint8_t* data, std::size_t size, std::uint32_t max_blocks,
    const FrameReference* reference = nullptr);

/// @brief A frame parsed and checked up to its segment streams: what a
///        reader, on the host or the device, decodes them from.
struct ParsedFrame {
  FrameHeader header;
  /// A P-frame's reference, as the caller held it (M is
  /// `reference.header.block_count`); none for an intra frame, whose M is
  /// 0.
  FrameReference reference;
  /// A P-frame's reference entries per update segment, and its new blocks;
  /// 0 for an intra frame, as in @ref CodedFrame.
  std::uint32_t update_size = 0;
  std::uint32_t new_count = 0;
  /// @ref frame_model_count tables, in TABLES order, without the host
  /// reader's slot lookups (@ref decode_frame builds them).
  std::vector<FrequencyTable> tables;
  /// Each segment's stream length in bytes: even, at least 4, summing to
  /// `payload_size`.
  std::vector<std::uint32_t> segment_lengths;
  /// The segment streams, back to back, inside the caller's frame.
  const std::uint8_t* payload = nullptr;
  std::size_t payload_size = 0;
};

/// @brief Everything @ref read_frame checks before it decodes a
///        segment: the header, the sections, the tables, the segment
///        lengths, a P-frame against @p reference, and the block count
///        against @p max_blocks.
/// @return The parsed frame, or those refusals, as @ref read_frame
///         makes them.
VR_CODEC_API core::Result<ParsedFrame> parse_frame(
    const std::uint8_t* data, std::size_t size, std::uint32_t max_blocks,
    const FrameReference* reference);

/// @brief Decode a parsed frame's segments on the host: @ref read_frame
///        after @ref parse_frame.
/// @param parsed  Taken by value: its tables gain the slot lookups only this
///                reader needs.
/// @return The frame; `Status::Code::Unsupported` for a type it does not
///         decode (@ref check_frame_type); or @ref read_frame's refusals of
///         its segments and survivors (and `Status::Code::OutOfMemory` for
///         arrays past this platform's address space).
VR_CODEC_API core::Result<Frame> decode_frame(ParsedFrame parsed);

/// How a segment's stream failed to decode.
enum class SegmentFault : std::uint32_t {
  kNone = 0,
  /// A delta stepped a coordinate outside int32, before anything else
  /// failed.
  kCoordOverflow = 1,
  /// The stream did not decode cleanly and completely (@ref
  /// RansReader::finish).
  kCorrupt = 2,
};

/// @brief The reader's verdict on segment @p s, taken in segment order: its
///        fault, else whether its first coordinate follows the segment
///        before's last. One place for both readers' refusals.
/// @param prev_last  The last coordinate of the block segment before; null
///                   for the first.
/// @param first      Segment @p s's first coordinate; null for a P-frame's
///                   update segment, which has none.
VR_CODEC_API core::Status check_segment(std::uint64_t s, SegmentFault fault,
                                        const Vec3i* prev_last,
                                        const Vec3i* first);

/// @brief A P-frame's survivor count, which only a whole decode knows: one
///        place for both readers' refusal.
/// @return OK when exactly @p n - @p a of @p modes survive (keep, skip or
///         zero); otherwise `Status::Code::InvalidArgument`.
VR_CODEC_API core::Status check_survivors(const std::vector<BlockMode>& modes,
                                          std::uint32_t n, std::uint32_t a);

/// @ref ApplySource's value for a field the block does not take.
inline constexpr std::uint32_t kNoSource = ~0u;

/// @brief Where one block of a decoded P-frame comes from: laid out as the
///        apply kernel reads it, two u32.
struct ApplySource {
  /// Its reference entry, for keep and skip; else @ref kNoSource.
  std::uint32_t reference = kNoSource;
  /// Its stream position, for keep, zero and a new block; else
  /// @ref kNoSource.
  std::uint32_t payload = kNoSource;
};
static_assert(sizeof(ApplySource) == 8 &&
                  offsetof(ApplySource, reference) == 0 &&
                  offsetof(ApplySource, payload) == 4,
              "ApplySource is two u32, as the apply kernel reads it");

/// @brief A decoded P-frame's blocks in the frame's order.
struct MergedFrame {
  /// The frame's N coordinates, strictly increasing: the grid's block set,
  /// and the next reference's coordinates.
  std::vector<Vec3i> coords;
  /// Each block's sources, entry for entry with `coords`.
  std::vector<ApplySource> sources;
};

/// @brief Merge a P-frame's surviving reference entries with its new blocks
///        (@ref merge_by_coord): keep gives {e, e}, skip {e, none}, zero
///        {none, e}, new block q {none, M + q}, and remove nothing.
/// @param reference  The reference's M coordinates, strictly increasing.
/// @param modes      The P-frame's M modes.
/// @param added      Its A new coordinates, strictly increasing.
/// @return The merged blocks; `Status::Code::InvalidArgument` for a new
///         coordinate that is a reference coordinate, whatever that entry's
///         mode, or a mode count other than the reference's; or
///         `Status::Code::Unsupported` for a motion mode.
VR_CODEC_API core::Result<MergedFrame> merge_frame(
    const std::vector<Vec3i>& reference, const std::vector<BlockMode>& modes,
    const std::vector<Vec3i>& added);

}  // namespace volumetric_kit::recon::codec::detail
