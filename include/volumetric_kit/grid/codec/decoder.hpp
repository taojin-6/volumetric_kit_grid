// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file codec/decoder.hpp
/// @brief Decode a frame -- intra, or a P-frame against the frame decoded
///        before it -- into a @ref volume::VoxelBlockGrid, smoothed across its
///        block faces on request, and read a frame's header without decoding
///        it.

#include <cstddef>
#include <cstdint>
#include <memory>

#include "volumetric_kit/core/base/result.hpp"
#include "volumetric_kit/core/base/stage_metrics.hpp"
#include "volumetric_kit/core/vulkan/command_batch.hpp"
#include "volumetric_kit/core/vulkan/fwd.hpp"
#include "volumetric_kit/core/vulkan/gpu_timer.hpp"
#include "volumetric_kit/core/vulkan/host_ring.hpp"
#include "volumetric_kit/recon/codec/codec_params.hpp"
#include "volumetric_kit/recon/codec/export.hpp"
#include "volumetric_kit/recon/volume/voxel_block_grid.hpp"

namespace volumetric_kit::recon::codec {

namespace detail {
class BlockLists;
class DctTransform;
class Deblock;
class DeviceFrameReader;
class FrameGate;
class ReferenceStore;
struct RecordedFrame;
}  // namespace detail

/// @brief What a frame's header says: enough to build a grid it decodes into.
struct FrameInfo {
  /// Metres per voxel edge. A grid decoding the frame must match it exactly.
  float voxel_size = 0.0f;
  /// Metres. A grid decoding the frame must match it exactly: the
  /// coefficients are fractions of it.
  float trunc_dist = 0.0f;
  /// Blocks the decoded frame holds (a P-frame's reference blocks that
  /// survive, and its new ones); a grid decoding it needs at least this many
  /// (`num_blocks`), plus the hash-table room to place them.
  std::uint32_t block_count = 0;
  /// The coefficient count, scale, and kept bases' weights it was coded with;
  /// weights beyond the K cutoff are not carried and read as 1.
  CodecParams params;
  /// Intra, or a P-frame, which decodes only after the frame it was coded
  /// against (@ref Decoder).
  FrameType type = FrameType::kIntra;
};

/// @brief What the deblock did to the frame a @ref Decoder last decoded or
///        rebuilt (@ref Decoder::set_deblock): enough for a player to show,
///        say, "LF, 12 of 1,920 blocks exact".
struct DeblockInfo {
  /// The filter the frame's steps chose and the decoder ran on its fully
  /// observed blocks: `"LF"`, a line fit across each block face; `"LG"`,
  /// that and a light Gaussian; `"R"`, that and a stronger one; or `"off"`,
  /// for the deblock off or a step too fine to need one. A string literal.
  const char* filter = "off";
  /// T, in metres: the largest change one quantum of the frame's
  /// lowest-frequency coefficients makes to a voxel, by which the filter is
  /// chosen and its changes are clipped. The frame's, the deblock on or off.
  float step = 0.0f;
  /// The frame's blocks.
  std::uint32_t block_count = 0;
  /// Of them, those the per-block fallback kept at their exact decode: with
  /// a filter, the blocks holding an unobserved voxel, which it never writes
  /// (@ref Decoder::set_deblock), counted on the device; with `"off"`,
  /// every block. A fully observed block the filter happens to leave
  /// unchanged is not counted. A frame with them all decodes exactly as with
  /// the deblock off.
  std::uint32_t exact_blocks = 0;
};

/// @brief Read a frame's complete header, including its quantization weights.
///
/// What a player calls on its first frame to build a grid of the geometry the
/// stream carries (`voxel_size`, `trunc_dist`, `block_size` 8, and room for
/// `block_count` blocks), before it can @ref Decoder::decode anything;
/// @ref make_decode_grid builds that grid. No entropy payload is read.
/// @param data  The frame (at least its complete header: 44 bytes plus four
///              per kept coefficient).
/// @param size  Bytes available at @p data.
/// @return The header, a P-frame's included, or
///         `Status::Code::Unsupported` for another version, a frame type
///         past @ref FrameType::kInter or another block size, and
///         `Status::Code::InvalidArgument` for anything else malformed. It
///         does not check the rest of the frame, which @ref Decoder::decode
///         does.
VR_CODEC_API core::Result<FrameInfo> read_frame_info(const std::uint8_t* data,
                                                     std::size_t size);

/// @brief A frame recorded into a caller's batch (@ref Decoder::record), for
///        @ref Decoder::finish or @ref Decoder::abandon once that batch has
///        resolved.
///
/// A token: the decoder keeps what finishing needs, and its next record
/// supersedes an unfinished one. Its @ref gate lives in the decoder's own
/// heap-held buffers, so it survives a move of the decoder.
///
/// @code
/// VKC_ASSIGN(core::HostSlot slot, decoder.claim_input(record.size));
/// VKC_TRY(reader.read(record, slot.data(), slot.capacity()));
/// VKC_ASSIGN(const codec::PendingDecode frame,
///            decoder.record(batch, slot, record.size, grid));
/// @endcode
struct PendingDecode {
  /// Which of the decoder's records this is; 0 for none.
  std::uint64_t serial = 0;
  /// N: the blocks the grid holds once the frame is in, which an extract
  /// recorded after it plans for.
  std::uint32_t blocks = 0;
  /// 1 once every step succeeded on the device, else 0 (bound 1): what an
  /// extract recorded after it gates on.
  volume::DeviceCount gate;
  /// The slot the record was parsed from (`core::HostSlot`), held so the
  /// frame can be decoded again (@ref finish_growing).
  core::HostSlot input;
  /// The record's bytes at the start of @ref input.
  std::size_t size = 0;
};

/// @brief Decodes frames into a caller's @ref volume::VoxelBlockGrid -- the
///        codec's playback side (the 2026-09-26 decision).
///
/// After a successful @ref decode the grid holds **exactly** the frame's
/// blocks, whatever it held before: blocks the frame lacks are removed (and
/// their attributes zeroed), missing ones are allocated, and every frame
/// block's `tsdf` and `weight` are rewritten in full -- an observed voxel with
/// its decoded SDF (smoothed where the deblock runs) and weight 1.0, an
/// unobserved one as a fresh block holds it. So a player decodes frame after
/// frame into one grid and meshes it after each; nothing is cleared between
/// frames, and a block present in both costs no allocation. A block the decode
/// leaves different is stamped `changed` (@ref volume::BlockStamp), and with
/// the deblock off one it leaves as it was is not, so a consumer of the
/// stamps sees only what the stream changed.
///
/// The deblock (@ref set_deblock, off by default) smooths each frame it runs
/// on across its block faces, where quantization leaves steps the mesher
/// would draw, by a filter the frame's header chooses (@ref DeblockInfo), on
/// the device inside the decode's batch. It is out of loop -- a P-frame
/// decodes from its reference's coefficients, never from a grid -- so it may
/// change between any two frames, and it costs no bits. It writes the `tsdf`
/// of fully observed blocks alone -- a block holding an unobserved voxel keeps
/// its exact decode -- and stamps a block it writes `changed`. With it on,
/// the stamps can also name a block whose values end as they were -- the
/// inverse compares the exact decode with the smoothed values the grid held
/// -- but never miss one that changed.
///
/// Only `tsdf` and `weight` are written, so a grid that declares any other
/// attribute is refused: a block kept across frames would carry that
/// attribute's previous values (a `color`, say) under the new geometry, and
/// nothing could tell which blocks those were.
///
/// A P-frame (@ref FrameType::kInter) decodes against the last frame this
/// decoder decoded, its reference, which the decoder keeps on the device as
/// it decoded it -- quantized coefficients, masks and block list, never
/// rebuilt from a grid -- so it decodes into any grid of its geometry,
/// a fresh one included, to exactly the blocks an intra frame of the same
/// content decodes to. Every decode that succeeds, intra or P, makes its
/// frame the reference; one refused or failed keeps the reference it had, so
/// the same frame decoded again, after a resize say, decodes against it.
/// The reference and the frame being decoded take `80 + 4 * ceil(K / 2)`
/// bytes of device memory a block each, and the frame 24 more (its sources,
/// and the blocks the grid lacks); the rANS decode takes
/// `64 + 4 * ceil(K / 2)` an entry it decodes (a P-frame's reference entries
/// and its new blocks), 16 more a new block and 8 a reference entry, and
/// 16 KiB a frequency table for its slot lookup, beside the frame's payload
/// and tables; the blocks the grid must drop take 16 bytes a hash slot; and
/// with the deblock on, a frame it filters takes 4 bytes a block slot of the
/// grid's heap, and one whose filter has a Gaussian 1,184 bytes a block
/// more, each grown with 1.5x headroom. All of it is kept, sized for the
/// largest frame decoded. The record itself is read in place from one of two
/// input slots (@ref claim_input), and its results land in one of two
/// results slots, each a `core::HostRing` slot.
///
/// Everything but I/O and control runs on the device: the rANS decode, one
/// segment an invocation; the reader's verdict; the merge of a P-frame's
/// survivors with its new blocks; and the diff of the grid's blocks against
/// the frame's. The host parses the header and tables; every verdict and
/// count stays on the device, where a one-thread gate judges rows 11 to 13
/// before the grid changes, and every later grid write in the frame's batch,
/// the deblock's included, is gated on it. A frame the device refuses changes
/// neither the grid's blocks nor the reference, but moves the map's
/// `topology_epoch` and tick. A frame the device cannot hold is refused,
/// never decoded on the host.
///
/// @warning The `Device` and `Allocator` passed to @ref create must
///          outlive this object. Not thread-safe: one call at a time, and
///          nothing else may modify the grid during one.
class VR_CODEC_API Decoder {
 public:
  /// @brief Build the kernels: the rANS decoder's, the block lists' (the
  ///        merge's and the grid diff's among them), the gate's, the apply's,
  ///        the inverse transform's and the deblock's; and make the input and
  ///        results rings, two slots each.
  /// @param device     The compute device (must outlive this object).
  /// @param allocator  The allocator per-call buffers come from (must outlive
  ///                   this object).
  /// @return The decoder, or a pipeline or allocation failure: a device
  ///         whose kernels do not build has no decoder.
  static core::Result<Decoder> create(core::Device& device,
                                      core::Allocator& allocator);

  ~Decoder();
  Decoder(Decoder&& other) noexcept;
  Decoder& operator=(Decoder&& other) noexcept;
  Decoder(const Decoder&) = delete;
  Decoder& operator=(const Decoder&) = delete;

  /// @brief Decode one frame into @p grid: an intra frame, or a P-frame
  ///        against the reference this decoder holds.
  ///
  /// Everything that can be checked before the grid is touched is: the grid's
  /// attributes, its geometry against the frame's header, the whole frame
  /// parsed and entropy-decoded, a P-frame against the held reference, and
  /// the grid's heap against its block count. A refusal there leaves @p grid
  /// exactly as it was. A failure **after** the grid starts to change -- the
  /// hash table cannot place every block, its bucket locks keep losing races,
  /// a dispatch fails, or an internal fault -- leaves it valid but holding
  /// **neither** frame: blocks the two frames share still hold the previous
  /// frame's voxels, and blocks new to this one are empty, so it is not worth
  /// meshing. The next successful decode restores it exactly, since each
  /// decode starts from whatever the grid holds.
  ///
  /// It is @ref claim_input, a copy of @p data, @ref record, one submit and
  /// @ref finish.
  /// @param data     The frame.
  /// @param size     Its exact length.
  /// @param grid     The grid to decode into: 4-byte float `tsdf` and
  ///                 `weight` and no other attribute, `block_size` 8, and the
  ///                 frame's `voxel_size` and `trunc_dist` exactly (see
  ///                 @ref read_frame_info).
  /// @param metrics  Optional `StageMetrics` collecting a `"codec decode"`
  ///                 row with its host, wait and device halves -- the whole
  ///                 decode's, the deblock's included -- over the breakdown
  ///                 rows, each with all three, that tile its one submit:
  ///                 `"  ..rans decode"` (parsing the frame, decoding it,
  ///                 merging a P-frame's blocks with its reference's, and
  ///                 diffing the grid's block set against the frame's),
  ///                 `"  ..apply"` (the gate, then removing and allocating
  ///                 blocks: the zeroing, the hash deletes and the
  ///                 allocations, every round) and `"  ..inverse"` (making
  ///                 the frame's blocks from the reference and the payload,
  ///                 the transform and the deblock, then gathering the
  ///                 results). Named apart from the @ref Encoder's, so both
  ///                 timed into one `StageMetrics` stay apart. `nullptr`
  ///                 measures nothing.
  /// @return OK, or: whatever the frame reader refuses
  ///         (`Status::Code::Unsupported`,
  ///         `Status::Code::InvalidArgument` for a malformed or corrupt
  ///         frame, a segment longer than @ref kMaxDeviceDecodeSegmentSize
  ///         and a P-frame whose coding parameters are not its reference's
  ///         included); `Status::Code::NotFound` for a P-frame whose
  ///         reference this decoder does not hold (none, or another frame):
  ///         the stream's P-frames decode again from its next I-frame; a
  ///         frame the device cannot hold: `Status::Code::InvalidArgument`
  ///         for a buffer past `maxStorageBufferRange`,
  ///         `Status::Code::OutOfMemory` past free device memory, which a
  ///         resize does not recover;
  ///         `Status::Code::InvalidArgument` for a moved-from decoder, a
  ///         grid that is moved-from, has another block size or geometry,
  ///         lacks a float `tsdf` / `weight`, or declares any other
  ///         attribute, or a grid whose free heap refuses a removed block
  ///         (its accounting was already broken, which nothing here mends),
  ///         or, after the grid changed, an internal fault: a block's source
  ///         outside the reference or payload bound (a bug, never a sound
  ///         frame's);
  ///         `Status::Code::OutOfMemory` when the grid is too small for
  ///         the frame -- fewer block slots (`num_blocks`) than the frame has
  ///         blocks, or a hash table that cannot place them all -- which
  ///         @ref volume::VoxelBlockGrid::resize and decoding again recovers
  ///         (@ref decode_growing does both);
  ///         `Status::Code::IoError` only when bucket-lock contention
  ///         outlasts the device rounds
  ///         (`volume::VoxelHashMap::kRecordedRounds`) over a table with room
  ///         (decode again; a resize would not help); otherwise a buffer or
  ///         dispatch failure.
  core::Status decode(const std::uint8_t* data, std::size_t size,
                      volume::VoxelBlockGrid& grid,
                      core::StageMetrics* metrics = nullptr);

  /// @brief @ref decode the record already at the start of @p slot: @ref
  ///        record, one submit and @ref finish, with no copy -- what
  ///        @ref finish_growing decodes a frame again with.
  /// @param slot     From @ref claim_input, the record at its start.
  /// @param size     The record's exact length.
  /// @param grid     As @ref decode.
  /// @param metrics  As @ref decode.
  /// @return As @ref decode and @ref record.
  core::Status decode(const core::HostSlot& slot, std::size_t size,
                      volume::VoxelBlockGrid& grid,
                      core::StageMetrics* metrics = nullptr);

  /// @brief A slot of this decoder's input ring with room for a record of
  ///        @p bytes and the 8 bytes of slack after it: read the record into
  ///        `data()`, then @ref record it.
  ///
  /// Two persistent slots, grown as a larger record asks: one held by a
  /// recorded frame until its batch has resolved and its token is dropped,
  /// and one being filled. @ref record zeroes the slack.
  /// @param bytes  The record's exact size; not 0.
  /// @return The slot; `Status::Code::InvalidArgument` for a moved-from
  ///         decoder, 0 bytes, or both slots held; or an allocation failure.
  core::Result<core::HostSlot> claim_input(std::size_t bytes);

  /// @brief Record decoding the record at the start of @p slot into @p grid,
  ///        into @p batch: the whole decode, every count on the device,
  ///        nothing read back.
  ///
  /// The host parses the record in place and checks what @ref decode checks
  /// before the grid changes: the grid, the header, sections and tables, a
  /// P-frame's reference, and N against the grid's heap. Then it records:
  /// - the rANS decode, the merge and the diff (`"  ..rans decode"`);
  /// - the gate, and the removal and allocation rounds, which fold their
  ///   outcome into @ref PendingDecode::gate (`"  ..apply"`);
  /// - the apply, the inverse, the deblock as @ref set_deblock stands now
  ///   (each sized by N and gated), then the results @ref finish reads
  ///   (`"  ..inverse"`).
  /// @param batch    Records the decode; the caller submits it.
  /// @param slot     From @ref claim_input, the record at its start; unchanged
  ///                 until the batch has resolved.
  /// @param size     The record's exact length.
  /// @param grid     As @ref decode takes it; nothing but @p batch changes it
  ///                 until the batch has resolved.
  /// @param metrics  As @ref decode's rows: host halves now, device halves
  ///                 from `PendingBatch::device_rows`.
  /// @return The pending frame; or, with nothing recorded:
  ///         - what the frame reader refuses (`Status::Code::Unsupported`;
  ///           `Status::Code::InvalidArgument` for a malformed frame);
  ///         - `Status::Code::NotFound` for a P-frame whose reference this
  ///           decoder does not hold;
  ///         - `Status::Code::OutOfMemory` for more blocks than the grid's
  ///           heap (@ref record_growing grows it first);
  ///         - `Status::Code::InvalidArgument` for a moved-from decoder, a
  ///           slot shorter than @p size and its slack, or a grid
  ///           @ref decode refuses.
  ///
  ///         Or, possibly after the first commands, so the caller drops
  ///         @p batch: `Status::Code::OutOfMemory` past free device memory,
  ///         or another buffer failure.
  /// @pre The batch of this decoder's last record has resolved.
  core::Result<PendingDecode> record(core::CommandBatch& batch,
                                     const core::HostSlot& slot,
                                     std::size_t size,
                                     volume::VoxelBlockGrid& grid,
                                     core::StageMetrics* metrics = nullptr);

  /// @brief Once @p pending's batch has resolved: judge it and, for a frame
  ///        that succeeded, make it the reference.
  ///
  /// Settles the grid's map from the rounds, reads the results (the verdict,
  /// row 13's collision, the apply's faults, the inverse's rejects, the
  /// blocks the deblock kept), then sets @ref last_deblock and
  /// @ref last_rounds.
  /// @param pending  From this decoder's last @ref record.
  /// @param grid     The grid it recorded into.
  /// @return OK; or, the reference unchanged, the statuses @ref decode
  ///         returns once a frame is parsed (rows 11 to 13's
  ///         `Status::Code::InvalidArgument` with the grid's blocks untouched,
  ///         a heap that refused a removed block, `Status::Code::OutOfMemory`
  ///         for a table that cannot place the frame, `Status::Code::IoError`
  ///         for lost races past the rounds, the apply's and the inverse's
  ///         faults); or `Status::Code::InvalidArgument` for a pending this
  ///         decoder no longer holds, or another grid.
  core::Status finish(const PendingDecode& pending,
                      volume::VoxelBlockGrid& grid);

  /// @brief Drop @p pending without judging it: its batch failed, was never
  ///        submitted, or ran a frame no longer wanted. The reference stays;
  ///        the grid holds what the batch left, and the map's heap copy stays
  ///        as it was, until the next decode.
  /// @param pending  From this decoder; one it no longer holds, or another
  ///                 decoder's, is ignored.
  void abandon(const PendingDecode& pending) noexcept;

  /// @return The most rounds the removal or the allocation ran in the frame
  ///         the last @ref finish judged (`AllocFailures::rounds`), for M2's
  ///         histogram; 0 before one.
  std::uint32_t last_rounds() const noexcept { return last_rounds_; }

  /// @brief Turn the deblock on or off for every frame recorded or rebuilt
  ///        from here on (@ref record and @ref rebuild read it): on, a frame
  ///        is smoothed across its block faces by the filter its header picks
  ///        (none at a fine step), after its inverse transform, in the same
  ///        batch; off, the default, the grid holds the exact decode.
  ///
  /// The per-block fallback: a block holding an unobserved voxel -- as
  /// nearly every block of a fused RGB-D frame does, and a true-SDF frame's
  /// few blocks with cut voxels (@ref EncoderConfig::true_sdf) -- keeps its
  /// exact decode. No pass writes it and no line is fitted across a face it
  /// shares, so neither side of that face is line-fitted; the Gaussian of a
  /// neighbour reads its observed voxels as decoded, its unobserved ones not
  /// at all. Only fully observed blocks are smoothed, so a frame all of whose
  /// blocks are masked decodes exactly as with the deblock off;
  /// @ref DeblockInfo::exact_blocks counts the blocks kept.
  void set_deblock(bool on) noexcept { deblock_on_ = on; }
  /// @return Whether the deblock is on.
  bool deblock() const noexcept { return deblock_on_; }

  /// @brief Rebuild the frame this decoder last decoded into @p grid, from
  ///        the reference it holds, with the deblock as it is now: what a
  ///        paused player calls after @ref set_deblock to show the change on
  ///        the frame it shows, without reading the frame again.
  ///
  /// One batch, submitted and waited on: the diff of the grid's blocks
  /// against the frame's, then the inverse transform and the deblock as the
  /// decode records them, gated on that diff and stamping as they do, and
  /// the results gathered into a results slot. The reference is unchanged.
  /// @param grid     The grid the last decode wrote: the frame's geometry,
  ///                 and exactly its blocks, whatever their values.
  /// @param metrics  Optional; a `"codec decode"` row with its host, wait and
  ///                 device halves.
  /// @return OK; `Status::Code::NotFound` before a decode has succeeded;
  ///         `Status::Code::InvalidArgument` for a moved-from decoder, a
  ///         grid @ref decode refuses before it changes one, or one whose
  ///         blocks are not exactly the frame's, found on the device before
  ///         anything is written (decode the frame into it instead);
  ///         otherwise a buffer or dispatch failure, after which the grid
  ///         holds a partial rebuild until a decode or rebuild succeeds.
  /// @pre The batch of this decoder's last @ref record has resolved.
  core::Status rebuild(volume::VoxelBlockGrid& grid,
                       core::StageMetrics* metrics = nullptr);

  /// @return What the deblock did to the frame the last @ref finish or
  ///         @ref rebuild that succeeded left; `"off"` and zeros before one.
  const DeblockInfo& last_deblock() const noexcept { return last_deblock_; }

  /// @return `true` if this owns a live transform (`false` when moved-from).
  bool valid() const noexcept;

 private:
  Decoder();

  // The body of record and decode, under the "codec decode" row their
  // caller opened.
  core::Result<PendingDecode> record_frame(core::CommandBatch& batch,
                                           const core::HostSlot& slot,
                                           std::size_t size,
                                           volume::VoxelBlockGrid& grid,
                                           core::StageMetrics* metrics);
  // Whether @p pending is this decoder's unfinished last record.
  bool holds(const PendingDecode& pending) const noexcept;

  // Borrowed (must outlive this); what the decode's batches run on.
  core::Device* device_ = nullptr;
  core::Allocator* allocator_ = nullptr;
  std::unique_ptr<detail::DctTransform> transform_;
  std::unique_ptr<detail::DeviceFrameReader> reader_;
  // The reference a P-frame decodes against, and the apply into it.
  std::unique_ptr<detail::ReferenceStore> store_;
  // The frame's blocks merged into frame order, and diffed against the grid.
  std::unique_ptr<detail::BlockLists> lists_;
  // The deblock's kernels and snapshot, and its switch.
  std::unique_ptr<detail::Deblock> deblock_;
  // The gate, its slices and the results gather.
  std::unique_ptr<detail::FrameGate> gate_;
  // The records' input slots and the frames' result slots, two each.
  std::unique_ptr<core::HostRing> inputs_;
  std::unique_ptr<core::HostRing> results_;
  // The frame recorded and not yet finished or abandoned, and the serial
  // the next record takes.
  std::unique_ptr<detail::RecordedFrame> pending_;
  std::uint64_t serial_ = 0;
  bool deblock_on_ = false;
  DeblockInfo last_deblock_;
  std::uint32_t last_rounds_ = 0;
  // Device spans for the transform; idle until a caller asks for metrics.
  core::GpuTimer gpu_timer_;
};

}  // namespace volumetric_kit::recon::codec
