// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file codec/encoder.hpp
/// @brief Encode a grid's geometry, frame after frame, as intra frames or
///        P-frames against the frame encoded before.

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "volumetric_kit/core/base/result.hpp"
#include "volumetric_kit/core/base/stage_metrics.hpp"
#include "volumetric_kit/core/vulkan/buffer.hpp"
#include "volumetric_kit/core/vulkan/command_batch.hpp"
#include "volumetric_kit/core/vulkan/gpu_timer.hpp"
#include "volumetric_kit/core/vulkan/host_ring.hpp"
#include "volumetric_kit/recon/codec/codec_params.hpp"
#include "volumetric_kit/recon/codec/export.hpp"
#include "volumetric_kit/recon/core/fwd.hpp"
#include "volumetric_kit/recon/volume/voxel_block_grid.hpp"

namespace volumetric_kit::recon::codec {

namespace detail {
class BlockLists;
class DctTransform;
class DeviceFrameWriter;
class ReferenceStore;
struct RecordedEncode;
}  // namespace detail

/// The band, in voxels from the zero level, that holds every voxel an
/// @ref EncoderConfig::true_sdf frame codes: `ceil(1 + 8 sqrt(3))`.
inline constexpr std::uint32_t kTrueSdfBandVoxels = 15;

/// @brief How an @ref Encoder codes a frame.
struct VR_CODEC_API EncoderConfig {
  /// Coefficients kept per block and their quantization steps.
  CodecParams params;
  /// Blocks per independently coded segment, from 1 to
  /// @ref kMaxDeviceDecodeSegmentSize. The device codes and decodes one
  /// segment an invocation, so smaller segments give it more parallelism, at
  /// a little more size (each segment restates its first coordinate); the
  /// default costs 2-3% of a frame.
  std::uint32_t segment_size = 64;
  /// A P-frame's reference entries per independently coded update segment:
  /// 0, the default, takes @ref segment_size, which gives a P-frame at least
  /// as many segments as its reference has coded intra, at any segment size
  /// (the 2026-10-17 decision); otherwise 1 to
  /// @ref kMaxDeviceDecodeSegmentSize. As @ref segment_size is for its new
  /// blocks, smaller segments give the device more parallelism, at a little
  /// more size (each segment costs a 4-byte length and its stream's 4-byte
  /// final state); at a @ref segment_size of 64, 256 against 64 would save
  /// about 4% of a stream at a period of 30 and code its P-frames 20-140%
  /// slower (the 2026-10-16 decision). An I-frame has no update segments.
  std::uint32_t update_size = 0;
  /// Frames from one I-frame to the next, at least 1. At 1, the default,
  /// every frame is an I-frame; at N, an I-frame and N - 1 P-frames
  /// (@ref FrameType::kInter) take turns; a period past the stream's length
  /// codes P-frames after the first frame for as long as the rules on
  /// @ref Encoder::encode allow.
  std::uint32_t intra_period = 1;
  /// Whether the grid holds a true (Euclidean) signed distance, as
  /// @ref voxelize::MeshIntegrator converts a mesh to. `false`, the default,
  /// codes every block with an observed voxel under its observed mask, as
  /// fused depth needs. `true` codes, without writing the grid:
  /// - only the blocks holding a corner of a cell whose 8 voxels, across
  ///   block faces too, are observed and not cut, and whose sign changes or
  ///   that has a corner within one voxel of zero;
  /// - each of their observed voxels as observed, but the cut: every corner
  ///   farther than `sqrt(3)` voxels from zero of a cell whose 8 voxels are
  ///   observed and whose sign changes.
  ///
  /// Every voxel of those blocks lies within @ref kTrueSdfBandVoxels voxels
  /// of the zero level, so a grid observed that far out codes every mask as
  /// all ones but the cuts, and every voxel of a coded block but the cuts
  /// decodes with weight 1; a narrower band leaves the voxels past it
  /// unobserved.
  /// @warning Only for a true distance: fused depth has crossings with
  ///          corners past `sqrt(3)` voxels, which the cut would erase.
  bool true_sdf = false;

  /// @brief The update segment size a P-frame of this config codes with.
  /// @return @ref update_size, or @ref segment_size when @ref update_size is
  ///         0: what @ref validate checks and @ref Encoder::create keeps.
  constexpr std::uint32_t resolved_update_size() const noexcept {
    return update_size != 0 ? update_size : segment_size;
  }

  /// @brief Check every field against what the @ref Encoder takes: what
  ///        @ref Encoder::create checks first, and what a caller checks
  ///        before any GPU work.
  /// @return OK, or `Status::invalid_argument` naming the field: invalid
  ///         @ref params (@ref CodecParams::validate), a @ref segment_size of
  ///         0, a @ref segment_size or nonzero @ref update_size past
  ///         @ref kMaxDeviceDecodeSegmentSize, or an @ref intra_period of 0.
  core::Status validate() const;
};

/// @brief A frame whose front half was recorded into a caller's batch
///        (@ref Encoder::record_front): for @ref Encoder::record_back,
///        @ref Encoder::finish or @ref Encoder::abandon.
///
/// A token: the encoder keeps what the halves need, and its next
/// @ref Encoder::record_front supersedes an unfinished one.
struct PendingEncode {
  /// Which record this is, unique in the process; 0 for none.
  std::uint64_t serial = 0;
  /// What the frame is coded as, decided at record by @ref Encoder::encode's
  /// rules; a P-frame whose grid turns out to hold no block to code finishes
  /// as an I-frame of none.
  FrameType type = FrameType::kIntra;
};

/// @brief A frame @ref Encoder::finish returned: its bytes are @ref head then
///        the payload, back to back, the payload where the device wrote it in
///        one of the encoder's slots, held until this is dropped.
///
/// @code
/// VKC_ASSIGN(const codec::EncodedFrame frame, encoder.finish(pending));
/// const stream::RecordPayload bytes = stream::RecordPayload::of(
///     frame.head.data(), frame.head.size(), frame.payload,
///     frame.payload_size);
/// @endcode
struct EncodedFrame {
  FrameType type = FrameType::kIntra;  ///< An I-frame or a P-frame.
  std::uint32_t blocks = 0;            ///< The blocks it carries.
  /// The header, the section table and every section but PAYLOAD's body,
  /// which comes last.
  std::vector<std::uint8_t> head;
  /// The PAYLOAD section's body, @ref payload_size bytes in @ref slot; null
  /// for a frame of no blocks.
  const std::uint8_t* payload = nullptr;
  std::size_t payload_size = 0;  ///< Its bytes.
  core::HostSlot slot;           ///< Holds @ref payload.
  /// @return The frame's bytes: the head's and the payload's.
  std::size_t size() const noexcept { return head.size() + payload_size; }
};

/// @brief Encodes a @ref volume::VoxelBlockGrid's geometry as frames -- the
///        codec's capture side (the 2026-09-26 decision).
///
/// A frame holds every active block that has at least one observed voxel
/// (`weight >= volume::kObservedWeight`), or with @ref EncoderConfig::true_sdf
/// every block that holds surface: its coordinate, a 1-bit-per-voxel observed
/// mask, and the first K coefficients of its 8^3 DCT, entropy-coded (the
/// 2026-09-27 format). A block with **no** observed voxel is left out -- the
/// mesher uses nothing from it, and a fused grid allocates a whole truncation
/// band, much of which is never integrated -- so a decoded grid can hold
/// fewer blocks than the one encoded, and the same surface (with
/// @ref EncoderConfig::true_sdf, the surface less its cuts).
///
/// The blocks are coded in (z, y, x) order, whatever order the hash table
/// holds them in, so the same content always makes the same bytes on the same
/// device (a frame is reproducible per device, decodable on any): at an
/// @ref EncoderConfig::intra_period of 1 any frame, and above it the same
/// chain of grids, from a fresh encoder or after the same I-frame.
///
/// A P-frame (@ref FrameType::kInter) codes a frame against the last one
/// @ref encode returned, its reference, which the encoder keeps on the
/// device as it coded it -- quantized coefficients, masks and block
/// list -- so the reference is the one a @ref Decoder holds after decoding
/// that frame: each block is skipped when unchanged, kept as residuals
/// against the reference, recoded in full, or removed, and blocks the
/// reference lacks follow as in an intra frame. A frame decodes to the same
/// grid whichever type it is. Only a frame @ref encode returns becomes the
/// reference, so a refusal keeps the chain, and a frame returned and then
/// lost -- never decoded, or decoded out of order -- breaks it until the
/// next I-frame: call @ref force_intra. The reference and the frame being
/// encoded take `80 + 4 * ceil(K / 2)` bytes of device memory a block each,
/// and the frame 8 more; a P-frame's entries `80 + 4 * ceil(K / 2)` a block
/// of the frame and `72 + 4 * ceil(K / 2)` a reference block, all kept,
/// sized for the largest frame; @ref EncoderConfig::true_sdf 68 bytes more
/// a block slot of the grid's heap, kept too.
///
/// Geometry only: colour is not coded. A player textures the decoded mesh
/// from RGB that travels beside the frame.
///
/// The rANS coding runs on the device, one segment an invocation: the
/// coefficients and masks stay in device memory, and only the symbol counts
/// and the frame cross the bus. A frame the device cannot hold is refused,
/// never coded on the host.
///
/// A frame is two batches, the caller's: the front half
/// (@ref record_front), recorded after what the batch holds -- a sender's
/// fusion -- with every count on the device, and the back half
/// (@ref record_back), the rANS coding, sized on the host from the front's
/// counts. @ref finish lays the frame out. @ref encode is those, a submit
/// and a wait each.
///
/// @code
/// VKC_ASSIGN(const codec::PendingEncode pending,
///            encoder.record_front(front, grid));
/// VKC_TRY(front.submit());
/// core::CommandBatch back(device, allocator);
/// VKC_TRY(encoder.record_back(back, pending));
/// VKC_TRY(back.submit());
/// VKC_ASSIGN(const codec::EncodedFrame frame, encoder.finish(pending));
/// @endcode
///
/// @warning The `Device` and `Allocator` passed to @ref create must
///          outlive this object. Not thread-safe: one call at a time.
class VR_CODEC_API Encoder {
 public:
  /// @brief Build the kernels: the forward transform's, the block lists'
  ///        (the sort's among them), the reference store's (the P-frame
  ///        diff's among them) and the rANS coder's.
  /// @param device     The compute device (must outlive this object).
  /// @param allocator  The allocator per-call buffers come from (must outlive
  ///                   this object).
  /// @param config     How frames are coded.
  /// @return The encoder, holding no reference, or
  ///         `Status::Code::InvalidArgument` for a @p config that
  ///         @ref EncoderConfig::validate refuses; otherwise a pipeline or
  ///         allocation failure: a device whose kernels do not build has no
  ///         encoder.
  static core::Result<Encoder> create(core::Device& device,
                                      core::Allocator& allocator,
                                      const EncoderConfig& config = {});

  ~Encoder();
  Encoder(Encoder&& other) noexcept;
  Encoder& operator=(Encoder&& other) noexcept;
  Encoder(const Encoder&) = delete;
  Encoder& operator=(const Encoder&) = delete;

  /// @brief Encode @p grid's blocks (see the class notes) as one frame: an
  ///        I-frame, or a P-frame against the last frame this returned.
  ///
  /// The frame is an I-frame when it is the first, when
  /// @ref EncoderConfig::intra_period frames have been coded since the last
  /// I-frame, after @ref force_intra, when @p grid's `voxel_size` or
  /// `trunc_dist` is not the reference's, when the reference holds no block,
  /// or when @p grid has no block to code (an I-frame is then the smaller);
  /// otherwise a P-frame. A frame returned becomes the reference and counts
  /// toward the period; a refusal changes neither.
  ///
  /// @ref record_front, a submit and its wait, @ref record_back, a submit
  /// and its wait, and @ref finish, the head and payload joined into one
  /// vector: 2 submits.
  /// @param grid     A grid with 4-byte float `tsdf` and `weight` attributes
  ///                 and `block_size` 8. Its blocks and voxels are read,
  ///                 never written. The active set is the map's own list:
  ///                 the one it holds
  ///                 (@ref volume::VoxelHashMap::held_device_list), or one
  ///                 compacted into the front's batch
  ///                 (@ref volume::VoxelHashMap::record_compaction), which an
  ///                 extract after this compacts again.
  /// @param metrics  Optional `StageMetrics` collecting a `"codec encode"`
  ///                 row with both halves over the breakdown rows, each with
  ///                 its own device half:
  ///                 `"  ..encode set"` (the compaction, when the map holds
  ///                 no list that is still its active set), `"  ..observed"`
  ///                 (finding the blocks to code, with
  ///                 @ref EncoderConfig::true_sdf's passes),
  ///                 `"  ..forward"` (sorting them, the transform, a
  ///                 P-frame's diff against the reference, and the symbol
  ///                 count: the front's batch)
  ///                 and `"  ..rans encode"` (coding and laying out the
  ///                 frame); and `"  ..tables"`, the host's normalization of
  ///                 the counts, with its host half alone.
  ///                 Named apart from the
  ///                 @ref Decoder's, so both timed into one
  ///                 `StageMetrics` stay apart. `nullptr` measures
  ///                 nothing.
  /// @return The frame's bytes (an empty grid is a valid frame of no blocks),
  ///         or `Status::Code::InvalidArgument` for a moved-from encoder,
  ///         a grid the transform refuses (moved-from, another block size,
  ///         no float `tsdf` / `weight`) or a non-finite `voxel_size`; a
  ///         frame the device cannot hold: `Status::Code::InvalidArgument`
  ///         for a buffer past `maxStorageBufferRange`,
  ///         `Status::Code::OutOfMemory` past free device memory; otherwise a
  ///         compaction, buffer, pipeline or dispatch failure.
  core::Result<std::vector<std::uint8_t>> encode(
      const volume::VoxelBlockGrid& grid,
      core::StageMetrics* metrics = nullptr);

  /// @brief Record the front half of coding @p grid into @p batch, after what
  ///        it holds -- the set's fusion -- with every count on the device:
  ///        the active set, the blocks to code and their sort, the transform,
  ///        a P-frame's diff against the reference, the symbol count and the
  ///        coder steps' layout, then the results @ref record_back reads.
  ///
  /// Decides I or P now by @ref encode's rules, all but "no block to code",
  /// which the device answers. Every step is sized for the grid's block slots
  /// and runs for the count the device found; the sort records its levels for
  /// that bound, and a level the count does not need passes its entries
  /// through. While the map's last device list still holds, it is bound with
  /// its count instead of a compaction.
  /// @param batch    Records the half; the caller submits it.
  /// @param grid     As @ref encode; nothing but @p batch changes it until the
  ///                 batch has resolved.
  /// @param metrics  Optional: `"codec encode"`'s host half now, its row and
  ///                 then `"  ..encode set"`'s, `"  ..observed"`'s and
  ///                 `"  ..forward"`'s placed as they open, so a caller
  ///                 sharing @p batch tells the encode's device rows by
  ///                 name; from `PendingBatch::device_rows` the device halves
  ///                 of those breakdowns.
  /// @return The pending frame; or, with nothing recorded, @ref encode's
  ///         refusals of the encoder and the grid, and of a frame the device
  ///         cannot hold at the grid's block slots; or, after the first
  ///         commands, so the caller drops @p batch, a buffer failure.
  /// @pre Every batch of this encoder's last frame has resolved.
  core::Result<PendingEncode> record_front(
      core::CommandBatch& batch, const volume::VoxelBlockGrid& grid,
      core::StageMetrics* metrics = nullptr);

  /// @brief Once @p pending's front batch has resolved: build the frame's
  ///        tables from the device's counts, on the host, then record the rANS
  ///        coding into @p batch -- the tables, each entry's coder steps, each
  ///        segment's stream, and their gathering into a payload slot.
  ///
  /// Records nothing for a frame of no blocks, which @ref finish makes an
  /// I-frame of none. The payload slot has room for the coding's bound.
  /// Where it is copied (`core::HostRing::in_place` false), the batch copies
  /// the payload's likely size, and @ref finish copies any rest.
  /// @param batch    Records the half; the caller submits it.
  /// @param pending  From this encoder's last @ref record_front.
  /// @param metrics  Optional: `"codec encode"`'s host half over
  ///                 `"  ..tables"`, and from `PendingBatch::device_rows` the
  ///                 device half of `"  ..rans encode"`.
  /// @return OK; with nothing recorded, the front's refusals (entries the
  ///         transform found no block for, a motion mode or a -32768
  ///         written); `Status::Code::InvalidArgument` for a pending this
  ///         encoder no longer holds or whose back is recorded, or coded
  ///         segments that could outgrow 4 GiB; or a buffer failure.
  ///         On failure, discard @p batch before retrying @p pending.
  /// @pre At most one @ref EncodedFrame of this encoder's is held.
  core::Status record_back(core::CommandBatch& batch,
                           const PendingEncode& pending,
                           core::StageMetrics* metrics = nullptr);

  /// @brief Once @p pending's back batch has resolved: check the coding,
  ///        assemble the head, and make the frame the reference.
  ///
  /// A payload that outgrew what its batch copied is gathered again over the
  /// rest, in a batch of its own that this waits on (`"  ..tail"`, rare). The
  /// reference's hash is XXH64 over the head and the payload.
  /// @param pending  From this encoder's last @ref record_front, its back
  ///                 recorded and resolved.
  /// @param metrics  Optional: `"codec encode"`'s host half, and a tail's
  ///                 `"  ..tail"`.
  /// @return The frame, now the reference; `Status::Code::IoError` if a
  ///         kernel's table refused a symbol it was counted from (a bug);
  ///         `Status::Code::InvalidArgument` for a pending this encoder no
  ///         longer holds or whose back is not recorded, or a frame the
  ///         container would refuse; or a tail's failure. A refusal keeps
  ///         the reference, and the pending frame is gone.
  core::Result<EncodedFrame> finish(const PendingEncode& pending,
                                    core::StageMetrics* metrics = nullptr);

  /// @brief Drop @p pending: its batch failed, or the grid it read changed
  ///        before the frame was finished (a fusion done again). The
  ///        reference and the period stay.
  /// @param pending  From this encoder; one it no longer holds is ignored.
  void abandon(const PendingEncode& pending) noexcept;

  /// @brief Make the next frame @ref encode returns an I-frame: what a
  ///        caller does when a frame it was returned did not reach the
  ///        decoder, so the P-frames after it would be refused there
  ///        (`Status::Code::NotFound`).
  void force_intra() noexcept;

  /// @return `true` if this owns a live transform (`false` when moved-from).
  bool valid() const noexcept;

 private:
  Encoder();

  // The halves' bodies, inside the caller's "codec encode" scope, which
  // encode() holds open across both batches and their waits.
  core::Result<PendingEncode> front(core::CommandBatch& batch,
                                    const volume::VoxelBlockGrid& grid,
                                    core::StageMetrics* metrics);
  core::Status back(core::CommandBatch& batch, const PendingEncode& pending,
                    core::StageMetrics* metrics);
  core::Result<EncodedFrame> finish_frame(
      const PendingEncode& pending, core::StageMetrics* metrics,
      std::vector<std::uint8_t>* bytes = nullptr);
  // Whether `pending` is this encoder's unfinished last record.
  bool holds(const PendingEncode& pending) const noexcept;

  EncoderConfig config_;
  // Borrowed (must outlive this); what the coding's batches run on.
  core::Device* device_ = nullptr;
  core::Allocator* allocator_ = nullptr;
  std::unique_ptr<detail::DctTransform> transform_;
  std::unique_ptr<detail::BlockLists> lists_;
  std::unique_ptr<detail::DeviceFrameWriter> writer_;
  // The reference a P-frame is coded against, and the diff against it.
  std::unique_ptr<detail::ReferenceStore> store_;
  // A held active list's count, set in the front's batch, for the passes
  // the device sizes from it.
  core::Buffer list_count_;
  // Device spans for the kernels; idle until a caller asks for metrics.
  core::GpuTimer gpu_timer_;
  // Frames coded since the last I-frame, below config_.intra_period: the
  // next frame is an I-frame at 0.
  std::uint32_t gop_position_ = 0;
  bool force_intra_ = false;
  // The transform holds the quantization steps, which never change: a
  // finished frame's front batch wrote them, so no later front records them.
  bool steps_held_ = false;
  // What the halves need of the last record (encoder.cpp).
  std::unique_ptr<detail::RecordedEncode> pending_;
};

}  // namespace volumetric_kit::recon::codec
