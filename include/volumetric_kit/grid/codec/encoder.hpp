// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file codec/encoder.hpp
/// @brief Encode a grid's geometry, frame after frame, as intra frames or
///        P-frames against the frame encoded before.

#include <cstdint>
#include <memory>
#include <vector>

#include "volumetric_kit/core/base/result.hpp"
#include "volumetric_kit/core/base/stage_metrics.hpp"
#include "volumetric_kit/core/vulkan/gpu_timer.hpp"
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
}  // namespace detail

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
  /// A P-frame's reference entries per independently coded update segment,
  /// from 1 to @ref kMaxDeviceDecodeSegmentSize: as @ref segment_size is for
  /// its new blocks, smaller segments give the device more parallelism, at a
  /// little more size (each segment costs a 4-byte length and its stream's
  /// 4-byte final state); 256 would save about 4% of a stream at a period of
  /// 30 and code its P-frames 20-140% slower (the 2026-10-16 decision). An
  /// I-frame has no update segments.
  std::uint32_t update_size = 64;
  /// Frames from one I-frame to the next, at least 1. At 1, the default,
  /// every frame is an I-frame; at N, an I-frame and N - 1 P-frames
  /// (@ref FrameType::kInter, experimental) take turns; a period past the
  /// stream's length codes P-frames after the first frame for as long as
  /// the rules on @ref Encoder::encode allow.
  std::uint32_t intra_period = 1;

  /// @brief Check every field against what the @ref Encoder takes: what
  ///        @ref Encoder::create checks first, and what a caller checks
  ///        before any GPU work.
  /// @return OK, or `Status::invalid_argument` naming the field: invalid
  ///         @ref params (@ref CodecParams::validate), a @ref segment_size or
  ///         @ref update_size of 0 or past @ref kMaxDeviceDecodeSegmentSize,
  ///         or an @ref intra_period of 0.
  core::Status validate() const;
};

/// @brief Encodes a @ref volume::VoxelBlockGrid's geometry as frames -- the
///        codec's capture side (the 2026-09-26 decision).
///
/// A frame holds every active block that has at least one observed voxel
/// (`weight >= volume::kObservedWeight`): its coordinate, a 1-bit-per-voxel
/// observed mask, and the first K coefficients of its 8^3 DCT, entropy-coded
/// (the 2026-09-27 format). A block with **no** observed voxel is left out --
/// the mesher uses nothing from it, and a fused grid allocates a whole
/// truncation band, much of which is never integrated -- so a decoded grid can
/// hold fewer blocks than the one encoded, and the same surface.
///
/// The blocks are coded in (z, y, x) order, whatever order the hash table
/// holds them in, so the same content always makes the same bytes on the same
/// device (a frame is reproducible per device, decodable on any): at an
/// @ref EncoderConfig::intra_period of 1 any frame, and above it the same
/// chain of grids, from a fresh encoder or after the same I-frame.
///
/// A P-frame (@ref FrameType::kInter, experimental) codes a frame against
/// the last one @ref encode returned, its reference, which the encoder keeps
/// on the device as it coded it -- quantized coefficients, masks and block
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
/// sized for the largest frame.
///
/// Geometry only: colour is not coded. A player textures the decoded mesh
/// from RGB that travels beside the frame.
///
/// The rANS coding runs on the device, one segment an invocation: the
/// coefficients and masks stay in device memory, and only the symbol counts
/// and the frame cross the bus. A frame the device cannot hold is refused,
/// never coded on the host.
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

  /// @brief Encode every observed block of @p grid as one frame: an
  ///        I-frame, or a P-frame against the last frame this returned.
  ///
  /// The frame is an I-frame when it is the first, when
  /// @ref EncoderConfig::intra_period frames have been coded since the last
  /// I-frame, after @ref force_intra, when @p grid's `voxel_size` or
  /// `trunc_dist` is not the reference's, when the reference holds no block,
  /// or when @p grid has no observed block (an I-frame is then the smaller);
  /// otherwise a P-frame. A frame returned becomes the reference and counts
  /// toward the period; a refusal changes neither.
  /// @param grid     A grid with 4-byte float `tsdf` and `weight` attributes
  ///                 and `block_size` 8. Its blocks and voxels are read,
  ///                 never written; the reference is non-const only because
  ///                 the active set is the map's own list
  ///                 (@ref
  ///                 volume::VoxelHashMap::compact_active_blocks_on_device),
  ///                 which an extract after this takes back from it.
  /// @param metrics  Optional `StageMetrics` collecting a `"codec encode"`
  ///                 row with both halves -- its device half is the transform
  ///                 and the rANS kernels -- over the breakdown rows
  ///                 `"  ..active set"` (the compaction, when the map holds
  ///                 no list that is still its active set), `"  ..observed"`
  ///                 (finding the blocks with an observed voxel),
  ///                 `"  ..forward"` (sorting them, the transform, a
  ///                 P-frame's diff against the reference, and the symbol
  ///                 count: one batch, and a second for a P-frame's count)
  ///                 and `"  ..rans encode"` (coding and laying out the
  ///                 frame).
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
      volume::VoxelBlockGrid& grid, core::StageMetrics* metrics = nullptr);

  /// @brief Make the next frame @ref encode returns an I-frame: what a
  ///        caller does when a frame it was returned did not reach the
  ///        decoder, so the P-frames after it would be refused there
  ///        (`Status::Code::NotFound`).
  void force_intra() noexcept;

  /// @return `true` if this owns a live transform (`false` when moved-from).
  bool valid() const noexcept;

 private:
  Encoder();

  EncoderConfig config_;
  // Borrowed (must outlive this); what the coding's batches run on.
  core::Device* device_ = nullptr;
  core::Allocator* allocator_ = nullptr;
  std::unique_ptr<detail::DctTransform> transform_;
  std::unique_ptr<detail::BlockLists> lists_;
  std::unique_ptr<detail::DeviceFrameWriter> writer_;
  // The reference a P-frame is coded against, and the diff against it.
  std::unique_ptr<detail::ReferenceStore> store_;
  // Device spans for the kernels; idle until a caller asks for metrics.
  core::GpuTimer gpu_timer_;
  // Frames coded since the last I-frame, below config_.intra_period: the
  // next frame is an I-frame at 0.
  std::uint32_t gop_position_ = 0;
  bool force_intra_ = false;
};

}  // namespace volumetric_kit::recon::codec
