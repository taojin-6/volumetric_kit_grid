// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file codec/decoder.hpp
/// @brief Decode an intra frame into a @ref volume::VoxelBlockGrid, and read
///        a frame's header without decoding it.

#include <cstddef>
#include <cstdint>
#include <memory>

#include "volumetric_kit/core/base/result.hpp"
#include "volumetric_kit/core/base/stage_metrics.hpp"
#include "volumetric_kit/core/vulkan/gpu_timer.hpp"
#include "volumetric_kit/recon/codec/codec_params.hpp"
#include "volumetric_kit/recon/codec/export.hpp"
#include "volumetric_kit/recon/core/fwd.hpp"
#include "volumetric_kit/recon/volume/voxel_block_grid.hpp"

namespace volumetric_kit::recon::codec {

namespace detail {
class DctTransform;
class DeviceFrameReader;
struct ParsedFrame;
struct ResidentBlocks;
}  // namespace detail

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
  /// The coefficient count, scale, and kept bases' weights it was coded with;
  /// weights beyond the K cutoff are not carried and read as 1.
  CodecParams params;
};

/// @brief Read a frame's complete header, including its quantization weights.
///
/// What a player calls on its first frame to build a grid of the geometry the
/// stream carries (`voxel_size`, `trunc_dist`, `block_size` 8, and room for
/// `block_count` blocks), before it can @ref Decoder::decode anything.
/// No entropy payload is read.
/// @param data  The frame (at least its complete header: 44 bytes plus four
///              per kept coefficient).
/// @param size  Bytes available at @p data.
/// @return The header, or `Status::Code::Unsupported` for another version,
///         frame type or block size, and `Status::Code::InvalidArgument`
///         for anything else malformed. It does not check the rest of the
///         frame, which @ref Decoder::decode does.
VR_CODEC_API core::Result<FrameInfo> read_frame_info(const std::uint8_t* data,
                                                     std::size_t size);

/// @brief How a @ref Decoder decodes a frame.
struct DecoderConfig {
  /// Where the rANS decoding runs. The device decodes one segment per
  /// invocation, so a frame of smaller segments gives it more parallelism.
  EntropyCoding entropy = EntropyCoding::kAuto;
};

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
/// no allocation. A block the decode leaves different is stamped `changed`
/// (@ref volume::BlockStamp) and one it leaves as it was is not, so a
/// consumer of the stamps sees only what the stream changed.
///
/// Only `tsdf` and `weight` are written, so a grid that declares any other
/// attribute is refused: a block kept across frames would carry that
/// attribute's previous values (a `color`, say) under the new geometry, and
/// nothing could tell which blocks those were.
///
/// @warning The `Device` and `Allocator` passed to @ref create must
///          outlive this object. Not thread-safe: one call at a time, and
///          nothing else may modify the grid during one.
class VR_CODEC_API Decoder {
 public:
  /// @brief Build the inverse transform's kernels, and with
  ///        @ref EntropyCoding::kDevice the rANS decoder's; @ref
  ///        EntropyCoding::kAuto builds those at its first device frame.
  /// @param device     The compute device (must outlive this object).
  /// @param allocator  The allocator per-call buffers come from (must outlive
  ///                   this object).
  /// @param config     How frames are decoded.
  /// @return The decoder, or a pipeline or allocation failure.
  static core::Result<Decoder> create(core::Device& device,
                                      core::Allocator& allocator,
                                      const DecoderConfig& config = {});

  ~Decoder();
  Decoder(Decoder&& other) noexcept;
  Decoder& operator=(Decoder&& other) noexcept;
  Decoder(const Decoder&) = delete;
  Decoder& operator=(const Decoder&) = delete;

  /// @brief Decode one intra frame into @p grid.
  ///
  /// Everything that can be checked before the grid is touched is: the grid's
  /// attributes, its geometry against the frame's header, the whole frame
  /// parsed and entropy-decoded, and the grid's heap against its block count.
  /// A refusal there leaves @p grid exactly as it was. A failure **after** the
  /// grid starts to change -- the hash table cannot place every block, its
  /// bucket locks keep losing races, or a dispatch fails -- leaves it valid but
  /// holding **neither** frame: blocks the two frames share still hold the
  /// previous frame's voxels, and blocks new to this one are empty, so it is
  /// not worth meshing. The next successful decode restores it exactly, since
  /// each decode starts from whatever the grid holds.
  /// @param data     The frame.
  /// @param size     Its exact length.
  /// @param grid     The grid to decode into: 4-byte float `tsdf` and
  ///                 `weight` and no other attribute, `block_size` 8, and the
  ///                 frame's `voxel_size` and `trunc_dist` exactly (see
  ///                 @ref read_frame_info).
  /// @param metrics  Optional `StageMetrics` collecting a `"codec decode"`
  ///                 row with both halves -- its device half is the inverse
  ///                 transform and, decoded on the device, the rANS
  ///                 kernels -- over the breakdown rows `"  ..rans decode"`
  ///                 (parsing and decoding the frame), `"  ..active set"`
  ///                 (the compaction), `"  ..apply"` (merging the two block
  ///                 sets, and removing and allocating blocks) and
  ///                 `"  ..inverse"` (the transform). Named apart from the @ref
  ///                 Encoder's, so both timed into one `StageMetrics` stay
  ///                 apart. `nullptr` measures nothing.
  /// @return OK, or: whatever the frame reader refuses
  ///         (`Status::Code::Unsupported`, @ref
  ///         Status::Code::InvalidArgument for a malformed or corrupt frame);
  ///         with @ref EntropyCoding::kDevice, a frame the device cannot
  ///         decode (see @ref EntropyCoding);
  ///         `Status::Code::InvalidArgument` for a moved-from decoder, a
  ///         grid that is moved-from, has another block size or geometry,
  ///         lacks a float `tsdf` / `weight`, or declares any other
  ///         attribute, or a grid whose free heap refuses a removed block
  ///         (its accounting was already broken, which nothing here mends);
  ///         `Status::Code::OutOfMemory` when the grid is too small for
  ///         the frame -- fewer block slots (`num_blocks`) than the frame has
  ///         blocks, or a hash table that cannot place them all -- which
  ///         @ref volume::VoxelBlockGrid::resize and decoding again recovers;
  ///         `Status::Code::IoError` only when bucket-lock contention
  ///         outlasts the retries over a table with room (decode again; a
  ///         resize would not help); otherwise a compaction, buffer or
  ///         dispatch failure.
  core::Status decode(const std::uint8_t* data, std::size_t size,
                      volume::VoxelBlockGrid& grid,
                      core::StageMetrics* metrics = nullptr);

  /// @return `true` if this owns a live transform (`false` when moved-from).
  bool valid() const noexcept;

 private:
  Decoder();

  /// Build @ref reader_ unless it is built, or return the failure of the
  /// one build tried.
  core::Status ensure_reader();
  /// Decode @p frame's segments on the built reader, through the submit; the
  /// reader's check() then judges them.
  core::Result<detail::ResidentBlocks> decode_on_device(
      const detail::ParsedFrame& frame, core::GpuStageScope& stage);

  DecoderConfig config_;
  // Borrowed (must outlive this); what the device decode's batch runs on.
  core::Device* device_ = nullptr;
  core::Allocator* allocator_ = nullptr;
  std::unique_ptr<detail::DctTransform> transform_;
  // Built at create for kDevice, at the first device frame for kAuto, and
  // never for kHost; and why that build failed, after which kAuto decodes
  // every frame on the host.
  std::unique_ptr<detail::DeviceFrameReader> reader_;
  core::Status reader_failure_;
  // Device spans for the transform; idle until a caller asks for metrics.
  core::GpuTimer gpu_timer_;
};

}  // namespace volumetric_kit::recon::codec
