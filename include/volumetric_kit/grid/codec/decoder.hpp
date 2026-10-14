// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file codec/decoder.hpp
/// @brief Decode a frame -- intra, or a P-frame against the frame decoded
///        before it -- into a @ref volume::VoxelBlockGrid, and read a frame's
///        header without decoding it.

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
class BlockLists;
class DctTransform;
class DeviceFrameReader;
class ReferenceStore;
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

/// @brief Read a frame's complete header, including its quantization weights.
///
/// What a player calls on its first frame to build a grid of the geometry the
/// stream carries (`voxel_size`, `trunc_dist`, `block_size` 8, and room for
/// `block_count` blocks), before it can @ref Decoder::decode anything.
/// No entropy payload is read.
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

/// @brief Decodes frames into a caller's @ref volume::VoxelBlockGrid -- the
///        codec's playback side (the 2026-09-26 decision).
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
/// A P-frame (@ref FrameType::kInter, experimental) decodes against the last
/// frame this decoder decoded, its reference, which the decoder keeps on the
/// device as it decoded it -- quantized coefficients, masks and block list,
/// never rebuilt from a grid -- so it decodes into any grid of its geometry,
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
/// and tables; and the blocks the grid must drop take 16 bytes a hash slot.
/// All of it is kept, sized for the largest frame decoded.
///
/// Everything but I/O and control runs on the device: the rANS decode, one
/// segment an invocation; the reader's verdict; the merge of a P-frame's
/// survivors with its new blocks; and the diff of the grid's blocks against
/// the frame's. The host parses the header and tables and reads back 24
/// bytes of verdict and counts before the grid changes. A frame the device
/// cannot hold is refused, never decoded on the host.
///
/// @warning The `Device` and `Allocator` passed to @ref create must
///          outlive this object. Not thread-safe: one call at a time, and
///          nothing else may modify the grid during one.
class VR_CODEC_API Decoder {
 public:
  /// @brief Build the kernels: the rANS decoder's, the apply's and the
  ///        inverse transform's.
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
  /// or a dispatch fails -- leaves it valid but holding **neither** frame:
  /// blocks the two frames share still hold the previous frame's voxels, and
  /// blocks new to this one are empty, so it is not worth meshing. The next
  /// successful decode restores it exactly, since each decode starts from
  /// whatever the grid holds.
  /// @param data     The frame.
  /// @param size     Its exact length.
  /// @param grid     The grid to decode into: 4-byte float `tsdf` and
  ///                 `weight` and no other attribute, `block_size` 8, and the
  ///                 frame's `voxel_size` and `trunc_dist` exactly (see
  ///                 @ref read_frame_info).
  /// @param metrics  Optional `StageMetrics` collecting a `"codec decode"`
  ///                 row with both halves -- its device half is the rANS
  ///                 kernels, the apply and the inverse transform -- over
  ///                 the breakdown rows
  ///                 `"  ..rans decode"` (parsing the frame, then one
  ///                 submit: decoding it, merging a P-frame's blocks with
  ///                 its reference's, and diffing the grid's block set
  ///                 against the frame's), `"  ..apply"` (removing and
  ///                 allocating blocks) and `"  ..inverse"` (making the
  ///                 frame's blocks from the reference and the payload, and
  ///                 the transform, in one submit). Named apart from the
  ///                 @ref Encoder's, so both timed into one `StageMetrics`
  ///                 stay apart. `nullptr` measures nothing.
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
  ///         (its accounting was already broken, which nothing here mends);
  ///         `Status::Code::OutOfMemory` when the grid is too small for
  ///         the frame -- fewer block slots (`num_blocks`) than the frame has
  ///         blocks, or a hash table that cannot place them all -- which
  ///         @ref volume::VoxelBlockGrid::resize and decoding again recovers;
  ///         `Status::Code::IoError` only when bucket-lock contention
  ///         outlasts the retries over a table with room (decode again; a
  ///         resize would not help); otherwise a buffer or
  ///         dispatch failure.
  core::Status decode(const std::uint8_t* data, std::size_t size,
                      volume::VoxelBlockGrid& grid,
                      core::StageMetrics* metrics = nullptr);

  /// @return `true` if this owns a live transform (`false` when moved-from).
  bool valid() const noexcept;

 private:
  Decoder();

  // Borrowed (must outlive this); what the decode's batches run on.
  core::Device* device_ = nullptr;
  core::Allocator* allocator_ = nullptr;
  std::unique_ptr<detail::DctTransform> transform_;
  std::unique_ptr<detail::DeviceFrameReader> reader_;
  // The reference a P-frame decodes against, and the apply into it.
  std::unique_ptr<detail::ReferenceStore> store_;
  // The frame's blocks merged into frame order, and diffed against the grid.
  std::unique_ptr<detail::BlockLists> lists_;
  // Device spans for the transform; idle until a caller asks for metrics.
  core::GpuTimer gpu_timer_;
};

}  // namespace volumetric_kit::recon::codec
