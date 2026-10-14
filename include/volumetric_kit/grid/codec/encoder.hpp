// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file codec/encoder.hpp
/// @brief Encode a grid's geometry as one self-contained intra frame.

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
class DctTransform;
class DeviceFrameWriter;
}  // namespace detail

/// @brief How an @ref Encoder codes a frame.
struct EncoderConfig {
  /// Coefficients kept per block and their quantization steps.
  CodecParams params;
  /// Blocks per independently coded segment, from 1 to
  /// @ref kMaxDeviceDecodeSegmentSize. The device codes and decodes one
  /// segment an invocation, so smaller segments give it more parallelism, at
  /// a little more size (each segment restates its first coordinate); the
  /// default costs 2-3% of a frame.
  std::uint32_t segment_size = 64;
};

/// @brief Encodes a @ref volume::VoxelBlockGrid's geometry as one intra frame
///        -- the codec's capture side (the 2026-09-26 decision).
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
/// device (a frame is reproducible per device, decodable on any).
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
  /// @brief Build the kernels: the forward transform's and the rANS
  ///        coder's.
  /// @param device     The compute device (must outlive this object).
  /// @param allocator  The allocator per-call buffers come from (must outlive
  ///                   this object).
  /// @param config     How frames are coded.
  /// @return The encoder, or `Status::Code::InvalidArgument` for invalid
  ///         `config.params` or a `segment_size` of 0 or past
  ///         @ref kMaxDeviceDecodeSegmentSize; otherwise a pipeline or
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

  /// @brief Encode every observed block of @p grid as one intra frame.
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
  ///                 `"  ..sort"`, `"  ..forward"` (the transform, and the
  ///                 symbol count, which shares its batch) and
  ///                 `"  ..rans encode"` (coding and laying out the frame).
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

  /// @return `true` if this owns a live transform (`false` when moved-from).
  bool valid() const noexcept;

 private:
  Encoder();

  EncoderConfig config_;
  // Borrowed (must outlive this); what the coding's batches run on.
  core::Device* device_ = nullptr;
  core::Allocator* allocator_ = nullptr;
  std::unique_ptr<detail::DctTransform> transform_;
  std::unique_ptr<detail::DeviceFrameWriter> writer_;
  // Device spans for the kernels; idle until a caller asks for metrics.
  core::GpuTimer gpu_timer_;
};

}  // namespace volumetric_kit::recon::codec
