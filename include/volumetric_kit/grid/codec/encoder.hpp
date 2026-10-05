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
  /// Blocks per independently decodable segment, at least 1. Smaller costs a
  /// little more (each segment restates its first coordinate) and decodes
  /// with more parallelism; the default is about 1% overhead.
  std::uint32_t segment_size = 64;
  /// Where the rANS coding runs. The device codes one segment per
  /// invocation, so smaller segments give it more parallelism.
  EntropyCoding entropy = EntropyCoding::kAuto;
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
/// @warning The @ref Device and @ref Allocator passed to @ref create must
///          outlive this object. Not thread-safe: one call at a time.
class VR_CODEC_API Encoder {
 public:
  /// @brief Build the forward transform's kernels, and with
  ///        @ref EntropyCoding::kDevice the rANS coder's; @ref
  ///        EntropyCoding::kAuto builds those at its first device frame.
  /// @param device     The compute device (must outlive this object).
  /// @param allocator  The allocator per-call buffers come from (must outlive
  ///                   this object).
  /// @param config     How frames are coded.
  /// @return The encoder, or @ref Status::Code::InvalidArgument for invalid
  ///         `config.params` or a `segment_size` of 0; otherwise a pipeline
  ///         or allocation failure.
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
  ///                 which an extract after this takes back from it as it
  ///                 would a fuse's.
  /// @param metrics  Optional @ref StageMetrics collecting a `"codec encode"`
  ///                 row with both halves -- its device half is the transform
  ///                 and, coded on the device, the rANS kernels -- over the
  ///                 breakdown rows `"  ..active set"` (the compaction, when
  ///                 the map holds no list that is still its active set),
  ///                 `"  ..observed"` (finding the blocks with an observed
  ///                 voxel), `"  ..sort"`, `"  ..forward"` (the transform;
  ///                 on the device, also the symbol count, which shares its
  ///                 batch) and `"  ..rans encode"` (writing the frame; on
  ///                 the host, counting included). Named apart from the
  ///                 @ref Decoder's, so both timed into one
  ///                 @ref StageMetrics stay apart. `nullptr` measures
  ///                 nothing.
  /// @return The frame's bytes (an empty grid is a valid frame of no blocks),
  ///         or @ref Status::Code::InvalidArgument for a moved-from encoder,
  ///         a grid the transform refuses (moved-from, another block size,
  ///         no float `tsdf` / `weight`) or a non-finite `voxel_size`;
  ///         otherwise a compaction, buffer, pipeline or dispatch failure.
  core::Result<std::vector<std::uint8_t>> encode(
      volume::VoxelBlockGrid& grid, core::StageMetrics* metrics = nullptr);

  /// @return The configuration frames are coded with; all zeros when
  ///         moved-from, which no valid encoder has.
  const EncoderConfig& config() const noexcept { return config_; }

  /// @return `true` if this owns a live transform (`false` when moved-from).
  bool valid() const noexcept;

 private:
  Encoder();

  /// Build @ref writer_ unless it is built, or return the failure of the
  /// one build tried.
  core::Status ensure_writer();
  /// The forward and the rANS coding on the built writer.
  core::Result<std::vector<std::uint8_t>> encode_on_device(
      volume::VoxelBlockGrid& grid,
      const std::vector<volume::BlockIndex>& blocks,
      core::StageMetrics* metrics, core::GpuStageScope& stage);

  EncoderConfig config_;
  // Borrowed (must outlive this); what the device coding's batch runs on.
  core::Device* device_ = nullptr;
  core::Allocator* allocator_ = nullptr;
  std::unique_ptr<detail::DctTransform> transform_;
  // Built at create for kDevice, at the first device frame for kAuto, and
  // never for kHost; and why that build failed, after which kAuto codes
  // every frame on the host.
  std::unique_ptr<detail::DeviceFrameWriter> writer_;
  core::Status writer_failure_;
  // Device spans for the transform; idle until a caller asks for metrics.
  core::GpuTimer gpu_timer_;
};

}  // namespace volumetric_kit::recon::codec
