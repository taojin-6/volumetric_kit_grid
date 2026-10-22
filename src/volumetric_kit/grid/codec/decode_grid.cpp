// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include "volumetric_kit/recon/codec/decode_grid.hpp"

#include <algorithm>
#include <limits>
#include <new>
#include <string>
#include <utility>

#include "bitstream.hpp"
#include "volumetric_kit/core/base/exception_status.hpp"
#include "volumetric_kit/recon/volume/grid_growth.hpp"
#include "volumetric_kit/recon/volume/voxel_grid.hpp"

namespace volumetric_kit::recon::codec {
namespace {

// The grid's fewest buckets.
constexpr std::int32_t kMinBuckets = 1024;
// Decodes again after the decoder's IoError, its lock contention.
constexpr int kContendedRetries = 2;

// Buckets for twice @p blocks, at least kMinBuckets. Capped so the heap's
// slot count fits an int32_t; no grid holds that many blocks anyway, which
// VoxelBlockGrid::create and grow_grid refuse.
std::int32_t decode_buckets(std::uint32_t blocks) {
  constexpr std::int32_t kBucket = volume::kDefaultBucketSize;
  const std::int64_t need = (2 * std::int64_t(blocks) + kBucket - 1) / kBucket;
  return std::int32_t(std::clamp<std::int64_t>(
      need, kMinBuckets, std::numeric_limits<std::int32_t>::max() / kBucket));
}

// The caller's ceiling on a frame's blocks, refused as a frame to skip.
core::Status check_ceiling(const char* who, std::uint32_t blocks,
                           std::uint32_t max_blocks) {
  if (blocks <= max_blocks) return {};
  return core::Status::invalid_argument(
      std::string(who) + ": the frame holds " + std::to_string(blocks) +
      " blocks, more than the caller's ceiling of " +
      std::to_string(max_blocks));
}

}  // namespace

core::Result<volume::VoxelBlockGrid> make_decode_grid(
    core::Device& device, core::Allocator& allocator, const std::uint8_t* frame,
    std::size_t size, std::uint32_t max_blocks) try {
  VKC_ASSIGN(const FrameInfo info, read_frame_info(frame, size));
  VKC_TRY(check_ceiling("make_decode_grid", info.block_count, max_blocks));
  // The block count against the segment table, as the decoder parses the
  // frame. A sound P-frame stops at the reference only a decoder holds
  // (NotFound); anything else is the decoder's refusal of the frame.
  {
    const core::Result<detail::ParsedFrame> parsed =
        detail::parse_frame(frame, size, info.block_count, nullptr);
    if (!parsed.ok() &&
        parsed.status().domain() != core::Status::Code::NotFound) {
      return parsed.status();
    }
  }
  volume::VoxelGridParams grid{};
  grid.voxel_size = info.voxel_size;
  grid.block_size = kBlockSize;
  grid.voxels_per_block = std::int32_t(kVoxelsPerBlock);
  grid.trunc_dist = info.trunc_dist;
  grid.bucket_size = volume::kDefaultBucketSize;
  grid.num_buckets = decode_buckets(info.block_count);
  grid.num_blocks = grid.bucket_size * grid.num_buckets;
  grid.max_chain = volume::kDefaultMaxChain;
  const volume::AttributeSpec attrs[] = {{"tsdf", sizeof(float)},
                                         {"weight", sizeof(float)}};
  return volume::VoxelBlockGrid::create(device, allocator, grid, attrs, 2);
} catch (const std::bad_alloc&) {
  return core::exception_status("make_decode_grid",
                                core::Status::Code::OutOfMemory);
}

core::Status decode_growing(Decoder& decoder, const std::uint8_t* frame,
                            std::size_t size, volume::VoxelBlockGrid& grid,
                            std::uint32_t max_blocks,
                            core::StageMetrics* metrics) try {
  VKC_ASSIGN(const FrameInfo info, read_frame_info(frame, size));
  VKC_TRY(check_ceiling("decode_growing", info.block_count, max_blocks));
  int contended = 0;
  bool grown = false;
  for (;;) {
    core::StageMetrics attempt;
    const core::Status s = decoder.decode(
        frame, size, grid, metrics != nullptr ? &attempt : nullptr);
    if (s.ok()) {
      if (metrics != nullptr) {
        try {
          core::StageMetrics merged = *metrics;
          merged.merge(attempt);
          *metrics = std::move(merged);
        } catch (const std::bad_alloc&) {
          // The decoder already committed its reference. Dropping this
          // attempt's diagnostics must not turn that success into a failure
          // the caller would retry; keep the caller's previous rows intact.
        }
      }
      return s;
    }
    if (s.domain() == core::Status::Code::OutOfMemory) {
      if (grown) return s;
      // grow_grid's target, in slots, against make_decode_grid's grid for
      // the ceiling.
      const std::int32_t at_least = decode_buckets(info.block_count);
      const volume::VoxelGridParams& gp = grid.grid();
      const std::int64_t slots =
          std::int64_t(gp.bucket_size) *
          std::max<std::int64_t>(2 * std::int64_t(gp.num_buckets), at_least);
      const std::int64_t ceiling =
          std::int64_t(volume::kDefaultBucketSize) * decode_buckets(max_blocks);
      if (slots > ceiling) {
        return core::Status::invalid_argument(
            "decode_growing: growing the grid to " + std::to_string(slots) +
            " slots would pass the " + std::to_string(ceiling) +
            " the caller's ceiling of " + std::to_string(max_blocks) +
            " blocks allows (" + s.message() + ")");
      }
      VKC_TRY(volume::grow_grid(grid, at_least));
      grown = true;
    } else if (s.domain() != core::Status::Code::IoError ||
               ++contended > kContendedRetries) {
      return s;
    }
  }
} catch (const std::bad_alloc&) {
  return core::exception_status("decode_growing",
                                core::Status::Code::OutOfMemory);
}

}  // namespace volumetric_kit::recon::codec
