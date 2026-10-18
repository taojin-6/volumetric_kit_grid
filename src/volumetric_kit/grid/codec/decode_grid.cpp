// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include "volumetric_kit/recon/codec/decode_grid.hpp"

#include <algorithm>
#include <limits>
#include <new>

#include "volumetric_kit/recon/volume/grid_growth.hpp"
#include "volumetric_kit/recon/volume/voxel_grid.hpp"

namespace volumetric_kit::recon::codec {
namespace {

// The grid's hash layout: entries, and heap slots, per bucket; chain length.
constexpr std::int32_t kBucketSize = 8;
constexpr std::int32_t kMaxChain = 128;
constexpr std::int32_t kMinBuckets = 1024;
// Decodes again after the decoder's IoError, its lock contention.
constexpr int kContendedRetries = 2;

// Buckets for twice @p blocks, at least kMinBuckets. Capped so the heap's
// slot count fits an int32_t; no grid holds that many blocks anyway, which
// VoxelBlockGrid::create and grow_grid refuse.
std::int32_t decode_buckets(std::uint32_t blocks) {
  const std::int64_t need =
      (2 * std::int64_t(blocks) + kBucketSize - 1) / kBucketSize;
  return std::int32_t(std::clamp<std::int64_t>(
      need, kMinBuckets,
      std::numeric_limits<std::int32_t>::max() / kBucketSize));
}

}  // namespace

core::Result<volume::VoxelBlockGrid> make_decode_grid(
    core::Device& device, core::Allocator& allocator, const std::uint8_t* frame,
    std::size_t size) {
  VKC_ASSIGN(const FrameInfo info, read_frame_info(frame, size));
  volume::VoxelGridParams grid{};
  grid.voxel_size = info.voxel_size;
  grid.block_size = kBlockSize;
  grid.voxels_per_block = std::int32_t(kVoxelsPerBlock);
  grid.trunc_dist = info.trunc_dist;
  grid.bucket_size = kBucketSize;
  grid.num_buckets = decode_buckets(info.block_count);
  grid.num_blocks = kBucketSize * grid.num_buckets;
  grid.max_chain = kMaxChain;
  const volume::AttributeSpec attrs[] = {{"tsdf", sizeof(float)},
                                         {"weight", sizeof(float)}};
  return volume::VoxelBlockGrid::create(device, allocator, grid, attrs, 2);
}

core::Status decode_growing(Decoder& decoder, const std::uint8_t* frame,
                            std::size_t size, volume::VoxelBlockGrid& grid,
                            core::StageMetrics* metrics) try {
  int contended = 0;
  bool grown = false;
  for (;;) {
    core::StageMetrics attempt;
    const core::Status s = decoder.decode(
        frame, size, grid, metrics != nullptr ? &attempt : nullptr);
    if (s.ok()) {
      if (metrics != nullptr) metrics->merge(attempt);
      return s;
    }
    if (s.domain() == core::Status::Code::OutOfMemory) {
      if (grown) return s;
      // The decoder read this header before any OutOfMemory.
      VKC_ASSIGN(const FrameInfo info, read_frame_info(frame, size));
      VKC_TRY(volume::grow_grid(grid, decode_buckets(info.block_count)));
      grown = true;
    } else if (s.domain() != core::Status::Code::IoError ||
               ++contended > kContendedRetries) {
      return s;
    }
  }
} catch (const std::bad_alloc&) {
  return core::Status::out_of_memory("decode_growing: host allocation failed");
}

}  // namespace volumetric_kit::recon::codec
