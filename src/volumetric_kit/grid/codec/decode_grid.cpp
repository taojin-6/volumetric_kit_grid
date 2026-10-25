// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include "volumetric_kit/recon/codec/decode_grid.hpp"

#include <algorithm>
#include <cstring>
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

namespace {

// Grow @p grid once for a frame of @p blocks that the decoder refused as too
// small (@p refused), unless that passes @p max_blocks' grid: to twice its
// buckets or the frame's make_decode_grid buckets, whichever is more.
core::Status grow_for(const char* who, volume::VoxelBlockGrid& grid,
                      std::uint32_t blocks, std::uint32_t max_blocks,
                      const core::Status& refused) {
  // grow_grid's target, in slots, against make_decode_grid's grid for the
  // ceiling.
  const std::int32_t at_least = decode_buckets(blocks);
  const volume::VoxelGridParams& gp = grid.grid();
  const std::int64_t slots =
      std::int64_t(gp.bucket_size) *
      std::max<std::int64_t>(2 * std::int64_t(gp.num_buckets), at_least);
  const std::int64_t ceiling =
      std::int64_t(volume::kDefaultBucketSize) * decode_buckets(max_blocks);
  if (slots > ceiling) {
    return core::Status::invalid_argument(
        std::string(who) + ": growing the grid to " + std::to_string(slots) +
        " slots would pass the " + std::to_string(ceiling) +
        " the caller's ceiling of " + std::to_string(max_blocks) +
        " blocks allows (" + refused.message() + ")");
  }
  return volume::grow_grid(grid, at_least);
}

// Decode the record of @p blocks blocks at the start of @p slot again,
// blocking, in batches of its own, as the decoder's refusal @p s asks: after
// a grow for `Status::Code::OutOfMemory` (once), or as it is for
// `Status::Code::IoError` (twice at most). Each recovery, its decode again
// included, is one `"  ..grow"` or `"  ..retry"` row of @p metrics inside a
// `"codec decode"` span, so the stage's row holds it. Returns OK, or the last
// error.
core::Status recover(Decoder& decoder, const core::HostSlot& slot,
                     std::size_t size, volume::VoxelBlockGrid& grid,
                     std::uint32_t blocks, std::uint32_t max_blocks,
                     core::Status s, core::StageMetrics* metrics) {
  int contended = 0;
  bool grown = false;
  // The stage's row above its breakdown's, though its span closes after.
  if (metrics != nullptr) metrics->seed("codec decode");
  for (;;) {
    const bool grow = s.domain() == core::Status::Code::OutOfMemory && !grown;
    if (!grow && (s.domain() != core::Status::Code::IoError ||
                  contended == kContendedRetries)) {
      return s;
    }
    const core::StageScope stage(metrics, "codec decode");
    const core::StageScope row(metrics, grow ? "  ..grow" : "  ..retry");
    if (grow) {
      VKC_TRY(grow_for("decode_growing", grid, blocks, max_blocks, s));
      grown = true;
    } else {
      ++contended;
    }
    s = decoder.decode(slot, size, grid, nullptr);
    if (s.ok()) return s;
  }
}

}  // namespace

core::Result<PendingDecode> record_growing(
    Decoder& decoder, core::CommandBatch& batch, const core::HostSlot& slot,
    std::size_t size, volume::VoxelBlockGrid& grid, std::uint32_t max_blocks,
    core::StageMetrics* metrics) try {
  VKC_ASSIGN(const FrameInfo info, read_frame_info(slot.data(), size));
  VKC_TRY(check_ceiling("record_growing", info.block_count, max_blocks));
  core::Result<PendingDecode> recorded =
      decoder.record(batch, slot, size, grid, metrics);
  // A heap too small is refused with nothing recorded, so the frame records
  // again once the grid has grown. The device's own out of memory can come
  // after the first commands, so the batch is the caller's to drop.
  if (recorded.ok() ||
      recorded.status().domain() != core::Status::Code::OutOfMemory ||
      info.block_count <= static_cast<std::uint32_t>(grid.grid().num_blocks)) {
    return recorded;
  }
  {
    // Part of the decode, whose row the record above opened.
    const core::StageScope stage(metrics, "codec decode");
    const core::StageScope grow(metrics, "  ..grow");
    VKC_TRY(grow_for("record_growing", grid, info.block_count, max_blocks,
                     recorded.status()));
  }
  return decoder.record(batch, slot, size, grid, metrics);
} catch (const std::bad_alloc&) {
  return core::exception_status("record_growing",
                                core::Status::Code::OutOfMemory);
}

core::Result<Finished> finish_growing(Decoder& decoder,
                                      const PendingDecode& pending,
                                      volume::VoxelBlockGrid& grid,
                                      std::uint32_t max_blocks,
                                      core::StageMetrics* metrics) try {
  const core::Status s = decoder.finish(pending, grid);
  if (s.ok()) return Finished{};
  VKC_TRY(recover(decoder, pending.input, pending.size, grid, pending.blocks,
                  max_blocks, s, metrics));
  return Finished{true};
} catch (const std::bad_alloc&) {
  return core::exception_status("finish_growing",
                                core::Status::Code::OutOfMemory);
}

core::Status decode_growing(Decoder& decoder, const std::uint8_t* frame,
                            std::size_t size, volume::VoxelBlockGrid& grid,
                            std::uint32_t max_blocks,
                            core::StageMetrics* metrics) try {
  VKC_ASSIGN(const FrameInfo info, read_frame_info(frame, size));
  VKC_TRY(check_ceiling("decode_growing", info.block_count, max_blocks));
  VKC_ASSIGN(const core::HostSlot slot, decoder.claim_input(size));
  std::memcpy(slot.data(), frame, size);
  // The whole call is timed, the first decode and a recovery, and its rows
  // land in the caller's metrics once the decode succeeds. A heap too small
  // is the record's refusal and a table that cannot place the frame the
  // finish's, and either recovers alike.
  core::StageMetrics call;
  core::StageMetrics* const rows = metrics != nullptr ? &call : nullptr;
  const core::Status s = decoder.decode(slot, size, grid, rows);
  if (!s.ok()) {
    VKC_TRY(recover(decoder, slot, size, grid, info.block_count, max_blocks, s,
                    rows));
  }
  if (metrics != nullptr) {
    try {
      core::StageMetrics merged = *metrics;
      merged.merge(call);
      *metrics = std::move(merged);
    } catch (const std::bad_alloc&) {
      // The decoder already committed its reference. Dropping this decode's
      // diagnostics must not turn that success into a failure the caller
      // would retry; keep the caller's previous rows intact.
    }
  }
  return {};
} catch (const std::bad_alloc&) {
  return core::exception_status("decode_growing",
                                core::Status::Code::OutOfMemory);
}

}  // namespace volumetric_kit::recon::codec
