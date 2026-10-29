// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// GPU test for codec/decode_grid.hpp, on sphere grids: make_decode_grid's
// geometry, layout and attributes off a frame's header, at the minimum bucket
// count and past it, and the headers it refuses; decode_growing leaving a grid
// the frame fits alone, growing one too small exactly once to decode what a
// roomy grid decodes, from bytes or an input slot (a slot short of the record
// refused), returning the second OutOfMemory of a grid one grow
// leaves too small, timing the whole call once it succeeds, and passing the
// decoder's refusals through with their codes, ungrown; and the caller's
// ceiling, which refuses a forged header, a forged frame and a grow past it
// with the grid as it was. Skips with no device.

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <string_view>
#include <vector>

#include "bitstream.hpp"
#include "volumetric_kit/core/base/stage_metrics.hpp"
#include "volumetric_kit/core/testing/gpu_test.hpp"
#include "volumetric_kit/grid/base/device_requirements.hpp"
#include "volumetric_kit/grid/codec/decode_grid.hpp"
#include "volumetric_kit/grid/codec/decoder.hpp"
#include "volumetric_kit/grid/codec/encoder.hpp"
#include "volumetric_kit/grid/testing/codec_fixture.hpp"
#include "volumetric_kit/grid/testing/stage_rows.hpp"

namespace vgrid = volumetric_kit::grid;
namespace vkc = volumetric_kit::core;
namespace codec = volumetric_kit::grid::codec;
namespace d = volumetric_kit::grid::codec::detail;
using namespace codec_fixture;

namespace {

using Bytes = std::vector<std::uint8_t>;
using Code = vkc::Status::Code;

// The ceiling the cases pass where they do not test it: far past their frames.
constexpr std::uint32_t kMaxBlocks = 1u << 20;

// The frames: a small sphere's at its own geometry, and a large sphere's of
// more blocks than the minimum bucket count holds at half the heap.
struct Frames {
  Bytes small;
  std::uint32_t small_blocks = 0;
  Bytes large;
  std::uint32_t large_blocks = 0;
};

constexpr float kSmallVoxel = 0.0075f;
constexpr float kSmallTrunc = 0.03f;

vkc::Result<Bytes> encode(Gpu& gpu, vol::VoxelBlockGrid& grid) {
  VKC_ASSIGN(codec::Encoder enc,
             codec::Encoder::create(gpu.device, gpu.allocator, {}));
  return enc.encode(grid);
}

vkc::Result<Frames> make_frames(Gpu& gpu) {
  Frames f;
  GridShape own;
  own.voxel_size = kSmallVoxel;
  own.trunc_dist = kSmallTrunc;
  VKC_ASSIGN(
      vol::VoxelBlockGrid small,
      sphere_grid(gpu, Sphere{vgrid::Vec3f(0.01f, 0.0f, -0.02f), 0.1f}, own));
  VKC_ASSIGN(f.small, encode(gpu, small));
  GridShape roomy;
  roomy.num_buckets = 2048;
  VKC_ASSIGN(vol::VoxelBlockGrid large,
             sphere_grid(gpu, Sphere{vgrid::Vec3f(0.0f), 0.6f}, roomy));
  VKC_ASSIGN(f.large, encode(gpu, large));
  VKC_ASSIGN(const codec::FrameInfo s,
             codec::read_frame_info(f.small.data(), f.small.size()));
  VKC_ASSIGN(const codec::FrameInfo l,
             codec::read_frame_info(f.large.data(), f.large.size()));
  f.small_blocks = s.block_count;
  f.large_blocks = l.block_count;
  return f;
}

// make_decode_grid's documented floor, and the most blocks it holds at half
// its heap.
constexpr std::int32_t kMinBuckets = 1024;
constexpr std::uint32_t kMinBucketBlocks =
    kMinBuckets * vol::kDefaultBucketSize / 2;

// The buckets make_decode_grid documents: twice the blocks, at least 1024.
std::int32_t decode_buckets(std::uint32_t blocks) {
  constexpr std::int32_t kBucket = vol::kDefaultBucketSize;
  return std::max(
      std::int32_t((2 * std::int64_t(blocks) + kBucket - 1) / kBucket),
      kMinBuckets);
}

bool has_row(const vkc::StageMetrics& m, const char* name) {
  return vgrid_test::find_row(m, name) != nullptr;
}

// The grid from a frame's header: its geometry exactly, 8^3 blocks, the
// default hash layout with a slot per entry, twice the blocks' buckets at
// least 1024, float tsdf and weight alone, and no block yet; a frame decodes
// into it. A ceiling of the frame's own blocks admits it.
int make_grid_case(Gpu& gpu, const Frames& f) {
  CHECK(f.small_blocks > 32 && f.small_blocks < kMinBucketBlocks);
  CHECK(f.large_blocks > kMinBucketBlocks);
  struct Want {
    const Bytes* frame;
    std::uint32_t blocks;
    float voxel;
    float trunc;
    std::int32_t buckets;
  };
  const Want wants[] = {
      {&f.small, f.small_blocks, kSmallVoxel, kSmallTrunc, kMinBuckets},
      {&f.large, f.large_blocks, kVoxel, kTrunc,
       decode_buckets(f.large_blocks)},
  };
  for (const Want& want : wants) {
    vkc::Result<vol::VoxelBlockGrid> grid =
        codec::make_decode_grid(gpu.device, gpu.allocator, want.frame->data(),
                                want.frame->size(), want.blocks);
    CHECK(grid.ok());
    const vol::VoxelGridParams& gp = grid->grid();
    CHECK(gp.voxel_size == want.voxel && gp.trunc_dist == want.trunc);
    CHECK(gp.block_size == 8 && gp.voxels_per_block == 512);
    CHECK(gp.bucket_size == vol::kDefaultBucketSize &&
          gp.num_buckets == want.buckets);
    CHECK(gp.num_blocks == vol::kDefaultBucketSize * want.buckets &&
          gp.max_chain == vol::kDefaultMaxChain);
    CHECK(grid->attribute_count() == 2);
    for (const std::string_view name :
         {vol::kTsdfAttribute, vol::kWeightAttribute}) {
      vkc::Result<vol::AttributeView> view = grid->attribute(name);
      CHECK(view.ok() && view->element_size == sizeof(float));
    }
    vkc::Result<std::vector<vol::BlockIndex>> active = active_sorted(*grid);
    CHECK(active.ok() && active->empty());
    vkc::Result<codec::Decoder> dec =
        codec::Decoder::create(gpu.device, gpu.allocator);
    CHECK(dec.ok());
    CHECK(dec->decode(want.frame->data(), want.frame->size(), *grid).ok());
  }
  const vkc::Result<vol::VoxelBlockGrid> cut = codec::make_decode_grid(
      gpu.device, gpu.allocator, f.small.data(), 10, kMaxBlocks);
  CHECK(cut.status().domain() == Code::InvalidArgument);
  // A header claiming more blocks than any grid holds (block_count, at byte
  // 32) is refused, not built.
  Bytes most = f.small;
  const std::uint32_t all = 0xFFFFFFFFu;
  std::memcpy(most.data() + 32, &all, sizeof all);
  const vkc::Result<vol::VoxelBlockGrid> huge = codec::make_decode_grid(
      gpu.device, gpu.allocator, most.data(), most.size(), kMaxBlocks);
  CHECK(huge.status().domain() == Code::InvalidArgument);
  return 0;
}

// A frame of `blocks` blocks that a reader parses up to its block count:
// `real`'s coding and tables, segments of 1024 blocks, each the shortest
// stream a reader takes (4 bytes), so 8 bytes carry 1024 blocks.
vkc::Result<Bytes> forge(const Bytes& real, std::uint32_t blocks) {
  VKC_ASSIGN(const d::ParsedFrame parsed,
             d::parse_frame(real.data(), real.size(), kMaxBlocks, nullptr));
  d::CodedFrame f;
  f.voxel_size = parsed.header.voxel_size;
  f.trunc_dist = parsed.header.trunc_dist;
  f.params = parsed.header.params;
  f.block_count = blocks;
  f.segment_size = codec::kMaxDeviceDecodeSegmentSize;
  f.tables = parsed.tables;
  const std::uint64_t segments = d::frame_segment_count(blocks, f.segment_size);
  f.segment_lengths.assign(segments, 4);
  f.payload.assign(segments * 4, 0);
  return d::assemble_frame(codec::FrameType::kIntra, f);
}

// The ceiling, as InvalidArgument (a frame to skip): make_decode_grid refuses
// a header whose block count its segment table does not carry, and a frame
// past the ceiling; decode_growing refuses such a frame before it decodes it,
// and a grow past make_decode_grid's grid for the ceiling before it grows;
// each leaves the grid's buffers as they were.
int ceiling_case(Gpu& gpu, const Frames& f) {
  // A header that claims blocks its segment table does not carry, under the
  // ceiling.
  Bytes lying = f.small;
  const std::uint32_t claimed = 10000;
  std::memcpy(lying.data() + 32, &claimed, sizeof claimed);
  CHECK(codec::make_decode_grid(gpu.device, gpu.allocator, lying.data(),
                                lying.size(), kMaxBlocks)
            .status()
            .domain() == Code::InvalidArgument);
  // A frame one block past the ceiling.
  CHECK(codec::make_decode_grid(gpu.device, gpu.allocator, f.large.data(),
                                f.large.size(), f.large_blocks - 1)
            .status()
            .domain() == Code::InvalidArgument);

  // A forged frame whose segment table carries its blocks, in fewer bytes
  // than the small frame's: past the grid's 4096 slots, so the decoder alone
  // would ask for a grow, and one past the ceiling.
  constexpr std::uint32_t kForged = 5000;
  const vkc::Result<Bytes> forged = forge(f.small, kForged);
  CHECK(forged.ok() && forged->size() < f.small.size());
  CHECK(codec::make_decode_grid(gpu.device, gpu.allocator, forged->data(),
                                forged->size(), kForged - 1)
            .status()
            .domain() == Code::InvalidArgument);
  GridShape own;
  own.voxel_size = kSmallVoxel;
  own.trunc_dist = kSmallTrunc;
  vkc::Result<vol::VoxelBlockGrid> grid = make_grid(gpu, own);
  vkc::Result<codec::Decoder> dec =
      codec::Decoder::create(gpu.device, gpu.allocator);
  CHECK(grid.ok() && dec.ok() &&
        grid->grid().num_blocks < std::int32_t(kForged));
  CHECK(codec::decode_growing(*dec, f.small.data(), f.small.size(), *grid,
                              kForged - 1)
            .ok());
  vkc::Result<Snapshot> before = snapshot(gpu, *grid);
  CHECK(before.ok());
  const vkc::Status s = codec::decode_growing(
      *dec, forged->data(), forged->size(), *grid, kForged - 1);
  CHECK(s.domain() == Code::InvalidArgument);
  CHECK(grid->grid().num_buckets == own.num_buckets);
  vkc::Result<Snapshot> after = snapshot(gpu, *grid);
  CHECK(after.ok() && *after == *before);

  // A sound frame under the ceiling that a grid of about a slot a block, with
  // one-link chains, cannot place: doubling the grid would pass the slots of
  // make_decode_grid's grid for the ceiling, the frame's own blocks.
  GridShape cramped;
  cramped.num_buckets = std::int32_t(
      (f.large_blocks + vol::kDefaultBucketSize - 1) / vol::kDefaultBucketSize +
      1);
  cramped.max_chain = 1;
  vkc::Result<vol::VoxelBlockGrid> tight = make_grid(gpu, cramped);
  CHECK(tight.ok());
  const vkc::Status g = codec::decode_growing(
      *dec, f.large.data(), f.large.size(), *tight, f.large_blocks);
  CHECK(g.domain() == Code::InvalidArgument);
  CHECK(g.message().find("decode_growing") != std::string::npos);
  CHECK(tight->grid().num_buckets == cramped.num_buckets);
  return 0;
}

// A grid too small for the frame (32 slots) grows once, to the frame's
// buckets (not twice its own, nor more), and holds what a decode into a
// roomy grid holds; the whole call is timed, the refused decode and the grow
// with its decode again, in "codec decode", over "  ..grow".
int grows_once_case(Gpu& gpu, const Frames& f) {
  GridShape few;
  few.voxel_size = kSmallVoxel;
  few.trunc_dist = kSmallTrunc;
  few.num_buckets = 4;
  vkc::Result<vol::VoxelBlockGrid> grid = make_grid(gpu, few);
  CHECK(grid.ok() && grid->grid().num_blocks < std::int32_t(f.small_blocks));
  vkc::Result<codec::Decoder> dec =
      codec::Decoder::create(gpu.device, gpu.allocator);
  CHECK(dec.ok());
  vkc::StageMetrics metrics;
  CHECK(codec::decode_growing(*dec, f.small.data(), f.small.size(), *grid,
                              kMaxBlocks, &metrics)
            .ok());
  CHECK(grid->grid().num_buckets == decode_buckets(f.small_blocks));
  CHECK(metrics.rows().front().name == std::string("codec decode"));
  CHECK(has_row(metrics, "  ..grow") && vgrid_test::breakdowns_nest(metrics));

  vkc::Result<vol::VoxelBlockGrid> roomy = codec::make_decode_grid(
      gpu.device, gpu.allocator, f.small.data(), f.small.size(), kMaxBlocks);
  CHECK(roomy.ok());
  CHECK(dec->decode(f.small.data(), f.small.size(), *roomy).ok());
  vkc::Result<Snapshot> got = snapshot(gpu, *grid);
  vkc::Result<Snapshot> want = snapshot(gpu, *roomy);
  CHECK(got.ok() && want.ok() && got->coords.size() == f.small_blocks);
  CHECK(*got == *want);
  return 0;
}

// The slot form: a record copied into a claimed input slot decodes, grown
// once, to what the bytes' form decodes; a slot shorter than the record, or
// none, is refused with the grid as it was.
int slot_case(Gpu& gpu, const Frames& f) {
  GridShape few;
  few.voxel_size = kSmallVoxel;
  few.trunc_dist = kSmallTrunc;
  few.num_buckets = 4;
  vkc::Result<vol::VoxelBlockGrid> grid = make_grid(gpu, few);
  vkc::Result<vol::VoxelBlockGrid> bytes_grid = make_grid(gpu, few);
  vkc::Result<codec::Decoder> dec =
      codec::Decoder::create(gpu.device, gpu.allocator);
  vkc::Result<codec::Decoder> bytes_dec =
      codec::Decoder::create(gpu.device, gpu.allocator);
  CHECK(grid.ok() && bytes_grid.ok() && dec.ok() && bytes_dec.ok());

  // A slot may hold more than it was claimed for: a record a byte past it.
  vkc::Result<vkc::HostSlot> short_slot = dec->claim_input(f.small.size() / 2);
  CHECK(short_slot.ok());
  CHECK(codec::decode_growing(*dec, *short_slot,
                              std::size_t(short_slot->capacity()) + 1, *grid,
                              kMaxBlocks)
            .domain() == Code::InvalidArgument);
  CHECK(codec::decode_growing(*dec, vkc::HostSlot{}, f.small.size(), *grid,
                              kMaxBlocks)
            .domain() == Code::InvalidArgument);
  CHECK(grid->grid().num_buckets == few.num_buckets);

  vkc::Result<vkc::HostSlot> slot = dec->claim_input(f.small.size());
  CHECK(slot.ok());
  std::memcpy(slot->data(), f.small.data(), f.small.size());
  vkc::StageMetrics metrics;
  CHECK(codec::decode_growing(*dec, *slot, f.small.size(), *grid, kMaxBlocks,
                              &metrics)
            .ok());
  CHECK(grid->grid().num_buckets == decode_buckets(f.small_blocks));
  CHECK(has_row(metrics, "codec decode") && has_row(metrics, "  ..grow"));
  CHECK(codec::decode_growing(*bytes_dec, f.small.data(), f.small.size(),
                              *bytes_grid, kMaxBlocks)
            .ok());
  vkc::Result<Snapshot> got = snapshot(gpu, *grid);
  vkc::Result<Snapshot> want = snapshot(gpu, *bytes_grid);
  CHECK(got.ok() && want.ok() && got->coords.size() == f.small_blocks);
  CHECK(*got == *want);
  return 0;
}

// Two-slot buckets: one grow, to the frame's buckets, leaves half the slots
// the frame needs, and the decoder's second OutOfMemory is returned, ungrown
// past and untimed.
int second_out_of_memory_case(Gpu& gpu, const Frames& f) {
  GridShape pairs;
  pairs.bucket_size = 2;
  pairs.num_buckets = 16;
  vkc::Result<vol::VoxelBlockGrid> grid = make_grid(gpu, pairs);
  CHECK(grid.ok());
  vkc::Result<codec::Decoder> dec =
      codec::Decoder::create(gpu.device, gpu.allocator);
  CHECK(dec.ok());
  vkc::StageMetrics metrics;
  const vkc::Status s = codec::decode_growing(
      *dec, f.large.data(), f.large.size(), *grid, kMaxBlocks, &metrics);
  CHECK(s.domain() == Code::OutOfMemory);
  CHECK(s.message().find("Decoder::decode") != std::string::npos);
  CHECK(grid->grid().num_buckets == decode_buckets(f.large_blocks));
  CHECK(metrics.rows().empty());
  return 0;
}

// A grid the frames fit is never grown: an I-frame, timed, and a P-frame
// with no metrics, the default. A P-frame builds a grid as an I-frame does,
// though only a decoder holding its reference decodes it.
int fits_case(Gpu& gpu) {
  const Sphere a{vgrid::Vec3f(0.0f), 0.08f};
  const Sphere b{vgrid::Vec3f(0.02f, 0.0f, 0.0f), 0.08f};
  vkc::Result<vol::VoxelBlockGrid> ga = sphere_grid(gpu, a);
  vkc::Result<vol::VoxelBlockGrid> gb = sphere_grid(gpu, b);
  CHECK(ga.ok() && gb.ok());
  codec::EncoderConfig config;
  config.intra_period = 30;
  vkc::Result<codec::Encoder> enc =
      codec::Encoder::create(gpu.device, gpu.allocator, config);
  CHECK(enc.ok());
  vkc::Result<Bytes> intra = enc->encode(*ga);
  vkc::Result<Bytes> inter = enc->encode(*gb);
  CHECK(intra.ok() && inter.ok());
  CHECK(codec::read_frame_info(inter->data(), inter->size())->type ==
        codec::FrameType::kInter);
  vkc::Result<vol::VoxelBlockGrid> grid = codec::make_decode_grid(
      gpu.device, gpu.allocator, intra->data(), intra->size(), kMaxBlocks);
  vkc::Result<codec::Decoder> dec =
      codec::Decoder::create(gpu.device, gpu.allocator);
  CHECK(grid.ok() && dec.ok());
  const std::int32_t buckets = grid->grid().num_buckets;
  vkc::StageMetrics metrics;
  CHECK(codec::decode_growing(*dec, intra->data(), intra->size(), *grid,
                              kMaxBlocks, &metrics)
            .ok());
  CHECK(grid->grid().num_buckets == buckets);
  CHECK(has_row(metrics, "codec decode") && has_row(metrics, "  ..inverse"));
  CHECK(codec::decode_growing(*dec, inter->data(), inter->size(), *grid,
                              kMaxBlocks)
            .ok());
  CHECK(grid->grid().num_buckets == buckets);
  vkc::Result<vol::VoxelBlockGrid> from_inter = codec::make_decode_grid(
      gpu.device, gpu.allocator, inter->data(), inter->size(), kMaxBlocks);
  vkc::Result<codec::FrameInfo> inter_info =
      codec::read_frame_info(inter->data(), inter->size());
  CHECK(from_inter.ok() && inter_info.ok());
  CHECK(from_inter->grid().num_buckets ==
        decode_buckets(inter_info->block_count));
  return 0;
}

// The decoder's refusals come back with their codes, the grid ungrown and
// empty and nothing timed: a P-frame without its reference (NotFound), a
// frame of other geometry and a cut one (InvalidArgument).
int refusals_case(Gpu& gpu, const Frames& f) {
  vkc::Result<vol::VoxelBlockGrid> ga =
      sphere_grid(gpu, Sphere{vgrid::Vec3f(0.0f), 0.08f});
  CHECK(ga.ok());
  codec::EncoderConfig config;
  config.intra_period = 30;
  vkc::Result<codec::Encoder> enc =
      codec::Encoder::create(gpu.device, gpu.allocator, config);
  CHECK(enc.ok());
  vkc::Result<Bytes> intra = enc->encode(*ga);
  vkc::Result<Bytes> inter = enc->encode(*ga);
  CHECK(intra.ok() && inter.ok());
  vkc::Result<vol::VoxelBlockGrid> grid = codec::make_decode_grid(
      gpu.device, gpu.allocator, intra->data(), intra->size(), kMaxBlocks);
  vkc::Result<codec::Decoder> dec =
      codec::Decoder::create(gpu.device, gpu.allocator);
  CHECK(grid.ok() && dec.ok());
  struct Refused {
    const Bytes* frame;
    std::size_t size;
    Code code;
  };
  const Refused refusals[] = {
      {&*inter, inter->size(), Code::NotFound},
      {&f.small, f.small.size(), Code::InvalidArgument},
      {&*intra, intra->size() - 1, Code::InvalidArgument},
  };
  const std::int32_t buckets = grid->grid().num_buckets;
  for (const Refused& r : refusals) {
    vkc::StageMetrics metrics;
    const vkc::Status s = codec::decode_growing(*dec, r.frame->data(), r.size,
                                                *grid, kMaxBlocks, &metrics);
    CHECK(s.domain() == r.code);
    CHECK(grid->grid().num_buckets == buckets && metrics.rows().empty());
    vkc::Result<std::vector<vol::BlockIndex>> active = active_sorted(*grid);
    CHECK(active.ok() && active->empty());
  }
  return 0;
}

}  // namespace

int gpu_main(vkc::test::GpuContext& gpu) {
  vkc::Result<Frames> frames = make_frames(gpu);
  CHECK(frames.ok());
  if (make_grid_case(gpu, *frames) != 0) return 1;
  if (grows_once_case(gpu, *frames) != 0) return 1;
  if (slot_case(gpu, *frames) != 0) return 1;
  if (second_out_of_memory_case(gpu, *frames) != 0) return 1;
  if (fits_case(gpu) != 0) return 1;
  if (ceiling_case(gpu, *frames) != 0) return 1;
  if (refusals_case(gpu, *frames) != 0) return 1;
  std::printf("codec decode grid: OK\n");
  return 0;
}

int main() {
  return vkc::test::run_on_gpu(vgrid::device_requirements(), gpu_main);
}
