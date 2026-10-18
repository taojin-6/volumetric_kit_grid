// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// GPU test for codec/decode_grid.hpp, on sphere grids: make_decode_grid's
// geometry, layout and attributes off a frame's header, at the minimum bucket
// count and past it, and the headers it refuses; decode_growing leaving a grid
// the frame fits alone, growing one too small exactly once to decode what a
// roomy grid decodes, returning the second OutOfMemory of a grid one grow
// leaves too small, timing only the attempt that succeeds, and passing the
// decoder's refusals through with their codes, ungrown. Skips with no device.

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "codec_fixture.hpp"
#include "gpu_test.hpp"
#include "volumetric_kit/core/base/stage_metrics.hpp"
#include "volumetric_kit/recon/codec/decode_grid.hpp"
#include "volumetric_kit/recon/codec/decoder.hpp"
#include "volumetric_kit/recon/codec/encoder.hpp"

namespace codec = volumetric_kit::recon::codec;
using namespace codec_fixture;

namespace {

using Bytes = std::vector<std::uint8_t>;
using Code = vkc::Status::Code;

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
      sphere_grid(gpu, Sphere{vr::Vec3f(0.01f, 0.0f, -0.02f), 0.1f}, own));
  VKC_ASSIGN(f.small, encode(gpu, small));
  GridShape roomy;
  roomy.num_buckets = 2048;
  VKC_ASSIGN(vol::VoxelBlockGrid large,
             sphere_grid(gpu, Sphere{vr::Vec3f(0.0f), 0.6f}, roomy));
  VKC_ASSIGN(f.large, encode(gpu, large));
  VKC_ASSIGN(const codec::FrameInfo s,
             codec::read_frame_info(f.small.data(), f.small.size()));
  VKC_ASSIGN(const codec::FrameInfo l,
             codec::read_frame_info(f.large.data(), f.large.size()));
  f.small_blocks = s.block_count;
  f.large_blocks = l.block_count;
  return f;
}

// The buckets make_decode_grid documents: twice the blocks, at least 1024.
std::int32_t decode_buckets(std::uint32_t blocks) {
  return std::max(std::int32_t((2 * std::int64_t(blocks) + 7) / 8), 1024);
}

bool has_row(const vkc::StageMetrics& m, const char* name) {
  for (const vkc::StageRow& row : m.rows()) {
    if (std::strcmp(row.name, name) == 0) return true;
  }
  return false;
}

// The grid from a frame's header: its geometry exactly, 8^3 blocks, buckets
// of 8 with a slot each and chains of 128, twice the blocks' buckets at
// least 1024, float tsdf and weight alone, and no block yet; a frame decodes
// into it.
int make_grid_case(Gpu& gpu, const Frames& f) {
  CHECK(f.small_blocks > 32 && f.small_blocks < 4096);
  CHECK(f.large_blocks > 4096);
  struct Want {
    const Bytes* frame;
    float voxel;
    float trunc;
    std::int32_t buckets;
  };
  const Want wants[] = {
      {&f.small, kSmallVoxel, kSmallTrunc, 1024},
      {&f.large, kVoxel, kTrunc, std::int32_t((2 * f.large_blocks + 7) / 8)},
  };
  for (const Want& want : wants) {
    vkc::Result<vol::VoxelBlockGrid> grid = codec::make_decode_grid(
        gpu.device, gpu.allocator, want.frame->data(), want.frame->size());
    CHECK(grid.ok());
    const vol::VoxelGridParams& gp = grid->grid();
    CHECK(gp.voxel_size == want.voxel && gp.trunc_dist == want.trunc);
    CHECK(gp.block_size == 8 && gp.voxels_per_block == 512);
    CHECK(gp.bucket_size == 8 && gp.num_buckets == want.buckets);
    CHECK(gp.num_blocks == 8 * want.buckets && gp.max_chain == 128);
    CHECK(grid->attribute_count() == 2);
    for (const char* name : {"tsdf", "weight"}) {
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
  const vkc::Result<vol::VoxelBlockGrid> cut =
      codec::make_decode_grid(gpu.device, gpu.allocator, f.small.data(), 10);
  CHECK(cut.status().domain() == Code::InvalidArgument);
  // A header claiming more blocks than any grid holds (block_count, at byte
  // 32) is refused, not built.
  Bytes most = f.small;
  const std::uint32_t all = 0xFFFFFFFFu;
  std::memcpy(most.data() + 32, &all, sizeof all);
  const vkc::Result<vol::VoxelBlockGrid> huge = codec::make_decode_grid(
      gpu.device, gpu.allocator, most.data(), most.size());
  CHECK(huge.status().domain() == Code::InvalidArgument);
  return 0;
}

// A grid too small for the frame (32 slots) grows once, to the frame's
// buckets (not twice its own, nor more), and holds what a decode into a
// roomy grid holds; only that successful decode is timed.
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
                              &metrics)
            .ok());
  CHECK(grid->grid().num_buckets == decode_buckets(f.small_blocks));
  CHECK(has_row(metrics, "codec decode") && has_row(metrics, "  ..inverse"));

  vkc::Result<vol::VoxelBlockGrid> roomy = codec::make_decode_grid(
      gpu.device, gpu.allocator, f.small.data(), f.small.size());
  CHECK(roomy.ok());
  CHECK(dec->decode(f.small.data(), f.small.size(), *roomy).ok());
  vkc::Result<Snapshot> got = snapshot(gpu, *grid);
  vkc::Result<Snapshot> want = snapshot(gpu, *roomy);
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
  const vkc::Status s = codec::decode_growing(*dec, f.large.data(),
                                              f.large.size(), *grid, &metrics);
  CHECK(s.domain() == Code::OutOfMemory);
  CHECK(s.message().find("Decoder::decode") != std::string::npos);
  CHECK(grid->grid().num_buckets == decode_buckets(f.large_blocks));
  CHECK(metrics.rows().empty());
  return 0;
}

// A grid the frames fit is never grown: an I-frame, timed, and a P-frame
// with no metrics, the default.
int fits_case(Gpu& gpu) {
  const Sphere a{vr::Vec3f(0.0f), 0.08f};
  const Sphere b{vr::Vec3f(0.02f, 0.0f, 0.0f), 0.08f};
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
      gpu.device, gpu.allocator, intra->data(), intra->size());
  vkc::Result<codec::Decoder> dec =
      codec::Decoder::create(gpu.device, gpu.allocator);
  CHECK(grid.ok() && dec.ok());
  const std::int32_t buckets = grid->grid().num_buckets;
  vkc::StageMetrics metrics;
  CHECK(
      codec::decode_growing(*dec, intra->data(), intra->size(), *grid, &metrics)
          .ok());
  CHECK(grid->grid().num_buckets == buckets);
  CHECK(has_row(metrics, "codec decode") && has_row(metrics, "  ..inverse"));
  CHECK(codec::decode_growing(*dec, inter->data(), inter->size(), *grid).ok());
  CHECK(grid->grid().num_buckets == buckets);
  return 0;
}

// The decoder's refusals come back with their codes, the grid ungrown and
// empty and nothing timed: a P-frame without its reference (NotFound), a
// frame of other geometry and a cut one (InvalidArgument).
int refusals_case(Gpu& gpu, const Frames& f) {
  vkc::Result<vol::VoxelBlockGrid> ga =
      sphere_grid(gpu, Sphere{vr::Vec3f(0.0f), 0.08f});
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
      gpu.device, gpu.allocator, intra->data(), intra->size());
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
    const vkc::Status s =
        codec::decode_growing(*dec, r.frame->data(), r.size, *grid, &metrics);
    CHECK(s.domain() == r.code);
    CHECK(grid->grid().num_buckets == buckets && metrics.rows().empty());
    vkc::Result<std::vector<vol::BlockIndex>> active = active_sorted(*grid);
    CHECK(active.ok() && active->empty());
  }
  return 0;
}

}  // namespace

int gpu_main(vr_test::GpuContext& gpu) {
  vkc::Result<Frames> frames = make_frames(gpu);
  CHECK(frames.ok());
  if (make_grid_case(gpu, *frames) != 0) return 1;
  if (grows_once_case(gpu, *frames) != 0) return 1;
  if (second_out_of_memory_case(gpu, *frames) != 0) return 1;
  if (fits_case(gpu) != 0) return 1;
  if (refusals_case(gpu, *frames) != 0) return 1;
  std::printf("codec decode grid: OK\n");
  return 0;
}

int main() { return vr_test::run_on_gpu(gpu_main); }
