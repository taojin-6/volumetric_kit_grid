// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// GPU test for codec::Encoder: an empty grid is a valid frame; the bytes are
// independent of the order the hash table holds the blocks in; blocks with no
// observed voxel are left out; its stage rows; its refusals and its moves.
// Decoding is codec_decoder_test's. Exits 0 (skip) with no device.

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <utility>
#include <vector>

#include "codec_fixture.hpp"
#include "volumetric_kit/recon/codec/decoder.hpp"
#include "volumetric_kit/recon/codec/encoder.hpp"
#include "volumetric_kit/recon/core/gpu_timer.hpp"
#include "volumetric_kit/recon/core/stage_metrics.hpp"

namespace codec = volumetric_kit::recon::codec;
using namespace codec_fixture;

namespace {

const vr::StageRow* find_row(const vr::StageMetrics& m, const char* name) {
  for (const vr::StageRow& row : m.rows()) {
    if (std::strcmp(row.name, name) == 0) {
      return &row;
    }
  }
  return nullptr;
}

int empty_grid_case(Gpu& gpu, codec::Encoder& enc) {
  vr::Result<vol::VoxelBlockGrid> g = make_grid(gpu);
  CHECK(g.ok());
  vol::VoxelBlockGrid grid = std::move(g).value();
  vr::Result<std::vector<std::uint8_t>> bytes = enc.encode(grid);
  CHECK(bytes.ok());
  vr::Result<codec::FrameInfo> info =
      codec::read_frame_info(bytes.value().data(), bytes.value().size());
  CHECK(info.ok());
  CHECK(info.value().block_count == 0);
  CHECK(info.value().voxel_size == kVoxel);
  CHECK(info.value().trunc_dist == kTrunc);
  CHECK(info.value().params.coefficient_count ==
        enc.config().params.coefficient_count);
  return 0;
}

// The same content in two grids whose hash tables hold it differently -- the
// blocks allocated in opposite orders into tables of different sizes, so
// every ptr and the compaction order differ -- encodes to the same bytes.
int order_independent_case(Gpu& gpu, codec::Encoder& enc) {
  const Sphere s{vr::Vec3f(0.01f, -0.02f, 0.03f), 0.09f};
  std::vector<vr::Vec3i> coords = band_blocks(s);
  CHECK(coords.size() > 100);

  vr::Result<vol::VoxelBlockGrid> ga = make_grid(gpu);
  CHECK(ga.ok());
  vol::VoxelBlockGrid a = std::move(ga).value();
  CHECK(allocate(a, coords).ok());
  CHECK(write_sphere(gpu, a, s).ok());

  GridShape bigger;
  bigger.num_buckets = 2048;
  vr::Result<vol::VoxelBlockGrid> gb = make_grid(gpu, bigger);
  CHECK(gb.ok());
  vol::VoxelBlockGrid b = std::move(gb).value();
  std::vector<vr::Vec3i> reversed(coords.rbegin(), coords.rend());
  CHECK(allocate(b, reversed).ok());
  CHECK(write_sphere(gpu, b, s).ok());

  vr::Result<std::vector<vol::BlockIndex>> pa = a.map().compact_active_blocks();
  vr::Result<std::vector<vol::BlockIndex>> pb = b.map().compact_active_blocks();
  CHECK(pa.ok() && pb.ok());
  CHECK(pa.value().size() == pb.value().size());
  bool any_ptr_differs = false;
  for (std::size_t i = 0; i < pa.value().size(); ++i) {
    any_ptr_differs = any_ptr_differs ||
                      !(pa.value()[i].coord == pb.value()[i].coord) ||
                      pa.value()[i].ptr != pb.value()[i].ptr;
  }
  CHECK(any_ptr_differs);  // the premise: the tables really do differ

  vr::Result<std::vector<std::uint8_t>> fa = enc.encode(a);
  vr::Result<std::vector<std::uint8_t>> fb = enc.encode(b);
  CHECK(fa.ok() && fb.ok());
  CHECK(fa.value() == fb.value());
  // And encoding is repeatable.
  vr::Result<std::vector<std::uint8_t>> again = enc.encode(a);
  CHECK(again.ok() && again.value() == fa.value());
  return 0;
}

// A block with no observed voxel is left out of the frame.
int unobserved_dropped_case(Gpu& gpu, codec::Encoder& enc) {
  const Sphere s{vr::Vec3f(0.0f, 0.0f, 0.0f), 0.06f};
  const std::vector<vr::Vec3i> far = {{40, 40, 40}, {41, 40, 40}, {-40, 3, 9}};
  vr::Result<vol::VoxelBlockGrid> g = sphere_grid(gpu, s, {}, far);
  CHECK(g.ok());
  vol::VoxelBlockGrid grid = std::move(g).value();

  // Which blocks have an observed voxel, counted on the host.
  vr::Result<Snapshot> snap = snapshot(gpu, grid);
  CHECK(snap.ok());
  std::uint32_t observed_blocks = 0;
  for (std::size_t i = 0; i < snap.value().coords.size(); ++i) {
    bool any = false;
    for (std::uint32_t v = 0; v < kVpb; ++v) {
      any = any || snap.value().weight[i * kVpb + v] >= vol::kObservedWeight;
    }
    observed_blocks += any ? 1u : 0u;
  }
  const std::size_t active = snap.value().coords.size();
  CHECK(observed_blocks + far.size() <= active);  // the far ones at least

  vr::Result<std::vector<std::uint8_t>> bytes = enc.encode(grid);
  CHECK(bytes.ok());
  vr::Result<codec::FrameInfo> info =
      codec::read_frame_info(bytes.value().data(), bytes.value().size());
  CHECK(info.ok());
  CHECK(info.value().block_count == observed_blocks);
  CHECK(info.value().block_count < active);
  return 0;
}

// The rows the header promises, and the device half on the parent row where
// the device can time at all.
int metrics_case(Gpu& gpu, codec::Encoder& enc) {
  const Sphere s{vr::Vec3f(0.0f), 0.08f};
  vr::Result<vol::VoxelBlockGrid> g = sphere_grid(gpu, s);
  CHECK(g.ok());
  vol::VoxelBlockGrid grid = std::move(g).value();
  vr::StageMetrics m;
  CHECK(enc.encode(grid, &m).ok());
  const vr::StageRow* top = find_row(m, "codec encode");
  CHECK(top != nullptr);
  for (const char* sub : {"  ..active set", "  ..observed", "  ..sort",
                          "  ..forward", "  ..rans encode"}) {
    const vr::StageRow* row = find_row(m, sub);
    CHECK(row != nullptr);
    CHECK(row->cpu_ms <= top->cpu_ms);
  }
  // Breakdown rows are inside the parent, so the host total is the parent.
  CHECK(m.total_cpu_ms() == top->cpu_ms);
  vr::Result<vr::GpuTimer> probe = vr::GpuTimer::create(gpu.device);
  CHECK(probe.ok());
  if (probe.value().available()) {
    CHECK(top->has_gpu);
    CHECK(top->gpu_ms > 0.0 && top->gpu_ms < top->cpu_ms);
  }
  // A refused call still costs its row.
  vr::StageMetrics refused;
  vol::VoxelBlockGrid moved = std::move(grid);
  CHECK(!enc.encode(grid, &refused).ok());  // NOLINT(bugprone-use-after-move)
  CHECK(find_row(refused, "codec encode") != nullptr);
  return 0;
}

// The active set is the map's own list: a list a fuse left that still holds
// is taken back, compacting nothing, and is still the map's for the extract
// after -- and one the encoder compacts is left for it too.
int device_list_case(Gpu& gpu, codec::Encoder& enc) {
  vr::Result<vol::VoxelBlockGrid> g =
      sphere_grid(gpu, Sphere{vr::Vec3f(0.0f), 0.07f});
  CHECK(g.ok());
  vol::VoxelBlockGrid grid = std::move(g).value();
  vr::Result<vol::DeviceBlockList> fused =
      grid.map().compact_active_blocks_on_device();
  CHECK(fused.ok());
  vr::StageMetrics m;
  vr::Result<std::vector<std::uint8_t>> a = enc.encode(grid, &m);
  CHECK(a.ok());
  CHECK(find_row(m, "  ..active set") == nullptr);
  CHECK(grid.map().check_device_block_list(fused.value(), "test").ok());

  // A list the encoder compacted itself, after an allocation, holds for the
  // next caller, and the content -- one block more, never observed -- codes
  // the same.
  CHECK(allocate(grid, {{50, 50, 50}}).ok());
  vr::StageMetrics again;
  vr::Result<std::vector<std::uint8_t>> b = enc.encode(grid, &again);
  CHECK(b.ok() && b.value() == a.value());
  CHECK(find_row(again, "  ..active set") != nullptr);
  vr::StageMetrics extract;  // the extract's compaction, which it skips
  vr::Result<vol::DeviceBlockList> next =
      grid.map().compact_active_blocks_on_device(&extract);
  CHECK(next.ok() && extract.rows().empty());
  CHECK(next.value().serial != fused.value().serial);
  return 0;
}

int refusals_case(Gpu& gpu) {
  codec::EncoderConfig bad;
  bad.segment_size = 0;
  CHECK(!codec::Encoder::create(gpu.device, gpu.allocator, bad).ok());
  bad = codec::EncoderConfig{};
  bad.params.coefficient_count = 0;
  CHECK(!codec::Encoder::create(gpu.device, gpu.allocator, bad).ok());
  bad = codec::EncoderConfig{};
  bad.params.ac_step = 0.0f;
  CHECK(!codec::Encoder::create(gpu.device, gpu.allocator, bad).ok());

  vr::Result<codec::Encoder> e =
      codec::Encoder::create(gpu.device, gpu.allocator);
  CHECK(e.ok());
  codec::Encoder enc = std::move(e).value();
  // Another block size, and no weight attribute.
  GridShape four;
  four.block_size = 4;
  vr::Result<vol::VoxelBlockGrid> g4 = make_grid(gpu, four);
  CHECK(g4.ok());
  CHECK(!enc.encode(g4.value()).ok());
  GridShape no_weight;
  no_weight.weight = false;
  vr::Result<vol::VoxelBlockGrid> gw = make_grid(gpu, no_weight);
  CHECK(gw.ok());
  CHECK(!enc.encode(gw.value()).ok());
  return 0;
}

int moves_case(Gpu& gpu) {
  vr::Result<codec::Encoder> a_r =
      codec::Encoder::create(gpu.device, gpu.allocator);
  CHECK(a_r.ok());
  codec::Encoder a = std::move(a_r).value();
  CHECK(a.valid());
  codec::Encoder b(std::move(a));
  CHECK(b.valid());
  CHECK(!a.valid());  // NOLINT(bugprone-use-after-move): asserting the source
  // Its configuration went with it: nothing a moved-from encoder reports
  // looks like one that could encode.
  CHECK(a.config().params.coefficient_count == 0);  // NOLINT
  CHECK(a.config().segment_size == 0);              // NOLINT
  CHECK(!a.config().params.validate().ok());        // NOLINT

  codec::EncoderConfig other;
  other.params.coefficient_count = 8;
  vr::Result<codec::Encoder> c_r =
      codec::Encoder::create(gpu.device, gpu.allocator, other);
  CHECK(c_r.ok());
  codec::Encoder c = std::move(c_r).value();
  c = std::move(b);  // over a live encoder: takes b's config too
  CHECK(c.valid());
  CHECK(!b.valid());  // NOLINT(bugprone-use-after-move)
  CHECK(c.config().params.coefficient_count ==
        codec::CodecParams{}.coefficient_count);
  CHECK(b.config().segment_size == 0);  // NOLINT(bugprone-use-after-move)

  codec::Encoder* alias = &c;
  c = std::move(*alias);  // self-move, laundered past -Wself-move
  CHECK(c.valid());

  vr::Result<vol::VoxelBlockGrid> g = make_grid(gpu);
  CHECK(g.ok());
  CHECK(c.encode(g.value()).ok());
  CHECK(!a.encode(g.value()).ok());  // a moved-from encoder refuses
  return 0;
}

}  // namespace

int main() {
  vr::Result<vr::Instance> instance = vr::Instance::create({});
  if (!instance) {
    std::fprintf(stderr, "no Vulkan instance (%s); skipping\n",
                 instance.status().message().c_str());
    return 0;
  }
  vr::Result<VkPhysicalDevice> physical =
      instance.value().select_physical_device();
  if (!physical) {
    std::fprintf(stderr, "no compute-capable device (%s); skipping\n",
                 physical.status().message().c_str());
    return 0;
  }
  vr::Result<vr::Device> device =
      vr::Device::create(instance.value(), physical.value(), {});
  CHECK(device.ok());
  vr::Result<vr::Allocator> allocator =
      vr::Allocator::create(instance.value().handle(), device.value());
  CHECK(allocator.ok());
  Gpu gpu{device.value(), allocator.value()};

  vr::Result<codec::Encoder> e =
      codec::Encoder::create(gpu.device, gpu.allocator);
  CHECK(e.ok());
  codec::Encoder enc = std::move(e).value();

  if (empty_grid_case(gpu, enc) != 0) return 1;
  if (order_independent_case(gpu, enc) != 0) return 1;
  if (unobserved_dropped_case(gpu, enc) != 0) return 1;
  if (metrics_case(gpu, enc) != 0) return 1;
  if (device_list_case(gpu, enc) != 0) return 1;
  if (refusals_case(gpu) != 0) return 1;
  if (moves_case(gpu) != 0) return 1;
  std::printf("codec Encoder: OK\n");
  return 0;
}
