// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// GPU test for codec::Encoder: an empty grid is a valid frame; the bytes are
// the CPU oracle's -- its blocks sorted on the host -- blocks at int32's
// extremes included, and independent of the order the hash table holds the
// blocks in; blocks with no observed voxel are left out; its stage rows; its
// refusals (a segment or update segment longer than the format allows and an
// intra period of 0 among them, each as EncoderConfig::validate refuses it)
// and its moves; and its frame types -- an I-frame first, every period,
// after force_intra(), on a geometry change, for a grid of no blocks and
// after one, a P-frame otherwise -- a refusal keeping the reference and the
// period's count, and a chain repeated by a fresh encoder. With
// EncoderConfig::true_sdf: the oracle's bytes, its blocks and cut set found
// by the host rule -- cuts across block faces, partial masks, blocks the near
// rule alone keeps, and none of a true distance cut -- the same bytes
// whatever the hash table's order or heap, its scratch rebound as the heap
// grows and cleared between frames, and the same rows. Decoding is
// codec_decoder_test's, and a P-frame's bytes codec_inter_test's. The two
// halves, each batch the caller's, give encode()'s bytes -- I and P, at
// several K, masked and true-SDF, and on a grid of far more slots than
// blocks -- a P-frame over nothing finishes as an I-frame of none with no
// back batch, a token is the encoder's last record only, abandon keeps the
// chain, a copied payload past its guess takes a tail to the same bytes, and
// a warm encode() is 2 submits and no staging buffer. encode() and the front
// half take the grid read-only, the map's own list or one compacted into the
// front's batch alike. Skips with no device.

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "codec_fixture.hpp"
#include "codec_oracle_encode.hpp"
#include "grid_readback.hpp"
#include "volumetric_kit/core/base/stage_metrics.hpp"
#include "volumetric_kit/core/vulkan/gpu_timer.hpp"
#include "volumetric_kit/recon/codec/decode_grid.hpp"
#include "volumetric_kit/recon/codec/decoder.hpp"
#include "volumetric_kit/recon/codec/encoder.hpp"
#include "volumetric_kit/recon/volume/grid_growth.hpp"

#include "volumetric_kit/core/testing/gpu_test.hpp"
#include "volumetric_kit/recon/core/device_requirements.hpp"

namespace vr = volumetric_kit::recon;
namespace vkc = volumetric_kit::core;

namespace codec = volumetric_kit::recon::codec;
using namespace codec_fixture;

namespace {

using Bytes = std::vector<std::uint8_t>;

const vkc::StageRow* find_row(const vkc::StageMetrics& m, const char* name) {
  for (const vkc::StageRow& row : m.rows()) {
    if (std::strcmp(row.name, name) == 0) {
      return &row;
    }
  }
  return nullptr;
}

int empty_grid_case(Gpu& gpu, codec::Encoder& enc) {
  vkc::Result<vol::VoxelBlockGrid> g = make_grid(gpu);
  CHECK(g.ok());
  vol::VoxelBlockGrid grid = std::move(g).value();
  vkc::Result<std::vector<std::uint8_t>> bytes = enc.encode(grid);
  CHECK(bytes.ok());
  CHECK(bytes.value() == codec_oracle::encode(gpu, grid, {}).value());
  vkc::Result<codec::FrameInfo> info =
      codec::read_frame_info(bytes.value().data(), bytes.value().size());
  CHECK(info.ok());
  CHECK(info.value().block_count == 0);
  CHECK(info.value().voxel_size == kVoxel);
  CHECK(info.value().trunc_dist == kTrunc);
  CHECK(info.value().params.coefficient_count ==
        codec::CodecParams{}.coefficient_count);
  return 0;
}

// The same content in two grids whose hash tables hold it differently -- the
// blocks allocated in opposite orders into tables of different sizes, so
// every ptr and the compaction order differ -- encodes to the same bytes.
int order_independent_case(Gpu& gpu, codec::Encoder& enc) {
  const Sphere s{vr::Vec3f(0.01f, -0.02f, 0.03f), 0.09f};
  std::vector<vr::Vec3i> coords = band_blocks(s);
  CHECK(coords.size() > 100);

  vkc::Result<vol::VoxelBlockGrid> ga = make_grid(gpu);
  CHECK(ga.ok());
  vol::VoxelBlockGrid a = std::move(ga).value();
  CHECK(allocate(gpu, a, coords).ok());
  CHECK(write_sphere(gpu, a, s).ok());

  GridShape bigger;
  bigger.num_buckets = 2048;
  vkc::Result<vol::VoxelBlockGrid> gb = make_grid(gpu, bigger);
  CHECK(gb.ok());
  vol::VoxelBlockGrid b = std::move(gb).value();
  std::vector<vr::Vec3i> reversed(coords.rbegin(), coords.rend());
  CHECK(allocate(gpu, b, reversed).ok());
  CHECK(write_sphere(gpu, b, s).ok());

  vkc::Result<std::vector<vol::BlockIndex>> pa =
      a.map().compact_active_blocks();
  vkc::Result<std::vector<vol::BlockIndex>> pb =
      b.map().compact_active_blocks();
  CHECK(pa.ok() && pb.ok());
  CHECK(pa.value().size() == pb.value().size());
  bool any_ptr_differs = false;
  for (std::size_t i = 0; i < pa.value().size(); ++i) {
    any_ptr_differs = any_ptr_differs ||
                      !(pa.value()[i].coord == pb.value()[i].coord) ||
                      pa.value()[i].ptr != pb.value()[i].ptr;
  }
  CHECK(any_ptr_differs);  // the premise: the tables really do differ

  vkc::Result<std::vector<std::uint8_t>> fa = enc.encode(a);
  vkc::Result<std::vector<std::uint8_t>> fb = enc.encode(b);
  CHECK(fa.ok() && fb.ok());
  CHECK(fa.value() == fb.value());
  // And encoding is repeatable.
  vkc::Result<std::vector<std::uint8_t>> again = enc.encode(a);
  CHECK(again.ok() && again.value() == fa.value());
  return 0;
}

// A block with no observed voxel is left out of the frame.
int unobserved_dropped_case(Gpu& gpu, codec::Encoder& enc) {
  const Sphere s{vr::Vec3f(0.0f, 0.0f, 0.0f), 0.06f};
  const std::vector<vr::Vec3i> far = {{40, 40, 40}, {41, 40, 40}, {-40, 3, 9}};
  vkc::Result<vol::VoxelBlockGrid> g = sphere_grid(gpu, s, {}, far);
  CHECK(g.ok());
  vol::VoxelBlockGrid grid = std::move(g).value();

  // Which blocks have an observed voxel, counted on the host.
  vkc::Result<Snapshot> snap = snapshot(gpu, grid);
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

  vkc::Result<std::vector<std::uint8_t>> bytes = enc.encode(grid);
  CHECK(bytes.ok());
  vkc::Result<codec::FrameInfo> info =
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
  vkc::Result<vol::VoxelBlockGrid> g = sphere_grid(gpu, s);
  CHECK(g.ok());
  vol::VoxelBlockGrid grid = std::move(g).value();
  vkc::StageMetrics m;
  CHECK(enc.encode(grid, &m).ok());
  const vkc::StageRow* top = find_row(m, "codec encode");
  CHECK(top != nullptr);
  for (const char* sub :
       {"  ..encode set", "  ..observed", "  ..forward", "  ..rans encode"}) {
    const vkc::StageRow* row = find_row(m, sub);
    CHECK(row != nullptr);
    CHECK(row->cpu_ms <= top->cpu_ms);
  }
  // Breakdown rows are inside the parent, so the host total is the parent.
  CHECK(m.total_cpu_ms() == top->cpu_ms);
  vkc::Result<vkc::GpuTimer> probe = vkc::GpuTimer::create(gpu.device);
  CHECK(probe.ok());
  if (probe.value().available()) {
    CHECK(top->has_gpu);
    CHECK(top->gpu_ms > 0.0 && top->gpu_ms < top->cpu_ms);
  }
  // A refused call still costs its row.
  vkc::StageMetrics refused;
  vol::VoxelBlockGrid moved = std::move(grid);
  CHECK(!enc.encode(grid, &refused).ok());  // NOLINT(bugprone-use-after-move)
  CHECK(find_row(refused, "codec encode") != nullptr);

  // true_sdf's passes are inside "  ..observed": the same rows.
  codec::EncoderConfig config;
  config.true_sdf = true;
  vkc::Result<codec::Encoder> t =
      codec::Encoder::create(gpu.device, gpu.allocator, config);
  vkc::Result<vol::VoxelBlockGrid> wide = sphere_grid(gpu, s, wide_band());
  CHECK(t.ok() && wide.ok());
  vkc::StageMetrics tm;
  CHECK(t.value().encode(wide.value(), &tm).ok());
  CHECK(tm.rows().size() == m.rows().size());
  for (std::size_t i = 0; i < m.rows().size(); ++i) {
    CHECK(std::strcmp(tm.rows()[i].name, m.rows()[i].name) == 0);
  }
  return 0;
}

// The active set is the map's own list: the list the map last compacted
// (an extract's, say) is taken back while it still holds, compacting
// nothing, and is still the map's for the extract after. One the encoder
// compacts into its batch leaves the map no count to hold, so the extract
// after compacts its own. Both encodes are handed the grid read-only.
int device_list_case(Gpu& gpu, codec::Encoder& enc) {
  vkc::Result<vol::VoxelBlockGrid> g =
      sphere_grid(gpu, Sphere{vr::Vec3f(0.0f), 0.07f});
  CHECK(g.ok());
  vol::VoxelBlockGrid grid = std::move(g).value();
  const vol::VoxelBlockGrid& view = grid;
  vkc::Result<vol::DeviceBlockList> last =
      grid.map().compact_active_blocks_on_device();
  CHECK(last.ok());
  vkc::StageMetrics m;
  vkc::Result<std::vector<std::uint8_t>> a = enc.encode(view, &m);
  CHECK(a.ok());
  CHECK(find_row(m, "  ..encode set") == nullptr);
  CHECK(grid.map().check_device_block_list(last.value(), "test").ok());

  // After an allocation the encoder compacts into its batch, and the content
  // -- one block more, never observed -- codes the same; the map holds no
  // list after it, so a compaction after it compacts.
  CHECK(allocate(gpu, grid, {{50, 50, 50}}).ok());
  vkc::StageMetrics again;
  vkc::Result<std::vector<std::uint8_t>> b = enc.encode(view, &again);
  CHECK(b.ok() && b.value() == a.value());
  CHECK(find_row(again, "  ..encode set") != nullptr);
  CHECK(!grid.map().held_device_list().has_value());
  vkc::StageMetrics extract;
  vkc::Result<vol::DeviceBlockList> next =
      grid.map().compact_active_blocks_on_device(&extract);
  CHECK(next.ok() && !extract.rows().empty());
  CHECK(next.value().serial != last.value().serial);
  const std::optional<vol::DeviceBlockList> held =
      grid.map().held_device_list();
  CHECK(held.has_value() && held->serial == next.value().serial);
  return 0;
}

// The Encoder writes the CPU oracle's bytes for a fused-looking grid, at an
// odd K, for the default and a finer segmentation and the longest segments.
int oracle_case(Gpu& gpu) {
  const Sphere s{vr::Vec3f(0.01f, -0.02f, 0.03f), 0.09f};
  vkc::Result<vol::VoxelBlockGrid> g = sphere_grid(gpu, s);
  CHECK(g.ok());
  vol::VoxelBlockGrid grid = std::move(g).value();
  for (std::uint32_t segment_size :
       {64u, 3u, codec::kMaxDeviceDecodeSegmentSize}) {
    codec::EncoderConfig config;
    config.params.coefficient_count = 37;
    config.segment_size = segment_size;
    vkc::Result<codec::Encoder> enc =
        codec::Encoder::create(gpu.device, gpu.allocator, config);
    CHECK(enc.ok());
    vkc::Result<std::vector<std::uint8_t>> frame = enc.value().encode(grid);
    vkc::Result<std::vector<std::uint8_t>> oracle =
        codec_oracle::encode(gpu, grid, config);
    CHECK(frame.ok() && oracle.ok());
    CHECK(frame.value().size() > 1000);
    CHECK(frame.value() == oracle.value());
  }
  return 0;
}

// Blocks at int32's extremes on every axis, beside a sphere's: the device
// sort orders them as the oracle's std::sort does, and the frame's deltas span
// the whole range, so the bytes are still the oracle's -- with true_sdf too,
// whose neighbours there wrap around int32 on the device and in the oracle.
int extremes_case(Gpu& gpu, codec::Encoder& enc) {
  const Sphere s{vr::Vec3f(0.0f), 0.05f};
  vkc::Result<vol::VoxelBlockGrid> g = sphere_grid(gpu, s);
  CHECK(g.ok());
  vol::VoxelBlockGrid grid = std::move(g).value();
  constexpr std::int32_t lo = std::numeric_limits<std::int32_t>::min();
  constexpr std::int32_t hi = std::numeric_limits<std::int32_t>::max();
  const std::vector<vr::Vec3i> far = {
      {lo, 0, 0},   {hi, 0, 0},   {0, lo, 0},   {0, hi, 0},
      {0, 0, lo},   {0, 0, hi},   {lo, lo, lo}, {hi, hi, hi},
      {hi, lo, hi}, {lo, hi, lo}, {-1, hi, lo}, {lo + 1, -1, hi - 1}};
  CHECK(allocate(gpu, grid, far).ok());
  // Observed by hand, as write_sphere's block positions overflow here.
  vkc::Result<std::vector<vol::BlockIndex>> active =
      grid.map().compact_active_blocks();
  vkc::Result<std::vector<float>> tsdf = vr_test::read_attribute<float>(
      gpu.device, gpu.allocator, grid, vr::volume::kTsdfAttribute);
  vkc::Result<std::vector<float>> weight = vr_test::read_attribute<float>(
      gpu.device, gpu.allocator, grid, vr::volume::kWeightAttribute);
  CHECK(active.ok() && tsdf.ok() && weight.ok());
  std::size_t marked = 0;
  for (const vol::BlockIndex& b : active.value()) {
    if (std::find(far.begin(), far.end(), b.coord) == far.end()) continue;
    for (std::uint32_t v = 0; v < kVpb; ++v) {
      const bool obs = (v + std::uint32_t(b.ptr) / kVpb) % 3 != 0;
      tsdf.value()[std::uint32_t(b.ptr) + v] =
          obs ? kTrunc * (float(v % 17) / 8.0f - 1.0f) : 0.0f;
      weight.value()[std::uint32_t(b.ptr) + v] = obs ? 2.0f : 0.0f;
    }
    ++marked;
  }
  CHECK(marked == far.size());
  CHECK(vr_test::write_attribute(gpu.device, gpu.allocator, grid,
                                 vr::volume::kTsdfAttribute, tsdf.value())
            .ok());
  CHECK(vr_test::write_attribute(gpu.device, gpu.allocator, grid,
                                 vr::volume::kWeightAttribute, weight.value())
            .ok());
  vkc::Result<std::vector<std::uint8_t>> frame = enc.encode(grid);
  vkc::Result<std::vector<std::uint8_t>> oracle =
      codec_oracle::encode(gpu, grid, {});
  CHECK(frame.ok() && oracle.ok());
  CHECK(frame.value() == oracle.value());
  vkc::Result<codec::FrameInfo> info =
      codec::read_frame_info(frame.value().data(), frame.value().size());
  CHECK(info.ok() && info.value().block_count > far.size());
  codec::EncoderConfig config;
  config.true_sdf = true;
  vkc::Result<codec::Encoder> t =
      codec::Encoder::create(gpu.device, gpu.allocator, config);
  CHECK(t.ok());
  frame = t.value().encode(grid);
  oracle = codec_oracle::encode(gpu, grid, config);
  CHECK(frame.ok() && oracle.ok() && frame.value() == oracle.value());
  return 0;
}

// What EncoderConfig::validate refuses, create refuses, with the same
// message, naming the field: a segment size of 0, or a segment or update
// size the Decoder cannot decode; a segment size of 0 is named before the
// update size of 0 that would take it. The longest either can be is
// accepted, and an update size of 0, the segment size's.
int sizes_case(Gpu& gpu) {
  const std::uint32_t kLongest = codec::kMaxDeviceDecodeSegmentSize;
  const std::string over = std::to_string(kLongest + 1);
  struct Sizes {
    std::uint32_t segment_size;
    std::uint32_t update_size;
    std::string refusal;  // empty when accepted
  };
  for (const Sizes& c :
       {Sizes{0, 0, "segment_size 0"},
        Sizes{kLongest + 1, 0, "segment_size " + over}, Sizes{kLongest, 0, ""},
        Sizes{64, kLongest + 1, "update_size " + over}, Sizes{64, 0, ""},
        Sizes{64, kLongest, ""}}) {
    codec::EncoderConfig config;
    config.segment_size = c.segment_size;
    config.update_size = c.update_size;
    const vkc::Status checked = config.validate();
    const vkc::Result<codec::Encoder> made =
        codec::Encoder::create(gpu.device, gpu.allocator, config);
    CHECK(checked.ok() == made.ok());
    if (c.refusal.empty()) {
      CHECK(checked.ok());
      continue;
    }
    CHECK(checked.domain() == vkc::Status::Code::InvalidArgument &&
          made.status().message() == checked.message());
    CHECK(checked.message().find(c.refusal) != std::string::npos);
    CHECK(c.update_size != 0 ||
          checked.message().find("update_size") == std::string::npos);
  }
  CHECK(codec::EncoderConfig{}.validate().ok());
  return 0;
}

int refusals_case(Gpu& gpu) {
  codec::EncoderConfig bad;
  bad.params.coefficient_count = 0;
  CHECK(!codec::Encoder::create(gpu.device, gpu.allocator, bad).ok());
  bad = codec::EncoderConfig{};
  bad.params.quantization_weights[0] = 0.0f;
  CHECK(!codec::Encoder::create(gpu.device, gpu.allocator, bad).ok());

  vkc::Result<codec::Encoder> e =
      codec::Encoder::create(gpu.device, gpu.allocator);
  CHECK(e.ok());
  codec::Encoder enc = std::move(e).value();
  // Another block size, and no weight attribute.
  GridShape four;
  four.block_size = 4;
  vkc::Result<vol::VoxelBlockGrid> g4 = make_grid(gpu, four);
  CHECK(g4.ok());
  CHECK(!enc.encode(g4.value()).ok());
  GridShape no_weight;
  no_weight.weight = false;
  vkc::Result<vol::VoxelBlockGrid> gw = make_grid(gpu, no_weight);
  CHECK(gw.ok());
  CHECK(!enc.encode(gw.value()).ok());
  return 0;
}

int moves_case(Gpu& gpu) {
  vkc::Result<codec::Encoder> a_r =
      codec::Encoder::create(gpu.device, gpu.allocator);
  CHECK(a_r.ok());
  codec::Encoder a = std::move(a_r).value();
  CHECK(a.valid());
  codec::Encoder b(std::move(a));
  CHECK(b.valid());
  CHECK(!a.valid());  // NOLINT(bugprone-use-after-move): asserting the source

  codec::EncoderConfig other;
  other.params.coefficient_count = 8;
  vkc::Result<codec::Encoder> c_r =
      codec::Encoder::create(gpu.device, gpu.allocator, other);
  CHECK(c_r.ok());
  codec::Encoder c = std::move(c_r).value();
  c = std::move(b);  // over a live encoder: takes b's config too
  CHECK(c.valid());
  CHECK(!b.valid());  // NOLINT(bugprone-use-after-move)

  codec::Encoder* alias = &c;
  c = std::move(*alias);  // self-move, laundered past -Wself-move
  CHECK(c.valid());

  vkc::Result<vol::VoxelBlockGrid> g = make_grid(gpu);
  CHECK(g.ok());
  vkc::Result<std::vector<std::uint8_t>> bytes = c.encode(g.value());
  CHECK(bytes.ok());
  // b's configuration, not the one c was made with.
  vkc::Result<codec::FrameInfo> info =
      codec::read_frame_info(bytes.value().data(), bytes.value().size());
  CHECK(info.ok() && info.value().params.coefficient_count ==
                         codec::CodecParams{}.coefficient_count);
  CHECK(!a.encode(g.value()).ok());  // a moved-from encoder refuses
  return 0;
}

codec::FrameType type_of(const Bytes& frame) {
  return codec::read_frame_info(frame.data(), frame.size()).value().type;
}

// A sphere at step @p t of a chain: moving and growing.
Sphere chain_sphere(int t) {
  return Sphere{vr::Vec3f(0.008f * float(t), -0.004f * float(t), 0.0f),
                0.08f + 0.005f * float(t)};
}

// An intra period of 0 is refused; 1 and the longest are taken.
int period_case(Gpu& gpu) {
  codec::EncoderConfig config;
  config.intra_period = 0;
  const vkc::Result<codec::Encoder> none =
      codec::Encoder::create(gpu.device, gpu.allocator, config);
  CHECK(!none.ok() &&
        none.status().domain() == vkc::Status::Code::InvalidArgument &&
        none.status().message().find("intra_period 0") != std::string::npos);
  for (const std::uint32_t period : {1u, ~0u}) {
    config.intra_period = period;
    CHECK(codec::Encoder::create(gpu.device, gpu.allocator, config).ok());
  }
  return 0;
}

// Each rule that makes an I-frame, at a period of 3: the first frame, the
// period, force_intra(), a grid of another voxel size or of another
// trunc_dist alone -- and the grid after each, against that one -- a grid of
// no blocks after one of some, and a reference of no blocks, whatever the
// grid; a P-frame otherwise, costing the same rows as an I-frame. A grid of
// no blocks, planned as a P-frame or not, is an I-frame of none, with no
// back half and so no "  ..rans encode".
int intra_rules_case(Gpu& gpu) {
  codec::EncoderConfig config;
  config.intra_period = 3;
  vkc::Result<codec::Encoder> made =
      codec::Encoder::create(gpu.device, gpu.allocator, config);
  vkc::Result<vol::VoxelBlockGrid> a = sphere_grid(gpu, chain_sphere(0));
  vkc::Result<vol::VoxelBlockGrid> b = sphere_grid(gpu, chain_sphere(1));
  GridShape coarse;
  coarse.voxel_size = 0.006f;
  vkc::Result<vol::VoxelBlockGrid> c =
      sphere_grid(gpu, chain_sphere(1), coarse);
  GridShape wide;
  wide.trunc_dist = 0.05f;
  vkc::Result<vol::VoxelBlockGrid> w = sphere_grid(gpu, chain_sphere(1), wide);
  vkc::Result<vol::VoxelBlockGrid> empty = make_grid(gpu);
  CHECK(made.ok() && a.ok() && b.ok() && c.ok() && w.ok() && empty.ok());
  codec::Encoder& enc = made.value();
  using T = codec::FrameType;
  constexpr T I = T::kIntra;
  constexpr T P = T::kInter;
  struct Step {
    vol::VoxelBlockGrid* grid;
    bool force;
    T type;
  };
  const Step steps[] = {
      {&a.value(), false, I},      // the first frame
      {&b.value(), false, P},      //
      {&a.value(), false, P},      //
      {&b.value(), false, I},      // the period
      {&a.value(), false, P},      //
      {&b.value(), true, I},       // force_intra()
      {&a.value(), false, P},      //
      {&c.value(), false, I},      // another voxel size
      {&a.value(), false, I},      // and back
      {&w.value(), false, I},      // another trunc_dist
      {&a.value(), false, I},      // and back
      {&empty.value(), false, I},  // a grid of no blocks
      {&empty.value(), false, I},  // a reference of no blocks
      {&a.value(), false, I},      // and a grid of some
      {&b.value(), false, P},      //
  };
  int i = 0;
  for (const Step& step : steps) {
    if (step.force) enc.force_intra();
    vkc::StageMetrics m;
    vkc::Result<Bytes> frame = enc.encode(*step.grid, &m);
    CHECK(frame.ok());
    if (type_of(frame.value()) != step.type) {
      std::fprintf(stderr, "step %d: the wrong frame type\n", i);
      return 1;
    }
    for (const char* row :
         {"codec encode", "  ..observed", "  ..forward", "  ..rans encode"}) {
      // A frame of none codes nothing: it records no back half.
      const bool coded = std::strcmp(row, "  ..rans encode") != 0 ||
                         step.grid != &empty.value();
      CHECK((find_row(m, row) != nullptr) == coded);
    }
    ++i;
  }
  return 0;
}

// A refused encode -- a grid of another block size, or with no weight --
// keeps the reference, the period's count and a pending force_intra():
// the frames after it are the ones an encoder that never saw it writes.
// Both refuse before the store is written; a later refusal (the diff's, the
// writer's) has no public trigger, and keeps them alike, since only the
// spare and scratch are written before the commit.
int refusal_keeps_chain_case(Gpu& gpu) {
  codec::EncoderConfig config;
  config.intra_period = 3;
  vkc::Result<codec::Encoder> refused_one =
      codec::Encoder::create(gpu.device, gpu.allocator, config);
  vkc::Result<codec::Encoder> clean =
      codec::Encoder::create(gpu.device, gpu.allocator, config);
  GridShape four;
  four.block_size = 4;
  GridShape no_weight;
  no_weight.weight = false;
  vkc::Result<vol::VoxelBlockGrid> bad4 = make_grid(gpu, four);
  vkc::Result<vol::VoxelBlockGrid> bad_weight = make_grid(gpu, no_weight);
  CHECK(refused_one.ok() && clean.ok() && bad4.ok() && bad_weight.ok());
  // The grids of a chain, and whether to force an I-frame before each.
  const bool force[] = {false, false, false, false, true, false};
  for (int t = 0; t < 6; ++t) {
    vkc::Result<vol::VoxelBlockGrid> g = sphere_grid(gpu, chain_sphere(t));
    CHECK(g.ok());
    if (force[t]) {
      refused_one.value().force_intra();
      clean.value().force_intra();
    }
    // Refusals before the frame, the force left pending across them.
    CHECK(!refused_one.value().encode(bad4.value()).ok());
    CHECK(!refused_one.value().encode(bad_weight.value()).ok());
    vkc::Result<Bytes> got = refused_one.value().encode(g.value());
    vkc::Result<Bytes> want = clean.value().encode(g.value());
    CHECK(got.ok() && want.ok() && got.value() == want.value());
    // The first, the period's and the forced.
    const codec::FrameType expected = t == 0 || t == 3 || t == 4
                                          ? codec::FrameType::kIntra
                                          : codec::FrameType::kInter;
    CHECK(type_of(got.value()) == expected);
  }
  return 0;
}

// A chain fed to a fresh encoder repeats, frame for frame, I and P alike.
int repeat_case(Gpu& gpu) {
  codec::EncoderConfig config;
  config.intra_period = 4;
  std::vector<Bytes> first;
  for (int run = 0; run < 2; ++run) {
    vkc::Result<codec::Encoder> enc =
        codec::Encoder::create(gpu.device, gpu.allocator, config);
    CHECK(enc.ok());
    for (int t = 0; t < 6; ++t) {
      vkc::Result<vol::VoxelBlockGrid> g = sphere_grid(gpu, chain_sphere(t));
      CHECK(g.ok());
      vkc::Result<Bytes> frame = enc.value().encode(g.value());
      CHECK(frame.ok());
      if (run == 0) {
        first.push_back(std::move(frame).value());
      } else {
        CHECK(frame.value() == first[std::size_t(t)]);
      }
    }
  }
  CHECK(type_of(first[1]) == codec::FrameType::kInter);
  CHECK(type_of(first[4]) == codec::FrameType::kIntra);
  return 0;
}

// --- EncoderConfig::true_sdf. ------------------------------------------------

// Off the grid, so the curtain's slab faces lie inside blocks.
Curtain test_curtain(bool with_speck = true) {
  return Curtain{Sphere{vr::Vec3f(0.013f, -0.007f, 0.004f), 0.09f}, with_speck};
}

codec::EncoderConfig true_sdf_config() {
  codec::EncoderConfig config;
  config.true_sdf = true;
  return config;
}

// The block ceiling decoded() gives the player's helpers: far past any
// fixture's.
constexpr std::uint32_t kMaxBlocks = 1u << 20;

// A frame decoded as a player decodes it, into a grid its header builds.
vkc::Result<Snapshot> decoded(Gpu& gpu, const Bytes& frame) {
  VKC_ASSIGN(vol::VoxelBlockGrid grid,
             codec::make_decode_grid(gpu.device, gpu.allocator, frame.data(),
                                     frame.size(), kMaxBlocks));
  VKC_ASSIGN(codec::Decoder dec,
             codec::Decoder::create(gpu.device, gpu.allocator));
  VKC_TRY(
      codec::decode_growing(dec, frame.data(), frame.size(), grid, kMaxBlocks));
  return snapshot(gpu, grid);
}

// The voxels of a decoded grid's blocks that their masks leave unobserved.
std::size_t unobserved(const Snapshot& s) {
  return std::size_t(std::count(s.weight.begin(), s.weight.end(), 0.0f));
}

std::vector<vr::Vec3i> coords_of(const std::vector<vol::BlockIndex>& blocks) {
  std::vector<vr::Vec3i> out;
  for (const vol::BlockIndex& b : blocks) out.push_back(b.coord);
  return out;
}

// The Encoder writes the CPU oracle's bytes, its blocks and cut set found by
// the host rule, for a curtain and a speck at a 16-voxel band, at an odd K,
// for the default and a finer segmentation and the longest segments. The
// premises: the curtain is cut, across block faces too; the speck's 8 blocks
// are kept by the near rule alone; fewer blocks are coded than observed; and
// coded blocks hold cut voxels, coded unobserved.
int true_sdf_oracle_case(Gpu& gpu) {
  vkc::Result<vol::VoxelBlockGrid> g = curtain_grid(gpu, test_curtain());
  CHECK(g.ok());
  vol::VoxelBlockGrid grid = std::move(g).value();
  vkc::Result<codec_oracle::SurfaceSet> set =
      codec_oracle::surface_set(gpu, grid);
  CHECK(set.ok());
  const codec_oracle::SurfaceSet& host = set.value();
  CHECK(host.cut_voxels > 0 && host.cut_across_faces > 0);
  CHECK(host.near_only_blocks >= 8);
  CHECK(!host.blocks.empty() && host.blocks.size() < host.observed_blocks);
  Bytes last;
  for (std::uint32_t segment_size :
       {64u, 3u, codec::kMaxDeviceDecodeSegmentSize}) {
    codec::EncoderConfig config = true_sdf_config();
    config.params.coefficient_count = 37;
    config.segment_size = segment_size;
    vkc::Result<codec::Encoder> enc =
        codec::Encoder::create(gpu.device, gpu.allocator, config);
    CHECK(enc.ok());
    vkc::Result<Bytes> frame = enc.value().encode(grid);
    vkc::Result<Bytes> oracle = codec_oracle::encode(gpu, grid, config);
    CHECK(frame.ok() && oracle.ok());
    CHECK(frame.value() == oracle.value());
    last = std::move(frame).value();
  }
  vkc::Result<Snapshot> out = decoded(gpu, last);
  CHECK(out.ok());
  CHECK(out.value().coords == coords_of(host.blocks));
  CHECK(unobserved(out.value()) > 0);
  std::printf(
      "true SDF curtain: %zu of %zu observed blocks coded, %zu by the near "
      "rule alone; %zu voxels cut, %zu across block faces, %zu in coded "
      "blocks\n",
      host.blocks.size(), host.observed_blocks, host.near_only_blocks,
      host.cut_voxels, host.cut_across_faces, unobserved(out.value()));
  return 0;
}

// On a true distance -- two spheres' union and a speck -- the frame codes
// the host rule's blocks, fewer than are observed, the speck's by the near
// rule alone, and cuts nothing: every voxel of a coded block decodes
// observed.
int surface_blocks_case(Gpu& gpu) {
  const Sphere a{vr::Vec3f(0.01f, -0.02f, 0.03f), 0.09f};
  const Sphere b{vr::Vec3f(0.31f, 0.0f, 0.0f), 0.05f};
  const GridShape shape = wide_band();
  vkc::Result<vol::VoxelBlockGrid> g = make_grid(gpu, shape);
  CHECK(g.ok());
  vol::VoxelBlockGrid grid = std::move(g).value();
  CHECK(allocate(gpu, grid, union_blocks({a, b, speck(10)}, shape)).ok());
  CHECK(write_field(gpu, grid, [&](const vr::Vec3f& p) {
          return std::min({a.sdf(p), b.sdf(p), speck(10).sdf(p)});
        }).ok());
  vkc::Result<codec_oracle::SurfaceSet> set =
      codec_oracle::surface_set(gpu, grid);
  CHECK(set.ok());
  const codec_oracle::SurfaceSet& host = set.value();
  CHECK(host.cut_voxels == 0 && host.near_only_blocks >= 8);
  CHECK(host.blocks.size() < host.observed_blocks);
  vkc::Result<codec::Encoder> enc =
      codec::Encoder::create(gpu.device, gpu.allocator, true_sdf_config());
  CHECK(enc.ok());
  vkc::Result<Bytes> frame = enc.value().encode(grid);
  CHECK(frame.ok());
  vkc::Result<Snapshot> out = decoded(gpu, frame.value());
  CHECK(out.ok());
  CHECK(out.value().coords == coords_of(host.blocks));
  CHECK(unobserved(out.value()) == 0);
  return 0;
}

// The same curtain in two grids whose hash tables hold it differently --
// allocated in opposite orders into tables of different sizes -- codes to
// the same bytes, again and again.
int true_sdf_order_case(Gpu& gpu) {
  GridShape bigger = wide_band();
  bigger.num_buckets = 2048;
  vkc::Result<vol::VoxelBlockGrid> a = curtain_grid(gpu, test_curtain());
  vkc::Result<vol::VoxelBlockGrid> b =
      curtain_grid(gpu, test_curtain(), bigger, false);
  vkc::Result<codec::Encoder> enc =
      codec::Encoder::create(gpu.device, gpu.allocator, true_sdf_config());
  CHECK(a.ok() && b.ok() && enc.ok());
  vkc::Result<std::vector<vol::BlockIndex>> pa =
      a.value().map().compact_active_blocks();
  vkc::Result<std::vector<vol::BlockIndex>> pb =
      b.value().map().compact_active_blocks();
  CHECK(pa.ok() && pb.ok() && pa.value().size() == pb.value().size());
  bool any_differs = false;
  for (std::size_t i = 0; i < pa.value().size(); ++i) {
    any_differs = any_differs ||
                  !(pa.value()[i].coord == pb.value()[i].coord) ||
                  pa.value()[i].ptr != pb.value()[i].ptr;
  }
  CHECK(any_differs);  // the premise: the tables really do differ
  vkc::Result<Bytes> fa = enc.value().encode(a.value());
  vkc::Result<Bytes> fb = enc.value().encode(b.value());
  vkc::Result<Bytes> again = enc.value().encode(a.value());
  CHECK(fa.ok() && fb.ok() && again.ok());
  CHECK(fa.value() == fb.value() && again.value() == fa.value());
  return 0;
}

// One encoder across a heap that grows between frames: the scratch the
// passes regrow is bound afresh, so each frame is the oracle's, and the
// same content the same bytes.
int true_sdf_heap_case(Gpu& gpu) {
  vkc::Result<codec::Encoder> enc =
      codec::Encoder::create(gpu.device, gpu.allocator, true_sdf_config());
  vkc::Result<vol::VoxelBlockGrid> g = curtain_grid(gpu, test_curtain());
  CHECK(enc.ok() && g.ok());
  vol::VoxelBlockGrid& grid = g.value();
  const std::int32_t slots = grid.grid().num_blocks;
  Bytes first;
  for (int grown = 0; grown < 3; ++grown) {
    if (grown != 0) CHECK(vol::grow_grid(grid).ok());
    vkc::Result<Bytes> frame = enc.value().encode(grid);
    vkc::Result<Bytes> oracle =
        codec_oracle::encode(gpu, grid, true_sdf_config());
    CHECK(frame.ok() && oracle.ok() && frame.value() == oracle.value());
    if (grown == 0) first = frame.value();
    CHECK(frame.value() == first);
  }
  CHECK(grid.grid().num_blocks >= 4 * slots);  // the premise: it grew
  return 0;
}

// One encoder, a frame with cuts, then a frame without any on the same grid
// cleared and allocated again -- its slots now other blocks': no cut and no
// kept block of the first frame reaches the second, which is the oracle's
// and codes every voxel of its blocks observed. clear() refills the heap as
// a new grid's, so the same blocks may draw the same slots again: far blocks,
// drawn one at a time until one holds a slot of the first frame's, keep that
// slot from its block while the curtain's blocks are drawn, then go.
int stale_cut_case(Gpu& gpu) {
  const Curtain curtain = test_curtain(false);
  vkc::Result<codec::Encoder> enc =
      codec::Encoder::create(gpu.device, gpu.allocator, true_sdf_config());
  vkc::Result<vol::VoxelBlockGrid> g = curtain_grid(gpu, curtain);
  CHECK(enc.ok() && g.ok());
  vol::VoxelBlockGrid& grid = g.value();
  vkc::Result<Bytes> cut = enc.value().encode(grid);
  CHECK(cut.ok());
  vkc::Result<Snapshot> with_cuts = decoded(gpu, cut.value());
  CHECK(with_cuts.ok() && unobserved(with_cuts.value()) > 0);
  vkc::Result<std::vector<vol::BlockIndex>> before = active_sorted(grid);
  CHECK(before.ok());
  CHECK(grid.clear().ok());
  std::vector<vol::BlockIndex> far;
  bool holds_one = false;
  while (!holds_one) {
    CHECK(far.size() < std::size_t(grid.grid().num_blocks));
    far.push_back(vol::BlockIndex{vr::Vec3i(1000 + int(far.size()), 0, 0), 0});
    CHECK(allocate(gpu, grid, {far.back().coord}).ok());
    vkc::Result<std::vector<vol::BlockIndex>> drawn = active_sorted(grid);
    CHECK(drawn.ok());
    for (const vol::BlockIndex& f : drawn.value()) {
      for (const vol::BlockIndex& b : before.value()) {
        holds_one = holds_one || f.ptr == b.ptr;
      }
    }
  }
  CHECK(allocate(gpu, grid, union_blocks({curtain.sphere}, wide_band())).ok());
  vkc::Result<std::uint32_t> failed = vr_test::remove_blocks(gpu, grid, far);
  CHECK(failed.ok() && failed.value() == 0);
  CHECK(write_sphere(gpu, grid, curtain.sphere).ok());
  vkc::Result<std::vector<vol::BlockIndex>> after = active_sorted(grid);
  CHECK(after.ok() && coords_of(before.value()) == coords_of(after.value()));
  bool moved = false;
  for (std::size_t i = 0; i < after.value().size(); ++i) {
    moved = moved || before.value()[i].ptr != after.value()[i].ptr;
  }
  CHECK(moved);  // the premise: the blocks sit in other slots
  vkc::Result<Bytes> clean = enc.value().encode(grid);
  vkc::Result<Bytes> oracle =
      codec_oracle::encode(gpu, grid, true_sdf_config());
  CHECK(clean.ok() && oracle.ok() && clean.value() == oracle.value());
  vkc::Result<Snapshot> none = decoded(gpu, clean.value());
  CHECK(none.ok() && unobserved(none.value()) == 0);
  return 0;
}

// --- The two halves. ---------------------------------------------------------

// A frame by the halves, each batch the caller's, its head and payload joined
// as encode() joins them; with its type and whether its back batch held a
// command, read from the device's submit count. The front is handed the grid
// read-only.
struct Halves {
  Bytes bytes;
  codec::FrameType planned = codec::FrameType::kIntra;
  codec::FrameType type = codec::FrameType::kIntra;
  bool back_submitted = false;
};

vkc::Result<Halves> by_halves(Gpu& gpu, codec::Encoder& enc,
                              const vol::VoxelBlockGrid& grid,
                              vkc::StageMetrics* m = nullptr) {
  Halves out;
  vkc::CommandBatch front(gpu.device, gpu.allocator);
  VKC_ASSIGN(const codec::PendingEncode pending,
             enc.record_front(front, grid, m));
  out.planned = pending.type;
  VKC_TRY(front.submit());
  vkc::CommandBatch back(gpu.device, gpu.allocator);
  VKC_TRY(enc.record_back(back, pending, m));
  const std::uint64_t before = gpu.device.counters().submits;
  VKC_TRY(back.submit());
  out.back_submitted = gpu.device.counters().submits != before;
  VKC_ASSIGN(const codec::EncodedFrame frame, enc.finish(pending, m));
  out.type = frame.type;
  out.bytes.assign(frame.head.begin(), frame.head.end());
  out.bytes.insert(out.bytes.end(), frame.payload,
                   frame.payload + frame.payload_size);
  if (out.bytes.size() != frame.size() ||
      (frame.payload == nullptr) != (frame.payload_size == 0)) {
    return vkc::Status::invalid_argument("the frame's sizes disagree");
  }
  return out;
}

// The halves give encode()'s bytes frame for frame, along a chain of I- and
// P-frames at a period of 3, at K 20, 56 and 64, masked and as a true
// distance; and on a grid of 64 times the block slots its blocks need, whose
// every step runs for the count under that bound, the sort's levels past it
// copying their entries through.
int halves_case(Gpu& gpu) {
  struct Setting {
    std::uint32_t k;
    bool true_sdf;
  };
  using T = codec::FrameType;
  const T types[] = {T::kIntra, T::kInter, T::kInter, T::kIntra};
  for (const Setting s : {Setting{20, false}, Setting{56, false},
                          Setting{64, false}, Setting{56, true}}) {
    codec::EncoderConfig config;
    config.intra_period = 3;
    config.params.coefficient_count = s.k;
    config.true_sdf = s.true_sdf;
    vkc::Result<codec::Encoder> a =
        codec::Encoder::create(gpu.device, gpu.allocator, config);
    vkc::Result<codec::Encoder> b =
        codec::Encoder::create(gpu.device, gpu.allocator, config);
    CHECK(a.ok() && b.ok());
    for (int t = 0; t < 4; ++t) {
      vkc::Result<vol::VoxelBlockGrid> grid =
          s.true_sdf ? sphere_grid(gpu, chain_sphere(t), wide_band())
                     : chain_grid(gpu, t);
      CHECK(grid.ok());
      vkc::Result<Bytes> want = a.value().encode(grid.value());
      vkc::Result<Halves> got = by_halves(gpu, b.value(), grid.value());
      CHECK(want.ok() && got.ok());
      CHECK(got.value().bytes == want.value());
      CHECK(got.value().planned == types[t] && got.value().type == types[t]);
      CHECK(got.value().back_submitted);
    }
  }

  const Sphere s{vr::Vec3f(0.003f, 0.001f, -0.002f), 0.08f};
  const std::size_t blocks = band_blocks(s).size();
  GridShape slots;
  slots.num_buckets = static_cast<std::int32_t>(8 * blocks);  // 64x
  vkc::Result<vol::VoxelBlockGrid> tight = sphere_grid(gpu, s);
  vkc::Result<vol::VoxelBlockGrid> loose = sphere_grid(gpu, s, slots);
  vkc::Result<codec::Encoder> a =
      codec::Encoder::create(gpu.device, gpu.allocator);
  vkc::Result<codec::Encoder> b =
      codec::Encoder::create(gpu.device, gpu.allocator);
  CHECK(tight.ok() && loose.ok() && a.ok() && b.ok());
  CHECK(loose.value().grid().num_blocks >= std::int32_t(64 * blocks));
  vkc::Result<Bytes> want = a.value().encode(tight.value());
  vkc::Result<Halves> got = by_halves(gpu, b.value(), loose.value());
  CHECK(want.ok() && got.ok() && got.value().bytes == want.value());
  return 0;
}

// A P-frame planned over a grid with no block to code finishes as an I-frame
// of none, recorded with no back batch: the bytes encode() gives a grid of
// none. And the halves' own refusals: a back recorded twice, a finish before
// its back, a token abandoned or superseded, or another encoder's.
int halves_rules_case(Gpu& gpu) {
  codec::EncoderConfig config;
  config.intra_period = 30;
  vkc::Result<codec::Encoder> made =
      codec::Encoder::create(gpu.device, gpu.allocator, config);
  vkc::Result<codec::Encoder> other =
      codec::Encoder::create(gpu.device, gpu.allocator, config);
  vkc::Result<codec::Encoder> fresh =
      codec::Encoder::create(gpu.device, gpu.allocator);
  vkc::Result<vol::VoxelBlockGrid> a = sphere_grid(gpu, chain_sphere(0));
  vkc::Result<vol::VoxelBlockGrid> b = sphere_grid(gpu, chain_sphere(1));
  vkc::Result<vol::VoxelBlockGrid> none = make_grid(gpu);
  CHECK(made.ok() && other.ok() && fresh.ok() && a.ok() && b.ok() && none.ok());
  codec::Encoder& enc = made.value();
  CHECK(enc.encode(a.value()).ok());
  vkc::Result<Halves> empty = by_halves(gpu, enc, none.value());
  CHECK(empty.ok());
  CHECK(empty.value().planned == codec::FrameType::kInter);
  CHECK(empty.value().type == codec::FrameType::kIntra);
  CHECK(!empty.value().back_submitted);
  CHECK(empty.value().bytes == fresh.value().encode(none.value()).value());

  const auto refused = [](const vkc::Status& s) {
    return s.domain() == vkc::Status::Code::InvalidArgument;
  };
  vkc::CommandBatch front(gpu.device, gpu.allocator);
  vkc::Result<codec::PendingEncode> p = enc.record_front(front, b.value());
  CHECK(p.ok() && front.submit().ok());
  CHECK(refused(enc.finish(p.value()).status()));  // its back not recorded
  CHECK(refused(other.value().finish(p.value()).status()));
  vkc::CommandBatch back(gpu.device, gpu.allocator);
  CHECK(refused(other.value().record_back(back, p.value())));
  CHECK(enc.record_back(back, p.value()).ok());
  CHECK(refused(enc.record_back(back, p.value())));  // twice
  CHECK(back.submit().ok());
  CHECK(enc.finish(p.value()).ok());
  CHECK(refused(enc.finish(p.value()).status()));  // finished

  // A record supersedes the last unfinished one.
  vkc::CommandBatch first(gpu.device, gpu.allocator);
  vkc::Result<codec::PendingEncode> p1 = enc.record_front(first, a.value());
  CHECK(p1.ok() && first.submit().ok());
  vkc::CommandBatch second(gpu.device, gpu.allocator);
  vkc::Result<codec::PendingEncode> p2 = enc.record_front(second, b.value());
  CHECK(p2.ok() && second.submit().ok());
  vkc::CommandBatch late(gpu.device, gpu.allocator);
  CHECK(refused(enc.record_back(late, p1.value())));
  enc.abandon(p1.value());  // ignored: superseded
  CHECK(enc.record_back(late, p2.value()).ok() && late.submit().ok());
  enc.abandon(p2.value());
  CHECK(refused(enc.finish(p2.value()).status()));
  return 0;
}

// abandon keeps the reference and the period: an encoder that recorded a
// frame and dropped it writes the frames after as one that never saw it --
// a P-frame against the frame before, then the period's I-frame; and a cold
// encoder whose first front was never submitted codes as a fresh one.
int abandon_case(Gpu& gpu) {
  codec::EncoderConfig config;
  config.intra_period = 3;
  vkc::Result<codec::Encoder> x =
      codec::Encoder::create(gpu.device, gpu.allocator, config);
  vkc::Result<codec::Encoder> y =
      codec::Encoder::create(gpu.device, gpu.allocator, config);
  CHECK(x.ok() && y.ok());
  std::vector<vol::VoxelBlockGrid> grids;
  for (int t = 0; t < 4; ++t) {
    vkc::Result<vol::VoxelBlockGrid> g = sphere_grid(gpu, chain_sphere(t));
    CHECK(g.ok());
    grids.push_back(std::move(g).value());
  }
  vkc::Result<Bytes> first = y.value().encode(grids[0]);
  CHECK(first.ok() && x.value().encode(grids[0]).value() == first.value());
  // A cold encoder's first front, dropped unsubmitted, wrote no quantization
  // steps, so its next front still records them: the same frame.
  {
    vkc::Result<codec::Encoder> cold =
        codec::Encoder::create(gpu.device, gpu.allocator, config);
    CHECK(cold.ok());
    {
      vkc::CommandBatch front(gpu.device, gpu.allocator);
      vkc::Result<codec::PendingEncode> p =
          cold.value().record_front(front, grids[0]);
      CHECK(p.ok());
      cold.value().abandon(p.value());
    }
    vkc::Result<Bytes> got = cold.value().encode(grids[0]);
    CHECK(got.ok() && got.value() == first.value());
  }
  for (const bool after_back : {false, true}) {
    vkc::CommandBatch front(gpu.device, gpu.allocator);
    vkc::Result<codec::PendingEncode> p =
        x.value().record_front(front, grids[3]);
    CHECK(p.ok() && front.submit().ok());
    if (after_back) {
      vkc::CommandBatch back(gpu.device, gpu.allocator);
      CHECK(x.value().record_back(back, p.value()).ok() && back.submit().ok());
    }
    x.value().abandon(p.value());
  }
  for (int t = 1; t < 4; ++t) {
    vkc::Result<Bytes> got = x.value().encode(grids[std::size_t(t)]);
    vkc::Result<Bytes> want = y.value().encode(grids[std::size_t(t)]);
    CHECK(got.ok() && want.ok() && got.value() == want.value());
    CHECK(type_of(got.value()) ==
          (t == 3 ? codec::FrameType::kIntra : codec::FrameType::kInter));
  }
  return 0;
}

// A payload past what the back batch copied is gathered again over the rest
// ("  ..tail"), to the same bytes: a smooth frame sets the next frame's
// guess, and a frame of noise outgrows it. In place the batch writes the
// slot itself, so there is never a tail.
int tail_case(Gpu& gpu) {
  vkc::Result<codec::Encoder> enc =
      codec::Encoder::create(gpu.device, gpu.allocator);
  vkc::Result<codec::Encoder> fresh =
      codec::Encoder::create(gpu.device, gpu.allocator);
  vkc::Result<vol::VoxelBlockGrid> smooth = make_grid(gpu);
  vkc::Result<vol::VoxelBlockGrid> noise = make_grid(gpu);
  CHECK(enc.ok() && fresh.ok() && smooth.ok() && noise.ok());
  std::vector<vr::Vec3i> coords;
  for (int i = 0; i < 64; ++i) coords.push_back(vr::Vec3i(i % 8, i / 8, 49));
  CHECK(allocate(gpu, smooth.value(), coords).ok());
  CHECK(allocate(gpu, noise.value(), coords).ok());
  CHECK(write_field(
            gpu, smooth.value(), [](const vr::Vec3f& p) { return p.z - 1.98f; },
            true)
            .ok());
  CHECK(write_field(
            gpu, noise.value(),
            [](const vr::Vec3f& p) {
              const auto h = std::uint32_t(p.x * 7919.0f + p.y * 104729.0f +
                                           p.z * 1299709.0f) *
                             2654435761u;
              return (float(h % 2001u) / 1000.0f - 1.0f) * kTrunc;
            },
            true)
            .ok());
  vkc::Result<Halves> first = by_halves(gpu, enc.value(), smooth.value());
  vkc::StageMetrics m;
  vkc::Result<Halves> noisy = by_halves(gpu, enc.value(), noise.value(), &m);
  CHECK(first.ok() && noisy.ok());
  CHECK(noisy.value().bytes == fresh.value().encode(noise.value()).value());
  const bool in_place = gpu.device.caps().device_mapped_cached();
  std::printf("  smooth %zu bytes, noise %zu bytes, in place %d\n",
              first.value().bytes.size(), noisy.value().bytes.size(),
              int(in_place));
  CHECK((find_row(m, "  ..tail") != nullptr) == !in_place);
  return 0;
}

// After warm-up encode() is 2 submits -- the front's and the back's -- and
// no staging buffer, an I-frame or a P-frame.
int counters_case(Gpu& gpu) {
  codec::EncoderConfig config;
  config.intra_period = 30;
  vkc::Result<codec::Encoder> enc =
      codec::Encoder::create(gpu.device, gpu.allocator, config);
  CHECK(enc.ok());
  for (int t = 0; t < 6; ++t) {
    vkc::Result<vol::VoxelBlockGrid> grid =
        sphere_grid(gpu, chain_sphere(t % 2));
    CHECK(grid.ok());
    const vkc::DeviceCounters was = gpu.device.counters();
    vkc::Result<Bytes> frame = enc.value().encode(grid.value());
    const vkc::DeviceCounters now = gpu.device.counters();
    CHECK(frame.ok());
    if (t < 3) continue;  // the rings and scratch grow while warming
    CHECK(type_of(frame.value()) == codec::FrameType::kInter);
    CHECK(now.submits - was.submits == 2);
    // A wait that finds its batch done blocks nothing and counts none.
    CHECK(now.host_waits - was.host_waits <= 2);
    CHECK(now.staging_buffers == was.staging_buffers);
  }
  return 0;
}

int gpu_main(vkc::test::GpuContext& gpu) {
  vkc::Result<codec::Encoder> made =
      codec::Encoder::create(gpu.device, gpu.allocator);
  CHECK(made.ok());
  codec::Encoder enc = std::move(made).value();
  if (empty_grid_case(gpu, enc) != 0) return 1;
  if (order_independent_case(gpu, enc) != 0) return 1;
  if (unobserved_dropped_case(gpu, enc) != 0) return 1;
  if (metrics_case(gpu, enc) != 0) return 1;
  if (device_list_case(gpu, enc) != 0) return 1;
  if (oracle_case(gpu) != 0) return 1;
  if (extremes_case(gpu, enc) != 0) return 1;
  if (sizes_case(gpu) != 0) return 1;
  if (refusals_case(gpu) != 0) return 1;
  if (moves_case(gpu) != 0) return 1;
  if (period_case(gpu) != 0) return 1;
  if (intra_rules_case(gpu) != 0) return 1;
  if (refusal_keeps_chain_case(gpu) != 0) return 1;
  if (repeat_case(gpu) != 0) return 1;
  if (true_sdf_oracle_case(gpu) != 0) return 1;
  if (surface_blocks_case(gpu) != 0) return 1;
  if (true_sdf_order_case(gpu) != 0) return 1;
  if (true_sdf_heap_case(gpu) != 0) return 1;
  if (stale_cut_case(gpu) != 0) return 1;
  if (halves_case(gpu) != 0) return 1;
  if (halves_rules_case(gpu) != 0) return 1;
  if (abandon_case(gpu) != 0) return 1;
  if (tail_case(gpu) != 0) return 1;
  if (counters_case(gpu) != 0) return 1;
  std::printf("codec Encoder: OK\n");
  return 0;
}

}  // namespace

int main() {
  return vkc::test::run_on_gpu(vr::device_requirements(), gpu_main);
}
