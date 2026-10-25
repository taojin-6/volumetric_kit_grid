// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// GPU test for a frame decoded and meshed in the caller's batch
// (Decoder::claim_input, record, finish and abandon; record_growing and
// finish_growing; MarchingCubes::record_extract gated on the decode): the
// grid is what decode() leaves, the deblock's report too, chain after chain;
// a refusal before recording records nothing and moves neither the map's
// epoch nor its tick; a frame the device refuses leaves the extract after it
// meshing nothing; a superseded record does not finish, an abandoned one
// keeps the reference; both input slots held refuse a third; the rows; a
// steady frame costs one submit and no staging buffer; a rafa2-like P-frame's
// one batch, decode and mesh, records a pinned number of dispatches and
// barriers; and the growing calls grow before recording, or decode again
// from the pending frame's own slot after the device refused it. Links
// recon_mesh, which the codec tier itself may not. Skips with no device.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "codec_fixture.hpp"
#include "volumetric_kit/core/base/stage_metrics.hpp"
#include "volumetric_kit/core/vulkan/command_batch.hpp"
#include "volumetric_kit/core/vulkan/gpu_timer.hpp"
#include "volumetric_kit/core/vulkan/host_ring.hpp"
#include "volumetric_kit/recon/codec/decode_grid.hpp"
#include "volumetric_kit/recon/codec/decoder.hpp"
#include "volumetric_kit/recon/codec/encoder.hpp"
#include "volumetric_kit/recon/mesh/marching_cubes.hpp"

#include "gpu_test.hpp"

namespace vkc = volumetric_kit::core;
namespace codec = volumetric_kit::recon::codec;
namespace mesh = volumetric_kit::recon::mesh;
using namespace codec_fixture;
using C = vkc::Status::Code;

namespace {

using Bytes = std::vector<std::uint8_t>;

constexpr std::uint32_t kMaxBlocks = 1u << 20;

// pin_case's P-frame batch, decode and mesh, on an in-place ring.
constexpr std::uint64_t kPinnedDispatches = 39;
constexpr std::uint64_t kPinnedBarriers = 47;

const vkc::StageRow* find_row(const vkc::StageMetrics& m, const char* name) {
  for (const vkc::StageRow& row : m.rows()) {
    if (std::strcmp(row.name, name) == 0) return &row;
  }
  return nullptr;
}

vkc::Result<codec::Encoder> encoder(Gpu& gpu, std::uint32_t k, float scale,
                                    std::uint32_t period) {
  codec::EncoderConfig config;
  config.params.coefficient_count = k;
  if (scale > 0.0f) config.params.quantization_scale = scale;
  config.intra_period = period;
  return codec::Encoder::create(gpu.device, gpu.allocator, config);
}

vkc::Result<codec::Decoder> decoder(Gpu& gpu, bool deblock = false) {
  VKC_ASSIGN(codec::Decoder dec,
             codec::Decoder::create(gpu.device, gpu.allocator));
  dec.set_deblock(deblock);
  return dec;
}

// The chain's frames: chain_grid's steps 0 to @p n - 1.
vkc::Result<std::vector<Bytes>> chain(Gpu& gpu, codec::Encoder& enc, int n) {
  std::vector<Bytes> out;
  for (int t = 0; t < n; ++t) {
    VKC_ASSIGN(vol::VoxelBlockGrid g, chain_grid(gpu, t));
    VKC_ASSIGN(Bytes frame, enc.encode(g));
    out.push_back(std::move(frame));
  }
  return out;
}

// @p frame in an input slot, as a reader leaves it.
vkc::Result<vkc::HostSlot> slot_of(codec::Decoder& dec, const Bytes& frame) {
  VKC_ASSIGN(vkc::HostSlot slot, dec.claim_input(frame.size()));
  std::memcpy(slot.data(), frame.data(), frame.size());
  return slot;
}

// @p frame recorded into a batch of its own, submitted without blocking,
// waited on and finished: what a player's frame does, the device rows merged.
vkc::Status recorded(Gpu& gpu, codec::Decoder& dec, const Bytes& frame,
                     vol::VoxelBlockGrid& grid,
                     vkc::StageMetrics* metrics = nullptr) {
  VKC_ASSIGN(const vkc::HostSlot slot, slot_of(dec, frame));
  vkc::CommandBatch batch(gpu.device, gpu.allocator);
  VKC_ASSIGN(const codec::PendingDecode pending,
             dec.record(batch, slot, frame.size(), grid, metrics));
  VKC_ASSIGN(vkc::PendingBatch ran, batch.submit_async({}, {}));
  VKC_TRY(ran.wait());
  if (metrics != nullptr) metrics->merge_device(ran.device_rows());
  return dec.finish(pending, grid);
}

bool same_deblock(const codec::DeblockInfo& a, const codec::DeblockInfo& b) {
  return std::strcmp(a.filter, b.filter) == 0 && a.step == b.step &&
         a.block_count == b.block_count && a.exact_blocks == b.exact_blocks;
}

// --- The grid.
// ----------------------------------------------------------------

// Chains at K 20 and K 64, the deblock off and on (LG at K 64's step 0.2),
// open-ended and at a period of 3: frame after frame, the recorded decode
// leaves its grid bit for bit as decode() leaves another, and reports the
// deblock alike.
int equals_blocking_case(Gpu& gpu) {
  struct Setting {
    std::uint32_t k;
    float scale;
    bool deblock;
    std::uint32_t period;
  };
  for (const Setting& s :
       {Setting{20, 0.0f, false, ~0u}, Setting{64, 0.2f, true, ~0u},
        Setting{64, 0.2f, true, 3}}) {
    vkc::Result<codec::Encoder> enc = encoder(gpu, s.k, s.scale, s.period);
    CHECK(enc.ok());
    vkc::Result<std::vector<Bytes>> frames = chain(gpu, *enc, 5);
    vkc::Result<codec::Decoder> a = decoder(gpu, s.deblock);
    vkc::Result<codec::Decoder> b = decoder(gpu, s.deblock);
    vkc::Result<vol::VoxelBlockGrid> ga = make_grid(gpu);
    vkc::Result<vol::VoxelBlockGrid> gb = make_grid(gpu);
    CHECK(frames.ok() && a.ok() && b.ok() && ga.ok() && gb.ok());
    for (const Bytes& f : *frames) {
      CHECK(a->decode(f.data(), f.size(), *ga).ok());
      CHECK(recorded(gpu, *b, f, *gb).ok());
      vkc::Result<Snapshot> want = snapshot(gpu, *ga);
      vkc::Result<Snapshot> got = snapshot(gpu, *gb);
      CHECK(want.ok() && got.ok() && *want == *got && !got->coords.empty());
      CHECK(same_deblock(a->last_deblock(), b->last_deblock()));
      CHECK(b->last_rounds() >= 1 &&
            b->last_rounds() <= vol::VoxelHashMap::kRecordedRounds);
      CHECK(gb->map().allocated_blocks() == got->coords.size());
    }
    if (s.deblock) CHECK(std::strcmp(b->last_deblock().filter, "LG") == 0);
  }
  return 0;
}

// --- Refusals.
// ----------------------------------------------------------------

// What the host refuses records nothing: the batch submits empty, the grid,
// the map's epoch and tick stay as they were, and the reference holds, so the
// chain's next frame decodes.
int refusals_case(Gpu& gpu) {
  vkc::Result<codec::Encoder> enc = encoder(gpu, 20, 0.0f, ~0u);
  CHECK(enc.ok());
  vkc::Result<std::vector<Bytes>> frames = chain(gpu, *enc, 3);
  vkc::Result<codec::Decoder> dec = decoder(gpu);
  vkc::Result<codec::Decoder> fresh = decoder(gpu);
  vkc::Result<vol::VoxelBlockGrid> grid = make_grid(gpu);
  CHECK(frames.ok() && dec.ok() && fresh.ok() && grid.ok());
  CHECK(recorded(gpu, *dec, (*frames)[0], *grid).ok());
  vkc::Result<Snapshot> before = snapshot(gpu, *grid);
  CHECK(before.ok());
  const std::uint64_t epoch = grid->map().topology_epoch();
  const std::uint32_t tick = grid->map().tick();
  GridShape small;
  small.num_buckets = 2;
  vkc::Result<vol::VoxelBlockGrid> tiny = make_grid(gpu, small);
  CHECK(tiny.ok());
  Bytes corrupt = (*frames)[1];
  corrupt[0] ^= 0xFFu;
  const auto refused = [&](codec::Decoder& d, const Bytes& f,
                           vol::VoxelBlockGrid& g, std::size_t size, C code) {
    vkc::Result<vkc::HostSlot> slot = slot_of(d, f);
    if (!slot.ok()) return false;
    const vkc::DeviceCounters was = gpu.device.counters();
    vkc::CommandBatch batch(gpu.device, gpu.allocator);
    const vkc::Status s = d.record(batch, *slot, size, g).status();
    const bool empty = batch.submit().ok() &&
                       gpu.device.counters().dispatches == was.dispatches;
    if (!(s.domain() == code && empty)) {
      std::fprintf(stderr, "refusal: %s\n", s.message().c_str());
      return false;
    }
    return true;
  };
  CHECK(refused(*dec, corrupt, *grid, corrupt.size(), C::InvalidArgument));
  CHECK(refused(*fresh, (*frames)[1], *grid, (*frames)[1].size(), C::NotFound));
  CHECK(
      refused(*dec, (*frames)[1], *tiny, (*frames)[1].size(), C::OutOfMemory));
  CHECK(refused(*dec, (*frames)[1], *grid, (*frames)[1].size() + 8,
                C::InvalidArgument));
  CHECK(snapshot(gpu, *grid).value() == *before);
  CHECK(grid->map().topology_epoch() == epoch && grid->map().tick() == tick);
  CHECK(recorded(gpu, *dec, (*frames)[1], *grid).ok());
  CHECK(grid->map().tick() != tick);
  return 0;
}

// A frame the device refuses -- the table, its chains one long, cannot place
// it -- closes the decode's gate, so the extract recorded after it in the
// same batch meshes nothing; the epoch has moved, as for any frame recorded.
int device_refusal_case(Gpu& gpu) {
  vkc::Result<codec::Encoder> enc = encoder(gpu, 20, 0.0f, 1);
  CHECK(enc.ok());
  vkc::Result<std::vector<Bytes>> frames = chain(gpu, *enc, 1);
  CHECK(frames.ok());
  const Bytes& f = (*frames)[0];
  vkc::Result<codec::FrameInfo> info =
      codec::read_frame_info(f.data(), f.size());
  CHECK(info.ok());
  GridShape cramped;
  cramped.num_buckets = std::int32_t((info->block_count + 7) / 8 + 1);
  cramped.max_chain = 1;
  vkc::Result<vol::VoxelBlockGrid> grid = make_grid(gpu, cramped);
  vkc::Result<codec::Decoder> dec = decoder(gpu);
  vkc::Result<mesh::MarchingCubes> mc =
      mesh::MarchingCubes::create(gpu.device, gpu.allocator);
  CHECK(grid.ok() && dec.ok() && mc.ok());
  const std::uint64_t epoch = grid->map().topology_epoch();
  vkc::Result<vkc::HostSlot> slot = slot_of(*dec, f);
  CHECK(slot.ok());
  vkc::CommandBatch batch(gpu.device, gpu.allocator);
  vkc::Result<codec::PendingDecode> pending =
      dec->record(batch, *slot, f.size(), *grid);
  CHECK(pending.ok() && pending->blocks == info->block_count);
  vkc::Result<mesh::PendingExtract> extract =
      mc->record_extract(batch, *grid, pending->blocks, &pending->gate);
  CHECK(extract.ok());
  vkc::Result<vkc::PendingBatch> ran =
      batch.submit_async({}, {extract->signal});
  CHECK(ran.ok() && ran->wait().ok());
  CHECK(dec->finish(*pending, *grid).domain() == C::OutOfMemory);
  vkc::Result<std::optional<mesh::DeviceMesh>> none =
      mc->finish(*extract, *grid);
  CHECK(none.ok() && !none->has_value());
  CHECK(grid->map().topology_epoch() != epoch);
  return 0;
}

// --- Tokens and slots.
// --------------------------------------------------------

// A record superseded by the next does not finish; one abandoned keeps the
// reference, so its frame decodes again and the frame after it, coded
// against it, is NotFound; two input slots held refuse a third.
int tokens_case(Gpu& gpu) {
  vkc::Result<codec::Encoder> enc = encoder(gpu, 20, 0.0f, ~0u);
  CHECK(enc.ok());
  vkc::Result<std::vector<Bytes>> frames = chain(gpu, *enc, 3);
  vkc::Result<codec::Decoder> dec = decoder(gpu);
  vkc::Result<vol::VoxelBlockGrid> grid = make_grid(gpu);
  CHECK(frames.ok() && dec.ok() && grid.ok());
  const Bytes& i0 = (*frames)[0];
  const Bytes& p1 = (*frames)[1];
  const Bytes& p2 = (*frames)[2];
  CHECK(dec->decode(i0.data(), i0.size(), *grid).ok());

  // A token holds its input slot until it is dropped, so each lives in a
  // scope of its own.
  const auto run = [&](codec::Decoder& d,
                       const Bytes& f) -> vkc::Result<codec::PendingDecode> {
    VKC_ASSIGN(const vkc::HostSlot slot, slot_of(d, f));
    vkc::CommandBatch batch(gpu.device, gpu.allocator);
    VKC_ASSIGN(codec::PendingDecode p, d.record(batch, slot, f.size(), *grid));
    VKC_TRY(batch.submit());
    return p;
  };
  {
    vkc::Result<codec::PendingDecode> first = run(*dec, p1);
    CHECK(first.ok());
    vkc::Result<codec::PendingDecode> second = run(*dec, p1);
    CHECK(second.ok() && second->serial != first->serial);
    CHECK(dec->finish(*first, *grid).domain() == C::InvalidArgument);
    CHECK(dec->finish(*second, *grid).ok());
    dec->abandon(*first);  // ignored
  }
  {
    vkc::Result<codec::PendingDecode> dropped = run(*dec, p2);
    CHECK(dropped.ok());
    dec->abandon(*dropped);
    CHECK(dec->finish(*dropped, *grid).domain() == C::InvalidArgument);
  }
  // The reference is still p1's: p2 decodes against it.
  CHECK(dec->decode(p2.data(), p2.size(), *grid).ok());

  vkc::Result<codec::Decoder> other = decoder(gpu);
  CHECK(other.ok());
  CHECK(other->decode(i0.data(), i0.size(), *grid).ok());
  {
    vkc::Result<codec::PendingDecode> lost = run(*other, p1);
    CHECK(lost.ok());
    other->abandon(*lost);
  }
  CHECK(other->decode(p2.data(), p2.size(), *grid).domain() == C::NotFound);
  CHECK(other->decode(p1.data(), p1.size(), *grid).ok());

  vkc::Result<vkc::HostSlot> s1 = dec->claim_input(64);
  vkc::Result<vkc::HostSlot> s2 = dec->claim_input(64);
  CHECK(s1.ok() && s2.ok());
  CHECK(dec->claim_input(64).status().domain() == C::InvalidArgument);
  CHECK(dec->claim_input(0).status().domain() == C::InvalidArgument);
  return 0;
}

// --- Rows and cost.
// -------------------------------------------------------------

// The recorded decode's rows: its stage over the three breakdowns, host
// halves at the record and, where the device times, device halves from the
// batch, the breakdowns tiling the stage.
int rows_case(Gpu& gpu) {
  vkc::Result<codec::Encoder> enc = encoder(gpu, 20, 0.0f, ~0u);
  CHECK(enc.ok());
  vkc::Result<std::vector<Bytes>> frames = chain(gpu, *enc, 2);
  vkc::Result<codec::Decoder> dec = decoder(gpu);
  vkc::Result<vol::VoxelBlockGrid> grid = make_grid(gpu);
  CHECK(frames.ok() && dec.ok() && grid.ok());
  vkc::Result<vkc::GpuTimer> probe = vkc::GpuTimer::create(gpu.device);
  CHECK(probe.ok());
  for (const Bytes& f : *frames) {
    vkc::StageMetrics m;
    CHECK(recorded(gpu, *dec, f, *grid, &m).ok());
    const vkc::StageRow* top = find_row(m, "codec decode");
    CHECK(top != nullptr);
    double sum = 0.0;
    for (const char* sub : {"  ..rans decode", "  ..apply", "  ..inverse"}) {
      const vkc::StageRow* row = find_row(m, sub);
      CHECK(row != nullptr && row->cpu_ms <= top->cpu_ms);
      if (probe->available()) {
        CHECK(row->device == vkc::DeviceTime::Measured);
        sum += row->gpu_ms;
      }
    }
    if (probe->available()) {
      CHECK(top->device == vkc::DeviceTime::Measured && top->gpu_ms > 0.0);
      CHECK(std::fabs(sum - top->gpu_ms) < 1e-9);
    }
  }
  return 0;
}

// Once warm, a P-frame costs one submit and no staging buffer, recorded and
// through decode().
int counters_case(Gpu& gpu) {
  vkc::Result<codec::Encoder> enc = encoder(gpu, 20, 0.0f, ~0u);
  CHECK(enc.ok());
  vkc::Result<std::vector<Bytes>> frames = chain(gpu, *enc, 6);
  CHECK(frames.ok());
  for (const bool blocking : {false, true}) {
    vkc::Result<codec::Decoder> dec = decoder(gpu, true);
    vkc::Result<vol::VoxelBlockGrid> grid = make_grid(gpu);
    CHECK(dec.ok() && grid.ok());
    for (std::size_t i = 0; i < frames->size(); ++i) {
      const Bytes& f = (*frames)[i];
      const vkc::DeviceCounters before = gpu.device.counters();
      CHECK(blocking ? dec->decode(f.data(), f.size(), *grid).ok()
                     : recorded(gpu, *dec, f, *grid).ok());
      const vkc::DeviceCounters after = gpu.device.counters();
      if (i < 3) continue;  // the rings and the buffers grow at first
      CHECK(after.submits - before.submits == 1);
      CHECK(after.staging_buffers - before.staging_buffers == 0);
    }
  }
  return 0;
}

// A rafa2-like P-frame -- K 56, the LG deblock, blocks removed, kept and new
// -- decoded and meshed in one batch records the dispatches and barriers
// pinned here (DECISIONS.md, 2026-10-24). A change to the batch's commands
// moves them; one that records an upload after a dispatch adds barriers.
int pin_case(vr_test::GpuContext& gpu) {
  vkc::Result<codec::Encoder> enc = encoder(gpu, 56, 0.2f, ~0u);
  CHECK(enc.ok());
  vkc::Result<std::vector<Bytes>> frames = chain(gpu, *enc, 4);
  vkc::Result<codec::Decoder> dec = decoder(gpu, true);
  vkc::Result<mesh::MarchingCubes> mc =
      mesh::MarchingCubes::create(gpu.device, gpu.allocator);
  vkc::Result<vol::VoxelBlockGrid> grid = make_grid(gpu);
  CHECK(frames.ok() && dec.ok() && mc.ok() && grid.ok());
  for (std::size_t i = 0; i < frames->size(); ++i) {
    const Bytes& f = (*frames)[i];
    vkc::Result<vkc::HostSlot> slot = slot_of(*dec, f);
    CHECK(slot.ok());
    const vkc::DeviceCounters before = gpu.device.counters();
    vkc::CommandBatch batch(gpu.device, gpu.allocator);
    vkc::Result<codec::PendingDecode> pending =
        dec->record(batch, *slot, f.size(), *grid);
    CHECK(pending.ok());
    vkc::Result<mesh::PendingExtract> extract =
        mc->record_extract(batch, *grid, pending->blocks, &pending->gate);
    CHECK(extract.ok());
    vkc::Result<vkc::PendingBatch> ran =
        batch.submit_async({}, {extract->signal});
    CHECK(ran.ok() && ran->wait().ok());
    const vkc::DeviceCounters after = gpu.device.counters();
    CHECK(dec->finish(*pending, *grid).ok());
    vkc::Result<std::optional<mesh::DeviceMesh>> m =
        mc->finish(*extract, *grid);
    CHECK(m.ok() && m->has_value() && !(*m)->empty());
    mc->release_through((*m)->generation);
    if (i < 2) continue;
    CHECK(std::strcmp(dec->last_deblock().filter, "LG") == 0);
    const std::uint64_t dispatches = after.dispatches - before.dispatches;
    const std::uint64_t barriers = after.barriers - before.barriers;
    std::printf("P-frame %zu: %llu dispatches, %llu barriers (%s)\n", i,
                static_cast<unsigned long long>(dispatches),
                static_cast<unsigned long long>(barriers),
                gpu.physical.device_mapped_cached() ? "in place" : "copied");
    CHECK(after.submits - before.submits == 1);
    CHECK(dispatches == kPinnedDispatches);
    // A copied ring adds the barrier before the batch's output copies.
    CHECK(barriers ==
          kPinnedBarriers + (gpu.physical.device_mapped_cached() ? 0 : 1));
  }
  return 0;
}

// --- The growing calls.
// ---------------------------------------------------------

// record_growing grows a grid whose heap the frame outgrows before it
// records, with its "  ..grow" row; finish_growing then has nothing to redo.
// A table that cannot place the frame (chains one long) fails on the device:
// finish_growing grows the grid and decodes the frame again from the pending
// frame's own slot while the caller holds the other, and the extract
// recorded after it ran gated off, so the caller meshes again.
int growing_case(Gpu& gpu) {
  vkc::Result<codec::Encoder> enc = encoder(gpu, 20, 0.0f, 1);
  CHECK(enc.ok());
  vkc::Result<std::vector<Bytes>> frames = chain(gpu, *enc, 1);
  CHECK(frames.ok());
  const Bytes& f = (*frames)[0];
  vkc::Result<codec::FrameInfo> info =
      codec::read_frame_info(f.data(), f.size());
  vkc::Result<vol::VoxelBlockGrid> roomy = make_grid(gpu);
  vkc::Result<codec::Decoder> ref = decoder(gpu);
  CHECK(info.ok() && roomy.ok() && ref.ok());
  CHECK(ref->decode(f.data(), f.size(), *roomy).ok());
  vkc::Result<Snapshot> want = snapshot(gpu, *roomy);
  CHECK(want.ok());

  {
    GridShape few;
    few.num_buckets = 2;
    vkc::Result<vol::VoxelBlockGrid> grid = make_grid(gpu, few);
    vkc::Result<codec::Decoder> dec = decoder(gpu);
    CHECK(grid.ok() && dec.ok());
    vkc::Result<vkc::HostSlot> slot = slot_of(*dec, f);
    CHECK(slot.ok());
    vkc::StageMetrics m;
    vkc::CommandBatch batch(gpu.device, gpu.allocator);
    vkc::Result<codec::PendingDecode> pending = codec::record_growing(
        *dec, batch, *slot, f.size(), *grid, kMaxBlocks, &m);
    CHECK(pending.ok() && find_row(m, "  ..grow") != nullptr);
    CHECK(grid->grid().num_buckets > few.num_buckets);
    CHECK(batch.submit().ok());
    vkc::Result<codec::Finished> done =
        codec::finish_growing(*dec, *pending, *grid, kMaxBlocks);
    CHECK(done.ok() && !done->redone);
    CHECK(snapshot(gpu, *grid).value() == *want);
  }

  GridShape cramped;
  cramped.num_buckets = std::int32_t((info->block_count + 7) / 8 + 1);
  cramped.max_chain = 1;
  vkc::Result<vol::VoxelBlockGrid> grid = make_grid(gpu, cramped);
  vkc::Result<codec::Decoder> dec = decoder(gpu);
  vkc::Result<mesh::MarchingCubes> mc =
      mesh::MarchingCubes::create(gpu.device, gpu.allocator);
  CHECK(grid.ok() && dec.ok() && mc.ok());
  vkc::Result<vkc::HostSlot> slot = slot_of(*dec, f);
  CHECK(slot.ok());
  vkc::CommandBatch batch(gpu.device, gpu.allocator);
  vkc::Result<codec::PendingDecode> pending =
      codec::record_growing(*dec, batch, *slot, f.size(), *grid, kMaxBlocks);
  CHECK(pending.ok());
  vkc::Result<mesh::PendingExtract> extract =
      mc->record_extract(batch, *grid, pending->blocks, &pending->gate);
  CHECK(extract.ok());
  vkc::Result<vkc::PendingBatch> ran =
      batch.submit_async({}, {extract->signal});
  CHECK(ran.ok() && ran->wait().ok());
  // The next frame's slot, held while the frame is decoded again.
  vkc::Result<vkc::HostSlot> next = dec->claim_input(f.size());
  CHECK(next.ok());
  vkc::StageMetrics m;
  vkc::Result<codec::Finished> done =
      codec::finish_growing(*dec, *pending, *grid, kMaxBlocks, &m);
  CHECK(done.ok() && done->redone);
  CHECK(find_row(m, "  ..grow") != nullptr);
  CHECK(grid->grid().num_buckets > cramped.num_buckets);
  CHECK(snapshot(gpu, *grid).value() == *want);
  vkc::Result<std::optional<mesh::DeviceMesh>> gated =
      mc->finish(*extract, *grid);
  CHECK(gated.ok() && !gated->has_value());
  vkc::Result<mesh::DeviceMesh> again = mc->extract_device(*grid);
  CHECK(again.ok() && !again->empty());
  return 0;
}

int gpu_main(vr_test::GpuContext& gpu) {
  if (equals_blocking_case(gpu) != 0) return 1;
  if (refusals_case(gpu) != 0) return 1;
  if (device_refusal_case(gpu) != 0) return 1;
  if (tokens_case(gpu) != 0) return 1;
  if (rows_case(gpu) != 0) return 1;
  if (counters_case(gpu) != 0) return 1;
  if (pin_case(gpu) != 0) return 1;
  if (growing_case(gpu) != 0) return 1;
  std::printf("recon codec recorded test passed\n");
  return 0;
}

}  // namespace

int main() { return vr_test::run_on_gpu(gpu_main); }
