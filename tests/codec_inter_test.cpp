// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// GPU test for decoding on the device: the reference store's apply against
// independent uint16 arithmetic, at every source kind and K, in one dispatch
// or several, as its slots grow, and its sources past what is bound counted
// and refused; and the Decoder's frames -- the Encoder's, and chains of an
// I-frame, then P-frames, new blocks at int32's extremes among them --
// leaving the grid exactly as the CPU oracle's decode of the same content
// does, bit for bit, with every mode, new blocks and the empty cases; a
// P-frame whose reference is not held refused as NotFound, a segment longer
// than the format allows refused, every refusal and failure leaving the
// reference as it was, and the device buffers grown and shrunk frame to
// frame. Internal headers by include path; skips with no device.

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "bitstream.hpp"
#include "codec_fixture.hpp"
#include "codec_frames.hpp"
#include "codec_oracle.hpp"
#include "codec_oracle_device.hpp"
#include "dct_transform.hpp"
#include "reference_store.hpp"
#include "volumetric_kit/core/vulkan/command_batch.hpp"
#include "volumetric_kit/recon/codec/decoder.hpp"
#include "volumetric_kit/recon/codec/encoder.hpp"

#include "gpu_test.hpp"

namespace vkc = volumetric_kit::core;
namespace codec = volumetric_kit::recon::codec;
namespace d = volumetric_kit::recon::codec::detail;
namespace oracle = codec_oracle;
using namespace codec_fixture;
using codec_frames::append_payload;
using codec_frames::apply_residual;
using codec_frames::inter_frame;
using codec_frames::Lcg;
using codec_frames::make_frame;
using codec_frames::Payload;
using codec_frames::payload_at;
using codec_frames::reference_of;
using codec_frames::successor;
using codec_oracle::Resident;
using codec_oracle::resident;
using C = vkc::Status::Code;
using Bytes = std::vector<std::uint8_t>;

namespace {

constexpr std::uint32_t kMaxBlocks = 1u << 20;

// --- The store. -------------------------------------------------------------

// @p n entries of K coefficients drawn from all of int16, and masks of every
// kind.
oracle::DctBlocks random_entries(std::size_t n, std::uint32_t k, Lcg& rng) {
  oracle::DctBlocks b;
  b.params.coefficient_count = k;
  b.trunc_dist = kTrunc;
  for (std::size_t i = 0; i < n; ++i) {
    for (std::uint32_t w = 0; w < codec::kMaskWordsPerBlock; ++w) {
      b.masks.push_back(i % 3 == 0 ? ~0u : rng.next());
    }
    for (std::uint32_t j = 0; j < k; ++j) {
      b.coefficients.push_back(static_cast<std::int16_t>(rng.next()));
    }
  }
  return b;
}

// Strictly increasing coordinates, @p n of them.
std::vector<vr::Vec3i> row_of(std::size_t n) {
  std::vector<vr::Vec3i> out;
  for (std::size_t i = 0; i < n; ++i) out.push_back(vr::Vec3i(int(i), 0, 0));
  return out;
}

// A frame's sources and blocks written where the merge writes them, the
// store's frame buffers, after a reserve.
vkc::Status stage_frame(Gpu& gpu, const d::ReferenceStore& store,
                        const std::vector<d::ApplySource>& sources,
                        const std::vector<vr::Vec3i>& coords) {
  const d::FrameBuffers frame = store.frame_buffers();
  std::vector<vol::BlockIndex> list;
  for (const vr::Vec3i& c : coords) list.push_back(vol::BlockIndex{c, 0});
  if (!sources.empty()) {
    VKC_TRY(vr_test::write_back(gpu.device, gpu.allocator, *frame.sources,
                                sources));
  }
  if (!list.empty()) {
    VKC_TRY(vr_test::write_back(gpu.device, gpu.allocator, *frame.list, list));
  }
  return {};
}

// The held reference's coordinates, read back from the store's list.
std::vector<vr::Vec3i> held_coords(Gpu& gpu, const d::ReferenceStore& store) {
  std::vector<vr::Vec3i> out;
  if (store.held() == nullptr) return out;
  const std::uint32_t m = store.held()->header.block_count;
  if (m == 0) return out;
  for (const vol::BlockIndex& b :
       vr_test::read_back<vol::BlockIndex>(gpu.device, gpu.allocator,
                                           store.reference_list(), m)
           .value()) {
    out.push_back(b.coord);
  }
  return out;
}

// The spare as the last apply left it, unpacked: each block's payload, and
// whether every odd K's pad half was written 0.
struct Applied {
  std::vector<Payload> blocks;
  std::vector<vr::Vec3i> coords;
  bool pads_zero = true;
};

vkc::Result<Applied> read_spare(Gpu& gpu, const d::ResidentBlocks& spare) {
  const std::uint32_t k = spare.coefficient_count;
  const std::uint32_t words = d::coefficient_words(k);
  Applied out;
  if (spare.count == 0) return out;
  VKC_ASSIGN(const std::vector<std::uint32_t> masks,
             vr_test::read_back<std::uint32_t>(
                 gpu.device, gpu.allocator, *spare.masks,
                 std::size_t(spare.count) * codec::kMaskWordsPerBlock));
  VKC_ASSIGN(const std::vector<std::uint32_t> packed,
             vr_test::read_back<std::uint32_t>(
                 gpu.device, gpu.allocator, *spare.coefficients,
                 std::size_t(spare.count) * words));
  VKC_ASSIGN(const std::vector<vol::BlockIndex> list,
             vr_test::read_back<vol::BlockIndex>(gpu.device, gpu.allocator,
                                                 *spare.list, spare.count));
  for (std::size_t i = 0; i < spare.count; ++i) {
    Payload p;
    p.mask.assign(masks.begin() + std::ptrdiff_t(i * 16),
                  masks.begin() + std::ptrdiff_t(i * 16 + 16));
    for (std::uint32_t j = 0; j < k; ++j) {
      p.q.push_back(static_cast<std::int16_t>(packed[i * words + j / 2] >>
                                              (16 * (j & 1))));
    }
    if (k % 2 != 0 && (packed[i * words + words - 1] >> 16) != 0) {
      out.pads_zero = false;
    }
    out.blocks.push_back(std::move(p));
    out.coords.push_back(list[i].coord);
    if (list[i].ptr != 0) out.pads_zero = false;
  }
  return out;
}

// What each source kind makes, in plain uint16 arithmetic: keep adds the
// residual to the reference mod 2^16 and XORs the mask, skip is the
// reference, and the payload alone is itself.
Payload expected(const Payload* reference, const Payload* payload) {
  if (reference == nullptr) return *payload;
  if (payload == nullptr) return *reference;
  Payload t = *reference;
  for (std::size_t j = 0; j < t.q.size(); ++j) {
    t.q[j] = apply_residual(reference->q[j], payload->q[j]);
  }
  for (std::size_t w = 0; w < t.mask.size(); ++w) {
    t.mask[w] = reference->mask[w] ^ payload->mask[w];
  }
  return t;
}

// An intra frame's apply, committed: what the store holds after decoding it.
int commit_intra(Gpu& gpu, d::ReferenceStore& store, const oracle::DctBlocks& r,
                 std::uint64_t hash) {
  const std::size_t m = r.masks.size() / codec::kMaskWordsPerBlock;
  const std::uint32_t k = r.params.coefficient_count;
  std::vector<d::ApplySource> sources(m);
  for (std::size_t i = 0; i < m; ++i) {
    sources[i] = {d::kNoSource, static_cast<std::uint32_t>(i)};
  }
  CHECK(store.reserve(std::uint32_t(m), k).ok());
  CHECK(stage_frame(gpu, store, sources, row_of(m)).ok());
  vkc::Result<Resident> up = resident(gpu, r);
  CHECK(up.ok());
  vkc::CommandBatch batch(gpu.device, gpu.allocator);
  vkc::Result<d::ResidentBlocks> spare =
      store.record_apply(batch, up.value().blocks, std::uint32_t(m));
  CHECK(spare.ok());
  CHECK(batch.submit().ok() && store.check_apply().ok());
  vkc::Result<Applied> got = read_spare(gpu, spare.value());
  CHECK(got.ok() && got.value().pads_zero);
  for (std::size_t i = 0; i < m; ++i) {
    CHECK(got.value().blocks[i] == payload_at(r, i));
  }
  d::FrameReference id;
  id.hash = hash;
  id.header.block_count = static_cast<std::uint32_t>(m);
  id.header.params.coefficient_count = k;
  store.commit(id);
  return 0;
}

// Every source kind against a held reference, its payload where the device
// reader leaves it, pads not 0, at K of 1, 21, 64 and 512, with residuals
// from all of int16 so many wrap; at @p max_blocks_per_dispatch 4, the nine
// blocks in three dispatches.
int apply_case(Gpu& gpu, std::uint32_t max_blocks_per_dispatch) {
  vkc::Result<std::unique_ptr<d::ReferenceStore>> made =
      d::ReferenceStore::create(gpu.device, gpu.allocator,
                                max_blocks_per_dispatch);
  CHECK(made.ok());
  d::ReferenceStore& store = *made.value();
  CHECK(store.held() == nullptr);
  Lcg rng{17};
  std::uint64_t hash = 1;
  for (std::uint32_t k : {1u, 21u, 64u, codec::kVoxelsPerBlock}) {
    const std::size_t m = 9;
    const oracle::DctBlocks r = random_entries(m, k, rng);
    CHECK(commit_intra(gpu, store, r, hash) == 0);
    CHECK(store.held() != nullptr && store.held()->hash == hash);
    CHECK(held_coords(gpu, store) == row_of(m));

    // A P-frame's payload by stream position: M entries, then two new.
    const oracle::DctBlocks payload = random_entries(m + 2, k, rng);
    // keep, skip, zero, keep, new, skip, zero, keep, new: the reference's
    // entries 0, 1, 2, 4, 5, 6, 8 survive, 3 and 7 are gone.
    const std::vector<d::ApplySource> sources = {{0, 0},
                                                 {1, d::kNoSource},
                                                 {d::kNoSource, 2},
                                                 {4, 4},
                                                 {d::kNoSource, 9},
                                                 {5, d::kNoSource},
                                                 {d::kNoSource, 6},
                                                 {8, 8},
                                                 {d::kNoSource, 10}};
    const std::vector<vr::Vec3i> coords = row_of(sources.size());
    CHECK(store.reserve(std::uint32_t(sources.size()), k).ok());
    CHECK(stage_frame(gpu, store, sources, coords).ok());
    vkc::Result<Resident> up = resident(gpu, payload, {}, {}, 0xBEEF);
    CHECK(up.ok());
    d::ResidentBlocks entries = up.value().blocks;
    entries.reference_count = std::uint32_t(m);
    vkc::CommandBatch batch(gpu.device, gpu.allocator);
    vkc::Result<d::ResidentBlocks> spare =
        store.record_apply(batch, entries, std::uint32_t(sources.size()));
    CHECK(spare.ok());
    CHECK(batch.submit().ok() && store.check_apply().ok());
    vkc::Result<Applied> got = read_spare(gpu, spare.value());
    CHECK(got.ok());
    CHECK(got.value().pads_zero);
    CHECK(got.value().coords == coords);
    for (std::size_t i = 0; i < sources.size(); ++i) {
      const d::ApplySource& src = sources[i];
      const Payload ref = src.reference != d::kNoSource
                              ? payload_at(r, src.reference)
                              : Payload{};
      const Payload pay = src.payload != d::kNoSource
                              ? payload_at(payload, src.payload)
                              : Payload{};
      const Payload want =
          expected(src.reference != d::kNoSource ? &ref : nullptr,
                   src.payload != d::kNoSource ? &pay : nullptr);
      if (!(got.value().blocks[i] == want)) {
        std::fprintf(stderr, "K %u, block %zu\n", k, i);
        return 1;
      }
    }
    // Applied and submitted, but not committed: the reference is as it was.
    CHECK(store.held()->hash == hash && held_coords(gpu, store) == row_of(m));
    ++hash;
  }
  return 0;
}

// The slots grow under a held reference, whose words a growing spare never
// disturbs; and only commit changes what is held, whatever was recorded or
// refused before it. Seven blocks a dispatch, so 50 blocks take eight
// dispatches, the last of one block, and 700 take 100.
int grow_case(Gpu& gpu) {
  vkc::Result<std::unique_ptr<d::ReferenceStore>> made =
      d::ReferenceStore::create(gpu.device, gpu.allocator, 7);
  CHECK(made.ok());
  d::ReferenceStore& store = *made.value();
  Lcg rng{23};
  const std::uint32_t k = 64;
  oracle::DctBlocks held = random_entries(3, k, rng);
  CHECK(commit_intra(gpu, store, held, 100) == 0);
  for (const std::size_t n : {std::size_t(50), std::size_t(700)}) {
    // Every block a skip of a held one, then the rest new: a frame of n
    // blocks, the spare grown to hold it, its payload by stream position.
    const std::size_t m = store.held()->header.block_count;
    const oracle::DctBlocks fresh = random_entries(n, k, rng);
    std::vector<d::ApplySource> sources;
    for (std::size_t e = 0; e < m; ++e) {
      sources.push_back({std::uint32_t(e), d::kNoSource});
    }
    for (std::size_t q = 0; q < n - m; ++q) {
      sources.push_back({d::kNoSource, std::uint32_t(m + q)});
    }
    CHECK(store.reserve(std::uint32_t(n), k).ok());
    CHECK(stage_frame(gpu, store, sources, row_of(n)).ok());
    vkc::Result<Resident> up = resident(gpu, fresh);
    CHECK(up.ok());
    d::ResidentBlocks entries = up.value().blocks;
    entries.reference_count = std::uint32_t(m);
    vkc::CommandBatch batch(gpu.device, gpu.allocator);
    vkc::Result<d::ResidentBlocks> spare =
        store.record_apply(batch, entries, std::uint32_t(n));
    CHECK(spare.ok());
    CHECK(batch.submit().ok() && store.check_apply().ok());
    vkc::Result<Applied> got = read_spare(gpu, spare.value());
    CHECK(got.ok() && got.value().coords == row_of(n));
    oracle::DctBlocks next;
    next.params.coefficient_count = k;
    for (std::size_t i = 0; i < n; ++i) {
      const Payload want = i < m ? payload_at(held, i) : payload_at(fresh, i);
      CHECK(got.value().blocks[i] == want);
      append_payload(next, want);
    }
    // Refused -- twice the blocks, past the spare's headroom -- and recorded
    // but never committed: nothing changes.
    const d::FrameReference before = *store.held();
    vkc::CommandBatch refused(gpu.device, gpu.allocator);
    CHECK(store.record_apply(refused, entries, 2 * std::uint32_t(n))
              .status()
              .domain() == C::InvalidArgument);
    CHECK(store.held()->hash == before.hash &&
          held_coords(gpu, store) == row_of(m));
    d::FrameReference id;
    id.hash = before.hash + 1;
    id.header.block_count = std::uint32_t(n);
    id.header.params.coefficient_count = k;
    store.commit(id);
    CHECK(store.held()->hash == id.hash &&
          held_coords(gpu, store) == row_of(n));
    held = std::move(next);
  }
  return 0;
}

// What the apply refuses: on the host, before it records anything, a
// P-frame on another reference's M or K, more blocks than reserved and no
// payload; and on the device, a source past what is bound -- a reference
// entry past M, or any for an intra frame, a payload entry past the
// payload -- read as absent, counted and refused after the submit.
int apply_refusals_case(Gpu& gpu) {
  vkc::Result<std::unique_ptr<d::ReferenceStore>> made =
      d::ReferenceStore::create(gpu.device, gpu.allocator);
  CHECK(made.ok());
  d::ReferenceStore& store = *made.value();
  Lcg rng{29};
  CHECK(commit_intra(gpu, store, random_entries(4, 8, rng), 7) == 0);
  vkc::Result<Resident> up = resident(gpu, random_entries(4, 8, rng));
  CHECK(up.ok());
  const std::uint32_t none = d::kNoSource;
  CHECK(store.reserve(4, 8).ok());
  // The payload, its first @p m entries the reference's, at @p k over
  // @p sources; the host's refusal, or the device's after the submit.
  auto refused = [&](const std::vector<d::ApplySource>& sources,
                     std::uint32_t m, std::uint32_t k, std::uint32_t blocks,
                     const char* says) {
    if (!stage_frame(gpu, store, sources, row_of(sources.size())).ok()) {
      return false;
    }
    vkc::CommandBatch batch(gpu.device, gpu.allocator);
    d::ResidentBlocks payload = up.value().blocks;
    payload.reference_count = m;
    payload.coefficient_count = k;
    vkc::Result<d::ResidentBlocks> r =
        store.record_apply(batch, payload, blocks);
    vkc::Status s = r.status();
    if (r.ok()) {
      s = batch.submit();
      if (s.ok()) s = store.check_apply();
    }
    return !s.ok() && s.domain() == C::InvalidArgument &&
           s.message().find(says) != std::string::npos;
  };
  CHECK(refused({{0, none}, {4, none}}, 4, 8, 2,
                "1 of the frame's blocks read past"));
  CHECK(refused({{0, none}, {1, 1}}, 0, 8, 2,
                "2 of the frame's blocks read past"));
  CHECK(refused({{none, 4}}, 4, 8, 1, "1 of the frame's blocks read past"));
  CHECK(refused({{0, 0}}, 4, 9, 1, "a P-frame of 4 reference entries at K 9"));
  CHECK(refused({{0, 0}}, 3, 8, 1, "on a reference of 4 at K 8"));
  CHECK(refused(std::vector<d::ApplySource>(4, {0, none}), 4, 8, 100,
                "more than were reserved"));
  vkc::CommandBatch batch(gpu.device, gpu.allocator);
  CHECK(store.record_apply(batch, d::ResidentBlocks{}, 1)
            .status()
            .message()
            .find("no payload buffers") != std::string::npos);
  CHECK(store.reserve(1u << 31, 8).domain() == C::InvalidArgument);
  CHECK(store.held()->hash == 7 && held_coords(gpu, store) == row_of(4));
  // A fault is the call's: the next apply, in range, is clean.
  CHECK(stage_frame(gpu, store, {{0, none}}, row_of(1)).ok());
  d::ResidentBlocks payload = up.value().blocks;
  payload.reference_count = 4;
  vkc::CommandBatch clean(gpu.device, gpu.allocator);
  CHECK(store.record_apply(clean, payload, 1).ok());
  CHECK(clean.submit().ok() && store.check_apply().ok());
  return 0;
}

// --- The Decoder. -----------------------------------------------------------

// A grid built from a frame's header, as a player builds one.
vkc::Result<vol::VoxelBlockGrid> grid_for(Gpu& gpu, const Bytes& frame,
                                          std::int32_t num_buckets = 512) {
  VKC_ASSIGN(const codec::FrameInfo info,
             codec::read_frame_info(frame.data(), frame.size()));
  GridShape shape;
  shape.voxel_size = info.voxel_size;
  shape.trunc_dist = info.trunc_dist;
  shape.num_buckets = num_buckets;
  return make_grid(gpu, shape);
}

// What an intra frame decodes to by the oracle: the CPU rANS reader's
// entries (oracle::read_frame, which pins the format's bytes),
// inverse-transformed into a fresh grid of the frame's blocks -- no device
// reader, no apply, no reference.
vkc::Result<Snapshot> oracle_decode(Gpu& gpu, const Bytes& frame) {
  VKC_ASSIGN(const oracle::Frame content,
             oracle::read_frame(frame.data(), frame.size(), kMaxBlocks));
  VKC_ASSIGN(vol::VoxelBlockGrid grid, grid_for(gpu, frame));
  VKC_TRY(allocate(grid, content.coords));
  // The grid's blocks in frame order, entry for entry with the content's.
  VKC_ASSIGN(const std::vector<vol::BlockIndex> blocks, active_sorted(grid));
  VKC_ASSIGN(std::unique_ptr<d::DctTransform> transform,
             d::DctTransform::create(gpu.device, gpu.allocator));
  VKC_TRY(oracle::inverse(gpu, *transform, grid, blocks, content.blocks));
  return snapshot(gpu, grid);
}

// Whether @p grid holds what @p frame, an intra frame, decodes to.
bool holds(Gpu& gpu, vol::VoxelBlockGrid& grid, const Bytes& frame) {
  vkc::Result<Snapshot> want = oracle_decode(gpu, frame);
  vkc::Result<Snapshot> got = snapshot(gpu, grid);
  return want.ok() && got.ok() && want.value() == got.value();
}

oracle::FrameWriteOptions sizes(std::uint32_t r, std::uint32_t u) {
  oracle::FrameWriteOptions o;
  o.segment_size = r;
  o.update_size = u;
  return o;
}

// One step of a chain: the content a frame decodes to, its intra frame, and
// the frame sent, an I-frame or a P-frame against the step before.
struct Step {
  oracle::Frame content;
  Bytes intra;
  Bytes sent;
};

// The P-frame of @p next against @p prev's sent bytes, at @p options, with
// @p overrides' modes.
Step p_step(const Step& prev, oracle::Frame next,
            const oracle::FrameWriteOptions& options,
            const std::map<std::size_t, d::BlockMode>& overrides = {}) {
  Step s;
  s.content = std::move(next);
  s.intra = oracle::write_frame(s.content).value();
  s.sent = oracle::write_frame(inter_frame(reference_of(prev.sent),
                                           prev.content, s.content, overrides),
                               options)
               .value();
  return s;
}

Step i_step(const Bytes& intra) {
  Step s;
  s.content =
      oracle::read_frame(intra.data(), intra.size(), kMaxBlocks).value();
  s.intra = intra;
  s.sent = intra;
  return s;
}

// Decode a chain's frames in order into one grid, which after each must hold
// the oracle's decode of that step's intra frame.
int run_chain(Gpu& gpu, const std::vector<Step>& chain, const char* what) {
  vkc::Result<codec::Decoder> dec =
      codec::Decoder::create(gpu.device, gpu.allocator);
  vkc::Result<vol::VoxelBlockGrid> grid = grid_for(gpu, chain[0].sent);
  CHECK(dec.ok() && grid.ok());
  for (std::size_t i = 0; i < chain.size(); ++i) {
    const Bytes& f = chain[i].sent;
    const vkc::Status s = dec.value().decode(f.data(), f.size(), grid.value());
    if (!s.ok() || !holds(gpu, grid.value(), chain[i].intra)) {
      std::fprintf(stderr, "%s: step %zu: %s\n", what, i,
                   s.ok() ? "differs" : s.message().c_str());
      return 1;
    }
  }
  return 0;
}

// The modes a P-frame holds, and its new blocks.
struct Census {
  std::size_t modes[5] = {};
  std::size_t added = 0;
};

Census census(const Step& prev, const Step& p) {
  const d::FrameReference id = reference_of(prev.sent);
  const oracle::Frame f =
      oracle::read_frame(p.sent.data(), p.sent.size(), kMaxBlocks, &id).value();
  Census c;
  for (d::BlockMode m : f.modes) ++c.modes[std::size_t(m)];
  c.added = f.coords.size();
  return c;
}

// Synthetic chains: every mode and new blocks; then no new blocks (A = 0);
// every block removed (N = 0); and only new ones, against that empty frame
// (M = 0) -- at odd K, K = 64 and K = 512, update and new segments of
// several sizes; and the extreme coefficients, whose differences wrap and,
// at +-32768, code zero.
int synthetic_case(Gpu& gpu) {
  for (std::uint32_t k : {21u, 64u, codec::kVoxelsPerBlock}) {
    const std::size_t n = k == codec::kVoxelsPerBlock ? 30 : 90;
    std::vector<Step> chain;
    chain.push_back(
        i_step(oracle::write_frame(make_frame(n, k, 500 + k)).value()));
    const oracle::Frame f1 = successor(chain[0].content, k, 80, 400, n / 5);
    chain.push_back(p_step(chain[0], f1, sizes(3, 7),
                           codec_frames::spellings(chain[0].content, f1)));
    chain.push_back(p_step(chain[1],
                           successor(chain[1].content, k + 1, 60, 300, 0),
                           sizes(64, 1)));
    chain.push_back(p_step(
        chain[2], successor(chain[2].content, k + 2, 1000, 0, 0), sizes(5, 5)));
    chain.push_back(
        p_step(chain[3], make_frame(n / 3, k, 600 + k), sizes(2, 64)));
    // The premise: every mode and new blocks, A = 0, N = 0, M = 0.
    const Census first = census(chain[0], chain[1]);
    for (std::size_t m = 0; m < 4; ++m) CHECK(first.modes[m] > 0);
    CHECK(first.added > 0);
    CHECK(census(chain[1], chain[2]).added == 0);
    CHECK(chain[3].content.coords.empty());
    CHECK(census(chain[3], chain[4]).added == n / 3);
    std::string what = "synthetic, K " + std::to_string(k);
    CHECK(run_chain(gpu, chain, what.c_str()) == 0);

    const codec_frames::Extremes x = codec_frames::extremes(k);
    std::vector<Step> extreme;
    extreme.push_back(i_step(oracle::write_frame(x.reference).value()));
    extreme.push_back(p_step(extreme[0], x.target, sizes(4, 3)));
    CHECK(
        census(extreme[0], extreme[1]).modes[std::size_t(d::BlockMode::kZero)] >
        0);
    what = "extremes, K " + std::to_string(k);
    CHECK(run_chain(gpu, extreme, what.c_str()) == 0);
  }
  return 0;
}

// A P-frame whose new blocks are at int32's extremes, deltas of 2^32 - 1
// among them, beside a reference that keeps, skips, zeroes and removes; then
// back to the reference, which drops them: the merge and the grid diff
// search them as any other.
int extremes_chain_case(Gpu& gpu) {
  const oracle::Frame reference = make_frame(60, 4, 31);
  const oracle::Frame kept = successor(reference, 32, 100, 400, 0);
  const oracle::Frame added = codec_frames::extreme_frame();
  std::map<vr::Vec3i, Payload, codec_frames::ZyxLess> blocks;
  for (std::size_t i = 0; i < kept.coords.size(); ++i) {
    blocks.emplace(kept.coords[i], payload_at(kept.blocks, i));
  }
  for (std::size_t i = 0; i < added.coords.size(); ++i) {
    blocks.emplace(added.coords[i], payload_at(added.blocks, i));
  }
  oracle::Frame target = make_frame(0, 4, 1);
  for (const auto& [c, payload] : blocks) {
    target.coords.push_back(c);
    append_payload(target.blocks, payload);
  }
  std::vector<Step> chain;
  chain.push_back(i_step(oracle::write_frame(reference).value()));
  chain.push_back(p_step(chain[0], target, sizes(2, 5)));
  chain.push_back(p_step(chain[1], reference, sizes(3, 7)));
  CHECK(census(chain[0], chain[1]).added == added.coords.size());
  return run_chain(gpu, chain, "int32 extremes");
}

// The Encoder's frames of one sphere at an odd K, at the default segments
// and finer ones, each into a fresh grid: as the oracle decodes them.
int encoder_case(Gpu& gpu) {
  vkc::Result<vol::VoxelBlockGrid> g =
      sphere_grid(gpu, Sphere{vr::Vec3f(0.01f, -0.02f, 0.03f), 0.09f});
  vkc::Result<codec::Decoder> dec =
      codec::Decoder::create(gpu.device, gpu.allocator);
  CHECK(g.ok() && dec.ok());
  for (std::uint32_t segment_size : {64u, 3u}) {
    codec::EncoderConfig config;
    config.params.coefficient_count = 37;
    config.segment_size = segment_size;
    vkc::Result<codec::Encoder> enc =
        codec::Encoder::create(gpu.device, gpu.allocator, config);
    CHECK(enc.ok());
    vkc::Result<Bytes> frame = enc.value().encode(g.value());
    CHECK(frame.ok());
    CHECK(codec::read_frame_info(frame.value().data(), frame.value().size())
              .value()
              .block_count > 0);
    vkc::Result<vol::VoxelBlockGrid> player = grid_for(gpu, frame.value());
    CHECK(player.ok());
    CHECK(
        dec.value()
            .decode(frame.value().data(), frame.value().size(), player.value())
            .ok());
    CHECK(holds(gpu, player.value(), frame.value()));
  }
  return 0;
}

// I-frames from the Encoder, of a sphere that moves and grows.
struct Spheres {
  std::vector<Step> chain;  // I, P, P
};

vkc::Result<Bytes> encode_sphere(Gpu& gpu, const Sphere& s) {
  VKC_ASSIGN(vol::VoxelBlockGrid g, sphere_grid(gpu, s));
  codec::EncoderConfig config;
  config.segment_size = 3;
  VKC_ASSIGN(codec::Encoder enc,
             codec::Encoder::create(gpu.device, gpu.allocator, config));
  return enc.encode(g);
}

vkc::Result<Spheres> spheres(Gpu& gpu) {
  Spheres out;
  std::vector<Bytes> intra;
  for (int t = 0; t < 3; ++t) {
    VKC_ASSIGN(Bytes frame,
               encode_sphere(gpu, Sphere{vr::Vec3f(0.006f * float(t), 0.0f,
                                                   0.002f * float(t)),
                                         0.1f + 0.008f * float(t)}));
    intra.push_back(std::move(frame));
  }
  out.chain.push_back(i_step(intra[0]));
  // Long segments, then short ones.
  const oracle::FrameWriteOptions few = sizes(64, 64);
  const oracle::FrameWriteOptions many = sizes(1, 2);
  for (int t = 1; t < 3; ++t) {
    Step s = p_step(out.chain.back(), i_step(intra[std::size_t(t)]).content,
                    t == 1 ? few : many);
    s.intra = intra[std::size_t(t)];  // the target's own I-frame
    out.chain.push_back(std::move(s));
  }
  return out;
}

// The sphere chain; the last P-frame into a fresh grid built from its own
// header.
int sphere_case(Gpu& gpu, const Spheres& s) {
  const std::vector<Step>& chain = s.chain;
  const Census c = census(chain[0], chain[1]);
  CHECK(c.modes[std::size_t(d::BlockMode::kKeep)] > 0);
  CHECK(c.modes[std::size_t(d::BlockMode::kRemove)] > 0);
  CHECK(c.added > 0);
  CHECK(run_chain(gpu, chain, "spheres") == 0);

  vkc::Result<codec::Decoder> dec =
      codec::Decoder::create(gpu.device, gpu.allocator);
  vkc::Result<vol::VoxelBlockGrid> g = grid_for(gpu, chain[0].sent);
  CHECK(dec.ok() && g.ok());
  for (int i = 0; i < 2; ++i) {
    const Bytes& f = chain[std::size_t(i)].sent;
    CHECK(dec.value().decode(f.data(), f.size(), g.value()).ok());
  }
  const Bytes& p = chain[2].sent;
  vkc::Result<codec::FrameInfo> info =
      codec::read_frame_info(p.data(), p.size());
  CHECK(info.ok() && info.value().type == codec::FrameType::kInter);
  CHECK(info.value().block_count == chain[2].content.coords.size());
  vkc::Result<vol::VoxelBlockGrid> fresh = grid_for(gpu, p);
  CHECK(fresh.ok());
  CHECK(dec.value().decode(p.data(), p.size(), fresh.value()).ok());
  CHECK(holds(gpu, fresh.value(), chain[2].intra));
  return 0;
}

// A refusal: its code, and the grid as it was.
int refused(Gpu& gpu, codec::Decoder& dec, const Bytes& frame,
            vol::VoxelBlockGrid& grid, C code, const char* says) {
  vkc::Result<Snapshot> before = snapshot(gpu, grid);
  CHECK(before.ok());
  const vkc::Status s = dec.decode(frame.data(), frame.size(), grid);
  if (s.ok() || s.domain() != code ||
      s.message().find(says) == std::string::npos) {
    std::fprintf(stderr, "expected a refusal with \"%s\", got: %s\n", says,
                 s.ok() ? "OK" : s.message().c_str());
    return 1;
  }
  vkc::Result<Snapshot> after = snapshot(gpu, grid);
  CHECK(after.ok() && after.value() == before.value());
  return 0;
}

// NotFound, with the grid and the reference untouched and the chain going on:
// no reference held; a later I-frame held; another decoder's reference.
int not_found_case(Gpu& gpu, const Spheres& s) {
  const std::vector<Step>& chain = s.chain;
  vkc::Result<codec::Decoder> made =
      codec::Decoder::create(gpu.device, gpu.allocator);
  vkc::Result<vol::VoxelBlockGrid> g = grid_for(gpu, chain[0].sent);
  CHECK(made.ok() && g.ok());
  codec::Decoder& dec = made.value();
  vol::VoxelBlockGrid& grid = g.value();
  CHECK(refused(gpu, dec, chain[1].sent, grid, C::NotFound,
                "no reference frame is held") == 0);
  CHECK(dec.decode(chain[0].sent.data(), chain[0].sent.size(), grid).ok());
  CHECK(dec.decode(chain[1].sent.data(), chain[1].sent.size(), grid).ok());
  CHECK(holds(gpu, grid, chain[1].intra));

  // A later I-frame replaces the reference: the P-frame on the one before
  // is refused, and one on the I-frame decodes.
  CHECK(dec.decode(chain[2].intra.data(), chain[2].intra.size(), grid).ok());
  CHECK(refused(gpu, dec, chain[2].sent, grid, C::NotFound,
                "not the one held") == 0);
  const Step again =
      p_step(i_step(chain[2].intra), chain[1].content, sizes(4, 4));
  CHECK(dec.decode(again.sent.data(), again.sent.size(), grid).ok());
  CHECK(holds(gpu, grid, chain[1].intra));

  // Each decoder holds its own: a second one that never saw the I-frame
  // refuses what the first decodes.
  vkc::Result<codec::Decoder> other =
      codec::Decoder::create(gpu.device, gpu.allocator);
  vkc::Result<codec::Decoder> first =
      codec::Decoder::create(gpu.device, gpu.allocator);
  vkc::Result<vol::VoxelBlockGrid> og = grid_for(gpu, chain[0].sent);
  CHECK(other.ok() && first.ok() && og.ok());
  CHECK(first.value()
            .decode(chain[0].sent.data(), chain[0].sent.size(), og.value())
            .ok());
  vkc::Result<vol::VoxelBlockGrid> fresh = grid_for(gpu, chain[1].sent);
  CHECK(fresh.ok());
  CHECK(refused(gpu, other.value(), chain[1].sent, fresh.value(), C::NotFound,
                "no reference frame is held") == 0);
  CHECK(first.value()
            .decode(chain[1].sent.data(), chain[1].sent.size(), fresh.value())
            .ok());
  CHECK(holds(gpu, fresh.value(), chain[1].intra));
  return 0;
}

// InvalidArgument from checks outside the device decode -- a new block on a
// reference block (after it), and a scale not the reference's (before it) --
// and rows 11 and 12 judged before row 13 on a frame that fails both,
// leaving the grid and the reference as they were.
int invalid_case(Gpu& gpu, const Spheres& s) {
  const std::vector<Step>& chain = s.chain;
  // The second P-frame with one more new block: the coordinate of a block
  // its reference has and it removes.
  const d::FrameReference id = reference_of(chain[1].sent);
  oracle::Frame p = inter_frame(id, chain[1].content, chain[2].content);
  const std::size_t m = p.modes.size();
  std::size_t gone = m;
  for (std::size_t e = 0; e < m && gone == m; ++e) {
    if (p.modes[e] == d::BlockMode::kRemove) gone = e;
  }
  CHECK(gone < m);
  const vr::Vec3i c = chain[1].content.coords[gone];
  const auto at =
      std::upper_bound(p.coords.begin(), p.coords.end(), c, d::coord_less) -
      p.coords.begin();
  p.coords.insert(p.coords.begin() + at, c);
  oracle::DctBlocks blocks;
  blocks.params = p.blocks.params;
  blocks.trunc_dist = p.blocks.trunc_dist;
  for (std::size_t e = 0; e < m + std::size_t(at); ++e) {
    append_payload(blocks, payload_at(p.blocks, e));
  }
  append_payload(blocks, payload_at(chain[1].content.blocks, gone));
  for (std::size_t e = m + std::size_t(at); e + 1 < m + p.coords.size(); ++e) {
    append_payload(blocks, payload_at(p.blocks, e));
  }
  p.blocks = std::move(blocks);
  const Bytes on_reference = oracle::write_frame(p, sizes(1, 2)).value();
  // That frame broken before row 13 too, where the merge still meets the
  // new block on the reference block: one block more than survive and are
  // new (row 12), and its first update segment corrupt (row 11).
  Bytes miscounted = on_reference;
  codec_frames::put_u32(miscounted, 32,
                        codec_frames::get_u32(on_reference, 32) + 1);
  std::vector<codec_frames::Section> t =
      codec_frames::sections_of(on_reference);
  t[3].body[0] = 0;
  t[3].body[1] = 0;
  const Bytes corrupt = codec_frames::assemble(on_reference, t);

  Bytes rescaled = chain[2].sent;
  codec_frames::put_f32(rescaled, 24, 0.3f);

  vkc::Result<codec::Decoder> made =
      codec::Decoder::create(gpu.device, gpu.allocator);
  vkc::Result<vol::VoxelBlockGrid> g = grid_for(gpu, chain[0].sent);
  CHECK(made.ok() && g.ok());
  codec::Decoder& dec = made.value();
  for (int i = 0; i < 2; ++i) {
    const Bytes& f = chain[std::size_t(i)].sent;
    CHECK(dec.decode(f.data(), f.size(), g.value()).ok());
  }
  CHECK(refused(gpu, dec, on_reference, g.value(), C::InvalidArgument,
                "is on reference block") == 0);
  CHECK(refused(gpu, dec, miscounted, g.value(), C::InvalidArgument,
                "are new, but the frame holds") == 0);
  CHECK(refused(gpu, dec, corrupt, g.value(), C::InvalidArgument,
                "segment 0 is corrupt") == 0);
  CHECK(refused(gpu, dec, rescaled, g.value(), C::InvalidArgument,
                "quantization scale is not its reference's") == 0);
  CHECK(dec.decode(chain[2].sent.data(), chain[2].sent.size(), g.value()).ok());
  CHECK(holds(gpu, g.value(), chain[2].intra));
  return 0;
}

// OutOfMemory for a grid too small for a P-frame -- its heap, before the grid
// changes, and its hash table, after -- then the documented resize, and the
// same P-frame decodes against the same reference. A P-frame on the frame
// that failed is NotFound, since a failed decode holds nothing new.
int out_of_memory_case(Gpu& gpu, const Spheres& s) {
  const std::vector<Step>& chain = s.chain;
  // A few blocks, which short chains in a table of as many slots cannot all
  // place, and a resize can.
  std::vector<Step> few;
  few.push_back(i_step(oracle::write_frame(make_frame(12, 64, 41)).value()));
  few.push_back(
      p_step(few[0], successor(few[0].content, 42, 100, 400, 6), sizes(4, 4)));
  few.push_back(
      p_step(few[1], successor(few[1].content, 43, 100, 400, 2), sizes(4, 4)));
  vkc::Result<codec::Decoder> made =
      codec::Decoder::create(gpu.device, gpu.allocator);
  vkc::Result<vol::VoxelBlockGrid> big = grid_for(gpu, chain[0].sent);
  CHECK(made.ok() && big.ok());
  codec::Decoder& dec = made.value();
  CHECK(
      dec.decode(chain[0].sent.data(), chain[0].sent.size(), big.value()).ok());

  // Too few slots: refused at the parse, the grid untouched.
  const Bytes& p = chain[1].sent;
  const std::uint32_t n =
      codec::read_frame_info(p.data(), p.size()).value().block_count;
  GridShape tiny;
  tiny.num_buckets = 4;
  vkc::Result<vol::VoxelBlockGrid> small = make_grid(gpu, tiny);
  CHECK(small.ok() && n > 32);
  CHECK(refused(gpu, dec, p, small.value(), C::OutOfMemory,
                "the caller can hold") == 0);
  CHECK(small.value().resize(std::int32_t(n)).ok());
  CHECK(dec.decode(p.data(), p.size(), small.value()).ok());
  CHECK(holds(gpu, small.value(), chain[1].intra));

  // A table that cannot place them: refused once the grid has changed.
  CHECK(dec.decode(few[0].sent.data(), few[0].sent.size(), big.value()).ok());
  const Bytes& q = few[1].sent;
  const std::uint32_t m =
      codec::read_frame_info(q.data(), q.size()).value().block_count;
  GridShape cramped;
  cramped.bucket_size = 2;
  cramped.num_buckets = std::int32_t((m + 1) / 2);
  cramped.max_chain = 1;
  vkc::Result<vol::VoxelBlockGrid> tight = make_grid(gpu, cramped);
  CHECK(tight.ok());
  const vkc::Status st = dec.decode(q.data(), q.size(), tight.value());
  CHECK(!st.ok() && st.domain() == C::OutOfMemory);
  // The failed frame is not held: a P-frame on it is NotFound.
  CHECK(refused(gpu, dec, few[2].sent, big.value(), C::NotFound,
                "not the one held") == 0);
  CHECK(tight.value().resize(cramped.num_buckets * 8).ok());
  CHECK(dec.decode(q.data(), q.size(), tight.value()).ok());
  CHECK(holds(gpu, tight.value(), few[1].intra));
  CHECK(dec.decode(few[2].sent.data(), few[2].sent.size(), tight.value()).ok());
  CHECK(holds(gpu, tight.value(), few[2].intra));
  return 0;
}

// The device buffers grow with a frame and are kept for smaller ones: a
// large frame, a small one and the large one again, each into a grid of its
// own and each as the oracle decodes it.
int scratch_case(Gpu& gpu, const Spheres& s) {
  vkc::Result<codec::Decoder> dec =
      codec::Decoder::create(gpu.device, gpu.allocator);
  CHECK(dec.ok());
  const Bytes& large = s.chain[0].intra;
  const Bytes small = oracle::write_frame(make_frame(12, 64, 41)).value();
  for (const Bytes* f : {&large, &small, &large}) {
    vkc::Result<vol::VoxelBlockGrid> g = grid_for(gpu, *f);
    CHECK(g.ok());
    CHECK(dec.value().decode(f->data(), f->size(), g.value()).ok());
    CHECK(holds(gpu, g.value(), *f));
  }
  return 0;
}

// A decoder moved mid-chain, by construction and by assignment, takes its
// reference with it.
int moves_case(Gpu& gpu, const Spheres& s) {
  const std::vector<Step>& chain = s.chain;
  vkc::Result<codec::Decoder> made =
      codec::Decoder::create(gpu.device, gpu.allocator);
  vkc::Result<vol::VoxelBlockGrid> g = grid_for(gpu, chain[0].sent);
  CHECK(made.ok() && g.ok());
  codec::Decoder a = std::move(made).value();
  CHECK(a.decode(chain[0].sent.data(), chain[0].sent.size(), g.value()).ok());
  codec::Decoder b(std::move(a));
  CHECK(!a.valid());  // NOLINT(bugprone-use-after-move): asserting the source
  CHECK(b.decode(chain[1].sent.data(), chain[1].sent.size(), g.value()).ok());
  CHECK(holds(gpu, g.value(), chain[1].intra));
  vkc::Result<codec::Decoder> other =
      codec::Decoder::create(gpu.device, gpu.allocator);
  CHECK(other.ok());
  codec::Decoder c = std::move(other).value();
  c = std::move(b);  // over a live decoder
  CHECK(c.decode(chain[2].sent.data(), chain[2].sent.size(), g.value()).ok());
  CHECK(holds(gpu, g.value(), chain[2].intra));
  return 0;
}

// A frame that names a segment longer than kMaxDeviceDecodeSegmentSize -- its
// segment size, or a P-frame's update size -- is malformed: refused before
// any device work, with the grid and the reference as they were. Segments of
// exactly the limit decode.
int segment_rule_case(Gpu& gpu) {
  constexpr std::uint32_t kLimit = codec::kMaxDeviceDecodeSegmentSize;
  const oracle::Frame first = make_frame(kLimit, 8, 71);
  std::vector<Step> chain;
  chain.push_back(i_step(oracle::write_frame(first, sizes(kLimit, 1)).value()));
  const oracle::Frame next = successor(first, 72, 50, 300, 4);
  chain.push_back(p_step(chain[0], next, sizes(64, kLimit)));
  // One entry more named in the header, and in REFERENCE: the same segments.
  Bytes long_intra = chain[0].sent;
  codec_frames::put_u32(long_intra, 36, kLimit + 1);
  std::vector<codec_frames::Section> s =
      codec_frames::sections_of(chain[1].sent);
  codec_frames::put_u32(s[0].body, 12, kLimit + 1);
  const Bytes long_update = codec_frames::assemble(chain[1].sent, s);

  vkc::Result<codec::Decoder> made =
      codec::Decoder::create(gpu.device, gpu.allocator);
  vkc::Result<vol::VoxelBlockGrid> g = grid_for(gpu, chain[0].sent);
  CHECK(made.ok() && g.ok());
  codec::Decoder& dec = made.value();
  const Bytes& i = chain[0].sent;
  CHECK(dec.decode(i.data(), i.size(), g.value()).ok());
  CHECK(holds(gpu, g.value(), chain[0].intra));
  const std::string past = std::to_string(kLimit + 1) + " is not in";
  CHECK(refused(gpu, dec, long_intra, g.value(), C::InvalidArgument,
                ("segment size " + past).c_str()) == 0);
  CHECK(refused(gpu, dec, long_update, g.value(), C::InvalidArgument,
                ("update size " + past).c_str()) == 0);
  // Both refusals kept the reference: the P-frame on it decodes.
  const Bytes& p = chain[1].sent;
  CHECK(dec.decode(p.data(), p.size(), g.value()).ok());
  CHECK(holds(gpu, g.value(), chain[1].intra));
  return 0;
}

}  // namespace

int gpu_main(vr_test::GpuContext& gpu) {
  if (apply_case(gpu, 0) != 0) return 1;
  if (apply_case(gpu, 4) != 0) return 1;
  if (grow_case(gpu) != 0) return 1;
  if (apply_refusals_case(gpu) != 0) return 1;
  if (synthetic_case(gpu) != 0) return 1;
  if (extremes_chain_case(gpu) != 0) return 1;
  if (encoder_case(gpu) != 0) return 1;
  if (segment_rule_case(gpu) != 0) return 1;
  vkc::Result<Spheres> s = spheres(gpu);
  CHECK(s.ok());
  if (sphere_case(gpu, s.value()) != 0) return 1;
  if (not_found_case(gpu, s.value()) != 0) return 1;
  if (invalid_case(gpu, s.value()) != 0) return 1;
  if (out_of_memory_case(gpu, s.value()) != 0) return 1;
  if (scratch_case(gpu, s.value()) != 0) return 1;
  if (moves_case(gpu, s.value()) != 0) return 1;
  std::printf("codec P-frame decode: OK\n");
  return 0;
}

int main() { return vr_test::run_on_gpu(gpu_main); }
