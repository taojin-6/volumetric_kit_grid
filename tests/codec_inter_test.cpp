// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// GPU test for P-frames on the device. Decoding: the reference store's apply
// against independent uint16 arithmetic, at every source kind and K, in one
// dispatch or several, as its slots grow, and its sources past what is bound
// counted and refused; and the Decoder's frames -- the Encoder's, and chains
// of an I-frame, then P-frames, new blocks at int32's extremes among them --
// leaving the grid exactly as the CPU oracle's decode of the same content
// does, bit for bit, with every mode, new blocks and the empty cases; a
// P-frame whose reference is not held refused as NotFound, a segment longer
// than the format allows refused, every refusal and failure leaving the
// reference as it was, and the device buffers grown and shrunk frame to
// frame. Encoding: every frame of the Encoder's chains the CPU oracle's
// bytes, its P-frames priced as the oracle prices them, counted in the
// diff's batch, and coded in update segments of the segment size, or of an
// update size set; each of the diff's choices from one edit of a frame; a
// difference of 32768 coded zero, in either half of a coefficient word; and
// the diff's refusals and bounded entries, counted in its batch. Internal
// headers by include path; skips with no device.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
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
#include "codec_oracle_encode.hpp"
#include "dct_transform.hpp"
#include "device_frame_writer.hpp"
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
  id.header.voxel_size = kVoxel;
  id.header.trunc_dist = r.trunc_dist;
  id.header.params = r.params;
  id.header.block_count = static_cast<std::uint32_t>(m);
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

// The device, with what the cases share on it, each made once: the oracle's
// transform, and a Decoder for the cases whose chains start with an I-frame.
struct Shared : Gpu {
  d::DctTransform& transform;
  codec::Decoder& dec;
};

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
vkc::Result<Snapshot> oracle_decode(Shared& gpu, const Bytes& frame) {
  VKC_ASSIGN(const oracle::Frame content,
             oracle::read_frame(frame.data(), frame.size(), kMaxBlocks));
  VKC_ASSIGN(vol::VoxelBlockGrid grid, grid_for(gpu, frame));
  VKC_TRY(allocate(gpu, grid, content.coords));
  // The grid's blocks in frame order, entry for entry with the content's.
  VKC_ASSIGN(const std::vector<vol::BlockIndex> blocks, active_sorted(grid));
  VKC_TRY(oracle::inverse(gpu, gpu.transform, grid, blocks, content.blocks));
  return snapshot(gpu, grid);
}

// Whether @p grid holds what @p frame, an intra frame, decodes to.
bool holds(Shared& gpu, vol::VoxelBlockGrid& grid, const Bytes& frame) {
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
int run_chain(Shared& gpu, const std::vector<Step>& chain, const char* what) {
  vkc::Result<vol::VoxelBlockGrid> grid = grid_for(gpu, chain[0].sent);
  CHECK(grid.ok());
  for (std::size_t i = 0; i < chain.size(); ++i) {
    const Bytes& f = chain[i].sent;
    const vkc::Status s = gpu.dec.decode(f.data(), f.size(), grid.value());
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
int synthetic_case(Shared& gpu) {
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
int extremes_chain_case(Shared& gpu) {
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
int encoder_case(Shared& gpu) {
  vkc::Result<vol::VoxelBlockGrid> g =
      sphere_grid(gpu, Sphere{vr::Vec3f(0.01f, -0.02f, 0.03f), 0.09f});
  CHECK(g.ok());
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
        gpu.dec
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

vkc::Result<Spheres> spheres(Gpu& gpu) {
  Spheres out;
  codec::EncoderConfig config;
  config.segment_size = 3;
  VKC_ASSIGN(codec::Encoder enc,
             codec::Encoder::create(gpu.device, gpu.allocator, config));
  std::vector<Bytes> intra;
  for (int t = 0; t < 3; ++t) {
    const Sphere s{vr::Vec3f(0.006f * float(t), 0.0f, 0.002f * float(t)),
                   0.1f + 0.008f * float(t)};
    VKC_ASSIGN(vol::VoxelBlockGrid g, sphere_grid(gpu, s));
    VKC_ASSIGN(Bytes frame, enc.encode(g));
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
int sphere_case(Shared& gpu, const Spheres& s) {
  const std::vector<Step>& chain = s.chain;
  const Census c = census(chain[0], chain[1]);
  CHECK(c.modes[std::size_t(d::BlockMode::kKeep)] > 0);
  CHECK(c.modes[std::size_t(d::BlockMode::kRemove)] > 0);
  CHECK(c.added > 0);
  CHECK(run_chain(gpu, chain, "spheres") == 0);

  codec::Decoder& dec = gpu.dec;
  vkc::Result<vol::VoxelBlockGrid> g = grid_for(gpu, chain[0].sent);
  CHECK(g.ok());
  for (int i = 0; i < 2; ++i) {
    const Bytes& f = chain[std::size_t(i)].sent;
    CHECK(dec.decode(f.data(), f.size(), g.value()).ok());
  }
  const Bytes& p = chain[2].sent;
  vkc::Result<codec::FrameInfo> info =
      codec::read_frame_info(p.data(), p.size());
  CHECK(info.ok() && info.value().type == codec::FrameType::kInter);
  CHECK(info.value().block_count == chain[2].content.coords.size());
  vkc::Result<vol::VoxelBlockGrid> fresh = grid_for(gpu, p);
  CHECK(fresh.ok());
  CHECK(dec.decode(p.data(), p.size(), fresh.value()).ok());
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
int not_found_case(Shared& gpu, const Spheres& s) {
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
  codec::Decoder& first = gpu.dec;
  vkc::Result<vol::VoxelBlockGrid> og = grid_for(gpu, chain[0].sent);
  CHECK(other.ok() && og.ok());
  CHECK(first.decode(chain[0].sent.data(), chain[0].sent.size(), og.value())
            .ok());
  vkc::Result<vol::VoxelBlockGrid> fresh = grid_for(gpu, chain[1].sent);
  CHECK(fresh.ok());
  CHECK(refused(gpu, other.value(), chain[1].sent, fresh.value(), C::NotFound,
                "no reference frame is held") == 0);
  CHECK(first.decode(chain[1].sent.data(), chain[1].sent.size(), fresh.value())
            .ok());
  CHECK(holds(gpu, fresh.value(), chain[1].intra));
  return 0;
}

// InvalidArgument from checks outside the device decode -- a new block on a
// reference block (after it), and a scale not the reference's (before it) --
// and rows 11 and 12 judged before row 13 on a frame that fails both,
// leaving the grid and the reference as they were.
int invalid_case(Shared& gpu, const Spheres& s) {
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

  vkc::Result<vol::VoxelBlockGrid> g = grid_for(gpu, chain[0].sent);
  CHECK(g.ok());
  codec::Decoder& dec = gpu.dec;
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
int out_of_memory_case(Shared& gpu, const Spheres& s) {
  const std::vector<Step>& chain = s.chain;
  // A few blocks, which short chains in a table of as many slots cannot all
  // place, and a resize can.
  std::vector<Step> few;
  few.push_back(i_step(oracle::write_frame(make_frame(12, 64, 41)).value()));
  few.push_back(
      p_step(few[0], successor(few[0].content, 42, 100, 400, 6), sizes(4, 4)));
  few.push_back(
      p_step(few[1], successor(few[1].content, 43, 100, 400, 2), sizes(4, 4)));
  vkc::Result<vol::VoxelBlockGrid> big = grid_for(gpu, chain[0].sent);
  CHECK(big.ok());
  codec::Decoder& dec = gpu.dec;
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
int scratch_case(Shared& gpu, const Spheres& s) {
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
int moves_case(Shared& gpu, const Spheres& s) {
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
int segment_rule_case(Shared& gpu) {
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

  vkc::Result<vol::VoxelBlockGrid> g = grid_for(gpu, chain[0].sent);
  CHECK(g.ok());
  codec::Decoder& dec = gpu.dec;
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

// --- The Encoder's P-frames. -------------------------------------------------

// One frame of an Encoder's chain: what it sent, the oracle's intra frame of
// the same grid, and the content both code, read off that intra frame.
struct Coded {
  Bytes sent;
  Bytes intra;
  oracle::Frame content;
};

// Encode @p grid with @p enc and hold the bytes to the oracle's: an I-frame
// to the oracle's intra frame of the grid, a P-frame to the oracle's writer
// over encoder_frame against the chain's last frame, at @p config's sizes.
// Appends the frame to @p chain.
int encode_checked(Shared& gpu, codec::Encoder& enc,
                   const codec::EncoderConfig& config,
                   vol::VoxelBlockGrid& grid, std::vector<Coded>& chain) {
  vkc::Result<Bytes> sent = enc.encode(grid);
  vkc::Result<Bytes> intra = oracle::encode(gpu, gpu.transform, grid, config);
  CHECK(sent.ok() && intra.ok());
  vkc::Result<oracle::Frame> content = oracle::read_frame(
      intra.value().data(), intra.value().size(), kMaxBlocks);
  vkc::Result<codec::FrameInfo> info =
      codec::read_frame_info(sent.value().data(), sent.value().size());
  CHECK(content.ok() && info.ok());
  if (info.value().type == codec::FrameType::kIntra) {
    CHECK(sent.value() == intra.value());
  } else {
    CHECK(!chain.empty());
    const Coded& prev = chain.back();
    oracle::FrameWriteOptions options;
    options.segment_size = config.segment_size;
    options.update_size = config.resolved_update_size();
    vkc::Result<Bytes> want = oracle::write_frame(
        codec_frames::encoder_frame(reference_of(prev.sent), prev.content,
                                    content.value()),
        options);
    CHECK(want.ok());
    if (sent.value() != want.value()) {
      std::fprintf(stderr, "frame %zu: the P-frame is not the oracle's\n",
                   chain.size());
      return 1;
    }
  }
  chain.push_back(Coded{std::move(sent).value(), std::move(intra).value(),
                        std::move(content).value()});
  return 0;
}

// The modes a P-frame of @p chain's last two holds, and its new blocks.
Census census_of(const std::vector<Coded>& chain) {
  const Coded& prev = chain[chain.size() - 2];
  const Coded& p = chain.back();
  const d::FrameReference id = reference_of(prev.sent);
  const oracle::Frame f =
      oracle::read_frame(p.sent.data(), p.sent.size(), kMaxBlocks, &id).value();
  Census c;
  for (d::BlockMode m : f.modes) ++c.modes[std::size_t(m)];
  c.added = f.coords.size();
  return c;
}

codec::FrameType type_of(const Bytes& frame) {
  return codec::read_frame_info(frame.data(), frame.size()).value().type;
}

// A sphere that moves and grows beside one that stays, at step @p t.
vkc::Result<vol::VoxelBlockGrid> moving_spheres(Gpu& gpu, int t) {
  const Sphere moving{
      vr::Vec3f(0.01f * float(t), 0.003f * float(t), -0.004f * float(t)),
      0.09f + 0.006f * float(t)};
  return two_sphere_grid(gpu, moving,
                         Sphere{vr::Vec3f(0.6f, 0.0f, 0.0f), 0.05f});
}

// The moving spheres in chains at period 4 and open-ended, at an odd K in
// segments of 3 and update segments of 5 and at the defaults: every frame
// the oracle's bytes, every P-frame decoding to its I-frame's grid; and
// among the P-frames every mode the encoder chooses but motion, and new
// blocks.
int encoder_chain_case(Shared& gpu) {
  for (const std::uint32_t period : {4u, ~0u}) {
    for (const bool odd : {true, false}) {
      codec::EncoderConfig config;
      config.intra_period = period;
      if (odd) {
        config.params.coefficient_count = 37;
        config.segment_size = 3;
        config.update_size = 5;
      }
      vkc::Result<codec::Encoder> enc =
          codec::Encoder::create(gpu.device, gpu.allocator, config);
      CHECK(enc.ok());
      std::vector<Coded> chain;
      Census all;
      vkc::Result<vol::VoxelBlockGrid> player = make_grid(gpu);
      CHECK(player.ok());
      for (int t = 0; t < 7; ++t) {
        vkc::Result<vol::VoxelBlockGrid> g = moving_spheres(gpu, t);
        CHECK(g.ok());
        CHECK(encode_checked(gpu, enc.value(), config, g.value(), chain) == 0);
        const bool intra = t == 0 || (period == 4 && t == 4);
        CHECK(type_of(chain.back().sent) ==
              (intra ? codec::FrameType::kIntra : codec::FrameType::kInter));
        const Bytes& f = chain.back().sent;
        CHECK(gpu.dec.decode(f.data(), f.size(), player.value()).ok());
        CHECK(holds(gpu, player.value(), chain.back().intra));
        if (!intra) {
          const Census c = census_of(chain);
          for (std::size_t m = 0; m < 5; ++m) all.modes[m] += c.modes[m];
          all.added += c.added;
        }
      }
      CHECK(all.modes[std::size_t(d::BlockMode::kKeep)] > 0);
      CHECK(all.modes[std::size_t(d::BlockMode::kSkip)] > 0);
      CHECK(all.modes[std::size_t(d::BlockMode::kRemove)] > 0);
      CHECK(all.modes[std::size_t(d::BlockMode::kMotion)] == 0);
      CHECK(all.added > 0);
    }
  }
  return 0;
}

// The update size: left at 0, a P-frame's update segments take the segment
// size, whatever it is; set, they take it. The moving spheres' first two
// frames, an I-frame and a P-frame: each the oracle's bytes at the config's
// sizes, the P-frame's REFERENCE carrying the U expected, and the chain
// decoding to the second grid's I-frame.
int default_update_size_case(Shared& gpu) {
  const std::uint32_t kLongest = codec::kMaxDeviceDecodeSegmentSize;
  struct Sizes {
    std::uint32_t segment_size;
    std::uint32_t update_size;
    std::uint32_t coded;  // the U the P-frame must carry
  };
  for (const Sizes& s : {Sizes{1, 0, 1}, Sizes{16, 0, 16}, Sizes{64, 0, 64},
                         Sizes{kLongest, 0, kLongest}, Sizes{16, 5, 5},
                         Sizes{16, kLongest, kLongest}}) {
    codec::EncoderConfig config;
    config.intra_period = 2;
    config.segment_size = s.segment_size;
    config.update_size = s.update_size;
    vkc::Result<codec::Encoder> enc =
        codec::Encoder::create(gpu.device, gpu.allocator, config);
    vkc::Result<vol::VoxelBlockGrid> player = make_grid(gpu);
    CHECK(enc.ok() && player.ok());
    std::vector<Coded> chain;
    for (int t = 0; t < 2; ++t) {
      vkc::Result<vol::VoxelBlockGrid> g = moving_spheres(gpu, t);
      CHECK(g.ok());
      CHECK(encode_checked(gpu, enc.value(), config, g.value(), chain) == 0);
      const Bytes& f = chain.back().sent;
      CHECK(gpu.dec.decode(f.data(), f.size(), player.value()).ok());
    }
    CHECK(type_of(chain[0].sent) == codec::FrameType::kIntra);
    CHECK(type_of(chain[1].sent) == codec::FrameType::kInter);
    const d::FrameReference id = reference_of(chain[0].sent);
    const Bytes& p = chain[1].sent;
    vkc::Result<d::ParsedFrame> parsed =
        d::parse_frame(p.data(), p.size(), kMaxBlocks, &id);
    CHECK(parsed.ok() && parsed.value().update_size == s.coded);
    CHECK(holds(gpu, player.value(), chain[1].intra));
  }
  return 0;
}

// A block's voxels: SDF as a fraction of the band, and whether each is
// observed.
struct Voxels {
  std::vector<float> sdf = std::vector<float>(kVpb, 0.0f);
  std::vector<bool> observed = std::vector<bool>(kVpb, true);
};

using Content = std::map<vr::Vec3i, Voxels, codec_frames::ZyxLess>;

Voxels constant(float sdf) {
  Voxels v;
  v.sdf.assign(kVpb, sdf);
  return v;
}

// Values drawn from all of [-1, 1]: every coefficient large.
Voxels noise(Lcg& rng) {
  Voxels v;
  for (float& f : v.sdf) f = float(rng.below(2001)) / 1000.0f - 1.0f;
  return v;
}

// A plane through the block: a fused surface's smooth field.
Voxels slope(int tilt) {
  Voxels v;
  for (std::uint32_t i = 0; i < kVpb; ++i) {
    const int x = int(i % 8);
    const int y = int(i / 8 % 8);
    const int z = int(i / 64);
    v.sdf[i] = float(x + tilt * y + z - 7) / 16.0f;
  }
  return v;
}

// A grid holding exactly @p content, of the fixture's geometry.
vkc::Result<vol::VoxelBlockGrid> content_grid(Gpu& gpu,
                                              const Content& content) {
  VKC_ASSIGN(vol::VoxelBlockGrid grid, make_grid(gpu));
  std::vector<vr::Vec3i> coords;
  for (const auto& [c, voxels] : content) coords.push_back(c);
  VKC_TRY(allocate(gpu, grid, coords));
  VKC_ASSIGN(const std::vector<vol::BlockIndex> blocks, active_sorted(grid));
  VKC_ASSIGN(vr_test::AttributeSpan<float> tsdf,
             vr_test::read_span<float>(gpu.device, gpu.allocator, grid, "tsdf",
                                       blocks));
  VKC_ASSIGN(vr_test::AttributeSpan<float> weight,
             vr_test::read_span<float>(gpu.device, gpu.allocator, grid,
                                       "weight", blocks));
  for (const vol::BlockIndex& b : blocks) {
    const Voxels& v = content.at(b.coord);
    const std::size_t at = std::size_t(b.ptr) - tsdf.first;
    for (std::uint32_t i = 0; i < kVpb; ++i) {
      tsdf.values[at + i] = v.observed[i] ? v.sdf[i] * kTrunc : 0.0f;
      weight.values[at + i] = v.observed[i] ? 2.0f : 0.0f;
    }
  }
  VKC_TRY(vr_test::write_span(gpu.device, gpu.allocator, grid, "tsdf", tsdf));
  VKC_TRY(
      vr_test::write_span(gpu.device, gpu.allocator, grid, "weight", weight));
  return grid;
}

// What the diff chooses, each from one edit of a reference frame: an
// unchanged grid all skip; one block changed, one entry that is not; a mask
// changed alone, keep with residuals of 0; a block gone and one added; new
// content against unrelated content zero, a small change keep; and an empty
// grid an I-frame, smaller than the P-frame removing every entry, and the
// frame after it an I-frame too. Every frame is the oracle's, and the prices
// agree.
int diff_case(Shared& gpu) {
  codec::EncoderConfig config;
  config.intra_period = 1000;
  config.segment_size = 2;
  vkc::Result<codec::Encoder> made =
      codec::Encoder::create(gpu.device, gpu.allocator, config);
  CHECK(made.ok());
  codec::Encoder& enc = made.value();
  Lcg rng{91};
  Content base;
  base[vr::Vec3i(0, 0, 0)] = constant(0.3f);
  base[vr::Vec3i(1, 0, 0)] = slope(1);
  base[vr::Vec3i(3, 0, 0)] = slope(2);
  base[vr::Vec3i(0, 2, 0)] = noise(rng);
  base[vr::Vec3i(-4, 1, 7)] = noise(rng);
  base[vr::Vec3i(5, -3, 2)] = constant(-0.6f);
  std::vector<Coded> chain;
  // Encode @p content, a P-frame on the chain's last frame: its modes, in
  // reference order, and its new blocks.
  auto next = [&](const Content& content, oracle::Frame& p) {
    vkc::Result<vol::VoxelBlockGrid> g = content_grid(gpu, content);
    if (!g.ok() || encode_checked(gpu, enc, config, g.value(), chain) != 0 ||
        type_of(chain.back().sent) != codec::FrameType::kInter) {
      return false;
    }
    const d::FrameReference id = reference_of(chain[chain.size() - 2].sent);
    const Bytes& f = chain.back().sent;
    vkc::Result<oracle::Frame> read =
        oracle::read_frame(f.data(), f.size(), kMaxBlocks, &id);
    if (!read.ok()) return false;
    p = std::move(read).value();
    return true;
  };
  auto count = [](const oracle::Frame& p, d::BlockMode mode) {
    return std::size_t(std::count(p.modes.begin(), p.modes.end(), mode));
  };
  vkc::Result<vol::VoxelBlockGrid> first = content_grid(gpu, base);
  CHECK(first.ok());
  CHECK(encode_checked(gpu, enc, config, first.value(), chain) == 0);
  CHECK(type_of(chain.back().sent) == codec::FrameType::kIntra);
  const std::size_t m = base.size();

  oracle::Frame p;
  CHECK(next(base, p));
  CHECK(count(p, d::BlockMode::kSkip) == m && p.coords.empty());

  // A plane of one block lowered: several coefficients change.
  Content changed = base;
  for (std::uint32_t i = 0; i < 64; ++i) {
    changed.at(vr::Vec3i(1, 0, 0)).sdf[i] -= 0.3f;
  }
  CHECK(next(changed, p));
  CHECK(count(p, d::BlockMode::kSkip) == m - 1 && p.coords.empty());

  // One voxel of the constant block unobserved: the forward fills it with
  // the same value, so the coefficients stay and the mask alone changes.
  Content masked = changed;
  masked.at(vr::Vec3i(0, 0, 0)).observed[77] = false;
  CHECK(next(masked, p));
  CHECK(p.modes[0] == d::BlockMode::kKeep);
  CHECK(count(p, d::BlockMode::kSkip) == m - 1);
  const Payload x = payload_at(p.blocks, 0);
  CHECK(std::all_of(x.q.begin(), x.q.end(),
                    [](std::int16_t v) { return v == 0; }));
  std::uint32_t bits = 0;
  for (std::uint32_t w : x.mask) bits += std::uint32_t(__builtin_popcount(w));
  CHECK(bits == 1 && (x.mask[77 / 32] >> (77 % 32) & 1u) == 1u);

  Content moved = masked;
  moved.erase(vr::Vec3i(3, 0, 0));
  moved[vr::Vec3i(2, 0, 0)] = slope(3);
  CHECK(next(moved, p));
  CHECK(count(p, d::BlockMode::kRemove) == 1 &&
        count(p, d::BlockMode::kSkip) == m - 1);
  CHECK(p.coords == std::vector<vr::Vec3i>{vr::Vec3i(2, 0, 0)});

  // A smooth block where noise was is zero, its residuals costing more than
  // its values; noise with one plane lowered is keep, the other way round.
  Content unrelated = moved;
  unrelated.at(vr::Vec3i(0, 2, 0)) = slope(1);
  for (std::uint32_t i = 448; i < kVpb; ++i) {
    float& v = unrelated.at(vr::Vec3i(-4, 1, 7)).sdf[i];
    v = std::max(v - 0.2f, -1.0f);
  }
  CHECK(next(unrelated, p));
  const oracle::Frame& ref = chain[chain.size() - 2].content;
  std::map<vr::Vec3i, d::BlockMode, codec_frames::ZyxLess> mode_of;
  for (std::size_t e = 0; e < ref.coords.size(); ++e) {
    mode_of[ref.coords[e]] = p.modes[e];
  }
  CHECK(mode_of.at(vr::Vec3i(0, 2, 0)) == d::BlockMode::kZero);
  CHECK(mode_of.at(vr::Vec3i(-4, 1, 7)) == d::BlockMode::kKeep);

  // Every block gone: an I-frame of no blocks, where a P-frame would remove
  // every entry; then, on that empty reference, an I-frame again.
  vkc::Result<vol::VoxelBlockGrid> none = content_grid(gpu, Content{});
  CHECK(none.ok());
  CHECK(encode_checked(gpu, enc, config, none.value(), chain) == 0);
  const Coded& empty = chain.back();
  CHECK(type_of(empty.sent) == codec::FrameType::kIntra);
  CHECK(codec::read_frame_info(empty.sent.data(), empty.sent.size())
            .value()
            .block_count == 0);
  const Coded& last = chain[chain.size() - 2];
  oracle::FrameWriteOptions options;
  options.segment_size = config.segment_size;
  options.update_size = config.resolved_update_size();
  const vkc::Result<Bytes> removing = oracle::write_frame(
      codec_frames::encoder_frame(reference_of(last.sent), last.content,
                                  empty.content),
      options);
  CHECK(removing.ok() && empty.sent.size() < removing.value().size());
  vkc::Result<vol::VoxelBlockGrid> again = content_grid(gpu, base);
  CHECK(again.ok());
  CHECK(encode_checked(gpu, enc, config, again.value(), chain) == 0);
  CHECK(type_of(chain.back().sent) == codec::FrameType::kIntra);
  return 0;
}

// A difference of 32768 in coefficient @p j -- +16384 against -16384, at a
// scale that puts it exactly there, which the I-frames confirm -- is the one
// residual no class carries, so the block is zero, where the price would
// keep it: what the two blocks share makes the target's values cost more
// than the residuals. At j = 0, a word's low half, the DC of a mean SDF of
// +-0.9 beside a shared ramp along x; at j = 3, the (1, 0, 0) basis and a
// word's high half, its cosine at +-0.75 on a shared level. The P-frame
// decodes to the target's grid.
int wrap_case(Shared& gpu, std::uint32_t j) {
  codec::EncoderConfig config;
  config.intra_period = 2;
  config.params.coefficient_count = 4;
  // The DC of a block of normalized SDF v is v sqrt(512); the (1, 0, 0)
  // coefficient of a cos(pi (2x + 1) / 16) is 16 a.
  const float amplitude = j == 0 ? 0.9f : 0.75f;
  config.params.quantization_scale =
      amplitude * (j == 0 ? 22.627417f : 16.0f) / 16384.0f;
  CHECK(config.params.quantization_scale > codec::kMinStep);
  vkc::Result<codec::Encoder> enc =
      codec::Encoder::create(gpu.device, gpu.allocator, config);
  CHECK(enc.ok());
  // The block of coefficient j at +-16384, the other part shared.
  auto block = [&](float sign) {
    Voxels v;
    for (std::uint32_t i = 0; i < kVpb; ++i) {
      const int x = int(i % 8);
      const double wave = std::cos(3.14159265358979 * (2 * x + 1) / 16.0);
      v.sdf[i] = j == 0 ? sign * amplitude + 0.1f * float(2 * x - 7) / 7.0f
                        : 0.2f + sign * amplitude * float(wave);
    }
    return v;
  };
  Content up;
  up[vr::Vec3i(0, 0, 0)] = block(1.0f);
  up[vr::Vec3i(1, 0, 0)] = slope(1);
  Content down = up;
  down.at(vr::Vec3i(0, 0, 0)) = block(-1.0f);
  std::vector<Coded> chain;
  vkc::Result<vol::VoxelBlockGrid> a = content_grid(gpu, up);
  vkc::Result<vol::VoxelBlockGrid> b = content_grid(gpu, down);
  CHECK(a.ok() && b.ok());
  CHECK(encode_checked(gpu, enc.value(), config, a.value(), chain) == 0);
  CHECK(encode_checked(gpu, enc.value(), config, b.value(), chain) == 0);
  const Payload r = payload_at(chain[0].content.blocks, 0);
  const Payload t = payload_at(chain[1].content.blocks, 0);
  CHECK(r.q[j] == 16384 && t.q[j] == -16384);
  // The premise: keep's residuals, the -32768 included, price below the
  // target's values.
  Payload x = t;
  for (std::size_t i = 0; i < x.q.size(); ++i) {
    x.q[i] = codec_frames::residual(t.q[i], r.q[i]);
  }
  for (std::size_t w = 0; w < x.mask.size(); ++w) x.mask[w] ^= r.mask[w];
  CHECK(x.q[j] == std::numeric_limits<std::int16_t>::min());
  CHECK(oracle::payload_price(x.mask.data(), x.q.data(), 4) <
        oracle::payload_price(t.mask.data(), t.q.data(), 4));
  CHECK(type_of(chain[1].sent) == codec::FrameType::kInter);
  const Census c = census_of(chain);
  CHECK(c.modes[std::size_t(d::BlockMode::kZero)] == 1);
  CHECK(c.modes[std::size_t(d::BlockMode::kSkip)] == 1);
  vkc::Result<vol::VoxelBlockGrid> player = make_grid(gpu);
  CHECK(player.ok());
  for (const Coded& f : chain) {
    CHECK(gpu.dec.decode(f.sent.data(), f.sent.size(), player.value()).ok());
  }
  CHECK(holds(gpu, player.value(), chain[1].intra));
  return 0;
}

// What the diff refuses before it records: no reference held, a K not the
// reference's, more blocks than reserved; and the entries it leaves, which
// the device frame writer counts in the diff's batch: a frame of no blocks,
// bounded at M, every entry removed, written as the oracle's P-frame.
int diff_refusals_case(Gpu& gpu) {
  vkc::Result<std::unique_ptr<d::ReferenceStore>> made =
      d::ReferenceStore::create(gpu.device, gpu.allocator);
  vkc::Result<std::unique_ptr<d::DeviceFrameWriter>> writer =
      d::DeviceFrameWriter::create(gpu.device, gpu.allocator);
  CHECK(made.ok() && writer.ok());
  d::ReferenceStore& store = *made.value();
  auto refused = [&](std::uint32_t n, std::uint32_t k, const char* says) {
    vkc::CommandBatch batch(gpu.device, gpu.allocator);
    const vkc::Status s = store.record_diff(batch, n, k).status();
    return !s.ok() && s.domain() == C::InvalidArgument &&
           s.message().find(says) != std::string::npos;
  };
  CHECK(refused(0, 8, "no reference is held"));
  Lcg rng{37};
  const oracle::DctBlocks held = random_entries(5, 8, rng);
  CHECK(commit_intra(gpu, store, held, 3) == 0);
  CHECK(store.reserve(4, 8).ok());
  CHECK(refused(4, 9, "a frame at K 9 on a reference at K 8"));
  CHECK(refused(1000, 8, "1000 blocks, more than were reserved"));
  vkc::CommandBatch batch(gpu.device, gpu.allocator);
  vkc::Result<d::ResidentBlocks> entries = store.record_diff(batch, 0, 8);
  CHECK(entries.ok() && entries.value().count == 5 &&
        entries.value().reference_count == 5 && entries.value().bounded);
  const oracle::FrameWriteOptions sizes;
  CHECK(writer.value()
            ->record_count(batch, entries.value(), sizes.segment_size,
                           sizes.update_size)
            .ok());
  CHECK(batch.submit().ok());
  vkc::Result<std::vector<std::uint32_t>> modes =
      vr_test::read_back<std::uint32_t>(gpu.device, gpu.allocator,
                                        *entries.value().modes, 5);
  CHECK(modes.ok() &&
        modes.value() == std::vector<std::uint32_t>(
                             5, std::uint32_t(d::BlockMode::kRemove)));
  oracle::Frame removed;
  removed.type = d::FrameType::kInter;
  removed.voxel_size = kVoxel;
  removed.reference = *store.held();
  removed.modes.assign(5, d::BlockMode::kRemove);
  removed.blocks.params = held.params;
  removed.blocks.trunc_dist = held.trunc_dist;
  removed.blocks.masks.assign(5 * codec::kMaskWordsPerBlock, 0);
  removed.blocks.coefficients.assign(5 * 8, 0);
  vkc::Result<Bytes> written = writer.value()->finish(
      entries.value(), kVoxel, held.trunc_dist, held.params, store.held());
  CHECK(written.ok() &&
        written.value() == oracle::write_frame(removed, sizes).value());
  CHECK(store.held()->hash == 3);
  return 0;
}

}  // namespace

int gpu_main(vr_test::GpuContext& context) {
  vkc::Result<std::unique_ptr<d::DctTransform>> transform =
      d::DctTransform::create(context.device, context.allocator);
  vkc::Result<codec::Decoder> dec =
      codec::Decoder::create(context.device, context.allocator);
  CHECK(transform.ok() && dec.ok());
  Shared gpu{
      {context.device, context.allocator}, *transform.value(), dec.value()};
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
  if (encoder_chain_case(gpu) != 0) return 1;
  if (default_update_size_case(gpu) != 0) return 1;
  if (diff_case(gpu) != 0) return 1;
  if (wrap_case(gpu, 0) != 0) return 1;
  if (wrap_case(gpu, 3) != 0) return 1;
  if (diff_refusals_case(gpu) != 0) return 1;
  std::printf("codec P-frames: OK\n");
  return 0;
}

int main() { return vr_test::run_on_gpu(gpu_main); }
