// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// GPU test for P-frame decoding: the reference store's apply against
// independent uint16 arithmetic, at every source kind and K, in one dispatch
// or several, as its slots grow; and the Decoder's chains -- an I-frame, then
// P-frames -- leaving the grid exactly as an intra decode of the same content
// does, bit for bit, on every entropy path, kAuto crossing between them
// mid-chain, with every mode, new blocks and the empty cases; a P-frame whose
// reference is not held refused as NotFound, and every refusal and failure
// leaving the reference as it was. Internal headers by include path; skips
// with no device.

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
#include "reference_store.hpp"
#include "volumetric_kit/core/vulkan/command_batch.hpp"
#include "volumetric_kit/recon/codec/decoder.hpp"
#include "volumetric_kit/recon/codec/encoder.hpp"

#include "gpu_test.hpp"

namespace vkc = volumetric_kit::core;
namespace codec = volumetric_kit::recon::codec;
namespace d = volumetric_kit::recon::codec::detail;
using namespace codec_fixture;
using codec_frames::append_payload;
using codec_frames::apply_residual;
using codec_frames::inter_frame;
using codec_frames::Lcg;
using codec_frames::make_frame;
using codec_frames::pack;
using codec_frames::Payload;
using codec_frames::payload_at;
using codec_frames::reference_of;
using codec_frames::successor;
using C = vkc::Status::Code;
using Bytes = std::vector<std::uint8_t>;

namespace {

constexpr std::uint32_t kMaxBlocks = 1u << 20;

// --- The store. -------------------------------------------------------------

// @p n entries of K coefficients drawn from all of int16, and masks of every
// kind.
d::DctBlocks random_entries(std::size_t n, std::uint32_t k, Lcg& rng) {
  d::DctBlocks b;
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
int commit_intra(Gpu& gpu, d::ReferenceStore& store, const d::DctBlocks& r,
                 std::uint64_t hash) {
  const std::size_t m = r.masks.size() / codec::kMaskWordsPerBlock;
  const std::uint32_t k = r.params.coefficient_count;
  std::vector<d::ApplySource> sources(m);
  for (std::size_t i = 0; i < m; ++i) {
    sources[i] = {d::kNoSource, static_cast<std::uint32_t>(i)};
  }
  CHECK(store.reserve(std::uint32_t(m), k, &sources).ok());
  vkc::CommandBatch batch(gpu.device, gpu.allocator);
  vkc::Result<d::ResidentBlocks> upload =
      store.record_upload(batch, r, sources);
  CHECK(upload.ok());
  vkc::Result<d::ResidentBlocks> spare =
      store.record_apply(batch, upload.value(), sources, row_of(m));
  CHECK(spare.ok());
  CHECK(batch.submit().ok());
  vkc::Result<Applied> got = read_spare(gpu, spare.value());
  CHECK(got.ok() && got.value().pads_zero);
  for (std::size_t i = 0; i < m; ++i) {
    CHECK(got.value().blocks[i] == payload_at(r, i));
  }
  d::FrameReference id;
  id.hash = hash;
  id.header.block_count = static_cast<std::uint32_t>(m);
  id.header.params.coefficient_count = k;
  store.commit(id, row_of(m));
  return 0;
}

// Every source kind against a held reference, its payload uploaded from the
// host or bound where the device left it, at K of 1, 21, 64 and 512, with
// residuals from all of int16 so many wrap; at @p max_blocks_per_dispatch 4,
// the nine blocks in three dispatches.
int apply_case(Gpu& gpu, std::uint32_t max_blocks_per_dispatch) {
  vkc::Result<std::unique_ptr<d::ReferenceStore>> made =
      d::ReferenceStore::create(gpu.device, gpu.allocator,
                                max_blocks_per_dispatch);
  CHECK(made.ok());
  d::ReferenceStore& store = *made.value();
  CHECK(store.held() == nullptr && store.coords().empty());
  Lcg rng{17};
  std::uint64_t hash = 1;
  for (std::uint32_t k : {1u, 21u, 64u, codec::kVoxelsPerBlock}) {
    const std::size_t m = 9;
    const d::DctBlocks r = random_entries(m, k, rng);
    CHECK(commit_intra(gpu, store, r, hash) == 0);
    CHECK(store.held() != nullptr && store.held()->hash == hash);
    CHECK(store.coords() == row_of(m));

    // A P-frame's payload by stream position: M entries, then two new.
    const d::DctBlocks payload = random_entries(m + 2, k, rng);
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
    for (const bool device : {false, true}) {
      std::vector<d::ApplySource> s = sources;
      CHECK(store.reserve(std::uint32_t(s.size()), k, device ? nullptr : &s)
                .ok());
      vkc::CommandBatch batch(gpu.device, gpu.allocator);
      // The device reader's layout, its pads not 0, which the apply drops.
      vkc::Result<vkc::Buffer> masks = vr_test::upload_device_buffer(
          gpu.device, gpu.allocator, payload.masks.data(),
          payload.masks.size() * sizeof(std::uint32_t));
      const std::vector<std::uint32_t> packed = pack(payload, 0xBEEF);
      vkc::Result<vkc::Buffer> coefficients = vr_test::upload_device_buffer(
          gpu.device, gpu.allocator, packed.data(),
          packed.size() * sizeof(std::uint32_t));
      CHECK(masks.ok() && coefficients.ok());
      d::ResidentBlocks resident;
      resident.masks = &masks.value();
      resident.coefficients = &coefficients.value();
      resident.count = std::uint32_t(m + 2);
      resident.coefficient_count = k;
      if (!device) {
        vkc::Result<d::ResidentBlocks> up =
            store.record_upload(batch, payload, s);
        CHECK(up.ok());
        resident = up.value();
        // Only the seven entries with a payload went up.
        CHECK(resident.count == 7);
      }
      vkc::Result<d::ResidentBlocks> spare =
          store.record_apply(batch, resident, s, coords);
      CHECK(spare.ok());
      CHECK(batch.submit().ok());
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
          std::fprintf(stderr, "K %u, block %zu (%s payload)\n", k, i,
                       device ? "device" : "host");
          return 1;
        }
      }
      // Applied and submitted, but not committed: the reference is as it
      // was.
      CHECK(store.held()->hash == hash && store.coords() == row_of(m));
    }
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
  d::DctBlocks held = random_entries(3, k, rng);
  CHECK(commit_intra(gpu, store, held, 100) == 0);
  for (const std::size_t n : {std::size_t(50), std::size_t(700)}) {
    // Every block a skip of a held one, then the rest new: a frame of n
    // blocks, the spare grown to hold it.
    const std::size_t m = store.coords().size();
    const d::DctBlocks fresh = random_entries(n - m, k, rng);
    std::vector<d::ApplySource> sources;
    for (std::size_t e = 0; e < m; ++e) {
      sources.push_back({std::uint32_t(e), d::kNoSource});
    }
    for (std::size_t q = 0; q < n - m; ++q) {
      sources.push_back({d::kNoSource, std::uint32_t(q)});
    }
    CHECK(store.reserve(std::uint32_t(n), k, &sources).ok());
    vkc::CommandBatch batch(gpu.device, gpu.allocator);
    vkc::Result<d::ResidentBlocks> up =
        store.record_upload(batch, fresh, sources);
    CHECK(up.ok());
    vkc::Result<d::ResidentBlocks> spare =
        store.record_apply(batch, up.value(), sources, row_of(n));
    CHECK(spare.ok());
    CHECK(batch.submit().ok());
    vkc::Result<Applied> got = read_spare(gpu, spare.value());
    CHECK(got.ok());
    d::DctBlocks next;
    next.params.coefficient_count = k;
    for (std::size_t i = 0; i < n; ++i) {
      const Payload want =
          i < m ? payload_at(held, i) : payload_at(fresh, i - m);
      CHECK(got.value().blocks[i] == want);
      append_payload(next, want);
    }
    // Refused, and recorded but never committed: nothing changes.
    const d::FrameReference before = *store.held();
    vkc::CommandBatch refused(gpu.device, gpu.allocator);
    CHECK(store.record_apply(refused, up.value(), sources, row_of(n - 1))
              .status()
              .domain() == C::InvalidArgument);
    CHECK(store.held()->hash == before.hash && store.coords() == row_of(m));
    d::FrameReference id;
    id.hash = before.hash + 1;
    id.header.block_count = std::uint32_t(n);
    id.header.params.coefficient_count = k;
    store.commit(id, row_of(n));
    CHECK(store.held()->hash == id.hash && store.coords() == row_of(n));
    held = std::move(next);
  }
  return 0;
}

// What the apply refuses before it records anything.
int apply_refusals_case(Gpu& gpu) {
  vkc::Result<std::unique_ptr<d::ReferenceStore>> made =
      d::ReferenceStore::create(gpu.device, gpu.allocator);
  CHECK(made.ok());
  d::ReferenceStore& store = *made.value();
  Lcg rng{29};
  CHECK(commit_intra(gpu, store, random_entries(4, 8, rng), 7) == 0);
  const d::DctBlocks payload = random_entries(4, 8, rng);
  const std::uint32_t none = d::kNoSource;
  // Room for the whole payload uploaded.
  const std::vector<d::ApplySource> every = {
      {none, 0}, {none, 1}, {none, 2}, {none, 3}};
  CHECK(store.reserve(4, 8, &every).ok());
  // The payload uploaded as given, then applied at @p k over @p blocks.
  auto refuses = [&](std::vector<d::ApplySource> sources, std::uint32_t k,
                     std::size_t blocks, const char* says) {
    vkc::CommandBatch batch(gpu.device, gpu.allocator);
    vkc::Result<d::ResidentBlocks> r =
        store.record_upload(batch, payload, sources);
    if (r.ok()) {
      d::ResidentBlocks up = r.value();
      up.coefficient_count = k;
      r = store.record_apply(batch, up, sources, row_of(blocks));
    }
    return !r.ok() && r.status().domain() == C::InvalidArgument &&
           r.status().message().find(says) != std::string::npos;
  };
  CHECK(refuses({{0, none}, {4, none}}, 8, 2, "reads reference entry 4"));
  CHECK(refuses({{0, 0}}, 9, 1, "at K 9"));
  CHECK(refuses({{none, 4}}, 8, 1, "names entry 4 of 4"));
  CHECK(refuses(std::vector<d::ApplySource>(100, {0, none}), 8, 100,
                "more than were reserved"));
  CHECK(refuses({{0, none}}, 8, 2, "1 sources for 2 blocks"));
  CHECK(store.reserve(1u << 31, 8, nullptr).domain() == C::InvalidArgument);
  CHECK(store.held()->hash == 7 && store.coords() == row_of(4));
  return 0;
}

// --- The Decoder. -----------------------------------------------------------

codec::DecoderConfig on(codec::EntropyCoding entropy) {
  codec::DecoderConfig c;
  c.entropy = entropy;
  return c;
}

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

// What an intra frame decodes to: a decoder of its own, a fresh grid.
vkc::Result<Snapshot> intra_decode(Gpu& gpu, const Bytes& frame) {
  VKC_ASSIGN(codec::Decoder dec,
             codec::Decoder::create(gpu.device, gpu.allocator,
                                    on(codec::EntropyCoding::kHost)));
  VKC_ASSIGN(vol::VoxelBlockGrid grid, grid_for(gpu, frame));
  VKC_TRY(dec.decode(frame.data(), frame.size(), grid));
  return snapshot(gpu, grid);
}

// Whether @p grid holds what @p frame, an intra frame, decodes to.
bool holds(Gpu& gpu, vol::VoxelBlockGrid& grid, const Bytes& frame) {
  vkc::Result<Snapshot> want = intra_decode(gpu, frame);
  vkc::Result<Snapshot> got = snapshot(gpu, grid);
  return want.ok() && got.ok() && want.value() == got.value();
}

// The segments a frame holds, from its SEGMENTS section.
std::size_t segments_of(const Bytes& frame) {
  for (const codec_frames::Section& s : codec_frames::sections_of(frame)) {
    if (s.id == static_cast<std::uint16_t>(d::SectionId::kSegments)) {
      return s.body.size() / 4;
    }
  }
  return 0;
}

d::FrameWriteOptions sizes(std::uint32_t r, std::uint32_t u) {
  d::FrameWriteOptions o;
  o.segment_size = r;
  o.update_size = u;
  return o;
}

// One step of a chain: the content a frame decodes to, its intra frame, and
// the frame sent, an I-frame or a P-frame against the step before.
struct Step {
  d::Frame content;
  Bytes intra;
  Bytes sent;
};

// The P-frame of @p next against @p prev's sent bytes, at @p options, with
// @p overrides' modes.
Step p_step(const Step& prev, d::Frame next,
            const d::FrameWriteOptions& options,
            const std::map<std::size_t, d::BlockMode>& overrides = {}) {
  Step s;
  s.content = std::move(next);
  s.intra = d::write_frame(s.content).value();
  s.sent = d::write_frame(inter_frame(reference_of(prev.sent), prev.content,
                                      s.content, overrides),
                          options)
               .value();
  return s;
}

Step i_step(const Bytes& intra) {
  Step s;
  s.content = d::read_frame(intra.data(), intra.size(), kMaxBlocks).value();
  s.intra = intra;
  s.sent = intra;
  return s;
}

// Decode a chain's frames in order into one grid, which after each must hold
// that step's intra decode; under each entropy path.
int run_chain(Gpu& gpu, const std::vector<Step>& chain, const char* what) {
  for (codec::EntropyCoding entropy :
       {codec::EntropyCoding::kHost, codec::EntropyCoding::kDevice,
        codec::EntropyCoding::kAuto}) {
    vkc::Result<codec::Decoder> dec =
        codec::Decoder::create(gpu.device, gpu.allocator, on(entropy));
    vkc::Result<vol::VoxelBlockGrid> grid = grid_for(gpu, chain[0].sent);
    CHECK(dec.ok() && grid.ok());
    for (std::size_t i = 0; i < chain.size(); ++i) {
      const Bytes& f = chain[i].sent;
      const vkc::Status s =
          dec.value().decode(f.data(), f.size(), grid.value());
      if (!s.ok() || !holds(gpu, grid.value(), chain[i].intra)) {
        std::fprintf(stderr, "%s: step %zu, entropy %d: %s\n", what, i,
                     int(entropy), s.ok() ? "differs" : s.message().c_str());
        return 1;
      }
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
  const d::Frame f =
      d::read_frame(p.sent.data(), p.sent.size(), kMaxBlocks, &id).value();
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
    chain.push_back(i_step(d::write_frame(make_frame(n, k, 500 + k)).value()));
    const d::Frame f1 = successor(chain[0].content, k, 80, 400, n / 5);
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
    extreme.push_back(i_step(d::write_frame(x.reference).value()));
    extreme.push_back(p_step(extreme[0], x.target, sizes(4, 3)));
    CHECK(
        census(extreme[0], extreme[1]).modes[std::size_t(d::BlockMode::kZero)] >
        0);
    what = "extremes, K " + std::to_string(k);
    CHECK(run_chain(gpu, extreme, what.c_str()) == 0);
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
  config.segment_size = 3;  // enough segments for kAuto to decode on device
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
  // Long segments, then short ones: kAuto decodes the I-frame and the
  // second P-frame on the device, and the first on the host.
  const d::FrameWriteOptions few = sizes(64, 64);
  const d::FrameWriteOptions many = sizes(1, 2);
  for (int t = 1; t < 3; ++t) {
    Step s = p_step(out.chain.back(), i_step(intra[std::size_t(t)]).content,
                    t == 1 ? few : many);
    s.intra = intra[std::size_t(t)];  // the target's own I-frame
    out.chain.push_back(std::move(s));
  }
  return out;
}

// The sphere chain on every entropy path, kAuto crossing from the device to
// the host and back; the last P-frame into a fresh grid built from its own
// header.
int sphere_case(Gpu& gpu, const Spheres& s) {
  const std::vector<Step>& chain = s.chain;
  CHECK(segments_of(chain[0].sent) >= codec::kMinDeviceDecodeSegments);
  CHECK(segments_of(chain[1].sent) < codec::kMinDeviceDecodeSegments);
  CHECK(segments_of(chain[2].sent) >= codec::kMinDeviceDecodeSegments);
  const Census c = census(chain[0], chain[1]);
  CHECK(c.modes[std::size_t(d::BlockMode::kKeep)] > 0);
  CHECK(c.modes[std::size_t(d::BlockMode::kRemove)] > 0);
  CHECK(c.added > 0);
  CHECK(run_chain(gpu, chain, "spheres") == 0);

  vkc::Result<codec::Decoder> dec = codec::Decoder::create(
      gpu.device, gpu.allocator, on(codec::EntropyCoding::kAuto));
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
  vkc::Result<codec::Decoder> made = codec::Decoder::create(
      gpu.device, gpu.allocator, on(codec::EntropyCoding::kAuto));
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
  vkc::Result<codec::Decoder> other = codec::Decoder::create(
      gpu.device, gpu.allocator, on(codec::EntropyCoding::kHost));
  vkc::Result<codec::Decoder> first = codec::Decoder::create(
      gpu.device, gpu.allocator, on(codec::EntropyCoding::kHost));
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

// InvalidArgument from checks outside the device decode, under kAuto at a
// size it decodes on the device -- a new block on a reference block (after
// it), and a scale not the reference's (before it) -- leaving the grid and
// the reference as they were.
int invalid_case(Gpu& gpu, const Spheres& s) {
  const std::vector<Step>& chain = s.chain;
  // The second P-frame with one more new block: the coordinate of a block
  // its reference has and it removes.
  const d::FrameReference id = reference_of(chain[1].sent);
  d::Frame p = inter_frame(id, chain[1].content, chain[2].content);
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
  d::DctBlocks blocks;
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
  const Bytes on_reference = d::write_frame(p, sizes(1, 2)).value();
  CHECK(segments_of(on_reference) >= codec::kMinDeviceDecodeSegments);

  Bytes rescaled = chain[2].sent;
  codec_frames::put_f32(rescaled, 24, 0.3f);

  vkc::Result<codec::Decoder> made = codec::Decoder::create(
      gpu.device, gpu.allocator, on(codec::EntropyCoding::kAuto));
  vkc::Result<vol::VoxelBlockGrid> g = grid_for(gpu, chain[0].sent);
  CHECK(made.ok() && g.ok());
  codec::Decoder& dec = made.value();
  for (int i = 0; i < 2; ++i) {
    const Bytes& f = chain[std::size_t(i)].sent;
    CHECK(dec.decode(f.data(), f.size(), g.value()).ok());
  }
  CHECK(refused(gpu, dec, on_reference, g.value(), C::InvalidArgument,
                "is on reference block") == 0);
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
  few.push_back(i_step(d::write_frame(make_frame(12, 64, 41)).value()));
  few.push_back(
      p_step(few[0], successor(few[0].content, 42, 100, 400, 6), sizes(4, 4)));
  few.push_back(
      p_step(few[1], successor(few[1].content, 43, 100, 400, 2), sizes(4, 4)));
  for (codec::EntropyCoding entropy :
       {codec::EntropyCoding::kHost, codec::EntropyCoding::kDevice}) {
    vkc::Result<codec::Decoder> made =
        codec::Decoder::create(gpu.device, gpu.allocator, on(entropy));
    vkc::Result<vol::VoxelBlockGrid> big = grid_for(gpu, chain[0].sent);
    CHECK(made.ok() && big.ok());
    codec::Decoder& dec = made.value();
    CHECK(dec.decode(chain[0].sent.data(), chain[0].sent.size(), big.value())
              .ok());

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
    CHECK(
        dec.decode(few[2].sent.data(), few[2].sent.size(), tight.value()).ok());
    CHECK(holds(gpu, tight.value(), few[2].intra));
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

}  // namespace

int gpu_main(vr_test::GpuContext& gpu) {
  if (apply_case(gpu, 0) != 0) return 1;
  if (apply_case(gpu, 4) != 0) return 1;
  if (grow_case(gpu) != 0) return 1;
  if (apply_refusals_case(gpu) != 0) return 1;
  if (synthetic_case(gpu) != 0) return 1;
  vkc::Result<Spheres> s = spheres(gpu);
  CHECK(s.ok());
  if (sphere_case(gpu, s.value()) != 0) return 1;
  if (not_found_case(gpu, s.value()) != 0) return 1;
  if (invalid_case(gpu, s.value()) != 0) return 1;
  if (out_of_memory_case(gpu, s.value()) != 0) return 1;
  if (moves_case(gpu, s.value()) != 0) return 1;
  std::printf("codec P-frame decode: OK\n");
  return 0;
}

int main() { return vr_test::run_on_gpu(gpu_main); }
