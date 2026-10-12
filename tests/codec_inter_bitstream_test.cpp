// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// Host-only tests for P-frames in the v3 frame (bitstream.hpp, type 1): the
// reference hash's known answers; the worked example, built by hand, read and
// written byte for byte; round trips of every mode across K, update and
// segment sizes and the empty cases; I -> P -> P chains on real hashes against
// an independent oracle (codec_frames.hpp); merge_by_coord against std::set's
// algorithms; every reader refusal, each one edit from a valid P-frame, in its
// order and with its code; every writer refusal; and truncation and random
// corruption, which must fail cleanly and never read out of bounds.
// CPU-only, so these always run.

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iterator>
#include <limits>
#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "bitstream.hpp"
#include "codec_frames.hpp"
#include "frame_hash.hpp"
#include "test_check.hpp"
#include "volumetric_kit/recon/codec/codec_params.hpp"

namespace vr = volumetric_kit::recon;
namespace vkc = volumetric_kit::core;
namespace codec = volumetric_kit::recon::codec;
namespace d = volumetric_kit::recon::codec::detail;

namespace {

using codec_frames::get_u32;
using codec_frames::inter_frame;
using codec_frames::Lcg;
using codec_frames::make_frame;
using codec_frames::put_u32;
using codec_frames::reference_of;
using codec_frames::resolve;
using codec_frames::successor;
using codec_frames::ZyxLess;
using C = vkc::Status::Code;
using Bytes = std::vector<std::uint8_t>;
using Modes = std::map<std::size_t, d::BlockMode>;

constexpr std::int32_t kMax32 = std::numeric_limits<std::int32_t>::max();
constexpr std::int32_t kMin32 = std::numeric_limits<std::int32_t>::min();

bool same_bits(float a, float b) { return std::memcmp(&a, &b, sizeof(a)) == 0; }

// The same content, parameters included: what an intra frame holds.
bool same_content(const d::Frame& a, const d::Frame& b) {
  return same_bits(a.voxel_size, b.voxel_size) &&
         same_bits(a.blocks.trunc_dist, b.blocks.trunc_dist) &&
         a.blocks.params.coefficient_count ==
             b.blocks.params.coefficient_count &&
         same_bits(a.blocks.params.quantization_scale,
                   b.blocks.params.quantization_scale) &&
         a.coords == b.coords &&
         a.blocks.coefficients == b.blocks.coefficients &&
         a.blocks.masks == b.blocks.masks;
}

// A P-frame as written and as read: its reference, modes, new blocks and
// every entry's payload (skip and remove are zero on both sides).
bool same_inter(const d::Frame& a, const d::Frame& b) {
  return a.type == d::FrameType::kInter && b.type == d::FrameType::kInter &&
         a.reference.hash == b.reference.hash &&
         a.reference.header.block_count == b.reference.header.block_count &&
         a.modes == b.modes && same_content(a, b);
}

bool same_merge(const d::MergedFrame& a, const d::MergedFrame& b) {
  if (a.coords != b.coords || a.sources.size() != b.sources.size()) {
    return false;
  }
  for (std::size_t i = 0; i < a.sources.size(); ++i) {
    if (a.sources[i].reference != b.sources[i].reference ||
        a.sources[i].payload != b.sources[i].payload) {
      return false;
    }
  }
  return true;
}

d::FrameWriteOptions sizes(std::uint32_t r, std::uint32_t u) {
  d::FrameWriteOptions o;
  o.segment_size = r;
  o.update_size = u;
  return o;
}

// --- The reference hash. ----------------------------------------------------

// xxHash's own sanity input (cli/xsum_sanity_check.c): each byte the top
// byte of a generator that starts at PRIME32_1 and multiplies by PRIME64_1.
Bytes xxhash_sanity_buffer(std::size_t n) {
  Bytes out(n);
  std::uint64_t gen = 2654435761u;
  for (std::uint8_t& b : out) {
    b = static_cast<std::uint8_t>(gen >> 56);
    gen *= 11400714785074694797ull;
  }
  return out;
}

std::uint64_t hash_of(const std::string& s) {
  return d::frame_hash(reinterpret_cast<const std::uint8_t*>(s.data()),
                       s.size());
}

// Known answers of XXH64, seed 0: the short inputs, and xxHash's own vectors
// over its sanity buffer, 222 bytes of which run the four-lane loop six
// times and then every tail (8, 4 and 1 bytes). Checked against the
// reference implementation (python-xxhash 4.0.1, xxHash 0.8.3).
int hash_case() {
  CHECK(d::frame_hash(nullptr, 0) == 0xEF46DB3751D8E999ull);
  CHECK(hash_of("") == 0xEF46DB3751D8E999ull);
  CHECK(hash_of("a") == 0xD24EC4F1A98C6E5Bull);
  CHECK(hash_of("abc") == 0x44BC2CF5AD770999ull);
  CHECK(hash_of("The quick brown fox jumps over the lazy dog") ==
        0x0B242D361FDA71BCull);
  const Bytes sanity = xxhash_sanity_buffer(222);
  CHECK(d::frame_hash(sanity.data(), 14) == 0x8282DCC4994E35C8ull);
  CHECK(d::frame_hash(sanity.data(), 222) == 0xB641AE8CB691C174ull);
  // Bytes read with shifts: any alignment gives the same value.
  Bytes shifted(sanity.size() + 1);
  std::memcpy(shifted.data() + 1, sanity.data(), sanity.size());
  CHECK(d::frame_hash(shifted.data() + 1, 222) == 0xB641AE8CB691C174ull);
  return 0;
}

// --- The worked example. ----------------------------------------------------

// The reference of the worked example: a = (0,0,0) and b = (4,0,0), K = 1,
// voxel 1 cm, band 4 cm, scale 0.2, weight 1.
d::Frame example_reference() {
  d::Frame f = make_frame(0, 1, 1);
  f.voxel_size = 0.01f;
  f.blocks.trunc_dist = 0.04f;
  f.coords = {{0, 0, 0}, {4, 0, 0}};
  f.blocks.masks.assign(2 * codec::kMaskWordsPerBlock, ~0u);
  for (std::uint32_t w = 0; w < codec::kMaskWordsPerBlock; w += 2) {
    f.blocks.masks[w] = 0x0F0F0F0Fu;  // a partial
  }
  f.blocks.coefficients = {5, -2};
  return f;
}

// The worked example's P-frame, by hand: a kept with residual +1 and its mask
// unchanged, b skipped, and c = (2,0,0) new with a full mask and coefficient
// 3. Its REFERENCE names @p hash.
Bytes example_bytes(std::uint64_t hash) {
  Bytes b = {0x56, 0x52, 0x54, 0x43, 0x03, 0x00, 0x01,
             0x00,  // VRTC, v3, type 1
             0x0A, 0xD7, 0x23, 0x3C, 0x0A, 0xD7, 0x23, 0x3D,  // 0.01, 0.04
             0x08, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00,  // block size, K
             0xCD, 0xCC, 0x4C, 0x3E, 0x01, 0x00, 0x00, 0x00,  // scale, weights
             0x03, 0x00, 0x00, 0x00, 0x40, 0x00, 0x00, 0x00,  // N = 3, R = 64
             0x04, 0x00, 0x00, 0x00, 0x00, 0x00, 0x80, 0x3F,  // S = 4, weight 1
             // Sections: REFERENCE, TABLES, SEGMENTS, PAYLOAD.
             0x04, 0x00, 0x01, 0x00, 0x14, 0x00, 0x00, 0x00,  //
             0x01, 0x00, 0x01, 0x00, 0x21, 0x00, 0x00, 0x00,  //
             0x02, 0x00, 0x01, 0x00, 0x08, 0x00, 0x00, 0x00,  //
             0x03, 0x00, 0x01, 0x00, 0x14, 0x00, 0x00, 0x00,  //
             // REFERENCE: the hash (below), M = 2, U = 64, A = 1.
             0, 0, 0, 0, 0, 0, 0, 0,                          //
             0x02, 0x00, 0x00, 0x00, 0x40, 0x00, 0x00, 0x00,  //
             0x01, 0x00, 0x00, 0x00,                          //
             // TABLES: five coordinate models unused; the mask class, full and
             // empty; seven plane, line and byte models unused; coefficient 0,
             // classes 1 and 2; the modes, keep and skip.
             0x00, 0x00, 0x00, 0x00, 0x00,              //
             0x02, 0x00, 0x80, 0x10, 0x00, 0x80, 0x10,  //
             0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,  //
             0x02, 0x01, 0x80, 0x10, 0x00, 0x80, 0x10,  //
             0x02, 0x00, 0x80, 0x10, 0x00, 0x80, 0x10,  //
             // SEGMENTS: an update segment of 4 bytes, a new segment of 16.
             0x04, 0x00, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00,  //
             // keep, mask unchanged, class 1 with sign 0, then skip: the final
             // state alone.
             0x20, 0x00, 0x00, 0x90,  //
             // (2,0,0) in 96 raw bits, mask full, class 2 with field 10.
             0x10, 0x00, 0x02, 0x30, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  for (int i = 0; i < 8; ++i) {
    b[80 + std::size_t(i)] = static_cast<std::uint8_t>(hash >> (8 * i));
  }
  return b;
}

// The worked example of the format: 161 bytes, read as the format says and
// written back byte for byte. Pinned as the example; it may change until
// golden P-frames freeze type 1.
int worked_example_case() {
  const d::Frame reference = example_reference();
  const d::FrameReference id = reference_of(d::write_frame(reference).value());
  const Bytes bytes = example_bytes(id.hash);
  CHECK(bytes.size() == 161);

  const vkc::Result<d::Frame> read =
      d::read_frame(bytes.data(), bytes.size(), 3, &id);
  CHECK(read.ok());
  const d::Frame& p = read.value();
  CHECK(p.type == d::FrameType::kInter);
  CHECK(p.modes ==
        std::vector<d::BlockMode>({d::BlockMode::kKeep, d::BlockMode::kSkip}));
  CHECK(p.coords == std::vector<vr::Vec3i>({{2, 0, 0}}));
  // Positions 0 (a), 1 (b) and 2 (c).
  CHECK(p.blocks.coefficients == std::vector<std::int16_t>({1, 0, 3}));
  const std::vector<std::uint32_t> none(codec::kMaskWordsPerBlock, 0u);
  const std::vector<std::uint32_t> full(codec::kMaskWordsPerBlock, ~0u);
  std::vector<std::uint32_t> masks = none;
  masks.insert(masks.end(), none.begin(), none.end());
  masks.insert(masks.end(), full.begin(), full.end());
  CHECK(p.blocks.masks == masks);

  d::Frame written = p;
  CHECK(d::write_frame(written).value() == bytes);

  // a <- (0, 0), c <- (none, 2), b <- (1, none).
  const vkc::Result<d::MergedFrame> merged =
      d::merge_frame(reference.coords, p.modes, p.coords);
  CHECK(merged.ok());
  CHECK(merged.value().coords ==
        std::vector<vr::Vec3i>({{0, 0, 0}, {2, 0, 0}, {4, 0, 0}}));
  const std::uint32_t no = d::kNoSource;
  const d::ApplySource expected[] = {{0, 0}, {no, 2}, {1, no}};
  for (std::size_t i = 0; i < 3; ++i) {
    CHECK(merged.value().sources[i].reference == expected[i].reference);
    CHECK(merged.value().sources[i].payload == expected[i].payload);
  }
  // q = (q_a + 1, 3, q_b), masks (mask_a, full, mask_b).
  const d::Frame resolved = resolve(reference, p);
  CHECK(resolved.blocks.coefficients == std::vector<std::int16_t>({6, 3, -2}));
  std::vector<std::uint32_t> resolved_masks(
      reference.blocks.masks.begin(),
      reference.blocks.masks.begin() + codec::kMaskWordsPerBlock);
  resolved_masks.insert(resolved_masks.end(), full.begin(), full.end());
  resolved_masks.insert(
      resolved_masks.end(),
      reference.blocks.masks.begin() + codec::kMaskWordsPerBlock,
      reference.blocks.masks.end());
  CHECK(resolved.blocks.masks == resolved_masks);

  // At U = 1, a's entry and b's are update segments of their own, and c
  // follows: the layout laid out here from the format text, each stream
  // straight from the coder over the example's tables.
  auto table = [](std::uint32_t alphabet, std::vector<std::uint32_t> symbols) {
    std::vector<std::uint64_t> counts(alphabet, 0);
    for (std::uint32_t symbol : symbols) ++counts[symbol];
    return d::normalize_counts(counts);
  };
  const d::FrequencyTable mask_class = table(3, {0, 1});
  const d::FrequencyTable coefficient = table(16, {1, 2});
  const d::FrequencyTable mode = table(5, {0, 1});
  Bytes payload;
  std::vector<std::uint32_t> lengths;
  auto stream = [&](auto&& put) {
    d::RansWriter w;
    put(w);
    const std::size_t at = payload.size();
    CHECK(w.finish(payload));
    lengths.push_back(static_cast<std::uint32_t>(payload.size() - at));
    return 0;
  };
  CHECK(stream([&](d::RansWriter& w) {  // a: keep, unchanged, +1
          w.put(mode, 0);
          w.put(mask_class, 1);
          w.put(coefficient, 1);
          w.put_bits(0, 1);
        }) == 0);
  CHECK(stream([&](d::RansWriter& w) { w.put(mode, 1); }) == 0);  // b: skip
  CHECK(stream([&](d::RansWriter& w) {  // c: (2,0,0), full, +3
          w.put_bits(2, 32);
          w.put_bits(0, 32);
          w.put_bits(0, 32);
          w.put(mask_class, 0);
          w.put(coefficient, 2);
          w.put_bits(2, 2);
        }) == 0);
  std::vector<codec_frames::Section> s = codec_frames::sections_of(bytes);
  put_u32(s[0].body, 12, 1);
  s[2].body.clear();
  for (std::uint32_t length : lengths) {
    s[2].body.resize(s[2].body.size() + 4);
    put_u32(s[2].body, s[2].body.size() - 4, length);
  }
  s[3].body = payload;
  const Bytes one_each = codec_frames::assemble(bytes, s);
  CHECK(d::write_frame(written, sizes(64, 1)).value() == one_each);
  CHECK(d::read_frame(one_each.data(), one_each.size(), 3, &id).ok());

  // Had b been removed: mode 3, N = 2, the mode table remove for skip, and
  // the payload unchanged.
  Bytes removed = bytes;
  put_u32(removed, 32, 2);
  removed[130] = 0x02;  // the gap from keep to remove
  written.modes[1] = d::BlockMode::kRemove;
  CHECK(d::write_frame(written).value() == removed);
  const vkc::Result<d::Frame> read_removed =
      d::read_frame(removed.data(), removed.size(), 2, &id);
  CHECK(read_removed.ok());
  CHECK(read_removed.value().modes[1] == d::BlockMode::kRemove);
  return 0;
}

// --- Round trips. -----------------------------------------------------------

// Write @p p (coding @p target against @p reference), read it back against
// its reference, and check it is @p p, writes the same bytes again, merges
// as the oracle does and resolves to @p target, whose intra bytes the
// resolved frame writes too.
int round_trip(const d::Frame& reference, const d::Frame& target,
               const d::Frame& p, const d::FrameWriteOptions& options,
               Bytes* out = nullptr) {
  const vkc::Result<Bytes> bytes = d::write_frame(p, options);
  if (!bytes.ok()) {
    std::fprintf(stderr, "write: %s\n", bytes.status().message().c_str());
  }
  CHECK(bytes.ok());
  const Bytes& b = bytes.value();
  const auto n = static_cast<std::uint32_t>(target.coords.size());
  CHECK(get_u32(b, 32) == n);
  const vkc::Result<d::Frame> back =
      d::read_frame(b.data(), b.size(), n, &p.reference);
  if (!back.ok()) {
    std::fprintf(stderr, "read: %s\n", back.status().message().c_str());
  }
  CHECK(back.ok());
  CHECK(same_inter(p, back.value()));
  CHECK(d::write_frame(back.value(), options).value() == b);

  const vkc::Result<d::MergedFrame> merged =
      d::merge_frame(reference.coords, back.value().modes, back.value().coords);
  CHECK(merged.ok());
  CHECK(same_merge(merged.value(), codec_frames::merge_oracle(
                                       reference.coords, back.value().modes,
                                       back.value().coords)));
  CHECK(merged.value().coords == target.coords);

  const d::Frame resolved = resolve(reference, back.value());
  CHECK(same_content(resolved, target));
  d::FrameWriteOptions intra;
  intra.segment_size = options.segment_size;
  CHECK(d::write_frame(resolved, intra).value() ==
        d::write_frame(target, intra).value());
  if (out != nullptr) *out = b;
  return 0;
}

// Spellings beyond the automatic choice: zero for some surviving blocks,
// and keep, with zero residuals and an unchanged mask, for some unchanged
// ones -- each a second spelling of the same block.
Modes spellings(const d::Frame& reference, const d::Frame& target) {
  std::set<vr::Vec3i, ZyxLess> unchanged;
  std::map<vr::Vec3i, std::size_t, ZyxLess> at;
  for (std::size_t i = 0; i < target.coords.size(); ++i) {
    at.emplace(target.coords[i], i);
  }
  Modes out;
  for (std::size_t e = 0; e < reference.coords.size(); ++e) {
    const auto it = at.find(reference.coords[e]);
    if (it == at.end()) continue;
    if (e % 9 == 4) out[e] = d::BlockMode::kZero;
    if (e % 7 == 3 && codec_frames::payload_at(reference.blocks, e) ==
                          codec_frames::payload_at(target.blocks, it->second)) {
      out[e] = d::BlockMode::kKeep;
    }
  }
  return out;
}

// Extreme coefficients: residuals of +-32767, differences that wrap, and
// differences of +-32768, which no class carries, so the block is zero.
struct Extremes {
  d::Frame reference;
  d::Frame target;
};

Extremes extremes(std::uint32_t k) {
  Extremes x;
  x.reference = make_frame(12, k, 99);
  x.target = x.reference;
  const std::int16_t from[] = {0, 0, -32767, 32767, -16384, 16384, -1};
  const std::int16_t to[] = {32767, -32767, 32767, -32767,
                             16384, -16384, 32767};
  for (std::size_t i = 0; i < 7; ++i) {
    x.reference.blocks.coefficients[i * k] = from[i];
    x.target.blocks.coefficients[i * k] = to[i];
  }
  return x;
}

// Every mode, K of 1, 64 and 512, update and segment sizes of 1, 7, 64 and
// past the frame, and the empty cases: no reference blocks (M = 0), no new
// ones (A = 0), none left (N = 0), all removed with new ones, all skipped,
// and the extreme coefficients.
int round_trip_case() {
  std::uint64_t seen[5] = {};
  for (std::uint32_t k : {1u, 64u, codec::kVoxelsPerBlock}) {
    const std::size_t m = k == codec::kVoxelsPerBlock ? 24 : 60;
    const d::Frame reference = make_frame(m, k, 1000 + k);
    const d::Frame empty = make_frame(0, k, 1);
    const Extremes x = extremes(k);
    struct Case {
      const char* what;
      const d::Frame* reference;
      d::Frame target;
      bool spell;
    };
    const Case cases[] = {
        {"every mode", &reference, successor(reference, k, 80, 400, m / 6),
         true},
        {"M = 0", &empty, make_frame(m / 2, k, 2000 + k), false},
        {"A = 0", &reference, successor(reference, k + 1, 80, 400, 0), true},
        {"N = 0", &reference, successor(reference, k + 2, 1000, 0, 0), false},
        {"all removed", &reference, successor(reference, k + 3, 1000, 0, 9),
         false},
        {"all skipped", &reference, reference, false},
        {"extremes", &x.reference, x.target, false},
    };
    for (const Case& c : cases) {
      const d::FrameReference id =
          reference_of(d::write_frame(*c.reference).value());
      const d::Frame p =
          inter_frame(id, *c.reference, c.target,
                      c.spell ? spellings(*c.reference, c.target) : Modes{});
      for (const d::BlockMode mode : p.modes) ++seen[std::size_t(mode)];
      for (std::uint32_t u : {1u, 7u, 64u, 100000u}) {
        for (std::uint32_t r : {1u, 7u, 64u, 100000u}) {
          if (round_trip(*c.reference, c.target, p, sizes(r, u)) != 0) {
            std::fprintf(stderr, "%s: K %u, U %u, R %u\n", c.what, k, u, r);
            return 1;
          }
        }
      }
    }
    // The extremes code as they must: keep with residuals +-32767 and
    // wrapped to -2 and 2, and zero where the difference is +-32768.
    const d::FrameReference id =
        reference_of(d::write_frame(x.reference).value());
    const d::Frame p = inter_frame(id, x.reference, x.target);
    const std::int16_t residuals[] = {32767, -32767, -2, 2};
    for (std::size_t e = 0; e < 4; ++e) {
      CHECK(p.modes[e] == d::BlockMode::kKeep);
      CHECK(p.blocks.coefficients[e * k] == residuals[e]);
    }
    for (std::size_t e = 4; e < 7; ++e) {
      CHECK(p.modes[e] == d::BlockMode::kZero);
    }
    for (std::size_t e = 7; e < p.modes.size(); ++e) {
      CHECK(p.modes[e] == d::BlockMode::kSkip);
    }
  }
  std::printf("round trips: %llu keep, %llu skip, %llu zero, %llu remove\n",
              static_cast<unsigned long long>(seen[0]),
              static_cast<unsigned long long>(seen[1]),
              static_cast<unsigned long long>(seen[2]),
              static_cast<unsigned long long>(seen[3]));
  for (std::size_t mode = 0; mode < 4; ++mode) CHECK(seen[mode] > 0);
  CHECK(seen[4] == 0);
  return 0;
}

// I -> P -> P ..., each P-frame against the frame before on its real hash:
// it decodes to its target, as the oracle resolves it, and only against its
// own reference.
int chain_case() {
  d::Frame previous = make_frame(300, 64, 41);
  d::FrameWriteOptions intra;
  intra.segment_size = 16;
  d::FrameReference id = reference_of(d::write_frame(previous, intra).value());
  d::FrameReference stale;
  bool have_stale = false;
  const std::uint32_t update_sizes[] = {64, 1, 7, 300, 16, 64};
  for (std::uint32_t step = 0; step < 6; ++step) {
    const d::Frame target =
        successor(previous, 100 + step, 20 + 20 * step, 300, 10 + 5 * step);
    const d::Frame p = inter_frame(id, previous, target);
    Bytes bytes;
    CHECK(round_trip(previous, target, p, sizes(16, update_sizes[step]),
                     &bytes) == 0);
    if (have_stale) {
      const vkc::Result<d::Frame> wrong =
          d::read_frame(bytes.data(), bytes.size(), 1u << 20, &stale);
      CHECK(!wrong.ok() && wrong.status().domain() == C::NotFound);
    }
    stale = id;
    have_stale = true;
    id = reference_of(bytes);
    CHECK(id.header.type == d::FrameType::kInter);
    CHECK(id.header.block_count == target.coords.size());
    previous = target;
  }
  return 0;
}

// --- merge_by_coord. --------------------------------------------------------

// Against std::set's algorithms on random sorted lists, empty ones and int32
// extremes included: one visit per distinct coordinate, in order, each side
// once and in order, and the walk stops at the first refusal.
int merge_by_coord_case() {
  Lcg rng{7};
  const std::int32_t picks[] = {kMin32, -1, 0, 1, 2, kMax32};
  auto draw = [&]() {
    const std::int32_t x = picks[rng.below(6)];
    const std::int32_t y = picks[rng.below(6)];
    const std::int32_t z = picks[rng.below(6)];
    return vr::Vec3i(x, y, z);
  };
  using Set = std::set<vr::Vec3i, ZyxLess>;
  for (int trial = 0; trial < 300; ++trial) {
    Set left;
    Set right;
    const std::uint32_t nl = trial % 10 == 0 ? 0 : rng.below(40);
    const std::uint32_t nr = trial % 10 == 1 ? 0 : rng.below(40);
    for (std::uint32_t i = 0; i < nl; ++i) left.insert(draw());
    for (std::uint32_t i = 0; i < nr; ++i) right.insert(draw());
    const std::vector<vr::Vec3i> l(left.begin(), left.end());
    const std::vector<vr::Vec3i> r(right.begin(), right.end());
    std::vector<std::pair<std::size_t, std::size_t>> visits;
    CHECK(d::merge_by_coord(
              l.size(), [&](std::size_t i) -> const vr::Vec3i& { return l[i]; },
              r.size(), [&](std::size_t j) -> const vr::Vec3i& { return r[j]; },
              [&](std::size_t i, std::size_t j) {
                visits.emplace_back(i, j);
                return vkc::Status{};
              })
              .ok());
    std::vector<vr::Vec3i> all, both, only_left, only_right;
    std::size_t next_i = 0;
    std::size_t next_j = 0;
    for (const auto& [i, j] : visits) {
      CHECK(i != d::kAbsent || j != d::kAbsent);
      if (i != d::kAbsent) CHECK(i == next_i++);
      if (j != d::kAbsent) CHECK(j == next_j++);
      if (i != d::kAbsent && j != d::kAbsent) CHECK(l[i] == r[j]);
      const vr::Vec3i& c = i != d::kAbsent ? l[i] : r[j];
      all.push_back(c);
      if (i != d::kAbsent && j != d::kAbsent) both.push_back(c);
      if (j == d::kAbsent) only_left.push_back(c);
      if (i == d::kAbsent) only_right.push_back(c);
    }
    CHECK(next_i == l.size() && next_j == r.size());
    std::vector<vr::Vec3i> expected;
    std::set_union(l.begin(), l.end(), r.begin(), r.end(),
                   std::back_inserter(expected), ZyxLess{});
    CHECK(all == expected);
    expected.clear();
    std::set_intersection(l.begin(), l.end(), r.begin(), r.end(),
                          std::back_inserter(expected), ZyxLess{});
    CHECK(both == expected);
    expected.clear();
    std::set_difference(l.begin(), l.end(), r.begin(), r.end(),
                        std::back_inserter(expected), ZyxLess{});
    CHECK(only_left == expected);
    expected.clear();
    std::set_difference(r.begin(), r.end(), l.begin(), l.end(),
                        std::back_inserter(expected), ZyxLess{});
    CHECK(only_right == expected);

    // A refusal at the third visit ends the walk there.
    if (all.size() >= 3) {
      int calls = 0;
      const vkc::Status s = d::merge_by_coord(
          l.size(), [&](std::size_t i) -> const vr::Vec3i& { return l[i]; },
          r.size(), [&](std::size_t j) -> const vr::Vec3i& { return r[j]; },
          [&](std::size_t, std::size_t) {
            return ++calls == 3 ? vkc::Status::not_found("third")
                                : vkc::Status{};
          });
      CHECK(s.domain() == C::NotFound && s.message() == "third");
      CHECK(calls == 3);
    }
  }
  return 0;
}

// --- Refusals. --------------------------------------------------------------

// Every check a P-frame reader makes, each refusing one edit of a valid
// frame with its code; and merge_frame's, a new block on a reference block,
// whatever that block's mode.
int refusals_case() {
  const codec_frames::RefusalBase base = codec_frames::refusal_base();
  const codec_frames::InterCase& good = base.good;
  const vkc::Result<d::Frame> read =
      d::read_frame(good.bytes.data(), good.bytes.size(), good.max_blocks,
                    good.held_reference());
  CHECK(read.ok());
  // What the edits rely on: a removed block, M not a multiple of U, and the
  // new blocks over three segments.
  const std::uint32_t n = get_u32(good.bytes, 32);
  CHECK(read.value().modes.size() == 40 && read.value().coords.size() == 8);
  CHECK(n < 40 + 8);
  CHECK(std::count(read.value().modes.begin(), read.value().modes.end(),
                   d::BlockMode::kKeep) > 0);
  CHECK(std::count(read.value().modes.begin(), read.value().modes.end(),
                   d::BlockMode::kSkip) > 0);

  std::set<int> rows;
  int last_row = 0;
  for (const codec_frames::InterRefusal& r :
       codec_frames::inter_refusals(base)) {
    const codec_frames::InterCase& c = r.edited;
    const vkc::Result<d::Frame> got = d::read_frame(
        c.bytes.data(), c.bytes.size(), c.max_blocks, c.held_reference());
    const bool as_expected = !got.ok() && got.status().domain() == r.code;
    if (!as_expected) {
      std::fprintf(stderr, "row %d, %s: %s\n", r.row, r.what,
                   got.ok() ? "read" : got.status().message().c_str());
    }
    CHECK(as_expected);
    CHECK(r.row >= last_row);
    last_row = r.row;
    rows.insert(r.row);
    // The order across the reader's stages: the format before the held
    // reference (a corrupt frame is InvalidArgument, a motion frame
    // Unsupported, whatever is held), the reference before the coding
    // parameters, those before the caller's room, and that before decoding.
    codec_frames::InterCase later = c;
    C code = r.code;
    if (r.row < 8) {
      later.held = false;
    } else if (r.row == 9) {
      later.reference.hash ^= 1;
      code = C::NotFound;
    } else if (r.row == 10) {
      later.reference.header.params.quantization_scale *= 2.0f;
      code = C::InvalidArgument;
    } else if (r.row > 10) {
      later.max_blocks = 0;
      code = C::OutOfMemory;
    }
    if (r.row != 8) {
      const vkc::Result<d::Frame> both =
          d::read_frame(later.bytes.data(), later.bytes.size(),
                        later.max_blocks, later.held_reference());
      if (both.ok() || both.status().domain() != code) {
        std::fprintf(stderr, "row %d, %s, with a later refusal: %s\n", r.row,
                     r.what,
                     both.ok() ? "read" : both.status().message().c_str());
        return 1;
      }
    }
  }
  CHECK(rows.size() == 12 && *rows.begin() == 1 && *rows.rbegin() == 12);

  // Row 13: a new block on a reference block, removed or skipped. The
  // writer cannot see it, and the reader's checks pass; the merge refuses
  // it.
  const d::Frame& reference = base.reference;
  const d::Frame p = d::read_frame(good.bytes.data(), good.bytes.size(),
                                   good.max_blocks, &good.reference)
                         .value();
  for (const d::BlockMode mode : {d::BlockMode::kRemove, d::BlockMode::kSkip}) {
    const auto at = std::find(p.modes.begin(), p.modes.end(), mode);
    CHECK(at != p.modes.end());
    const auto e = static_cast<std::size_t>(at - p.modes.begin());
    d::Frame twice = p;
    const vr::Vec3i c = reference.coords[e];
    const auto q = static_cast<std::size_t>(
        std::lower_bound(twice.coords.begin(), twice.coords.end(), c,
                         ZyxLess{}) -
        twice.coords.begin());
    twice.coords.insert(twice.coords.begin() + std::ptrdiff_t(q), c);
    const std::size_t position = p.modes.size() + q;
    const std::uint32_t k = p.blocks.params.coefficient_count;
    twice.blocks.masks.insert(
        twice.blocks.masks.begin() +
            std::ptrdiff_t(position * codec::kMaskWordsPerBlock),
        codec::kMaskWordsPerBlock, ~0u);
    twice.blocks.coefficients.insert(
        twice.blocks.coefficients.begin() + std::ptrdiff_t(position * k), k,
        std::int16_t(1));
    const Bytes b = d::write_frame(twice).value();
    const vkc::Result<d::Frame> back =
        d::read_frame(b.data(), b.size(), 1u << 20, &good.reference);
    CHECK(back.ok());
    const vkc::Result<d::MergedFrame> merged = d::merge_frame(
        reference.coords, back.value().modes, back.value().coords);
    CHECK(!merged.ok() && merged.status().domain() == C::InvalidArgument);
  }
  // A mode count other than the reference's, and the motion mode.
  std::vector<d::BlockMode> modes = p.modes;
  modes.pop_back();
  CHECK(d::merge_frame(reference.coords, modes, p.coords).status().domain() ==
        C::InvalidArgument);
  modes = p.modes;
  modes[0] = d::BlockMode::kMotion;
  CHECK(d::merge_frame(reference.coords, modes, p.coords).status().domain() ==
        C::Unsupported);
  // check_survivors on its own.
  CHECK(d::check_survivors(p.modes, n, 8).ok());
  CHECK(d::check_survivors(p.modes, n + 1, 8).domain() == C::InvalidArgument);
  CHECK(d::check_survivors(p.modes, n, 7).domain() == C::InvalidArgument);
  return 0;
}

// Every refusal write_frame and assemble_frame make of a P-frame.
int writer_refusals_case() {
  const d::Frame reference = make_frame(30, 8, 51);
  const d::FrameReference id = reference_of(d::write_frame(reference).value());
  const d::Frame target = successor(reference, 52, 150, 400, 5);
  const d::Frame good = inter_frame(id, reference, target);
  const Bytes bytes = d::write_frame(good).value();
  auto refused = [](const d::Frame& f, C code,
                    const d::FrameWriteOptions& o = {}) {
    const vkc::Result<Bytes> r = d::write_frame(f, o);
    return !r.ok() && r.status().domain() == code;
  };
  auto first = [&](d::BlockMode mode) {
    return static_cast<std::size_t>(
        std::find(good.modes.begin(), good.modes.end(), mode) -
        good.modes.begin());
  };
  const std::size_t keep = first(d::BlockMode::kKeep);
  const std::size_t skip = first(d::BlockMode::kSkip);
  const std::size_t removed = first(d::BlockMode::kRemove);
  CHECK(keep < good.modes.size() && skip < good.modes.size() &&
        removed < good.modes.size());
  const std::uint32_t k = 8;
  d::Frame f;

  f = good;
  f.type = static_cast<d::FrameType>(2);
  CHECK(refused(f, C::Unsupported));
  f = good;
  f.modes[keep] = d::BlockMode::kMotion;
  CHECK(refused(f, C::Unsupported));
  f = good;
  f.modes[keep] = static_cast<d::BlockMode>(5);
  CHECK(refused(f, C::InvalidArgument));
  f = good;
  f.modes.pop_back();
  CHECK(refused(f, C::InvalidArgument));
  f = good;
  f.modes.push_back(d::BlockMode::kSkip);
  CHECK(refused(f, C::InvalidArgument));
  CHECK(refused(good, C::InvalidArgument, sizes(64, 0)));
  CHECK(refused(good, C::InvalidArgument, sizes(0, 64)));
  f = good;
  f.blocks.coefficients.pop_back();
  CHECK(refused(f, C::InvalidArgument));
  f = good;
  f.blocks.masks.push_back(0);
  CHECK(refused(f, C::InvalidArgument));
  f = good;
  std::swap(f.coords[0], f.coords[1]);
  CHECK(refused(f, C::InvalidArgument));
  f = good;
  f.coords[1] = f.coords[0];
  CHECK(refused(f, C::InvalidArgument));
  // A written value outside +-32767: a keep residual of -32768 is refused,
  // while a skipped or removed entry's ignored payload is not written.
  f = good;
  f.blocks.coefficients[keep * k] = std::numeric_limits<std::int16_t>::min();
  CHECK(refused(f, C::InvalidArgument));
  f = good;
  f.blocks.coefficients[skip * k] = std::numeric_limits<std::int16_t>::min();
  f.blocks.masks[removed * codec::kMaskWordsPerBlock] = 7;
  CHECK(d::write_frame(f).value() == bytes);
  f = good;
  f.blocks.coefficients.back() = std::numeric_limits<std::int16_t>::min();
  CHECK(refused(f, C::InvalidArgument));  // a new block's
  // M + A past 2^32 - 1, refused before the mode count would be.
  f = good;
  f.reference.header.block_count = 0xFFFFFFFFu;
  CHECK(refused(f, C::InvalidArgument));
  // Coding parameters other than the reference's.
  for (int which = 0; which < 5; ++which) {
    f = good;
    d::FrameHeader& h = f.reference.header;
    if (which == 0) h.voxel_size *= 2.0f;
    if (which == 1) h.trunc_dist *= 2.0f;
    if (which == 2) h.params.coefficient_count = 9;
    if (which == 3) h.params.quantization_scale *= 2.0f;
    if (which == 4) h.params.quantization_weights[0] *= 2.0f;
    CHECK(refused(f, C::InvalidArgument));
  }
  // A weight beyond K is not a coding parameter.
  f = good;
  f.reference.header.params.quantization_weights[511] = 3.0f;
  CHECK(d::write_frame(f).value() == bytes);

  // assemble_frame, from the frame's own parsed sections.
  const d::ParsedFrame parsed =
      d::parse_frame(bytes.data(), bytes.size(), 1u << 20, &id).value();
  d::CodedFrame coded;
  coded.voxel_size = parsed.header.voxel_size;
  coded.trunc_dist = parsed.header.trunc_dist;
  coded.params = parsed.header.params;
  coded.block_count = parsed.header.block_count;
  coded.segment_size = parsed.header.segment_size;
  coded.reference = parsed.reference;
  coded.update_size = parsed.update_size;
  coded.new_count = parsed.new_count;
  coded.tables = parsed.tables;
  coded.segment_lengths = parsed.segment_lengths;
  coded.payload.assign(parsed.payload, parsed.payload + parsed.payload_size);
  CHECK(d::assemble_frame(d::FrameType::kInter, coded).value() == bytes);
  auto assemble_refused = [](d::FrameType type, const d::CodedFrame& c,
                             C code) {
    const vkc::Result<Bytes> r = d::assemble_frame(type, c);
    return !r.ok() && r.status().domain() == code;
  };
  CHECK(assemble_refused(static_cast<d::FrameType>(2), coded, C::Unsupported));
  CHECK(assemble_refused(d::FrameType::kIntra, coded, C::InvalidArgument));
  d::CodedFrame c = coded;
  c.tables.pop_back();  // an intra frame's tables
  CHECK(assemble_refused(d::FrameType::kInter, c, C::InvalidArgument));
  c = coded;
  c.segment_lengths.pop_back();
  CHECK(assemble_refused(d::FrameType::kInter, c, C::InvalidArgument));
  c = coded;
  c.update_size = 0;
  CHECK(assemble_refused(d::FrameType::kInter, c, C::InvalidArgument));
  c = coded;
  c.block_count = c.new_count - 1;
  CHECK(assemble_refused(d::FrameType::kInter, c, C::InvalidArgument));
  c = coded;
  c.block_count = c.reference.header.block_count + c.new_count + 1;
  CHECK(assemble_refused(d::FrameType::kInter, c, C::InvalidArgument));
  c = coded;
  c.reference.header.block_count = 0xFFFFFFFFu;
  CHECK(assemble_refused(d::FrameType::kInter, c, C::InvalidArgument));
  c = coded;
  c.params.quantization_scale *= 2.0f;
  CHECK(assemble_refused(d::FrameType::kInter, c, C::InvalidArgument));
  // More than 2^30 - 1 segments: 2^31 reference entries, one an update
  // segment, with nothing allocated for them.
  c = coded;
  c.reference.header.block_count = 1u << 31;
  c.update_size = 1;
  c.new_count = 0;
  c.block_count = 0;
  CHECK(assemble_refused(d::FrameType::kInter, c, C::InvalidArgument));
  return 0;
}

// --- Corruption. ------------------------------------------------------------

// Every truncation of a P-frame fails cleanly, and random corruption never
// gets the reader out of bounds -- what the sanitizer job checks.
int corruption_case() {
  const d::Frame reference = make_frame(300, 32, 13);
  d::FrameWriteOptions sixteen;
  sixteen.segment_size = 16;
  const d::FrameReference id =
      reference_of(d::write_frame(reference, sixteen).value());
  const d::Frame target = successor(reference, 14, 60, 500, 30);
  const Bytes good =
      d::write_frame(inter_frame(id, reference, target), sizes(16, 16)).value();
  CHECK(d::read_frame(good.data(), good.size(), 1u << 20, &id).ok());
  for (std::size_t len = 0; len < good.size(); ++len) {
    CHECK(!d::read_frame(good.data(), len, 1u << 20, &id).ok());
  }
  Lcg rng{17};
  int refused = 0;
  const int trials = 3000;
  for (int i = 0; i < trials; ++i) {
    Bytes b = good;
    const std::uint32_t edits = 1 + rng.below(3);
    for (std::uint32_t e = 0; e < edits; ++e) {
      b[rng.below(std::uint32_t(b.size()))] ^=
          static_cast<std::uint8_t>(1 + rng.below(255));
    }
    refused += d::read_frame(b.data(), b.size(), 1u << 20, &id).ok() ? 0 : 1;
  }
  std::printf("P-frame corruption: %d of %d refused\n", refused, trials);
  // Most edits land in the payload, where the segments' end checks catch
  // them; one in a raw field or a header float can leave a valid frame.
  CHECK(refused >= trials * 9 / 10);
  return 0;
}

}  // namespace

int main() {
  if (hash_case() != 0) return 1;
  if (worked_example_case() != 0) return 1;
  if (round_trip_case() != 0) return 1;
  if (chain_case() != 0) return 1;
  if (merge_by_coord_case() != 0) return 1;
  if (refusals_case() != 0) return 1;
  if (writer_refusals_case() != 0) return 1;
  if (corruption_case() != 0) return 1;
  std::printf("codec inter bitstream: OK\n");
  return 0;
}
