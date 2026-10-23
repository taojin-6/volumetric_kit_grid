// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// Host-only tests for P-frames in the v3 frame (bitstream.hpp, type 1),
// through the CPU oracle (codec_oracle.hpp) that pins its bytes: the
// reference hash's known answers; the worked example, built by hand, read and
// written byte for byte; round trips of every mode across K, update and
// segment sizes and the empty cases; the golden P-frames' pinned bytes;
// I -> P -> P chains on real hashes against
// an independent model (codec_frames.hpp); every reader refusal, each one
// edit from a valid P-frame, with its code and message, in the order of the
// reader's stages; every writer refusal, with its code and message; and
// truncation and random corruption, which must fail cleanly and never read
// out of bounds. The decoder's merge is codec_block_lists_test's.
// CPU-only, so these always run.

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "bitstream.hpp"
#include "codec_frames.hpp"
#include "codec_oracle.hpp"
#include "codec_oracle_rans.hpp"
#include "fnv1a.hpp"
#include "frame_hash.hpp"
#include "test_check.hpp"
#include "volumetric_kit/recon/codec/codec_params.hpp"

namespace vr = volumetric_kit::recon;
namespace vkc = volumetric_kit::core;
namespace codec = volumetric_kit::recon::codec;
namespace d = volumetric_kit::recon::codec::detail;
namespace oracle = codec_oracle;

namespace {

using codec_frames::get_u32;
using codec_frames::inter_frame;
using codec_frames::Lcg;
using codec_frames::make_frame;
using codec_frames::put_u32;
using codec_frames::reference_of;
using codec_frames::resolve;
using codec_frames::same_bits;
using codec_frames::successor;
using codec_frames::ZyxLess;
using C = vkc::Status::Code;
using Bytes = std::vector<std::uint8_t>;

// The same content, parameters included: what an intra frame holds.
bool same_content(const oracle::Frame& a, const oracle::Frame& b) {
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
bool same_inter(const oracle::Frame& a, const oracle::Frame& b) {
  return a.type == d::FrameType::kInter && b.type == d::FrameType::kInter &&
         a.reference.hash == b.reference.hash &&
         a.reference.header.block_count == b.reference.header.block_count &&
         a.modes == b.modes && same_content(a, b);
}

// A refusal with @p code whose message holds @p says: the check under test,
// not a later one with the same code. Says what it got otherwise.
bool refusal_is(const vkc::Status& s, C code, const std::string& says,
                const char* what) {
  if (!s.ok() && s.domain() == code &&
      s.message().find(says) != std::string::npos) {
    return true;
  }
  std::fprintf(stderr, "%s: %s\n", what,
               s.ok() ? "not refused" : s.message().c_str());
  return false;
}

oracle::FrameWriteOptions sizes(std::uint32_t r, std::uint32_t u) {
  oracle::FrameWriteOptions o;
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

// The worked example of the format: 161 bytes, read as the format says and
// written back byte for byte.
int worked_example_case() {
  const oracle::Frame reference = codec_frames::example_reference();
  const d::FrameReference id =
      reference_of(oracle::write_frame(reference).value());
  const Bytes bytes = codec_frames::example_bytes(id.hash);
  CHECK(bytes.size() == 161);

  const vkc::Result<oracle::Frame> read =
      oracle::read_frame(bytes.data(), bytes.size(), 3, &id);
  CHECK(read.ok());
  const oracle::Frame& p = read.value();
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

  oracle::Frame written = p;
  CHECK(oracle::write_frame(written).value() == bytes);

  // q = (q_a + 1, 3, q_b), masks (mask_a, full, mask_b).
  const oracle::Frame resolved = resolve(reference, p);
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
    oracle::RansWriter w;
    put(w);
    const std::size_t at = payload.size();
    CHECK(w.finish(payload));
    lengths.push_back(static_cast<std::uint32_t>(payload.size() - at));
    return 0;
  };
  CHECK(stream([&](oracle::RansWriter& w) {  // a: keep, unchanged, +1
          w.put(mode, 0);
          w.put(mask_class, 1);
          w.put(coefficient, 1);
          w.put_bits(0, 1);
        }) == 0);
  CHECK(stream([&](oracle::RansWriter& w) { w.put(mode, 1); }) ==
        0);                                  // b: skip
  CHECK(stream([&](oracle::RansWriter& w) {  // c: (2,0,0), full, +3
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
  CHECK(oracle::write_frame(written, sizes(64, 1)).value() == one_each);
  CHECK(oracle::read_frame(one_each.data(), one_each.size(), 3, &id).ok());

  // Had b been removed: mode 3, N = 2, the mode table remove for skip, and
  // the payload unchanged.
  Bytes removed = bytes;
  put_u32(removed, 32, 2);
  removed[130] = 0x02;  // the gap from keep to remove
  written.modes[1] = d::BlockMode::kRemove;
  CHECK(oracle::write_frame(written).value() == removed);
  const vkc::Result<oracle::Frame> read_removed =
      oracle::read_frame(removed.data(), removed.size(), 2, &id);
  CHECK(read_removed.ok());
  CHECK(read_removed.value().modes[1] == d::BlockMode::kRemove);
  return 0;
}

// --- Round trips. -----------------------------------------------------------

// Write @p p (coding @p target against @p reference), read it back against
// its reference, and check it is @p p, writes the same bytes again, holds
// @p target's blocks by the merge's model and resolves to @p target, whose
// intra bytes the resolved frame writes too.
int round_trip(const oracle::Frame& reference, const oracle::Frame& target,
               const oracle::Frame& p, const oracle::FrameWriteOptions& options,
               Bytes* out = nullptr) {
  const vkc::Result<Bytes> bytes = oracle::write_frame(p, options);
  if (!bytes.ok()) {
    std::fprintf(stderr, "write: %s\n", bytes.status().message().c_str());
  }
  CHECK(bytes.ok());
  const Bytes& b = bytes.value();
  const auto n = static_cast<std::uint32_t>(target.coords.size());
  CHECK(get_u32(b, 32) == n);
  const vkc::Result<oracle::Frame> back =
      oracle::read_frame(b.data(), b.size(), n, &p.reference);
  if (!back.ok()) {
    std::fprintf(stderr, "read: %s\n", back.status().message().c_str());
  }
  CHECK(back.ok());
  CHECK(same_inter(p, back.value()));
  CHECK(oracle::write_frame(back.value(), options).value() == b);

  CHECK(codec_frames::merge_oracle(reference.coords, back.value().modes,
                                   back.value().coords)
            .coords == target.coords);

  const oracle::Frame resolved = resolve(reference, back.value());
  CHECK(same_content(resolved, target));
  oracle::FrameWriteOptions intra;
  intra.segment_size = options.segment_size;
  CHECK(oracle::write_frame(resolved, intra).value() ==
        oracle::write_frame(target, intra).value());
  if (out != nullptr) *out = b;
  return 0;
}

// The matrix (codec_frames.hpp): every mode, K of 1, 64 and 512, and the
// empty cases, each at update and segment sizes of 1, 7, 64 and the
// longest, past the frame.
int round_trip_case() {
  std::uint64_t seen[5] = {};
  for (const codec_frames::InterCode& c : codec_frames::inter_matrix()) {
    for (const d::BlockMode mode : c.p.modes) ++seen[std::size_t(mode)];
    for (std::uint32_t u : codec_frames::kInterSizes) {
      for (std::uint32_t r : codec_frames::kInterSizes) {
        if (round_trip(c.reference, c.target, c.p, sizes(r, u)) != 0) {
          std::fprintf(stderr, "%s: U %u, R %u\n", c.what.c_str(), u, r);
          return 1;
        }
      }
    }
  }
  // The extremes code as they must: keep with residuals +-32767 and wrapped
  // to -2 and 2, and zero where the difference is +-32768.
  for (std::uint32_t k : {1u, 64u, codec::kVoxelsPerBlock}) {
    const codec_frames::Extremes x = codec_frames::extremes(k);
    const d::FrameReference id =
        reference_of(oracle::write_frame(x.reference).value());
    const oracle::Frame p = inter_frame(id, x.reference, x.target);
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

// The golden P-frames' bytes (codec_frames.hpp), which pin type 1 -- and,
// through each REFERENCE, frame_hash -- as the golden frames pin type 0: a
// writer and a reader changed together still round-trip, but no longer
// write these. Each row holds every mode and new blocks, and the second a
// partial last segment of each kind.
int golden_case() {
  for (const codec_frames::GoldenInterFrame& g :
       codec_frames::kGoldenInterFrames) {
    const codec_frames::InterCode c = codec_frames::golden_inter_code(g);
    Bytes bytes;
    CHECK(round_trip(c.reference, c.target, c.p,
                     sizes(g.segment_size, g.update_size), &bytes) == 0);
    const std::uint32_t content = codec_frames::content_hash(c.p);
    const std::uint32_t hash = vr_test::fnv1a(bytes);
    if (content != g.content || bytes.size() != g.size || hash != g.hash) {
      std::fprintf(stderr,
                   "%s, R %u, U %u: content 0x%08xu, %zu bytes, hash "
                   "0x%08xu\n",
                   c.what.c_str(), g.segment_size, g.update_size, content,
                   bytes.size(), hash);
    }
    CHECK(content == g.content);
    CHECK(bytes.size() == g.size);
    CHECK(hash == g.hash);
    std::size_t seen[5] = {};
    for (const d::BlockMode mode : c.p.modes) ++seen[std::size_t(mode)];
    for (std::size_t mode = 0; mode < 4; ++mode) CHECK(seen[mode] > 0);
    CHECK(c.p.coords.size() == g.added && g.added > 0);
  }
  const codec_frames::GoldenInterFrame& partial =
      codec_frames::kGoldenInterFrames[1];
  const codec_frames::InterCode c = codec_frames::golden_inter_code(partial);
  CHECK(c.p.modes.size() % partial.update_size != 0);
  CHECK(c.p.coords.size() % partial.segment_size != 0);
  return 0;
}

// I -> P -> P ..., each P-frame against the frame before on its real hash:
// it decodes to its target, as the oracle resolves it, and only against its
// own reference.
int chain_case() {
  oracle::Frame previous = make_frame(300, 64, 41);
  oracle::FrameWriteOptions intra;
  intra.segment_size = 16;
  d::FrameReference id =
      reference_of(oracle::write_frame(previous, intra).value());
  d::FrameReference stale;
  bool have_stale = false;
  const std::uint32_t update_sizes[] = {64, 1, 7, 300, 16, 64};
  for (std::uint32_t step = 0; step < 6; ++step) {
    const oracle::Frame target =
        successor(previous, 100 + step, 20 + 20 * step, 300, 10 + 5 * step);
    const oracle::Frame p = inter_frame(id, previous, target);
    Bytes bytes;
    CHECK(round_trip(previous, target, p, sizes(16, update_sizes[step]),
                     &bytes) == 0);
    if (have_stale) {
      const vkc::Result<oracle::Frame> wrong =
          oracle::read_frame(bytes.data(), bytes.size(), 1u << 20, &stale);
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

// --- Refusals. --------------------------------------------------------------

// Every check a P-frame reader makes, each refusing one edit of a valid
// frame with its code and message; and row 13's words, for a new block on a
// reference block, which the reader's checks pass.
int refusals_case() {
  const codec_frames::RefusalBase base = codec_frames::refusal_base();
  const codec_frames::InterCase& good = base.good;
  const vkc::Result<oracle::Frame> read =
      oracle::read_frame(good.bytes.data(), good.bytes.size(), good.max_blocks,
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

  auto read_status = [](const codec_frames::InterCase& c) {
    return oracle::read_frame(c.bytes.data(), c.bytes.size(), c.max_blocks,
                              c.held_reference())
        .status();
  };
  std::set<int> rows;
  int last_row = 0;
  for (const codec_frames::InterRefusal& r :
       codec_frames::inter_refusals(base)) {
    const codec_frames::InterCase& c = r.edited;
    CHECK(refusal_is(read_status(c), r.code, r.says, r.what));
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
      const vkc::Status both = read_status(later);
      if (both.ok() || both.domain() != code) {
        std::fprintf(stderr, "row %d, %s, with a later refusal: %s\n", r.row,
                     r.what, both.ok() ? "read" : both.message().c_str());
        return 1;
      }
    }
    // REFERENCE's body before TABLES: a malformed table changes nothing.
    if (r.row == 4) {
      std::vector<codec_frames::Section> t = codec_frames::sections_of(c.bytes);
      for (codec_frames::Section& section : t) {
        if (section.id == std::uint16_t(d::SectionId::kTables)) {
          section.body = {0xFF};
        }
      }
      later = c;
      later.bytes = codec_frames::assemble(c.bytes, t);
      CHECK(refusal_is(read_status(later), r.code, r.says, r.what));
    }
  }
  CHECK(rows.size() == 12 && *rows.begin() == 1 && *rows.rbegin() == 12);

  // Row 13: a new block on a reference block, removed or skipped. The
  // writer cannot see it, and the reader's checks pass; the decoder's merge
  // finds it (codec_block_lists_test), and check_new_block words it.
  const oracle::Frame& reference = base.reference;
  const oracle::Frame p =
      oracle::read_frame(good.bytes.data(), good.bytes.size(), good.max_blocks,
                         &good.reference)
          .value();
  for (const d::BlockMode mode : {d::BlockMode::kRemove, d::BlockMode::kSkip}) {
    const auto at = std::find(p.modes.begin(), p.modes.end(), mode);
    CHECK(at != p.modes.end());
    const auto e = static_cast<std::size_t>(at - p.modes.begin());
    oracle::Frame twice = p;
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
    const Bytes b = oracle::write_frame(twice).value();
    CHECK(
        oracle::read_frame(b.data(), b.size(), 1u << 20, &good.reference).ok());
  }
  CHECK(refusal_is(d::check_new_block(3, 17), C::InvalidArgument,
                   "new block 3 is on reference block 17", "check_new_block"));
  // check_survivors and check_segment on their own.
  const auto survivors = static_cast<std::uint64_t>(std::count_if(
      p.modes.begin(), p.modes.end(),
      [](d::BlockMode mode) { return mode != d::BlockMode::kRemove; }));
  CHECK(d::check_survivors(survivors, n, 8).ok());
  CHECK(d::check_survivors(survivors, n + 1, 8).domain() == C::InvalidArgument);
  CHECK(d::check_survivors(survivors, n, 7).domain() == C::InvalidArgument);
  CHECK(d::check_segment(5, d::SegmentFault::kNone).ok());
  CHECK(refusal_is(
      d::check_segment(5, d::SegmentFault::kOutOfOrder), C::InvalidArgument,
      "segment 5 does not start after the one before ends", "check_segment"));
  return 0;
}

// Every refusal write_frame and assemble_frame make of a P-frame, each with
// its code and message.
int writer_refusals_case() {
  const oracle::Frame reference = make_frame(30, 8, 51);
  const d::FrameReference id =
      reference_of(oracle::write_frame(reference).value());
  const oracle::Frame target = successor(reference, 52, 150, 400, 5);
  const oracle::Frame good = inter_frame(id, reference, target);
  const Bytes bytes = oracle::write_frame(good).value();
  auto refused = [](const oracle::Frame& f, C code, const std::string& says,
                    const oracle::FrameWriteOptions& o = {}) {
    return refusal_is(oracle::write_frame(f, o).status(), code, says,
                      "write_frame");
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
  const std::string range = "outside +-32767";
  oracle::Frame f;

  f = good;
  f.type = static_cast<d::FrameType>(2);
  CHECK(refused(f, C::Unsupported, "frame type 2"));
  f = good;
  f.modes[keep] = d::BlockMode::kMotion;
  CHECK(refused(f, C::Unsupported, "the motion mode"));
  f = good;
  f.modes[keep] = static_cast<d::BlockMode>(5);
  CHECK(refused(f, C::InvalidArgument, "mode 5 is not a mode"));
  f = good;
  f.modes.pop_back();
  CHECK(refused(f, C::InvalidArgument, "modes, one per reference block"));
  f = good;
  f.modes.push_back(d::BlockMode::kSkip);
  CHECK(refused(f, C::InvalidArgument, "modes, one per reference block"));
  // Sizes from 1 to the longest a device invocation decodes.
  const std::uint32_t longest = codec::kMaxDeviceDecodeSegmentSize;
  CHECK(refused(good, C::InvalidArgument, "update_size 0 is not in",
                sizes(64, 0)));
  CHECK(refused(good, C::InvalidArgument, "segment_size 0 is not in",
                sizes(0, 64)));
  CHECK(refused(good, C::InvalidArgument,
                "update_size " + std::to_string(longest + 1),
                sizes(64, longest + 1)));
  CHECK(refused(good, C::InvalidArgument,
                "segment_size " + std::to_string(longest + 1),
                sizes(longest + 1, 64)));
  CHECK(oracle::write_frame(good, sizes(longest, longest)).ok());
  f = good;
  f.blocks.coefficients.pop_back();
  CHECK(refused(f, C::InvalidArgument, "coefficients, got"));
  f = good;
  f.blocks.masks.push_back(0);
  CHECK(refused(f, C::InvalidArgument, "mask words, got"));
  f = good;
  std::swap(f.coords[0], f.coords[1]);
  CHECK(refused(f, C::InvalidArgument, "not strictly increasing"));
  f = good;
  f.coords[1] = f.coords[0];
  CHECK(refused(f, C::InvalidArgument, "not strictly increasing"));
  // A written value outside +-32767: a keep residual of -32768 is refused,
  // while a skipped or removed entry's ignored payload is not written.
  f = good;
  f.blocks.coefficients[keep * k] = std::numeric_limits<std::int16_t>::min();
  CHECK(refused(f, C::InvalidArgument, range));
  f = good;
  f.blocks.coefficients[skip * k] = std::numeric_limits<std::int16_t>::min();
  f.blocks.masks[removed * codec::kMaskWordsPerBlock] = 7;
  CHECK(oracle::write_frame(f).value() == bytes);
  f = good;
  f.blocks.coefficients.back() = std::numeric_limits<std::int16_t>::min();
  CHECK(refused(f, C::InvalidArgument, range));  // a new block's
  f = good;
  f.reference.header.block_count = 0xFFFFFFFFu;
  CHECK(refused(f, C::InvalidArgument, "2^32 - 1 entries"));
  // Coding parameters other than the reference's.
  const char* const parameters[] = {"voxel_size", "trunc_dist",
                                    "coefficient count", "quantization scale",
                                    "kept quantization weights"};
  for (int which = 0; which < 5; ++which) {
    f = good;
    d::FrameHeader& h = f.reference.header;
    if (which == 0) h.voxel_size *= 2.0f;
    if (which == 1) h.trunc_dist *= 2.0f;
    if (which == 2) h.params.coefficient_count = 9;
    if (which == 3) h.params.quantization_scale *= 2.0f;
    if (which == 4) h.params.quantization_weights[0] *= 2.0f;
    CHECK(refused(f, C::InvalidArgument,
                  std::string(parameters[which]) + " is not its reference's"));
  }
  // A weight beyond K is not a coding parameter.
  f = good;
  f.reference.header.params.quantization_weights[511] = 3.0f;
  CHECK(oracle::write_frame(f).value() == bytes);

  // assemble_frame, from the frame's own parsed sections.
  const d::CodedFrame coded = codec_frames::coded_frame(
      d::parse_frame(bytes.data(), bytes.size(), 1u << 20, &id).value());
  CHECK(d::assemble_frame(d::FrameType::kInter, coded).value() == bytes);
  auto assemble_refused = [](d::FrameType type, const d::CodedFrame& c, C code,
                             const std::string& says) {
    return refusal_is(d::assemble_frame(type, c).status(), code, says,
                      "assemble_frame");
  };
  CHECK(assemble_refused(static_cast<d::FrameType>(2), coded, C::Unsupported,
                         "frame type 2"));
  CHECK(assemble_refused(d::FrameType::kIntra, coded, C::InvalidArgument,
                         "an intra frame has no reference"));
  const std::string disagree = "segment lengths disagree with the header";
  d::CodedFrame c = coded;
  c.tables.pop_back();  // an intra frame's tables
  CHECK(
      assemble_refused(d::FrameType::kInter, c, C::InvalidArgument, disagree));
  c = coded;
  c.segment_lengths.pop_back();
  CHECK(
      assemble_refused(d::FrameType::kInter, c, C::InvalidArgument, disagree));
  c = coded;
  c.update_size = 0;
  CHECK(assemble_refused(d::FrameType::kInter, c, C::InvalidArgument,
                         "update_size 0 is not in"));
  c = coded;
  c.update_size = longest + 1;
  CHECK(assemble_refused(d::FrameType::kInter, c, C::InvalidArgument,
                         "update_size " + std::to_string(longest + 1)));
  const std::string survivors = "not the survivors plus the new blocks";
  c = coded;
  c.block_count = c.new_count - 1;
  CHECK(
      assemble_refused(d::FrameType::kInter, c, C::InvalidArgument, survivors));
  c = coded;
  c.block_count = c.reference.header.block_count + c.new_count + 1;
  CHECK(
      assemble_refused(d::FrameType::kInter, c, C::InvalidArgument, survivors));
  c = coded;
  c.reference.header.block_count = 0xFFFFFFFFu;
  CHECK(assemble_refused(d::FrameType::kInter, c, C::InvalidArgument,
                         "2^32 - 1 entries"));
  c = coded;
  c.params.quantization_scale *= 2.0f;
  CHECK(assemble_refused(d::FrameType::kInter, c, C::InvalidArgument,
                         "quantization scale is not its reference's"));
  // More than 2^30 - 1 segments: 2^31 reference entries, one an update
  // segment, with nothing allocated for them.
  c = coded;
  c.reference.header.block_count = 1u << 31;
  c.update_size = 1;
  c.new_count = 0;
  c.block_count = 0;
  CHECK(assemble_refused(d::FrameType::kInter, c, C::InvalidArgument,
                         "2^30 - 1 segments"));
  return 0;
}

// --- Corruption. ------------------------------------------------------------

// Every truncation of a P-frame fails cleanly, and random corruption never
// gets the reader out of bounds -- what the sanitizer job checks.
int corruption_case() {
  const oracle::Frame reference = make_frame(300, 32, 13);
  oracle::FrameWriteOptions sixteen;
  sixteen.segment_size = 16;
  const d::FrameReference id =
      reference_of(oracle::write_frame(reference, sixteen).value());
  const oracle::Frame target = successor(reference, 14, 60, 500, 30);
  const Bytes good =
      oracle::write_frame(inter_frame(id, reference, target), sizes(16, 16))
          .value();
  CHECK(oracle::read_frame(good.data(), good.size(), 1u << 20, &id).ok());
  for (std::size_t len = 0; len < good.size(); ++len) {
    CHECK(!oracle::read_frame(good.data(), len, 1u << 20, &id).ok());
  }
  Lcg rng{17};
  int refused = 0;
  const int trials = 500;
  for (int i = 0; i < trials; ++i) {
    Bytes b = good;
    const std::uint32_t edits = 1 + rng.below(3);
    for (std::uint32_t e = 0; e < edits; ++e) {
      b[rng.below(std::uint32_t(b.size()))] ^=
          static_cast<std::uint8_t>(1 + rng.below(255));
    }
    refused +=
        oracle::read_frame(b.data(), b.size(), 1u << 20, &id).ok() ? 0 : 1;
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
  if (golden_case() != 0) return 1;
  if (chain_case() != 0) return 1;
  if (refusals_case() != 0) return 1;
  if (writer_refusals_case() != 0) return 1;
  if (corruption_case() != 0) return 1;
  std::printf("codec inter bitstream: OK\n");
  return 0;
}
