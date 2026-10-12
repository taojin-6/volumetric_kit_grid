// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

// Synthetic frames for the codec's frame tests: the host writer's round
// trips and the device writer's byte comparison draw the same frames, both
// writers must write the golden frames' pinned bytes, and P-frames come with
// an oracle of their own and the refusals each check makes.

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <map>
#include <set>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "bitstream.hpp"
#include "fnv1a.hpp"
#include "frame_hash.hpp"
#include "volumetric_kit/core/base/result.hpp"
#include "volumetric_kit/recon/codec/codec_params.hpp"

namespace codec_frames {

namespace vr = volumetric_kit::recon;
namespace vkc = volumetric_kit::core;
namespace codec = volumetric_kit::recon::codec;
namespace d = volumetric_kit::recon::codec::detail;

inline constexpr std::int32_t kMax32 = std::numeric_limits<std::int32_t>::max();
inline constexpr std::int32_t kMin32 = std::numeric_limits<std::int32_t>::min();

inline bool same_bits(float a, float b) {
  return std::memcmp(&a, &b, sizeof(a)) == 0;
}

struct Lcg {
  std::uint64_t state;
  std::uint32_t next() {
    state = state * 6364136223846793005ull + 1442695040888963407ull;
    return static_cast<std::uint32_t>(state >> 32);
  }
  std::uint32_t below(std::uint32_t n) { return next() % n; }
};

// A partial mask shaped like a fused band's edge: the voxels on one side of a
// plane through the block, so its planes and lines repeat, run full or empty,
// or cut across -- every symbol the mask code has.
inline void slab_mask(Lcg& rng, std::uint32_t* mask) {
  const int a = int(rng.below(7)) - 3;
  const int b = int(rng.below(7)) - 3;
  const int c = int(rng.below(7)) - 3;
  const int d = int(rng.below(40)) - 20;
  for (std::uint32_t w = 0; w < codec::kMaskWordsPerBlock; ++w) {
    mask[w] = 0;
  }
  for (std::uint32_t v = 0; v < codec::kVoxelsPerBlock; ++v) {
    const int x = int(v % 8), y = int(v / 8 % 8), z = int(v / 64);
    if (a * x + b * y + c * z < d) {
      mask[v / 32] |= 1u << (v % 32);
    }
  }
}

// A frame shaped like a real one: blocks in runs along x with jumps between
// them, masks mostly full with some empty and some partial -- noise, or a
// slab's edge -- coefficients small and sparser at higher indices, and the
// odd extreme value.
inline d::Frame make_frame(std::size_t n, std::uint32_t k, std::uint64_t seed) {
  Lcg rng{seed};
  auto less = [](const vr::Vec3i& a, const vr::Vec3i& b) {
    return d::coord_less(a, b);
  };
  std::set<vr::Vec3i, decltype(less)> coords(less);
  while (coords.size() < n) {
    // One draw per statement: the order a function's arguments are evaluated
    // in is unspecified (GCC and Clang disagree), and a fixture must be the
    // same frame on every compiler.
    const int sx = int(rng.below(200)) - 100;
    const int sy = int(rng.below(60)) - 30;
    const int sz = int(rng.below(60)) - 30;
    const vr::Vec3i start(sx, sy, sz);
    const std::uint32_t run = 1 + rng.below(12);
    for (std::uint32_t i = 0; i < run && coords.size() < n; ++i) {
      coords.insert(vr::Vec3i(start.x + int(i), start.y, start.z));
    }
  }
  d::Frame f;
  f.voxel_size = 0.005f;
  f.coords.assign(coords.begin(), coords.end());
  f.blocks.params.coefficient_count = k;
  f.blocks.trunc_dist = 0.04f;
  for (std::size_t i = 0; i < n; ++i) {
    const std::uint32_t cls = rng.below(10);
    std::uint32_t mask[codec::kMaskWordsPerBlock];
    if (cls < 8) {
      for (std::uint32_t& w : mask) {
        w = cls < 6 ? ~0u : cls < 7 ? 0u : rng.next();
      }
    } else {
      slab_mask(rng, mask);
    }
    f.blocks.masks.insert(f.blocks.masks.end(), mask,
                          mask + codec::kMaskWordsPerBlock);
    for (std::uint32_t j = 0; j < k; ++j) {
      std::int32_t v = 0;
      const std::uint32_t spread = 1 + 64 / (1 + j);
      if (rng.below(1 + j / 4) == 0) {
        v = int(rng.below(2 * spread + 1)) - int(spread);
      }
      if (rng.below(500) == 0) {
        v = rng.below(2) == 0 ? codec::kMaxQuantizedMagnitude
                              : -codec::kMaxQuantizedMagnitude;
      }
      f.blocks.coefficients.push_back(static_cast<std::int16_t>(v));
    }
  }
  return f;
}

// Coordinates at the ends of int32, and deltas of 2^32 - 1 on every axis.
inline d::Frame extreme_frame() {
  d::Frame f = make_frame(0, 4, 1);
  f.coords = {
      {kMin32, kMin32, kMin32}, {kMax32, kMin32, kMin32},  // dx = 2^32 - 1
      {kMin32, kMax32, kMin32},                            // dy, dx back
      {0, 0, kMax32},                                      // dz
      {1, 0, kMax32},           {kMax32, kMax32, kMax32},
  };
  f.blocks.masks.assign(f.coords.size() * codec::kMaskWordsPerBlock, ~0u);
  f.blocks.coefficients.assign(f.coords.size() * 4, 0);
  return f;
}

// What a frame holds, hashed: coordinates, masks and coefficients.
inline std::uint32_t content_hash(const d::Frame& f) {
  std::vector<std::uint32_t> words;
  for (const vr::Vec3i& c : f.coords) {
    words.insert(words.end(), {static_cast<std::uint32_t>(c.x),
                               static_cast<std::uint32_t>(c.y),
                               static_cast<std::uint32_t>(c.z)});
  }
  words.insert(words.end(), f.blocks.masks.begin(), f.blocks.masks.end());
  for (std::int16_t c : f.blocks.coefficients) {
    words.push_back(static_cast<std::uint16_t>(c));
  }
  return vr_test::fnv1a(words);
}

// --- Golden frames. ---------------------------------------------------------
//
// The v3 bytes every writer, host or device, writes for these frames on every
// machine: what pins the format, where a round trip would pass a writer and a
// reader changed together. `content` pins the fixture, so a failure says which
// changed. A new kFrameVersion brings new values; nothing else may.
static_assert(d::kFrameVersion == 3, "the golden frames are v3's");

struct GoldenFrame {
  bool extreme;  // extreme_frame(), else make_frame(blocks, k, seed)
  std::size_t blocks;
  std::uint32_t k;
  std::uint64_t seed;
  std::uint32_t segment_size;
  std::uint32_t content;  // content_hash of the frame
  std::size_t size;       // its bytes
  std::uint32_t hash;     // vr_test::fnv1a of them
};

// The default K and segment size over many segments, every K with segments
// of 16, an odd K with odd segments, and coordinates at the ends of int32,
// as steps of up to 2^32 - 1 and each in full.
inline constexpr GoldenFrame kGoldenFrames[] = {
    {false, 700, 64, 7, 64, 0x3f9773d8u, 17582, 0x411a37d6u},
    {false, 40, codec::kVoxelsPerBlock, 5, 16, 0xebbf784au, 6520, 0x8e023b27u},
    {false, 65, 21, 3, 7, 0x4fb1073du, 1907, 0x29533280u},
    {true, 6, 4, 0, 64, 0x11b607ddu, 198, 0xe081c9b3u},
    {true, 6, 4, 0, 1, 0x11b607ddu, 236, 0x92cc1c51u},
};

inline d::Frame golden_frame(const GoldenFrame& g) {
  return g.extreme ? extreme_frame() : make_frame(g.blocks, g.k, g.seed);
}

// --- Editing a valid frame. -------------------------------------------------

inline void put_u32(std::vector<std::uint8_t>& b, std::size_t at,
                    std::uint32_t v) {
  for (int i = 0; i < 4; ++i) {
    b[at + std::size_t(i)] = static_cast<std::uint8_t>(v >> (8 * i));
  }
}

inline void put_f32(std::vector<std::uint8_t>& b, std::size_t at, float f) {
  std::uint32_t v = 0;
  std::memcpy(&v, &f, sizeof(v));
  put_u32(b, at, v);
}

inline std::uint32_t get_u32(const std::vector<std::uint8_t>& b,
                             std::size_t at) {
  std::uint32_t v = 0;
  for (int i = 0; i < 4; ++i) {
    v |= std::uint32_t(b[at + std::size_t(i)]) << (8 * i);
  }
  return v;
}

struct Section {
  std::uint16_t id;
  std::uint16_t flags;
  std::vector<std::uint8_t> body;
};

// Split a frame into its fixed header and sections, and put one back
// together, so a test can drop, repeat, reorder or rewrite a section.
inline std::vector<Section> sections_of(const std::vector<std::uint8_t>& b) {
  std::vector<Section> out;
  const std::uint32_t count = get_u32(b, 40);
  const std::size_t header = d::frame_header_bytes(get_u32(b, 20));
  std::size_t body = header + count * d::kSectionEntryBytes;
  for (std::uint32_t i = 0; i < count; ++i) {
    const std::size_t e = header + i * d::kSectionEntryBytes;
    Section s;
    s.id = static_cast<std::uint16_t>(b[e] | (b[e + 1] << 8));
    s.flags = static_cast<std::uint16_t>(b[e + 2] | (b[e + 3] << 8));
    const std::uint32_t len = get_u32(b, e + 4);
    s.body.assign(b.begin() + std::ptrdiff_t(body),
                  b.begin() + std::ptrdiff_t(body + len));
    body += len;
    out.push_back(std::move(s));
  }
  return out;
}

inline std::vector<std::uint8_t> assemble(
    const std::vector<std::uint8_t>& frame,
    const std::vector<Section>& sections) {
  std::vector<std::uint8_t> b(
      frame.begin(), frame.begin() + std::ptrdiff_t(d::frame_header_bytes(
                                         get_u32(frame, 20))));
  put_u32(b, 40, static_cast<std::uint32_t>(sections.size()));
  for (const Section& s : sections) {
    b.push_back(static_cast<std::uint8_t>(s.id));
    b.push_back(static_cast<std::uint8_t>(s.id >> 8));
    b.push_back(static_cast<std::uint8_t>(s.flags));
    b.push_back(static_cast<std::uint8_t>(s.flags >> 8));
    for (int i = 0; i < 4; ++i) {
      b.push_back(static_cast<std::uint8_t>(s.body.size() >> (8 * i)));
    }
  }
  for (const Section& s : sections) {
    b.insert(b.end(), s.body.begin(), s.body.end());
  }
  return b;
}

// Rewrite the low 12 bits of the x of the first block of the block segment
// whose stream starts at @p at in PAYLOAD. A segment's first raw chunk is the
// low 12 bits of its stream's initial state word (bytes 2-3), and raw bits
// leave the state untouched, so this moves that one coordinate and nothing
// else -- the edit no end check can see, and so how a test reaches what only
// a flipped raw bit makes.
inline void set_first_x_low_bits(std::vector<std::uint8_t>& payload,
                                 std::size_t at, std::uint32_t low12) {
  payload[at + 2] = static_cast<std::uint8_t>(low12 & 0xFF);
  payload[at + 3] = static_cast<std::uint8_t>((payload[at + 3] & 0xF0) |
                                              ((low12 >> 8) & 0x0F));
}

// A SEGMENTS body with its @p count lengths from @p first replaced by their
// sum: the same PAYLOAD read as fewer segments.
inline std::vector<std::uint8_t> join_lengths(
    const std::vector<std::uint8_t>& segments, std::size_t first,
    std::size_t count) {
  std::uint32_t sum = 0;
  for (std::size_t i = first; i < first + count; ++i) {
    sum += get_u32(segments, 4 * i);
  }
  std::vector<std::uint8_t> out(
      segments.begin(), segments.begin() + std::ptrdiff_t(4 * first + 4));
  put_u32(out, 4 * first, sum);
  out.insert(out.end(), segments.begin() + std::ptrdiff_t(4 * (first + count)),
             segments.end());
  return out;
}

// What assemble_frame lays a parsed frame out from: every field it reads, so
// it writes the frame's bytes again.
inline d::CodedFrame coded_frame(const d::ParsedFrame& p) {
  d::CodedFrame c;
  c.voxel_size = p.header.voxel_size;
  c.trunc_dist = p.header.trunc_dist;
  c.params = p.header.params;
  c.block_count = p.header.block_count;
  c.segment_size = p.header.segment_size;
  c.reference = p.reference;
  c.update_size = p.update_size;
  c.new_count = p.new_count;
  c.tables = p.tables;
  c.segment_lengths = p.segment_lengths;
  c.payload.assign(p.payload, p.payload + p.payload_size);
  return c;
}

// --- P-frames. --------------------------------------------------------------
//
// An independent model of the type-1 rules in bitstream.hpp: block identity
// by a std::map of its own order, and residuals and masks in plain integer
// arithmetic, so a test that compares the library with it compares two
// implementations.

// The frame order, (z, y, x), written out rather than taken from the library.
struct ZyxLess {
  bool operator()(const vr::Vec3i& a, const vr::Vec3i& b) const {
    return std::tie(a.z, a.y, a.x) < std::tie(b.z, b.y, b.x);
  }
};

// One block's payload.
struct Payload {
  std::vector<std::uint32_t> mask;
  std::vector<std::int16_t> q;
  bool operator==(const Payload& o) const { return mask == o.mask && q == o.q; }
};

inline Payload payload_at(const d::DctBlocks& b, std::size_t i) {
  const std::uint32_t k = b.params.coefficient_count;
  const std::size_t w = codec::kMaskWordsPerBlock;
  return Payload{
      std::vector<std::uint32_t>(b.masks.begin() + std::ptrdiff_t(i * w),
                                 b.masks.begin() + std::ptrdiff_t(i * w + w)),
      std::vector<std::int16_t>(
          b.coefficients.begin() + std::ptrdiff_t(i * k),
          b.coefficients.begin() + std::ptrdiff_t(i * k + k))};
}

inline void append_payload(d::DctBlocks& b, const Payload& p) {
  b.masks.insert(b.masks.end(), p.mask.begin(), p.mask.end());
  b.coefficients.insert(b.coefficients.end(), p.q.begin(), p.q.end());
}

// (t - r) mod 2^16, as int16.
inline std::int16_t residual(std::int16_t t, std::int16_t r) {
  return static_cast<std::int16_t>(static_cast<std::uint16_t>(
      static_cast<std::uint16_t>(t) - static_cast<std::uint16_t>(r)));
}

// (r + x) mod 2^16, as int16.
inline std::int16_t apply_residual(std::int16_t r, std::int16_t x) {
  return static_cast<std::int16_t>(static_cast<std::uint16_t>(
      static_cast<std::uint16_t>(r) + static_cast<std::uint16_t>(x)));
}

// What a reader holds of a frame it decoded: its hash and header.
inline d::FrameReference reference_of(const std::vector<std::uint8_t>& bytes) {
  d::FrameReference r;
  r.hash = d::frame_hash(bytes.data(), bytes.size());
  r.header = d::read_frame_header(bytes.data(), bytes.size()).value();
  return r;
}

// A frame after @p reference: of every 1000 reference blocks about
// @p removed are gone, @p changed change (half of them by a little, the rest
// anywhere, masks included) and the rest stay as they are; then @p added
// blocks that the reference lacks. The parameters are the reference's.
inline d::Frame successor(const d::Frame& reference, std::uint64_t seed,
                          std::uint32_t removed, std::uint32_t changed,
                          std::size_t added) {
  Lcg rng{seed};
  const std::uint32_t k = reference.blocks.params.coefficient_count;
  std::map<vr::Vec3i, Payload, ZyxLess> blocks;
  for (std::size_t e = 0; e < reference.coords.size(); ++e) {
    const std::uint32_t roll = rng.below(1000);
    if (roll < removed) continue;
    Payload p = payload_at(reference.blocks, e);
    if (roll < removed + changed) {
      const bool small = rng.below(2) == 0;
      for (std::uint32_t j = 0; j < k; ++j) {
        if (small) {
          const int v = p.q[j] + int(rng.below(5)) - 2;
          if (v >= -codec::kMaxQuantizedMagnitude &&
              v <= codec::kMaxQuantizedMagnitude) {
            p.q[j] = static_cast<std::int16_t>(v);
          }
        } else {
          p.q[j] = static_cast<std::int16_t>(int(rng.below(201)) - 100);
        }
      }
      const std::uint32_t mask_roll = rng.below(4);
      if (mask_roll == 0) {
        const std::uint32_t w = rng.below(codec::kMaskWordsPerBlock);
        p.mask[w] ^= rng.next() | 1u;
      } else if (mask_roll == 1) {
        slab_mask(rng, p.mask.data());
      }
    }
    blocks.emplace(reference.coords[e], std::move(p));
  }
  const d::Frame fresh = make_frame(added, k, seed ^ 0x5eedu);
  std::set<vr::Vec3i, ZyxLess> taken(reference.coords.begin(),
                                     reference.coords.end());
  for (std::size_t i = 0; i < fresh.coords.size(); ++i) {
    vr::Vec3i c = fresh.coords[i];
    while (taken.count(c) != 0) c.z += 1000;  // away from the reference
    taken.insert(c);
    blocks.emplace(c, payload_at(fresh.blocks, i));
  }
  d::Frame f;
  f.voxel_size = reference.voxel_size;
  f.blocks.params = reference.blocks.params;
  f.blocks.trunc_dist = reference.blocks.trunc_dist;
  for (const auto& [c, p] : blocks) {
    f.coords.push_back(c);
    append_payload(f.blocks, p);
  }
  return f;
}

// The P-frame that codes @p target against @p reference, whose bytes
// @p id names: a reference block @p target lacks is removed, one it holds
// unchanged is skipped, one whose difference is +-32768 somewhere (the one
// residual a class cannot carry) is zero, and any other is keep; then the
// blocks @p reference lacks, new. @p overrides sets a surviving reference
// block's mode, written as given.
inline d::Frame inter_frame(
    const d::FrameReference& id, const d::Frame& reference,
    const d::Frame& target,
    const std::map<std::size_t, d::BlockMode>& overrides = {}) {
  const std::uint32_t k = reference.blocks.params.coefficient_count;
  std::map<vr::Vec3i, std::size_t, ZyxLess> in_target;
  for (std::size_t i = 0; i < target.coords.size(); ++i) {
    in_target.emplace(target.coords[i], i);
  }
  const std::set<vr::Vec3i, ZyxLess> in_reference(reference.coords.begin(),
                                                  reference.coords.end());
  d::Frame p;
  p.type = d::FrameType::kInter;
  p.voxel_size = target.voxel_size;
  p.reference = id;
  p.blocks.params = target.blocks.params;
  p.blocks.trunc_dist = target.blocks.trunc_dist;
  const Payload none{std::vector<std::uint32_t>(codec::kMaskWordsPerBlock, 0),
                     std::vector<std::int16_t>(k, 0)};
  for (std::size_t e = 0; e < reference.coords.size(); ++e) {
    const auto it = in_target.find(reference.coords[e]);
    if (it == in_target.end()) {
      p.modes.push_back(d::BlockMode::kRemove);
      append_payload(p.blocks, none);
      continue;
    }
    const Payload r = payload_at(reference.blocks, e);
    const Payload t = payload_at(target.blocks, it->second);
    Payload x = t;
    bool wraps = false;
    for (std::uint32_t j = 0; j < k; ++j) {
      x.q[j] = residual(t.q[j], r.q[j]);
      wraps = wraps || x.q[j] == std::numeric_limits<std::int16_t>::min();
    }
    for (std::size_t w = 0; w < x.mask.size(); ++w) x.mask[w] ^= r.mask[w];
    d::BlockMode mode = t == r  ? d::BlockMode::kSkip
                        : wraps ? d::BlockMode::kZero
                                : d::BlockMode::kKeep;
    if (const auto o = overrides.find(e); o != overrides.end()) {
      mode = o->second;
    }
    p.modes.push_back(mode);
    append_payload(p.blocks, mode == d::BlockMode::kKeep   ? x
                             : mode == d::BlockMode::kZero ? t
                                                           : none);
  }
  for (std::size_t i = 0; i < target.coords.size(); ++i) {
    if (in_reference.count(target.coords[i]) != 0) continue;
    p.coords.push_back(target.coords[i]);
    append_payload(p.blocks, payload_at(target.blocks, i));
  }
  return p;
}

// The intra frame a decoded P-frame @p p makes of @p reference's blocks
// (refusing nothing: a new block at a reference coordinate replaces it).
inline d::Frame resolve(const d::Frame& reference, const d::Frame& p) {
  const std::size_t m = reference.coords.size();
  std::map<vr::Vec3i, Payload, ZyxLess> blocks;
  for (std::size_t e = 0; e < m; ++e) {
    const Payload r = payload_at(reference.blocks, e);
    const Payload x = payload_at(p.blocks, e);
    switch (p.modes[e]) {
      case d::BlockMode::kKeep: {
        Payload t = r;
        for (std::size_t j = 0; j < t.q.size(); ++j) {
          t.q[j] = apply_residual(r.q[j], x.q[j]);
        }
        for (std::size_t w = 0; w < t.mask.size(); ++w) t.mask[w] ^= x.mask[w];
        blocks[reference.coords[e]] = t;
        break;
      }
      case d::BlockMode::kSkip:
        blocks[reference.coords[e]] = r;
        break;
      case d::BlockMode::kZero:
        blocks[reference.coords[e]] = x;
        break;
      default:
        break;
    }
  }
  for (std::size_t q = 0; q < p.coords.size(); ++q) {
    blocks[p.coords[q]] = payload_at(p.blocks, m + q);
  }
  d::Frame f;
  f.voxel_size = p.voxel_size;
  f.blocks.params = p.blocks.params;
  f.blocks.trunc_dist = p.blocks.trunc_dist;
  for (const auto& [c, payload] : blocks) {
    f.coords.push_back(c);
    append_payload(f.blocks, payload);
  }
  return f;
}

// A P-frame's blocks in frame order with their sources, by the oracle's own
// map: what merge_frame must give.
inline d::MergedFrame merge_oracle(const std::vector<vr::Vec3i>& reference,
                                   const std::vector<d::BlockMode>& modes,
                                   const std::vector<vr::Vec3i>& added) {
  std::map<vr::Vec3i, d::ApplySource, ZyxLess> blocks;
  for (std::size_t e = 0; e < reference.size(); ++e) {
    const auto i = static_cast<std::uint32_t>(e);
    if (modes[e] == d::BlockMode::kKeep) blocks[reference[e]] = {i, i};
    if (modes[e] == d::BlockMode::kSkip) {
      blocks[reference[e]] = {i, d::kNoSource};
    }
    if (modes[e] == d::BlockMode::kZero) {
      blocks[reference[e]] = {d::kNoSource, i};
    }
  }
  for (std::size_t q = 0; q < added.size(); ++q) {
    blocks[added[q]] = {d::kNoSource,
                        static_cast<std::uint32_t>(reference.size() + q)};
  }
  d::MergedFrame out;
  for (const auto& [c, source] : blocks) {
    out.coords.push_back(c);
    out.sources.push_back(source);
  }
  return out;
}

// --- The P-frame refusals. --------------------------------------------------

// A P-frame and what its reader holds.
struct InterCase {
  std::vector<std::uint8_t> bytes;
  bool held = true;  // whether the reader holds `reference`
  d::FrameReference reference;
  std::uint32_t max_blocks = 1u << 20;
  const d::FrameReference* held_reference() const {
    return held ? &reference : nullptr;
  }
};

// The P-frame every refusal below edits, and what it was made from. Built so
// each check is one edit away: a removed block (so the block count can rise
// past the survivors), M = 40 not a multiple of U = 7 (so M can move without
// moving the update segments), and new blocks in segments of 3 -- a run along
// x split across the first two, then two blocks just short of INT32_MAX in x
// in the last, in which one raw edit steps out of int32.
struct RefusalBase {
  d::Frame reference;  // its intra content
  d::Frame target;
  InterCase good;
};

inline RefusalBase refusal_base() {
  RefusalBase base;
  base.reference = make_frame(40, 8, 77);
  const std::vector<std::uint8_t> reference_bytes =
      d::write_frame(base.reference).value();
  d::Frame target = successor(base.reference, 78, 100, 400, 0);
  std::map<vr::Vec3i, Payload, ZyxLess> blocks;
  for (std::size_t i = 0; i < target.coords.size(); ++i) {
    blocks.emplace(target.coords[i], payload_at(target.blocks, i));
  }
  const d::Frame extra = make_frame(8, 8, 79);
  const vr::Vec3i added[] = {{500, 0, 0},
                             {501, 0, 0},
                             {502, 0, 0},
                             {503, 0, 0},
                             {504, 0, 0},
                             {505, 0, 0},
                             {0x7FFFF000, 0, kMax32},
                             {0x7FFFF001, 0, kMax32}};
  for (std::size_t i = 0; i < 8; ++i) {
    blocks.emplace(added[i], payload_at(extra.blocks, i));
  }
  base.target = make_frame(0, 8, 1);
  base.target.voxel_size = target.voxel_size;
  for (const auto& [c, p] : blocks) {
    base.target.coords.push_back(c);
    append_payload(base.target.blocks, p);
  }
  base.good.reference = reference_of(reference_bytes);
  d::FrameWriteOptions options;
  options.segment_size = 3;
  options.update_size = 7;
  base.good.bytes = d::write_frame(inter_frame(base.good.reference,
                                               base.reference, base.target),
                                   options)
                        .value();
  return base;
}

struct InterRefusal {
  const char* what;
  int row;  // in bitstream.hpp's refusal order
  vkc::Status::Code code;
  std::string says;  // a part of the message only the check under test makes
  InterCase edited;
};

// The refusal of each check a P-frame reader makes, each one edit from
// refusal_base()'s frame, in the order the reader makes them; merge_frame's,
// which needs the reference's coordinates, is the test's. An edit that a
// later check would refuse too is matched by its message.
inline std::vector<InterRefusal> inter_refusals(const RefusalBase& base) {
  using C = vkc::Status::Code;
  const InterCase& good = base.good;
  const std::vector<std::uint8_t>& b = good.bytes;
  const std::size_t header = d::frame_header_bytes(get_u32(b, 20));
  const std::vector<Section> s = sections_of(b);  // REFERENCE first
  const std::uint32_t n = get_u32(b, 32);
  const std::uint32_t m = get_u32(s[0].body, 8);
  const std::uint32_t u = get_u32(s[0].body, 12);
  const std::uint32_t a = get_u32(s[0].body, 16);
  const std::uint64_t updates = d::frame_segment_count(m, u);
  const std::uint64_t news = d::frame_segment_count(a, get_u32(b, 36));
  const std::string mode_table_name =
      "table " +
      std::to_string(
          d::frame_model_count(d::FrameType::kInter, get_u32(b, 20)) - 1);
  // Where segment i's stream starts in PAYLOAD.
  auto stream_at = [&](std::uint64_t i) {
    std::size_t at = 0;
    for (std::uint64_t j = 0; j < i; ++j) at += get_u32(s[2].body, 4 * j);
    return at;
  };
  auto segment = [](std::uint64_t i, const char* then) {
    return "segment " + std::to_string(i) + then;
  };
  std::vector<InterRefusal> out;
  auto bytes = [&](const char* what, int row, C code, std::string says,
                   std::vector<std::uint8_t> edited) {
    InterCase c = good;
    c.bytes = std::move(edited);
    out.push_back({what, row, code, std::move(says), std::move(c)});
  };
  auto sections = [&](const char* what, int row, C code, std::string says,
                      const std::vector<Section>& edited) {
    bytes(what, row, code, std::move(says), assemble(b, edited));
  };
  auto held = [&](const char* what, int row, C code, std::string says,
                  InterCase c) {
    out.push_back({what, row, code, std::move(says), std::move(c)});
  };
  std::vector<std::uint8_t> e;
  std::vector<Section> t;

  e = b;
  e[6] = 2;
  bytes("frame type 2", 1, C::Unsupported, "frame type 2", e);
  e = b;
  e[4] = 4;
  bytes("version 4", 1, C::Unsupported, "version 4", e);
  e = b;
  e[7] = 1;
  bytes("reserved byte", 1, C::InvalidArgument, "reserved header byte", e);

  sections("unknown required section", 2, C::Unsupported,
           "unknown required section 9",
           {s[0], s[1], s[2], s[3], Section{9, d::kSectionRequired, {1}}});
  t = s;
  t[0].flags |= 2;
  sections("REFERENCE with an undefined flag", 2, C::Unsupported,
           "section 4 sets flags 3", t);

  sections("REFERENCE twice", 3, C::InvalidArgument, "section 4 appears twice",
           {s[0], s[0], s[1], s[2], s[3]});
  sections("REFERENCE missing", 3, C::InvalidArgument,
           "REFERENCE section is missing", {s[1], s[2], s[3]});
  e = b;
  e[6] = 0;
  bytes("REFERENCE on an intra frame", 3, C::InvalidArgument,
        "an intra frame has a REFERENCE section", e);
  e = b;
  e.push_back(0);
  bytes("a trailing byte", 3, C::InvalidArgument, "bytes after the last", e);

  t = s;
  t[0].body.pop_back();
  sections("REFERENCE of 19 bytes", 4, C::InvalidArgument, "is not 20 bytes",
           t);
  t = s;
  t[0].body.push_back(0);
  sections("REFERENCE of 21 bytes", 4, C::InvalidArgument, "is not 20 bytes",
           t);
  t = s;
  put_u32(t[0].body, 12, 0);
  sections("update size 0", 4, C::InvalidArgument, "update size 0", t);
  e = b;
  put_u32(e, 32, a - 1);
  bytes("fewer blocks than new blocks", 4, C::InvalidArgument,
        "not between its", e);
  e = b;
  put_u32(e, 32, m + a + 1);
  bytes("more blocks than survivors and new blocks", 4, C::InvalidArgument,
        "not between its", e);
  // The update segments as one, so SEGMENTS still agrees.
  t = s;
  put_u32(t[0].body, 8, 0xFFFFFFFFu);
  put_u32(t[0].body, 12, 0xFFFFFFFFu);
  t[2].body = join_lengths(s[2].body, 0, updates);
  sections("M + A past 2^32 - 1", 4, C::InvalidArgument, "2^32 - 1 entries", t);
  // What A <= N bounds: a frame of one block that names 2^22 new ones in one
  // segment, all else in agreement, would have the reader hold 2^22 + M
  // entries against a caller's room for one.
  e = b;
  put_u32(e, 32, 1);
  put_u32(e, 36, 0xFFFFFFFFu);
  t = sections_of(e);
  put_u32(t[0].body, 16, 1u << 22);
  t[2].body = join_lengths(s[2].body, updates, news);
  {
    InterCase c = good;
    c.bytes = assemble(e, t);
    c.max_blocks = 1;
    held("2^22 new blocks in a frame of 1", 4, C::InvalidArgument,
         "not between its", c);
  }

  t = s;
  t[1].body.pop_back();
  sections("mode table cut short", 5, C::InvalidArgument,
           mode_table_name + " is malformed", t);
  // The mode table is the last: its bytes are whatever follows the others.
  std::size_t mode_table = 0;
  {
    const std::vector<std::uint8_t>& tb = s[1].body;
    std::size_t at = 0;
    auto varint = [&]() {
      std::uint32_t v = 0;
      for (std::uint32_t shift = 0;; shift += 7) {
        const std::uint8_t byte = tb[at++];
        v |= std::uint32_t(byte & 0x7F) << shift;
        if ((byte & 0x80) == 0) return v;
      }
    };
    for (std::uint32_t model = 0;
         model + 1 < d::frame_model_count(d::FrameType::kInter, get_u32(b, 20));
         ++model) {
      const std::uint32_t used = varint();
      for (std::uint32_t i = 0; i < 2 * used; ++i) varint();
    }
    mode_table = at;
  }
  auto with_mode_table = [&](std::vector<std::uint8_t> table) {
    std::vector<Section> r = s;
    r[1].body.resize(mode_table);
    r[1].body.insert(r[1].body.end(), table.begin(), table.end());
    return r;
  };
  sections("mode table not summing to 4096", 5, C::InvalidArgument,
           mode_table_name + " does not sum to 4096",
           with_mode_table({2, 0, 0x80, 0x10, 0, 0x80, 0x0F}));
  sections("mode past the alphabet", 5, C::InvalidArgument,
           mode_table_name + " is malformed",
           with_mode_table({1, 5, 0x80, 0x20}));
  // keep and motion at 2048 each.
  sections("motion", 6, C::Unsupported, "uses the motion mode",
           with_mode_table({2, 0, 0x80, 0x10, 3, 0x80, 0x10}));
  sections("motion, then appended table bytes", 6, C::Unsupported,
           "uses the motion mode",
           with_mode_table({2, 0, 0x80, 0x10, 3, 0x80, 0x10, 7, 7, 7}));

  t = s;
  t[2].body.resize(t[2].body.size() - 4);
  sections("one segment length fewer", 7, C::InvalidArgument,
           "one length per segment", t);
  t = s;
  put_u32(t[2].body, 0, 3);
  sections("an odd segment length", 7, C::InvalidArgument,
           segment(0, " has an impossible length"), t);
  t = s;
  put_u32(t[2].body, 0, get_u32(t[2].body, 0) + 2);
  sections("lengths not summing to PAYLOAD", 7, C::InvalidArgument,
           "do not add up to the PAYLOAD", t);

  InterCase c = good;
  c.held = false;
  held("no reference held", 8, C::NotFound, "no reference frame is held", c);
  c = good;
  c.reference.hash ^= 1;
  held("another hash", 8, C::NotFound, "not the one held", c);
  c = good;
  c.reference.header.block_count += 1;
  held("another held M", 8, C::NotFound, "not the one held", c);
  t = s;
  put_u32(t[0].body, 8, m + 1);  // the same update segments: 41 / 7
  sections("another M in the frame", 8, C::NotFound, "not the one held", t);

  const std::string differs = " is not its reference's";
  c = good;
  c.reference.header.voxel_size *= 2.0f;
  held("held voxel_size", 9, C::InvalidArgument, "voxel_size" + differs, c);
  c = good;
  c.reference.header.trunc_dist *= 2.0f;
  held("held trunc_dist", 9, C::InvalidArgument, "trunc_dist" + differs, c);
  c = good;
  c.reference.header.params.coefficient_count += 1;
  held("held K", 9, C::InvalidArgument, "coefficient count" + differs, c);
  c = good;
  c.reference.header.params.quantization_scale *= 2.0f;
  held("held scale", 9, C::InvalidArgument, "quantization scale" + differs, c);
  c = good;
  c.reference.header.params.quantization_weights[0] *= 2.0f;
  held("held DC weight", 9, C::InvalidArgument,
       "kept quantization weights" + differs, c);
  e = b;
  put_f32(e, 8, 0.006f);
  bytes("the frame's voxel_size", 9, C::InvalidArgument, "voxel_size" + differs,
        e);
  e = b;
  put_f32(e, 24, 0.3f);
  bytes("the frame's scale", 9, C::InvalidArgument,
        "quantization scale" + differs, e);
  e = b;
  put_f32(e, header - 4, 1.5f);
  bytes("the frame's last kept weight", 9, C::InvalidArgument,
        "kept quantization weights" + differs, e);

  c = good;
  c.max_blocks = n - 1;
  held("more blocks than the caller holds", 10, C::OutOfMemory,
       "the caller can hold", c);

  t = s;
  t[3].body[0] = 0;
  t[3].body[1] = 0;
  sections("an update segment's state below L", 11, C::InvalidArgument,
           segment(0, " is corrupt"), t);
  t = s;
  t[3].body[stream_at(updates)] = 0;
  t[3].body[stream_at(updates) + 1] = 0;
  sections("a new segment's state below L", 11, C::InvalidArgument,
           segment(updates, " is corrupt"), t);
  t = s;
  set_first_x_low_bits(t[3].body, stream_at(updates + 1), 502);
  sections("a new segment starting on the one before's end", 11,
           C::InvalidArgument,
           segment(updates + 1, " does not start after the one before"), t);
  t = s;
  set_first_x_low_bits(t[3].body, stream_at(updates + 2), 0xFFF);
  sections("a new segment stepping out of int32", 11, C::InvalidArgument,
           segment(updates + 2, " steps a coordinate outside int32"), t);

  e = b;
  put_u32(e, 32, n + 1);
  bytes("one block more than survive", 12, C::InvalidArgument,
        "are new, but the frame holds", e);
  e = b;
  put_u32(e, 32, n - 1);
  bytes("one block fewer than survive", 12, C::InvalidArgument,
        "are new, but the frame holds", e);
  return out;
}

}  // namespace codec_frames
