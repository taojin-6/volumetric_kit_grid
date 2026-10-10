// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// Host-only tests for the v1 intra frame (bitstream.hpp): round trips across
// block counts, coefficient counts and segment sizes; coordinates at the ends
// of int32; every writer refusal; every reader refusal, each made by editing
// one field or section of a valid frame; the section rules; coordinates out
// of order across segments; truncation and random corruption, which must fail
// cleanly and never read out of bounds; and what an all-zero frame costs.
// CPU-only, so these always run.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <set>
#include <string>
#include <tuple>
#include <vector>

#include "bitstream.hpp"
#include "codec_frames.hpp"
#include "dct_tables.hpp"
#include "volumetric_kit/recon/codec/codec_params.hpp"

namespace vr = volumetric_kit::recon;
namespace vkc = volumetric_kit::core;
namespace codec = volumetric_kit::recon::codec;
namespace d = volumetric_kit::recon::codec::detail;

#define CHECK(cond)                                                        \
  do {                                                                     \
    if (!(cond)) {                                                         \
      std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
      return 1;                                                            \
    }                                                                      \
  } while (0)

namespace {

constexpr std::int32_t kMax32 = std::numeric_limits<std::int32_t>::max();

using codec_frames::Lcg;
using codec_frames::make_frame;

bool same_bits(float a, float b) { return std::memcmp(&a, &b, sizeof(a)) == 0; }

// A frame carries the weights of the K kept bases only.
bool same_kept_weights(const codec::CodecParams& a,
                       const codec::CodecParams& b) {
  const auto zigzag = codec::detail::zigzag_order();
  for (std::uint32_t j = 0; j < a.coefficient_count; ++j) {
    if (!same_bits(a.quantization_weights[zigzag[j]],
                   b.quantization_weights[zigzag[j]])) {
      return false;
    }
  }
  return true;
}

bool same_frame(const d::IntraFrame& a, const d::IntraFrame& b) {
  return same_bits(a.voxel_size, b.voxel_size) &&
         same_bits(a.blocks.trunc_dist, b.blocks.trunc_dist) &&
         a.blocks.params.coefficient_count ==
             b.blocks.params.coefficient_count &&
         same_bits(a.blocks.params.quantization_scale,
                   b.blocks.params.quantization_scale) &&
         same_kept_weights(a.blocks.params, b.blocks.params) &&
         a.coords == b.coords &&
         a.blocks.coefficients == b.blocks.coefficients &&
         a.blocks.masks == b.blocks.masks;
}

vkc::Result<d::IntraFrame> read(const std::vector<std::uint8_t>& bytes,
                                std::uint32_t max_blocks = 1u << 20) {
  return d::read_intra_frame(bytes.data(), bytes.size(), max_blocks);
}

// Write, read back, compare, and write again: the frame survives exactly and
// re-encodes to the same bytes.
int round_trip(const d::IntraFrame& f, std::uint32_t segment_size) {
  d::FrameWriteOptions opt;
  opt.segment_size = segment_size;
  vkc::Result<std::vector<std::uint8_t>> bytes = d::write_intra_frame(f, opt);
  CHECK(bytes.ok());
  vkc::Result<d::IntraFrame> back =
      read(bytes.value(), static_cast<std::uint32_t>(f.coords.size()));
  CHECK(back.ok());
  CHECK(same_frame(f, back.value()));
  vkc::Result<std::vector<std::uint8_t>> again =
      d::write_intra_frame(back.value(), opt);
  CHECK(again.ok());
  CHECK(again.value() == bytes.value());
  return 0;
}

int round_trip_case() {
  for (std::size_t n : {1u, 2u, 63u, 64u, 65u, 1000u}) {
    for (std::uint32_t k : {1u, 32u}) {
      for (std::uint32_t r : {1u, 7u, 64u, 100000u}) {
        CHECK(round_trip(make_frame(n, k, 1000 * n + k), r) == 0);
      }
    }
  }
  CHECK(round_trip(make_frame(40, codec::kVoxelsPerBlock, 5), 16) == 0);
  // The default segment size, and non-default steps carried bit for bit.
  d::IntraFrame f = make_frame(300, 32, 9);
  f.blocks.params.quantization_scale = 0.3141f;
  for (std::size_t i = 0; i < f.blocks.params.quantization_weights.size();
       ++i) {
    f.blocks.params.quantization_weights[i] = 0.25f + float(i % 97) * 0.03125f;
  }
  CHECK(round_trip(f, d::kDefaultSegmentSize) == 0);
  // No blocks at all: no segments, empty tables.
  CHECK(round_trip(make_frame(0, 32, 1), 64) == 0);
  return 0;
}

// Coordinates at the ends of int32, and deltas of 2^32 - 1 on every axis.
int extreme_coords_case() {
  const d::IntraFrame f = codec_frames::extreme_frame();
  CHECK(round_trip(f, 64) == 0);
  CHECK(round_trip(f, 1) == 0);  // every coordinate in full
  return 0;
}

// The golden frames' bytes, which pin v3 itself: a writer and a reader changed
// together still round-trip, but no longer write these.
int golden_case() {
  for (const codec_frames::GoldenFrame& g : codec_frames::kGoldenFrames) {
    const d::IntraFrame f = codec_frames::golden_frame(g);
    d::FrameWriteOptions opt;
    opt.segment_size = g.segment_size;
    const vkc::Result<std::vector<std::uint8_t>> bytes =
        d::write_intra_frame(f, opt);
    CHECK(bytes.ok());
    const std::uint32_t content = codec_frames::content_hash(f);
    const std::uint32_t hash = vr_test::fnv1a(bytes.value());
    if (content != g.content || bytes.value().size() != g.size ||
        hash != g.hash) {
      std::fprintf(stderr,
                   "golden frame %zu blocks, K %u, R %u: content 0x%08xu, "
                   "%zu bytes, hash 0x%08xu\n",
                   f.coords.size(), g.k, g.segment_size, content,
                   bytes.value().size(), hash);
    }
    CHECK(content == g.content);
    CHECK(bytes.value().size() == g.size);
    CHECK(hash == g.hash);
    CHECK(round_trip(f, g.segment_size) == 0);
  }
  return 0;
}

int write_refusals_case() {
  const d::IntraFrame good = make_frame(10, 8, 3);
  CHECK(d::write_intra_frame(good).ok());
  auto refused = [](const d::IntraFrame& f, d::FrameWriteOptions opt = {}) {
    return !d::write_intra_frame(f, opt).ok();
  };
  d::IntraFrame f = good;
  std::swap(f.coords[3], f.coords[4]);  // out of order
  CHECK(refused(f));
  f = good;
  f.coords[4] = f.coords[3];  // a duplicate
  CHECK(refused(f));
  f = good;
  f.blocks.coefficients.pop_back();
  CHECK(refused(f));
  f = good;
  f.blocks.masks.push_back(0);
  CHECK(refused(f));
  f = good;  // the one int16 past the clamp
  f.blocks.coefficients[5] = std::numeric_limits<std::int16_t>::min();
  CHECK(refused(f));
  f = good;
  f.blocks.params.quantization_weights[0] = 0.0f;
  CHECK(refused(f));
  f = good;  // a weight beyond K is never used, so never refused
  f.blocks.params.quantization_weights[511] = 0.0f;
  CHECK(!refused(f));
  f = good;
  f.voxel_size = std::numeric_limits<float>::quiet_NaN();
  CHECK(refused(f));
  f = good;
  f.blocks.trunc_dist = -0.04f;
  CHECK(refused(f));
  d::FrameWriteOptions zero;
  zero.segment_size = 0;
  CHECK(refused(good, zero));
  return 0;
}

// The models' layout, which the kernels index their counts and tables by:
// each model's first entry is every earlier model's alphabet summed, and one
// past the last model is the arrays' size.
int model_layout_case() {
  for (std::uint32_t k : {1u, 64u, codec::kVoxelsPerBlock}) {
    CHECK(d::frame_model_base(0) == 0);
    std::uint32_t sum = 0;
    for (std::uint32_t m = 0; m < d::frame_model_count(k); ++m) {
      CHECK(d::frame_model_base(m) == sum);
      sum += d::frame_model_alphabet(m);
    }
    CHECK(d::frame_model_base(d::frame_model_count(k)) == sum);
  }
  return 0;
}

// The container check the device writer relies on: assemble_intra_frame
// refuses a coded frame whose header fields or sections disagree.
int assemble_refusals_case() {
  d::CodedFrame good;
  good.voxel_size = 0.005f;
  good.trunc_dist = 0.04f;
  good.params.coefficient_count = 8;
  good.block_count = 10;
  good.segment_size = 4;  // three segments
  std::vector<std::vector<std::uint64_t>> counts(d::frame_model_count(8));
  for (std::uint32_t m = 0; m < counts.size(); ++m) {
    counts[m].assign(d::frame_model_alphabet(m), 1);
  }
  good.tables = d::frame_tables(counts);
  good.segment_lengths = {4, 4, 4};
  good.payload.assign(12, 0);
  CHECK(d::assemble_intra_frame(good).ok());
  auto refused = [](const d::CodedFrame& f) {
    return !d::assemble_intra_frame(f).ok();
  };
  d::CodedFrame f = good;
  f.voxel_size = std::numeric_limits<float>::infinity();
  CHECK(refused(f));
  f = good;
  f.trunc_dist = 0.0f;
  CHECK(refused(f));
  f = good;
  f.params.coefficient_count = codec::kVoxelsPerBlock + 1;
  CHECK(refused(f));
  f = good;
  f.segment_size = 0;
  CHECK(refused(f));
  f = good;
  f.tables.pop_back();
  CHECK(refused(f));
  f = good;
  f.block_count = 13;  // four segments, three lengths
  CHECK(refused(f));
  f = good;
  f.payload.pop_back();
  CHECK(refused(f));
  return 0;
}

// --- Editing a valid frame.
// ----------------------------------------------------

void put_u32(std::vector<std::uint8_t>& b, std::size_t at, std::uint32_t v) {
  for (int i = 0; i < 4; ++i) {
    b[at + std::size_t(i)] = static_cast<std::uint8_t>(v >> (8 * i));
  }
}

void put_f32(std::vector<std::uint8_t>& b, std::size_t at, float f) {
  std::uint32_t v = 0;
  std::memcpy(&v, &f, sizeof(v));
  put_u32(b, at, v);
}

std::uint32_t get_u32(const std::vector<std::uint8_t>& b, std::size_t at) {
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
std::vector<Section> sections_of(const std::vector<std::uint8_t>& b) {
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

std::vector<std::uint8_t> assemble(const std::vector<std::uint8_t>& frame,
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

bool refused_as(const std::vector<std::uint8_t>& bytes, vkc::Status::Code code,
                std::uint32_t max_blocks = 1u << 20) {
  vkc::Result<d::IntraFrame> r = read(bytes, max_blocks);
  return !r.ok() && r.status().domain() == code;
}

int header_refusals_case() {
  const d::IntraFrame f = make_frame(100, 16, 4);
  const std::vector<std::uint8_t> good = d::write_intra_frame(f).value();
  CHECK(read(good).ok());
  using C = vkc::Status::Code;
  std::vector<std::uint8_t> b;

  b = good;
  b[0] = 'X';
  CHECK(refused_as(b, C::InvalidArgument));  // magic
  b = good;
  b[4] = 4;
  CHECK(refused_as(b, C::Unsupported));  // future version 4
  b = good;
  b[4] = 2;
  CHECK(refused_as(b, C::Unsupported));  // version 2, whose quantizer differs
  b = good;
  b[4] = 1;
  CHECK(refused_as(b, C::Unsupported));  // version 1, whose mask code differs
  b = good;
  b[6] = 1;
  CHECK(refused_as(b, C::Unsupported));  // frame type 1 (inter)
  b = good;
  b[7] = 1;
  CHECK(refused_as(b, C::InvalidArgument));  // reserved
  b = good;
  put_f32(b, 8, 0.0f);
  CHECK(refused_as(b, C::InvalidArgument));  // voxel_size
  b = good;
  put_f32(b, 12, std::numeric_limits<float>::infinity());
  CHECK(refused_as(b, C::InvalidArgument));  // trunc_dist
  b = good;
  put_u32(b, 16, 4);
  CHECK(refused_as(b, C::Unsupported));  // block size
  b = good;
  put_u32(b, 20, 0);
  CHECK(refused_as(b, C::InvalidArgument));  // K
  b = good;
  put_f32(b, 24, std::numeric_limits<float>::quiet_NaN());
  CHECK(refused_as(b, C::InvalidArgument));  // quantization scale
  // A step past kMaxStep, finite but enough to make a decoded coefficient
  // infinite and the inverse's sums NaN.
  b = good;
  put_f32(b, d::kFramePrefixBytes, 1e38f);
  CHECK(refused_as(b, C::InvalidArgument));  // effective step exceeds ceiling
  b = good;
  put_u32(b, 28, 15);
  CHECK(refused_as(b, C::InvalidArgument));  // weight count other than K
  // The first and last kept weights; a subnormal one is refused too.
  for (std::size_t i : {std::size_t(0), std::size_t(15)}) {
    for (float value : {0.0f, -1.0f, std::numeric_limits<float>::quiet_NaN(),
                        std::numeric_limits<float>::infinity(), 1e-20f,
                        std::numeric_limits<float>::denorm_min()}) {
      b = good;
      put_f32(b, d::kFramePrefixBytes + i * sizeof(float), value);
      CHECK(refused_as(b, C::InvalidArgument));
      CHECK(!d::read_frame_header(b.data(), b.size()).ok());
    }
  }
  b = good;
  put_u32(b, 36, 0);
  CHECK(refused_as(b, C::InvalidArgument));  // segment size
  b = good;
  put_u32(b, 40, 0x40000000u);
  CHECK(refused_as(b, C::InvalidArgument));  // section table past the end
  // The caller's limit, which the frame's size cannot stand in for: a sound
  // frame past it is too big to hold, not corrupt ...
  CHECK(refused_as(good, C::OutOfMemory, 99));
  CHECK(read(good, 100).ok());
  // ... while a count the segment table disagrees with is corrupt, however
  // far past the limit it claims to be.
  b = good;
  put_u32(b, 32, 0x7FFFFFFFu);
  CHECK(refused_as(b, C::InvalidArgument, 99));
  // Exact length: a trailing byte is refused, as is a null frame.
  b = good;
  b.push_back(0);
  CHECK(refused_as(b, C::InvalidArgument));
  CHECK(!d::read_intra_frame(nullptr, 64, 100).ok());
  return 0;
}

int section_rules_case() {
  const d::IntraFrame f = make_frame(100, 16, 6);
  const std::vector<std::uint8_t> good = d::write_intra_frame(f).value();
  const std::vector<Section> s = sections_of(good);
  CHECK(s.size() == 3);
  CHECK(assemble(good, s) == good);  // the helpers are faithful
  using C = vkc::Status::Code;

  // Order does not matter.
  CHECK(read(assemble(good, {s[2], s[0], s[1]})).ok());
  // Each required section must be there, once.
  for (std::size_t drop = 0; drop < 3; ++drop) {
    std::vector<Section> fewer = s;
    fewer.erase(fewer.begin() + std::ptrdiff_t(drop));
    CHECK(refused_as(assemble(good, fewer), C::InvalidArgument));
  }
  CHECK(
      refused_as(assemble(good, {s[0], s[1], s[1], s[2]}), C::InvalidArgument));
  // An unknown optional section is skipped; an unknown required one stops
  // the read.
  Section extra{9, 0, {1, 2, 3}};
  vkc::Result<d::IntraFrame> with_extra =
      read(assemble(good, {s[0], extra, s[1], s[2]}));
  CHECK(with_extra.ok());
  CHECK(same_frame(f, with_extra.value()));
  extra.flags = d::kSectionRequired;
  CHECK(refused_as(assemble(good, {s[0], extra, s[1], s[2]}), C::Unsupported));
  // A flag v1 does not define is refused on a known section, whose body it
  // may reinterpret, and ignored on an unknown optional one, which is skipped.
  for (std::size_t known = 0; known < 3; ++known) {
    std::vector<Section> flagged = s;
    flagged[known].flags = static_cast<std::uint16_t>(flagged[known].flags | 2);
    CHECK(refused_as(assemble(good, flagged), C::Unsupported));
  }
  extra.flags = 2;
  CHECK(read(assemble(good, {s[0], extra, s[1], s[2]})).ok());

  // SEGMENTS must list one length per segment, each a whole number of words
  // and at least a state, adding up to the PAYLOAD.
  std::vector<Section> t = s;
  t[1].body.resize(t[1].body.size() - 4);
  CHECK(refused_as(assemble(good, t), C::InvalidArgument));
  t = s;
  put_u32(t[1].body, 0, 3);
  CHECK(refused_as(assemble(good, t), C::InvalidArgument));
  t = s;
  put_u32(t[1].body, 0, get_u32(t[1].body, 0) + 2);
  CHECK(refused_as(assemble(good, t), C::InvalidArgument));
  // A segment whose stream is malformed is refused, not returned: zeroing its
  // initial state puts it below L, which no encoder can produce. (An
  // arbitrary flipped byte is NOT guaranteed to be refused -- one in raw bits
  // decodes to a different, valid frame; see codec_rans_test.)
  t = s;
  t[2].body[0] = 0;
  t[2].body[1] = 0;
  CHECK(refused_as(assemble(good, t), C::InvalidArgument));
  return 0;
}

// TABLES, on a frame with no blocks so the tables are all that is read. With
// K = 1 there are 14 models -- five for the coordinates, the mask class, the
// plane, three line and three byte models, and one coefficient -- each written
// as a varint count of used symbols.
int table_rules_case() {
  constexpr std::size_t kModels = 14;
  const d::IntraFrame f = make_frame(0, 1, 1);
  const std::vector<std::uint8_t> good = d::write_intra_frame(f).value();
  std::vector<Section> s = sections_of(good);
  CHECK(s[0].body == std::vector<std::uint8_t>(kModels, 0));
  using C = vkc::Status::Code;
  // @p head, then an empty table for each model it leaves.
  auto with_tables = [&](std::vector<std::uint8_t> head, std::size_t models) {
    head.insert(head.end(), kModels - models, 0);
    std::vector<Section> t = s;
    t[0].body = std::move(head);
    return assemble(good, t);
  };
  // Model 0 with one symbol at 4096 (a two-byte varint): valid.
  CHECK(read(with_tables({1, 0, 0x80, 0x20}, 1)).ok());
  // Not summing to 4096.
  CHECK(refused_as(with_tables({1, 0, 100}, 1), C::InvalidArgument));
  // A symbol past its alphabet: model 5, the mask class, has 3, and model 6,
  // the plane, has 4.
  CHECK(refused_as(with_tables({0, 0, 0, 0, 0, 1, 5, 0x80, 0x20}, 6),
                   C::InvalidArgument));
  CHECK(read(with_tables({0, 0, 0, 0, 0, 0, 1, 3, 0x80, 0x20}, 7)).ok());
  CHECK(refused_as(with_tables({0, 0, 0, 0, 0, 0, 1, 4, 0x80, 0x20}, 7),
                   C::InvalidArgument));
  // A zero frequency.
  CHECK(
      refused_as(with_tables({2, 0, 0, 0, 0x80, 0x20}, 1), C::InvalidArgument));
  // Too few models, and a trailing byte.
  CHECK(refused_as(with_tables({}, kModels - 3), C::InvalidArgument));
  CHECK(refused_as(with_tables({0}, 0), C::InvalidArgument));
  // A varint of more than 32 bits.
  CHECK(refused_as(with_tables({0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x0F}, 1),
                   C::InvalidArgument));
  // An overlong varint: the valid table above with its gap of 0 spelled
  // {0x80, 0x00}, then with 4096 spelled in three bytes. Each value has one
  // spelling, so a frame's tables have one too.
  CHECK(refused_as(with_tables({1, 0x80, 0x00, 0x80, 0x20}, 1),
                   C::InvalidArgument));
  CHECK(
      refused_as(with_tables({1, 0, 0x80, 0xA0, 0x00}, 1), C::InvalidArgument));
  return 0;
}

// Rewrite the low 12 bits of the x of the first block of the segment whose
// stream starts at @p at in PAYLOAD. A segment's first raw chunk is the low 12
// bits of its stream's initial state word (bytes 2-3), and raw bits leave the
// state untouched, so this moves that one coordinate and nothing else -- the
// edit no end check can see, and so how a test reaches what only a flipped
// raw bit makes.
void set_first_x_low_bits(std::vector<std::uint8_t>& payload, std::size_t at,
                          std::uint32_t low12) {
  payload[at + 2] = static_cast<std::uint8_t>(low12 & 0xFF);
  payload[at + 3] = static_cast<std::uint8_t>((payload[at + 3] & 0xF0) |
                                              ((low12 >> 8) & 0x0F));
}

// Blocks all observed with zero coefficients at @p coords, K = 1.
d::IntraFrame plain_frame(const std::vector<vr::Vec3i>& coords) {
  d::IntraFrame f = make_frame(0, 1, 1);
  f.coords = coords;
  f.blocks.masks.assign(coords.size() * codec::kMaskWordsPerBlock, ~0u);
  f.blocks.coefficients.assign(coords.size(), 0);
  return f;
}

// A delta that steps a coordinate out of int32 is refused, not wrapped. No
// writer makes one, so it is made by hand: x = 0x7FFFF000 becomes INT32_MAX,
// and the next block's run step (+1) leaves int32.
int coord_overflow_case() {
  d::IntraFrame f = plain_frame({{0x7FFFF000, 0, 0}, {0x7FFFF001, 0, 0}});
  const std::vector<std::uint8_t> good = d::write_intra_frame(f).value();
  CHECK(read(good).ok());
  std::vector<Section> s = sections_of(good);
  set_first_x_low_bits(s[2].body, 0, 0xFFF);
  vkc::Result<d::IntraFrame> r = read(assemble(good, s));
  CHECK(!r.ok());
  CHECK(r.status().message().find("outside int32") != std::string::npos);
  // The same patch one block earlier is merely a different, valid frame:
  // raw bits carry no redundancy (see codec_rans_test).
  f = plain_frame({{0x7FFFF000, 0, 0}});
  const std::vector<std::uint8_t> one = d::write_intra_frame(f).value();
  s = sections_of(one);
  set_first_x_low_bits(s[2].body, 0, 0xFFF);
  vkc::Result<d::IntraFrame> moved = read(assemble(one, s));
  CHECK(moved.ok());
  CHECK(moved.value().coords[0].x == kMax32);

  // A stream that fails partway through a block's deltas is reported as
  // corrupt, not as leaving int32, although its zeros decode as a run step
  // off INT32_MAX: a one-block frame whose header claims two reaches for the
  // never-used z-step table.
  f = plain_frame({{kMax32, 0, 0}});
  std::vector<std::uint8_t> b = d::write_intra_frame(f).value();
  put_u32(b, 32, 2);
  r = read(b);
  CHECK(!r.ok());
  CHECK(r.status().message().find("corrupt") != std::string::npos);
  return 0;
}

// Coordinates must increase across segments too. Within one the delta code
// cannot step backwards, but a segment's first coordinate is raw bits, so a
// flip there decodes cleanly: the reader must check the order itself, or a
// duplicate reaches the inverse transform as two workgroups on one block.
int segment_order_case() {
  const d::IntraFrame f = plain_frame({{4, 0, 0}, {5, 0, 0}});
  d::FrameWriteOptions one_each;
  one_each.segment_size = 1;
  const std::vector<std::uint8_t> good =
      d::write_intra_frame(f, one_each).value();
  CHECK(read(good).ok());
  const std::vector<Section> s = sections_of(good);
  const std::size_t second = get_u32(s[1].body, 0);  // segment 1's offset
  auto with_second_x = [&](std::uint32_t x) {
    std::vector<Section> t = s;
    set_first_x_low_bits(t[2].body, second, x);
    return read(assemble(good, t));
  };
  // A duplicate, and a step backwards: both refused.
  for (std::uint32_t x : {4u, 3u}) {
    vkc::Result<d::IntraFrame> r = with_second_x(x);
    CHECK(!r.ok());
    CHECK(r.status().message().find("does not start after") !=
          std::string::npos);
  }
  // A step forwards is only a different, valid frame.
  vkc::Result<d::IntraFrame> r = with_second_x(9);
  CHECK(r.ok());
  CHECK(r.value().coords[1] == vr::Vec3i(9, 0, 0));
  return 0;
}

// Every truncation fails cleanly, and random corruption never gets the reader
// out of bounds -- the property the sanitizer job checks on every leg.
int corruption_case() {
  const d::IntraFrame small = make_frame(20, 8, 12);
  d::FrameWriteOptions opt;
  opt.segment_size = 4;
  const std::vector<std::uint8_t> good =
      d::write_intra_frame(small, opt).value();
  for (std::size_t len = 0; len < good.size(); ++len) {
    CHECK(!d::read_intra_frame(good.data(), len, 1000).ok());
  }

  const d::IntraFrame f = make_frame(300, 32, 13);
  opt.segment_size = 16;
  const std::vector<std::uint8_t> big = d::write_intra_frame(f, opt).value();
  Lcg rng{17};
  int refused = 0;
  const int trials = 3000;
  for (int i = 0; i < trials; ++i) {
    std::vector<std::uint8_t> b = big;
    const std::uint32_t edits = 1 + rng.below(3);
    for (std::uint32_t e = 0; e < edits; ++e) {
      b[rng.below(std::uint32_t(b.size()))] ^=
          static_cast<std::uint8_t>(1 + rng.below(255));
    }
    refused += read(b).ok() ? 0 : 1;
  }
  std::printf("frame corruption: %d of %d refused\n", refused, trials);
  // Most edits land in the payload, where the segment's end check catches
  // them; an edit to a header float (a step, the voxel size) can still leave a
  // valid frame, which is why this is a floor rather than all of them.
  CHECK(refused >= trials * 9 / 10);
  return 0;
}

// What the format costs when there is nothing to say: 640 blocks in one run
// along x, all observed, every coefficient zero. Each segment is then its
// state plus its first coordinate in full; every other symbol has
// probability one and costs nothing.
int zero_frame_cost_case() {
  d::IntraFrame f = make_frame(0, 32, 1);
  for (int i = 0; i < 640; ++i) {
    f.coords.push_back(vr::Vec3i(i, 0, 0));
  }
  f.blocks.masks.assign(640 * codec::kMaskWordsPerBlock, ~0u);
  f.blocks.coefficients.assign(640 * 32, 0);
  const std::vector<std::uint8_t> b = d::write_intra_frame(f).value();
  const std::vector<Section> s = sections_of(b);
  const std::size_t segments = 640 / d::kDefaultSegmentSize;
  std::printf("all-zero frame: %zu bytes for 640 blocks (payload %zu)\n",
              b.size(), s[2].body.size());
  // 4 bytes of state + 12 of coordinate per segment, give or take a word.
  CHECK(s[2].body.size() <= segments * 18);
  // The 172-byte header (32 kept weights) plus segment/table overhead.
  const std::size_t header = d::frame_header_bytes(32);
  CHECK(b.size() <= header + 406);
  // And a frame with something to say costs far more: the zero case is not
  // passing by encoding nothing.
  const std::vector<std::uint8_t> busy =
      d::write_intra_frame(make_frame(640, 32, 2)).value();
  CHECK(busy.size() - header > 20 * (b.size() - header));
  return 0;
}

}  // namespace

int main() {
  if (round_trip_case() != 0) return 1;
  if (extreme_coords_case() != 0) return 1;
  if (golden_case() != 0) return 1;
  if (write_refusals_case() != 0) return 1;
  if (model_layout_case() != 0) return 1;
  if (assemble_refusals_case() != 0) return 1;
  if (header_refusals_case() != 0) return 1;
  if (section_rules_case() != 0) return 1;
  if (table_rules_case() != 0) return 1;
  if (coord_overflow_case() != 0) return 1;
  if (segment_order_case() != 0) return 1;
  if (corruption_case() != 0) return 1;
  if (zero_frame_cost_case() != 0) return 1;
  std::printf("codec bitstream: OK\n");
  return 0;
}
