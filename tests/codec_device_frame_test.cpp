// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// The device frame writer and reader against the CPU oracle's: every frame
// the writer writes, intra or P -- a P-frame from its exact entries and from
// bounded ones, as the reference store's diff leaves them, with garbage past
// them -- must be oracle::write_frame's, byte for byte, or be refused as it
// refuses it, and every frame the reader reads must decode to
// oracle::read_frame's entries or be refused as it refuses it, with
// the same code and message -- the test the 2026-09-26 decision set for the
// GPU coder -- the reader's verdict naming the first failing segment however
// many fail, and its survivor ranks an exclusive scan. Skips where no device
// is present.

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "bitstream.hpp"
#include "codec_frames.hpp"
#include "codec_oracle.hpp"
#include "codec_oracle_device.hpp"
#include "device_frame_reader.hpp"
#include "device_frame_writer.hpp"
#include "test_check.hpp"
#include "volumetric_kit/core/vulkan/allocator.hpp"
#include "volumetric_kit/core/vulkan/command_batch.hpp"
#include "volumetric_kit/core/vulkan/compute_util.hpp"
#include "volumetric_kit/core/vulkan/device.hpp"

#include "gpu_test.hpp"

namespace vr = volumetric_kit::recon;
namespace vkc = volumetric_kit::core;
namespace codec = volumetric_kit::recon::codec;
namespace d = volumetric_kit::recon::codec::detail;
namespace oracle = codec_oracle;
using codec_frames::extreme_frame;
using codec_frames::inter_frame;
using codec_frames::Lcg;
using codec_frames::make_frame;
using codec_frames::reference_of;
using codec_frames::successor;
using codec_oracle::Resident;
using C = vkc::Status::Code;
using Bytes = std::vector<std::uint8_t>;

namespace {

constexpr std::uint32_t kMaxBlocks = 1u << 20;

// The device the writer and reader run on, for the helpers below.
vkc::Device* g_device = nullptr;
vkc::Allocator* g_allocator = nullptr;

oracle::FrameWriteOptions sizes(std::uint32_t r, std::uint32_t u) {
  oracle::FrameWriteOptions o;
  o.segment_size = r;
  o.update_size = u;
  return o;
}

// A host frame's entries in device buffers, as the encoder holds the
// transform's output; with @p tail, bounded as its diff leaves a P-frame's
// (oracle::resident).
vkc::Result<Resident> resident(
    const oracle::Frame& frame,
    std::optional<std::uint32_t> tail = std::nullopt) {
  return oracle::resident({*g_device, *g_allocator}, frame.blocks, frame.modes,
                          frame.coords, 0, tail);
}

// The device writer's bytes for a host frame: its entries uploaded, counted
// and finished, as the encoder does with the transform's output, or with
// @p tail as it does with a P-frame's diff.
vkc::Result<Bytes> device_write(
    d::DeviceFrameWriter& writer, const oracle::Frame& frame,
    const oracle::FrameWriteOptions& options,
    std::optional<std::uint32_t> tail = std::nullopt) {
  VKC_ASSIGN(const Resident r, resident(frame, tail));
  vkc::CommandBatch batch(*g_device, *g_allocator);
  VKC_TRY(writer.record_count(batch, r.blocks, options.segment_size,
                              options.update_size));
  VKC_TRY(batch.submit());
  return writer.finish(
      r.blocks, frame.voxel_size, frame.blocks.trunc_dist, frame.blocks.params,
      frame.type == d::FrameType::kInter ? &frame.reference : nullptr);
}

// The garbage tails a bounded frame of @p m reference entries is written
// with: 0, 1, 7 and M positions, at most M, since no more survive.
std::set<std::uint32_t> tails(std::size_t m) {
  std::set<std::uint32_t> out;
  for (const std::size_t g :
       {std::size_t(0), std::size_t(1), std::size_t(7), m}) {
    if (g <= m) out.insert(static_cast<std::uint32_t>(g));
  }
  return out;
}

// The device reader's decode of a frame, read back as the oracle's: a skip or
// remove entry's payload, which the device leaves as it was, zero as the
// oracle's.
vkc::Result<oracle::Frame> device_read(d::DeviceFrameReader& reader,
                                       const std::uint8_t* data,
                                       std::size_t size,
                                       const d::FrameReference* reference,
                                       std::uint32_t max_blocks = kMaxBlocks) {
  VKC_ASSIGN(const d::ParsedFrame parsed,
             d::parse_frame(data, size, max_blocks, reference));
  const std::uint32_t k = parsed.header.params.coefficient_count;
  const std::uint32_t words = d::coefficient_words(k);
  oracle::Frame frame;
  frame.type = parsed.header.type;
  frame.voxel_size = parsed.header.voxel_size;
  frame.reference = parsed.reference;
  frame.blocks.trunc_dist = parsed.header.trunc_dist;
  frame.blocks.params = parsed.header.params;
  vkc::CommandBatch batch(*g_device, *g_allocator);
  VKC_ASSIGN(const d::ResidentBlocks resident,
             reader.record_decode(batch, parsed));
  const std::uint32_t entries = resident.count;
  const std::uint32_t m = resident.reference_count;
  std::vector<std::uint32_t> packed(std::size_t(entries) * words);
  std::vector<std::uint32_t> modes(m);
  std::vector<vr::volume::BlockIndex> list(entries - m);
  frame.blocks.masks.resize(std::size_t(entries) * codec::kMaskWordsPerBlock);
  if (entries != 0) {
    VKC_TRY(batch.readback(*resident.masks, 0,
                           frame.blocks.masks.size() * sizeof(std::uint32_t),
                           frame.blocks.masks.data()));
    VKC_TRY(batch.readback(*resident.coefficients, 0,
                           packed.size() * sizeof(std::uint32_t),
                           packed.data()));
  }
  if (m != 0) {
    VKC_TRY(batch.readback(*resident.modes, 0, m * sizeof(std::uint32_t),
                           modes.data()));
  }
  if (!list.empty()) {
    VKC_TRY(batch.readback(*resident.list, 0,
                           list.size() * sizeof(vr::volume::BlockIndex),
                           list.data()));
  }
  VKC_TRY(batch.submit());
  VKC_TRY(reader.check());
  for (const std::uint32_t mode : modes) {
    frame.modes.push_back(static_cast<d::BlockMode>(mode));
  }
  for (const vr::volume::BlockIndex& b : list) {
    frame.coords.push_back(b.coord);
  }
  frame.blocks.coefficients.resize(std::size_t(entries) * k);
  for (std::size_t p = 0; p < entries; ++p) {
    const bool payload = p >= frame.modes.size() ||
                         frame.modes[p] == d::BlockMode::kKeep ||
                         frame.modes[p] == d::BlockMode::kZero;
    for (std::uint32_t j = 0; j < k; ++j) {
      frame.blocks.coefficients[p * k + j] =
          payload ? static_cast<std::int16_t>(packed[p * words + j / 2] >>
                                              (16 * (j & 1)))
                  : std::int16_t(0);
    }
    for (std::uint32_t w = 0; !payload && w < codec::kMaskWordsPerBlock; ++w) {
      frame.blocks.masks[p * codec::kMaskWordsPerBlock + w] = 0;
    }
  }
  return frame;
}

// The oracle's decode of a frame and the device's: the same entries, or the
// same refusal -- code and message -- which @p refusal receives.
int same_read(d::DeviceFrameReader& reader, const Bytes& bytes,
              const d::FrameReference* reference = nullptr,
              std::string* refusal = nullptr,
              std::uint32_t max_blocks = kMaxBlocks) {
  const vkc::Result<oracle::Frame> expected =
      oracle::read_frame(bytes.data(), bytes.size(), max_blocks, reference);
  const vkc::Result<oracle::Frame> device =
      device_read(reader, bytes.data(), bytes.size(), reference, max_blocks);
  if (expected.ok() != device.ok() ||
      (!expected.ok() &&
       expected.status().message() != device.status().message())) {
    std::fprintf(stderr, "oracle: %s; device: %s\n",
                 expected.ok() ? "OK" : expected.status().message().c_str(),
                 device.ok() ? "OK" : device.status().message().c_str());
    return 1;
  }
  if (!expected.ok()) {
    CHECK(expected.status().domain() == device.status().domain());
    if (refusal != nullptr) *refusal = expected.status().message();
    return 0;
  }
  if (refusal != nullptr) refusal->clear();
  const oracle::Frame& a = expected.value();
  const oracle::Frame& b = device.value();
  CHECK(a.type == b.type);
  CHECK(a.voxel_size == b.voxel_size);
  CHECK(a.reference.hash == b.reference.hash);
  CHECK(a.reference.header.block_count == b.reference.header.block_count);
  CHECK(a.modes == b.modes);
  CHECK(a.coords == b.coords);
  CHECK(a.blocks.masks == b.blocks.masks);
  CHECK(a.blocks.coefficients == b.blocks.coefficients);
  return 0;
}

// The oracle's frame and the device's, for one frame and its sizes -- a
// P-frame also written bounded, at every tail, unless @p bounded is false --
// and the device's decode of it.
int same_bytes(d::DeviceFrameWriter& writer, d::DeviceFrameReader& reader,
               const oracle::Frame& frame,
               const oracle::FrameWriteOptions& options, bool bounded = true) {
  const bool inter = frame.type == d::FrameType::kInter;
  const vkc::Result<Bytes> expected = oracle::write_frame(frame, options);
  CHECK(expected.ok());
  std::vector<std::optional<std::uint32_t>> writes = {std::nullopt};
  if (inter && bounded) {
    for (const std::uint32_t g : tails(frame.modes.size())) writes.push_back(g);
  }
  for (const std::optional<std::uint32_t>& tail : writes) {
    const vkc::Result<Bytes> device =
        device_write(writer, frame, options, tail);
    if (!device.ok()) {
      std::fprintf(stderr, "device write: %s\n",
                   device.status().message().c_str());
    }
    CHECK(device.ok());
    if (device.value() != expected.value()) {
      std::fprintf(stderr,
                   "differs: %zu modes, %zu blocks, K %u, R %u, U %u, tail %d "
                   "(%zu vs %zu bytes)\n",
                   frame.modes.size(), frame.coords.size(),
                   frame.blocks.params.coefficient_count, options.segment_size,
                   options.update_size, tail ? int(*tail) : -1,
                   device.value().size(), expected.value().size());
    }
    CHECK(device.value() == expected.value());
  }
  return same_read(reader, expected.value(),
                   inter ? &frame.reference : nullptr);
}

int matches_oracle_case(d::DeviceFrameWriter& writer,
                        d::DeviceFrameReader& reader) {
  // Block counts on and off a segment boundary, K odd and even up to the
  // whole transform, and segments of one block through more than the frame.
  for (std::size_t n : {std::size_t(0), std::size_t(1), std::size_t(63),
                        std::size_t(64), std::size_t(65), std::size_t(700)}) {
    for (std::uint32_t k : {1u, 20u, 64u, codec::kVoxelsPerBlock}) {
      for (std::uint32_t r : {1u, 16u, 64u, 1024u}) {
        CHECK(same_bytes(writer, reader, make_frame(n, k, 31 * n + k),
                         sizes(r, 64)) == 0);
      }
    }
  }
  // Steps across all of int32, and a segment of first blocks only.
  const oracle::Frame f = extreme_frame();
  CHECK(same_bytes(writer, reader, f, sizes(64, 64)) == 0);
  CHECK(same_bytes(writer, reader, f, sizes(1, 64)) == 0);
  // A smaller frame after larger ones reuses the retained buffers.
  CHECK(same_bytes(writer, reader, make_frame(5, 64, 7), sizes(64, 64)) == 0);
  return 0;
}

// P-frames: the worked example's pinned bytes, at U = 1 too; the matrix
// (codec_frames.hpp) at every update and segment size, every mode, K and
// empty case among them, each written bounded too at U = R = 7, several
// segments of each, and at U past every frame with R = 1; one writer
// alternating exact and bounded entries; and a P-frame of no entries at all.
int inter_case(d::DeviceFrameWriter& writer, d::DeviceFrameReader& reader) {
  const d::FrameReference example = reference_of(
      oracle::write_frame(codec_frames::example_reference()).value());
  const Bytes bytes = codec_frames::example_bytes(example.hash);
  const oracle::Frame p =
      oracle::read_frame(bytes.data(), bytes.size(), 3, &example).value();
  CHECK(device_write(writer, p, {}).value() == bytes);
  CHECK(same_read(reader, bytes, &example) == 0);
  CHECK(same_bytes(writer, reader, p, sizes(64, 1)) == 0);

  int written = 0;
  const std::vector<codec_frames::InterCode> matrix =
      codec_frames::inter_matrix();
  constexpr std::uint32_t kLongest = codec::kMaxDeviceDecodeSegmentSize;
  for (const codec_frames::InterCode& c : matrix) {
    for (std::uint32_t u : codec_frames::kInterSizes) {
      for (std::uint32_t r : codec_frames::kInterSizes) {
        const bool bounded = (u == 7 && r == 7) || (u == kLongest && r == 1);
        if (same_bytes(writer, reader, c.p, sizes(r, u), bounded) != 0) {
          std::fprintf(stderr, "%s: U %u, R %u\n", c.what.c_str(), u, r);
          return 1;
        }
        ++written;
      }
    }
  }
  std::printf("P-frames: the worked example and %d of the matrix\n", written);

  // On one writer: exact, bounded, exact, then bounded at another tail,
  // each the oracle's bytes, so no frame's surplus outlives it.
  const oracle::Frame& every = matrix.front().p;  // every mode, K 1
  const auto m = static_cast<std::uint32_t>(every.modes.size());
  CHECK(every.coords.size() > 7 && m > 7);
  const Bytes want = oracle::write_frame(every, sizes(7, 7)).value();
  const std::optional<std::uint32_t> exact;
  for (const std::optional<std::uint32_t>& tail :
       {exact, std::optional<std::uint32_t>(7), exact,
        std::optional<std::uint32_t>(m)}) {
    CHECK(device_write(writer, every, sizes(7, 7), tail).value() == want);
  }

  // M = 0, A = 0: REFERENCE and an empty mode table, no segment.
  const oracle::Frame empty = make_frame(0, 8, 1);
  const d::FrameReference id = reference_of(oracle::write_frame(empty).value());
  CHECK(same_bytes(writer, reader, inter_frame(id, empty, empty), {}) == 0);
  return 0;
}

// The golden frames, intra and P: the device writes their pinned bytes, so
// every machine writes the same frame, a P-frame from its bounded entries
// too, at every tail, and reads them as the oracle does.
int golden_case(d::DeviceFrameWriter& writer, d::DeviceFrameReader& reader) {
  for (const codec_frames::GoldenFrame& g : codec_frames::kGoldenFrames) {
    const vkc::Result<Bytes> bytes = device_write(
        writer, codec_frames::golden_frame(g), sizes(g.segment_size, 64));
    CHECK(bytes.ok());
    CHECK(bytes.value().size() == g.size);
    CHECK(vr_test::fnv1a(bytes.value()) == g.hash);
    CHECK(same_read(reader, bytes.value()) == 0);
  }
  for (const codec_frames::GoldenInterFrame& g :
       codec_frames::kGoldenInterFrames) {
    const oracle::Frame p = codec_frames::golden_inter_code(g).p;
    CHECK(codec_frames::content_hash(p) == g.content);
    const vkc::Result<Bytes> bytes =
        device_write(writer, p, sizes(g.segment_size, g.update_size));
    CHECK(bytes.ok());
    CHECK(bytes.value().size() == g.size);
    CHECK(vr_test::fnv1a(bytes.value()) == g.hash);
    CHECK(same_read(reader, bytes.value(), &p.reference) == 0);
    for (const std::uint32_t tail : tails(p.modes.size())) {
      const vkc::Result<Bytes> bounded =
          device_write(writer, p, sizes(g.segment_size, g.update_size), tail);
      CHECK(bounded.ok());
      CHECK(bounded.value().size() == g.size);
      CHECK(vr_test::fnv1a(bounded.value()) == g.hash);
    }
  }
  return 0;
}

// What a refusal names, as the corruption cases count them.
struct Verdicts {
  int decoded = 0;
  int overflow = 0;
  int corrupt = 0;
  int order = 0;
  int survivors = 0;
  void count(const std::string& why) {
    decoded += why.empty() ? 1 : 0;
    overflow += why.find("outside int32") != std::string::npos ? 1 : 0;
    corrupt += why.find("is corrupt") != std::string::npos ? 1 : 0;
    order += why.find("does not start after") != std::string::npos ? 1 : 0;
    survivors += why.find("are new, but") != std::string::npos ? 1 : 0;
  }
};

// @p trials edits of 1 to 3 bytes in @p good's payload, each read by the
// oracle and the device against @p reference, into @p verdicts.
int corrupt_payloads(d::DeviceFrameReader& reader, const Bytes& good,
                     const d::FrameReference* reference, int trials, Lcg& rng,
                     Verdicts& verdicts) {
  const vkc::Result<d::ParsedFrame> parsed =
      d::parse_frame(good.data(), good.size(), kMaxBlocks, reference);
  CHECK(parsed.ok());
  const auto payload =
      static_cast<std::uint32_t>(parsed.value().payload - good.data());
  for (int trial = 0; trial < trials; ++trial) {
    Bytes b = good;
    const std::uint32_t edits = 1 + rng.below(3);
    for (std::uint32_t e = 0; e < edits; ++e) {
      b[payload + rng.below(std::uint32_t(b.size()) - payload)] ^=
          static_cast<std::uint8_t>(1 + rng.below(255));
    }
    std::string why;
    CHECK(same_read(reader, b, reference, &why) == 0);
    verdicts.count(why);
  }
  return 0;
}

// Corrupt payloads: the device refuses each one the oracle refuses, with the
// same message, and decodes the rest to the same blocks. Every verdict the
// segments have is reached.
int corruption_case(d::DeviceFrameReader& reader) {
  Lcg rng{29};
  Verdicts v;
  CHECK(corrupt_payloads(
            reader,
            oracle::write_frame(make_frame(300, 32, 13), sizes(16, 64)).value(),
            nullptr, 400, rng, v) == 0);
  CHECK(corrupt_payloads(
            reader, oracle::write_frame(extreme_frame(), sizes(2, 64)).value(),
            nullptr, 400, rng, v) == 0);
  std::printf(
      "corrupt payloads: %d decoded, %d overflow, %d corrupt, %d "
      "out of order\n",
      v.decoded, v.overflow, v.corrupt, v.order);
  CHECK(v.decoded > 0 && v.overflow > 0 && v.corrupt > 0 && v.order > 0);
  return 0;
}

// A P-frame whose new blocks are extreme_frame's, in segments of 2 -- steps
// of 2^32 - 1 that one raw edit takes out of int32 -- after a reference that
// keeps, skips, zeroes and removes.
Bytes extreme_inter(d::FrameReference& id) {
  const oracle::Frame reference = make_frame(60, 4, 31);
  id = reference_of(oracle::write_frame(reference).value());
  const oracle::Frame kept = successor(reference, 32, 100, 400, 0);
  const oracle::Frame added = extreme_frame();
  std::map<vr::Vec3i, codec_frames::Payload, codec_frames::ZyxLess> blocks;
  for (std::size_t i = 0; i < kept.coords.size(); ++i) {
    blocks.emplace(kept.coords[i], codec_frames::payload_at(kept.blocks, i));
  }
  for (std::size_t i = 0; i < added.coords.size(); ++i) {
    blocks.emplace(added.coords[i], codec_frames::payload_at(added.blocks, i));
  }
  oracle::Frame target = make_frame(0, 4, 1);
  for (const auto& [c, payload] : blocks) {
    target.coords.push_back(c);
    codec_frames::append_payload(target.blocks, payload);
  }
  return oracle::write_frame(inter_frame(id, reference, target), sizes(2, 5))
      .value();
}

// P-frame refusals: every reader refusal of C2's edit list, each one edit
// from a valid P-frame, and 800 random payload corruptions of two P-frames,
// each refused by the device with the oracle's code and message, or decoded
// to the oracle's entries.
int inter_refusals_case(d::DeviceFrameReader& reader) {
  const codec_frames::RefusalBase base = codec_frames::refusal_base();
  int edits = 0;
  int decoded_on_device = 0;
  for (const codec_frames::InterRefusal& r :
       codec_frames::inter_refusals(base)) {
    const codec_frames::InterCase& c = r.edited;
    std::string why;
    if (same_read(reader, c.bytes, c.held_reference(), &why, c.max_blocks) !=
        0) {
      std::fprintf(stderr, "row %d, %s\n", r.row, r.what);
      return 1;
    }
    CHECK(why.find(r.says) != std::string::npos);
    ++edits;
    decoded_on_device += r.row >= 11 ? 1 : 0;
  }
  std::printf("P-frame refusals: %d edits, %d judged after a device decode\n",
              edits, decoded_on_device);
  CHECK(decoded_on_device >= 6);

  Lcg rng{31};
  Verdicts v;
  const codec_frames::InterCase& good = base.good;
  CHECK(corrupt_payloads(reader, good.bytes, &good.reference, 400, rng, v) ==
        0);
  d::FrameReference id;
  const Bytes extreme = extreme_inter(id);
  CHECK(corrupt_payloads(reader, extreme, &id, 400, rng, v) == 0);
  std::printf(
      "corrupt P-frame payloads: %d decoded, %d overflow, %d corrupt, %d out "
      "of order, %d survivor counts\n",
      v.decoded, v.overflow, v.corrupt, v.order, v.survivors);
  CHECK(v.decoded > 0 && v.overflow > 0 && v.corrupt > 0 && v.order > 0);
  return 0;
}

// The device writer refuses what write_frame refuses of a frame's entries,
// with its code and message: a mode past the alphabet (the largest named),
// motion, and a written -32768 -- a keep residual, a zero or new block's
// coefficient, an intra frame's -- which the count clamps so no count
// indexes past its model. Bad coding parameters come first on both, with
// any of these. A skipped entry's -32768 is not written, so both write the
// frame.
int writer_refusals_case(d::DeviceFrameWriter& writer) {
  const oracle::Frame reference = make_frame(30, 8, 51);
  const d::FrameReference id =
      reference_of(oracle::write_frame(reference).value());
  const oracle::Frame target = successor(reference, 52, 150, 400, 5);
  auto first = [](const oracle::Frame& p, d::BlockMode mode) {
    const auto it = std::find(p.modes.begin(), p.modes.end(), mode);
    return static_cast<std::size_t>(it - p.modes.begin());
  };
  // A skip spelled as zero: the frame writes a zero entry too.
  const std::size_t spelled =
      first(inter_frame(id, reference, target), d::BlockMode::kSkip);
  const oracle::Frame good =
      inter_frame(id, reference, target, {{spelled, d::BlockMode::kZero}});
  const std::size_t keep = first(good, d::BlockMode::kKeep);
  const std::size_t skip = first(good, d::BlockMode::kSkip);
  const std::size_t zero = first(good, d::BlockMode::kZero);
  CHECK(keep < good.modes.size() && skip < good.modes.size() &&
        zero < good.modes.size());
  const std::uint32_t k = 8;
  const std::int16_t low = std::numeric_limits<std::int16_t>::min();
  struct Edit {
    const char* what;
    C code;
    oracle::Frame frame;
  };
  std::vector<Edit> edits;
  oracle::Frame f = good;
  f.modes[keep] = d::BlockMode::kMotion;
  edits.push_back({"motion", C::Unsupported, f});
  f = good;
  f.modes[keep] = static_cast<d::BlockMode>(5);
  edits.push_back({"mode 5", C::InvalidArgument, f});
  f.modes[skip] = static_cast<d::BlockMode>(7);
  f.modes[zero] = d::BlockMode::kMotion;
  edits.push_back({"modes 5 and 7, and motion", C::InvalidArgument, f});
  f = good;
  f.blocks.coefficients[keep * k + 3] = low;
  edits.push_back({"a keep residual of -32768", C::InvalidArgument, f});
  f = good;
  f.blocks.coefficients[zero * k] = low;
  edits.push_back({"a zero coefficient of -32768", C::InvalidArgument, f});
  f = good;
  f.blocks.coefficients.back() = low;
  edits.push_back({"a new block's -32768", C::InvalidArgument, f});
  f = make_frame(10, 1, 3);
  f.blocks.coefficients[4] = low;
  edits.push_back({"an intra frame's -32768", C::InvalidArgument, f});
  f = good;
  f.voxel_size = 0.0f;
  f.modes[keep] = d::BlockMode::kMotion;
  edits.push_back({"voxel_size 0, and motion", C::InvalidArgument, f});
  f.modes[keep] = static_cast<d::BlockMode>(9);
  edits.push_back({"voxel_size 0, and mode 9", C::InvalidArgument, f});
  f = good;
  f.blocks.trunc_dist = -1.0f;
  f.blocks.coefficients[keep * k + 3] = low;
  edits.push_back({"trunc_dist -1, and a -32768", C::InvalidArgument, f});
  // Each written exact, and bounded past a tail whose -32768s a count over
  // the bound would flag.
  auto writes = [](const oracle::Frame& frame) {
    return std::vector<std::optional<std::uint32_t>>{
        std::nullopt, static_cast<std::uint32_t>(
                          std::min<std::size_t>(7, frame.modes.size()))};
  };
  for (const Edit& e : edits) {
    const vkc::Result<Bytes> expected = oracle::write_frame(e.frame);
    for (const std::optional<std::uint32_t>& tail : writes(e.frame)) {
      const vkc::Result<Bytes> device = device_write(writer, e.frame, {}, tail);
      if (expected.ok() || device.ok() ||
          expected.status().domain() != e.code ||
          device.status().domain() != e.code ||
          expected.status().message() != device.status().message()) {
        std::fprintf(stderr, "%s, tail %d: oracle %s; device %s\n", e.what,
                     tail ? int(*tail) : -1,
                     expected.ok() ? "OK" : expected.status().message().c_str(),
                     device.ok() ? "OK" : device.status().message().c_str());
        return 1;
      }
    }
  }
  CHECK(oracle::write_frame(edits[2].frame).status().message() ==
        "frame writer: mode 7 is not a mode");
  CHECK(oracle::write_frame(edits.back().frame).status().message() ==
        "frame writer: trunc_dist must be finite and positive");
  f = good;
  f.blocks.coefficients[skip * k] = low;
  for (const std::optional<std::uint32_t>& tail : writes(f)) {
    CHECK(device_write(writer, f, {}, tail).value() ==
          oracle::write_frame(good).value());
  }
  return 0;
}

// Segments of at most kMaxDeviceDecodeSegmentSize entries are a rule of the
// format: both writers refuse a longer segment or update size, and both
// readers a frame that names one, with the same message, while frames at the
// limit -- one segment of that many blocks, and one update segment of that
// many reference entries -- are written and read alike.
int segment_rule_case(d::DeviceFrameWriter& writer,
                      d::DeviceFrameReader& reader) {
  constexpr std::uint32_t kLimit = codec::kMaxDeviceDecodeSegmentSize;
  const oracle::Frame many = make_frame(kLimit, 1, 11);
  CHECK(same_bytes(writer, reader, many, sizes(kLimit, 64)) == 0);
  const d::FrameReference id = reference_of(oracle::write_frame(many).value());
  const oracle::Frame skips = inter_frame(id, many, many);
  CHECK(same_bytes(writer, reader, skips, sizes(64, kLimit)) == 0);
  // The device writer refuses at the count, before it records any.
  for (const oracle::FrameWriteOptions& o :
       {sizes(kLimit + 1, 64), sizes(64, kLimit + 1)}) {
    CHECK(oracle::write_frame(skips, o).status().domain() ==
          C::InvalidArgument);
    const vkc::Status device = device_write(writer, skips, o).status();
    CHECK(device.domain() == C::InvalidArgument);
    CHECK(device.message().rfind("DeviceFrameWriter::record_count: ", 0) == 0);
  }

  // One entry more named in the header, and in REFERENCE: the same segments,
  // so the rule alone refuses them.
  const std::string past = std::to_string(kLimit + 1) + " is not in";
  Bytes intra = oracle::write_frame(many, sizes(kLimit, 64)).value();
  codec_frames::put_u32(intra, 36, kLimit + 1);
  std::string why;
  CHECK(same_read(reader, intra, nullptr, &why) == 0);
  CHECK(why.find("segment size " + past) != std::string::npos);
  const Bytes p = oracle::write_frame(skips, sizes(64, kLimit)).value();
  std::vector<codec_frames::Section> s = codec_frames::sections_of(p);
  codec_frames::put_u32(s[0].body, 12, kLimit + 1);
  CHECK(same_read(reader, codec_frames::assemble(p, s), &id, &why) == 0);
  CHECK(why.find("update size " + past) != std::string::npos);
  return 0;
}

// The reader checks a caller's parsed frame again, recording nothing for one
// it refuses: a type past the P-frame, which the oracle's decode_frame
// refuses too, and a segment or update size past the rule, which one
// invocation could not decode.
int parsed_frame_case(d::DeviceFrameReader& reader) {
  const oracle::Frame reference = make_frame(100, 16, 12);
  const Bytes intra = oracle::write_frame(reference).value();
  const d::FrameReference id = reference_of(intra);
  const Bytes p =
      oracle::write_frame(
          inter_frame(id, reference, successor(reference, 13, 100, 300, 20)))
          .value();
  CHECK(same_read(reader, intra) == 0);
  // The coordinates the reader holds after a decode of the intra frame.
  d::ResidentBlocks held;
  {
    vkc::CommandBatch batch(*g_device, *g_allocator);
    vkc::Result<d::ResidentBlocks> r = reader.record_decode(
        batch, d::parse_frame(intra.data(), intra.size(), kMaxBlocks, nullptr)
                   .value());
    CHECK(r.ok() && batch.submit().ok() && reader.check().ok());
    held = r.value();
  }
  auto decoded = [&held] {
    std::vector<vr::Vec3i> coords;
    for (const vr::volume::BlockIndex& b :
         vr_test::read_back<vr::volume::BlockIndex>(*g_device, *g_allocator,
                                                    *held.list, held.count)
             .value()) {
      coords.push_back(b.coord);
    }
    return coords;
  };
  const std::vector<vr::Vec3i> before = decoded();
  CHECK(before == reference.coords);
  auto refused = [&](const d::ParsedFrame& parsed, C code,
                     const std::string& says) {
    vkc::CommandBatch batch(*g_device, *g_allocator);
    const vkc::Status s = reader.record_decode(batch, parsed).status();
    if (s.domain() != code || s.message().rfind(says, 0) != 0) {
      std::fprintf(stderr, "%s: %s\n", says.c_str(),
                   s.ok() ? "not refused" : s.message().c_str());
      return 1;
    }
    CHECK(batch.submit().ok());
    CHECK(decoded() == before);
    return 0;
  };
  auto parsed = [](const Bytes& bytes, const d::FrameReference* held) {
    return d::parse_frame(bytes.data(), bytes.size(), kMaxBlocks, held).value();
  };

  d::ParsedFrame typed = parsed(intra, nullptr);
  typed.header.type = static_cast<d::FrameType>(2);
  CHECK(oracle::decode_frame(typed).status().domain() == C::Unsupported);
  CHECK(refused(typed, C::Unsupported,
                "DeviceFrameReader::record_decode: frame type 2") == 0);
  const std::string past =
      std::to_string(codec::kMaxDeviceDecodeSegmentSize + 1) + " is not in";
  for (const Bytes* frame : {&intra, &p}) {
    d::ParsedFrame longer = parsed(*frame, frame == &p ? &id : nullptr);
    longer.header.segment_size = codec::kMaxDeviceDecodeSegmentSize + 1;
    CHECK(refused(longer, C::InvalidArgument,
                  "DeviceFrameReader::record_decode: segment size " + past) ==
          0);
  }
  d::ParsedFrame longer = parsed(p, &id);
  longer.update_size = codec::kMaxDeviceDecodeSegmentSize + 1;
  CHECK(refused(longer, C::InvalidArgument,
                "DeviceFrameReader::record_decode: update size " + past) == 0);
  return 0;
}

// The reader's verdict is the first segment that fails, however many do and
// whichever way: a corrupt first segment, the second and the last starting
// before the segment before's end, the third exactly on it, and pairs of
// them -- each the oracle's code and message -- and it is reset for the next
// frame, which reads clean.
int verdict_case(d::DeviceFrameReader& reader) {
  oracle::Frame row = make_frame(20, 4, 41);
  row.coords.clear();
  for (int x = 0; x < 20; ++x) row.coords.push_back(vr::Vec3i(x, 3, -2));
  const Bytes good = oracle::write_frame(row, sizes(4, 64)).value();
  const d::ParsedFrame parsed =
      d::parse_frame(good.data(), good.size(), kMaxBlocks, nullptr).value();
  const auto payload = static_cast<std::size_t>(parsed.payload - good.data());
  // Where segment s's stream starts in the frame.
  auto at = [&](std::size_t s) {
    std::size_t offset = payload;
    for (std::size_t i = 0; i < s; ++i) offset += parsed.segment_lengths[i];
    return offset;
  };
  // Segment s's first x set to @p x, at or before the last of segment s - 1
  // (4s - 1): out of order. Or its state set below L: corrupt.
  auto out_of_order = [&](Bytes& b, std::size_t s, std::uint32_t x) {
    b[at(s) + 2] = static_cast<std::uint8_t>(x & 0xFF);
    b[at(s) + 3] = static_cast<std::uint8_t>((b[at(s) + 3] & 0xF0) | (x >> 8));
  };
  auto corrupt = [&](Bytes& b, std::size_t s) {
    b[at(s)] = 0;
    b[at(s) + 1] = 0;
  };
  struct Edit {
    const char* what;
    std::vector<std::pair<std::size_t, int>> edits;  // segment, x or -1
    std::string says;
  };
  const Edit edits[] = {
      {"segment 0 corrupt", {{0, -1}}, "segment 0 is corrupt"},
      {"segment 1 out of order", {{1, 0}}, "segment 1 does not start after"},
      {"the last out of order", {{4, 2}}, "segment 4 does not start after"},
      {"segment 2 on segment 1's last", {{2, 7}}, "segment 2 does not start"},
      {"1 out of order, 3 corrupt",
       {{1, 1}, {3, -1}},
       "segment 1 does not start after"},
      {"2 corrupt, 4 out of order", {{2, -1}, {4, 0}}, "segment 2 is corrupt"},
      {"1 corrupt, 3 out of order", {{3, 5}, {1, -1}}, "segment 1 is corrupt"},
  };
  for (const Edit& e : edits) {
    Bytes b = good;
    for (const auto& [s, x] : e.edits) {
      if (x < 0) {
        corrupt(b, s);
      } else {
        out_of_order(b, s, std::uint32_t(x));
      }
    }
    std::string why;
    if (same_read(reader, b, nullptr, &why) != 0 ||
        why.find(e.says) == std::string::npos) {
      std::fprintf(stderr, "%s: %s\n", e.what, why.c_str());
      return 1;
    }
    // The next frame reads clean: the verdict is the frame's own.
    CHECK(same_read(reader, good, nullptr, &why) == 0 && why.empty());
  }
  return 0;
}

// The survivor ranks: entry e's is how many of the reference entries before
// it survive, an exclusive scan, and entry M's the survivor count, at M on
// and off the scan's 256-entry chunks and past 65,536.
int ranks_case(d::DeviceFrameReader& reader) {
  for (const std::size_t m :
       {std::size_t(0), std::size_t(1), std::size_t(255), std::size_t(256),
        std::size_t(257), std::size_t(65537)}) {
    const oracle::Frame reference = make_frame(m, 2, 70 + m);
    const d::FrameReference id =
        reference_of(oracle::write_frame(reference).value());
    const oracle::Frame p =
        inter_frame(id, reference, successor(reference, 71 + m, 300, 200, 9));
    CHECK(p.modes.size() == m);
    const Bytes bytes = oracle::write_frame(p).value();
    const d::ParsedFrame parsed =
        d::parse_frame(bytes.data(), bytes.size(), kMaxBlocks, &id).value();
    vkc::CommandBatch batch(*g_device, *g_allocator);
    vkc::Result<d::ResidentBlocks> r = reader.record_decode(batch, parsed);
    CHECK(r.ok() && r.value().ranks != nullptr);
    CHECK(batch.submit().ok() && reader.check().ok());
    vkc::Result<std::vector<std::uint32_t>> ranks =
        vr_test::read_back<std::uint32_t>(*g_device, *g_allocator,
                                          *r.value().ranks, m + 1);
    CHECK(ranks.ok());
    std::vector<std::uint32_t> survives(m + 1, 0);
    for (std::size_t e = 0; e < m; ++e) {
      survives[e] = p.modes[e] == d::BlockMode::kRemove ? 0 : 1;
    }
    std::vector<std::uint32_t> expected(m + 1);
    std::exclusive_scan(survives.begin(), survives.end(), expected.begin(), 0u);
    CHECK(ranks.value() == expected);
  }
  return 0;
}

// The writer's own refusals: entries it cannot walk -- more reference
// entries than entries, or without the modes or list they need, or bounded
// without ranks -- and a segment or update size past the rule, each counting
// nothing, so finish refuses, rather than coding stale steps; a finish whose
// blocks have another entry or reference count than were counted, or are
// exact where bounded ones were counted or the reverse; a bounded set's
// survivors past its reference entries or its blocks; reference entries
// without an update size; and a reference that does not hold the blocks'
// reference entries, or none for blocks that have them.
int refusals_case(vkc::Device& device, vkc::Allocator& allocator,
                  d::DeviceFrameWriter& writer) {
  const oracle::Frame reference = make_frame(20, 8, 61);
  const d::FrameReference id =
      reference_of(oracle::write_frame(reference).value());
  oracle::Frame p =
      inter_frame(id, reference, successor(reference, 62, 100, 300, 4));
  CHECK(!p.modes.empty() && !p.coords.empty());
  const Resident r = resident(p).value();
  auto finish = [&](d::DeviceFrameWriter& w, const d::ResidentBlocks& blocks,
                    const d::FrameReference& held) {
    return w.finish(blocks, p.voxel_size, p.blocks.trunc_dist, p.blocks.params,
                    &held);
  };
  auto uncounted = [](const vkc::Result<Bytes>& written) {
    return !written.ok() &&
           written.status().message() ==
               "DeviceFrameWriter::finish: these blocks were not the ones "
               "record_count saw";
  };

  vkc::Result<std::unique_ptr<d::DeviceFrameWriter>> fresh =
      d::DeviceFrameWriter::create(device, allocator);
  CHECK(fresh.ok());
  std::vector<d::ResidentBlocks> unwalkable(3, r.blocks);
  unwalkable[0].reference_count = r.blocks.count + 1;
  unwalkable[1].modes = nullptr;
  unwalkable[2].list = nullptr;
  vkc::CommandBatch batch(*g_device, *g_allocator);
  for (const d::ResidentBlocks& blocks : unwalkable) {
    CHECK(fresh.value()->record_count(batch, blocks, 64, 64).domain() ==
          C::InvalidArgument);
  }
  d::ResidentBlocks unranked = r.blocks;
  unranked.bounded = true;
  const vkc::Status no_ranks =
      fresh.value()->record_count(batch, unranked, 64, 64);
  CHECK(no_ranks.domain() == C::InvalidArgument &&
        no_ranks.message() ==
            "DeviceFrameWriter::record_count: the entries lack a buffer they "
            "need");
  const std::uint32_t over = codec::kMaxDeviceDecodeSegmentSize + 1;
  CHECK(fresh.value()->record_count(batch, r.blocks, over, 64).domain() ==
        C::InvalidArgument);
  CHECK(fresh.value()->record_count(batch, r.blocks, 64, over).domain() ==
        C::InvalidArgument);
  CHECK(batch.submit().ok());
  CHECK(uncounted(finish(*fresh.value(), r.blocks, p.reference)));

  // Counted with N entries, M of them reference entries; finished with
  // N - 1 entries, whose layout the count did not make, and with M - 1
  // against a reference of M - 1 blocks.
  vkc::CommandBatch counted(*g_device, *g_allocator);
  CHECK(writer.record_count(counted, r.blocks, 64, 64).ok());
  CHECK(counted.submit().ok());
  d::ResidentBlocks shorn = r.blocks;
  CHECK(shorn.count > shorn.reference_count);
  shorn.count -= 1;
  CHECK(uncounted(finish(writer, shorn, p.reference)));
  d::ResidentBlocks fewer = r.blocks;
  fewer.reference_count -= 1;
  d::FrameReference shorter = p.reference;
  shorter.header.block_count -= 1;
  CHECK(uncounted(finish(writer, fewer, shorter)));

  // Bounded entries counted and exact ones finished, and the reverse: the
  // same count, so only the bound tells them apart.
  const Resident bounded = resident(p, 0u).value();
  CHECK(bounded.blocks.count == r.blocks.count);
  for (const auto& [counted_blocks, finished_blocks] :
       {std::pair(&bounded.blocks, &r.blocks),
        std::pair(&r.blocks, &bounded.blocks)}) {
    vkc::CommandBatch b(*g_device, *g_allocator);
    CHECK(writer.record_count(b, *counted_blocks, 64, 64).ok());
    CHECK(b.submit().ok());
    CHECK(uncounted(finish(writer, *finished_blocks, p.reference)));
  }

  // Ranks naming more survivors than the reference entries (a tail of
  // M + 1) or than the frame's blocks (word M planted at A + 1 over no
  // tail): the count clamps them, and finish refuses either.
  const std::uint32_t m = r.blocks.reference_count;
  const std::uint32_t a = r.blocks.count - m;
  CHECK(a + 1 <= m);
  auto survivors_refused = [&](const Resident& planted, std::uint32_t s,
                               std::uint32_t blocks) {
    vkc::CommandBatch b(*g_device, *g_allocator);
    CHECK(writer.record_count(b, planted.blocks, 64, 64).ok());
    CHECK(b.submit().ok());
    const vkc::Result<Bytes> written =
        finish(writer, planted.blocks, p.reference);
    const std::string says = "DeviceFrameWriter::finish: " + std::to_string(s) +
                             " survivors of " + std::to_string(m) +
                             " reference entries and " +
                             std::to_string(blocks) + " blocks";
    if (written.ok() || written.status().domain() != C::InvalidArgument ||
        written.status().message() != says) {
      std::fprintf(
          stderr, "%s: %s\n", says.c_str(),
          written.ok() ? "not refused" : written.status().message().c_str());
      return 1;
    }
    return 0;
  };
  CHECK(survivors_refused(resident(p, m + 1).value(), m + 1, a + m + 1) == 0);
  const Resident past_blocks = resident(p, 0u).value();
  CHECK(vr_test::write_back(*g_device, *g_allocator, *past_blocks.ranks,
                            std::vector<std::uint32_t>(m + 1, a + 1))
            .ok());
  CHECK(survivors_refused(past_blocks, a + 1, a) == 0);

  CHECK(device_write(writer, p, sizes(64, 0)).status().domain() ==
        C::InvalidArgument);
  p.reference.header.block_count += 1;
  CHECK(device_write(writer, p, {}).status().domain() == C::InvalidArgument);
  p.reference.header.block_count -= 1;
  p.type = d::FrameType::kIntra;  // no reference passed to finish
  CHECK(device_write(writer, p, {}).status().domain() == C::InvalidArgument);
  return 0;
}

int gpu_main(vr_test::GpuContext& gpu) {
  vkc::Result<std::unique_ptr<d::DeviceFrameWriter>> w =
      d::DeviceFrameWriter::create(gpu.device, gpu.allocator);
  CHECK(w.ok());
  vkc::Result<std::unique_ptr<d::DeviceFrameReader>> r =
      d::DeviceFrameReader::create(gpu.device, gpu.allocator);
  CHECK(r.ok());
  d::DeviceFrameWriter& writer = *w.value();
  d::DeviceFrameReader& reader = *r.value();
  g_device = &gpu.device;
  g_allocator = &gpu.allocator;
  if (matches_oracle_case(writer, reader) != 0) return 1;
  if (inter_case(writer, reader) != 0) return 1;
  if (golden_case(writer, reader) != 0) return 1;
  if (corruption_case(reader) != 0) return 1;
  if (inter_refusals_case(reader) != 0) return 1;
  if (writer_refusals_case(writer) != 0) return 1;
  if (segment_rule_case(writer, reader) != 0) return 1;
  if (parsed_frame_case(reader) != 0) return 1;
  if (verdict_case(reader) != 0) return 1;
  if (ranks_case(reader) != 0) return 1;
  if (refusals_case(gpu.device, gpu.allocator, writer) != 0) return 1;
  std::puts("codec device frame: OK");
  return 0;
}

}  // namespace

int main() { return vr_test::run_on_gpu(gpu_main); }
