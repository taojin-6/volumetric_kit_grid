// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// The device frame writer and reader against the host's: every frame the
// writer writes, intra or P, must be write_frame's, byte for byte, or be
// refused as it refuses it, and every frame the reader reads must decode to
// read_frame's entries or be refused as it refuses it, with the same code and
// message -- the test the 2026-09-26 decision set for the GPU coder. Skips
// where no device is present.

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "bitstream.hpp"
#include "codec_frames.hpp"
#include "codec_resident.hpp"
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
using codec_frames::extreme_frame;
using codec_frames::inter_frame;
using codec_frames::Lcg;
using codec_frames::make_frame;
using codec_frames::reference_of;
using codec_frames::Resident;
using codec_frames::successor;
using C = vkc::Status::Code;
using Bytes = std::vector<std::uint8_t>;

namespace {

constexpr std::uint32_t kMaxBlocks = 1u << 20;

// The device the writer and reader run on, for the helpers below.
vkc::Device* g_device = nullptr;
vkc::Allocator* g_allocator = nullptr;

d::FrameWriteOptions sizes(std::uint32_t r, std::uint32_t u) {
  d::FrameWriteOptions o;
  o.segment_size = r;
  o.update_size = u;
  return o;
}

// A host frame's entries in device buffers, as the encoder holds the
// transform's output.
vkc::Result<Resident> resident(const d::Frame& frame) {
  return codec_frames::resident({*g_device, *g_allocator}, frame.blocks,
                                frame.modes, frame.coords);
}

// The device writer's bytes for a host frame: its entries uploaded, counted
// and finished, as the encoder does with the transform's output.
vkc::Result<Bytes> device_write(d::DeviceFrameWriter& writer,
                                const d::Frame& frame,
                                const d::FrameWriteOptions& options) {
  VKC_ASSIGN(const Resident r, resident(frame));
  vkc::CommandBatch batch(*g_device, *g_allocator);
  VKC_TRY(writer.record_count(batch, r.blocks, options.segment_size,
                              options.update_size));
  VKC_TRY(batch.submit());
  return writer.finish(
      r.blocks, frame.voxel_size, frame.blocks.trunc_dist, frame.blocks.params,
      frame.type == d::FrameType::kInter ? &frame.reference : nullptr);
}

// The device reader's decode of a frame, read back as the host's: a skip or
// remove entry's payload, which the device leaves as it was, zero as the
// host's.
vkc::Result<d::Frame> device_read(d::DeviceFrameReader& reader,
                                  const std::uint8_t* data, std::size_t size,
                                  const d::FrameReference* reference,
                                  std::uint32_t max_blocks = kMaxBlocks) {
  VKC_ASSIGN(const d::ParsedFrame parsed,
             d::parse_frame(data, size, max_blocks, reference));
  const std::uint32_t k = parsed.header.params.coefficient_count;
  const std::uint32_t words = d::coefficient_words(k);
  d::Frame frame;
  frame.type = parsed.header.type;
  frame.voxel_size = parsed.header.voxel_size;
  frame.reference = parsed.reference;
  frame.blocks.trunc_dist = parsed.header.trunc_dist;
  frame.blocks.params = parsed.header.params;
  vkc::CommandBatch batch(*g_device, *g_allocator);
  VKC_ASSIGN(const d::ResidentBlocks resident,
             reader.record_decode(batch, parsed));
  const std::uint32_t entries = resident.count;
  std::vector<std::uint32_t> packed(std::size_t(entries) * words);
  frame.blocks.masks.resize(std::size_t(entries) * codec::kMaskWordsPerBlock);
  if (entries != 0) {
    VKC_TRY(batch.readback(*resident.masks, 0,
                           frame.blocks.masks.size() * sizeof(std::uint32_t),
                           frame.blocks.masks.data()));
    VKC_TRY(batch.readback(*resident.coefficients, 0,
                           packed.size() * sizeof(std::uint32_t),
                           packed.data()));
    VKC_TRY(batch.submit());
  }
  VKC_TRY(reader.check());
  frame.modes = reader.modes();
  for (const vr::volume::BlockIndex& b : reader.blocks()) {
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

// The host's decode of a frame and the device's: the same entries, or the
// same refusal -- code and message -- which @p refusal receives.
int same_read(d::DeviceFrameReader& reader, const Bytes& bytes,
              const d::FrameReference* reference = nullptr,
              std::string* refusal = nullptr,
              std::uint32_t max_blocks = kMaxBlocks) {
  const vkc::Result<d::Frame> host =
      d::read_frame(bytes.data(), bytes.size(), max_blocks, reference);
  const vkc::Result<d::Frame> device =
      device_read(reader, bytes.data(), bytes.size(), reference, max_blocks);
  if (host.ok() != device.ok() ||
      (!host.ok() && host.status().message() != device.status().message())) {
    std::fprintf(stderr, "host: %s; device: %s\n",
                 host.ok() ? "OK" : host.status().message().c_str(),
                 device.ok() ? "OK" : device.status().message().c_str());
    return 1;
  }
  if (!host.ok()) {
    CHECK(host.status().domain() == device.status().domain());
    if (refusal != nullptr) *refusal = host.status().message();
    return 0;
  }
  if (refusal != nullptr) refusal->clear();
  const d::Frame& a = host.value();
  const d::Frame& b = device.value();
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

// The host's frame and the device's, for one frame and its sizes, and the
// device's decode of it.
int same_bytes(d::DeviceFrameWriter& writer, d::DeviceFrameReader& reader,
               const d::Frame& frame, const d::FrameWriteOptions& options) {
  const vkc::Result<Bytes> host = d::write_frame(frame, options);
  const vkc::Result<Bytes> device = device_write(writer, frame, options);
  CHECK(host.ok());
  if (!device.ok()) {
    std::fprintf(stderr, "device write: %s\n",
                 device.status().message().c_str());
  }
  CHECK(device.ok());
  if (device.value() != host.value()) {
    std::fprintf(stderr,
                 "differs: %zu modes, %zu blocks, K %u, R %u, U %u (%zu vs %zu "
                 "bytes)\n",
                 frame.modes.size(), frame.coords.size(),
                 frame.blocks.params.coefficient_count, options.segment_size,
                 options.update_size, device.value().size(),
                 host.value().size());
  }
  CHECK(device.value() == host.value());
  return same_read(
      reader, host.value(),
      frame.type == d::FrameType::kInter ? &frame.reference : nullptr);
}

int matches_host_case(d::DeviceFrameWriter& writer,
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
  const d::Frame f = extreme_frame();
  CHECK(same_bytes(writer, reader, f, sizes(64, 64)) == 0);
  CHECK(same_bytes(writer, reader, f, sizes(1, 64)) == 0);
  // A smaller frame after larger ones reuses the retained buffers.
  CHECK(same_bytes(writer, reader, make_frame(5, 64, 7), sizes(64, 64)) == 0);
  return 0;
}

// P-frames: the worked example's pinned bytes, at U = 1 too; the matrix
// (codec_frames.hpp) at every update and segment size, every mode, K and
// empty case among them; and a P-frame of no entries at all.
int inter_case(d::DeviceFrameWriter& writer, d::DeviceFrameReader& reader) {
  const d::FrameReference example =
      reference_of(d::write_frame(codec_frames::example_reference()).value());
  const Bytes bytes = codec_frames::example_bytes(example.hash);
  const d::Frame p =
      d::read_frame(bytes.data(), bytes.size(), 3, &example).value();
  CHECK(device_write(writer, p, {}).value() == bytes);
  CHECK(same_read(reader, bytes, &example) == 0);
  CHECK(same_bytes(writer, reader, p, sizes(64, 1)) == 0);

  int written = 0;
  for (const codec_frames::InterCode& c : codec_frames::inter_matrix()) {
    for (std::uint32_t u : codec_frames::kInterSizes) {
      for (std::uint32_t r : codec_frames::kInterSizes) {
        if (same_bytes(writer, reader, c.p, sizes(r, u)) != 0) {
          std::fprintf(stderr, "%s: U %u, R %u\n", c.what.c_str(), u, r);
          return 1;
        }
        ++written;
      }
    }
  }
  std::printf("P-frames: the worked example and %d of the matrix\n", written);

  // M = 0, A = 0: REFERENCE and an empty mode table, no segment.
  const d::Frame empty = make_frame(0, 8, 1);
  const d::FrameReference id = reference_of(d::write_frame(empty).value());
  CHECK(same_bytes(writer, reader, inter_frame(id, empty, empty), {}) == 0);
  return 0;
}

// The golden frames: the device writes their pinned bytes, so every machine
// writes the same frame, and reads them as the host does.
int golden_case(d::DeviceFrameWriter& writer, d::DeviceFrameReader& reader) {
  for (const codec_frames::GoldenFrame& g : codec_frames::kGoldenFrames) {
    const vkc::Result<Bytes> bytes = device_write(
        writer, codec_frames::golden_frame(g), sizes(g.segment_size, 64));
    CHECK(bytes.ok());
    CHECK(bytes.value().size() == g.size);
    CHECK(vr_test::fnv1a(bytes.value()) == g.hash);
    CHECK(same_read(reader, bytes.value()) == 0);
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
// host and the device against @p reference, into @p verdicts.
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

// Corrupt payloads: the device refuses each one the host refuses, with the
// same message, and decodes the rest to the same blocks. Every verdict the
// segments have is reached.
int corruption_case(d::DeviceFrameReader& reader) {
  Lcg rng{29};
  Verdicts v;
  CHECK(corrupt_payloads(
            reader,
            d::write_frame(make_frame(300, 32, 13), sizes(16, 64)).value(),
            nullptr, 400, rng, v) == 0);
  CHECK(corrupt_payloads(reader,
                         d::write_frame(extreme_frame(), sizes(2, 64)).value(),
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
  const d::Frame reference = make_frame(60, 4, 31);
  id = reference_of(d::write_frame(reference).value());
  const d::Frame kept = successor(reference, 32, 100, 400, 0);
  const d::Frame added = extreme_frame();
  std::map<vr::Vec3i, codec_frames::Payload, codec_frames::ZyxLess> blocks;
  for (std::size_t i = 0; i < kept.coords.size(); ++i) {
    blocks.emplace(kept.coords[i], codec_frames::payload_at(kept.blocks, i));
  }
  for (std::size_t i = 0; i < added.coords.size(); ++i) {
    blocks.emplace(added.coords[i], codec_frames::payload_at(added.blocks, i));
  }
  d::Frame target = make_frame(0, 4, 1);
  for (const auto& [c, payload] : blocks) {
    target.coords.push_back(c);
    codec_frames::append_payload(target.blocks, payload);
  }
  return d::write_frame(inter_frame(id, reference, target), sizes(2, 5))
      .value();
}

// P-frame refusals: every reader refusal of C2's edit list, each one edit
// from a valid P-frame, and 800 random payload corruptions of two P-frames,
// each refused by the device with the host's code and message, or decoded to
// the host's entries.
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
  const d::Frame reference = make_frame(30, 8, 51);
  const d::FrameReference id = reference_of(d::write_frame(reference).value());
  const d::Frame target = successor(reference, 52, 150, 400, 5);
  auto first = [](const d::Frame& p, d::BlockMode mode) {
    const auto it = std::find(p.modes.begin(), p.modes.end(), mode);
    return static_cast<std::size_t>(it - p.modes.begin());
  };
  // A skip spelled as zero: the frame writes a zero entry too.
  const std::size_t spelled =
      first(inter_frame(id, reference, target), d::BlockMode::kSkip);
  const d::Frame good =
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
    d::Frame frame;
  };
  std::vector<Edit> edits;
  d::Frame f = good;
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
  for (const Edit& e : edits) {
    const vkc::Result<Bytes> host = d::write_frame(e.frame);
    const vkc::Result<Bytes> device = device_write(writer, e.frame, {});
    if (host.ok() || device.ok() || host.status().domain() != e.code ||
        device.status().domain() != e.code ||
        host.status().message() != device.status().message()) {
      std::fprintf(stderr, "%s: host %s; device %s\n", e.what,
                   host.ok() ? "OK" : host.status().message().c_str(),
                   device.ok() ? "OK" : device.status().message().c_str());
      return 1;
    }
  }
  CHECK(d::write_frame(edits[2].frame).status().message() ==
        "write_frame: mode 7 is not a mode");
  CHECK(d::write_frame(edits.back().frame).status().message() ==
        "write_frame: trunc_dist must be finite and positive");
  f = good;
  f.blocks.coefficients[skip * k] = low;
  CHECK(device_write(writer, f, {}).value() == d::write_frame(good).value());
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
  const d::Frame many = make_frame(kLimit, 1, 11);
  CHECK(same_bytes(writer, reader, many, sizes(kLimit, 64)) == 0);
  const d::FrameReference id = reference_of(d::write_frame(many).value());
  const d::Frame skips = inter_frame(id, many, many);
  CHECK(same_bytes(writer, reader, skips, sizes(64, kLimit)) == 0);
  for (const d::FrameWriteOptions& o :
       {sizes(kLimit + 1, 64), sizes(64, kLimit + 1)}) {
    CHECK(d::write_frame(skips, o).status().domain() == C::InvalidArgument);
    CHECK(device_write(writer, skips, o).status().domain() ==
          C::InvalidArgument);
  }

  // One entry more named in the header, and in REFERENCE: the same segments,
  // so the rule alone refuses them.
  const std::string past = std::to_string(kLimit + 1) + " is not in";
  Bytes intra = d::write_frame(many, sizes(kLimit, 64)).value();
  codec_frames::put_u32(intra, 36, kLimit + 1);
  std::string why;
  CHECK(same_read(reader, intra, nullptr, &why) == 0);
  CHECK(why.find("segment size " + past) != std::string::npos);
  const Bytes p = d::write_frame(skips, sizes(64, kLimit)).value();
  std::vector<codec_frames::Section> s = codec_frames::sections_of(p);
  codec_frames::put_u32(s[0].body, 12, kLimit + 1);
  CHECK(same_read(reader, codec_frames::assemble(p, s), &id, &why) == 0);
  CHECK(why.find("update size " + past) != std::string::npos);
  return 0;
}

// The reader refuses a parsed frame of a type past the P-frame, which the
// host's decode_frame refuses too, recording nothing.
int frame_type_case(d::DeviceFrameReader& reader) {
  const d::Frame reference = make_frame(100, 16, 12);
  const Bytes intra = d::write_frame(reference).value();
  CHECK(same_read(reader, intra) == 0);
  auto decoded = [&reader] {
    std::vector<vr::Vec3i> coords;
    for (const vr::volume::BlockIndex& b : reader.blocks()) {
      coords.push_back(b.coord);
    }
    return coords;
  };
  const std::vector<vr::Vec3i> before = decoded();
  CHECK(before == reference.coords);
  d::ParsedFrame typed =
      d::parse_frame(intra.data(), intra.size(), kMaxBlocks, nullptr).value();
  typed.header.type = static_cast<d::FrameType>(2);
  CHECK(d::decode_frame(typed).status().domain() == C::Unsupported);
  vkc::CommandBatch batch(*g_device, *g_allocator);
  CHECK(reader.record_decode(batch, typed).status().domain() == C::Unsupported);
  CHECK(batch.submit().ok());
  CHECK(decoded() == before);
  return 0;
}

// The writer's own refusals: entries it cannot walk -- more reference
// entries than entries, or without the modes or list they need -- each
// counting nothing, so finish refuses, rather than coding stale steps; a
// finish whose blocks have another reference count than were counted;
// reference entries without an update size; and a reference that does not
// hold the blocks' reference entries, or none for blocks that have them.
int refusals_case(vkc::Device& device, vkc::Allocator& allocator,
                  d::DeviceFrameWriter& writer) {
  const d::Frame reference = make_frame(20, 8, 61);
  const d::FrameReference id = reference_of(d::write_frame(reference).value());
  d::Frame p =
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
  CHECK(batch.submit().ok());
  CHECK(uncounted(finish(*fresh.value(), r.blocks, p.reference)));

  // Counted with M reference entries, finished with M - 1 against a
  // reference of M - 1 blocks.
  vkc::CommandBatch counted(*g_device, *g_allocator);
  CHECK(writer.record_count(counted, r.blocks, 64, 64).ok());
  CHECK(counted.submit().ok());
  d::ResidentBlocks fewer = r.blocks;
  fewer.reference_count -= 1;
  d::FrameReference shorter = p.reference;
  shorter.header.block_count -= 1;
  CHECK(uncounted(finish(writer, fewer, shorter)));

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
  if (matches_host_case(writer, reader) != 0) return 1;
  if (inter_case(writer, reader) != 0) return 1;
  if (golden_case(writer, reader) != 0) return 1;
  if (corruption_case(reader) != 0) return 1;
  if (inter_refusals_case(reader) != 0) return 1;
  if (writer_refusals_case(writer) != 0) return 1;
  if (segment_rule_case(writer, reader) != 0) return 1;
  if (frame_type_case(reader) != 0) return 1;
  if (refusals_case(gpu.device, gpu.allocator, writer) != 0) return 1;
  std::puts("codec device frame: OK");
  return 0;
}

}  // namespace

int main() { return vr_test::run_on_gpu(gpu_main); }
