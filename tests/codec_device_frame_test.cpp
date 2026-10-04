// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// The device frame writer and reader against the host's: every frame the
// writer writes must be write_intra_frame's, byte for byte, and every frame
// the reader reads must decode to read_intra_frame's blocks or be refused as
// it refuses it -- the test the 2026-09-26 decision set for the GPU coder.
// Skips where no device is present.

#include <cstdint>
#include <cstdio>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "bitstream.hpp"
#include "codec_frames.hpp"
#include "device_frame_reader.hpp"
#include "device_frame_writer.hpp"
#include "volumetric_kit/recon/core/allocator.hpp"
#include "volumetric_kit/recon/core/command_batch.hpp"
#include "volumetric_kit/recon/core/compute_util.hpp"
#include "volumetric_kit/recon/core/device.hpp"
#include "volumetric_kit/recon/core/instance.hpp"

namespace vr = volumetric_kit::recon;
namespace codec = volumetric_kit::recon::codec;
namespace d = volumetric_kit::recon::codec::detail;
using codec_frames::Lcg;
using codec_frames::make_frame;

#define CHECK(cond)                                                        \
  do {                                                                     \
    if (!(cond)) {                                                         \
      std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
      return 1;                                                            \
    }                                                                      \
  } while (0)

namespace {

constexpr std::uint32_t kMaxBlocks = 1u << 20;

// The host's decode of a frame and the device's: the same blocks, or the same
// refusal, which @p refusal receives.
int same_read(d::DeviceFrameReader& reader,
              const std::vector<std::uint8_t>& bytes,
              std::string* refusal = nullptr) {
  const vr::Result<d::IntraFrame> host =
      d::read_intra_frame(bytes.data(), bytes.size(), kMaxBlocks);
  const vr::Result<d::IntraFrame> device =
      reader.read(bytes.data(), bytes.size(), kMaxBlocks);
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
  const d::IntraFrame& a = host.value();
  const d::IntraFrame& b = device.value();
  CHECK(a.voxel_size == b.voxel_size);
  CHECK(a.coords == b.coords);
  CHECK(a.blocks.masks == b.blocks.masks);
  CHECK(a.blocks.coefficients == b.blocks.coefficients);
  return 0;
}

// The host's frame and the device's, for one frame and segment size, and the
// device's decode of it.
int same_bytes(d::DeviceFrameWriter& writer, d::DeviceFrameReader& reader,
               const d::IntraFrame& frame, std::uint32_t segment_size) {
  d::FrameWriteOptions options;
  options.segment_size = segment_size;
  const vr::Result<std::vector<std::uint8_t>> host =
      d::write_intra_frame(frame, options);
  const vr::Result<std::vector<std::uint8_t>> device =
      writer.write(frame, options);
  CHECK(host.ok());
  if (!device.ok()) {
    std::fprintf(stderr, "device write: %s\n",
                 device.status().message().c_str());
  }
  CHECK(device.ok());
  if (device.value() != host.value()) {
    std::fprintf(stderr, "differs: %zu blocks, K %u, R %u (%zu vs %zu bytes)\n",
                 frame.coords.size(), frame.blocks.params.coefficient_count,
                 segment_size, device.value().size(), host.value().size());
  }
  CHECK(device.value() == host.value());
  return same_read(reader, host.value());
}

d::IntraFrame extreme_frame() {
  d::IntraFrame f = make_frame(0, 4, 1);
  constexpr std::int32_t kMax = std::numeric_limits<std::int32_t>::max();
  constexpr std::int32_t kMin = std::numeric_limits<std::int32_t>::min();
  f.coords = {{kMin, kMin, kMin}, {kMax, kMin, kMin}, {kMin, kMax, kMin},
              {0, 0, kMax},       {1, 0, kMax},       {kMax, kMax, kMax}};
  f.blocks.masks.assign(f.coords.size() * codec::kMaskWordsPerBlock, ~0u);
  f.blocks.coefficients.assign(f.coords.size() * 4, 0);
  return f;
}

int matches_host_case(d::DeviceFrameWriter& writer,
                      d::DeviceFrameReader& reader) {
  // Block counts on and off a segment boundary, K odd and even up to the
  // whole transform, and segments of one block through more than the frame.
  for (std::size_t n : {std::size_t(0), std::size_t(1), std::size_t(63),
                        std::size_t(64), std::size_t(65), std::size_t(700)}) {
    for (std::uint32_t k : {1u, 20u, 64u, codec::kVoxelsPerBlock}) {
      for (std::uint32_t r : {1u, 16u, 64u, 1024u}) {
        CHECK(same_bytes(writer, reader, make_frame(n, k, 31 * n + k), r) == 0);
      }
    }
  }
  // Steps across all of int32, and a segment of first blocks only.
  const d::IntraFrame f = extreme_frame();
  CHECK(same_bytes(writer, reader, f, 64) == 0);
  CHECK(same_bytes(writer, reader, f, 1) == 0);
  // A smaller frame after larger ones reuses the retained buffers.
  CHECK(same_bytes(writer, reader, make_frame(5, 64, 7), 64) == 0);
  return 0;
}

// Corrupt payloads: the device refuses each one the host refuses, with the
// same message, and decodes the rest to the same blocks. Every verdict the
// segments have is reached.
int corruption_case(d::DeviceFrameReader& reader) {
  d::FrameWriteOptions sixteen;
  sixteen.segment_size = 16;
  d::FrameWriteOptions two;
  two.segment_size = 2;
  const std::vector<std::uint8_t> frames[] = {
      d::write_intra_frame(make_frame(300, 32, 13), sixteen).value(),
      d::write_intra_frame(extreme_frame(), two).value()};
  Lcg rng{29};
  int decoded = 0;
  int overflow = 0;
  int corrupt = 0;
  int order = 0;
  for (const std::vector<std::uint8_t>& good : frames) {
    const vr::Result<d::ParsedFrame> parsed =
        d::parse_intra_frame(good.data(), good.size(), kMaxBlocks);
    CHECK(parsed.ok());
    const auto payload =
        static_cast<std::uint32_t>(parsed.value().payload - good.data());
    for (int trial = 0; trial < 400; ++trial) {
      std::vector<std::uint8_t> b = good;
      const std::uint32_t edits = 1 + rng.below(3);
      for (std::uint32_t e = 0; e < edits; ++e) {
        b[payload + rng.below(std::uint32_t(b.size()) - payload)] ^=
            static_cast<std::uint8_t>(1 + rng.below(255));
      }
      std::string why;
      CHECK(same_read(reader, b, &why) == 0);
      decoded += why.empty() ? 1 : 0;
      overflow += why.find("outside int32") != std::string::npos ? 1 : 0;
      corrupt += why.find("is corrupt") != std::string::npos ? 1 : 0;
      order += why.find("does not start after") != std::string::npos ? 1 : 0;
    }
  }
  std::printf(
      "corrupt payloads: %d decoded, %d overflow, %d corrupt, %d "
      "out of order\n",
      decoded, overflow, corrupt, order);
  CHECK(decoded > 0 && overflow > 0 && corrupt > 0 && order > 0);
  return 0;
}

// -32768, which the transform never writes but a rejected entry's stale
// word can hold, is clamped into the model's 16 classes when counted: the
// frame codes -32767 rather than counting past its table.
int out_of_range_case(vr::Device& device, vr::Allocator& allocator,
                      d::DeviceFrameWriter& writer) {
  const vr::volume::BlockIndex block{};
  const std::uint32_t masks[codec::kMaskWordsPerBlock] = {};
  const std::uint32_t coefficient = 0x8000u;  // K = 1
  vr::Result<vr::Buffer> list = vr::device_storage_buffer(allocator, 16);
  vr::Result<vr::Buffer> mask = vr::device_storage_buffer(allocator, 64);
  vr::Result<vr::Buffer> coeff = vr::device_storage_buffer(allocator, 4);
  CHECK(list.ok() && mask.ok() && coeff.ok());
  d::ResidentBlocks blocks;
  blocks.list = &list.value();
  blocks.masks = &mask.value();
  blocks.coefficients = &coeff.value();
  blocks.count = 1;
  blocks.coefficient_count = 1;
  vr::CommandBatch batch(device, allocator);
  CHECK(batch.upload(list.value(), 0, &block, sizeof(block)).ok());
  CHECK(batch.upload(mask.value(), 0, masks, sizeof(masks)).ok());
  CHECK(batch.upload(coeff.value(), 0, &coefficient, 4).ok());
  CHECK(writer.record_count(batch, blocks, 64).ok());
  CHECK(batch.submit().ok());
  codec::CodecParams params;
  params.coefficient_count = 1;
  vr::Result<std::vector<std::uint8_t>> frame =
      writer.finish(blocks, 0.005f, 0.04f, params);
  CHECK(frame.ok());
  vr::Result<d::IntraFrame> read =
      d::read_intra_frame(frame.value().data(), frame.value().size(), 1);
  CHECK(read.ok());
  CHECK(read.value().blocks.coefficients[0] == -codec::kMaxQuantizedMagnitude);
  return 0;
}

// One invocation decodes a whole segment, so the reader refuses segments
// longer than kMaxDeviceDecodeSegmentSize, which the host decodes, and
// decodes one of exactly that length.
int segment_limit_case(d::DeviceFrameReader& reader) {
  constexpr std::uint32_t kLimit = codec::kMaxDeviceDecodeSegmentSize;
  d::FrameWriteOptions whole;
  whole.segment_size = kLimit + 1;
  const std::vector<std::uint8_t> long_segment =
      d::write_intra_frame(make_frame(kLimit + 1, 1, 11), whole).value();
  CHECK(
      d::read_intra_frame(long_segment.data(), long_segment.size(), kMaxBlocks)
          .ok());
  const vr::Result<d::IntraFrame> refused =
      reader.read(long_segment.data(), long_segment.size(), kMaxBlocks);
  CHECK(!refused.ok());
  CHECK(refused.status().domain() == vr::Status::Code::InvalidArgument);
  // The header's segment size alone does not count: this frame's one
  // segment holds kLimit blocks.
  const std::vector<std::uint8_t> at_limit =
      d::write_intra_frame(make_frame(kLimit, 1, 11), whole).value();
  CHECK(same_read(reader, at_limit) == 0);
  return 0;
}

int refusals_case(vr::Device& device, vr::Allocator& allocator,
                  d::DeviceFrameWriter& writer) {
  d::IntraFrame f = make_frame(10, 8, 3);
  std::swap(f.coords[3], f.coords[4]);
  CHECK(!writer.write(f, {}).ok());
  d::FrameWriteOptions zero;
  zero.segment_size = 0;
  CHECK(!writer.write(make_frame(10, 8, 3), zero).ok());
  // finish with nothing counted refuses rather than coding stale steps.
  vr::Result<std::unique_ptr<d::DeviceFrameWriter>> fresh =
      d::DeviceFrameWriter::create(device, allocator);
  CHECK(fresh.ok());
  d::ResidentBlocks uncounted;
  uncounted.count = 10;
  uncounted.coefficient_count = 8;
  codec::CodecParams params;
  params.coefficient_count = 8;
  CHECK(!fresh.value()->finish(uncounted, 0.005f, 0.04f, params).ok());
  return 0;
}

}  // namespace

int main() {
  vr::Result<vr::Instance> instance = vr::Instance::create({});
  if (!instance) {
    std::fprintf(stderr, "no Vulkan instance (%s); skipping\n",
                 instance.status().message().c_str());
    return 0;
  }
  vr::Result<vr::PhysicalDeviceInfo> gpu =
      instance.value().select_physical_device(vr::device_requirements());
  if (!gpu) {
    std::fprintf(stderr, "no compute-capable device (%s); skipping\n",
                 gpu.status().message().c_str());
    return 0;
  }
  vr::Result<vr::Device> device = vr::Device::create(
      instance.value(), gpu.value(), vr::device_requirements());
  CHECK(device.ok());
  vr::Result<vr::Allocator> allocator =
      vr::Allocator::create(instance.value().handle(), device.value());
  CHECK(allocator.ok());
  vr::Result<std::unique_ptr<d::DeviceFrameWriter>> w =
      d::DeviceFrameWriter::create(device.value(), allocator.value());
  CHECK(w.ok());
  vr::Result<std::unique_ptr<d::DeviceFrameReader>> r =
      d::DeviceFrameReader::create(device.value(), allocator.value());
  CHECK(r.ok());
  d::DeviceFrameWriter& writer = *w.value();
  d::DeviceFrameReader& reader = *r.value();
  if (matches_host_case(writer, reader) != 0) return 1;
  if (corruption_case(reader) != 0) return 1;
  if (segment_limit_case(reader) != 0) return 1;
  if (out_of_range_case(device.value(), allocator.value(), writer) != 0) {
    return 1;
  }
  if (refusals_case(device.value(), allocator.value(), writer) != 0) return 1;
  std::puts("codec device frame: OK");
  return 0;
}
