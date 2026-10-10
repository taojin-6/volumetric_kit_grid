// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// The device frame writer and reader against the host's: every frame the
// writer writes must be write_intra_frame's, byte for byte, and every frame
// the reader reads must decode to read_intra_frame's blocks or be refused as
// it refuses it -- the test the 2026-09-26 decision set for the GPU coder.
// Skips where no device is present.

#include <cstdint>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

#include "bitstream.hpp"
#include "codec_frames.hpp"
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
using codec_frames::Lcg;
using codec_frames::make_frame;

namespace {

constexpr std::uint32_t kMaxBlocks = 1u << 20;

// The device the writer and reader run on, for the helpers below.
vkc::Device* g_device = nullptr;
vkc::Allocator* g_allocator = nullptr;

// `bytes` in a new device storage buffer.
vkc::Result<std::unique_ptr<vkc::Buffer>> upload(const void* bytes,
                                                 VkDeviceSize size) {
  VKC_ASSIGN(vkc::Buffer buffer,
             vkc::device_storage_buffer(*g_allocator, size));
  vkc::CommandBatch batch(*g_device, *g_allocator);
  VKC_TRY(batch.upload(buffer, 0, bytes, size));
  VKC_TRY(batch.submit());
  return std::make_unique<vkc::Buffer>(std::move(buffer));
}

// The device writer's bytes for a host frame: its blocks uploaded, counted
// and finished, as the encoder does with the transform's output.
vkc::Result<std::vector<std::uint8_t>> device_write(
    d::DeviceFrameWriter& writer, const d::IntraFrame& frame,
    const d::FrameWriteOptions& options) {
  const auto n = static_cast<std::uint32_t>(frame.coords.size());
  const std::uint32_t k = frame.blocks.params.coefficient_count;
  const std::uint32_t words = (k + 1) / 2;
  d::ResidentBlocks blocks;
  blocks.count = n;
  blocks.coefficient_count = k;
  std::unique_ptr<vkc::Buffer> list, masks, coefficients;
  if (n != 0) {
    std::vector<vr::volume::BlockIndex> entries(n);
    for (std::uint32_t i = 0; i < n; ++i) entries[i].coord = frame.coords[i];
    // Two int16 a word, each entry starting a word of its own.
    std::vector<std::uint32_t> packed(std::size_t(n) * words, 0);
    for (std::uint32_t i = 0; i < n; ++i) {
      for (std::uint32_t j = 0; j < k; ++j) {
        const auto v = static_cast<std::uint16_t>(
            frame.blocks.coefficients[std::size_t(i) * k + j]);
        packed[std::size_t(i) * words + j / 2] |= std::uint32_t(v)
                                                  << (16 * (j & 1));
      }
    }
    VKC_ASSIGN(list, upload(entries.data(),
                            entries.size() * sizeof(vr::volume::BlockIndex)));
    VKC_ASSIGN(masks,
               upload(frame.blocks.masks.data(),
                      frame.blocks.masks.size() * sizeof(std::uint32_t)));
    VKC_ASSIGN(coefficients,
               upload(packed.data(), packed.size() * sizeof(std::uint32_t)));
    blocks.list = list.get();
    blocks.masks = masks.get();
    blocks.coefficients = coefficients.get();
  }
  vkc::CommandBatch batch(*g_device, *g_allocator);
  VKC_TRY(writer.record_count(batch, blocks, options.segment_size));
  VKC_TRY(batch.submit());
  return writer.finish(blocks, frame.voxel_size, frame.blocks.trunc_dist,
                       frame.blocks.params);
}

// The device reader's decode of a frame, read back as the host's.
vkc::Result<d::IntraFrame> device_read(d::DeviceFrameReader& reader,
                                       const std::uint8_t* data,
                                       std::size_t size) {
  VKC_ASSIGN(const d::ParsedFrame parsed,
             d::parse_intra_frame(data, size, kMaxBlocks));
  const std::uint32_t n = parsed.header.block_count;
  const std::uint32_t k = parsed.header.params.coefficient_count;
  const std::uint32_t words = (k + 1) / 2;
  std::vector<std::uint32_t> packed(std::size_t(n) * words);
  d::IntraFrame frame;
  frame.voxel_size = parsed.header.voxel_size;
  frame.blocks.trunc_dist = parsed.header.trunc_dist;
  frame.blocks.params = parsed.header.params;
  frame.blocks.masks.resize(std::size_t(n) * codec::kMaskWordsPerBlock);
  vkc::CommandBatch batch(*g_device, *g_allocator);
  VKC_ASSIGN(const d::ResidentBlocks resident,
             reader.record_decode(batch, parsed));
  if (n != 0) {
    VKC_TRY(batch.readback(*resident.masks, 0,
                           frame.blocks.masks.size() * sizeof(std::uint32_t),
                           frame.blocks.masks.data()));
    VKC_TRY(batch.readback(*resident.coefficients, 0,
                           packed.size() * sizeof(std::uint32_t),
                           packed.data()));
    VKC_TRY(batch.submit());
  }
  VKC_TRY(reader.check());
  for (const vr::volume::BlockIndex& b : reader.blocks()) {
    frame.coords.push_back(b.coord);
  }
  frame.blocks.coefficients.resize(std::size_t(n) * k);
  for (std::size_t i = 0; i < n; ++i) {
    for (std::uint32_t j = 0; j < k; ++j) {
      frame.blocks.coefficients[i * k + j] = static_cast<std::int16_t>(
          packed[i * words + j / 2] >> (16 * (j & 1)));
    }
  }
  return frame;
}

// The host's decode of a frame and the device's: the same blocks, or the same
// refusal, which @p refusal receives.
int same_read(d::DeviceFrameReader& reader,
              const std::vector<std::uint8_t>& bytes,
              std::string* refusal = nullptr) {
  const vkc::Result<d::IntraFrame> host =
      d::read_intra_frame(bytes.data(), bytes.size(), kMaxBlocks);
  const vkc::Result<d::IntraFrame> device =
      device_read(reader, bytes.data(), bytes.size());
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
  const vkc::Result<std::vector<std::uint8_t>> host =
      d::write_intra_frame(frame, options);
  const vkc::Result<std::vector<std::uint8_t>> device =
      device_write(writer, frame, options);
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

// The golden frames: the device writes their pinned bytes, so every machine
// writes the same frame, and reads them as the host does.
int golden_case(d::DeviceFrameWriter& writer, d::DeviceFrameReader& reader) {
  for (const codec_frames::GoldenFrame& g : codec_frames::kGoldenFrames) {
    d::FrameWriteOptions options;
    options.segment_size = g.segment_size;
    const vkc::Result<std::vector<std::uint8_t>> bytes =
        device_write(writer, codec_frames::golden_frame(g), options);
    CHECK(bytes.ok());
    CHECK(bytes.value().size() == g.size);
    CHECK(vr_test::fnv1a(bytes.value()) == g.hash);
    CHECK(same_read(reader, bytes.value()) == 0);
  }
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
    const vkc::Result<d::ParsedFrame> parsed =
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
int out_of_range_case(vkc::Device& device, vkc::Allocator& allocator,
                      d::DeviceFrameWriter& writer) {
  const vr::volume::BlockIndex block{};
  const std::uint32_t masks[codec::kMaskWordsPerBlock] = {};
  const std::uint32_t coefficient = 0x8000u;  // K = 1
  vkc::Result<vkc::Buffer> list = vkc::device_storage_buffer(allocator, 16);
  vkc::Result<vkc::Buffer> mask = vkc::device_storage_buffer(allocator, 64);
  vkc::Result<vkc::Buffer> coeff = vkc::device_storage_buffer(allocator, 4);
  CHECK(list.ok() && mask.ok() && coeff.ok());
  d::ResidentBlocks blocks;
  blocks.list = &list.value();
  blocks.masks = &mask.value();
  blocks.coefficients = &coeff.value();
  blocks.count = 1;
  blocks.coefficient_count = 1;
  vkc::CommandBatch batch(device, allocator);
  CHECK(batch.upload(list.value(), 0, &block, sizeof(block)).ok());
  CHECK(batch.upload(mask.value(), 0, masks, sizeof(masks)).ok());
  CHECK(batch.upload(coeff.value(), 0, &coefficient, 4).ok());
  CHECK(writer.record_count(batch, blocks, 64).ok());
  CHECK(batch.submit().ok());
  codec::CodecParams params;
  params.coefficient_count = 1;
  vkc::Result<std::vector<std::uint8_t>> frame =
      writer.finish(blocks, 0.005f, 0.04f, params);
  CHECK(frame.ok());
  vkc::Result<d::IntraFrame> read =
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
  const vkc::Result<d::IntraFrame> refused =
      device_read(reader, long_segment.data(), long_segment.size());
  CHECK(!refused.ok());
  CHECK(refused.status().domain() == vkc::Status::Code::InvalidArgument);
  // The header's segment size alone does not count: this frame's one
  // segment holds kLimit blocks.
  const std::vector<std::uint8_t> at_limit =
      d::write_intra_frame(make_frame(kLimit, 1, 11), whole).value();
  CHECK(same_read(reader, at_limit) == 0);
  return 0;
}

int refusals_case(vkc::Device& device, vkc::Allocator& allocator) {
  // finish with nothing counted refuses rather than coding stale steps.
  vkc::Result<std::unique_ptr<d::DeviceFrameWriter>> fresh =
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
  if (golden_case(writer, reader) != 0) return 1;
  if (corruption_case(reader) != 0) return 1;
  if (segment_limit_case(reader) != 0) return 1;
  if (out_of_range_case(gpu.device, gpu.allocator, writer) != 0) {
    return 1;
  }
  if (refusals_case(gpu.device, gpu.allocator) != 0) return 1;
  std::puts("codec device frame: OK");
  return 0;
}

}  // namespace

int main() { return vr_test::run_on_gpu(gpu_main); }
