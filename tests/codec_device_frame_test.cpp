// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// The device frame writer against the host's: every frame it writes must be
// write_intra_frame's, byte for byte -- the test the 2026-09-26 decision set
// for the GPU coder. Skips where no device is present.

#include <cstdint>
#include <cstdio>
#include <limits>
#include <vector>

#include "bitstream.hpp"
#include "codec_frames.hpp"
#include "device_frame_writer.hpp"
#include "volumetric_kit/recon/core/allocator.hpp"
#include "volumetric_kit/recon/core/command_batch.hpp"
#include "volumetric_kit/recon/core/compute_util.hpp"
#include "volumetric_kit/recon/core/device.hpp"
#include "volumetric_kit/recon/core/instance.hpp"

namespace vr = volumetric_kit::recon;
namespace codec = volumetric_kit::recon::codec;
namespace d = volumetric_kit::recon::codec::detail;
using codec_frames::make_frame;

#define CHECK(cond)                                                        \
  do {                                                                     \
    if (!(cond)) {                                                         \
      std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
      return 1;                                                            \
    }                                                                      \
  } while (0)

namespace {

// The host's frame and the device's, for one frame and segment size.
int same_bytes(d::DeviceFrameWriter& writer, const d::IntraFrame& frame,
               std::uint32_t segment_size) {
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
  return 0;
}

int matches_host_case(d::DeviceFrameWriter& writer) {
  // Block counts on and off a segment boundary, K odd and even up to the
  // whole transform, and segments of one block through more than the frame.
  for (std::size_t n : {std::size_t(0), std::size_t(1), std::size_t(63),
                        std::size_t(64), std::size_t(65), std::size_t(700)}) {
    for (std::uint32_t k : {1u, 20u, 64u, codec::kVoxelsPerBlock}) {
      for (std::uint32_t r : {1u, 16u, 64u, 1024u}) {
        CHECK(same_bytes(writer, make_frame(n, k, 31 * n + k), r) == 0);
      }
    }
  }
  // Steps across all of int32, and a segment of first blocks only.
  d::IntraFrame f = make_frame(0, 4, 1);
  constexpr std::int32_t kMax = std::numeric_limits<std::int32_t>::max();
  constexpr std::int32_t kMin = std::numeric_limits<std::int32_t>::min();
  f.coords = {{kMin, kMin, kMin}, {kMax, kMin, kMin}, {kMin, kMax, kMin},
              {0, 0, kMax},       {1, 0, kMax},       {kMax, kMax, kMax}};
  f.blocks.masks.assign(f.coords.size() * codec::kMaskWordsPerBlock, ~0u);
  f.blocks.coefficients.assign(f.coords.size() * 4, 0);
  CHECK(same_bytes(writer, f, 64) == 0);
  CHECK(same_bytes(writer, f, 1) == 0);
  // A smaller frame after larger ones reuses the retained buffers.
  CHECK(same_bytes(writer, make_frame(5, 64, 7), 64) == 0);
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

int refusals_case(vr::Device& device, vr::Allocator& allocator,
                  d::DeviceFrameWriter& writer) {
  d::IntraFrame f = make_frame(10, 8, 3);
  std::swap(f.coords[3], f.coords[4]);
  CHECK(!writer.write(f, {}).ok());
  d::FrameWriteOptions zero;
  zero.segment_size = 0;
  CHECK(!writer.write(make_frame(10, 8, 3), zero).ok());
  // finish with nothing counted refuses rather than coding stale steps.
  vr::Result<d::DeviceFrameWriter> fresh =
      d::DeviceFrameWriter::create(device, allocator);
  CHECK(fresh.ok());
  d::ResidentBlocks uncounted;
  uncounted.count = 10;
  uncounted.coefficient_count = 8;
  codec::CodecParams params;
  params.coefficient_count = 8;
  CHECK(!fresh.value().finish(uncounted, 0.005f, 0.04f, params).ok());
  return 0;
}

// The move rules: a moved-from writer is empty and refuses, a move-assign
// over a live one works, and a self-move leaves it intact.
int moves_case(vr::Device& device, vr::Allocator& allocator,
               d::DeviceFrameWriter& writer) {
  d::DeviceFrameWriter moved = std::move(writer);
  CHECK(moved.valid());
  CHECK(!writer.valid());  // NOLINT(bugprone-use-after-move): the source
  CHECK(!writer.write(make_frame(10, 8, 3), {}).ok());
  vr::Result<d::DeviceFrameWriter> live =
      d::DeviceFrameWriter::create(device, allocator);
  CHECK(live.ok());
  live.value() = std::move(moved);  // over a live writer
  CHECK(live.value().valid());
  CHECK(!moved.valid());  // NOLINT(bugprone-use-after-move)
  d::DeviceFrameWriter* alias = &live.value();
  live.value() = std::move(*alias);  // self-move, laundered past -Wself-move
  CHECK(live.value().valid());
  CHECK(same_bytes(live.value(), make_frame(70, 8, 5), 16) == 0);
  writer = std::move(live.value());
  CHECK(writer.valid());
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
  vr::Result<VkPhysicalDevice> gpu = instance.value().select_physical_device();
  if (!gpu) {
    std::fprintf(stderr, "no compute-capable device (%s); skipping\n",
                 gpu.status().message().c_str());
    return 0;
  }
  vr::Result<vr::Device> device =
      vr::Device::create(instance.value(), gpu.value(), {});
  CHECK(device.ok());
  vr::Result<vr::Allocator> allocator =
      vr::Allocator::create(instance.value().handle(), device.value());
  CHECK(allocator.ok());
  vr::Result<d::DeviceFrameWriter> writer =
      d::DeviceFrameWriter::create(device.value(), allocator.value());
  CHECK(writer.ok());
  if (matches_host_case(writer.value()) != 0) return 1;
  if (out_of_range_case(device.value(), allocator.value(), writer.value()) !=
      0) {
    return 1;
  }
  if (refusals_case(device.value(), allocator.value(), writer.value()) != 0) {
    return 1;
  }
  if (moves_case(device.value(), allocator.value(), writer.value()) != 0) {
    return 1;
  }
  std::puts("codec device frame: OK");
  return 0;
}
