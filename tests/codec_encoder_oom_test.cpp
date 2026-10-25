// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// A refused encode must not publish a reference or consume the GOP/force
// request. A refused back recording must keep the front's counts for retry.

#include <cstdio>
#include <cstring>
#include <new>
#include <optional>

#include "codec_fixture.hpp"
#include "gpu_test.hpp"
#include "test_allocation_failure.hpp"
#include "volumetric_kit/recon/codec/decoder.hpp"
#include "volumetric_kit/recon/codec/encoder.hpp"

namespace codec = volumetric_kit::recon::codec;
using namespace codec_fixture;

namespace {

using Code = vkc::Status::Code;

int returned_bytes_case(Gpu& gpu, vol::VoxelBlockGrid& a,
                        vol::VoxelBlockGrid& b, bool forced) {
  codec::EncoderConfig config;
  config.intra_period = 3;
  auto measured = codec::Encoder::create(gpu.device, gpu.allocator, config);
  auto encoder = codec::Encoder::create(gpu.device, gpu.allocator, config);
  auto decoder = codec::Decoder::create(gpu.device, gpu.allocator);
  auto output = make_grid(gpu);
  CHECK(measured.ok() && encoder.ok() && decoder.ok() && output.ok());
  CHECK(measured->encode(a).ok());
  const auto first = encoder->encode(a);
  CHECK(first.ok());
  CHECK(decoder->decode(first->data(), first->size(), *output).ok());
  if (forced) {
    measured->force_intra();
    encoder->force_intra();
  }

  // The last allocation joins the head and payload into the returned vector.
  // Capture its call site as well as its size: the driver and validation
  // layer may allocate that same size earlier in the encode.
  auto& failure = vr_test::allocation_failure;
  failure = {};
  failure.measure = true;
  const auto expected = measured->encode(b);
  const auto captured = failure;
  failure = {};
  CHECK(expected.ok());
  CHECK(captured.bytes == expected->size());
  CHECK(captured.measured_site != nullptr);
  failure.bytes = captured.bytes;
  failure.site = captured.measured_site;
  failure.armed = true;
  const auto refused = encoder->encode(b);
  const bool injected = failure.injected;
  failure = {};
  CHECK(injected);
  CHECK(refused.status().domain() == Code::OutOfMemory);

  const auto retry = encoder->encode(b);
  CHECK(retry.ok());
  // The decoder still holds A, so this also proves the refused B never
  // became the predictor. With force_intra, the retry must still be intra.
  CHECK(decoder->decode(retry->data(), retry->size(), *output).ok());
  CHECK(*retry == *expected);
  const auto info = codec::read_frame_info(retry->data(), retry->size());
  CHECK(info.ok());
  CHECK(info->type ==
        (forced ? codec::FrameType::kIntra : codec::FrameType::kInter));

  // The refused call counts toward neither the old nor the forced GOP.
  const auto next = encoder->encode(a);
  CHECK(next.ok());
  const auto next_info = codec::read_frame_info(next->data(), next->size());
  CHECK(next_info.ok() && next_info->type == codec::FrameType::kInter);
  CHECK(decoder->decode(next->data(), next->size(), *output).ok());
  const auto boundary = encoder->encode(b);
  CHECK(boundary.ok());
  const auto boundary_info =
      codec::read_frame_info(boundary->data(), boundary->size());
  CHECK(boundary_info.ok());
  CHECK(boundary_info->type ==
        (forced ? codec::FrameType::kInter : codec::FrameType::kIntra));
  CHECK(decoder->decode(boundary->data(), boundary->size(), *output).ok());
  return 0;
}

vkc::Result<codec::EncodedFrame> recorded_frame(Gpu& gpu,
                                                codec::Encoder& encoder,
                                                vol::VoxelBlockGrid& grid) {
  vkc::CommandBatch front(gpu.device, gpu.allocator);
  VKC_ASSIGN(const codec::PendingEncode pending,
             encoder.record_front(front, grid));
  VKC_TRY(front.submit());
  vkc::CommandBatch back(gpu.device, gpu.allocator);
  VKC_TRY(encoder.record_back(back, pending));
  VKC_TRY(back.submit());
  return encoder.finish(pending);
}

int back_retry_case(Gpu& gpu, vol::VoxelBlockGrid& grid, bool hold_previous) {
  codec::EncoderConfig config;
  config.intra_period = 30;
  auto measured = codec::Encoder::create(gpu.device, gpu.allocator, config);
  auto encoder = codec::Encoder::create(gpu.device, gpu.allocator, config);
  CHECK(measured.ok() && encoder.ok());
  std::optional<codec::EncodedFrame> measured_previous;
  std::optional<codec::EncodedFrame> previous;
  if (hold_previous) {
    auto a = recorded_frame(gpu, *measured, grid);
    auto b = recorded_frame(gpu, *encoder, grid);
    CHECK(a.ok() && b.ok());
    measured_previous.emplace(std::move(a).value());
    previous.emplace(std::move(b).value());
  }
  vkc::CommandBatch measured_front(gpu.device, gpu.allocator);
  const auto a = measured->record_front(measured_front, grid);
  CHECK(a.ok() && measured_front.submit().ok());
  vkc::CommandBatch measured_back(gpu.device, gpu.allocator);
  auto& failure = vr_test::allocation_failure;
  failure = {};
  failure.measure = true;
  const auto recorded = measured->record_back(measured_back, *a);
  const auto captured = failure;
  failure = {};
  CHECK(recorded.ok());
  CHECK(captured.bytes != 0 && captured.measured_site != nullptr);
  CHECK(measured_back.submit().ok());
  const auto expected = measured->finish(*a);
  CHECK(expected.ok() && expected->blocks == 1);

  vkc::CommandBatch front(gpu.device, gpu.allocator);
  const auto b = encoder->record_front(front, grid);
  CHECK(b.ok() && front.submit().ok());
  {
    vkc::CommandBatch back(gpu.device, gpu.allocator);
    failure.bytes = captured.bytes;
    failure.site = captured.measured_site;
    failure.armed = true;
    const auto refused = encoder->record_back(back, *b);
    const bool injected = failure.injected;
    failure = {};
    CHECK(injected);
    CHECK(refused.domain() == Code::OutOfMemory);
    // Drop the failed batch before retry, including any output-slot leases.
  }
  vkc::CommandBatch retry(gpu.device, gpu.allocator);
  CHECK(encoder->record_back(retry, *b).ok());
  CHECK(retry.submit().ok());
  const auto result = encoder->finish(*b);
  CHECK(result.ok());
  CHECK(result->blocks == expected->blocks);
  CHECK(result->head == expected->head);
  CHECK(result->payload_size == expected->payload_size);
  CHECK(result->payload_size == 0 ||
        std::memcmp(result->payload, expected->payload, result->payload_size) ==
            0);
  return 0;
}

int gpu_main(vr_test::GpuContext& gpu) {
  auto a = make_grid(gpu);
  auto b = make_grid(gpu);
  CHECK(a.ok() && b.ok());
  CHECK(allocate(gpu, *a, {vr::Vec3i(0)}).ok());
  CHECK(allocate(gpu, *b, {vr::Vec3i(0)}).ok());
  CHECK(
      write_field(gpu, *a, [](const vr::Vec3f&) { return 0.01f; }, true).ok());
  CHECK(
      write_field(gpu, *b, [](const vr::Vec3f&) { return -0.01f; }, true).ok());
  int failed = returned_bytes_case(gpu, *a, *b, false);
  failed |= returned_bytes_case(gpu, *a, *b, true);
  failed |= back_retry_case(gpu, *a, false);
  failed |= back_retry_case(gpu, *a, true);
  if (failed == 0) std::puts("codec encoder OOM: OK");
  return failed;
}

}  // namespace

int main() try {
  return vr_test::run_on_gpu(gpu_main);
} catch (const std::bad_alloc&) {
  vr_test::allocation_failure = {};
  std::fputs("FAIL: bad_alloc escaped the encoder\n", stderr);
  return 1;
}
