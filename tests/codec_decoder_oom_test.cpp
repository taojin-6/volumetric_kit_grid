// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// The pending-frame allocation and earlier diagnostics return OutOfMemory
// through every decode entry point. A failed record leaves the grid and
// reference intact, can be retried, and keeps a previously recorded token.

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <new>
#include <vector>

#include "volumetric_kit/core/testing/gpu_test.hpp"
#include "volumetric_kit/core/testing/test_allocation_failure.hpp"
#include "volumetric_kit/grid/base/device_requirements.hpp"
#include "volumetric_kit/grid/codec/decoder.hpp"
#include "volumetric_kit/grid/codec/encoder.hpp"
#include "volumetric_kit/grid/testing/codec_fixture.hpp"

namespace vgrid = volumetric_kit::grid;
namespace vkc = volumetric_kit::core;
namespace codec = volumetric_kit::grid::codec;
using namespace codec_fixture;

namespace {

using Bytes = std::vector<std::uint8_t>;
using Code = vkc::Status::Code;

vkc::Result<vkc::HostSlot> input(codec::Decoder& dec, const Bytes& frame) {
  VKC_ASSIGN(vkc::HostSlot slot, dec.claim_input(frame.size()));
  std::memcpy(slot.data(), frame.data(), frame.size());
  return slot;
}

int decoder_case(Gpu& gpu, const Bytes& intra, const Bytes& inter,
                 const Bytes& next, const Bytes& larger) {
  auto grid = make_grid(gpu);
  auto dec = codec::Decoder::create(gpu.device, gpu.allocator);
  CHECK(grid.ok() && dec.ok());
  // Warm every retained buffer before measuring the host bookkeeping.
  for (const Bytes* frame : {&intra, &inter, &next}) {
    CHECK(dec->decode(frame->data(), frame->size(), *grid).ok());
  }
  const auto expected_next = snapshot(gpu, *grid);
  CHECK(expected_next.ok());
  CHECK(dec->decode(intra.data(), intra.size(), *grid).ok());
  CHECK(dec->decode(inter.data(), inter.size(), *grid).ok());
  const auto expected = snapshot(gpu, *grid);
  CHECK(expected.ok());
  CHECK(dec->decode(intra.data(), intra.size(), *grid).ok());
  auto slot = input(*dec, inter);
  CHECK(slot.ok());

  // A batch already refused at input stops record at its first command.
  // With K=1, the parsed tables and result-slot lease are smaller than the
  // pending frame, so its allocation is the largest in this host-only call.
  // Measure its actual size and site instead of copying its private layout;
  // replaying both prevents a driver/layer allocation consuming the failure.
  auto& failure = vkc::test::allocation_failure;
  std::size_t bytes = 0;
  const void* site = nullptr;
  {
    vkc::CommandBatch batch(gpu.device, gpu.allocator);
    CHECK(!batch.input(vkc::HostSlot{}, 0, 1).ok());
    failure = {};
    failure.measure = true;
    const auto measured = dec->record(batch, *slot, inter.size(), *grid);
    bytes = failure.largest;
    site = failure.largest_site;
    failure = {};
    CHECK(measured.status().domain() == Code::InvalidArgument);
    CHECK(bytes > sizeof(codec::FrameInfo) && site != nullptr);
  }

  // record, decode(slot), and decode(bytes) all reach the same allocation.
  for (int api = 0; api != 3; ++api) {
    CHECK(dec->decode(intra.data(), intra.size(), *grid).ok());
    const auto before = snapshot(gpu, *grid);
    CHECK(before.ok());
    const auto epoch = grid->map().topology_epoch();
    const auto tick = grid->map().tick();
    vkc::CommandBatch batch(gpu.device, gpu.allocator);
    failure = {};
    failure.bytes = bytes;
    failure.site = site;
    failure.armed = true;
    Code code = Code::Ok;
    bool escaped = false;
    try {
      if (api == 0) {
        const auto result = dec->record(batch, *slot, inter.size(), *grid);
        code = result.status().domain();
        if (result.ok()) dec->abandon(*result);
      } else if (api == 1) {
        code = dec->decode(*slot, inter.size(), *grid).domain();
      } else {
        code = dec->decode(inter.data(), inter.size(), *grid).domain();
      }
    } catch (const std::bad_alloc&) {
      escaped = true;
    }
    const bool injected = failure.injected;
    failure = {};
    if (!injected || escaped || code != Code::OutOfMemory) {
      std::fprintf(stderr,
                   "decoder OOM api=%d: injected=%d escaped=%d code=%d\n", api,
                   injected, escaped, static_cast<int>(code));
      return 1;
    }
    if (api == 0) CHECK(batch.submit().ok());  // no commands were recorded
    CHECK(grid->map().topology_epoch() == epoch && grid->map().tick() == tick);
    const auto after = snapshot(gpu, *grid);
    CHECK(after.ok() && *after == *before);

    if (api == 0) {
      vkc::CommandBatch retry(gpu.device, gpu.allocator);
      const auto pending = dec->record(retry, *slot, inter.size(), *grid);
      CHECK(pending.ok() && retry.submit().ok());
      CHECK(dec->finish(*pending, *grid).ok());
    } else if (api == 1) {
      CHECK(dec->decode(*slot, inter.size(), *grid).ok());
    } else {
      CHECK(dec->decode(inter.data(), inter.size(), *grid).ok());
    }
    const auto retried = snapshot(gpu, *grid);
    CHECK(retried.ok() && *retried == *expected);
    CHECK(dec->decode(next.data(), next.size(), *grid).ok());
  }

  // The boundary starts before preflight diagnostics too: a truncated
  // header allocates its refusal before reaching pending-frame bookkeeping.
  for (int api = 0; api != 3; ++api) {
    vkc::CommandBatch batch(gpu.device, gpu.allocator);
    const auto truncated = [&]() {
      if (api == 0) {
        return dec->record(batch, *slot, 10, *grid).status().domain();
      }
      if (api == 1) return dec->decode(*slot, 10, *grid).domain();
      return dec->decode(inter.data(), 10, *grid).domain();
    };
    failure = {};
    failure.measure = true;
    const Code normal = truncated();
    failure.measure = false;
    CHECK(normal == Code::InvalidArgument);
    CHECK(failure.bytes != 0 && failure.measured_site != nullptr);
    failure.site = failure.measured_site;
    failure.armed = true;
    const Code refused = truncated();
    const bool injected = failure.injected;
    failure = {};
    CHECK(injected && refused == Code::OutOfMemory);
    CHECK(truncated() == normal);
  }

  // Allocating a larger replacement must not retire an already resolved
  // token or replace the spare buffers holding its reference coefficients.
  // This K=64 frame was never decoded here, so reserving it would grow those
  // buffers even though its block count is the same as the held frame's.
  CHECK(dec->decode(intra.data(), intra.size(), *grid).ok());
  vkc::CommandBatch held_batch(gpu.device, gpu.allocator);
  const auto held = dec->record(held_batch, *slot, inter.size(), *grid);
  CHECK(held.ok() && held_batch.submit().ok());
  auto larger_slot = input(*dec, larger);
  CHECK(larger_slot.ok());
  vkc::CommandBatch replacement(gpu.device, gpu.allocator);
  failure = {};
  failure.bytes = bytes;
  failure.site = site;
  failure.armed = true;
  const auto refused =
      dec->record(replacement, *larger_slot, larger.size(), *grid);
  const bool injected = failure.injected;
  failure = {};
  CHECK(injected && refused.status().domain() == Code::OutOfMemory);
  CHECK(dec->finish(*held, *grid).ok());
  CHECK(dec->rebuild(*grid).ok());
  const auto rebuilt = snapshot(gpu, *grid);
  CHECK(rebuilt.ok() && *rebuilt == *expected);
  // Release the second input slot before the blocking bytes overload claims
  // it for the next frame.
  larger_slot = vkc::HostSlot{};
  CHECK(dec->decode(next.data(), next.size(), *grid).ok());
  const auto continued = snapshot(gpu, *grid);
  CHECK(continued.ok() && *continued == *expected_next);
  return 0;
}

int gpu_main(vkc::test::GpuContext& gpu) {
  auto a = sphere_grid(gpu, Sphere{vgrid::Vec3f(0.0f), 0.04f});
  auto b = sphere_grid(gpu, Sphere{vgrid::Vec3f(0.02f, 0.0f, 0.0f), 0.05f});
  CHECK(a.ok() && b.ok());
  codec::EncoderConfig config;
  config.params.coefficient_count = 1;
  config.intra_period = 30;
  auto enc = codec::Encoder::create(gpu.device, gpu.allocator, config);
  CHECK(enc.ok());
  auto intra = enc->encode(*a);
  auto inter = enc->encode(*b);
  auto next = enc->encode(*a);
  CHECK(intra.ok() && inter.ok() && next.ok());
  config.params.coefficient_count = 64;
  auto larger_encoder =
      codec::Encoder::create(gpu.device, gpu.allocator, config);
  CHECK(larger_encoder.ok());
  auto larger = larger_encoder->encode(*b);
  CHECK(larger.ok());
  for (const Bytes* frame : {&*inter, &*next}) {
    const auto info = codec::read_frame_info(frame->data(), frame->size());
    CHECK(info.ok() && info->type == codec::FrameType::kInter);
  }
  CHECK(decoder_case(gpu, *intra, *inter, *next, *larger) == 0);
  std::puts("codec decoder OOM: OK");
  return 0;
}

}  // namespace

int main() try {
  return vkc::test::run_on_gpu(vgrid::device_requirements(), gpu_main);
} catch (const std::bad_alloc&) {
  vkc::test::allocation_failure = {};
  std::fputs("FAIL: bad_alloc escaped a decoder API\n", stderr);
  return 1;
}
