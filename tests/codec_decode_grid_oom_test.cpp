// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// Public grid helpers contain host allocation failures, including header
// diagnostics, and a failed metrics merge cannot fail a committed P-frame.

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <new>
#include <vector>

#include "codec_fixture.hpp"
#include "gpu_test.hpp"
#include "test_allocation_failure.hpp"
#include "volumetric_kit/recon/codec/decode_grid.hpp"
#include "volumetric_kit/recon/codec/encoder.hpp"

namespace codec = volumetric_kit::recon::codec;
using namespace codec_fixture;

namespace {

using Bytes = std::vector<std::uint8_t>;
using Code = vkc::Status::Code;

// Capture the diagnostic's allocation by size and call site, so allocations
// in the driver or validation layer cannot consume the failure.
int factory_refusal(Gpu& gpu, const std::uint8_t* frame, std::size_t size) {
  auto& failure = vr_test::allocation_failure;
  failure = {};
  failure.measure = true;
  const auto normal =
      codec::make_decode_grid(gpu.device, gpu.allocator, frame, size);
  failure.measure = false;
  CHECK(normal.status().domain() == Code::InvalidArgument);
  CHECK(failure.bytes != 0 && failure.measured_site != nullptr);
  failure.site = failure.measured_site;
  failure.armed = true;
  Code code = Code::Ok;
  bool escaped = false;
  try {
    code = codec::make_decode_grid(gpu.device, gpu.allocator, frame, size)
               .status()
               .domain();
  } catch (const std::bad_alloc&) {
    escaped = true;
  }
  const bool injected = failure.injected;
  failure = {};
  if (!injected || escaped || code != Code::OutOfMemory) {
    std::fprintf(stderr, "factory OOM: injected=%d escaped=%d code=%d\n",
                 injected, escaped, static_cast<int>(code));
    return 1;
  }
  const auto retry =
      codec::make_decode_grid(gpu.device, gpu.allocator, frame, size);
  CHECK(retry.status().domain() == normal.status().domain());
  CHECK(retry.status().message() == normal.status().message());
  return 0;
}

int factory_case(Gpu& gpu, const Bytes& intra) {
  int failed = factory_refusal(gpu, intra.data(), 10);
  // The header is readable, but VoxelBlockGrid::create rejects its attribute
  // sizes. Its diagnostic allocation must be inside the wrapper's boundary.
  Bytes huge = intra;
  const std::uint32_t blocks = 0xFFFFFFFFu;
  std::memcpy(huge.data() + 32, &blocks, sizeof(blocks));
  failed |= factory_refusal(gpu, huge.data(), huge.size());
  const auto retry = codec::make_decode_grid(gpu.device, gpu.allocator,
                                             intra.data(), intra.size());
  CHECK(retry.ok());
  return failed;
}

bool same_metrics(const vkc::StageMetrics& a, const vkc::StageMetrics& b) {
  if (a.rows().size() != b.rows().size()) return false;
  for (std::size_t i = 0; i < a.rows().size(); ++i) {
    const auto& x = a.rows()[i];
    const auto& y = b.rows()[i];
    if (std::strcmp(x.name, y.name) != 0 || x.cpu_ms != y.cpu_ms ||
        x.gpu_ms != y.gpu_ms || x.has_gpu != y.has_gpu) {
      return false;
    }
  }
  return true;
}

int metrics_case(Gpu& gpu, const Bytes& intra, const Bytes& inter,
                 const Bytes& next) {
  auto grid = codec::make_decode_grid(gpu.device, gpu.allocator, intra.data(),
                                      intra.size());
  auto dec = codec::Decoder::create(gpu.device, gpu.allocator);
  CHECK(grid.ok() && dec.ok());
  CHECK(codec::decode_growing(*dec, intra.data(), intra.size(), *grid).ok());

  // Existing rows force the successful attempt's merge to allocate beyond
  // anything its own small metrics set needs. The first row already matches
  // the decode, so a partial merge would corrupt its previous totals.
  vkc::StageMetrics before;
  before.add_cpu("  ..rans decode", 17.0);
  before.add_gpu("  ..rans decode", 3.0);
  std::array<std::array<char, 24>, 63> labels{};
  for (std::size_t i = 0; i < labels.size(); ++i) {
    std::snprintf(labels[i].data(), labels[i].size(), "prior row %zu", i);
    before.add_cpu(labels[i].data(), double(i + 1));
  }
  CHECK(before.rows().size() == before.rows().capacity());
  vkc::StageMetrics measured = before;
  auto& failure = vr_test::allocation_failure;
  failure = {};
  failure.measure = true;
  const auto normal =
      codec::decode_growing(*dec, inter.data(), inter.size(), *grid, &measured);
  failure.measure = false;
  CHECK(normal.ok());
  CHECK(measured.rows().size() > before.rows().size());
  CHECK(failure.bytes > before.rows().size() * sizeof(vkc::StageRow));
  CHECK(failure.measured_site != nullptr);
  const auto bytes = failure.bytes;
  const auto site = failure.measured_site;
  failure = {};
  const auto expected = snapshot(gpu, *grid);
  CHECK(expected.ok());
  CHECK(codec::decode_growing(*dec, intra.data(), intra.size(), *grid).ok());

  vkc::StageMetrics metrics = before;
  failure.bytes = bytes;
  failure.site = site;
  failure.armed = true;
  const auto decoded =
      codec::decode_growing(*dec, inter.data(), inter.size(), *grid, &metrics);
  const bool injected = failure.injected;
  failure = {};
  CHECK(injected);
  if (!decoded.ok()) {
    const auto retry =
        codec::decode_growing(*dec, inter.data(), inter.size(), *grid);
    std::fprintf(stderr, "metrics OOM: decode=%d, same-frame retry=%d\n",
                 static_cast<int>(decoded.domain()),
                 static_cast<int>(retry.domain()));
    return 1;
  }
  CHECK(same_metrics(metrics, before));
  const auto actual = snapshot(gpu, *grid);
  CHECK(actual.ok() && *actual == *expected);

  // The next P-frame requires the just-committed reference; normal metrics
  // collection also recovers on this call.
  CHECK(codec::decode_growing(*dec, next.data(), next.size(), *grid, &metrics)
            .ok());
  CHECK(metrics.rows().size() == measured.rows().size());
  CHECK(metrics.rows()[0].cpu_ms > before.rows()[0].cpu_ms);
  const auto continued = snapshot(gpu, *grid);
  CHECK(continued.ok());
  CHECK(codec::decode_growing(*dec, intra.data(), intra.size(), *grid).ok());
  CHECK(codec::decode_growing(*dec, inter.data(), inter.size(), *grid).ok());
  CHECK(codec::decode_growing(*dec, next.data(), next.size(), *grid).ok());
  const auto reference = snapshot(gpu, *grid);
  CHECK(reference.ok() && *continued == *reference);
  return 0;
}

int gpu_main(vr_test::GpuContext& gpu) {
  auto a = sphere_grid(gpu, Sphere{vr::Vec3f(0.0f), 0.08f});
  auto b = sphere_grid(gpu, Sphere{vr::Vec3f(0.02f, 0.0f, 0.0f), 0.10f});
  CHECK(a.ok() && b.ok());
  codec::EncoderConfig config;
  config.intra_period = 30;
  auto enc = codec::Encoder::create(gpu.device, gpu.allocator, config);
  CHECK(enc.ok());
  auto intra = enc->encode(*a);
  auto inter = enc->encode(*b);
  auto next = enc->encode(*a);
  CHECK(intra.ok() && inter.ok() && next.ok());
  for (const auto* frame : {&*inter, &*next}) {
    const auto info = codec::read_frame_info(frame->data(), frame->size());
    CHECK(info.ok() && info->type == codec::FrameType::kInter);
  }
  int failed = factory_case(gpu, *intra);
  failed |= metrics_case(gpu, *intra, *inter, *next);
  if (failed == 0) std::puts("codec decode grid OOM: OK");
  return failed;
}

}  // namespace

int main() try {
  return vr_test::run_on_gpu(gpu_main);
} catch (const std::bad_alloc&) {
  vr_test::allocation_failure = {};
  std::fputs("FAIL: bad_alloc escaped a decode grid helper\n", stderr);
  return 1;
}
