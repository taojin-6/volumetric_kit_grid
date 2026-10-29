// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// The voxel colour attribute's format, held together in its three places:
// volume/shaders/voxel_color.glsl, which fusion writes and the mesher reads
// with; volume/voxel_color.hpp, its host half; and core/color_space.hpp's
// published curve, the sensor boundary's. Two implementations of one curve is
// the silent drift the 2026-08-02 colour-space decision exists to prevent, so
// it is pinned rather than trusted.
//
// Host, always: volume's curve, decode and running-mean encode are
// color_space's, bit for bit, over every code and a sweep of linear values;
// the unobserved word decodes to opaque white.
//
// Device (shaders/voxel_color_cases.comp, skipped where no device is
// present), over every grey code, the colour-unobserved word and words with
// any high byte:
//   * the decoded LINEAR values agree with the host's within 1e-5 -- catches a
//     substituted pow(x, 2.2), a dropped linear toe or a mistyped constant;
//   * the 8-bit ROUND TRIP is exact for every word, alpha 0xFF -- catches a
//     rounding-mode difference between packUnorm4x8 and the host's +0.5;
//   * a first observation stores the word's R, G, B and alpha 0xFF, exactly
//     as the host's, never the unobserved word;
//   * the mesher reads every word as the host's voxel_color_to_linear does:
//     the unobserved word exactly (opaque white), any other within 1e-5.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <vector>

#include "volumetric_kit/core/testing/buffer_readback.hpp"
#include "volumetric_kit/core/testing/test_check.hpp"
#include "volumetric_kit/core/vulkan/compute_kernel.hpp"
#include "volumetric_kit/core/vulkan/compute_util.hpp"
#include "volumetric_kit/core/vulkan/vulkan.hpp"
#include "volumetric_kit/recon/core/color_space.hpp"
#include "volumetric_kit/recon/core/math/vector_types.hpp"
#include "volumetric_kit/recon/volume/voxel_color.hpp"

#include "volumetric_kit/core/testing/gpu_test.hpp"
#include "volumetric_kit/recon/core/device_requirements.hpp"
#include "voxel_color_cases_comp.spv.hpp"

namespace vr = volumetric_kit::recon;
namespace vol = volumetric_kit::recon::volume;
namespace vkc = volumetric_kit::core;

namespace {

bool same_bits(float a, float b) {
  std::uint32_t x = 0;
  std::uint32_t y = 0;
  std::memcpy(&x, &a, sizeof(x));
  std::memcpy(&y, &b, sizeof(y));
  return x == y;
}

bool same_bits(const vr::Vec3f& a, const vr::Vec3f& b) {
  return same_bits(a.x, b.x) && same_bits(a.y, b.y) && same_bits(a.z, b.z);
}

// volume's half against color_space's: the same formula, so the same bits.
int check_host() {
  for (std::uint32_t c = 0; c < 256; ++c) {
    const float e = static_cast<float>(c) / 255.0f;
    CHECK(same_bits(vol::srgb_to_linear(e), vr::srgb_to_linear(e)));
    const std::uint32_t word = c | (c << 8) | (c << 16) | 0xFF000000u;
    CHECK(same_bits(vol::voxel_color_to_linear(word),
                    vr::unpack_srgb_to_linear(word)));
  }
  // Unobserved reads opaque white, so it does not drag an average to black.
  CHECK(same_bits(vol::voxel_color_to_linear(vol::kVoxelColorUnobserved),
                  vr::Vec3f(1.0f)));
  const float specials[] = {-0.0f,
                            -1.0f,
                            2.0f,
                            0.0031308f,
                            std::numeric_limits<float>::quiet_NaN(),
                            std::numeric_limits<float>::infinity(),
                            -std::numeric_limits<float>::infinity()};
  std::vector<float> linear(std::begin(specials), std::end(specials));
  for (int i = 0; i <= 1 << 16; ++i) {
    linear.push_back(static_cast<float>(i) / static_cast<float>(1 << 16));
  }
  for (const float l : linear) {
    CHECK(std::isnan(l) ||
          same_bits(vol::linear_to_srgb(l), vr::linear_to_srgb(l)));
    const vr::Vec3f v(l, 1.0f - l, 0.5f * l);
    CHECK(vol::voxel_color_from_linear(v) == vr::pack_linear_to_srgb(v));
  }
  std::puts("recon volume voxel colour: the host half matched color_space");
  return 0;
}

// Every grey code, the unobserved word, and words with any high byte.
std::vector<std::uint32_t> words() {
  std::vector<std::uint32_t> out;
  for (std::uint32_t c = 0; c < 256; ++c) {
    out.push_back(c | (c << 8) | (c << 16) | 0xFF000000u);
  }
  out.insert(out.end(), {vol::kVoxelColorUnobserved, 0x00123456u, 0x01000000u,
                         0xAB000000u, 0xFFFFFFFFu, 0x7F80FF01u});
  std::uint32_t x = 0x9E3779B9u;
  for (int i = 0; i < 250; ++i) {
    x = x * 1664525u + 1013904223u;  // any word, high byte included
    out.push_back(x);
  }
  return out;
}

int gpu_main(vkc::test::GpuContext& gpu) {
  const std::vector<std::uint32_t> in = words();
  const auto n = static_cast<std::uint32_t>(in.size());

  vkc::ComputeKernel kernel;
  vkc::KernelSetBuilder builder(gpu.device);
  CHECK(builder
            .add(kernel, "voxel_color_cases", vr_voxel_color_cases_comp_spv,
                 vr_voxel_color_cases_comp_spv_size, 5)
            .ok());
  auto pool = builder.build();
  auto words_buf = vkc::test::upload_device_buffer(
      gpu.device, gpu.allocator, in.data(), n * sizeof(std::uint32_t));
  auto linear_buf =
      vkc::device_storage_buffer(gpu.allocator, n * sizeof(vr::Vec4f));
  auto round_buf =
      vkc::device_storage_buffer(gpu.allocator, n * sizeof(std::uint32_t));
  auto observed_buf =
      vkc::device_storage_buffer(gpu.allocator, n * sizeof(std::uint32_t));
  auto read_buf =
      vkc::device_storage_buffer(gpu.allocator, n * sizeof(vr::Vec4f));
  CHECK(pool.ok() && words_buf.ok() && linear_buf.ok() && round_buf.ok() &&
        observed_buf.ok() && read_buf.ok());
  kernel.set.write_storage_buffer(0, words_buf->handle(), 0, VK_WHOLE_SIZE);
  kernel.set.write_storage_buffer(1, linear_buf->handle(), 0, VK_WHOLE_SIZE);
  kernel.set.write_storage_buffer(2, round_buf->handle(), 0, VK_WHOLE_SIZE);
  kernel.set.write_storage_buffer(3, observed_buf->handle(), 0, VK_WHOLE_SIZE);
  kernel.set.write_storage_buffer(4, read_buf->handle(), 0, VK_WHOLE_SIZE);
  CHECK(vkc::dispatch(gpu.device, kernel, nullptr, 0, vkc::group_count(n, 64),
                      gpu.device.caps().limits().maxComputeWorkGroupCount[0])
            .ok());
  auto linear = vkc::test::read_back<vr::Vec4f>(gpu.device, gpu.allocator,
                                                *linear_buf, n);
  auto round = vkc::test::read_back<std::uint32_t>(gpu.device, gpu.allocator,
                                                   *round_buf, n);
  auto observed = vkc::test::read_back<std::uint32_t>(gpu.device, gpu.allocator,
                                                      *observed_buf, n);
  auto read =
      vkc::test::read_back<vr::Vec4f>(gpu.device, gpu.allocator, *read_buf, n);
  CHECK(linear.ok() && round.ok() && observed.ok() && read.ok());

  // 1e-5 absolute is far tighter than the ~0.0036 gap between the exact curve
  // and pow(x, 2.2) at its worst.
  const auto near = [](const vr::Vec4f& got, const vr::Vec3f& want) {
    return std::fabs(got.x - want.x) < 1e-5f &&
           std::fabs(got.y - want.y) < 1e-5f &&
           std::fabs(got.z - want.z) < 1e-5f;
  };
  int failures = 0;
  for (std::uint32_t i = 0; i < n; ++i) {
    const std::uint32_t word = in[i];
    const float inv = 1.0f / 255.0f;
    const vr::Vec3f decoded(
        vol::srgb_to_linear(static_cast<float>(word & 0xFFu) * inv),
        vol::srgb_to_linear(static_cast<float>((word >> 8) & 0xFFu) * inv),
        vol::srgb_to_linear(static_cast<float>((word >> 16) & 0xFFu) * inv));
    const vr::Vec4f& r = read.value()[i];
    const vr::Vec3f want_read = vol::voxel_color_to_linear(word);
    const bool read_ok =
        word == vol::kVoxelColorUnobserved
            ? r.x == want_read.x && r.y == want_read.y && r.z == want_read.z
            : near(r, want_read);
    const bool ok =
        near(linear.value()[i], decoded) &&
        round.value()[i] == ((word & 0x00FFFFFFu) | 0xFF000000u) &&
        round.value()[i] == vol::voxel_color_from_linear(decoded) &&
        observed.value()[i] == vol::voxel_color_from_observed(word) &&
        observed.value()[i] != vol::kVoxelColorUnobserved && read_ok;
    if (!ok && ++failures <= 20) {
      std::fprintf(stderr,
                   "word 0x%08x: linear (%.9g, %.9g, %.9g) host (%.9g, %.9g, "
                   "%.9g), round 0x%08x, observed 0x%08x, read (%.9g, %.9g, "
                   "%.9g)\n",
                   word, static_cast<double>(linear.value()[i].x),
                   static_cast<double>(linear.value()[i].y),
                   static_cast<double>(linear.value()[i].z),
                   static_cast<double>(decoded.x),
                   static_cast<double>(decoded.y),
                   static_cast<double>(decoded.z), round.value()[i],
                   observed.value()[i], static_cast<double>(r.x),
                   static_cast<double>(r.y), static_cast<double>(r.z));
    }
  }
  CHECK(failures == 0);
  std::printf(
      "recon volume voxel colour: %u words, the GLSL format matched the "
      "host's\n",
      n);
  return 0;
}

}  // namespace

int main() {
  if (check_host() != 0) return 1;
  return vkc::test::run_on_gpu(vr::device_requirements(), gpu_main);
}
