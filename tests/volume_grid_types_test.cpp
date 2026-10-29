// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// The voxel grid's device contract, volume/shaders/grid_types.glsl, against
// volume's host types on the device (shaders/grid_types_cases.comp), built
// through the include root grid_volume exports, as a consumer's kernel is:
//
//   * vgridTickAfter is volume::tick_after on every case, across the clock's
//     wrap;
//   * vgridLocalBase inverts the in-block index, x + bs (y + bs z), and
//     vgridLocalIndex gives it back, for every voxel of five block sizes;
//   * VoxelGridParams, BlockIndex and BlockStamp read every field of the host
//     PODs at its offset, under scalar block layout.
//
// Skips where no device is present.

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include "volumetric_kit/core/testing/buffer_readback.hpp"
#include "volumetric_kit/core/testing/test_check.hpp"
#include "volumetric_kit/core/vulkan/compute_kernel.hpp"
#include "volumetric_kit/core/vulkan/compute_util.hpp"
#include "volumetric_kit/core/vulkan/vulkan.hpp"
#include "volumetric_kit/grid/base/math/vector_types.hpp"
#include "volumetric_kit/grid/volume/hash_types.hpp"
#include "volumetric_kit/grid/volume/voxel_grid.hpp"

#include "grid_types_cases_comp.spv.hpp"
#include "volumetric_kit/core/testing/gpu_test.hpp"
#include "volumetric_kit/grid/base/device_requirements.hpp"

namespace vgrid = volumetric_kit::grid;
namespace vol = volumetric_kit::grid::volume;
namespace vkc = volumetric_kit::core;

namespace {

// Mirrors the Structs block of shaders/grid_types_cases.comp.
struct Structs {
  vol::VoxelGridParams grid;
  vol::BlockIndex blocks[2];
  vol::BlockStamp stamps[2];
};
static_assert(sizeof(Structs) == 88, "the Structs block, scalar layout");

std::uint32_t bits(float f) {
  std::uint32_t u = 0;
  std::memcpy(&u, &f, sizeof(u));
  return u;
}

std::vector<glm::uvec2> tick_cases() {
  std::vector<glm::uvec2> out = {
      {0u, 0u},          {1u, 0u},          {0u, 1u},
      {0x7FFFFFFFu, 0u}, {0x80000000u, 0u}, {0x80000001u, 0u},
      {0u, 0xFFFFFFFFu}, {0xFFFFFFFFu, 0u}, {1u, 0xFFFFFFFEu},
      {0xFFFFFFFEu, 1u}, {7u, 7u},          {0x80000000u, 0x7FFFFFFFu}};
  std::uint32_t x = 0x2545F491u;
  for (int i = 0; i < 116; ++i) {
    x = x * 1664525u + 1013904223u;
    const std::uint32_t a = x;
    x = x * 1664525u + 1013904223u;
    out.push_back({a, i % 2 == 0 ? x : a + (x >> 20)});
  }
  return out;
}

std::vector<glm::uvec2> local_cases() {
  std::vector<glm::uvec2> out;
  for (const std::uint32_t bs : {1u, 2u, 3u, 4u, 8u}) {
    for (std::uint32_t l = 0; l < bs * bs * bs; ++l) out.push_back({l, bs});
  }
  return out;
}

int gpu_main(vkc::test::GpuContext& gpu) {
  const std::vector<glm::uvec2> ticks = tick_cases();
  const std::vector<glm::uvec2> locals = local_cases();
  Structs structs{};
  structs.grid = {0.0125f, 8, 512, 0.04f, 4, 1024, 4096, 128};
  structs.blocks[0] = {vgrid::Vec3i(-3, 17, -2147483647), 512};
  structs.blocks[1] = {vgrid::Vec3i(2147483647, -1, 0), -1};
  structs.stamps[0] = {1u, 0xFFFFFFFFu, 0x80000000u};
  structs.stamps[1] = {7u, 0u, 123456789u};
  const auto n_ticks = static_cast<std::uint32_t>(ticks.size());
  const auto n_locals = static_cast<std::uint32_t>(locals.size());
  constexpr std::uint32_t kFields = 22;

  vkc::ComputeKernel kernel;
  vkc::KernelSetBuilder builder(gpu.device);
  CHECK(builder
            .add(kernel, "grid_types_cases", vgrid_grid_types_cases_comp_spv,
                 vgrid_grid_types_cases_comp_spv_size, 6)
            .ok());
  auto pool = builder.build();
  auto ticks_buf = vkc::test::upload_device_buffer(
      gpu.device, gpu.allocator, ticks.data(), n_ticks * sizeof(ticks[0]));
  auto after_buf = vkc::device_storage_buffer(gpu.allocator,
                                              n_ticks * sizeof(std::uint32_t));
  auto locals_buf = vkc::test::upload_device_buffer(
      gpu.device, gpu.allocator, locals.data(), n_locals * sizeof(locals[0]));
  auto bases_buf =
      vkc::device_storage_buffer(gpu.allocator, n_locals * sizeof(glm::ivec4));
  auto structs_buf = vkc::test::upload_device_buffer(gpu.device, gpu.allocator,
                                                     &structs, sizeof(structs));
  auto fields_buf = vkc::device_storage_buffer(gpu.allocator,
                                               kFields * sizeof(std::uint32_t));
  CHECK(pool.ok() && ticks_buf.ok() && after_buf.ok() && locals_buf.ok() &&
        bases_buf.ok() && structs_buf.ok() && fields_buf.ok());
  kernel.set.write_storage_buffer(0, ticks_buf->handle(), 0, VK_WHOLE_SIZE);
  kernel.set.write_storage_buffer(1, after_buf->handle(), 0, VK_WHOLE_SIZE);
  kernel.set.write_storage_buffer(2, locals_buf->handle(), 0, VK_WHOLE_SIZE);
  kernel.set.write_storage_buffer(3, bases_buf->handle(), 0, VK_WHOLE_SIZE);
  kernel.set.write_storage_buffer(4, structs_buf->handle(), 0, VK_WHOLE_SIZE);
  kernel.set.write_storage_buffer(5, fields_buf->handle(), 0, VK_WHOLE_SIZE);
  const std::uint32_t n = n_ticks > n_locals ? n_ticks : n_locals;
  CHECK(vkc::dispatch(gpu.device, kernel, nullptr, 0, vkc::group_count(n, 64),
                      gpu.device.caps().limits().maxComputeWorkGroupCount[0])
            .ok());
  auto after = vkc::test::read_back<std::uint32_t>(gpu.device, gpu.allocator,
                                                   *after_buf, n_ticks);
  auto bases = vkc::test::read_back<glm::ivec4>(gpu.device, gpu.allocator,
                                                *bases_buf, n_locals);
  auto fields = vkc::test::read_back<std::uint32_t>(gpu.device, gpu.allocator,
                                                    *fields_buf, kFields);
  CHECK(after.ok() && bases.ok() && fields.ok());

  int failures = 0;
  for (std::uint32_t i = 0; i < n_ticks; ++i) {
    const bool want = vol::tick_after(ticks[i].x, ticks[i].y);
    if (after.value()[i] != (want ? 1u : 0u) && ++failures <= 10) {
      std::fprintf(stderr, "tick_after(%u, %u): device %u, host %d\n",
                   ticks[i].x, ticks[i].y, after.value()[i], want ? 1 : 0);
    }
  }
  for (std::uint32_t i = 0; i < n_locals; ++i) {
    const auto l = static_cast<int>(locals[i].x);
    const auto bs = static_cast<int>(locals[i].y);
    const glm::ivec4 want(l % bs, (l / bs) % bs, l / (bs * bs), l);
    const glm::ivec4 got = bases.value()[i];
    if (got != want && ++failures <= 20) {
      std::fprintf(stderr,
                   "local %d at block size %d: device (%d, %d, %d) index %d\n",
                   l, bs, got.x, got.y, got.z, got.w);
    }
  }
  const vol::VoxelGridParams& g = structs.grid;
  std::vector<std::uint32_t> want_fields = {
      bits(g.voxel_size),
      static_cast<std::uint32_t>(g.block_size),
      static_cast<std::uint32_t>(g.voxels_per_block),
      bits(g.trunc_dist),
      static_cast<std::uint32_t>(g.bucket_size),
      static_cast<std::uint32_t>(g.num_buckets),
      static_cast<std::uint32_t>(g.num_blocks),
      static_cast<std::uint32_t>(g.max_chain)};
  for (const vol::BlockIndex& b : structs.blocks) {
    want_fields.insert(want_fields.end(),
                       {static_cast<std::uint32_t>(b.coord.x),
                        static_cast<std::uint32_t>(b.coord.y),
                        static_cast<std::uint32_t>(b.coord.z),
                        static_cast<std::uint32_t>(b.ptr)});
  }
  for (const vol::BlockStamp& s : structs.stamps) {
    want_fields.insert(want_fields.end(), {s.requested, s.weighted, s.changed});
  }
  CHECK(want_fields.size() == kFields);
  for (std::uint32_t i = 0; i < kFields; ++i) {
    if (fields.value()[i] != want_fields[i] && ++failures <= 30) {
      std::fprintf(stderr, "field %u: device 0x%08x, host 0x%08x\n", i,
                   fields.value()[i], want_fields[i]);
    }
  }
  CHECK(failures == 0);
  std::printf(
      "grid volume grid types: %u tick cases, %u in-block voxels and %u "
      "struct fields matched the host's\n",
      n_ticks, n_locals, kFields);
  return 0;
}

}  // namespace

int main() {
  return vkc::test::run_on_gpu(vgrid::device_requirements(), gpu_main);
}
