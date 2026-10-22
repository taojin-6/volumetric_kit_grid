// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

// A grid attribute as a host vector, and back, for a test: the arrays are
// device-local, so a test edits a copy and writes it back, or reads the blocks
// it compares.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <map>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

#include "buffer_readback.hpp"
#include "volumetric_kit/recon/volume/hash_types.hpp"
#include "volumetric_kit/recon/volume/voxel_block_grid.hpp"

namespace vr_test {

namespace vkc = volumetric_kit::core;

// The whole of attribute `name`, as elements of T.
template <typename T>
vkc::Result<std::vector<T>> read_attribute(
    const vkc::Device& device, vkc::Allocator& allocator,
    const volumetric_kit::recon::volume::VoxelBlockGrid& grid,
    std::string_view name) {
  VKC_ASSIGN(const auto view, grid.attribute(name));
  return read_back<T>(device, allocator, *view.buffer,
                      view.buffer->size() / sizeof(T));
}

// The voxels of `blocks` in attribute `name`, as elements of T: block i's
// voxels_per_block elements from element i * voxels_per_block. A test reads
// the blocks it compares, not the whole heap: one copy of the slots they
// span, which a grid hands out from one end, so about the blocks alone.
template <typename T>
vkc::Result<std::vector<T>> read_blocks(
    const Gpu& gpu, const volumetric_kit::recon::volume::VoxelBlockGrid& grid,
    std::string_view name,
    const std::vector<volumetric_kit::recon::volume::BlockIndex>& blocks) {
  VKC_ASSIGN(const auto view, grid.attribute(name));
  const auto vpb = static_cast<std::size_t>(grid.grid().voxels_per_block);
  std::vector<T> out(blocks.size() * vpb);
  if (blocks.empty()) return out;
  std::size_t lo = ~std::size_t{0}, hi = 0;  // the voxels the blocks span
  for (const auto& b : blocks) {
    lo = std::min(lo, static_cast<std::size_t>(b.ptr));
    hi = std::max(hi, static_cast<std::size_t>(b.ptr) + vpb);
  }
  std::vector<T> span(hi - lo);
  vkc::CommandBatch batch(gpu.device, gpu.allocator);
  VKC_TRY(batch.readback(*view.buffer, VkDeviceSize(lo) * sizeof(T),
                         VkDeviceSize(span.size()) * sizeof(T), span.data()));
  VKC_TRY(batch.submit());
  for (std::size_t i = 0; i < blocks.size(); ++i) {
    const std::size_t at = static_cast<std::size_t>(blocks[i].ptr) - lo;
    std::copy_n(span.data() + at, vpb, out.data() + i * vpb);
  }
  return out;
}

// Write `data` over the start of attribute `name`.
template <typename T>
vkc::Status write_attribute(
    const vkc::Device& device, vkc::Allocator& allocator,
    const volumetric_kit::recon::volume::VoxelBlockGrid& grid,
    std::string_view name, const std::vector<T>& data) {
  VKC_ASSIGN(const auto view, grid.attribute(name));
  return write_back(device, allocator, *view.buffer, data);
}

// The voxels of an attribute from the block at the lowest ptr of some blocks
// through the block at the highest: voxel ptr + v is values[ptr - first + v].
// A test grid's few blocks sit together, so this is a small part of what
// read_attribute reads.
template <typename T>
struct AttributeSpan {
  std::size_t first = 0;
  std::vector<T> values;
};

// The span of attribute `name`, of elements T, that `blocks` lie in; empty
// for no blocks.
template <typename T>
vkc::Result<AttributeSpan<T>> read_span(
    const vkc::Device& device, vkc::Allocator& allocator,
    const volumetric_kit::recon::volume::VoxelBlockGrid& grid,
    std::string_view name,
    const std::vector<volumetric_kit::recon::volume::BlockIndex>& blocks) {
  VKC_ASSIGN(const auto view, grid.attribute(name));
  if (view.element_size != sizeof(T)) {
    return vkc::Status::invalid_argument("read_span: another element size");
  }
  AttributeSpan<T> out;
  if (blocks.empty()) return out;
  const auto by_ptr = [](const auto& a, const auto& b) {
    return a.ptr < b.ptr;
  };
  const auto [lo, hi] =
      std::minmax_element(blocks.begin(), blocks.end(), by_ptr);
  out.first = std::size_t(lo->ptr);
  out.values.resize(std::size_t(hi->ptr) + grid.grid().voxels_per_block -
                    out.first);
  vkc::CommandBatch batch(device, allocator);
  VKC_TRY(batch.readback(*view.buffer, out.first * sizeof(T),
                         out.values.size() * sizeof(T), out.values.data()));
  VKC_TRY(batch.submit());
  return out;
}

// Write `span` back where read_span read it.
template <typename T>
vkc::Status write_span(
    const vkc::Device& device, vkc::Allocator& allocator,
    const volumetric_kit::recon::volume::VoxelBlockGrid& grid,
    std::string_view name, const AttributeSpan<T>& span) {
  VKC_ASSIGN(const auto view, grid.attribute(name));
  vkc::CommandBatch batch(device, allocator);
  VKC_TRY(batch.upload(*view.buffer, span.first * sizeof(T), span.values.data(),
                       span.values.size() * sizeof(T)));
  return batch.submit();
}

using BlockCoord = std::tuple<int, int, int>;

// Every active block's coordinate and first voxel (`ptr` is a voxel offset).
inline vkc::Result<std::map<BlockCoord, std::int32_t>> blocks_of(
    volumetric_kit::recon::volume::VoxelBlockGrid& grid) {
  VKC_ASSIGN(std::vector<volumetric_kit::recon::volume::BlockIndex> active,
             grid.map().compact_active_blocks());
  std::map<BlockCoord, std::int32_t> out;
  for (const auto& b : active) {
    out[BlockCoord{b.coord.x, b.coord.y, b.coord.z}] = b.ptr;
  }
  return out;
}

namespace detail {

// The blocks two grids share, in coordinate order: a[i] and b[i] are at the
// same coordinate, whatever slots the GPU gave them.
struct MatchedBlocks {
  std::vector<volumetric_kit::recon::volume::BlockIndex> a, b;
};

inline vkc::Result<MatchedBlocks> matching_blocks(
    volumetric_kit::recon::volume::VoxelBlockGrid& a,
    volumetric_kit::recon::volume::VoxelBlockGrid& b) {
  namespace vol = volumetric_kit::recon::volume;
  MatchedBlocks m;
  VKC_ASSIGN(m.a, a.map().compact_active_blocks());
  VKC_ASSIGN(m.b, b.map().compact_active_blocks());
  if (m.a.size() != m.b.size() ||
      a.grid().voxels_per_block != b.grid().voxels_per_block) {
    return vkc::Status::invalid_argument("grid shapes differ");
  }
  const auto by_coord = [](const vol::BlockIndex& x, const vol::BlockIndex& y) {
    return vol::coord_less(x.coord, y.coord);
  };
  std::sort(m.a.begin(), m.a.end(), by_coord);
  std::sort(m.b.begin(), m.b.end(), by_coord);
  for (std::size_t i = 0; i < m.a.size(); ++i) {
    if (m.a[i].coord != m.b[i].coord) {
      return vkc::Status::invalid_argument("grid block sets differ");
    }
  }
  return m;
}

// Each voxel of the matched blocks, by its index in both grids' read_blocks.
// Both comparison paths share this traversal, but choose how many attributes
// to hold at once.
template <typename Compare>
bool compare_voxels(const MatchedBlocks& blocks, std::size_t voxels_per_block,
                    Compare&& compare) {
  for (std::size_t i = 0; i < blocks.a.size(); ++i) {
    for (std::size_t k = 0; k < voxels_per_block; ++k) {
      if (!compare(i * voxels_per_block + k)) {
        const auto& c = blocks.a[i].coord;
        std::fprintf(stderr, "grid voxel %zu differs in (%d, %d, %d)\n", k, c.x,
                     c.y, c.z);
        return false;
      }
    }
  }
  return true;
}

}  // namespace detail

// Attribute bits at one voxel. A caller decides whether all bits matter,
// only observed voxels matter, or floating-point rounding is allowed.
struct VoxelWords {
  std::uint32_t weight, tsdf, color;
};

// Read all three attributes for a comparison that relates them. Callers keep
// their own nonempty and minimum-observed checks; empty grids are valid here.
template <typename Compare>
bool compare_grids(const Gpu& gpu,
                   volumetric_kit::recon::volume::VoxelBlockGrid& a,
                   volumetric_kit::recon::volume::VoxelBlockGrid& b,
                   Compare&& compare, std::size_t* observed = nullptr,
                   std::size_t* block_count = nullptr) {
  auto blocks = detail::matching_blocks(a, b);
  if (!blocks.ok()) {
    std::fprintf(stderr, "compare_grids: %s\n",
                 blocks.status().message().c_str());
    return false;
  }
  const char* names[] = {"weight", "tsdf", "color"};
  std::vector<std::uint32_t> wa[3], wb[3];
  for (int n = 0; n < 3; ++n) {
    auto ra = read_blocks<std::uint32_t>(gpu, a, names[n], blocks->a);
    auto rb = read_blocks<std::uint32_t>(gpu, b, names[n], blocks->b);
    if (!ra.ok() || !rb.ok()) {
      std::fprintf(stderr, "compare_grids: no %s attribute\n", names[n]);
      return false;
    }
    wa[n] = std::move(ra).value();
    wb[n] = std::move(rb).value();
  }
  std::size_t weighted = 0;
  if (!detail::compare_voxels(
          blocks.value(), a.grid().voxels_per_block, [&](std::size_t i) {
            const VoxelWords va{wa[0][i], wa[1][i], wa[2][i]};
            const VoxelWords vb{wb[0][i], wb[1][i], wb[2][i]};
            weighted += va.weight != 0 ? 1 : 0;
            return compare(va, vb);
          }))
    return false;
  if (observed != nullptr) *observed = weighted;
  if (block_count != nullptr) *block_count = blocks->a.size();
  return true;
}

// Bit-identical weight, tsdf and colour, including unobserved voxels. Read
// one attribute pair at a time so large-grid tests need no extra host memory.
inline bool same_grids(const Gpu& gpu,
                       volumetric_kit::recon::volume::VoxelBlockGrid& a,
                       volumetric_kit::recon::volume::VoxelBlockGrid& b,
                       std::size_t* observed = nullptr,
                       std::size_t* block_count = nullptr) {
  auto blocks = detail::matching_blocks(a, b);
  if (!blocks.ok()) {
    std::fprintf(stderr, "same_grids: %s\n", blocks.status().message().c_str());
    return false;
  }
  const char* names[] = {"weight", "tsdf", "color"};
  std::size_t weighted = 0;
  for (int n = 0; n < 3; ++n) {
    auto ra = read_blocks<std::uint32_t>(gpu, a, names[n], blocks->a);
    auto rb = read_blocks<std::uint32_t>(gpu, b, names[n], blocks->b);
    if (!ra.ok() || !rb.ok()) {
      std::fprintf(stderr, "same_grids: no %s attribute\n", names[n]);
      return false;
    }
    if (!detail::compare_voxels(blocks.value(), a.grid().voxels_per_block,
                                [&](std::size_t i) {
                                  if (n == 0)
                                    weighted += ra.value()[i] != 0 ? 1 : 0;
                                  return ra.value()[i] == rb.value()[i];
                                })) {
      std::fprintf(stderr, "same_grids: %s differs\n", names[n]);
      return false;
    }
  }
  if (observed != nullptr) *observed = weighted;
  if (block_count != nullptr) *block_count = blocks->a.size();
  return true;
}

}  // namespace vr_test
