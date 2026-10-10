// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

// A grid attribute as a host vector, and back, for a test: the arrays are
// device-local, so a test edits a copy and writes it back.

#include <cstdint>
#include <cstdio>
#include <map>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

#include "buffer_readback.hpp"
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

// Write `data` over the start of attribute `name`.
template <typename T>
vkc::Status write_attribute(
    const vkc::Device& device, vkc::Allocator& allocator,
    const volumetric_kit::recon::volume::VoxelBlockGrid& grid,
    std::string_view name, const std::vector<T>& data) {
  VKC_ASSIGN(const auto view, grid.attribute(name));
  return write_back(device, allocator, *view.buffer, data);
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

struct GridBlockPair {
  BlockCoord coord;
  std::size_t a, b;
};

inline vkc::Result<std::vector<GridBlockPair>> matching_blocks(
    volumetric_kit::recon::volume::VoxelBlockGrid& a,
    volumetric_kit::recon::volume::VoxelBlockGrid& b) {
  VKC_ASSIGN(auto ba, blocks_of(a));
  VKC_ASSIGN(auto bb, blocks_of(b));
  if (ba.size() != bb.size() ||
      a.grid().voxels_per_block != b.grid().voxels_per_block) {
    return vkc::Status::invalid_argument("grid shapes differ");
  }
  std::vector<GridBlockPair> pairs;
  for (const auto& [coord, ptr] : ba) {
    const auto other = bb.find(coord);
    if (other == bb.end()) {
      return vkc::Status::invalid_argument("grid block sets differ");
    }
    pairs.push_back({coord, std::size_t(ptr), std::size_t(other->second)});
  }
  return pairs;
}

// Match coordinates independently of GPU allocation order. Both comparison
// paths share this traversal, but choose how many attributes to hold at once.
template <typename Compare>
bool compare_voxels(const std::vector<GridBlockPair>& pairs,
                    std::size_t voxels_per_block, Compare&& compare) {
  for (const auto& p : pairs) {
    for (std::size_t k = 0; k < voxels_per_block; ++k) {
      if (!compare(p.a + k, p.b + k)) {
        std::fprintf(stderr, "grid voxel %zu differs in (%d, %d, %d)\n", k,
                     std::get<0>(p.coord), std::get<1>(p.coord),
                     std::get<2>(p.coord));
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
  auto pairs = detail::matching_blocks(a, b);
  if (!pairs.ok()) {
    std::fprintf(stderr, "compare_grids: %s\n",
                 pairs.status().message().c_str());
    return false;
  }
  const char* names[] = {"weight", "tsdf", "color"};
  std::vector<std::uint32_t> wa[3], wb[3];
  for (int n = 0; n < 3; ++n) {
    auto ra =
        read_attribute<std::uint32_t>(gpu.device, gpu.allocator, a, names[n]);
    auto rb =
        read_attribute<std::uint32_t>(gpu.device, gpu.allocator, b, names[n]);
    if (!ra.ok() || !rb.ok()) {
      std::fprintf(stderr, "compare_grids: no %s attribute\n", names[n]);
      return false;
    }
    wa[n] = std::move(ra).value();
    wb[n] = std::move(rb).value();
  }
  std::size_t weighted = 0;
  if (!detail::compare_voxels(
          pairs.value(), a.grid().voxels_per_block,
          [&](std::size_t ia, std::size_t ib) {
            const VoxelWords va{wa[0][ia], wa[1][ia], wa[2][ia]};
            const VoxelWords vb{wb[0][ib], wb[1][ib], wb[2][ib]};
            weighted += va.weight != 0 ? 1 : 0;
            return compare(va, vb);
          }))
    return false;
  if (observed != nullptr) *observed = weighted;
  if (block_count != nullptr) *block_count = pairs.value().size();
  return true;
}

// Bit-identical weight, tsdf and colour, including unobserved voxels. Read
// one attribute pair at a time so large-grid tests need no extra host memory.
inline bool same_grids(const Gpu& gpu,
                       volumetric_kit::recon::volume::VoxelBlockGrid& a,
                       volumetric_kit::recon::volume::VoxelBlockGrid& b,
                       std::size_t* observed = nullptr,
                       std::size_t* block_count = nullptr) {
  auto pairs = detail::matching_blocks(a, b);
  if (!pairs.ok()) {
    std::fprintf(stderr, "same_grids: %s\n", pairs.status().message().c_str());
    return false;
  }
  const char* names[] = {"weight", "tsdf", "color"};
  std::size_t weighted = 0;
  for (int n = 0; n < 3; ++n) {
    auto ra =
        read_attribute<std::uint32_t>(gpu.device, gpu.allocator, a, names[n]);
    auto rb =
        read_attribute<std::uint32_t>(gpu.device, gpu.allocator, b, names[n]);
    if (!ra.ok() || !rb.ok()) {
      std::fprintf(stderr, "same_grids: no %s attribute\n", names[n]);
      return false;
    }
    if (!detail::compare_voxels(pairs.value(), a.grid().voxels_per_block,
                                [&](std::size_t ia, std::size_t ib) {
                                  if (n == 0)
                                    weighted += ra.value()[ia] != 0 ? 1 : 0;
                                  return ra.value()[ia] == rb.value()[ib];
                                })) {
      std::fprintf(stderr, "same_grids: %s differs\n", names[n]);
      return false;
    }
  }
  if (observed != nullptr) *observed = weighted;
  if (block_count != nullptr) *block_count = pairs.value().size();
  return true;
}

}  // namespace vr_test
