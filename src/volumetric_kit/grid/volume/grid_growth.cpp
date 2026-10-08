// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include "volumetric_kit/recon/volume/grid_growth.hpp"

#include <algorithm>
#include <optional>
#include <utility>

#include "volumetric_kit/core/vulkan/vk_result.hpp"

namespace volumetric_kit::recon::volume {
namespace {

// A resize that ran out of memory: the allocator refused a grown buffer, or
// the grown table could not place the blocks (VoxelHashMap::resize). Either
// may pass, so it is a refusal that expires, not an error.
bool out_of_memory(const core::Status& status) {
  if (status.domain() == core::Status::Code::OutOfMemory) return true;
  const std::optional<VkResult> vk = core::vk_result(status);
  return vk && (*vk == VK_ERROR_OUT_OF_DEVICE_MEMORY ||
                *vk == VK_ERROR_OUT_OF_HOST_MEMORY);
}

}  // namespace

GridGrowth::GridGrowth(GrowthPolicy policy) : policy_(std::move(policy)) {}

core::Result<GrowthEvent> GridGrowth::grow_ahead(VoxelBlockGrid& grid,
                                                 core::StageMetrics* metrics) {
  if (!grid.valid()) {
    return core::Status::invalid_argument(
        "GridGrowth::grow_ahead: moved-from grid");
  }
  GrowthEvent event;
  VKC_ASSIGN(event.load_factor, grid.map().load_factor());
  event.from_buckets = grid.grid().num_buckets;
  if (!(event.load_factor > VoxelHashMap::kGrowThreshold)) {
    return event;
  }
  return decide(grid, std::move(event), 0, metrics);
}

core::Result<GrowthEvent> GridGrowth::grow(VoxelBlockGrid& grid,
                                           std::int32_t at_least,
                                           core::StageMetrics* metrics) {
  if (!grid.valid()) {
    return core::Status::invalid_argument("GridGrowth::grow: moved-from grid");
  }
  GrowthEvent event;
  VKC_ASSIGN(event.load_factor, grid.map().load_factor());
  event.from_buckets = grid.grid().num_buckets;
  return decide(grid, std::move(event), at_least, metrics);
}

core::Result<GrowthEvent> GridGrowth::decide(VoxelBlockGrid& grid,
                                             GrowthEvent event,
                                             std::int32_t at_least,
                                             core::StageMetrics* metrics) {
  if (policy_.max_buckets < 0) {
    return core::Status::invalid_argument(
        "GridGrowth: GrowthPolicy::max_buckets must be >= 0");
  }
  if (!policy_.enabled) {
    event.outcome = GrowthOutcome::Disabled;
    return event;
  }
  const std::int32_t buckets = event.from_buckets;
  std::int32_t ceiling = grid.max_num_buckets();
  if (policy_.max_buckets > 0) {
    ceiling = std::min(ceiling, policy_.max_buckets);
  }
  if (buckets >= ceiling) {
    event.outcome = GrowthOutcome::AtCeiling;
    return event;
  }
  event.to_buckets = static_cast<std::int32_t>(std::min<std::int64_t>(
      std::max<std::int64_t>(2 * std::int64_t(buckets), at_least), ceiling));
  event.needed_bytes = grid.bytes_at(event.to_buckets);

  // A refusal holds only at the size it was made at, and only for a while.
  if (refused_at_ != 0) {
    const auto age = std::uint32_t(grid.map().tick() - refused_tick_);
    if (refused_at_ != buckets || age >= policy_.retry_after) {
      refused_at_ = 0;
    } else {
      event.outcome = GrowthOutcome::Waiting;
      return event;
    }
  }
  if (policy_.headroom) {
    const std::optional<std::uint64_t> headroom = policy_.headroom();
    if (headroom && *headroom < event.needed_bytes) {
      refuse(grid);
      event.outcome = GrowthOutcome::DeclinedForMemory;
      event.headroom_bytes = *headroom;
      return event;
    }
  }
  core::Status resized;
  {
    core::StageScope span(metrics, "resize");
    resized = grid.resize(event.to_buckets);
  }
  if (resized.ok()) {
    event.outcome = GrowthOutcome::Grew;
    return event;
  }
  if (!out_of_memory(resized)) {
    return resized;
  }
  refuse(grid);
  event.outcome = GrowthOutcome::ResizeFailed;
  event.error = std::move(resized);
  return event;
}

void GridGrowth::refuse(const VoxelBlockGrid& grid) noexcept {
  refused_at_ = grid.grid().num_buckets;
  refused_tick_ = grid.map().tick();
}

}  // namespace volumetric_kit::recon::volume
