// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include "volumetric_kit/recon/volume/grid_growth.hpp"

#include <algorithm>
#include <new>

namespace volumetric_kit::recon::volume {

core::Status grow_grid(VoxelBlockGrid& grid, std::int32_t at_least,
                       core::StageMetrics* metrics) try {
  if (!grid.valid() || at_least < 0) {
    return core::Status::invalid_argument(
        "grow_grid: requires a live grid and a non-negative minimum");
  }
  const std::int32_t ceiling = grid.max_num_buckets();
  if (grid.grid().num_buckets >= ceiling || at_least > ceiling) {
    return core::Status::out_of_memory("grow_grid: grid capacity limit");
  }
  const auto target = static_cast<std::int32_t>(std::min<std::int64_t>(
      std::max<std::int64_t>(2 * std::int64_t(grid.grid().num_buckets),
                             at_least),
      ceiling));
  core::StageScope span(metrics, "resize");
  return grid.resize(target);
} catch (const std::bad_alloc&) {
  return core::Status::out_of_memory("grow_grid: host allocation failed");
}

}  // namespace volumetric_kit::recon::volume
