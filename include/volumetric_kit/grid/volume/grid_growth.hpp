// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file volume/grid_growth.hpp
/// @brief Shared grid-growth arithmetic for fusion and codec callers.

#include <cstdint>

#include "volumetric_kit/core/base/result.hpp"
#include "volumetric_kit/core/base/stage_metrics.hpp"
#include "volumetric_kit/recon/volume/export.hpp"
#include "volumetric_kit/recon/volume/voxel_block_grid.hpp"

namespace volumetric_kit::recon::volume {

/// @brief Double @p grid, or grow it to @p at_least buckets if larger.
///
/// Clamps a doubling to @ref VoxelBlockGrid::max_num_buckets. A requested
/// minimum past that ceiling is refused before resizing. This helper retains
/// no state; the caller decides when to grow and whether to retry a failure.
/// @param grid      The grid to grow; a failed resize preserves its contents.
/// @param at_least  Minimum bucket count; 0 requests a doubling.
/// @param metrics   Optional `"resize"` timing row.
/// @return OK after growth; `Status::Code::InvalidArgument` for a moved-from
///         grid or a negative minimum; `Status::Code::OutOfMemory` at the
///         ceiling, for an unreachable minimum or a host allocation failure;
///         otherwise @ref VoxelBlockGrid::resize's error.
VR_VOLUME_API core::Status grow_grid(VoxelBlockGrid& grid,
                                     std::int32_t at_least = 0,
                                     core::StageMetrics* metrics = nullptr);

}  // namespace volumetric_kit::recon::volume
