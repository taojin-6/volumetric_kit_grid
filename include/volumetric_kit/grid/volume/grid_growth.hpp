// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file volume/grid_growth.hpp
/// @brief When a @ref VoxelBlockGrid grows, and how far: ahead of need at
///        @ref VoxelHashMap::kGrowThreshold, or for an allocation that hit a
///        capacity limit, within a caller's ceiling and memory.

#include <cstdint>
#include <functional>
#include <optional>

#include "volumetric_kit/core/base/result.hpp"
#include "volumetric_kit/core/base/stage_metrics.hpp"
#include "volumetric_kit/recon/volume/export.hpp"
#include "volumetric_kit/recon/volume/voxel_block_grid.hpp"

namespace volumetric_kit::recon::volume {

/// @brief What a caller lets a grid grow to, and what it can afford.
struct GrowthPolicy {
  /// Grow at all. Off, the grid keeps the size it was created with, and an
  /// allocation past it leaves the blocks that did not fit out.
  bool enabled = true;
  /// The most buckets the grid may grow to; 0 is as far as
  /// @ref VoxelBlockGrid::max_num_buckets allows.
  std::int32_t max_buckets = 0;
  /// The device memory the process can still commit, in bytes, read only
  /// once a grow is due, so it may be a system call. A grow that needs more
  /// (@ref VoxelBlockGrid::bytes_at the new size) is declined. Return
  /// `std::nullopt` when the reading is unknown, which declines nothing; 0 is
  /// a known reading at the limit, which declines every grow. A caller with
  /// several ceilings (a process limit and a GPU working set, say) returns
  /// the least of them. Empty, the headroom is unknown.
  std::function<std::optional<std::uint64_t>()> headroom;
  /// Ticks of the map's clock (@ref VoxelHashMap::tick: one a fused set) a
  /// grow declined for memory, or whose resize ran out of it, waits before
  /// it is asked again at the same size. The pressure that refused it may
  /// pass, and growth is the only thing that lowers the occupancy, so a
  /// refusal never stands for the rest of a scan. 0 asks again every time.
  std::uint32_t retry_after = 60;
};

/// @brief What a @ref GridGrowth call did.
enum class GrowthOutcome : std::uint8_t {
  NotDue,             ///< Below the threshold: nothing to do.
  Grew,               ///< Resized to @ref GrowthEvent::to_buckets.
  Disabled,           ///< Due, and @ref GrowthPolicy::enabled is off.
  AtCeiling,          ///< Due, and the grid is at its ceiling.
  Waiting,            ///< Due, and a refusal at this size has not expired.
  DeclinedForMemory,  ///< Due, and @ref GrowthPolicy::headroom is too small.
  ResizeFailed,       ///< Due, and the resize ran out of memory.
};

/// @brief A @ref GridGrowth decision and the figures that report it.
struct GrowthEvent {
  GrowthOutcome outcome = GrowthOutcome::NotDue;
  /// The map's occupancy (@ref VoxelHashMap::load_factor) the decision read.
  float load_factor = 0.0f;
  /// The bucket count before the call.
  std::int32_t from_buckets = 0;
  /// The size grown to, or the one refused; 0 when no size was chosen
  /// (`NotDue`, `Disabled`, `AtCeiling`).
  std::int32_t to_buckets = 0;
  /// @ref VoxelBlockGrid::bytes_at @ref to_buckets: what the resize
  /// allocates beside the live grid.
  std::uint64_t needed_bytes = 0;
  /// The headroom read, for `DeclinedForMemory`.
  std::uint64_t headroom_bytes = 0;
  /// Why the resize failed, for `ResizeFailed`.
  core::Status error;

  /// @return Whether the grid grew.
  bool grew() const noexcept { return outcome == GrowthOutcome::Grew; }
};

/// @brief Grows a @ref VoxelBlockGrid by a @ref GrowthPolicy: ahead of need,
///        before the slow regime, or for an allocation that ran out of room.
///
/// Both entry points decide alike. A grow doubles the bucket count (or goes
/// further, to a size a caller asks for), clamped to the ceiling; it is
/// declined when the headroom will not cover @ref VoxelBlockGrid::bytes_at
/// the new size; and a refusal at a size, for memory or a failed resize,
/// holds for @ref GrowthPolicy::retry_after ticks, then the grow is asked
/// again. Any other resize failure is returned as an error. A grow is timed
/// as a `"resize"` row.
///
/// The refusal it remembers belongs to one grid: use one per grid.
class VR_VOLUME_API GridGrowth {
 public:
  /// @param policy  The ceiling, the memory and whether to grow at all.
  explicit GridGrowth(GrowthPolicy policy = {});

  /// @brief Grow @p grid ahead of need: double it once its
  ///        @ref VoxelHashMap::load_factor exceeds
  ///        @ref VoxelHashMap::kGrowThreshold.
  ///
  /// Linear probing slows sharply past that occupancy, so a caller that grows
  /// only once allocation fails has already run every insert before it at
  /// the slowest. Call this before allocating.
  /// @param grid     The grid.
  /// @param metrics  Optional rows: `"resize"` around a grow.
  /// @return The decision, or `Status::Code::InvalidArgument` for a
  ///         moved-from @p grid or a negative @ref GrowthPolicy::max_buckets,
  ///         or a resize failure other than running out of memory.
  core::Result<GrowthEvent> grow_ahead(VoxelBlockGrid& grid,
                                       core::StageMetrics* metrics = nullptr);

  /// @brief Grow @p grid for an allocation that hit a capacity limit
  ///        (@ref AllocFailures::capacity_limited, or a codec frame refused
  ///        as too large): double it, or grow it to @p at_least buckets if
  ///        that is more.
  /// @param grid      The grid.
  /// @param at_least  The fewest buckets the caller needs; 0 doubles.
  /// @param metrics   Optional rows: `"resize"` around a grow.
  /// @return As @ref grow_ahead; never `NotDue`.
  core::Result<GrowthEvent> grow(VoxelBlockGrid& grid,
                                 std::int32_t at_least = 0,
                                 core::StageMetrics* metrics = nullptr);

  /// @return The policy this applies.
  const GrowthPolicy& policy() const noexcept { return policy_; }

 private:
  core::Result<GrowthEvent> decide(VoxelBlockGrid& grid, GrowthEvent event,
                                   std::int32_t at_least,
                                   core::StageMetrics* metrics);
  void refuse(const VoxelBlockGrid& grid) noexcept;

  GrowthPolicy policy_;
  // The size a grow was last refused at (0: none), and the map's tick then.
  std::int32_t refused_at_ = 0;
  std::uint32_t refused_tick_ = 0;
};

}  // namespace volumetric_kit::recon::volume
