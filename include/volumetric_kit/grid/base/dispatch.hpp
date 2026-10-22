// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file dispatch.hpp
/// @brief A loop that records a 1-D dispatch of items whose group count can
///        pass the device's limit, and its test seam.
///
/// Internal (under src/, never installed). Vulkan guarantees only 65535
/// groups along x (`maxComputeWorkGroupCount[0]`), which a room's blocks
/// already pass, so such a dispatch is recorded as several rather than
/// refused. The seam carries `VR_CORE_API` so a test reaches it in a
/// shared-library build too.

#include <algorithm>
#include <cstdint>

#include "volumetric_kit/core/base/result.hpp"
#include "volumetric_kit/core/vulkan/command_batch.hpp"
#include "volumetric_kit/core/vulkan/compute_kernel.hpp"
#include "volumetric_kit/core/vulkan/descriptor.hpp"
#include "volumetric_kit/recon/core/export.hpp"

namespace volumetric_kit::recon::detail {

/// @return The groups one dispatch of @ref dispatch_items holds:
///         @p max_groups, the device's `maxComputeWorkGroupCount[0]`, or the
///         cap @ref set_test_groups_per_dispatch set, if lower.
VR_CORE_API std::uint32_t groups_per_dispatch(std::uint32_t max_groups);

/// @brief For tests only: cap @ref groups_per_dispatch at @p groups, so a
///        small input takes the split path a device's limit gives a large
///        one; 0 lifts the cap. Process-wide.
VR_CORE_API void set_test_groups_per_dispatch(std::uint32_t groups);

/// Record @p kernel over @p items invocations of @p lanes a workgroup, in as
/// many dispatches of @ref groups_per_dispatch groups as it takes; each pushes
/// @p push with the first item it covers as `item_base`, and binds @p set, or
/// the kernel's own set when it is null. The batch's barrier between
/// dispatches orders each after the one before.
template <typename Push>
core::Status dispatch_items(core::CommandBatch& batch,
                            const core::ComputeKernel& kernel, Push push,
                            std::uint64_t items, std::uint32_t lanes,
                            std::uint32_t max_groups,
                            core::GpuStageScope* stage,
                            const core::DescriptorSet* set = nullptr) {
  const std::uint64_t per_dispatch =
      std::uint64_t(groups_per_dispatch(max_groups)) * lanes;
  for (std::uint64_t base = 0; base < items; base += per_dispatch) {
    push.item_base = static_cast<std::uint32_t>(base);
    const std::uint64_t count =
        std::min<std::uint64_t>(per_dispatch, items - base);
    const auto groups = static_cast<std::uint32_t>((count + lanes - 1) / lanes);
    VKC_TRY(set != nullptr ? batch.dispatch(kernel, *set, &push, sizeof(push),
                                            groups, max_groups, stage)
                           : batch.dispatch(kernel, &push, sizeof(push), groups,
                                            max_groups, stage));
  }
  return {};
}

}  // namespace volumetric_kit::recon::detail
