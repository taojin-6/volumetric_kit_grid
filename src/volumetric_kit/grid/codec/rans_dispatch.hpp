// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file rans_dispatch.hpp
/// @brief The host side of the rANS kernels, which the device frame writer
///        and reader share: their push constants, the table entries they
///        read, and the loop that dispatches them.
///
/// Internal (under src/, never installed).

#include <algorithm>
#include <cstdint>
#include <vector>

#include "rans.hpp"
#include "volumetric_kit/core/base/result.hpp"
#include "volumetric_kit/core/vulkan/command_batch.hpp"
#include "volumetric_kit/core/vulkan/compute_kernel.hpp"

namespace volumetric_kit::recon::codec::detail {

/// Mirrors `Push` in rans_push.glsl.
struct RansPush {
  std::uint32_t item_base = 0;
  std::uint32_t num_blocks = 0;
  std::uint32_t coefficient_count = 0;
  std::uint32_t segment_size = 0;
  std::uint32_t segment_count = 0;
};
static_assert(sizeof(RansPush) == 20, "RansPush must be 20 bytes");

/// @return Every model's symbols back to back, in TABLES order (each model
///         at rans_models.glsl's `model_base`), as the kernels read them:
///         `cum | freq << 16`.
inline std::vector<std::uint32_t> rans_table_entries(
    const std::vector<FrequencyTable>& tables) {
  std::vector<std::uint32_t> entries;
  for (const FrequencyTable& t : tables) {
    for (std::size_t s = 0; s < t.freq.size(); ++s) {
      entries.push_back(std::uint32_t(t.cum[s]) |
                        (std::uint32_t(t.freq[s]) << 16));
    }
  }
  return entries;
}

/// Record @p kernel over @p items invocations of @p lanes a workgroup, in as
/// many dispatches as the device's group limit needs; each pushes the first
/// item it covers as `item_base`.
inline core::Status dispatch_items(core::CommandBatch& batch,
                                   const core::ComputeKernel& kernel,
                                   RansPush push, std::uint64_t items,
                                   std::uint32_t lanes,
                                   std::uint32_t max_groups,
                                   core::GpuStageScope* stage) {
  const std::uint64_t per_dispatch = std::uint64_t(max_groups) * lanes;
  for (std::uint64_t base = 0; base < items; base += per_dispatch) {
    push.item_base = static_cast<std::uint32_t>(base);
    const std::uint64_t count =
        std::min<std::uint64_t>(per_dispatch, items - base);
    VKC_TRY(
        batch.dispatch(kernel, &push, sizeof(push),
                       static_cast<std::uint32_t>((count + lanes - 1) / lanes),
                       max_groups, stage));
  }
  return {};
}

}  // namespace volumetric_kit::recon::codec::detail
