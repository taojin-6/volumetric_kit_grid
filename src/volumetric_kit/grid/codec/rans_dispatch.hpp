// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file rans_dispatch.hpp
/// @brief The host side of the rANS kernels, which the device frame writer
///        and reader share: their push constants and the table entries they
///        read. They and the block lists' kernels dispatch through
///        `vgrid::detail::dispatch_items` (base/dispatch.hpp).
///
/// Internal (under src/, never installed).

#include <algorithm>
#include <cstdint>
#include <vector>

#include "rans.hpp"
#include "volumetric_kit/core/vulkan/vulkan.hpp"
#include "volumetric_kit/grid/base/dispatch.hpp"

namespace volumetric_kit::grid::codec::detail {

using volumetric_kit::grid::detail::dispatch_items;

/// Mirrors `Push` in rans_push.glsl.
struct RansPush {
  std::uint32_t item_base = 0;
  /// M + A, every stream position; or for the count and layout a bound
  /// whose surplus they bind (`counted_entries`).
  std::uint32_t num_entries = 0;
  std::uint32_t coefficient_count = 0;
  std::uint32_t segment_size = 0;
  std::uint32_t segment_count = 0;
  std::uint32_t frame_type = 0;
  std::uint32_t reference_count = 0;  ///< M.
  std::uint32_t update_size = 0;      ///< U.
  /// The decode's payload's first byte in its binding.
  std::uint32_t payload_byte = 0;
};
static_assert(sizeof(RansPush) == 36, "RansPush must be 36 bytes");

/// @return Every model's symbols back to back, in TABLES order (each model
///         at @ref frame_model_base), as the kernels read them:
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

/// @return A binding's range of @p bytes, but never less than one
///         @p element: a side a frame lacks (reference entries, or new
///         blocks) still binds a valid range, which its kernel never reads.
inline VkDeviceSize binding_range(VkDeviceSize bytes, VkDeviceSize element) {
  return std::max(bytes, element);
}

}  // namespace volumetric_kit::grid::codec::detail
