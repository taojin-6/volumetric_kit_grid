// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include "dispatch.hpp"

#include <atomic>

namespace volumetric_kit::recon::detail {
namespace {

std::atomic<std::uint32_t> g_test_groups{0};

}  // namespace

std::uint32_t groups_per_dispatch(std::uint32_t max_groups) {
  const std::uint32_t cap = g_test_groups.load(std::memory_order_relaxed);
  return cap == 0 ? max_groups : std::min(cap, max_groups);
}

void set_test_groups_per_dispatch(std::uint32_t groups) {
  g_test_groups.store(groups, std::memory_order_relaxed);
}

}  // namespace volumetric_kit::recon::detail
