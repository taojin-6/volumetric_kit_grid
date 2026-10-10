// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

// The tests' pin for a table or a byte stream too long to spell out: 32-bit
// FNV-1a, one value a step.

#include <cstdint>

namespace vr_test {

// FNV-1a over @p values, each taken as a 32-bit unsigned value: a byte of a
// stream, or an entry of a table.
template <typename Range>
std::uint32_t fnv1a(const Range& values) {
  std::uint32_t hash = 2166136261u;
  for (const auto& v : values) {
    hash = (hash ^ static_cast<std::uint32_t>(v)) * 16777619u;
  }
  return hash;
}

}  // namespace vr_test
