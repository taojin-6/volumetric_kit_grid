// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

// A StageMetrics' rows by name, and the rule its breakdowns keep: each comes
// after a stage's row, and its host time is inside that stage's, so the
// totals, which skip breakdowns, still count it.

#include <cstring>

#include "volumetric_kit/core/base/stage_metrics.hpp"

namespace vr_test {

// The row named @p name, or null.
inline const volumetric_kit::core::StageRow* find_row(
    const volumetric_kit::core::StageMetrics& m, const char* name) {
  for (const volumetric_kit::core::StageRow& row : m.rows()) {
    if (std::strcmp(row.name, name) == 0) return &row;
  }
  return nullptr;
}

// Whether every breakdown row of @p m follows a stage row, and has no more
// host time than the nearest one above it.
inline bool breakdowns_nest(const volumetric_kit::core::StageMetrics& m) {
  const char* const prefix =
      volumetric_kit::core::StageMetrics::kBreakdownPrefix;
  const volumetric_kit::core::StageRow* stage = nullptr;
  for (const volumetric_kit::core::StageRow& row : m.rows()) {
    if (std::strncmp(row.name, prefix, std::strlen(prefix)) != 0) {
      stage = &row;
    } else if (stage == nullptr || row.cpu_ms > stage->cpu_ms) {
      return false;
    }
  }
  return true;
}

}  // namespace vr_test
