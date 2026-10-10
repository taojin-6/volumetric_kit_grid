// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file examples/common/stage_table.hpp
/// @brief The one table the examples print stage rows in: host and device
///        milliseconds per unit of work, and the device's share of the host
///        span. Header-only, the core's `StageMetrics` alone.

#include <cstddef>
#include <cstdio>
#include <cstring>
#include <string>

#include "volumetric_kit/core/base/stage_metrics.hpp"

namespace vr_example {

namespace vkc = volumetric_kit::core;

/// @return @p rows' row named @p name, or null.
inline const vkc::StageRow* find_row(const vkc::StageMetrics& rows,
                                     const char* name) {
  for (const vkc::StageRow& row : rows.rows()) {
    if (std::strcmp(row.name, name) == 0) return &row;
  }
  return nullptr;
}

/// @brief @p rows as a table, each divided by @p per:
///
///     <title> (host ms / device ms):
///       frame prep            1.234     0.800   64.8%
///       allocate              0.500         -
///
/// The share is the device span over the host span: what is left of the host
/// span is submission, fence waits, host round trips, and any dispatch the
/// stage does not time. A breakdown row (`StageMetrics::kBreakdownPrefix`)
/// keeps its indent.
/// @param indent  Spaces before the title; the rows sit two further in.
/// @return The table, or an empty string when @p per is 0 or there is no
///         row.
inline std::string format_stage_rows(const std::string& title,
                                     const vkc::StageMetrics& rows,
                                     std::size_t per, int indent = 0) {
  if (per == 0 || rows.empty()) return {};
  const double n = static_cast<double>(per);
  const std::string pad(static_cast<std::size_t>(indent), ' ');
  std::string out = pad + title + " (host ms / device ms):\n";
  for (const vkc::StageRow& row : rows.rows()) {
    char line[160];
    if (row.has_gpu) {
      std::snprintf(line, sizeof line, "%s  %-18s %9.3f %9.3f  %5.1f%%\n",
                    pad.c_str(), row.name, row.cpu_ms / n, row.gpu_ms / n,
                    row.cpu_ms > 0.0 ? 100.0 * row.gpu_ms / row.cpu_ms : 0.0);
    } else {
      std::snprintf(line, sizeof line, "%s  %-18s %9.3f %9s\n", pad.c_str(),
                    row.name, row.cpu_ms / n, "-");
    }
    out += line;
  }
  return out;
}

/// @brief Print @ref format_stage_rows to stdout.
inline void print_stage_rows(const std::string& title,
                             const vkc::StageMetrics& rows, std::size_t per,
                             int indent = 0) {
  std::fputs(format_stage_rows(title, rows, per, indent).c_str(), stdout);
}

}  // namespace vr_example
