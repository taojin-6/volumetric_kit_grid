// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file examples/common/stage_table.hpp
/// @brief The one table the examples print stage rows in: host busy, wait and
///        device milliseconds per unit of work. Header-only, the core's
///        `StageMetrics` alone.

#include <cstddef>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "volumetric_kit/core/base/stage_metrics.hpp"

namespace vr_example {

namespace vkc = volumetric_kit::core;

/// @return The nearest-rank @p percent th percentile of @p sorted, ascending
///         and not empty: its ceil(@p percent / 100 n)-th smallest.
template <typename T>
T nearest_rank(const std::vector<T>& sorted, std::size_t percent) {
  return sorted[(sorted.size() * percent + 99) / 100 - 1];
}

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
///     <title> (ms: host busy / wait / device):
///       frame prep            0.934     0.300     0.800
///       allocate              0.500     0.000         -
///       integrate             1.100     2.400       n/t
///
/// Busy is the host span less its wait, the time it blocked on the device
/// (`StageRow::wait_ms`); the device column reads `n/t`
/// where the row's device work ran untimed, `n/a` where the queue cannot
/// time, and `-` for a host-only row. A breakdown row
/// (`StageMetrics::kBreakdownPrefix`) keeps its indent.
/// @param indent  Spaces before the title; the rows sit two further in.
/// @return The table, or an empty string when @p per is 0 or there is no
///         row.
inline std::string format_stage_rows(const std::string& title,
                                     const vkc::StageMetrics& rows,
                                     std::size_t per, int indent = 0) {
  if (per == 0 || rows.empty()) return {};
  const double n = static_cast<double>(per);
  const std::string pad(static_cast<std::size_t>(indent), ' ');
  std::string out = pad + title + " (ms: host busy / wait / device):\n";
  for (const vkc::StageRow& row : rows.rows()) {
    char device[32] = "-";
    switch (row.device) {
      case vkc::DeviceTime::Measured:
        std::snprintf(device, sizeof device, "%.3f", row.gpu_ms / n);
        break;
      case vkc::DeviceTime::NotTimed:
        std::snprintf(device, sizeof device, "n/t");
        break;
      case vkc::DeviceTime::Unavailable:
        std::snprintf(device, sizeof device, "n/a");
        break;
      case vkc::DeviceTime::None:
        break;
    }
    char line[160];
    std::snprintf(line, sizeof line, "%s  %-18s %9.3f %9.3f %9s\n", pad.c_str(),
                  row.name, (row.cpu_ms - row.wait_ms) / n, row.wait_ms / n,
                  device);
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
