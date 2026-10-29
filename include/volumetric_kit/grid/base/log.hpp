// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file log.hpp
/// @brief grid's side of the family's one log sink.
///
/// The sink is volumetric_kit_core's: one process-wide handler for every
/// sibling, defaulting to stderr for warnings and errors. The library imposes
/// no logging framework on consumers; an application installs its own handler
/// with the core's `set_log_handler` and receives each message with its level
/// and source. grid's messages carry the source @ref kLogSource, so the default
/// sink prints `[vgrid <level>]`.

#include <string_view>

#include "volumetric_kit/core/base/log.hpp"

namespace volumetric_kit::grid {

/// @brief The source grid's diagnostics carry; the default sink prints
///        `[vgrid <level>]`.
inline constexpr std::string_view kLogSource = "vgrid";

/// @brief Emit a grid diagnostic through the family's sink, with source
///        @ref kLogSource. Thread-safe.
/// @param level    The message's severity.
/// @param message  The message; it need not be NUL-terminated.
inline void log_message(core::LogLevel level, std::string_view message) {
  core::log_message(level, kLogSource, message);
}

}  // namespace volumetric_kit::grid
