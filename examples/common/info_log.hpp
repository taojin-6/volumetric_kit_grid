// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file examples/common/info_log.hpp
/// @brief The senders' log: the core's default sink prints Warning and Error
///        only, so serve_stream and fuse_orbbec install a handler that prints
///        Info too, as networking::StreamServer logs each viewer joining and
///        leaving at Info.

#include <cstdio>
#include <string>
#include <string_view>

#include "volumetric_kit/core/base/log.hpp"

namespace vgrid_example {

/// @brief A log handler that writes each Info, Warning and Error message to
///        @p out as the core's default sink writes a line,
///        `[<source> <level>] <message>`, and drops Debug.
///
/// Each line goes out in one fwrite, so concurrent lines do not interleave.
inline volumetric_kit::core::LogHandler info_log_handler(std::FILE* out) {
  using volumetric_kit::core::LogLevel;
  return [out](LogLevel level, std::string_view source, std::string_view text) {
    if (level == LogLevel::Debug) return;
    const char* name = level == LogLevel::Info      ? "info"
                       : level == LogLevel::Warning ? "warning"
                                                    : "error";
    std::string line = "[";
    line.append(source.data(), source.size());
    line += ' ';
    line += name;
    line += "] ";
    line.append(text.data(), text.size());
    line += '\n';
    std::fwrite(line.data(), 1, line.size(), out);
  };
}

/// @brief Install @ref info_log_handler on stderr for the process.
inline void print_info_log() {
  volumetric_kit::core::set_log_handler(info_log_handler(stderr));
}

}  // namespace vgrid_example
