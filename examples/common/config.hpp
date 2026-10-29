// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file examples/common/config.hpp
/// @brief Start an example from the core's validated scenario.

#include <cstdio>
#include <utility>

#include "volumetric_kit/core/config/config.hpp"

namespace vgrid_example {

namespace config = volumetric_kit::core::config;

/// @brief Parse the common entry point, print help, and run validated settings.
/// @param argc Argument count.
/// @param argv Arguments.
/// @param application The core profile for this executable.
/// @param name Executable name for diagnostics.
/// @param run Application setup and work; returns its process exit status.
/// @return 0 for help, 2 for invalid configuration, or the application's
/// status.
template <typename Run>
int configured_main(int argc, char** argv, config::Application application,
                    const char* name, Run&& run) {
  const auto args = config::parse_args(argc, argv);
  if (!args) {
    std::fprintf(stderr, "%s: %s\n%s\n", name, args.status().message().c_str(),
                 config::usage());
    return 2;
  }
  if (args.value().help) {
    std::printf("%s\n", config::usage());
    return 0;
  }
  const auto settings = config::load(args.value().config_path, application);
  if (!settings) {
    std::fprintf(stderr, "%s: %s\n", name, settings.status().message().c_str());
    return 2;
  }
  return std::forward<Run>(run)(settings.value());
}

}  // namespace vgrid_example
