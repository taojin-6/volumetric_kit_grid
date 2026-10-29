// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file examples/common/interrupt.hpp
/// @brief Ctrl+C (or a SIGTERM) in the terminal ends a live example's run as
///        its own end would, rather than the process there, so the cameras
///        are released and what the run writes is finished. Killed outright,
///        the process would leave the cameras to notice the dropped
///        connection. The handler restores that signal's default, so a
///        second Ctrl+C, or a second SIGTERM, still ends a run that has hung.

#include <atomic>
#include <csignal>

namespace vgrid_example {

/// Set by the handler @ref catch_interrupts installs. Lock-free, as a signal
/// handler may only touch such an atomic.
inline std::atomic<bool> g_interrupted{false};
static_assert(std::atomic<bool>::is_always_lock_free,
              "the interrupt flag must be safe to set from a signal handler");

extern "C" inline void vgrid_example_on_interrupt(int signal) {
  g_interrupted.store(true);
  std::signal(signal, SIG_DFL);
}

/// @brief Catch SIGINT and SIGTERM: the first of either sets
///        @ref interrupted and restores that signal's default.
inline void catch_interrupts() {
  std::signal(SIGINT, vgrid_example_on_interrupt);
  std::signal(SIGTERM, vgrid_example_on_interrupt);
}

/// @return Whether a SIGINT or a SIGTERM has come since
///         @ref catch_interrupts.
inline bool interrupted() noexcept { return g_interrupted.load(); }

}  // namespace vgrid_example
