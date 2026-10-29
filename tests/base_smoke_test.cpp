// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// The base tier's own pieces: the version CMake wires through, and the log
// seam's "vgrid" source. Status, Result and the GLM types are the core's and
// GLM's, tested there; including vector_types.hpp proves grid's aliases
// compile. Returns non-zero on any failed check, so CTest reports it.

#include <cstdio>
#include <string>
#include <string_view>

#include "volumetric_kit/grid/base/log.hpp"
#include "volumetric_kit/grid/base/math/vector_types.hpp"
#include "volumetric_kit/grid/version.hpp"

namespace vgrid = volumetric_kit::grid;
namespace vkc = volumetric_kit::core;

namespace {

int g_failures = 0;

void check(bool cond, const char* what) {
  if (!cond) {
    std::fprintf(stderr, "FAIL: %s\n", what);
    ++g_failures;
  }
}

}  // namespace

int main() {
  // Version is wired through CMake: the integer accessors must compose to the
  // string form. This exercises all three accessors -- a swapped
  // MAJOR/MINOR/PATCH is caught, since version_string() comes from the
  // independent PROJECT_VERSION macro -- without hard-coding a literal that a
  // version bump would invalidate.
  const std::string composed = std::to_string(vgrid::version_major()) + "." +
                               std::to_string(vgrid::version_minor()) + "." +
                               std::to_string(vgrid::version_patch());
  check(composed == vgrid::version_string(),
        "version components compose to version_string");

  // Logging seam: an installed handler receives every level (including Info,
  // which the default sink drops) with grid's source; restoring the empty
  // handler falls back to the default sink.
  vkc::LogLevel seen_level = vkc::LogLevel::Error;
  std::string seen_source;
  std::string seen_message;
  int seen_count = 0;
  vkc::set_log_handler([&](vkc::LogLevel level, std::string_view source,
                           std::string_view message) {
    seen_level = level;
    seen_source.assign(source.data(), source.size());
    seen_message.assign(message.data(), message.size());
    ++seen_count;
  });
  vgrid::log_message(vkc::LogLevel::Info, "smoke test ran");
  check(seen_count == 1 && seen_level == vkc::LogLevel::Info &&
            seen_source == "vgrid" && seen_message == "smoke test ran",
        "installed log handler receives the message with source \"vgrid\"");
  vkc::set_log_handler({});  // restore the default sink
  vgrid::log_message(vkc::LogLevel::Info, "after reset");
  check(seen_count == 1, "empty handler restores default sink (Info dropped)");

  if (g_failures == 0) {
    std::puts("grid_base smoke test passed");
    return 0;
  }
  std::fprintf(stderr, "%d checks failed\n", g_failures);
  return 1;
}
