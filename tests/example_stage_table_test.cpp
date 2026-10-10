// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// The examples' one stage table (examples/common/stage_table.hpp): every row
// divided by the unit count, the device column and its share where a timer
// ran and a dash where none did, breakdown rows keeping their indent, the
// indent applied to title and rows, nothing for no rows or no units, and a
// row found by name. Host only.

#include <cstdio>
#include <string>

#include "stage_table.hpp"

namespace vkc = volumetric_kit::core;

#define CHECK(cond)                                                        \
  do {                                                                     \
    if (!(cond)) {                                                         \
      std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
      return 1;                                                            \
    }                                                                      \
  } while (0)

int main() {
  vkc::StageMetrics rows;
  rows.add_cpu("frame prep", 4.0);
  rows.add_gpu("frame prep", 1.0);
  rows.add_cpu("integrate", 10.0);
  rows.add_gpu("integrate", 8.0);
  rows.add_cpu("  ..active set", 2.0);
  rows.add_gpu("  ..active set", 1.5);
  rows.add_cpu("poll", 0.5);
  rows.seed("texture");

  const std::string table =
      vr_example::format_stage_rows("stages per frame", rows, 2);
  std::fputs(table.c_str(), stdout);
  CHECK(table ==
        "stages per frame (host ms / device ms):\n"
        "  frame prep             2.000     0.500   25.0%\n"
        "  integrate              5.000     4.000   80.0%\n"
        "    ..active set         1.000     0.750   75.0%\n"
        "  poll                   0.250         -\n"
        "  texture                0.000         -\n");

  // An indent moves the title and the rows alike; a zero host span has no
  // share to speak of.
  vkc::StageMetrics idle;
  idle.add_gpu("codec encode", 0.0);
  CHECK(vr_example::format_stage_rows("encode", idle, 1, 2) ==
        "  encode (host ms / device ms):\n"
        "    codec encode           0.000     0.000    0.0%\n");

  // Nothing to say: no rows, or no units to divide by.
  CHECK(vr_example::format_stage_rows("x", vkc::StageMetrics{}, 3).empty());
  CHECK(vr_example::format_stage_rows("x", rows, 0).empty());

  // A row by name, or null.
  const vkc::StageRow* row = vr_example::find_row(rows, "integrate");
  CHECK(row != nullptr && row->cpu_ms == 10.0 && row->has_gpu);
  CHECK(vr_example::find_row(rows, "extract") == nullptr);
  std::puts("example_stage_table: OK");
  return 0;
}
