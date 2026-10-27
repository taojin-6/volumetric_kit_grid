// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// The examples' one stage table (examples/common/stage_table.hpp): every row
// divided by the unit count, host busy beside its wait, the device column a
// number, n/t, n/a or a dash as the row's device half says, breakdown rows
// keeping their indent, the indent applied to title and rows, nothing for no
// rows or no units, a row found by name, and a figure's mean, nearest-rank
// p50 and p95 and max. Host only.

#include <cstdio>
#include <string>
#include <vector>

#include "stage_table.hpp"
#include "test_check.hpp"

namespace vkc = volumetric_kit::core;

int main() {
  vkc::StageMetrics rows;
  rows.add_cpu("frame prep", 4.0);
  rows.add_wait("frame prep", 1.0);
  rows.add_gpu("frame prep", 1.0);
  rows.add_cpu("integrate", 10.0);
  rows.add_wait("integrate", 6.0);
  rows.add_device("integrate", vkc::DeviceTime::NotTimed);
  rows.add_cpu("  ..active set", 2.0);
  rows.add_gpu("  ..active set", 1.5);
  rows.add_cpu("readback", 1.0);
  rows.add_device("readback", vkc::DeviceTime::Unavailable);
  rows.add_cpu("poll", 0.5);
  rows.seed("texture");

  const std::string table =
      vr_example::format_stage_rows("stages per frame", rows, 2);
  std::fputs(table.c_str(), stdout);
  CHECK(table ==
        "stages per frame (ms: host busy / wait / device):\n"
        "  frame prep             1.500     0.500     0.500\n"
        "  integrate              2.000     3.000       n/t\n"
        "    ..active set         1.000     0.000     0.750\n"
        "  readback               0.500     0.000       n/a\n"
        "  poll                   0.250     0.000         -\n"
        "  texture                0.000     0.000         -\n");

  // An indent moves the title and the rows alike.
  vkc::StageMetrics idle;
  idle.add_gpu("codec encode", 0.0);
  CHECK(vr_example::format_stage_rows("encode", idle, 1, 2) ==
        "  encode (ms: host busy / wait / device):\n"
        "    codec encode           0.000     0.000     0.000\n");

  // Nothing to say: no rows, or no units to divide by.
  CHECK(vr_example::format_stage_rows("x", vkc::StageMetrics{}, 3).empty());
  CHECK(vr_example::format_stage_rows("x", rows, 0).empty());

  // A row by name, or null.
  const vkc::StageRow* row = vr_example::find_row(rows, "frame prep");
  CHECK(row != nullptr && row->cpu_ms == 4.0 &&
        row->device == vkc::DeviceTime::Measured);
  CHECK(vr_example::find_row(rows, "extract") == nullptr);

  // A figure over many units: the mean and the nearest-rank p50 and p95 of
  // the values in any order, and the max.
  CHECK(vr_example::format_stats({4.0, 1.0, 3.0, 2.0}) ==
        "   2.500    2.000    4.000    4.000");
  CHECK(vr_example::format_stats({0.25}) ==
        "   0.250    0.250    0.250    0.250");
  std::vector<double> ramp;
  for (int i = 1; i <= 100; ++i) ramp.push_back(i);
  CHECK(vr_example::format_stats(ramp) ==
        "  50.500   50.000   95.000  100.000");
  std::puts("example_stage_table: OK");
  return 0;
}
