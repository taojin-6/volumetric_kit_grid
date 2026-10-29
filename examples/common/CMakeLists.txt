# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Tao Jin

# grid_example_common: the examples' scaffolding (config.hpp, info_log.hpp,
# interrupt.hpp, stage_table.hpp) and the codec examples' helpers
# (codec_config.hpp, codec_quantization.hpp, codec_report.hpp, codec_sweep.hpp,
# stream_replay.hpp), headers in namespace vgrid_example, built with the
# examples or the tests. grid's examples and the tests of these helpers link it,
# and so do a sibling's, recon's among them, which reach the helpers here. It
# carries the include path alone: a target links the core's config tier for
# config.hpp, and the grid tiers the helpers it includes use.
add_library(grid_example_common INTERFACE)
target_include_directories(grid_example_common
                           INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}")
target_compile_features(grid_example_common INTERFACE cxx_std_17)
