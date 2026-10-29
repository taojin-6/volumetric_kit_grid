# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Tao Jin

# grid_test_support: the helpers grid's tests share with a sibling's, as headers
# (include/volumetric_kit/grid/testing/): a grid's blocks made and read on the
# device (device_blocks.hpp, grid_readback.hpp), a sphere's field
# (sphere_scene.hpp), a codec test's device and grids (codec_fixture.hpp), a
# stage's rows (stage_rows.hpp), a .vrst file's bytes (stream_files.hpp) and a
# server and its clients on loopback (networking_loopback.hpp). Built with the
# tests, or for a sibling with VGRID_BUILD_TEST_SUPPORT. It carries the include
# path alone, so a host-only test stays Vulkan-free: a test links the grid tiers
# its helpers use and the core's test support, core_test_base and, for one that
# reads the device, core_test_policy.
add_library(grid_test_support INTERFACE)
add_library(volumetric_kit::grid_test_support ALIAS grid_test_support)
target_include_directories(grid_test_support
                           INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/include")
