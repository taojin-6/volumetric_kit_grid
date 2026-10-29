# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Tao Jin

# serve_stream: serve a recorded .vrst live over TCP, paced by its pts and
# looping (examples/common/stream_replay.hpp): a sender with no camera or GPU,
# so it builds wherever grid's examples do.
add_executable(serve_stream main.cpp)
target_link_libraries(
  serve_stream PRIVATE grid_example_common volumetric_kit::grid_networking
                       volumetric_kit::grid_stream)
vgrid_target_warnings(serve_stream)
