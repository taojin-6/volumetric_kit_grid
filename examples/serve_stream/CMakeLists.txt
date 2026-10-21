# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Tao Jin

# serve_stream: serve a recorded .vrst live over TCP, paced by its pts and
# looping (examples/common/stream_replay.hpp): a sender with no camera or GPU,
# so it builds wherever recon's examples do.
add_executable(serve_stream main.cpp)
target_include_directories(serve_stream
                           PRIVATE "${PROJECT_SOURCE_DIR}/examples/common")
target_link_libraries(serve_stream PRIVATE volumetric_kit::recon_networking
                                           volumetric_kit::recon_stream)
vr_target_warnings(serve_stream)
