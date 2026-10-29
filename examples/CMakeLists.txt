# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Tao Jin

# The examples: codec_mesh, which codes a mesh or a numbered sequence of them
# (with the Assimp loader), and serve_stream, which serves a recording live.
# Their scaffolding and the codec helpers are grid_example_common (common/,
# which the root adds).

if(TARGET grid_mesh_io_assimp)
  add_subdirectory(codec_mesh)
endif()
# A recording served live, with no camera or GPU.
add_subdirectory(serve_stream)

# Exercise each real executable's startup contract without creating resources.
if(VGRID_BUILD_TESTS)
  if(TARGET codec_mesh)
    add_test(
      NAME grid_codec_mesh_config_entry
      COMMAND
        ${CMAKE_COMMAND} -DPROGRAM=$<TARGET_FILE:codec_mesh>
        -DWORK_DIR=${CMAKE_CURRENT_BINARY_DIR}/config-entry/codec_mesh -P
        ${PROJECT_SOURCE_DIR}/tests/example_config_entry.cmake)
  endif()
  # serve_stream's arguments: usage and exit 2 for bad ones, 1 for a missing
  # file, before it listens; it is never run to serve, which would listen on
  # every interface.
  add_test(
    NAME grid_serve_stream_args
    COMMAND
      ${CMAKE_COMMAND} -DPROGRAM=$<TARGET_FILE:serve_stream>
      -DWORK_DIR=${CMAKE_CURRENT_BINARY_DIR}/serve-stream-args -P
      ${PROJECT_SOURCE_DIR}/tests/example_serve_stream_args.cmake)
  # codec_mesh's order on a device: an existing stream refused before any frame
  # is converted or written, then a two-frame run.
  if(TARGET codec_mesh)
    add_test(
      NAME grid_codec_mesh_run
      COMMAND
        ${CMAKE_COMMAND} -DPROGRAM=$<TARGET_FILE:codec_mesh>
        -DWORK_DIR=${CMAKE_CURRENT_BINARY_DIR}/codec-mesh-run -P
        ${PROJECT_SOURCE_DIR}/tests/example_codec_mesh_run.cmake)
    set_tests_properties(
      grid_codec_mesh_run PROPERTIES SKIP_REGULAR_EXPRESSION
                                     "skipping: no Vulkan device")
  endif()
endif()
