# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Tao Jin

# --- grid_networking: a session served and read live over TCP -----------------
# A sender serves one session's chunks, the .vrst file's own bytes, to viewers
# as it records; a viewer reads them as they arrive (the 2026-10-21 decision).
# Host-only, over grid_stream (the chunks: ChunkEncoder and ChunkParser) and the
# family core's base tier, with BSD sockets and threads of its own. Its own
# tier, apart from the format, so the sockets are tested apart from it. It logs
# through grid's log seam (base/log.hpp), header-only over that base tier.
include(GenerateExportHeader)

set(_vgrid_networking_generated_inc
    "${CMAKE_CURRENT_BINARY_DIR}/generated/include")

add_library(grid_networking)
add_library(volumetric_kit::grid_networking ALIAS grid_networking)

target_sources(grid_networking PRIVATE gop_log.cpp socket.cpp stream_client.cpp
                                       stream_server.cpp)

generate_export_header(
  grid_networking
  BASE_NAME
  VGRID_NETWORKING
  EXPORT_MACRO_NAME
  VGRID_NETWORKING_API
  EXPORT_FILE_NAME
  "${_vgrid_networking_generated_inc}/volumetric_kit/grid/networking/export.hpp"
)

set_target_properties(
  grid_networking
  PROPERTIES CXX_VISIBILITY_PRESET hidden
             VISIBILITY_INLINES_HIDDEN ON
             POSITION_INDEPENDENT_CODE ON)

get_target_property(_vgrid_networking_type grid_networking TYPE)
if(_vgrid_networking_type STREQUAL "STATIC_LIBRARY")
  target_compile_definitions(grid_networking
                             PUBLIC VGRID_NETWORKING_STATIC_DEFINE)
endif()

target_compile_features(grid_networking PUBLIC cxx_std_17)
# Catch allocation failures at the Status boundary, and in the server's threads,
# even when a consuming application disables exceptions. Private: no public
# header reaches it.
target_compile_options(grid_networking PRIVATE -fexceptions)

# PUBLIC: the headers return the core's Status/Result and the stream's Session
# and Record. The server and the client run threads of their own, so a static
# archive's link interface names Threads.
find_package(Threads REQUIRED)
target_link_libraries(
  grid_networking
  PUBLIC volumetric_kit::grid_stream volumetric_kit::core_base
  PRIVATE Threads::Threads)

target_include_directories(
  grid_networking
  PUBLIC "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>"
         "$<BUILD_INTERFACE:${_vgrid_networking_generated_inc}>")

vgrid_target_warnings(grid_networking)
