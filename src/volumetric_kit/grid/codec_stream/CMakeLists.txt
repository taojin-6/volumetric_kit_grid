# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Tao Jin

# --- grid_codec_stream: a codec stream ----------------------------------------
# Grids coded as frames, each written to a .vrst file, served to live viewers
# and decoded back as a viewer would (CodecStream), a frame whole or in halves
# around a caller's batch, with Audio tracks' records muxed in by pts
# (record_mux). The tier after networking: it links the codec, the container and
# the server, and neither the fuser nor the sensor tiers.
include(GenerateExportHeader)

set(_vgrid_codec_stream_generated_inc
    "${CMAKE_CURRENT_BINARY_DIR}/generated/include")

add_library(grid_codec_stream)
add_library(volumetric_kit::grid_codec_stream ALIAS grid_codec_stream)

target_sources(grid_codec_stream PRIVATE codec_stream.cpp record_mux.cpp)

generate_export_header(
  grid_codec_stream
  BASE_NAME
  VGRID_CODEC_STREAM
  EXPORT_MACRO_NAME
  VGRID_CODEC_STREAM_API
  EXPORT_FILE_NAME
  "${_vgrid_codec_stream_generated_inc}/volumetric_kit/grid/codec_stream/export.hpp"
)

set_target_properties(
  grid_codec_stream
  PROPERTIES CXX_VISIBILITY_PRESET hidden
             VISIBILITY_INLINES_HIDDEN ON
             POSITION_INDEPENDENT_CODE ON)

get_target_property(_vgrid_codec_stream_type grid_codec_stream TYPE)
if(_vgrid_codec_stream_type STREQUAL "STATIC_LIBRARY")
  target_compile_definitions(grid_codec_stream
                             PUBLIC VGRID_CODEC_STREAM_STATIC_DEFINE)
endif()

target_compile_features(grid_codec_stream PUBLIC cxx_std_17)

# PUBLIC: the header takes the encoder's settings and pending frame, the grid,
# the session and the server's options.
target_link_libraries(grid_codec_stream PUBLIC grid_codec grid_stream
                                               grid_networking)

target_include_directories(
  grid_codec_stream
  PUBLIC "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>"
         "$<BUILD_INTERFACE:${_vgrid_codec_stream_generated_inc}>")

vgrid_target_warnings(grid_codec_stream)
