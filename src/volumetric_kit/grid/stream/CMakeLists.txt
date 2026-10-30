# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Tao Jin

# --- grid_stream: the .vrst session container ---------------------------------
# A session header and the records appended after it, in tracks, each payload
# stored byte for byte, recoverable from a killed writer (the 2026-10-12
# decision). Host-only, over the family core's base tier and nothing else: no
# Vulkan, no other grid tier. A calibration travels as the camera tier's
# document, stored as bytes, and a geometry record is a codec frame, also bytes,
# so neither tier is linked.
include(GenerateExportHeader)

set(_vgrid_stream_generated_inc "${CMAKE_CURRENT_BINARY_DIR}/generated/include")

add_library(grid_stream)
add_library(volumetric_kit::grid_stream ALIAS grid_stream)

target_sources(grid_stream PRIVATE audio.cpp chunk.cpp chunks.cpp file.cpp
                                   stream_reader.cpp stream_writer.cpp)

generate_export_header(
  grid_stream
  BASE_NAME
  VGRID_STREAM
  EXPORT_MACRO_NAME
  VGRID_STREAM_API
  EXPORT_FILE_NAME
  "${_vgrid_stream_generated_inc}/volumetric_kit/grid/stream/export.hpp")

set_target_properties(
  grid_stream
  PROPERTIES CXX_VISIBILITY_PRESET hidden
             VISIBILITY_INLINES_HIDDEN ON
             POSITION_INDEPENDENT_CODE ON)

get_target_property(_vgrid_stream_type grid_stream TYPE)
if(_vgrid_stream_type STREQUAL "STATIC_LIBRARY")
  target_compile_definitions(grid_stream PUBLIC VGRID_STREAM_STATIC_DEFINE)
endif()

target_compile_features(grid_stream PUBLIC cxx_std_17)
# Catch allocation failures at the Status boundary even when a consuming
# application disables exceptions; a 64-bit off_t on every POSIX host, for files
# past 2 GiB. Both private: no public header reaches either.
target_compile_options(grid_stream PRIVATE -fexceptions)
target_compile_definitions(grid_stream PRIVATE _FILE_OFFSET_BITS=64)

# PUBLIC: the headers return the core's Status/Result and hold its UniqueFd.
target_link_libraries(grid_stream PUBLIC volumetric_kit::core_base)

target_include_directories(
  grid_stream PUBLIC "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>"
                     "$<BUILD_INTERFACE:${_vgrid_stream_generated_inc}>")

vgrid_target_warnings(grid_stream)
