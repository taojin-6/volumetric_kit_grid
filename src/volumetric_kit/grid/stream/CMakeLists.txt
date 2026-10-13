# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Tao Jin

# --- recon_stream: the .vrst session container --------------------------------
# A session header and the records appended after it, in tracks, each payload
# stored byte for byte, recoverable from a killed writer (the 2026-10-12
# decision). Host-only, over the family core's base tier and nothing else: no
# Vulkan, no other recon tier. A calibration travels as the camera tier's
# document, stored as bytes, and a geometry record is a codec frame, also bytes,
# so neither tier is linked.
include(GenerateExportHeader)
include(GNUInstallDirs)

set(_vr_stream_generated_inc "${CMAKE_CURRENT_BINARY_DIR}/generated/include")

add_library(recon_stream)
add_library(volumetric_kit::recon_stream ALIAS recon_stream)

target_sources(recon_stream PRIVATE chunk.cpp file.cpp stream_reader.cpp
                                    stream_writer.cpp)

generate_export_header(
  recon_stream
  BASE_NAME
  VR_STREAM
  EXPORT_MACRO_NAME
  VR_STREAM_API
  EXPORT_FILE_NAME
  "${_vr_stream_generated_inc}/volumetric_kit/recon/stream/export.hpp")

set_target_properties(
  recon_stream
  PROPERTIES CXX_VISIBILITY_PRESET hidden
             VISIBILITY_INLINES_HIDDEN ON
             POSITION_INDEPENDENT_CODE ON)

get_target_property(_vr_stream_type recon_stream TYPE)
if(_vr_stream_type STREQUAL "STATIC_LIBRARY")
  target_compile_definitions(recon_stream PUBLIC VR_STREAM_STATIC_DEFINE)
endif()

target_compile_features(recon_stream PUBLIC cxx_std_17)
# Catch allocation failures at the Status boundary even when a consuming
# application disables exceptions; a 64-bit off_t on every POSIX host, for files
# past 2 GiB. Both private: no public header reaches either.
target_compile_options(recon_stream PRIVATE -fexceptions)
target_compile_definitions(recon_stream PRIVATE _FILE_OFFSET_BITS=64)

# PUBLIC: the headers return the core's Status/Result and hold its UniqueFd.
target_link_libraries(recon_stream PUBLIC volumetric_kit::core_base)

target_include_directories(
  recon_stream
  PUBLIC "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>"
         "$<BUILD_INTERFACE:${_vr_stream_generated_inc}>"
         "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>")

vr_target_warnings(recon_stream)

if(VR_INSTALL)
  install(
    TARGETS recon_stream
    EXPORT volumetric_kit_reconTargets
    ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}")
  install(
    DIRECTORY "${PROJECT_SOURCE_DIR}/include/volumetric_kit/recon/stream"
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/volumetric_kit/recon"
    FILES_MATCHING
    PATTERN "*.hpp")
  install(
    DIRECTORY "${_vr_stream_generated_inc}/"
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
    FILES_MATCHING
    PATTERN "*.hpp")
endif()
