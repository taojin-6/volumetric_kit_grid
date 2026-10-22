# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Tao Jin

# --- recon_networking: a session served and read live over TCP ---------------
# A sender serves one session's chunks, the .vrst file's own bytes, to viewers
# as it records; a viewer reads them as they arrive (the 2026-10-21 decision).
# Host-only, over recon_stream (the chunks: ChunkEncoder and ChunkParser) and
# the family core's base tier, with BSD sockets and threads of its own. Its own
# tier, apart from the format, so the sockets are tested apart from it. It logs
# through recon's log seam (core/log.hpp), header-only over that base tier.
include(GenerateExportHeader)
include(GNUInstallDirs)

set(_vr_networking_generated_inc
    "${CMAKE_CURRENT_BINARY_DIR}/generated/include")

add_library(recon_networking)
add_library(volumetric_kit::recon_networking ALIAS recon_networking)

target_sources(recon_networking PRIVATE gop_log.cpp socket.cpp
                                        stream_client.cpp stream_server.cpp)

generate_export_header(
  recon_networking
  BASE_NAME
  VR_NETWORKING
  EXPORT_MACRO_NAME
  VR_NETWORKING_API
  EXPORT_FILE_NAME
  "${_vr_networking_generated_inc}/volumetric_kit/recon/networking/export.hpp")

set_target_properties(
  recon_networking
  PROPERTIES CXX_VISIBILITY_PRESET hidden
             VISIBILITY_INLINES_HIDDEN ON
             POSITION_INDEPENDENT_CODE ON)

get_target_property(_vr_networking_type recon_networking TYPE)
if(_vr_networking_type STREQUAL "STATIC_LIBRARY")
  target_compile_definitions(recon_networking
                             PUBLIC VR_NETWORKING_STATIC_DEFINE)
endif()

target_compile_features(recon_networking PUBLIC cxx_std_17)
# Catch allocation failures at the Status boundary, and in the server's threads,
# even when a consuming application disables exceptions. Private: no public
# header reaches it.
target_compile_options(recon_networking PRIVATE -fexceptions)

# PUBLIC: the headers return the core's Status/Result and the stream's Session
# and Record. The server and the client run threads of their own; the package
# config re-finds Threads, which a static archive's link interface names.
find_package(Threads REQUIRED)
target_link_libraries(
  recon_networking
  PUBLIC volumetric_kit::recon_stream volumetric_kit::core_base
  PRIVATE Threads::Threads)

target_include_directories(
  recon_networking
  PUBLIC "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>"
         "$<BUILD_INTERFACE:${_vr_networking_generated_inc}>"
         "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>")

vr_target_warnings(recon_networking)

if(VR_INSTALL)
  install(
    TARGETS recon_networking
    EXPORT volumetric_kit_reconTargets
    ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}")
  install(
    DIRECTORY "${PROJECT_SOURCE_DIR}/include/volumetric_kit/recon/networking"
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/volumetric_kit/recon"
    FILES_MATCHING
    PATTERN "*.hpp")
  install(
    DIRECTORY "${_vr_networking_generated_inc}/"
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
    FILES_MATCHING
    PATTERN "*.hpp")
endif()
