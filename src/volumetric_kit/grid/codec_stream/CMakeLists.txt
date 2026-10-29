# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Tao Jin

# --- recon_codec_stream: a codec stream ---------------------------------------
# Grids coded as frames, each written to a .vrst file, served to live viewers
# and decoded back as a viewer would (CodecStream), a frame whole or in halves
# around a caller's batch. The tier after networking: it links the codec, the
# container and the server, and neither the fuser nor the sensor tiers.
include(GenerateExportHeader)

set(_vr_codec_stream_generated_inc
    "${CMAKE_CURRENT_BINARY_DIR}/generated/include")

add_library(recon_codec_stream)
add_library(volumetric_kit::recon_codec_stream ALIAS recon_codec_stream)

target_sources(recon_codec_stream PRIVATE codec_stream.cpp)

generate_export_header(
  recon_codec_stream
  BASE_NAME
  VR_CODEC_STREAM
  EXPORT_MACRO_NAME
  VR_CODEC_STREAM_API
  EXPORT_FILE_NAME
  "${_vr_codec_stream_generated_inc}/volumetric_kit/recon/codec_stream/export.hpp"
)

set_target_properties(
  recon_codec_stream
  PROPERTIES CXX_VISIBILITY_PRESET hidden
             VISIBILITY_INLINES_HIDDEN ON
             POSITION_INDEPENDENT_CODE ON)

get_target_property(_vr_codec_stream_type recon_codec_stream TYPE)
if(_vr_codec_stream_type STREQUAL "STATIC_LIBRARY")
  target_compile_definitions(recon_codec_stream
                             PUBLIC VR_CODEC_STREAM_STATIC_DEFINE)
endif()

target_compile_features(recon_codec_stream PUBLIC cxx_std_17)

# PUBLIC: the header takes the encoder's settings and pending frame, the grid,
# the session and the server's options.
target_link_libraries(recon_codec_stream PUBLIC recon_codec recon_stream
                                                recon_networking)

target_include_directories(
  recon_codec_stream
  PUBLIC "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>"
         "$<BUILD_INTERFACE:${_vr_codec_stream_generated_inc}>")

vr_target_warnings(recon_codec_stream)
