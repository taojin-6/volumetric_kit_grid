# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Tao Jin

# --- recon_codec: the per-frame TSDF geometry codec --------------------------
# Encodes a grid's active blocks as a self-contained intra frame and decodes it
# back into a VoxelBlockGrid (the 2026-09-26 decision). Public: CodecParams, and
# the Encoder and Decoder built on the private pieces (encoder.cpp, decoder.cpp)
# -- the DctTransform (the GLSL forward/inverse 8^3 DCT that turns a block list
# into quantized coefficients plus an observed mask, and back), the host rANS
# reference coder (rans.hpp, header-only) and the v1 intra frame built on it
# (bitstream.cpp; the 2026-09-27 decisions). Links recon_volume ALONE -- it
# reads a grid's tsdf/weight and re-creates blocks through the map, and needs
# neither the integrator that produced the field nor the mesher that consumes
# the decoded one.
include(GenerateExportHeader)
include(GNUInstallDirs)

set(_vr_codec_generated_inc "${CMAKE_CURRENT_BINARY_DIR}/generated/include")

add_library(recon_codec)
add_library(volumetric_kit::recon_codec ALIAS recon_codec)

target_sources(recon_codec PRIVATE bitstream.cpp codec_params.cpp
                                   dct_transform.cpp decoder.cpp encoder.cpp)

# Per-tier export header (VR_CODEC_API / VR_CODEC_STATIC_DEFINE), isolated so
# the macros never collide with another tier's on one include path.
generate_export_header(
  recon_codec
  BASE_NAME
  VR_CODEC
  EXPORT_MACRO_NAME
  VR_CODEC_API
  EXPORT_FILE_NAME
  "${_vr_codec_generated_inc}/volumetric_kit/recon/codec/export.hpp")

set_target_properties(
  recon_codec
  PROPERTIES CXX_VISIBILITY_PRESET hidden
             VISIBILITY_INLINES_HIDDEN ON
             POSITION_INDEPENDENT_CODE ON)

get_target_property(_vr_codec_type recon_codec TYPE)
if(_vr_codec_type STREQUAL "STATIC_LIBRARY")
  target_compile_definitions(recon_codec PUBLIC VR_CODEC_STATIC_DEFINE)
endif()

target_compile_features(recon_codec PUBLIC cxx_std_17)

# PUBLIC: the codec headers expose volume types (VoxelBlockGrid, BlockList) and
# core types (Result, Status); recon_volume carries recon_core transitively.
target_link_libraries(recon_codec PUBLIC recon_volume)

target_include_directories(
  recon_codec
  PUBLIC "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>"
         "$<BUILD_INTERFACE:${_vr_codec_generated_inc}>"
         "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>")

# Compile the two DCT kernels to SPIR-V and embed them (dct_transform.cpp
# #includes the generated *.spv.hpp on a PRIVATE path), so recon_codec ships the
# transform with no runtime shader files.
include(vr_embed)
vr_embed_shaders(
  recon_codec SHADERS "${CMAKE_CURRENT_SOURCE_DIR}/shaders/dct_forward.comp"
  "${CMAKE_CURRENT_SOURCE_DIR}/shaders/dct_inverse.comp")

vr_target_warnings(recon_codec)

# Install the target into the shared export set + this tier's public headers
# (the codec/ subtree) + its generated export header. The private transform
# header under src/ is deliberately not installed. Mirrors the other tiers so a
# partial build never installs a header for a tier it did not build.
if(VR_INSTALL)
  install(
    TARGETS recon_codec
    EXPORT volumetric_kit_reconTargets
    ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}")
  install(
    DIRECTORY "${PROJECT_SOURCE_DIR}/include/volumetric_kit/recon/codec"
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/volumetric_kit/recon"
    FILES_MATCHING
    PATTERN "*.hpp")
  install(
    DIRECTORY "${_vr_codec_generated_inc}/"
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
    FILES_MATCHING
    PATTERN "*.hpp")
endif()
