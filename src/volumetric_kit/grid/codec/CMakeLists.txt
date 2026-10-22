# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Tao Jin

# --- recon_codec: the per-frame TSDF geometry codec --------------------------
# Encodes a grid's active blocks as an intra frame, or a P-frame against the
# frame coded before it, and decodes either back into a VoxelBlockGrid (the
# 2026-09-26 decision). Public: CodecParams, the player's grid and its growth
# (decode_grid.cpp), and the Encoder and Decoder built on the private pieces
# (encoder.cpp, decoder.cpp) -- the DctTransform (the GLSL forward/inverse 8^3
# DCT that turns a block list into quantized coefficients plus an observed mask,
# and back), the rANS tables (rans.hpp, header-only), the frame's layout,
# parsing and checks (bitstream.cpp; the 2026-09-27 decisions), its device
# coders and their step layout (step_layout.cpp), the block lists' sort, merge
# and grid diff (block_lists.cpp), and the reference a P-frame is coded and
# decoded against (reference_store.cpp). Links recon_volume ALONE -- it reads a
# grid's tsdf/weight and re-creates blocks through the map, and needs neither
# the integrator that produced the field nor the mesher that consumes the
# decoded one.
include(GenerateExportHeader)

set(_vr_codec_generated_inc "${CMAKE_CURRENT_BINARY_DIR}/generated/include")

add_library(recon_codec)
add_library(volumetric_kit::recon_codec ALIAS recon_codec)

target_sources(
  recon_codec
  PRIVATE bitstream.cpp
          block_lists.cpp
          codec_params.cpp
          dct_transform.cpp
          decode_grid.cpp
          decoder.cpp
          device_frame_reader.cpp
          device_frame_writer.cpp
          encoder.cpp
          frame_hash.cpp
          reference_store.cpp
          step_layout.cpp)

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

# PUBLIC: the codec headers expose volume types (VoxelBlockGrid) and core types
# (Result, Status); recon_volume carries recon_core transitively.
target_link_libraries(recon_codec PUBLIC recon_volume)

# PRIVATE src/: recon_core's internal dispatch.hpp, as recon_volume reaches it.
target_include_directories(
  recon_codec
  PUBLIC "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>"
         "$<BUILD_INTERFACE:${_vr_codec_generated_inc}>"
  PRIVATE "${PROJECT_SOURCE_DIR}/src")

# Compile the kernels -- the DCT's, the block lists', the P-frame apply and diff
# and the rANS coders' -- to SPIR-V and embed them (their sources #include the
# generated *.spv.hpp on a PRIVATE path), so recon_codec ships with no runtime
# shader files.
include(vr_embed)
vr_embed_shaders(
  recon_codec
  SHADERS
  "${CMAKE_CURRENT_SOURCE_DIR}/shaders/block_sort_merge.comp"
  "${CMAKE_CURRENT_SOURCE_DIR}/shaders/block_sort_tile.comp"
  "${CMAKE_CURRENT_SOURCE_DIR}/shaders/dct_forward.comp"
  "${CMAKE_CURRENT_SOURCE_DIR}/shaders/dct_inverse.comp"
  "${CMAKE_CURRENT_SOURCE_DIR}/shaders/dct_observed.comp"
  "${CMAKE_CURRENT_SOURCE_DIR}/shaders/frame_merge.comp"
  "${CMAKE_CURRENT_SOURCE_DIR}/shaders/grid_diff.comp"
  "${CMAKE_CURRENT_SOURCE_DIR}/shaders/inter_apply.comp"
  "${CMAKE_CURRENT_SOURCE_DIR}/shaders/inter_diff.comp"
  "${CMAKE_CURRENT_SOURCE_DIR}/shaders/rans_count.comp"
  "${CMAKE_CURRENT_SOURCE_DIR}/shaders/rans_decode.comp"
  "${CMAKE_CURRENT_SOURCE_DIR}/shaders/rans_ops.comp"
  "${CMAKE_CURRENT_SOURCE_DIR}/shaders/rans_encode.comp"
  "${CMAKE_CURRENT_SOURCE_DIR}/shaders/rans_layout.comp"
  "${CMAKE_CURRENT_SOURCE_DIR}/shaders/rans_scan.comp"
  "${CMAKE_CURRENT_SOURCE_DIR}/shaders/rans_slots.comp"
  "${CMAKE_CURRENT_SOURCE_DIR}/shaders/rans_gather.comp"
  "${CMAKE_CURRENT_SOURCE_DIR}/shaders/rans_verdict.comp"
  "${CMAKE_CURRENT_SOURCE_DIR}/shaders/survivor_ranks.comp")

vr_target_warnings(recon_codec)
