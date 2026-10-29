# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Tao Jin

# --- grid_codec: the per-frame TSDF geometry codec ----------------------------
# Encodes a grid's active blocks as an intra frame, or a P-frame against the
# frame coded before it, and decodes either back into a VoxelBlockGrid (the
# 2026-09-26 decision). Public: CodecParams, the player's grid and its growth
# (decode_grid.cpp), and the Encoder and Decoder built on the private pieces
# (encoder.cpp, decoder.cpp) -- the DctTransform (the GLSL forward/inverse 8^3
# DCT that turns a block list into quantized coefficients plus an observed mask,
# and back), the rANS tables (rans.hpp, header-only), the frame's layout,
# parsing and checks (bitstream.cpp; the 2026-09-27 decisions), its device
# coders and their step layout (step_layout.cpp), the block lists' sort, merge
# and grid diff (block_lists.cpp), the reference a P-frame is coded and decoded
# against (reference_store.cpp), and the decoder's gate and results on the
# device (frame_gate.cpp). Links grid_volume ALONE -- it reads a grid's
# tsdf/weight and re-creates blocks through the map, and needs neither the
# integrator that produced the field nor the mesher that consumes the decoded
# one.
include(GenerateExportHeader)

set(_vgrid_codec_generated_inc "${CMAKE_CURRENT_BINARY_DIR}/generated/include")

add_library(grid_codec)
add_library(volumetric_kit::grid_codec ALIAS grid_codec)

target_sources(
  grid_codec
  PRIVATE bitstream.cpp
          block_lists.cpp
          codec_params.cpp
          dct_transform.cpp
          deblock.cpp
          decode_grid.cpp
          decoder.cpp
          device_frame_reader.cpp
          device_frame_writer.cpp
          encoder.cpp
          frame_gate.cpp
          frame_hash.cpp
          reference_store.cpp
          step_layout.cpp)

# Per-tier export header (VGRID_CODEC_API / VGRID_CODEC_STATIC_DEFINE), isolated
# so the macros never collide with another tier's on one include path.
generate_export_header(
  grid_codec
  BASE_NAME
  VGRID_CODEC
  EXPORT_MACRO_NAME
  VGRID_CODEC_API
  EXPORT_FILE_NAME
  "${_vgrid_codec_generated_inc}/volumetric_kit/grid/codec/export.hpp")

set_target_properties(
  grid_codec
  PROPERTIES CXX_VISIBILITY_PRESET hidden
             VISIBILITY_INLINES_HIDDEN ON
             POSITION_INDEPENDENT_CODE ON)

get_target_property(_vgrid_codec_type grid_codec TYPE)
if(_vgrid_codec_type STREQUAL "STATIC_LIBRARY")
  target_compile_definitions(grid_codec PUBLIC VGRID_CODEC_STATIC_DEFINE)
endif()

target_compile_features(grid_codec PUBLIC cxx_std_17)

# PUBLIC: the codec headers expose volume types (VoxelBlockGrid) and core types
# (Result, Status); grid_volume carries grid_base transitively.
target_link_libraries(grid_codec PUBLIC grid_volume)

# PRIVATE src/: volume's internal hash_map_internal.hpp (the hash entries).
target_include_directories(
  grid_codec
  PUBLIC "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>"
         "$<BUILD_INTERFACE:${_vgrid_codec_generated_inc}>"
  PRIVATE "${PROJECT_SOURCE_DIR}/src")

# Compile the kernels -- the DCT's, the deblock's, the block lists', the P-frame
# apply and diff and the rANS coders' -- to SPIR-V and embed them (their sources
# #include the generated *.spv.hpp on a PRIVATE path), so grid_codec ships with
# no runtime shader files.
include(vgrid_embed)
vgrid_embed_shaders(
  grid_codec
  SHADERS
  "${CMAKE_CURRENT_SOURCE_DIR}/shaders/block_sort_merge.comp"
  "${CMAKE_CURRENT_SOURCE_DIR}/shaders/block_sort_tile.comp"
  "${CMAKE_CURRENT_SOURCE_DIR}/shaders/dct_forward.comp"
  "${CMAKE_CURRENT_SOURCE_DIR}/shaders/dct_inverse.comp"
  "${CMAKE_CURRENT_SOURCE_DIR}/shaders/dct_observed.comp"
  "${CMAKE_CURRENT_SOURCE_DIR}/shaders/dct_surface.comp"
  "${CMAKE_CURRENT_SOURCE_DIR}/shaders/deblock_flags.comp"
  "${CMAKE_CURRENT_SOURCE_DIR}/shaders/deblock_gauss.comp"
  "${CMAKE_CURRENT_SOURCE_DIR}/shaders/deblock_line.comp"
  "${CMAKE_CURRENT_SOURCE_DIR}/shaders/deblock_store.comp"
  "${CMAKE_CURRENT_SOURCE_DIR}/shaders/encode_results.comp"
  "${CMAKE_CURRENT_SOURCE_DIR}/shaders/frame_gate.comp"
  "${CMAKE_CURRENT_SOURCE_DIR}/shaders/frame_merge.comp"
  "${CMAKE_CURRENT_SOURCE_DIR}/shaders/frame_results.comp"
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

vgrid_target_warnings(grid_codec)
