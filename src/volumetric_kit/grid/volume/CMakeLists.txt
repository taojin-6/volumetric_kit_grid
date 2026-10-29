# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Tao Jin

# --- recon_volume: Tier 2 ---------------------------------------------------
# The sparse voxel hash map: the POD data model (hash_types.hpp,
# voxel_grid.hpp), the host coordinate math (voxel_coords.hpp), and the host
# VoxelHashMap that owns the device buffers + compute pipelines and dispatches
# the GLSL hash kernels (init / allocate-from-coords / compact). The table's
# internal layout (hash.hpp) stays under src/. Now a compiled STATIC library
# (was header-only) since VoxelHashMap + its embedded shaders landed. Depends
# only on recon_core, per the strict left-to-right tier rule.
include(GenerateExportHeader)

set(_vr_volume_generated_inc "${CMAKE_CURRENT_BINARY_DIR}/generated/include")

add_library(recon_volume)
add_library(volumetric_kit::recon_volume ALIAS recon_volume)

target_sources(recon_volume PRIVATE grid_growth.cpp triangle_candidates.cpp
                                    voxel_hash_map.cpp voxel_block_grid.cpp)

# Per-tier export header (VR_VOLUME_API / VR_VOLUME_STATIC_DEFINE), isolated
# from recon_core's so the macros never collide on one include path.
generate_export_header(
  recon_volume
  BASE_NAME
  VR_VOLUME
  EXPORT_MACRO_NAME
  VR_VOLUME_API
  EXPORT_FILE_NAME
  "${_vr_volume_generated_inc}/volumetric_kit/recon/volume/export.hpp")

set_target_properties(
  recon_volume
  PROPERTIES CXX_VISIBILITY_PRESET hidden
             VISIBILITY_INLINES_HIDDEN ON
             POSITION_INDEPENDENT_CODE ON)

get_target_property(_vr_volume_type recon_volume TYPE)
if(_vr_volume_type STREQUAL "STATIC_LIBRARY")
  target_compile_definitions(recon_volume PUBLIC VR_VOLUME_STATIC_DEFINE)
endif()

target_compile_features(recon_volume PUBLIC cxx_std_17)

# PUBLIC: the volume public headers (voxel_hash_map.hpp) expose core types
# (Result, Buffer, ComputePipeline, ...), so consumers compile against
# recon_core too. recon_core carries the Vulkan/GLM include wiring transitively.
target_link_libraries(recon_volume PUBLIC recon_core)

# PRIVATE src/: recon_core's internal dispatch.hpp, by root-relative path, as
# the shaders include another tier's GLSL.
target_include_directories(
  recon_volume
  PUBLIC "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>"
         "$<BUILD_INTERFACE:${_vr_volume_generated_inc}>"
  PRIVATE "${PROJECT_SOURCE_DIR}/src")

# Compile the GLSL hash kernels to SPIR-V and embed them as constexpr byte
# arrays into the library (voxel_hash_map.cpp #includes the generated *.spv.hpp
# on a PRIVATE path), so recon_volume ships the technique with no runtime shader
# files.
include(vr_embed)
vr_embed_shaders(
  recon_volume
  SHADERS
  "${CMAKE_CURRENT_SOURCE_DIR}/shaders/block_stamp.comp"
  "${CMAKE_CURRENT_SOURCE_DIR}/shaders/block_zero.comp"
  "${CMAKE_CURRENT_SOURCE_DIR}/shaders/hash_init.comp"
  "${CMAKE_CURRENT_SOURCE_DIR}/shaders/hash_allocate_coords.comp"
  "${CMAKE_CURRENT_SOURCE_DIR}/shaders/hash_allocate_depth.comp"
  "${CMAKE_CURRENT_SOURCE_DIR}/shaders/hash_allocate_triangles.comp"
  "${CMAKE_CURRENT_SOURCE_DIR}/shaders/hash_compact.comp"
  "${CMAKE_CURRENT_SOURCE_DIR}/shaders/hash_compact_frustum.comp"
  "${CMAKE_CURRENT_SOURCE_DIR}/shaders/hash_delete_coords.comp"
  "${CMAKE_CURRENT_SOURCE_DIR}/shaders/hash_rehash.comp"
  "${CMAKE_CURRENT_SOURCE_DIR}/shaders/hash_rounds.comp")

# The grid's device contract and its colour attribute's format, for any kernel's
# #include ("volumetric_kit/recon/volume/shaders/grid_types.glsl") through the
# core's INCLUDE_TARGETS volumetric_kit::recon_volume, as recon's tsdf does. The
# hash table's GLSL (hash_common.glsl, hash_lookup.glsl, ...) stays internal:
# the codec, mesh and voxelize kernels read it through src/.
vkc_export_glsl(recon_volume PREFIX volumetric_kit/recon/volume/shaders FILES
                shaders/grid_types.glsl shaders/voxel_color.glsl)

vr_target_warnings(recon_volume)
