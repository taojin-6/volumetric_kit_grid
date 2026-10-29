# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Tao Jin

# --- grid_volume: Tier 2 ------------------------------------------------------
# The sparse voxel hash map: the POD data model (hash_types.hpp,
# voxel_grid.hpp), the host coordinate math (voxel_coords.hpp), and the host
# VoxelHashMap that owns the device buffers + compute pipelines and dispatches
# the GLSL hash kernels (init / allocate-from-coords / compact). The table's
# internal layout (hash.hpp) stays under src/. Now a compiled STATIC library
# (was header-only) since VoxelHashMap + its embedded shaders landed. Depends
# only on grid_base, per the strict left-to-right tier rule.
include(GenerateExportHeader)

set(_vgrid_volume_generated_inc "${CMAKE_CURRENT_BINARY_DIR}/generated/include")

add_library(grid_volume)
add_library(volumetric_kit::grid_volume ALIAS grid_volume)

target_sources(grid_volume PRIVATE grid_growth.cpp triangle_candidates.cpp
                                   voxel_hash_map.cpp voxel_block_grid.cpp)

# Per-tier export header (VGRID_VOLUME_API / VGRID_VOLUME_STATIC_DEFINE),
# isolated from grid_base's so the macros never collide on one include path.
generate_export_header(
  grid_volume
  BASE_NAME
  VGRID_VOLUME
  EXPORT_MACRO_NAME
  VGRID_VOLUME_API
  EXPORT_FILE_NAME
  "${_vgrid_volume_generated_inc}/volumetric_kit/grid/volume/export.hpp")

set_target_properties(
  grid_volume
  PROPERTIES CXX_VISIBILITY_PRESET hidden
             VISIBILITY_INLINES_HIDDEN ON
             POSITION_INDEPENDENT_CODE ON)

get_target_property(_vgrid_volume_type grid_volume TYPE)
if(_vgrid_volume_type STREQUAL "STATIC_LIBRARY")
  target_compile_definitions(grid_volume PUBLIC VGRID_VOLUME_STATIC_DEFINE)
endif()

target_compile_features(grid_volume PUBLIC cxx_std_17)

# PUBLIC: the volume public headers (voxel_hash_map.hpp) expose core types
# (Result, Buffer, ComputePipeline, ...), so consumers compile against grid_base
# too. grid_base carries the Vulkan/GLM include wiring transitively.
target_link_libraries(grid_volume PUBLIC grid_base)

target_include_directories(
  grid_volume PUBLIC "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>"
                     "$<BUILD_INTERFACE:${_vgrid_volume_generated_inc}>")

# Compile the GLSL hash kernels to SPIR-V and embed them as constexpr byte
# arrays into the library (voxel_hash_map.cpp #includes the generated *.spv.hpp
# on a PRIVATE path), so grid_volume ships the technique with no runtime shader
# files.
include(vgrid_embed)
vgrid_embed_shaders(
  grid_volume
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
# #include ("volumetric_kit/grid/volume/shaders/grid_types.glsl") through the
# core's INCLUDE_TARGETS volumetric_kit::grid_volume, as recon's tsdf does. The
# hash table's GLSL (hash_common.glsl, hash_lookup.glsl, ...) stays internal:
# grid's codec, mesh and voxelize kernels read it through src/.
vkc_export_glsl(grid_volume PREFIX volumetric_kit/grid/volume/shaders FILES
                shaders/grid_types.glsl shaders/voxel_color.glsl)

vgrid_target_warnings(grid_volume)
