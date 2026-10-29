# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Tao Jin

# --- grid_voxelize: a triangle mesh's distance field --------------------------
# MeshIntegrator writes a mesh's truncated distance field into a
# VoxelBlockGrid's tsdf/weight, signed or as a shell, over the blocks
# VoxelHashMap::allocate_from_triangles allocates (the 2026-09-27 and 2026-10-28
# decisions). Links grid_volume ALONE: it needs the grid, the triangle
# candidates and grid_base's dispatch helper, not depth fusion, so a mesh caller
# need not link the fuser or its kernels.
include(GenerateExportHeader)

set(_vgrid_voxelize_generated_inc
    "${CMAKE_CURRENT_BINARY_DIR}/generated/include")

add_library(grid_voxelize)
add_library(volumetric_kit::grid_voxelize ALIAS grid_voxelize)

target_sources(grid_voxelize PRIVATE mesh_integrator.cpp)

# Per-tier export header (VGRID_VOXELIZE_API / VGRID_VOXELIZE_STATIC_DEFINE),
# isolated so the macros never collide with another tier's on one include path.
generate_export_header(
  grid_voxelize
  BASE_NAME
  VGRID_VOXELIZE
  EXPORT_MACRO_NAME
  VGRID_VOXELIZE_API
  EXPORT_FILE_NAME
  "${_vgrid_voxelize_generated_inc}/volumetric_kit/grid/voxelize/export.hpp")

set_target_properties(
  grid_voxelize
  PROPERTIES CXX_VISIBILITY_PRESET hidden
             VISIBILITY_INLINES_HIDDEN ON
             POSITION_INDEPENDENT_CODE ON)

get_target_property(_vgrid_voxelize_type grid_voxelize TYPE)
if(_vgrid_voxelize_type STREQUAL "STATIC_LIBRARY")
  target_compile_definitions(grid_voxelize PUBLIC VGRID_VOXELIZE_STATIC_DEFINE)
endif()

target_compile_features(grid_voxelize PUBLIC cxx_std_17)

# PUBLIC: the header exposes volume types (VoxelBlockGrid, VoxelGridParams) and
# core types (Result, Buffer, ComputeKernel).
target_link_libraries(grid_voxelize PUBLIC grid_volume)

# PRIVATE src/: volume's internal hash_map_internal.hpp (the hash entries).
target_include_directories(
  grid_voxelize
  PUBLIC "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>"
         "$<BUILD_INTERFACE:${_vgrid_voxelize_generated_inc}>"
  PRIVATE "${PROJECT_SOURCE_DIR}/src")

# The binning, scan, compaction and write kernels, compiled to SPIR-V and
# embedded (mesh_integrator.cpp #includes the generated *.spv.hpp on a PRIVATE
# path). The binning and write kernels include volume's GLSL by root-relative
# path.
include(vgrid_embed)
vgrid_embed_shaders(
  grid_voxelize
  SHADERS
  "${CMAKE_CURRENT_SOURCE_DIR}/shaders/mesh_bin.comp"
  "${CMAKE_CURRENT_SOURCE_DIR}/shaders/mesh_scan_counts.comp"
  "${CMAKE_CURRENT_SOURCE_DIR}/shaders/mesh_scan_sums.comp"
  "${CMAKE_CURRENT_SOURCE_DIR}/shaders/mesh_scan_add.comp"
  "${CMAKE_CURRENT_SOURCE_DIR}/shaders/mesh_compact_bins.comp"
  "${CMAKE_CURRENT_SOURCE_DIR}/shaders/mesh_bin_ranges.comp"
  "${CMAKE_CURRENT_SOURCE_DIR}/shaders/mesh_integrate.comp")

vgrid_target_warnings(grid_voxelize)
