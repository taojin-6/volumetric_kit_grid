# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Tao Jin

# --- recon_voxelize: a triangle mesh's distance field -------------------------
# MeshIntegrator writes a mesh's truncated distance field into a
# VoxelBlockGrid's tsdf/weight, signed or as a shell, over the blocks
# VoxelHashMap::allocate_from_triangles allocates (the 2026-09-27 and 2026-10-28
# decisions). Links recon_volume ALONE: it needs the grid, the triangle
# candidates and recon_core's dispatch helper, not depth fusion, so a mesh
# caller need not link the fuser or its kernels.
include(GenerateExportHeader)

set(_vr_voxelize_generated_inc "${CMAKE_CURRENT_BINARY_DIR}/generated/include")

add_library(recon_voxelize)
add_library(volumetric_kit::recon_voxelize ALIAS recon_voxelize)

target_sources(recon_voxelize PRIVATE mesh_integrator.cpp)

# Per-tier export header (VR_VOXELIZE_API / VR_VOXELIZE_STATIC_DEFINE), isolated
# so the macros never collide with another tier's on one include path.
generate_export_header(
  recon_voxelize
  BASE_NAME
  VR_VOXELIZE
  EXPORT_MACRO_NAME
  VR_VOXELIZE_API
  EXPORT_FILE_NAME
  "${_vr_voxelize_generated_inc}/volumetric_kit/recon/voxelize/export.hpp")

set_target_properties(
  recon_voxelize
  PROPERTIES CXX_VISIBILITY_PRESET hidden
             VISIBILITY_INLINES_HIDDEN ON
             POSITION_INDEPENDENT_CODE ON)

get_target_property(_vr_voxelize_type recon_voxelize TYPE)
if(_vr_voxelize_type STREQUAL "STATIC_LIBRARY")
  target_compile_definitions(recon_voxelize PUBLIC VR_VOXELIZE_STATIC_DEFINE)
endif()

target_compile_features(recon_voxelize PUBLIC cxx_std_17)

# PUBLIC: the header exposes volume types (VoxelBlockGrid, VoxelGridParams) and
# core types (Result, Buffer, ComputeKernel).
target_link_libraries(recon_voxelize PUBLIC recon_volume)

# PRIVATE src/: recon_core's internal dispatch.hpp, as recon_volume reaches it.
target_include_directories(
  recon_voxelize
  PUBLIC "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>"
         "$<BUILD_INTERFACE:${_vr_voxelize_generated_inc}>"
  PRIVATE "${PROJECT_SOURCE_DIR}/src")

# The binning, scan, compaction and write kernels, compiled to SPIR-V and
# embedded (mesh_integrator.cpp #includes the generated *.spv.hpp on a PRIVATE
# path). The binning and write kernels include volume's GLSL by root-relative
# path.
include(vr_embed)
vr_embed_shaders(
  recon_voxelize
  SHADERS
  "${CMAKE_CURRENT_SOURCE_DIR}/shaders/mesh_bin.comp"
  "${CMAKE_CURRENT_SOURCE_DIR}/shaders/mesh_scan_counts.comp"
  "${CMAKE_CURRENT_SOURCE_DIR}/shaders/mesh_scan_sums.comp"
  "${CMAKE_CURRENT_SOURCE_DIR}/shaders/mesh_scan_add.comp"
  "${CMAKE_CURRENT_SOURCE_DIR}/shaders/mesh_compact_bins.comp"
  "${CMAKE_CURRENT_SOURCE_DIR}/shaders/mesh_bin_ranges.comp"
  "${CMAKE_CURRENT_SOURCE_DIR}/shaders/mesh_integrate.comp")

vr_target_warnings(recon_voxelize)
