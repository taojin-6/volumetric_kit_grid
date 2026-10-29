# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Tao Jin

# --- grid_mesh: Tier 4 --------------------------------------------------------
# Marching-cubes surface extraction: the host mesh containers (mesh.hpp), the
# 256-case lookup tables (marching_cubes_tables.hpp), and the host MarchingCubes
# that owns the compute pipeline and drives the GLSL kernel turning a dense SDF
# grid into a triangle mesh. A compiled STATIC library carrying its embedded
# shader. Links grid_volume for the grid it meshes, per the strict left-to-right
# tier rule.
include(GenerateExportHeader)

set(_vgrid_mesh_generated_inc "${CMAKE_CURRENT_BINARY_DIR}/generated/include")

add_library(grid_mesh)
add_library(volumetric_kit::grid_mesh ALIAS grid_mesh)

target_sources(grid_mesh PRIVATE marching_cubes.cpp)

# Per-tier export header (VGRID_MESH_API / VGRID_MESH_STATIC_DEFINE), isolated
# from the other tiers' so the macros never collide on one include path.
generate_export_header(
  grid_mesh
  BASE_NAME
  VGRID_MESH
  EXPORT_MACRO_NAME
  VGRID_MESH_API
  EXPORT_FILE_NAME
  "${_vgrid_mesh_generated_inc}/volumetric_kit/grid/mesh/export.hpp")

set_target_properties(
  grid_mesh
  PROPERTIES CXX_VISIBILITY_PRESET hidden
             VISIBILITY_INLINES_HIDDEN ON
             POSITION_INDEPENDENT_CODE ON)

get_target_property(_vgrid_mesh_type grid_mesh TYPE)
if(_vgrid_mesh_type STREQUAL "STATIC_LIBRARY")
  target_compile_definitions(grid_mesh PUBLIC VGRID_MESH_STATIC_DEFINE)
endif()

target_compile_features(grid_mesh PUBLIC cxx_std_17)

# PUBLIC: the mesh public headers (marching_cubes.hpp) expose core types
# (Result, Buffer, ComputePipeline, Vec3f) and the volume grid
# (volume::VoxelBlockGrid), so consumers compile against grid_volume too.
# grid_volume carries grid_base (and its Vulkan/GLM include wiring)
# transitively.
target_link_libraries(grid_mesh PUBLIC grid_volume)

# PRIVATE src/: volume's internal hash_map_internal.hpp (the hash entries).
target_include_directories(
  grid_mesh
  PUBLIC "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>"
         "$<BUILD_INTERFACE:${_vgrid_mesh_generated_inc}>"
  PRIVATE "${PROJECT_SOURCE_DIR}/src")

# Compile the GLSL marching-cubes kernel to SPIR-V and embed it as a constexpr
# byte array into the library (marching_cubes.cpp #includes the generated
# *.spv.hpp on a PRIVATE path), so grid_mesh ships the technique with no runtime
# shader files.
include(vgrid_embed)
vgrid_embed_shaders(
  grid_mesh
  SHADERS
  "${CMAKE_CURRENT_SOURCE_DIR}/shaders/marching_cubes_sparse.comp"
  "${CMAKE_CURRENT_SOURCE_DIR}/shaders/marching_cubes_sparse_shared.comp"
  "${CMAKE_CURRENT_SOURCE_DIR}/shaders/mc_args.comp"
  "${CMAKE_CURRENT_SOURCE_DIR}/shaders/mc_finalize.comp")

vgrid_target_warnings(grid_mesh)
