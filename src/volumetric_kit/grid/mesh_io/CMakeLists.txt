# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Tao Jin

include(GenerateExportHeader)

set(_vgrid_mesh_io_generated_inc
    "${CMAKE_CURRENT_BINARY_DIR}/generated/include")

# --- grid_mesh_io: mesh files -------------------------------------------------
# Mesh -> PLY serialization, the host triangle containers and the opt-in mesh
# normalization helpers. PUBLIC grid_mesh because write_ply takes a mesh::Mesh.
# tinyply's implementation is private: no public header or target.
add_library(grid_mesh_io mesh_normalization.cpp ply_writer.cpp tinyply_impl.cpp)
add_library(volumetric_kit::grid_mesh_io ALIAS grid_mesh_io)
generate_export_header(
  grid_mesh_io
  BASE_NAME
  VGRID_MESH_IO
  EXPORT_MACRO_NAME
  VGRID_MESH_IO_API
  EXPORT_FILE_NAME
  "${_vgrid_mesh_io_generated_inc}/volumetric_kit/grid/mesh_io/mesh_io_export.hpp"
)
set_target_properties(
  grid_mesh_io
  PROPERTIES CXX_VISIBILITY_PRESET hidden
             VISIBILITY_INLINES_HIDDEN ON
             POSITION_INDEPENDENT_CODE ON)
get_target_property(_vgrid_mesh_io_type grid_mesh_io TYPE)
if(_vgrid_mesh_io_type STREQUAL "STATIC_LIBRARY")
  target_compile_definitions(grid_mesh_io PUBLIC VGRID_MESH_IO_STATIC_DEFINE)
endif()
target_compile_features(grid_mesh_io PUBLIC cxx_std_17)
# Catch backend / allocation exceptions at the Status boundary even when a
# consuming application globally disables exceptions. Keep this private.
target_compile_options(grid_mesh_io PRIVATE -fexceptions)
target_link_libraries(grid_mesh_io PUBLIC grid_mesh)
target_include_directories(
  grid_mesh_io PUBLIC "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>"
                      "$<BUILD_INTERFACE:${_vgrid_mesh_io_generated_inc}>")
target_include_directories(grid_mesh_io SYSTEM
                           PRIVATE "${VGRID_TINYPLY_INCLUDE_DIR}")
vgrid_target_warnings(grid_mesh_io)

# Only the vendored implementation TU opts out of first-party warnings.
set_source_files_properties(tinyply_impl.cpp PROPERTIES COMPILE_OPTIONS -w)
# No fused multiply-add, so a mesh transform's doubles, which a stream may
# store, are the same whichever compiler built them.
set_source_files_properties(mesh_normalization.cpp PROPERTIES COMPILE_OPTIONS
                                                              -ffp-contract=off)

# An installed prerequisite. Keeping it separate lets PLY consumers build
# without Assimp on every platform.
if(VGRID_WITH_ASSIMP)
  add_library(grid_mesh_io_assimp mesh_io.cpp)
  add_library(volumetric_kit::grid_mesh_io_assimp ALIAS grid_mesh_io_assimp)
  generate_export_header(
    grid_mesh_io_assimp
    BASE_NAME
    VGRID_MESH_IO_ASSIMP
    EXPORT_MACRO_NAME
    VGRID_MESH_IO_ASSIMP_API
    EXPORT_FILE_NAME
    "${_vgrid_mesh_io_generated_inc}/volumetric_kit/grid/mesh_io/mesh_io_assimp_export.hpp"
  )
  set_target_properties(
    grid_mesh_io_assimp
    PROPERTIES CXX_VISIBILITY_PRESET hidden
               VISIBILITY_INLINES_HIDDEN ON
               POSITION_INDEPENDENT_CODE ON)
  get_target_property(_vgrid_mesh_io_assimp_type grid_mesh_io_assimp TYPE)
  if(_vgrid_mesh_io_assimp_type STREQUAL "STATIC_LIBRARY")
    target_compile_definitions(grid_mesh_io_assimp
                               PUBLIC VGRID_MESH_IO_ASSIMP_STATIC_DEFINE)
  endif()
  target_compile_features(grid_mesh_io_assimp PUBLIC cxx_std_17)
  target_compile_options(grid_mesh_io_assimp PRIVATE -fexceptions)
  # grid_mesh_io carries the public and generated include directories.
  target_link_libraries(
    grid_mesh_io_assimp
    PUBLIC grid_mesh_io
    PRIVATE assimp::assimp)
  vgrid_target_warnings(grid_mesh_io_assimp)
endif()
