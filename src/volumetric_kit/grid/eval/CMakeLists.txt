# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Tao Jin

# --- grid_eval: quality measurement -------------------------------------------
# How far one mesh is from another -- point-to-surface distance, accuracy /
# coverage and the F-score -- for the tests and examples that judge a
# reconstruction or a codec (the 2026-09-27 eval decision). Host-only C++ with
# no dependency beyond the mesh container it reads; not a stage the pipeline
# runs, and nothing to the left of it may link it. Branches off grid_mesh.
include(GenerateExportHeader)

set(_vgrid_eval_generated_inc "${CMAKE_CURRENT_BINARY_DIR}/generated/include")

add_library(grid_eval)
add_library(volumetric_kit::grid_eval ALIAS grid_eval)

target_sources(grid_eval PRIVATE mesh_distance.cpp)

# Per-tier export header (VGRID_EVAL_API / VGRID_EVAL_STATIC_DEFINE), isolated
# so the macros never collide with another tier's on one include path.
generate_export_header(
  grid_eval
  BASE_NAME
  VGRID_EVAL
  EXPORT_MACRO_NAME
  VGRID_EVAL_API
  EXPORT_FILE_NAME
  "${_vgrid_eval_generated_inc}/volumetric_kit/grid/eval/export.hpp")

set_target_properties(
  grid_eval
  PROPERTIES CXX_VISIBILITY_PRESET hidden
             VISIBILITY_INLINES_HIDDEN ON
             POSITION_INDEPENDENT_CODE ON)

get_target_property(_vgrid_eval_type grid_eval TYPE)
if(_vgrid_eval_type STREQUAL "STATIC_LIBRARY")
  target_compile_definitions(grid_eval PUBLIC VGRID_EVAL_STATIC_DEFINE)
endif()

target_compile_features(grid_eval PUBLIC cxx_std_17)

# PUBLIC: the header exposes mesh::Mesh and core's Result / math aliases.
target_link_libraries(grid_eval PUBLIC grid_mesh)

target_include_directories(
  grid_eval PUBLIC "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>"
                   "$<BUILD_INTERFACE:${_vgrid_eval_generated_inc}>")

vgrid_target_warnings(grid_eval)
