# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Tao Jin

# --- recon_eval: quality measurement ------------------------------------------
# How far one mesh is from another -- point-to-surface distance, accuracy /
# coverage and the F-score -- for the tests and examples that judge a
# reconstruction or a codec (the 2026-09-27 eval decision). Host-only C++ with
# no dependency beyond the mesh container it reads; not a stage the pipeline
# runs, and nothing to the left of it may link it. Branches off recon_mesh.
include(GenerateExportHeader)
include(GNUInstallDirs)

set(_vr_eval_generated_inc "${CMAKE_CURRENT_BINARY_DIR}/generated/include")

add_library(recon_eval)
add_library(volumetric_kit::recon_eval ALIAS recon_eval)

target_sources(recon_eval PRIVATE mesh_distance.cpp)

# Per-tier export header (VR_EVAL_API / VR_EVAL_STATIC_DEFINE), isolated so the
# macros never collide with another tier's on one include path.
generate_export_header(
  recon_eval
  BASE_NAME
  VR_EVAL
  EXPORT_MACRO_NAME
  VR_EVAL_API
  EXPORT_FILE_NAME
  "${_vr_eval_generated_inc}/volumetric_kit/recon/eval/export.hpp")

set_target_properties(
  recon_eval
  PROPERTIES CXX_VISIBILITY_PRESET hidden
             VISIBILITY_INLINES_HIDDEN ON
             POSITION_INDEPENDENT_CODE ON)

get_target_property(_vr_eval_type recon_eval TYPE)
if(_vr_eval_type STREQUAL "STATIC_LIBRARY")
  target_compile_definitions(recon_eval PUBLIC VR_EVAL_STATIC_DEFINE)
endif()

target_compile_features(recon_eval PUBLIC cxx_std_17)

# PUBLIC: the header exposes mesh::Mesh and core's Result / math aliases.
target_link_libraries(recon_eval PUBLIC recon_mesh)

target_include_directories(
  recon_eval
  PUBLIC "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>"
         "$<BUILD_INTERFACE:${_vr_eval_generated_inc}>"
         "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>")

vr_target_warnings(recon_eval)

if(VR_INSTALL)
  install(
    TARGETS recon_eval
    EXPORT volumetric_kit_reconTargets
    ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}")
  install(
    DIRECTORY "${PROJECT_SOURCE_DIR}/include/volumetric_kit/recon/eval"
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/volumetric_kit/recon"
    FILES_MATCHING
    PATTERN "*.hpp")
  install(
    DIRECTORY "${_vr_eval_generated_inc}/"
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
    FILES_MATCHING
    PATTERN "*.hpp")
endif()
