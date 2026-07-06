# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Tao Jin

# --- recon_mesh: Tier 4 -----------------------------------------------------
# Marching-cubes surface extraction: the host mesh containers (mesh.hpp), the
# 256-case lookup tables (marching_cubes_tables.hpp), and the host MarchingCubes
# that owns the compute pipeline and drives the GLSL kernel turning a dense SDF
# grid into a triangle mesh. A compiled STATIC library carrying its embedded
# shader. Links recon_volume for the shared voxel POD layouts (volume::Voxel),
# per the strict left-to-right tier rule (tsdf slots between volume and mesh
# once it lands; mesh depends only on tiers to its left, and today reads the
# volume payload directly).
include(GenerateExportHeader)
include(GNUInstallDirs)

set(_vr_mesh_generated_inc "${CMAKE_CURRENT_BINARY_DIR}/generated/include")

add_library(recon_mesh)
add_library(volumetric_kit::recon_mesh ALIAS recon_mesh)

target_sources(recon_mesh PRIVATE marching_cubes.cpp)

# Per-tier export header (VR_MESH_API / VR_MESH_STATIC_DEFINE), isolated from
# the other tiers' so the macros never collide on one include path.
generate_export_header(
  recon_mesh
  BASE_NAME
  VR_MESH
  EXPORT_MACRO_NAME
  VR_MESH_API
  EXPORT_FILE_NAME
  "${_vr_mesh_generated_inc}/volumetric_kit/recon/mesh/export.hpp")

set_target_properties(
  recon_mesh
  PROPERTIES CXX_VISIBILITY_PRESET hidden
             VISIBILITY_INLINES_HIDDEN ON
             POSITION_INDEPENDENT_CODE ON)

get_target_property(_vr_mesh_type recon_mesh TYPE)
if(_vr_mesh_type STREQUAL "STATIC_LIBRARY")
  target_compile_definitions(recon_mesh PUBLIC VR_MESH_STATIC_DEFINE)
endif()

target_compile_features(recon_mesh PUBLIC cxx_std_17)

# PUBLIC: the mesh public headers (marching_cubes.hpp) expose core types
# (Result, Buffer, ComputePipeline, Vec3f) and the volume voxel payload
# (volume::Voxel), so consumers compile against recon_volume too. recon_volume
# carries recon_core (and its Vulkan/GLM include wiring) transitively.
target_link_libraries(recon_mesh PUBLIC recon_volume)

target_include_directories(
  recon_mesh
  PUBLIC "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>"
         "$<BUILD_INTERFACE:${_vr_mesh_generated_inc}>"
         "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>")

# Compile the GLSL marching-cubes kernel to SPIR-V and embed it as a constexpr
# byte array into the library (marching_cubes.cpp #includes the generated
# *.spv.hpp on a PRIVATE path), so recon_mesh ships the technique with no
# runtime shader files.
include(vr_embed)
vr_embed_shaders(
  recon_mesh SHADERS "${CMAKE_CURRENT_SOURCE_DIR}/shaders/marching_cubes.comp"
  "${CMAKE_CURRENT_SOURCE_DIR}/shaders/marching_cubes_sparse.comp")

vr_target_warnings(recon_mesh)

# Install the target into the shared export set + this tier's headers (the mesh/
# subtree) + its generated export header. Mirrors the other tiers so a partial
# build never installs a header for a tier it did not build.
if(VR_INSTALL)
  install(
    TARGETS recon_mesh
    EXPORT volumetric_kit_reconTargets
    ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}")
  install(
    DIRECTORY "${PROJECT_SOURCE_DIR}/include/volumetric_kit/recon/mesh"
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/volumetric_kit/recon"
    FILES_MATCHING
    PATTERN "*.hpp")
  install(
    DIRECTORY "${_vr_mesh_generated_inc}/"
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
    FILES_MATCHING
    PATTERN "*.hpp")
endif()
