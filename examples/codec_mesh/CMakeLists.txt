# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Tao Jin

# The sequence's steps, shared by codec_mesh and its test: Assimp-backed
# loading, so they live here rather than in vr_example_common.
add_library(vr_example_mesh STATIC mesh_sequence.cpp geometry_mesh.cpp)
target_include_directories(
  vr_example_mesh PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}"
                         "${CMAKE_CURRENT_SOURCE_DIR}/../common")
target_link_libraries(
  vr_example_mesh
  PUBLIC volumetric_kit::core_config volumetric_kit::recon_eval
         volumetric_kit::recon_sender volumetric_kit::recon_voxelize
         volumetric_kit::recon_mesh_io_assimp)
target_compile_features(vr_example_mesh PUBLIC cxx_std_17)
vr_target_warnings(vr_example_mesh)

add_executable(codec_mesh main.cpp)
target_link_libraries(codec_mesh PRIVATE vr_example_mesh Vulkan::Vulkan)
vr_target_warnings(codec_mesh)
