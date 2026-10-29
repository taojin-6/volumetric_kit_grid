# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Tao Jin

# The sequence's steps, shared by codec_mesh and its test: Assimp-backed
# loading, so they live here rather than in grid_example_common.
add_library(grid_example_mesh STATIC mesh_sequence.cpp geometry_mesh.cpp)
target_include_directories(grid_example_mesh
                           PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(
  grid_example_mesh
  PUBLIC grid_example_common volumetric_kit::core_config
         volumetric_kit::grid_eval volumetric_kit::grid_codec_stream
         volumetric_kit::grid_voxelize volumetric_kit::grid_mesh_io_assimp)
target_compile_features(grid_example_mesh PUBLIC cxx_std_17)
vgrid_target_warnings(grid_example_mesh)

add_executable(codec_mesh main.cpp)
target_link_libraries(codec_mesh PRIVATE grid_example_mesh Vulkan::Vulkan)
vgrid_target_warnings(codec_mesh)
