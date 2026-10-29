# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Tao Jin

# Orchestrator for the component tiers. Each subdirectory owns its own target
# and build settings; the strict dependency rule (a tier may depend only on
# tiers to its left) is expressed by what each links.

add_subdirectory(volumetric_kit/grid/base)
add_subdirectory(volumetric_kit/grid/volume)
# A triangle mesh's distance field, on volume alone: no depth fusion (the
# 2026-10-28 decision).
add_subdirectory(volumetric_kit/grid/voxelize)
# mesh reads the volume voxel payload directly (proven against a dense analytic
# SDF), so it depends on volume.
add_subdirectory(volumetric_kit/grid/mesh)
# Mesh files, off mesh for the public Mesh serialization API.
add_subdirectory(volumetric_kit/grid/mesh_io)
add_subdirectory(volumetric_kit/grid/codec)
# Quality measurement for the tests and examples that judge a reconstruction or
# a codec: mesh-to-mesh distance, accuracy / coverage, F-score (the 2026-09-27
# eval decision). Off grid_mesh.
add_subdirectory(volumetric_kit/grid/eval)
# The .vrst session container: recorded tracks of records, stored verbatim. On
# the family core's base tier alone, so it depends on no tier here.
add_subdirectory(volumetric_kit/grid/stream)
# A session served and read live over TCP: the stream's chunks on sockets, off
# stream (and grid's header-only log seam).
add_subdirectory(volumetric_kit/grid/networking)
# A codec stream, after networking: grids coded as frames into a .vrst file and
# to live viewers, decoded back. It links the codec, the container and the
# server.
add_subdirectory(volumetric_kit/grid/codec_stream)

# --- grid: umbrella alias -----------------------------------------------------
# Consumers link `volumetric_kit::grid` and get every tier. It carries grid's
# version, which a consumer's vkc_require_sibling reads.
add_library(grid INTERFACE)
add_library(volumetric_kit::grid ALIAS grid)
set_target_properties(grid PROPERTIES VKC_PACKAGE_VERSION "${PROJECT_VERSION}")
target_link_libraries(
  grid
  INTERFACE grid_base
            grid_volume
            grid_voxelize
            grid_mesh
            grid_mesh_io
            grid_codec
            grid_eval
            grid_stream
            grid_networking
            grid_codec_stream)
if(TARGET grid_mesh_io_assimp)
  target_link_libraries(grid INTERFACE grid_mesh_io_assimp)
endif()
