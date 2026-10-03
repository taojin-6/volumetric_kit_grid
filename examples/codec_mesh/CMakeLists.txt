# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Tao Jin

add_executable(codec_mesh main.cpp ../common/mesh_normalization.cpp)
target_include_directories(codec_mesh PRIVATE ../common)
target_link_libraries(
  codec_mesh
  PRIVATE volumetric_kit::recon_codec volumetric_kit::recon_eval
          volumetric_kit::recon_tsdf volumetric_kit::recon_io_assimp
          Vulkan::Vulkan)
vr_target_warnings(codec_mesh)
