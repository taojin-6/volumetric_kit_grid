# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Tao Jin

add_executable(codec_mesh main.cpp ../common/mesh_input.cpp)
target_link_libraries(
  codec_mesh
  PRIVATE vr_example_common
          volumetric_kit::recon_codec
          volumetric_kit::recon_eval
          volumetric_kit::recon_tsdf
          volumetric_kit::recon_mesh
          volumetric_kit::recon_volume
          Vulkan::Vulkan)
vr_target_warnings(codec_mesh)
