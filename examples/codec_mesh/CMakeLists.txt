# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Tao Jin

add_executable(codec_mesh main.cpp)
target_link_libraries(
  codec_mesh
  PRIVATE vr_example_mesh volumetric_kit::recon_codec
          volumetric_kit::recon_eval volumetric_kit::recon_tsdf Vulkan::Vulkan)
vr_target_warnings(codec_mesh)
