# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Tao Jin

# What a project whose kernels include recon's exported GLSL does, for the
# subproject/ and package/ consumers: read the include root
# volumetric_kit::recon_core names, and compile through it, with the core's
# vkc_compile_shaders, a kernel that includes core/shaders/pinhole.glsl
# (shaders/pinhole_cases.comp). The recon_*_glsl tests build the target,
# vr_glsl_consumer. The root holds the exported GLSL and none of recon's own.
get_target_property(_vr_glsl_root volumetric_kit::recon_core
                    VR_SHADER_INCLUDE_DIR)
set(_vr_pinhole "volumetric_kit/recon/core/shaders/pinhole.glsl")
if(NOT _vr_glsl_root OR NOT EXISTS "${_vr_glsl_root}/${_vr_pinhole}")
  message(
    FATAL_ERROR
      "volumetric_kit::recon_core's VR_SHADER_INCLUDE_DIR, '${_vr_glsl_root}', "
      "holds no ${_vr_pinhole}")
endif()
set(_vr_internal "volumetric_kit/recon/tsdf/shaders/tsdf_common.glsl")
if(EXISTS "${_vr_glsl_root}/${_vr_internal}")
  message(FATAL_ERROR "recon's GLSL include root holds ${_vr_internal}, "
                      "which recon does not export")
endif()
add_custom_target(vr_glsl_consumer)
vkc_compile_shaders(vr_glsl_consumer INCLUDE_DIRS "${_vr_glsl_root}" SHADERS
                    "${CMAKE_CURRENT_LIST_DIR}/shaders/pinhole_cases.comp")
