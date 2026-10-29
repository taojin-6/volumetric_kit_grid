# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Tao Jin

# What a project whose kernels include recon's exported GLSL does, for the
# subproject/ consumer: compile through the core's INCLUDE_TARGETS
# volumetric_kit::recon_core, with vkc_compile_shaders, a kernel that includes
# core/shaders/pinhole.glsl (shaders/pinhole_cases.comp). The
# recon_subproject_glsl test builds the target, vr_glsl_consumer. The include
# root recon_core exports holds none of recon's own GLSL.
get_target_property(_vr_glsl_roots volumetric_kit::recon_core
                    VKC_SHADER_INCLUDE_DIRS)
set(_vr_internal "volumetric_kit/recon/tsdf/shaders/tsdf_common.glsl")
foreach(_vr_glsl_root IN LISTS _vr_glsl_roots)
  if(EXISTS "${_vr_glsl_root}/${_vr_internal}")
    message(FATAL_ERROR "recon's GLSL include root holds ${_vr_internal}, "
                        "which recon does not export")
  endif()
endforeach()
add_custom_target(vr_glsl_consumer)
vkc_compile_shaders(
  vr_glsl_consumer INCLUDE_TARGETS volumetric_kit::recon_core SHADERS
  "${CMAKE_CURRENT_LIST_DIR}/shaders/pinhole_cases.comp")
