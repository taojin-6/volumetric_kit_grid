# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Tao Jin

# What a project whose kernels include recon's exported GLSL does, for the
# subproject/ consumer: compile through the core's INCLUDE_TARGETS, with
# vkc_compile_shaders, a kernel that includes core/shaders/pinhole.glsl
# (volumetric_kit::recon_core; shaders/pinhole_cases.comp) and kernels that
# include the voxel grid's contract and colour format
# (volumetric_kit::recon_volume; shaders/grid_types_cases.comp,
# shaders/voxel_color_cases.comp). The recon_subproject_glsl test builds the
# target, vr_glsl_consumer. Each include root holds exactly the GLSL its target
# exports: none of recon's own, and none of the hash table's.
function(_vr_check_glsl_root target)
  get_target_property(_roots ${target} VKC_SHADER_INCLUDE_DIRS)
  set(_held "")
  foreach(_root IN LISTS _roots)
    file(
      GLOB_RECURSE _files
      LIST_DIRECTORIES false
      RELATIVE "${_root}"
      "${_root}/*")
    list(APPEND _held ${_files})
  endforeach()
  list(SORT _held)
  set(_want ${ARGN})
  list(SORT _want)
  if(NOT _held STREQUAL _want)
    message(FATAL_ERROR "${target}'s GLSL include root holds [${_held}], "
                        "not exactly what it exports, [${_want}]")
  endif()
endfunction()
_vr_check_glsl_root(volumetric_kit::recon_core
                    volumetric_kit/recon/core/shaders/pinhole.glsl)
_vr_check_glsl_root(
  volumetric_kit::recon_volume
  volumetric_kit/recon/volume/shaders/grid_types.glsl
  volumetric_kit/recon/volume/shaders/voxel_color.glsl)

add_custom_target(vr_glsl_consumer)
vkc_compile_shaders(
  vr_glsl_consumer INCLUDE_TARGETS volumetric_kit::recon_core SHADERS
  "${CMAKE_CURRENT_LIST_DIR}/shaders/pinhole_cases.comp")
vkc_compile_shaders(
  vr_glsl_consumer
  INCLUDE_TARGETS
  volumetric_kit::recon_volume
  SPIRV_VAL_ARGS
  --scalar-block-layout
  SHADERS
  "${CMAKE_CURRENT_LIST_DIR}/shaders/grid_types_cases.comp"
  "${CMAKE_CURRENT_LIST_DIR}/shaders/voxel_color_cases.comp")
