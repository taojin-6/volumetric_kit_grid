# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Tao Jin

# What a project whose kernels include grid's exported GLSL does, for the
# subproject/ consumer: compile through the core's INCLUDE_TARGETS, with
# vkc_compile_shaders, a kernel that includes the split dispatch's
# base/shaders/dispatch_slices.glsl (volumetric_kit::grid_base;
# shaders/dispatch_slices_consumer.comp) and kernels that include the voxel
# grid's contract and colour format (volumetric_kit::grid_volume;
# shaders/grid_types_cases.comp, shaders/voxel_color_cases.comp). The
# grid_subproject_glsl test builds the target, vgrid_glsl_consumer. Each include
# root holds exactly the GLSL its target exports: none of grid's own, and none
# of the hash table's.
function(_vgrid_check_glsl_root target)
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
_vgrid_check_glsl_root(volumetric_kit::grid_base
                       volumetric_kit/grid/base/shaders/dispatch_slices.glsl)
_vgrid_check_glsl_root(
  volumetric_kit::grid_volume
  volumetric_kit/grid/volume/shaders/grid_types.glsl
  volumetric_kit/grid/volume/shaders/voxel_color.glsl)

add_custom_target(vgrid_glsl_consumer)
vkc_compile_shaders(
  vgrid_glsl_consumer
  INCLUDE_TARGETS
  volumetric_kit::grid_base
  SPIRV_VAL_ARGS
  --scalar-block-layout
  SHADERS
  "${CMAKE_CURRENT_LIST_DIR}/shaders/dispatch_slices_consumer.comp")
vkc_compile_shaders(
  vgrid_glsl_consumer
  INCLUDE_TARGETS
  volumetric_kit::grid_volume
  SPIRV_VAL_ARGS
  --scalar-block-layout
  SHADERS
  "${CMAKE_CURRENT_LIST_DIR}/shaders/grid_types_cases.comp"
  "${CMAKE_CURRENT_LIST_DIR}/shaders/voxel_color_cases.comp")
