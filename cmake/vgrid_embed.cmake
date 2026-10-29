# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Tao Jin

# vgrid_embed_shaders(<target> [SYMBOL_PREFIX <prefix>] [INCLUDE_TARGETS
# <target>...] SHADERS <file>...)
#
# The core's vkc_embed_shaders with spirv-val's --scalar-block-layout, the
# buffer ABI the kernels declare (2026-07-05), and the vgrid_ symbol prefix
# unless another is given: hash.comp embeds as hash_comp.spv.hpp, declaring
# `vgrid_hash_comp_spv[]` and `vgrid_hash_comp_spv_size`. The shaders include
# through grid's src/, so a cross-tier include reads like its C++ header path
# ("volumetric_kit/grid/volume/shaders/hash_common.glsl"); with INCLUDE_TARGETS,
# through those targets' exported GLSL alone instead, as a project that fetches
# grid does.
function(vgrid_embed_shaders target)
  cmake_parse_arguments(ARG "" "SYMBOL_PREFIX" "INCLUDE_TARGETS;SHADERS"
                        ${ARGN})
  if(NOT ARG_SYMBOL_PREFIX)
    set(ARG_SYMBOL_PREFIX vgrid_)
  endif()
  if(ARG_INCLUDE_TARGETS)
    set(_includes INCLUDE_TARGETS ${ARG_INCLUDE_TARGETS})
  else()
    set(_includes INCLUDE_DIRS "${PROJECT_SOURCE_DIR}/src")
  endif()
  vkc_embed_shaders(
    ${target}
    SYMBOL_PREFIX
    ${ARG_SYMBOL_PREFIX}
    ${_includes}
    SPIRV_VAL_ARGS
    --scalar-block-layout
    ${ARG_UNPARSED_ARGUMENTS}
    SHADERS
    ${ARG_SHADERS})
endfunction()
