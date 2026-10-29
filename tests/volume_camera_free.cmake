# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Tao Jin

# The volume tier is camera-free (DECISIONS.md, 2026-10-28): nothing in its
# headers or sources, shaders included, includes the camera vocabulary,
# core/camera_params.hpp or core/shaders/pinhole.glsl. Its depth allocation
# takes volume::DepthPinhole and unprojects through depth_pinhole.glsl. Run as
# `cmake -DSOURCE_DIR=<grid root> -P volume_camera_free.cmake`; fails naming
# every such include.
foreach(_vgrid_dir include/volumetric_kit/grid/volume
                   src/volumetric_kit/grid/volume)
  if(NOT IS_DIRECTORY "${SOURCE_DIR}/${_vgrid_dir}")
    message(
      FATAL_ERROR "no ${SOURCE_DIR}/${_vgrid_dir}: the check reads nothing")
  endif()
  file(GLOB_RECURSE _vgrid_files "${SOURCE_DIR}/${_vgrid_dir}/*")
  list(APPEND _vgrid_all ${_vgrid_files})
endforeach()
set(_vgrid_found "")
foreach(_vgrid_file IN LISTS _vgrid_all)
  file(
    STRINGS "${_vgrid_file}" _vgrid_lines
    REGEX
      "#[ \t]*include[ \t]*[\"<]([^\">]*/)?(camera_params\\.hpp|pinhole\\.glsl)[\">]"
  )
  foreach(_vgrid_line IN LISTS _vgrid_lines)
    file(RELATIVE_PATH _vgrid_rel "${SOURCE_DIR}" "${_vgrid_file}")
    string(APPEND _vgrid_found "\n  ${_vgrid_rel}: ${_vgrid_line}")
  endforeach()
endforeach()
if(_vgrid_found)
  message(FATAL_ERROR "the volume tier includes the camera vocabulary:"
                      "${_vgrid_found}")
endif()
list(LENGTH _vgrid_all _vgrid_count)
message(STATUS "volume is camera-free: ${_vgrid_count} files read")
