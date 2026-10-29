# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Tao Jin

# The volume tier is camera-free (DECISIONS.md, 2026-10-28): nothing in its
# headers or sources, shaders included, includes the camera vocabulary,
# core/camera_params.hpp or core/shaders/pinhole.glsl. Its depth allocation
# takes volume::DepthPinhole and unprojects through depth_pinhole.glsl. Run as
# `cmake -DSOURCE_DIR=<recon root> -P volume_camera_free.cmake`; fails naming
# every such include.
foreach(_vr_dir include/volumetric_kit/recon/volume
                src/volumetric_kit/recon/volume)
  if(NOT IS_DIRECTORY "${SOURCE_DIR}/${_vr_dir}")
    message(FATAL_ERROR "no ${SOURCE_DIR}/${_vr_dir}: the check reads nothing")
  endif()
  file(GLOB_RECURSE _vr_files "${SOURCE_DIR}/${_vr_dir}/*")
  list(APPEND _vr_all ${_vr_files})
endforeach()
set(_vr_found "")
foreach(_vr_file IN LISTS _vr_all)
  file(
    STRINGS "${_vr_file}" _vr_lines
    REGEX
      "#[ \t]*include[ \t]*[\"<]([^\">]*/)?(camera_params\\.hpp|pinhole\\.glsl)[\">]"
  )
  foreach(_vr_line IN LISTS _vr_lines)
    file(RELATIVE_PATH _vr_rel "${SOURCE_DIR}" "${_vr_file}")
    string(APPEND _vr_found "\n  ${_vr_rel}: ${_vr_line}")
  endforeach()
endforeach()
if(_vr_found)
  message(FATAL_ERROR "the volume tier includes the camera vocabulary:"
                      "${_vr_found}")
endif()
list(LENGTH _vr_all _vr_count)
message(STATUS "volume is camera-free: ${_vr_count} files read")
