# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Tao Jin

# grid sits below recon and sensor (DECISIONS.md, the package rule): no file
# under grid/ includes a header or GLSL file of either. Run as `cmake
# -DSOURCE_DIR=<grid root> -P package_boundary.cmake`; fails naming every such
# include. The pattern is spelt in pieces, so this file does not match itself.
if(NOT IS_DIRECTORY "${SOURCE_DIR}/src" OR NOT IS_DIRECTORY
                                           "${SOURCE_DIR}/include")
  message(FATAL_ERROR "no grid sources under ${SOURCE_DIR}: the check reads "
                      "nothing")
endif()
file(
  GLOB_RECURSE _vgrid_all
  LIST_DIRECTORIES false
  "${SOURCE_DIR}/*")
set(_vgrid_kit "volumetric_kit")
set(_vgrid_above "(recon|sensor)")
set(_vgrid_found "")
set(_vgrid_count 0)
foreach(_vgrid_file IN LISTS _vgrid_all)
  # Not a build tree that sits inside the checkout.
  file(RELATIVE_PATH _vgrid_rel "${SOURCE_DIR}" "${_vgrid_file}")
  if(_vgrid_rel MATCHES "^(build[^/]*|\\.git)/")
    continue()
  endif()
  math(EXPR _vgrid_count "${_vgrid_count} + 1")
  file(STRINGS "${_vgrid_file}" _vgrid_lines
       REGEX "#[ \t]*include[ \t]*[\"<]${_vgrid_kit}/${_vgrid_above}/")
  foreach(_vgrid_line IN LISTS _vgrid_lines)
    string(APPEND _vgrid_found "\n  ${_vgrid_rel}: ${_vgrid_line}")
  endforeach()
endforeach()
if(_vgrid_found)
  message(FATAL_ERROR "grid includes a package above it:${_vgrid_found}")
endif()
message(STATUS "grid includes nothing above it: ${_vgrid_count} files read")
