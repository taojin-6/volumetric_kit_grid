# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Tao Jin

# grid sits below recon and sensor (DECISIONS.md, the package rule): no file
# under grid/ includes a header or GLSL file of either. And grid makes no socket
# call of its own, the core's net tier making them (the 2026-11-02 net entry):
# nothing under src/ or include/ includes <sys/socket.h>, <netinet/...>,
# <arpa/inet.h> or <netdb.h>; <poll.h> stays, for POLLIN and POLLOUT. Run as
# `cmake -DSOURCE_DIR=<grid root> -P package_boundary.cmake`; fails naming every
# such include. The patterns are spelt in pieces, so this file does not match
# itself.
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
set(_vgrid_include "#[ \t]*include[ \t]*")
set(_vgrid_socket "(sys/socket\\.h|netinet/[^>]*|arpa/inet\\.h|netdb\\.h)")
set(_vgrid_found "")
set(_vgrid_sockets "")
set(_vgrid_count 0)
foreach(_vgrid_file IN LISTS _vgrid_all)
  # Not a build tree that sits inside the checkout.
  file(RELATIVE_PATH _vgrid_rel "${SOURCE_DIR}" "${_vgrid_file}")
  if(_vgrid_rel MATCHES "^(build[^/]*|\\.git)/")
    continue()
  endif()
  math(EXPR _vgrid_count "${_vgrid_count} + 1")
  file(STRINGS "${_vgrid_file}" _vgrid_lines
       REGEX "${_vgrid_include}[\"<]${_vgrid_kit}/${_vgrid_above}/")
  foreach(_vgrid_line IN LISTS _vgrid_lines)
    string(APPEND _vgrid_found "\n  ${_vgrid_rel}: ${_vgrid_line}")
  endforeach()
  if(_vgrid_rel MATCHES "^(src|include)/")
    file(STRINGS "${_vgrid_file}" _vgrid_lines
         REGEX "${_vgrid_include}<${_vgrid_socket}>")
    foreach(_vgrid_line IN LISTS _vgrid_lines)
      string(APPEND _vgrid_sockets "\n  ${_vgrid_rel}: ${_vgrid_line}")
    endforeach()
  endif()
endforeach()
if(_vgrid_found OR _vgrid_sockets)
  set(_vgrid_why "")
  if(_vgrid_found)
    string(APPEND _vgrid_why "grid includes a package above it:${_vgrid_found}")
  endif()
  if(_vgrid_sockets)
    if(_vgrid_why)
      string(APPEND _vgrid_why "\n")
    endif()
    string(APPEND _vgrid_why
           "grid's sources include a socket header, which the core's net tier "
           "alone includes:${_vgrid_sockets}")
  endif()
  message(FATAL_ERROR "${_vgrid_why}")
endif()
message(STATUS "grid includes nothing above it and no socket header: "
               "${_vgrid_count} files read")
