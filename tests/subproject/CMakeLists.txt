# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Tao Jin

# A stand-in for a sibling that fetches grid's source from VGRID_SOURCE_DIR, as
# the grid_subproject_* tests configure it: grid is then not the top-level
# project, so its tests, examples and -Werror must default off, and the consumer
# builds the library tiers alone, under its own compiler's warnings.
# grid_subproject_glsl builds its kernels on grid's exported GLSL
# (glsl_consumer.cmake), and grid_subproject_headers compiles the public headers
# that reach the volume tier, none of which may hold the hash table's internals
# (internal_names.cpp).
#
# With -DVGRID_CORE_FIRST=ON it makes the core available first, as an
# application that also fetches another sibling does, with VKC_WITH_VULKAN as
# given, so grid's pin yields to it; -DVGRID_CONSUMER_CORE_VERSION=<version>
# stamps that version on the core, standing in for an older one.
# -DVGRID_REQUIRE_GRID=<version> is the consumer's floor on grid.
cmake_minimum_required(VERSION 3.21)
project(vgrid_subproject_consumer LANGUAGES CXX)

if(VGRID_CORE_FIRST)
  include(FetchContent)
  set(_core_source "${FETCHCONTENT_SOURCE_DIR_VOLUMETRIC_KIT_CORE}")
  FetchContent_Declare(volumetric_kit_core SOURCE_DIR "${_core_source}")
  FetchContent_MakeAvailable(volumetric_kit_core)
  if(VGRID_CONSUMER_CORE_VERSION)
    get_target_property(_core volumetric_kit::core_base ALIASED_TARGET)
    set_target_properties(${_core} PROPERTIES VKC_CORE_VERSION
                                              ${VGRID_CONSUMER_CORE_VERSION})
  endif()
endif()
add_subdirectory("${VGRID_SOURCE_DIR}" volumetric_kit_grid)
foreach(_option VGRID_BUILD_TESTS VGRID_BUILD_EXAMPLES VGRID_BUILD_TEST_SUPPORT
                VGRID_WARNINGS_AS_ERRORS)
  if(${_option})
    message(FATAL_ERROR "${_option} must default OFF for a non-top-level grid")
  endif()
endforeach()

# With -DVGRID_REQUIRE_GRID=<version>, the consumer needs a grid that new, and
# checks it as a sibling does, with the core's vkc_require_sibling.
if(VGRID_REQUIRE_GRID)
  vkc_require_sibling(volumetric_kit::grid ${VGRID_REQUIRE_GRID})
endif()

include("${CMAKE_CURRENT_LIST_DIR}/../glsl_consumer.cmake")

# The public headers internal_names.cpp compiles, found in grid's include/ so a
# new one is checked too: every volume header, and every other grid header that
# includes one.
set(_include "${VGRID_SOURCE_DIR}/include")
file(
  GLOB_RECURSE _headers
  RELATIVE "${_include}"
  "${_include}/volumetric_kit/grid/*.hpp")
set(_volume_headers "")
foreach(_header IN LISTS _headers)
  file(STRINGS "${_include}/${_header}" _includes_volume
       REGEX "#include \"volumetric_kit/grid/volume/")
  if(_header MATCHES "^volumetric_kit/grid/volume/" OR _includes_volume)
    string(APPEND _volume_headers "#include \"${_header}\"\n")
  endif()
endforeach()
if(NOT _volume_headers)
  message(FATAL_ERROR "no public volume header under ${_include}")
endif()
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/vgrid_volume_headers.hpp"
     "${_volume_headers}")
# Compiled with the umbrella's usage requirements, not linked to it, so the
# check builds this one file, not grid.
add_library(vgrid_public_headers OBJECT EXCLUDE_FROM_ALL internal_names.cpp)
target_compile_features(vgrid_public_headers PRIVATE cxx_std_17)
target_include_directories(
  vgrid_public_headers
  PRIVATE
    "${CMAKE_CURRENT_BINARY_DIR}"
    "$<TARGET_PROPERTY:volumetric_kit::grid,INTERFACE_INCLUDE_DIRECTORIES>")
target_compile_definitions(
  vgrid_public_headers
  PRIVATE
    "$<TARGET_PROPERTY:volumetric_kit::grid,INTERFACE_COMPILE_DEFINITIONS>")
