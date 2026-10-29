# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Tao Jin

cmake_minimum_required(VERSION 3.21)

project(
  volumetric_kit_grid
  VERSION 0.2.1
  DESCRIPTION
    "Sparse voxel hashing, meshing and the TSDF geometry codec on Vulkan compute (MoltenVK on Apple) for the volumetric_kit family"
  HOMEPAGE_URL "https://github.com/taojin-6/volumetric_kit_grid"
  LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# --- Options ------------------------------------------------------------------
# PROJECT_IS_TOP_LEVEL is ON only when grid is the root project, not pulled in
# through add_subdirectory/FetchContent; the "build the extras" and -Werror
# defaults follow from it, so a consumer's newer compiler shows it grid's new
# warnings rather than failing its build (tests/subproject checks them). grid
# has no install rules: a consumer fetches it by commit.
option(VGRID_BUILD_EXAMPLES "Build example programs" ${PROJECT_IS_TOP_LEVEL})
option(VGRID_BUILD_TESTS "Build tests" ${PROJECT_IS_TOP_LEVEL})
# grid_test_support, the helpers a sibling's GPU tests share with grid's own
# (tests/support/), built for the tests and for a sibling that asks for it.
option(VGRID_BUILD_TEST_SUPPORT
       "Build grid_test_support for a sibling's tests (on with the tests)" OFF)
option(VGRID_WARNINGS_AS_ERRORS "Treat first-party compiler warnings as errors"
       ${PROJECT_IS_TOP_LEVEL})
# OFF by default: Assimp is a prerequisite you install, never fetched, and image
# / PLY consumers build without it on every platform.
option(VGRID_WITH_ASSIMP
       "Build the Assimp-backed mesh loader (requires an installed Assimp)" OFF)

# Semicolon list of sanitizers (e.g. "address;undefined"); empty = off. Applied
# globally -- see cmake/vgrid_sanitizers.cmake for why.
set(VGRID_SANITIZE
    ""
    CACHE STRING "Sanitizers to enable, e.g. address;undefined (empty = off)")

include(vgrid_warnings)
include(vgrid_sanitizers)

# --- Dependencies + library ---------------------------------------------------
# The Vulkan loader (linked at runtime) + headers, from the system SDK (MoltenVK
# on Apple). Public: grid's headers reach Vulkan through the core's
# core/vulkan/vulkan.hpp, so the requirement is carried on grid's targets.
find_package(Vulkan REQUIRED)
if(VGRID_WITH_ASSIMP)
  find_package(assimp CONFIG REQUIRED)
endif()

# Pinned dependencies: volumetric_kit_core, linked PUBLIC, and tinyply, private
# to the mesh files. The oldest core grid's libraries build with is checked in
# the core that arrives (third_party/, which also checks the config tier the
# examples and tests need): 0.30.0, whose output.max_viewers the examples' serve
# reads; 0.29.0 brought the vulkan/fwd.hpp the public headers include, the
# vkc_export_glsl that exports grid's GLSL and the test support the tests run
# on.
set(VGRID_CORE_MIN_VERSION 0.30.0)
add_subdirectory(third_party)

add_subdirectory(src)

if(VGRID_BUILD_TESTS OR VGRID_BUILD_TEST_SUPPORT)
  add_subdirectory(tests/support)
endif()
# The examples' scaffolding and codec helpers (grid_example_common), for the
# examples and the tests of their helpers, grid's and a sibling's.
if(VGRID_BUILD_EXAMPLES OR VGRID_BUILD_TESTS)
  add_subdirectory(examples/common)
endif()

if(VGRID_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()

if(VGRID_BUILD_EXAMPLES)
  add_subdirectory(examples)
endif()
