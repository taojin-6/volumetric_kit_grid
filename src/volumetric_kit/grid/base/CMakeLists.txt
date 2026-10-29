# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Tao Jin

# --- grid_base: Tier 1 --------------------------------------------------------
# Foundation tier: what is grid's own beneath every other tier -- the
# requirements grid's kernels place on a device, grid's log source, the
# GLM-backed vector/matrix types, the version and the split dispatch
# (dispatch.hpp, with its count-to-indirect-slices kernel and
# dispatch_slices.glsl) -- over volumetric_kit_core's error handling, log sink
# and contract checks (base tier) and its Vulkan foundation (vulkan tier), which
# grid's tiers use under the core's names. Every other tier depends on this.
# Vulkan is reached only through the core's core/vulkan/vulkan.hpp umbrella.
# Deps: volumetric_kit_core, header-only GLM and the Vulkan loader/headers.
include(GenerateExportHeader)

# The core's device serializes submits with std::mutex, so everything above
# grid_base needs the platform threading runtime. Link it explicitly rather than
# relying on the compiler's implicit pthread, so the static archive resolves on
# toolchains that require it -- GCC/libstdc++ on Linux with --as-needed,
# Android, and static links. (macOS folds pthread into libc, which would
# otherwise hide the gap.)
find_package(Threads REQUIRED)

# GLM: header-only vector/matrix math. PUBLIC -- it appears in public headers
# (math/vector_types.hpp aliases glm::vec*/glm::mat4), so consumers need it too.
# An imported CONFIG target brings glm's headers in as SYSTEM includes, so glm
# internals don't trip -Werror. We link the canonical glm::glm target, which
# every glm packaging provides: the newer glm::glm-header-only target does not
# exist on older packagings (e.g. Ubuntu's libglm-dev).
find_package(glm CONFIG REQUIRED)

# The Vulkan loader + headers. PUBLIC: the public headers reach Vulkan through
# the core's core/vulkan/vulkan.hpp and so include <vulkan/vulkan.h>, meaning
# every consumer compiles against it and links the loader. MoltenVK is the
# loader/driver on Apple.
find_package(Vulkan REQUIRED)

set(_vgrid_generated_inc "${CMAKE_CURRENT_BINARY_DIR}/generated/include")

add_library(grid_base)
add_library(volumetric_kit::grid_base ALIAS grid_base)

# The Vulkan foundation is the core's (core_vulkan); what grid compiles of its
# own here is the version query, the split dispatch's test seam and the one
# count-to-indirect-slices kernel. The rest of the tier is header-only.
target_sources(grid_base PRIVATE dispatch.cpp version.cpp)
include(vgrid_embed)
vgrid_embed_shaders(grid_base SHADERS
                    "${CMAKE_CURRENT_SOURCE_DIR}/shaders/dispatch_args.comp")

# Generated headers: the VGRID_BASE_API export macro + the configured version
# header. The export header is per-tier (base/export.hpp + VGRID_BASE_API +
# VGRID_BASE_STATIC_DEFINE) so each tier generates its own isolated macro/guard
# and they never collide on one path.
generate_export_header(
  grid_base
  BASE_NAME
  VGRID_BASE
  EXPORT_MACRO_NAME
  VGRID_BASE_API
  EXPORT_FILE_NAME
  "${_vgrid_generated_inc}/volumetric_kit/grid/base/export.hpp")
configure_file(
  "${PROJECT_SOURCE_DIR}/include/volumetric_kit/grid/version.hpp.in"
  "${_vgrid_generated_inc}/volumetric_kit/grid/version.hpp" @ONLY)

set_target_properties(
  grid_base
  PROPERTIES CXX_VISIBILITY_PRESET hidden
             VISIBILITY_INLINES_HIDDEN ON
             POSITION_INDEPENDENT_CODE ON)

# With a static archive, VGRID_BASE_API must expand to nothing for consumers.
get_target_property(_vgrid_type grid_base TYPE)
if(_vgrid_type STREQUAL "STATIC_LIBRARY")
  target_compile_definitions(grid_base PUBLIC VGRID_BASE_STATIC_DEFINE)
endif()

target_compile_features(grid_base PUBLIC cxx_std_17)

# PUBLIC: consumers compile our public headers (which include GLM and the core's
# base and vulkan tiers) and, for a static archive, must also link the threading
# runtime -- so the target carries these requirements.
target_link_libraries(
  grid_base PUBLIC volumetric_kit::core_base volumetric_kit::core_vulkan
                   Threads::Threads glm::glm Vulkan::Vulkan)

target_include_directories(
  grid_base PUBLIC "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>"
                   "$<BUILD_INTERFACE:${_vgrid_generated_inc}>")

vgrid_target_warnings(grid_base)

# The split dispatch's GLSL, for a consumer's kernels to #include as grid's own
# do ("volumetric_kit/grid/base/shaders/dispatch_slices.glsl"), through the
# core's INCLUDE_TARGETS volumetric_kit::grid_base. The other tiers' shaders
# read src/ (vgrid_embed_shaders).
vkc_export_glsl(grid_base PREFIX volumetric_kit/grid/base/shaders FILES
                shaders/dispatch_slices.glsl)
