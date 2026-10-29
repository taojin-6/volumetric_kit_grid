# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Tao Jin

# Vendored dependencies (pinned FetchContent). grid takes Vulkan and glm from
# the system (find_package in the root + base CMakeLists). Besides the family's
# volumetric_kit_core, what is vendored here is header-only with no system
# package to rely on across our targets (Linux/Android/macOS/iOS): tinyply
# below. The fetches run on the first configure (a network is required then;
# afterwards the FetchContent cache serves it).

include(FetchContent)

# --- volumetric_kit_core: the family's shared foundation ----------------------
# grid's Status / Result / VKC_CHECK / log sink come from its base tier, and its
# Vulkan foundation -- device, VMA allocator, buffers, compute kernels, command
# batches, GPU timers -- from its vulkan tier, so they are the same types every
# sibling uses, and one VkDevice can serve all of them. A compiled library and a
# PUBLIC dependency of grid_base. Pinned by commit, never main; a project that
# also fetches another sibling declares the core first, and FetchContent keeps
# the first declaration of a name, so this pin then yields to the project's, and
# vkc_require_core refuses that core if it is older than VGRID_CORE_MIN_VERSION
# or lacks the vulkan tier. Work on both checkouts at once with
# -DFETCHCONTENT_SOURCE_DIR_VOLUMETRIC_KIT_CORE=<core checkout>. The core's
# tests and -Werror stay off in a subproject.
#
# The pin is core 0.29.0 (a4a6fff, core #65's merge). The tests are main()s of
# their own on the core's test support.
FetchContent_Declare(
  volumetric_kit_core
  GIT_REPOSITORY https://github.com/taojin-6/volumetric_kit_core.git
  GIT_TAG a4a6fff4af65b26b25df27c8ed8c9eecbfbfcd87)
set(VKC_WITH_VULKAN ON)
if(VGRID_BUILD_TESTS OR VGRID_BUILD_TEST_SUPPORT)
  set(VKC_BUILD_TEST_SUPPORT ON)
endif()
# Scenario parsing is optional and used only by the examples and their host-side
# adapter tests. Library consumers need no parser.
if(VGRID_BUILD_EXAMPLES OR VGRID_BUILD_TESTS)
  set(VKC_WITH_CONFIG ON)
endif()
FetchContent_MakeAvailable(volumetric_kit_core)

# The set()s above are lost when the project made the core available first,
# without its vulkan tier, its config tier or its test support.
if(VGRID_BUILD_EXAMPLES OR VGRID_BUILD_TESTS)
  vkc_require_core(${VGRID_CORE_MIN_VERSION} VULKAN CONFIG)
else()
  vkc_require_core(${VGRID_CORE_MIN_VERSION} VULKAN)
endif()
if((VGRID_BUILD_TESTS OR VGRID_BUILD_TEST_SUPPORT)
   AND NOT (TARGET volumetric_kit::core_test_base
            AND TARGET volumetric_kit::core_test_policy))
  message(
    FATAL_ERROR
      "grid's tests need volumetric_kit_core's test support: set "
      "VKC_BUILD_TEST_SUPPORT ON before the first FetchContent_MakeAvailable "
      "of the core")
endif()

# --- Asset I/O implementation headers (never exported to consumers) -----------
# tinyply supplies PLY writing. Its implementation TU lives in grid_mesh_io; no
# helper target leaks through its link interface.
FetchContent_Declare(
  tinyply
  GIT_REPOSITORY https://github.com/ddiakopoulos/tinyply.git
  GIT_TAG c9bb690dfe5e9105961e9e28120c48c9ae084bc6 # 3.0
  GIT_SHALLOW FALSE
  SOURCE_SUBDIR do-not-add-tinyply-cmake)
FetchContent_MakeAvailable(tinyply)
set(VGRID_TINYPLY_INCLUDE_DIR
    "${tinyply_SOURCE_DIR}/source"
    PARENT_SCOPE)
