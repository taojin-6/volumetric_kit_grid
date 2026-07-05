# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Tao Jin

# --- recon_volume: Tier 2 ---------------------------------------------------
# The volumetric data model: the sparse voxel-hash POD layouts. Header-only for
# now (an INTERFACE library); it becomes a compiled STATIC library once the
# VoxelHashMap host class and its compute shaders land. Depends only on
# recon_core (POD math + device macros), per the strict left-to-right tier rule.
include(GNUInstallDirs)

add_library(recon_volume INTERFACE)
add_library(volumetric_kit::recon_volume ALIAS recon_volume)

# recon_core carries the public include dirs (build + install interface), so the
# transitive link is all the include wiring this header-only tier needs.
target_link_libraries(recon_volume INTERFACE recon_core)

# Install the target into the shared export set and ship this tier's own headers
# (the volume/ subtree only), mirroring recon_core so a partial build never
# installs a header for a tier it did not build.
if(VR_INSTALL)
  install(TARGETS recon_volume EXPORT volumetric_kit_reconTargets)
  install(
    DIRECTORY "${PROJECT_SOURCE_DIR}/include/volumetric_kit/recon/volume"
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/volumetric_kit/recon"
    FILES_MATCHING
    PATTERN "*.hpp")
endif()
