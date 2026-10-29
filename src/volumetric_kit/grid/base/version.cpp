// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include "volumetric_kit/grid/version.hpp"

namespace volumetric_kit::grid {

const char* version_string() noexcept { return VGRID_VERSION_STRING; }

int version_major() noexcept { return VGRID_VERSION_MAJOR; }

int version_minor() noexcept { return VGRID_VERSION_MINOR; }

int version_patch() noexcept { return VGRID_VERSION_PATCH; }

}  // namespace volumetric_kit::grid
