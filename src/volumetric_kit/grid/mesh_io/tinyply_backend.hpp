// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

// Keep the vendored implementation's symbols distinct from a consumer's own
// tinyply, including in static builds. Include at global scope so the vendor's
// standard-library includes remain in their normal namespaces. The remapping
// applies only during this private include and never reaches a public header.
#define tinyply volumetric_kit_recon_io_tinyply
#include "tinyply.h"
#undef tinyply

namespace volumetric_kit::recon::io::detail {
namespace tinyply = ::volumetric_kit_recon_io_tinyply;
}  // namespace volumetric_kit::recon::io::detail
