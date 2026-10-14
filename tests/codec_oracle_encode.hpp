// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

// The frame the CPU oracle writes for a grid, for a test that holds the
// Encoder to it through the public headers alone.

#include <cstdint>
#include <vector>

#include "buffer_readback.hpp"
#include "volumetric_kit/core/base/result.hpp"
#include "volumetric_kit/recon/codec/encoder.hpp"
#include "volumetric_kit/recon/volume/voxel_block_grid.hpp"

namespace codec_oracle {

// What an Encoder of @p config must write for @p grid: the grid's active
// blocks that hold an observed voxel, found on the host, in (z, y, x) order,
// through the device's forward transform, coded by write_frame.
volumetric_kit::core::Result<std::vector<std::uint8_t>> encode(
    const vr_test::Gpu& gpu,
    volumetric_kit::recon::volume::VoxelBlockGrid& grid,
    const volumetric_kit::recon::codec::EncoderConfig& config);

}  // namespace codec_oracle
