// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

// The frame the CPU oracle writes for a grid, for a test that holds the
// Encoder to it through the public headers alone.

#include <cstddef>
#include <cstdint>
#include <vector>

#include "volumetric_kit/core/base/result.hpp"
#include "volumetric_kit/core/testing/buffer_readback.hpp"
#include "volumetric_kit/grid/codec/encoder.hpp"
#include "volumetric_kit/grid/volume/voxel_block_grid.hpp"

namespace codec_oracle {

// What EncoderConfig::true_sdf codes of @p grid, by the study's rule on the
// host (one pass of far-flip cuts, then the cells that keep their blocks),
// from a snapshot: the kept blocks in (z, y, x) order, the cut set as
// DctTransform::record_forward takes it, and counts the tests assert on.
struct SurfaceSet {
  std::vector<volumetric_kit::grid::volume::BlockIndex> blocks;
  std::vector<std::uint32_t> cut;   // num_blocks * 16 words, by heap slot
  std::size_t observed_blocks = 0;  // active blocks with an observed voxel
  std::size_t cut_voxels = 0;
  // Cut voxels that a cell anchored in another block cut.
  std::size_t cut_across_faces = 0;
  // Kept blocks no sign-change cell holds a corner in: kept by the near rule
  // alone.
  std::size_t near_only_blocks = 0;
};

volumetric_kit::core::Result<SurfaceSet> surface_set(
    const volumetric_kit::core::test::Gpu& gpu,
    volumetric_kit::grid::volume::VoxelBlockGrid& grid);

// What an Encoder of @p config must write for @p grid: the grid's active
// blocks that hold an observed voxel -- with config.true_sdf, surface_set's
// blocks and cut set -- found on the host, in (z, y, x) order, through the
// device's forward transform, coded by write_frame.
volumetric_kit::core::Result<std::vector<std::uint8_t>> encode(
    const volumetric_kit::core::test::Gpu& gpu,
    volumetric_kit::grid::volume::VoxelBlockGrid& grid,
    const volumetric_kit::grid::codec::EncoderConfig& config);

}  // namespace codec_oracle
