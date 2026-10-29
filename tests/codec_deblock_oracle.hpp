// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

// The decoder-side deblock's CPU oracle: the 2026-10-21 quantization study's
// filters (its deblock.cpp), ported in double -- the line fit (linfit n 2,
// m 3, a 1, no gate, a clip) and the Gaussian (r 1, a clip) -- over a grid's
// host snapshot, under the per-block fallback (that decision's 2026-10-23
// amendment): a block holding an unobserved voxel keeps its values, no line
// is fitted across a face it shares, and its observed voxels are read as
// they are. Each pass reads a snapshot of the last pass's output, as the
// study's does, whatever order the device's workgroups run in; and it keeps
// the field between passes as the device does -- a line pass in metres, read
// back as float(tsdf / trunc_dist), the Gaussian in float across its three
// passes -- so its decisions match the device's and its values agree within
// float rounding. Test-only: the library runs the deblock on the device
// alone.

#include <cstddef>

#include "deblock.hpp"
#include "volumetric_kit/grid/codec/codec_params.hpp"
#include "volumetric_kit/grid/testing/codec_fixture.hpp"

namespace codec_deblock_oracle {

namespace d = volumetric_kit::grid::codec::detail;

// The study's T, its arithmetic as it wrote it (deblock.cpp's step_scale),
// which detail::step_scale must reproduce bit for bit.
float study_step_scale(const volumetric_kit::grid::codec::CodecParams& params);

// @p grid's tsdf filtered by @p passes in place, in metres, its band
// @p trunc_dist; a voxel is observed when its weight is at least
// volume::kObservedWeight, and a block's neighbours are found by coordinate.
// With @p fallback false, every block is filtered, its unobserved voxels
// alone kept, as before the fallback: what a test compares the fallback
// with.
void apply(codec_fixture::Snapshot& grid, const d::DeblockPasses& passes,
           float trunc_dist, bool fallback = true);

// Whether block @p b of @p grid holds an unobserved voxel: masked, which the
// fallback keeps as it is.
bool masked(const codec_fixture::Snapshot& grid, std::size_t b);

}  // namespace codec_deblock_oracle
