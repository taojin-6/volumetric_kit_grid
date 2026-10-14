// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

// The CPU oracle's ends on the device (codec_oracle.hpp): a host frame's
// entries uploaded where the device holds a frame's, the forward transform's
// output read back, and host coefficients through the inverse. The transform
// itself is the device's: what the oracle checks is the rANS coding around
// it.

#include <cstdint>
#include <memory>
#include <vector>

#include "buffer_readback.hpp"
#include "codec_oracle.hpp"
#include "dct_transform.hpp"
#include "volumetric_kit/core/base/result.hpp"
#include "volumetric_kit/core/vulkan/buffer.hpp"
#include "volumetric_kit/recon/volume/hash_types.hpp"
#include "volumetric_kit/recon/volume/voxel_block_grid.hpp"

namespace codec_oracle {

namespace vol = volumetric_kit::recon::volume;

// @p b's coefficients as the device holds them: two int16 a word,
// coefficient 2p low, each entry starting a word of its own, an odd K's pad
// half @p pad.
std::vector<std::uint32_t> pack(const DctBlocks& b, std::uint16_t pad = 0);

// Entries where the device holds them, and the buffers, by pointer so a moved
// Resident's blocks still point at them.
struct Resident {
  d::ResidentBlocks blocks;
  std::unique_ptr<vkc::Buffer> list, masks, coefficients, modes;
};

// @p entries uploaded by stream position, packed with @p pad: the first
// `modes.size()` a P-frame's reference entries, with @p modes, then the new
// blocks, whose @p coords go in the list -- as the device frame reader leaves
// a frame's, and as the device frame writer reads the transform's.
vkc::Result<Resident> resident(const vr_test::Gpu& gpu,
                               const DctBlocks& entries,
                               const std::vector<d::BlockMode>& modes = {},
                               const std::vector<vr::Vec3i>& coords = {},
                               std::uint16_t pad = 0);

// @p t's forward output over @p blocks read back, K coefficients to an entry;
// DctTransform::record_forward's refusals, or its reject count's, leaving
// @p out empty.
vkc::Status forward(const vr_test::Gpu& gpu, d::DctTransform& t,
                    const vol::VoxelBlockGrid& grid,
                    const vol::BlockList& blocks,
                    const codec::CodecParams& params, DctBlocks& out);

// @p in's blocks, the entries of @p blocks as given (ptrs included),
// reconstructed into @p grid by @p t's inverse; InvalidArgument for sizes
// that disagree with the list or a trunc_dist other than the grid's, else
// DctTransform::inverse's refusals.
vkc::Status inverse(const vr_test::Gpu& gpu, d::DctTransform& t,
                    vol::VoxelBlockGrid& grid, const vol::BlockList& blocks,
                    const DctBlocks& in);

}  // namespace codec_oracle
