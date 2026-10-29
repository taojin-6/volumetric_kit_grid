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
#include <optional>
#include <vector>

#include "codec_oracle.hpp"
#include "dct_transform.hpp"
#include "device_frame_writer.hpp"
#include "volumetric_kit/core/base/result.hpp"
#include "volumetric_kit/core/testing/buffer_readback.hpp"
#include "volumetric_kit/core/vulkan/buffer.hpp"
#include "volumetric_kit/grid/codec/encoder.hpp"
#include "volumetric_kit/grid/volume/hash_types.hpp"
#include "volumetric_kit/grid/volume/voxel_block_grid.hpp"

namespace codec_oracle {

namespace vol = volumetric_kit::grid::volume;

// @p b's coefficients as the device holds them: two int16 a word,
// coefficient 2p low, each entry starting a word of its own, an odd K's pad
// half @p pad.
std::vector<std::uint32_t> pack(const DctBlocks& b, std::uint16_t pad = 0);

// Entries where the device holds them, and the buffers, by pointer so a moved
// Resident's blocks still point at them.
struct Resident {
  d::ResidentBlocks blocks;
  std::unique_ptr<vkc::Buffer> list, masks, coefficients, modes, ranks;
};

// @p entries uploaded by stream position, packed with @p pad: the first
// `modes.size()` a P-frame's reference entries, with @p modes, then the new
// blocks, whose @p coords go in the list -- as the device frame reader leaves
// a frame's, and as the device frame writer reads the transform's. With
// @p tail, bounded as the reference store's diff leaves a P-frame's
// (ResidentBlocks::bounded): *tail positions past them of garbage -- random
// masks, coefficients with -32768 among them, coordinates out of order --
// and ranks whose word M is *tail, the rest garbage.
vkc::Result<Resident> resident(
    const vkc::test::Gpu& gpu, const DctBlocks& entries,
    const std::vector<d::BlockMode>& modes = {},
    const std::vector<vgrid::Vec3i>& coords = {}, std::uint16_t pad = 0,
    std::optional<std::uint32_t> tail = std::nullopt);

// @p t's forward output over @p blocks, uploaded as the device list it takes,
// read back, K coefficients to an entry, with @p cut, if given, as its cut
// set; DctTransform::record_forward's refusals, or its reject count's,
// leaving @p out empty.
vkc::Status forward(const vkc::test::Gpu& gpu, d::DctTransform& t,
                    const vol::VoxelBlockGrid& grid,
                    const std::vector<vol::BlockIndex>& blocks,
                    const codec::CodecParams& params, DctBlocks& out,
                    const vkc::Buffer* cut = nullptr);

// @p in's blocks, the entries of @p blocks as given (ptrs included),
// reconstructed into @p grid by @p t's inverse; InvalidArgument for sizes
// that disagree with the list or a trunc_dist other than the grid's, else
// DctTransform::record_inverse's refusals, or its reject count's.
vkc::Status inverse(const vkc::test::Gpu& gpu, d::DctTransform& t,
                    vol::VoxelBlockGrid& grid,
                    const std::vector<vol::BlockIndex>& blocks,
                    const DctBlocks& in);

// The device writer's frame for @p counted, which it counted in a batch the
// caller submitted: its back half recorded, submitted and finished, the head
// and the payload joined, as the Encoder's encode joins them; or the first
// refusal of record_code, the submit or finish.
vkc::Result<std::vector<std::uint8_t>> write_counted(
    const vkc::test::Gpu& gpu, d::DeviceFrameWriter& writer,
    const d::ResidentBlocks& counted, float voxel_size, float trunc_dist,
    const codec::CodecParams& params, const d::FrameReference* reference);

// encode (codec_oracle_encode.hpp) through @p t, for a test that encodes
// many grids on one transform.
vkc::Result<std::vector<std::uint8_t>> encode(
    const vkc::test::Gpu& gpu, d::DctTransform& t, vol::VoxelBlockGrid& grid,
    const codec::EncoderConfig& config);

}  // namespace codec_oracle
