// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

// The tests' CPU oracle for the codec's frames, which pins the format's
// bytes: a frame written and read on the host by the CPU rANS coder
// (codec_oracle_rans.hpp), over the block grammar the kernels include
// (shaders/frame_grammar.glsl), through the library's own container, tables
// and checks (bitstream.hpp). The device coders must write and read exactly
// what it does, refusals included. Host-only; codec_oracle_device.hpp holds
// its ends on the device.

#include <cstddef>
#include <cstdint>
#include <vector>

#include "bitstream.hpp"
#include "volumetric_kit/core/base/result.hpp"
#include "volumetric_kit/grid/base/math/vector_types.hpp"
#include "volumetric_kit/grid/codec/codec_params.hpp"

namespace codec_oracle {

namespace vkc = volumetric_kit::core;
namespace vgrid = volumetric_kit::grid;
namespace codec = volumetric_kit::grid::codec;
namespace d = volumetric_kit::grid::codec::detail;

// One block list's transform on the host: what the forward transform leaves
// on the device, read back K coefficients to an entry, and what its inverse
// takes (codec_oracle_device.hpp). By list position: entry i's coefficient j
// is coefficients[i * K + j], and its voxel v bit v % 32 of masks[i * 16 +
// v / 32]. The params and trunc_dist the coefficients were made with travel
// with them, since the steps are fractions of trunc_dist.
struct DctBlocks {
  codec::CodecParams params;
  float trunc_dist = 0.0f;
  std::vector<std::int16_t> coefficients;  // each within +-32767
  std::vector<std::uint32_t> masks;
};

// One frame's content, as write_frame takes it and read_frame returns it.
struct Frame {
  d::FrameType type = d::FrameType::kIntra;
  // The grid's voxel edge, metres: with the block coordinates it places the
  // frame in the world.
  float voxel_size = 0.0f;
  // A P-frame's reference; an intra frame has none (hash 0, block count 0).
  d::FrameReference reference;
  // A P-frame's mode for each of its reference's M entries, in reference
  // order; empty for an intra frame.
  std::vector<d::BlockMode> modes;
  // Block coordinates, strictly increasing in (z, y, x) order, as the format
  // requires: a P-frame's A new blocks, an intra frame's every block.
  std::vector<vgrid::Vec3i> coords;
  // The payload by stream position: a P-frame's M reference entries, then its
  // new blocks, entry for entry with coords; an intra frame's blocks. A keep
  // entry holds its XOR mask and residuals. Skip and remove entries are zero
  // when read and ignored when written.
  DctBlocks blocks;
};

struct FrameWriteOptions {
  // R: blocks per segment, a P-frame's new blocks per segment.
  std::uint32_t segment_size = d::kDefaultSegmentSize;
  // U: a P-frame's reference entries per update segment; the Encoder's
  // default config's U, its R.
  std::uint32_t update_size = d::kDefaultSegmentSize;
};

// A frame's bytes; or Unsupported for a type past a P-frame or a motion mode;
// InvalidArgument for a non-positive or non-finite voxel_size / trunc_dist,
// invalid params, a size check_segment_size refuses, a mode count other than
// the reference's block count, a mode past motion, M + A past 2^32 - 1,
// arrays whose sizes disagree with the entries, coordinates not strictly
// increasing, a written coefficient or residual outside +-32767, a P-frame
// whose coding parameters are not its reference's, an intra frame with a
// reference, or a frame whose SEGMENTS or PAYLOAD would outgrow its u32
// length; IoError only if the coder refuses a symbol its own tables were
// counted from, a bug here.
vkc::Result<std::vector<std::uint8_t>> write_frame(
    const Frame& frame, const FrameWriteOptions& options = {});

// d::parse_frame, then decode_frame: its refusals, in the order bitstream.hpp
// gives.
vkc::Result<Frame> read_frame(const std::uint8_t* data, std::size_t size,
                              std::uint32_t max_blocks,
                              const d::FrameReference* reference = nullptr);

// A parsed frame's segments decoded: Unsupported for a type it does not
// decode (check_frame_type), check_segment's and check_survivors' refusals,
// or OutOfMemory for arrays past this platform's address space (reachable on
// a 32-bit build).
vkc::Result<Frame> decode_frame(const d::ParsedFrame& parsed);

// The alphabet of frequency table @p model at @p k (frame_models.glsl).
std::uint32_t model_alphabet(std::uint32_t model, std::uint32_t k);

// Each of @p frame's entries' coder steps by stream position, in segments of
// @p segment_size new blocks: a symbol one, a raw field one per chunk of at
// most 12 bits -- the steps the device writer's count pass finds, and its
// step layout lays out. A mode past the alphabet takes none.
std::vector<std::uint32_t> entry_steps(const Frame& frame,
                                       std::uint32_t segment_size);

// The bits the Encoder's static price (inter_diff.comp) gives a payload of
// @p mask (16 words) and @p k coefficients: frame_grammar.glsl's
// emit_payload walked with a symbol costing bit_length(symbol) + 1 bits, a
// byte model's 8, and a raw field its bits -- the price that picks a
// P-frame's zero over keep when strictly lower.
std::uint64_t payload_price(const std::uint32_t* mask,
                            const std::int16_t* coefficients, std::uint32_t k);

// The Encoder's one-step hold (inter_hold.comp) on the host: @p target with
// each coefficient of a block on one of @p reference's coordinates set to
// that block's when the two lie within one step, |q - q_ref| <= 1; its masks,
// its other blocks and its pad as they are. Both are intra frames' content.
// A reference of no blocks, or whose coding parameters are not @p target's
// (d::coding_difference), holds nothing. Block identity by a std::map, the
// values in plain int arithmetic.
Frame held(const Frame& reference, const Frame& target);

// The device writer's step layout (step_layout.hpp), on the host, in 64
// bits: E, the steps before each entry, then their total; and each
// segment's slot's first 16-bit word -- a word a step and the state's two,
// rounded up to even -- then their total.
struct StepLayout {
  std::vector<std::uint64_t> starts;
  std::vector<std::uint64_t> slot_offsets;
};

// @p steps' layout in a frame of @p m reference entries in update segments
// of @p u, then new blocks in segments of @p r.
StepLayout step_layout(const std::vector<std::uint32_t>& steps, std::uint32_t m,
                       std::uint32_t u, std::uint32_t r);

}  // namespace codec_oracle
