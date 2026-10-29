// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file device_frame_reader.hpp
/// @brief Decodes a frame's segments on the device, intra or P: the
///        `Decoder`'s only reader, whose entries the tests' CPU oracle pins.
///
/// Internal (under src/, never installed). The host parses the frame
/// (@ref parse_frame) in the decoder's input slot and uploads only its
/// tables and segment words; the payload is read where the slot holds it.
/// The decoded entries and the verdict stay in device memory, where the
/// decoder's merge, its gate and the inverse transform read them. Every bulk
/// buffer is device-local and retained between frames
/// (`ensure_device_scratch`).

#include <cstdint>
#include <memory>

#include "bitstream.hpp"
#include "dct_transform.hpp"
#include "volumetric_kit/core/base/result.hpp"
#include "volumetric_kit/core/vulkan/buffer.hpp"
#include "volumetric_kit/core/vulkan/compute_kernel.hpp"
#include "volumetric_kit/core/vulkan/descriptor.hpp"
#include "volumetric_kit/core/vulkan/fwd.hpp"
#include "volumetric_kit/grid/codec/export.hpp"
#include "volumetric_kit/grid/volume/hash_types.hpp"

namespace volumetric_kit::grid::codec::detail {

/// The verdict word with no failing segment: a failing one is `s << 2 |
/// kind`, kind 1 to 3, and s below @ref kMaxFrameSegments, so it is never
/// this. The decoder's gate (`frame_gate.comp`) mirrors it.
inline constexpr std::uint32_t kNoVerdict = ~0u;

/// @brief Decodes frames on the device: one invocation per segment, as the
///        2026-09-27 format allows.
///
/// Recorded into the decode's batch: the tables go up and are expanded on the
/// device into each model's slot lookup, each segment decodes its entries
/// from the record where the input slot holds it into the inverse
/// transform's layout at their stream positions (@ref ResidentBlocks), the
/// segments are judged (`rans_verdict.comp`: the first that failed, its fault
/// or its order against the segment before), and the survivor ranks are
/// scanned (`survivor_ranks.comp`). The verdict -- the first failing segment
/// and the survivor count -- stays in @ref verdict, which the decoder's gate
/// and its results read; @ref judge words it. A skip or remove entry's
/// payload is left as it was: nothing reads it.
///
/// @warning The `Device` and `Allocator` passed to @ref create must
///          outlive this object. Not thread-safe: one frame at a time.
class VGRID_CODEC_API DeviceFrameReader {
 public:
  /// @brief Build the four kernels.
  /// @return The reader, or a pipeline or pool failure.
  static core::Result<std::unique_ptr<DeviceFrameReader>> create(
      core::Device& device, core::Allocator& allocator);

  ~DeviceFrameReader() = default;
  DeviceFrameReader(const DeviceFrameReader&) = delete;
  DeviceFrameReader& operator=(const DeviceFrameReader&) = delete;

  /// @brief Record the decode of @p frame's segments, and its verdict, into
  ///        @p batch.
  /// @param batch       The batch to record into; submitted by the caller.
  /// @param frame       Parsed from the bytes @p input holds on the device.
  /// @param input       The record and its slack (`CommandBatch::input`).
  /// @param input_bytes Its bytes bound, the slack included.
  /// @param payload_at  The payload's first byte in @p input.
  /// @param stage       Optional scope the dispatches record into.
  /// @return The decoded entries where the device holds them, with the
  ///         survivor ranks, valid until the next call (no entries for a
  ///         frame of no segments, which records only the verdict's reset);
  ///         `Status::Code::Unsupported`, recording nothing, for a
  ///         type it does not decode (@ref check_frame_type);
  ///         `Status::Code::InvalidArgument`, before any buffer grows, for a
  ///         size @ref check_segment_size refuses, a payload past
  ///         @p input_bytes, or a buffer past `maxStorageBufferRange`;
  ///         otherwise a buffer failure.
  core::Result<ResidentBlocks> record_decode(
      core::CommandBatch& batch, const ParsedFrame& frame, VkBuffer input,
      VkDeviceSize input_bytes, VkDeviceSize payload_at,
      core::GpuStageScope* stage = nullptr);

  /// @return The verdict the last recorded decode leaves once its batch has
  ///         run: the first failing segment as `s << 2 | kind`
  ///         (@ref kNoVerdict for none), then the survivor count, a u32 each.
  const core::Buffer& verdict() const noexcept { return verdict_; }

  /// @return The refusal a frame of @p type, @p n blocks and @p a new ones
  ///         earns for verdict words @p failed and @p survivors: the first
  ///         segment that failed (@ref check_segment) and, for a P-frame,
  ///         its survivors (@ref check_survivors); or OK.
  static core::Status judge(FrameType type, std::uint32_t n, std::uint32_t a,
                            std::uint32_t failed, std::uint32_t survivors);

 private:
  DeviceFrameReader() = default;

  // Borrowed (must outlive this).
  core::Device* device_ = nullptr;
  core::Allocator* allocator_ = nullptr;
  std::uint32_t max_workgroup_count_x_ = 0;
  VkDeviceSize max_storage_buffer_range_ = 0;
  VkDeviceSize min_offset_alignment_ = 0;

  // Declared before pool_, so the pool is destroyed first.
  core::ComputeKernel slots_kernel_;
  core::ComputeKernel decode_kernel_;
  core::ComputeKernel verdict_kernel_;
  core::ComputeKernel ranks_kernel_;
  core::DescriptorPool pool_;

  // Device-local, retained and grown by ensure_device_scratch.
  core::Buffer tables_;         // per model, per symbol: cum | freq << 16
  core::Buffer slots_;          // per model, per slot: rans_slots.comp's entry
  core::Buffer segment_words_;  // each segment's first 16-bit word, count + 1
  core::Buffer list_;           // volume::BlockIndex a new block
  core::Buffer masks_;          // kMaskWordsPerBlock words an entry
  core::Buffer coefficients_;   // coefficient_words(K) words an entry
  core::Buffer modes_;          // a u32 a reference entry
  core::Buffer faults_;         // a SegmentFault a segment
  core::Buffer ranks_;          // M + 1 u32: the survivors before each entry
  core::Buffer verdict_;        // the first failing segment, the survivors
};

}  // namespace volumetric_kit::grid::codec::detail
