// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file device_frame_reader.hpp
/// @brief Decodes a frame's segments on the device, intra or P: the
///        `Decoder`'s only reader, whose entries the tests' CPU oracle pins.
///
/// Internal (under src/, never installed). The host parses the frame
/// (@ref parse_frame) and uploads only its tables and payload; the decoded
/// entries stay in device memory, where the decoder's merge and the inverse
/// transform read them, and only the reader's verdict comes back. Every bulk
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
#include "volumetric_kit/recon/codec/export.hpp"
#include "volumetric_kit/recon/core/fwd.hpp"
#include "volumetric_kit/recon/volume/hash_types.hpp"

namespace volumetric_kit::recon::codec::detail {

/// @brief Decodes frames on the device: one invocation per segment, as the
///        2026-09-27 format allows.
///
/// One batch per frame: the tables go up and are expanded on the device into
/// each model's slot lookup, the payload goes up, each segment decodes its
/// entries into the inverse transform's layout at their stream positions
/// (@ref ResidentBlocks), the segments are judged (`rans_verdict.comp`: the
/// first that failed, its fault or its order against the segment before),
/// and the survivor ranks are scanned (`survivor_ranks.comp`). Eight bytes
/// come back: the first failing segment, and the survivor count. After the
/// caller submits it, @ref check gives the reader's verdict. A skip or
/// remove entry's payload is left as it was: nothing reads it.
///
/// @warning The `Device` and `Allocator` passed to @ref create must
///          outlive this object. Not thread-safe: one frame at a time.
class VR_CODEC_API DeviceFrameReader {
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
  ///
  /// The verdict arrives when the caller submits @p batch, which must happen
  /// before @ref check.
  /// @return The decoded entries where the device holds them, with the
  ///         survivor ranks, valid until the next call (no entries for a
  ///         frame of no segments, which records only the verdict's reset);
  ///         `Status::Code::Unsupported`, recording nothing, for a
  ///         type it does not decode (@ref check_frame_type);
  ///         `Status::Code::InvalidArgument`, before any buffer grows, for a
  ///         size @ref check_segment_size refuses or a buffer past
  ///         `maxStorageBufferRange`; otherwise a buffer failure.
  core::Result<ResidentBlocks> record_decode(
      core::CommandBatch& batch, const ParsedFrame& frame,
      core::GpuStageScope* stage = nullptr);

  /// @return After the submit, the refusal of the first segment that
  ///         failed (@ref check_segment) and, for a P-frame, of its
  ///         survivors (@ref check_survivors), or OK.
  core::Status check() const;

 private:
  DeviceFrameReader() = default;

  // Borrowed (must outlive this).
  core::Device* device_ = nullptr;
  core::Allocator* allocator_ = nullptr;
  std::uint32_t max_workgroup_count_x_ = 0;
  VkDeviceSize max_storage_buffer_range_ = 0;

  // Declared before pool_, so the pool is destroyed first.
  core::ComputeKernel slots_kernel_;
  core::ComputeKernel decode_kernel_;
  core::ComputeKernel verdict_kernel_;
  core::ComputeKernel ranks_kernel_;
  core::DescriptorPool pool_;

  // Device-local, retained and grown by ensure_device_scratch.
  core::Buffer tables_;         // per model, per symbol: cum | freq << 16
  core::Buffer slots_;          // per model, per slot: rans_slots.comp's entry
  core::Buffer payload_;        // the segment streams
  core::Buffer segment_words_;  // each segment's first 16-bit word, count + 1
  core::Buffer list_;           // volume::BlockIndex a new block
  core::Buffer masks_;          // kMaskWordsPerBlock words an entry
  core::Buffer coefficients_;   // coefficient_words(K) words an entry
  core::Buffer modes_;          // a u32 a reference entry
  core::Buffer faults_;         // a SegmentFault a segment
  core::Buffer ranks_;          // M + 1 u32: the survivors before each entry
  core::Buffer verdict_;        // the first failing segment, the survivors

  // What the last submit read back, which check() judges -- the first
  // failing segment as s << 2 | kind (kNoVerdict for none) and the survivor
  // count -- and what of the frame recorded last it judges by.
  std::uint32_t verdict_host_[2] = {};
  FrameType type_ = FrameType::kIntra;
  std::uint32_t block_count_ = 0;  // N
  std::uint32_t new_count_ = 0;    // A
};

}  // namespace volumetric_kit::recon::codec::detail
