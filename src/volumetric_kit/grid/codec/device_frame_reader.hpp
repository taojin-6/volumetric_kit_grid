// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file device_frame_reader.hpp
/// @brief Decodes an intra frame's segments on the device, block for block
///        what @ref read_intra_frame decodes on the host.
///
/// Internal (under src/, never installed). The host parses the frame
/// (@ref parse_intra_frame) and uploads only its tables and payload; the
/// decoded coefficients and masks stay in device memory, where the inverse
/// transform reads them, and only the coordinates and each segment's fault
/// come back. Every bulk buffer is device-local and retained between frames
/// (`ensure_device_scratch`).

#include <cstdint>
#include <memory>
#include <vector>

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

/// @brief Decodes intra frames on the device: one invocation per segment, as
///        the 2026-09-27 format allows.
///
/// One batch per frame: the tables go up and are expanded on the device into
/// each model's slot lookup, the payload goes up, each segment decodes into
/// the inverse transform's layout, and the coordinates and faults come back.
/// After the caller submits it, @ref check gives the host reader's verdict.
///
/// @warning The `Device` and `Allocator` passed to @ref create must
///          outlive this object. Not thread-safe: one frame at a time.
class VR_CODEC_API DeviceFrameReader {
 public:
  /// @brief Build the two kernels.
  /// @return The reader, or a pipeline or pool failure.
  static core::Result<std::unique_ptr<DeviceFrameReader>> create(
      core::Device& device, core::Allocator& allocator);

  ~DeviceFrameReader() = default;
  DeviceFrameReader(const DeviceFrameReader&) = delete;
  DeviceFrameReader& operator=(const DeviceFrameReader&) = delete;

  /// @brief Record the decode of @p frame's segments into @p batch.
  ///
  /// The coordinates and faults arrive when the caller submits @p batch,
  /// which must happen before @ref check.
  /// @return The decoded blocks where the device holds them, valid until the
  ///         next call (no entries for a frame of none, which records
  ///         nothing); `Status::Code::InvalidArgument`, before any buffer
  ///         grows, for a segment longer than
  ///         @ref kMaxDeviceDecodeSegmentSize or a buffer past
  ///         `maxStorageBufferRange`; otherwise a buffer failure.
  core::Result<ResidentBlocks> record_decode(
      core::CommandBatch& batch, const ParsedFrame& frame,
      core::GpuStageScope* stage = nullptr);

  /// @return After the submit, the refusal @ref read_intra_frame makes for
  ///         the same frame's segments (@ref check_segment), or OK.
  core::Status check() const;

  /// @return The decoded coordinates, `ptr` 0, in frame order, as the last
  ///         submit read them back.
  const std::vector<volume::BlockIndex>& blocks() const noexcept {
    return blocks_host_;
  }

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
  core::DescriptorPool pool_;

  // Device-local, retained and grown by ensure_device_scratch.
  core::Buffer tables_;         // per model, per symbol: cum | freq << 16
  core::Buffer slots_;          // per model, per slot: rans_slots.comp's entry
  core::Buffer payload_;        // the segment streams
  core::Buffer segment_words_;  // each segment's first 16-bit word, count + 1
  core::Buffer list_;           // volume::BlockIndex a block
  core::Buffer masks_;
  core::Buffer coefficients_;  // two int16 a word, (K + 1) / 2 words a block
  core::Buffer faults_;        // a SegmentFault a segment

  // What the last submit read back, which check() judges, and the segment
  // size of the frame recorded last.
  std::vector<volume::BlockIndex> blocks_host_;
  std::vector<std::uint32_t> faults_host_;
  std::uint32_t segment_size_ = 0;
};

}  // namespace volumetric_kit::recon::codec::detail
