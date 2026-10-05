// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file device_frame_writer.hpp
/// @brief Writes an intra frame with its rANS coding on the device, byte for
///        byte the frame @ref write_intra_frame writes on the host.
///
/// Internal (under src/, never installed). The forward output stays in device
/// memory: the host reads back only the symbol counts, to build the frame's
/// tables with the host's own @ref frame_tables, and the coded segments. Every
/// bulk buffer is device-local and retained between frames
/// (@ref ensure_device_scratch); the host reaches them only through
/// @ref CommandBatch, so a discrete GPU moves the same few bytes across PCIe
/// that unified memory would.

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

namespace volumetric_kit::recon::codec::detail {

/// @brief Codes intra frames on the device (the 2026-09-26 decision's fifth
///        PR): one invocation per segment, as the 2026-09-27 format allows.
///
/// A frame takes two batches. The caller's first batch carries the forward
/// transform (@ref DctTransform::record_forward) and @ref record_count, which
/// counts every model's symbols and each segment's coder steps. After it is
/// submitted, @ref finish builds the tables on the host and places every
/// block's steps. The device writes each block's steps in parallel, runs each
/// segment's serial rANS chain over its run of them into a slot sized by its
/// steps, packs the streams into one payload, and reads back only that payload
/// and the segment lengths. The payload is read up to the last frame's bytes
/// per block plus 25% with the lengths, and any rest in a second batch.
///
/// @warning The @ref Device and @ref Allocator passed to @ref create must
///          outlive this object. Not thread-safe: one frame at a time.
class VR_CODEC_API DeviceFrameWriter {
 public:
  /// @brief Build the five kernels.
  /// @return The writer, or a pipeline or pool failure.
  static core::Result<std::unique_ptr<DeviceFrameWriter>> create(
      core::Device& device, core::Allocator& allocator);

  ~DeviceFrameWriter() = default;
  DeviceFrameWriter(const DeviceFrameWriter&) = delete;
  DeviceFrameWriter& operator=(const DeviceFrameWriter&) = delete;

  /// @brief Record the counting pass over @p blocks into @p batch.
  ///
  /// The counts arrive when the caller submits @p batch, which must happen
  /// before @ref finish. @p blocks must stay unchanged until then.
  /// @param blocks        The forward output, coordinates strictly
  ///                      increasing in frame order and coefficients within
  ///                      +-32767 (what the transform writes).
  /// @param segment_size  Blocks per segment, at least 1, which @ref finish
  ///                      codes at.
  /// @return OK, or @ref Status::Code::InvalidArgument for a segment size of
  ///         0, more than 2^30 - 1 segments, or a buffer past
  ///         `maxStorageBufferRange`; otherwise a buffer failure.
  core::Status record_count(core::CommandBatch& batch,
                            const ResidentBlocks& blocks,
                            std::uint32_t segment_size,
                            core::GpuStageScope* stage = nullptr);

  /// @brief Code the frame @ref record_count counted, at the segment size it
  ///        counted with, and lay it out.
  /// @param blocks      The same output @ref record_count was given.
  /// @param voxel_size  The grid's voxel edge, metres.
  /// @param trunc_dist  The grid's `trunc_dist`, which the coefficients are
  ///                    fractions of.
  /// @param params      The params the coefficients were made with.
  /// @return The frame's bytes, identical to @ref write_intra_frame's for the
  ///         same content; @ref Status::Code::InvalidArgument for invalid
  ///         params, blocks @ref record_count did not count, or a frame
  ///         @ref assemble_intra_frame refuses; or a buffer or dispatch
  ///         failure. @ref Status::Code::IoError only if a kernel's
  ///         table refused a symbol it was counted from, which is a bug here.
  core::Result<std::vector<std::uint8_t>> finish(
      const ResidentBlocks& blocks, float voxel_size, float trunc_dist,
      const CodecParams& params, core::GpuStageScope* stage = nullptr);

  /// @brief Write @p frame from host arrays: upload them, then count and
  ///        finish. What tests compare against @ref write_intra_frame.
  /// @return As @ref write_intra_frame.
  core::Result<std::vector<std::uint8_t>> write(
      const IntraFrame& frame, const FrameWriteOptions& options);

 private:
  DeviceFrameWriter() = default;

  // Borrowed (must outlive this).
  core::Device* device_ = nullptr;
  core::Allocator* allocator_ = nullptr;
  std::uint32_t max_workgroup_count_x_ = 0;
  VkDeviceSize max_storage_buffer_range_ = 0;

  // Declared before pool_, so the pool is destroyed first.
  core::ComputeKernel count_kernel_;
  core::ComputeKernel ops_kernel_;
  core::ComputeKernel encode_kernel_;
  core::ComputeKernel scan_kernel_;
  core::ComputeKernel gather_kernel_;
  core::DescriptorPool pool_;

  // Device-local, retained and grown by ensure_device_scratch.
  core::Buffer counts_;         // per model, per symbol
  core::Buffer block_steps_;    // coder steps per block
  core::Buffer tables_;         // per model, per symbol: cum | freq << 16
  core::Buffer step_offsets_;   // each block's first step
  core::Buffer steps_;          // every coder step: start | freq << 16
  core::Buffer segment_steps_;  // each segment's first step, segment count + 1
  core::Buffer slot_offsets_;   // 16-bit words, segment count + 1
  core::Buffer slots_;          // each segment's stream, at its slot's top
  core::Buffer lengths_;        // bytes per segment
  core::Buffer payload_offsets_;  // bytes, segment count + 1
  core::Buffer payload_;
  core::Buffer gather_args_;  // the gather's dispatch, which the scan sizes
  core::Buffer failed_;       // set when a table refuses a symbol
  // write()'s uploads of a host frame.
  core::Buffer upload_list_;
  core::Buffer upload_masks_;
  core::Buffer upload_coefficients_;

  // Each model's first symbol entry (rans_walk.glsl's model_base), and what
  // record_count reads back at the caller's submit.
  std::vector<std::uint32_t> bases_host_;
  std::vector<std::uint32_t> counts_host_;
  std::vector<std::uint32_t> steps_host_;
  // The segment size record_count counted with (0 before any), and the last
  // frame's payload bytes per block plus 25%, the readback's prediction (0
  // before any).
  std::uint32_t segment_size_ = 0;
  std::uint32_t payload_per_block_ = 0;
};

}  // namespace volumetric_kit::recon::codec::detail
