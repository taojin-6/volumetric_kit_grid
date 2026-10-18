// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file device_frame_writer.hpp
/// @brief Writes a frame, intra or P, with its rANS coding on the device:
///        the `Encoder`'s only writer, whose bytes the tests' CPU oracle
///        pins.
///
/// Internal (under src/, never installed). The forward output stays in device
/// memory: the host reads back only the symbol counts, to build the frame's
/// tables (@ref frame_tables), the step layout's two totals, a P-frame's
/// survivor count, and the coded segments. Every bulk buffer is device-local
/// and retained between frames (`ensure_device_scratch`); the host reaches
/// them only through `CommandBatch`, so a discrete GPU moves the same few
/// bytes across PCIe that unified memory would.

#include <cstdint>
#include <memory>
#include <vector>

#include "bitstream.hpp"
#include "dct_transform.hpp"
#include "step_layout.hpp"
#include "volumetric_kit/core/base/result.hpp"
#include "volumetric_kit/core/vulkan/buffer.hpp"
#include "volumetric_kit/core/vulkan/compute_kernel.hpp"
#include "volumetric_kit/core/vulkan/descriptor.hpp"
#include "volumetric_kit/recon/codec/export.hpp"
#include "volumetric_kit/recon/core/fwd.hpp"

namespace volumetric_kit::recon::codec::detail {

/// @brief Codes frames on the device (the 2026-09-26 decision's fifth PR):
///        one invocation per segment, as the 2026-09-27 format allows.
///
/// A frame takes two batches. The caller's first batch carries the forward
/// transform (@ref DctTransform::record_forward) and @ref record_count, which
/// counts every model's symbols, each entry's coder steps, and what
/// @ref check_entries refuses, and lays the steps out (@ref StepLayout).
/// After it is submitted, @ref finish makes those refusals, builds the tables
/// on the host and sizes the coding from the layout's totals. The device
/// writes each entry's steps in parallel, runs each segment's serial rANS
/// chain over its run of them into a slot sized by its steps, packs the
/// streams into one payload, and reads back only that payload and the segment
/// lengths. The payload is read up to the last frame of its type's bytes per
/// entry plus 25% with the lengths, and any rest in a second batch.
///
/// A P-frame's entries are its stream positions (@ref ResidentBlocks): each
/// reference entry's mode and payload, which the caller has made against the
/// reference (a keep entry's XOR mask and residuals), then its new blocks.
/// Its count is recorded into its diff's batch, over the bounded entries
/// @ref ReferenceStore::record_diff returns: the count and the layout take
/// the survivors off the bound on the device, @ref record_count reads their
/// count back with the symbol counts, and @ref finish takes it off, so the
/// host learns the new blocks' count only after the submit.
///
/// @warning The `Device` and `Allocator` passed to @ref create must
///          outlive this object. Not thread-safe: one frame at a time.
class VR_CODEC_API DeviceFrameWriter {
 public:
  /// @brief Build the five kernels and the step layout's.
  /// @return The writer, or a pipeline or pool failure.
  static core::Result<std::unique_ptr<DeviceFrameWriter>> create(
      core::Device& device, core::Allocator& allocator);

  ~DeviceFrameWriter() = default;
  DeviceFrameWriter(const DeviceFrameWriter&) = delete;
  DeviceFrameWriter& operator=(const DeviceFrameWriter&) = delete;

  /// @brief Record the counting pass over @p blocks, and the layout of the
  ///        steps it counts, into @p batch.
  ///
  /// The counts arrive when the caller submits @p batch, which must happen
  /// before @ref finish. @p blocks must stay unchanged until then.
  /// @param blocks        The entries: the forward output, or a P-frame's
  ///                      stream positions, the new blocks' coordinates
  ///                      strictly increasing in frame order; or the
  ///                      bounded entries of a P-frame's diff, whose ranks
  ///                      the count reads its surplus from.
  /// @param segment_size  Blocks per segment: a P-frame's new blocks per
  ///                      segment.
  /// @param update_size   A P-frame's reference entries per update segment,
  ///                      read only when @p blocks holds reference entries.
  /// @return OK, or `Status::Code::InvalidArgument` for a size
  ///         @ref check_segment_size refuses, more reference entries than
  ///         entries, a buffer the entries need missing (a bounded set's
  ///         ranks among them), more than 2^30 - 1 segments, or a buffer
  ///         past `maxStorageBufferRange`; otherwise a buffer failure.
  core::Status record_count(core::CommandBatch& batch,
                            const ResidentBlocks& blocks,
                            std::uint32_t segment_size,
                            std::uint32_t update_size,
                            core::GpuStageScope* stage = nullptr);

  /// @brief Code the frame @ref record_count counted, at the sizes it
  ///        counted with, and lay it out.
  /// @param counted     The same entries @ref record_count was given.
  /// @param voxel_size  The grid's voxel edge, metres.
  /// @param trunc_dist  The grid's `trunc_dist`, which the coefficients are
  ///                    fractions of.
  /// @param params      The params the coefficients were made with.
  /// @param reference   A P-frame's reference, whose block count is the
  ///                    blocks' reference entries; null for an intra frame.
  /// @return The frame's bytes; @ref check_coding's refusals, then
  ///         @ref check_entries';
  ///         `Status::Code::InvalidArgument` for blocks @ref record_count
  ///         did not count, a bounded set's survivor count past its
  ///         reference entries or its blocks (a bug, never a frame's), a
  ///         reference whose block count is not the blocks' reference
  ///         entries, coded segments that could outgrow 4 GiB, or a frame
  ///         @ref assemble_frame refuses; or a buffer or dispatch failure.
  ///         `Status::Code::IoError` only if a kernel's table refused a
  ///         symbol it was counted from, which is a bug here.
  core::Result<std::vector<std::uint8_t>> finish(
      const ResidentBlocks& counted, float voxel_size, float trunc_dist,
      const CodecParams& params, const FrameReference* reference,
      core::GpuStageScope* stage = nullptr);

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

  // Where each entry's steps go and each segment's slot (record_count).
  std::unique_ptr<StepLayout> layout_;

  // Device-local, retained and grown by ensure_device_scratch.
  core::Buffer counts_;           // per model, per symbol
  core::Buffer entry_steps_;      // coder steps per entry
  core::Buffer tables_;           // per model, per symbol: cum | freq << 16
  core::Buffer steps_;            // every coder step: start | freq << 16
  core::Buffer slots_;            // each segment's stream, at its slot's top
  core::Buffer lengths_;          // bytes per segment
  core::Buffer payload_offsets_;  // bytes, segment count + 1
  core::Buffer payload_;
  core::Buffer gather_args_;  // the gather's dispatch, which the scan sizes
  core::Buffer failed_;       // set when a table refuses a symbol
  core::Buffer refusal_;      // what check_entries refuses: two words
  core::Buffer surplus_;      // the entries past the frame's (StepLayout)

  // What record_count reads back at the caller's submit, beside the
  // layout's totals: the counts, at frame_model_base in a P-frame's layout,
  // the largest mode past the alphabet and whether a written value is
  // -32768, and a bounded frame's surplus (0 otherwise).
  std::vector<std::uint32_t> counts_host_;
  std::uint32_t refusal_host_[2] = {};
  std::uint32_t surplus_host_ = 0;
  // The sizes and entries record_count counted with (segment size 0 before
  // any), and per frame type the last frame's payload bytes per entry plus
  // 25%, the readback's prediction (0 before any).
  std::uint32_t segment_size_ = 0;
  std::uint32_t update_size_ = 0;
  std::uint32_t entry_count_ = 0;
  std::uint32_t reference_count_ = 0;
  bool bounded_ = false;
  std::uint32_t payload_per_entry_[2] = {};
};

}  // namespace volumetric_kit::recon::codec::detail
