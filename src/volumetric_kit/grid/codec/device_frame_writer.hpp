// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file device_frame_writer.hpp
/// @brief Writes a frame, intra or P, with its rANS coding on the device:
///        the `Encoder`'s only writer, whose bytes the tests' CPU oracle
///        pins.
///
/// Internal (under src/, never installed). The forward output stays in device
/// memory: the host learns only the symbol counts, to build the frame's
/// tables (@ref frame_tables), the step layout's two totals, the frame's
/// block and survivor counts, and the coded segments, each batch's in one
/// results slot it binds (`core::HostRing`). Every bulk buffer is
/// device-local and retained between frames (`ensure_device_scratch`).

#include <cstdint>
#include <memory>
#include <vector>

#include "bitstream.hpp"
#include "dct_transform.hpp"
#include "step_layout.hpp"
#include "volumetric_kit/core/base/result.hpp"
#include "volumetric_kit/core/base/stage_metrics.hpp"
#include "volumetric_kit/core/vulkan/buffer.hpp"
#include "volumetric_kit/core/vulkan/compute_kernel.hpp"
#include "volumetric_kit/core/vulkan/descriptor.hpp"
#include "volumetric_kit/core/vulkan/fwd.hpp"
#include "volumetric_kit/core/vulkan/gpu_timer.hpp"
#include "volumetric_kit/core/vulkan/host_ring.hpp"
#include "volumetric_kit/recon/codec/encoder.hpp"
#include "volumetric_kit/recon/codec/export.hpp"
#include "volumetric_kit/recon/volume/voxel_hash_map.hpp"

namespace volumetric_kit::recon::codec::detail {

/// @brief What the front batch's last command gathers into its results slot
///        (`encode_results.comp`), before the symbol counts.
struct FrontResults {
  /// The transform's: entries its observed pass and its forward found no
  /// block for; 0 when the count was given none (@ref TransformWords).
  std::uint32_t observed_rejected = 0;
  std::uint32_t forward_rejected = 0;
  /// What @ref check_entries refuses: the largest mode past the alphabet,
  /// and whether a written value is -32768.
  std::uint32_t largest_mode = 0;
  std::uint32_t out_of_range = 0;
  /// The frame words as counted (`counted_entries`): the survivors a bounded
  /// P-frame's bound counts twice, and the frame's blocks.
  std::uint32_t surplus = 0;
  std::uint32_t blocks = 0;
  /// The step layout's totals (@ref StepLayout::Totals).
  std::uint32_t step_total = 0;
  std::uint32_t slot_total = 0;
};
static_assert(sizeof(FrontResults) == 32,
              "FrontResults mirrors encode_results.comp's struct");

/// @brief What the back batch leaves in its results slot (`rans_scan.comp`),
///        before each segment's length.
struct BackResults {
  std::uint32_t failed = 0;  ///< Nonzero when a table refused a symbol.
  std::uint32_t total = 0;   ///< The payload's bytes.
};
static_assert(sizeof(BackResults) == 8,
              "BackResults mirrors rans_scan.comp's struct");

/// @brief The transform's words a front gathers into its results, beside
///        the writer's own: the encoder's observed pass's and forward's
///        reject counts (@ref DctTransform::observed_counts,
///        @ref DctTransform::rejected).
struct TransformWords {
  const core::Buffer* observed = nullptr;  ///< Two u32: rejected, observed.
  const core::Buffer* rejected = nullptr;  ///< One u32.
};

/// @brief Codes frames on the device (the 2026-09-26 decision's fifth PR):
///        one invocation per segment, as the 2026-09-27 format allows.
///
/// A frame takes two batches, the caller's. The first carries the forward
/// transform (@ref DctTransform::record_forward) and @ref record_count, which
/// counts every model's symbols, each entry's coder steps, and what
/// @ref check_entries refuses, lays the steps out (@ref StepLayout) and
/// gathers what the host needs into a results slot. Once it has resolved,
/// @ref record_code makes those refusals, builds the tables on the host,
/// sizes the coding from the layout's totals and records it into the second:
/// the device writes each entry's steps in parallel, runs each segment's
/// serial rANS chain over its run of them into a slot sized by its steps,
/// and packs the streams into a payload slot, in place, or with the
/// payload's likely size copied (the last frame of its type's bytes per
/// entry plus 25%). @ref finish gathers any rest in a batch of its own (a
/// tail) and lays the frame out.
///
/// The entries' count may be a bound whose frame the device alone knows
/// (the encoder's forward): its blocks are a word a kernel wrote. A
/// P-frame's entries are its stream positions (@ref ResidentBlocks): each
/// reference entry's mode and payload, which the caller has made against the
/// reference (a keep entry's XOR mask and residuals), then its new blocks.
/// Over the bounded entries @ref ReferenceStore::record_diff returns, the
/// count and the layout take the survivors off on the device, and
/// @ref record_code takes them off on the host from the front's results.
///
/// @warning The `Device` and `Allocator` passed to @ref create must
///          outlive this object. Not thread-safe: one frame at a time.
class VR_CODEC_API DeviceFrameWriter {
 public:
  /// @brief Build the six kernels and the step layout's.
  /// @return The writer, or a pipeline or pool failure.
  static core::Result<std::unique_ptr<DeviceFrameWriter>> create(
      core::Device& device, core::Allocator& allocator);

  ~DeviceFrameWriter() = default;
  DeviceFrameWriter(const DeviceFrameWriter&) = delete;
  DeviceFrameWriter& operator=(const DeviceFrameWriter&) = delete;

  /// @brief Record the counting pass over @p blocks, the layout of the steps
  ///        it counts, and the gathering of the front's results into a slot,
  ///        into @p batch.
  ///
  /// @p blocks must stay unchanged until @ref record_code. Supersedes an
  /// unfinished frame.
  /// @param blocks        The entries: the forward output, or a P-frame's
  ///                      stream positions, the new blocks' coordinates
  ///                      strictly increasing in frame order; or the
  ///                      bounded entries of a P-frame's diff, whose ranks
  ///                      the count reads its surplus from.
  /// @param segment_size  Blocks per segment: a P-frame's new blocks per
  ///                      segment.
  /// @param update_size   A P-frame's reference entries per update segment,
  ///                      read only when @p blocks holds reference entries.
  /// @param stage         Optional scope the dispatches' device spans record
  ///                      into.
  /// @param frame_blocks  Optional: the frame's blocks, a word a kernel
  ///                      wrote, at most its bound, the entries' blocks
  ///                      (`count` less M); null takes them all.
  /// @param transform     The transform's words to gather; none reads 0.
  /// @return OK, or `Status::Code::InvalidArgument` for a size
  ///         @ref check_segment_size refuses, more reference entries than
  ///         entries, a buffer the entries need missing (a bounded set's
  ///         ranks among them), a @p frame_blocks word missing, misaligned,
  ///         past its buffer or bounding other than the entries' blocks,
  ///         more than 2^30 - 1 segments, or a buffer past
  ///         `maxStorageBufferRange`; otherwise a buffer failure.
  core::Status record_count(core::CommandBatch& batch,
                            const ResidentBlocks& blocks,
                            std::uint32_t segment_size,
                            std::uint32_t update_size,
                            core::GpuStageScope* stage = nullptr,
                            const volume::DeviceCount* frame_blocks = nullptr,
                            const TransformWords& transform = {});

  /// @return The front's results once its batch resolved, from the slot
  ///         @ref record_count gathered them into; zeros before one.
  FrontResults front() const noexcept;

  /// @brief Once the batch @ref record_count recorded into has resolved:
  ///        make the frame's refusals, build its tables, and record the
  ///        coding of the frame it counted into @p batch, at the sizes it
  ///        counted with.
  ///
  /// The tables are recorded first, inline, so MoltenVK opens no blit
  /// encoder mid-batch. Records nothing for a frame of no entries.
  /// @param counted     The same entries @ref record_count was given.
  /// @param voxel_size  The grid's voxel edge, metres.
  /// @param trunc_dist  The grid's `trunc_dist`, which the coefficients are
  ///                    fractions of.
  /// @param params      The params the coefficients were made with.
  /// @param reference   A P-frame's reference, whose block count is the
  ///                    blocks' reference entries; null for an intra frame.
  /// @param metrics     Optional: the tables' `"  ..tables"`, host only.
  /// @param stage       Optional scope the dispatches' device spans record
  ///                    into.
  /// @return OK; @ref check_coding's refusals, then @ref check_entries';
  ///         `Status::Code::InvalidArgument` for blocks @ref record_count
  ///         did not count, a frame word past its bound or a survivor count
  ///         past the reference entries or the blocks (a bug, never a
  ///         frame's), a reference whose block count is not the blocks'
  ///         reference entries, or coded segments that could outgrow 4 GiB;
  ///         or a buffer failure.
  core::Status record_code(core::CommandBatch& batch,
                           const ResidentBlocks& counted, float voxel_size,
                           float trunc_dist, const CodecParams& params,
                           const FrameReference* reference,
                           core::StageMetrics* metrics = nullptr,
                           core::GpuStageScope* stage = nullptr);

  /// @brief Code, in place of what @ref record_count counted, an I-frame of
  ///        no blocks: what the encoder makes of a frame whose front found
  ///        none, a planned P-frame's included. Records nothing; @ref finish
  ///        lays it out.
  /// @param voxel_size  As @ref record_code.
  /// @param trunc_dist  As @ref record_code.
  /// @param params      As @ref record_code.
  /// @return OK; @ref check_coding's refusals; or
  ///         `Status::Code::InvalidArgument` with nothing counted.
  core::Status record_none(float voxel_size, float trunc_dist,
                           const CodecParams& params);

  /// @brief Once the batch @ref record_code recorded into has resolved:
  ///        check the coding, gather any payload past what that batch copied
  ///        (a tail, its batch waited on), and lay the frame out.
  /// @param metrics  Optional: a tail's `"  ..tail"`, both halves.
  /// @param timer    The tail's device span; must outlive the call.
  /// @return The frame, its payload in a slot of this writer's;
  ///         `Status::Code::IoError` if a kernel's table refused a symbol it
  ///         was counted from, or the segments outgrew their slots, which is
  ///         a bug here; `Status::Code::InvalidArgument` for no coded frame,
  ///         or one @ref assemble_head refuses; or a tail's failure.
  core::Result<EncodedFrame> finish(core::StageMetrics* metrics,
                                    core::GpuTimer& timer);

  /// @brief Drop the frame counted or coded, and its slots.
  void drop() noexcept;

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
  core::ComputeKernel results_kernel_;
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
  core::Buffer gather_args_;      // the gather's dispatch, which the scan sizes
  core::Buffer failed_;           // set when a table refuses a symbol
  core::Buffer refusal_;          // what check_entries refuses: two words
  core::Buffer frame_words_;      // the surplus and the blocks (StepLayout)

  // The front's results, the back's and the payload, made at first use, two
  // slots each: one a batch in flight binds, and one the host fills.
  std::unique_ptr<core::HostRing> front_ring_;
  std::unique_ptr<core::HostRing> back_ring_;
  std::unique_ptr<core::HostRing> payload_ring_;
  core::HostSlot front_slot_;
  core::HostSlot back_slot_;
  core::HostSlot payload_slot_;

  // What record_count counted with (segment size 0 before any, or after a
  // drop), and the frame record_code coded: its type and container fields,
  // its segments, what the payload's batch copied out, and whether it holds
  // any entry. Per frame type, the last frame's payload bytes per entry
  // plus 25%, the copied payload's prediction (0 before any).
  std::uint32_t segment_size_ = 0;
  std::uint32_t update_size_ = 0;
  std::uint32_t entry_count_ = 0;
  std::uint32_t reference_count_ = 0;
  std::uint32_t coefficient_count_ = 0;
  bool bounded_ = false;
  bool coded_ = false;
  FrameType type_ = FrameType::kIntra;
  CodedFrame frame_;
  std::uint32_t entries_ = 0;
  std::uint32_t segments_ = 0;
  VkDeviceSize slot_bytes_ = 0;
  VkDeviceSize copied_ = 0;
  std::uint32_t payload_per_entry_[2] = {};
};

}  // namespace volumetric_kit::recon::codec::detail
