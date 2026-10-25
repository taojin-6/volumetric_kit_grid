// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file step_layout.hpp
/// @brief Where a frame's rANS coder steps go, laid out on the device: the
///        device frame writer's layout, which the host only sizes from.
///
/// Internal (under src/, never installed).

#include <cstdint>
#include <memory>

#include "volumetric_kit/core/base/result.hpp"
#include "volumetric_kit/core/vulkan/buffer.hpp"
#include "volumetric_kit/core/vulkan/compute_kernel.hpp"
#include "volumetric_kit/core/vulkan/descriptor.hpp"
#include "volumetric_kit/core/vulkan/vulkan.hpp"
#include "volumetric_kit/recon/codec/export.hpp"
#include "volumetric_kit/recon/core/fwd.hpp"

namespace volumetric_kit::recon::codec::detail {

/// @brief A frame's coder steps laid out from each entry's step count, on
///        the device (`rans_layout.comp`): where each entry's steps go in the
///        one array the encoder streams through, and each segment's output
///        slot.
///
/// The rANS encoder takes a segment's steps last first, so a segment's
/// entries are laid out last first: with E the exclusive scan of the entries'
/// steps in stream order (@ref starts), segment `[x, y)` takes steps `E[x]`
/// to `E[y]`, and entry p of it ends its steps at `E[x] + E[y] - E[p]`. Each
/// segment's slot holds a 16-bit word a step and the final state's two,
/// rounded up to whole 32-bit words so no two segments share one: a second
/// scan, of those sizes (@ref slot_offsets). Both run in one workgroup on
/// `scan.glsl`, recorded into the batch that counted the steps; their two
/// totals, which size the coding's buffers, reach the host with that batch's
/// results.
///
/// The kernel takes N on the device, from the frame words: its M reference
/// entries and a blocks word of the bound's blocks, less a surplus word, so
/// a frame whose blocks only the device knows (the encoder's forward), or a
/// bounded P-frame's count (@ref ResidentBlocks::bounded), lays out in the
/// batch that counted it. Its segments follow from N, and it writes both
/// totals to a buffer of their own, for a later command of the batch to
/// gather; nothing is read back.
///
/// Every buffer is device-local and retained, grown as a call needs
/// (`ensure_device_scratch`).
///
/// @warning The `Device` and `Allocator` passed to @ref create must
///          outlive this object. Not thread-safe: one call at a time.
class VR_CODEC_API StepLayout {
 public:
  /// @brief Build the kernel.
  /// @return The layout, or a pipeline or pool failure.
  static core::Result<std::unique_ptr<StepLayout>> create(
      core::Device& device, core::Allocator& allocator);

  ~StepLayout() = default;
  StepLayout(const StepLayout&) = delete;
  StepLayout& operator=(const StepLayout&) = delete;
  StepLayout(StepLayout&&) = delete;
  StepLayout& operator=(StepLayout&&) = delete;

  /// @brief What @ref totals holds once the batch ran.
  struct Totals {
    /// Every entry's steps, exact below 2^30, where it stops: 2^30 steps of
    /// 4 bytes reach 4 GiB, which no frame may code.
    std::uint32_t steps = 0;
    /// Every slot's 16-bit words, exact while @ref steps is below 2^30.
    std::uint32_t slot_words = 0;
  };

  /// The frame words' bytes: the surplus, then the blocks (@ref record).
  static constexpr VkDeviceSize kFrameWordsBytes = 2 * sizeof(std::uint32_t);

  /// @brief Record the layout of a frame's entries' coder steps into
  ///        @p batch, its totals left in @ref totals.
  ///
  /// The segments are the frame's (`frame_models.glsl`'s `segment_range`):
  /// @p reference_count update segments of @p update_size entries, then the
  /// rest in segments of @p segment_size. The kernel binds a set this object
  /// keeps, so a batch holds at most one of these calls.
  /// @param batch            The batch to record into; submitted by the
  ///                         caller.
  /// @param steps            Each entry's coder steps by stream position,
  ///                         at most 2^20 each (an entry of the grammar
  ///                         takes at most 1,685), so no sum past the step
  ///                         total's stop wraps; read, never written.
  /// @param words            The frame words, two u32: how many of the
  ///                         frame's blocks are past its entries, clamped
  ///                         to its blocks -- 0, or a bounded P-frame's
  ///                         survivors -- then the frame's blocks, clamped
  ///                         to the bound's, `entries` less M; read, never
  ///                         written.
  /// @param entries          A bound on N: every stream position @p steps
  ///                         holds.
  /// @param reference_count  M: its reference entries.
  /// @param update_size      U: reference entries an update segment, read
  ///                         only when M is nonzero.
  /// @param segment_size     R: new blocks a segment.
  /// @param stage            Optional scope the caller's stage row is open
  ///                         under; the dispatch records its device span
  ///                         into it.
  /// @return OK, recording only the totals' zeroing for no entries; or
  ///         `Status::Code::InvalidArgument` for a size
  ///         @ref check_segment_size refuses, more reference entries than
  ///         entries, a @p steps or @p words `StorageInput::check` refuses,
  ///         or a buffer past `maxStorageBufferRange`; otherwise a buffer
  ///         failure.
  core::Status record(core::CommandBatch& batch, const core::Buffer& steps,
                      const core::Buffer& words, std::uint32_t entries,
                      std::uint32_t reference_count, std::uint32_t update_size,
                      std::uint32_t segment_size,
                      core::GpuStageScope* stage = nullptr);

  /// @return The last @ref record's @ref Totals, once its batch ran.
  const core::Buffer& totals() const noexcept { return totals_; }
  /// @return E, once the batch ran: the steps before each entry in stream
  ///         order, N of them, then their total at N.
  const core::Buffer& starts() const noexcept { return starts_; }
  /// @return Each segment's slot's first 16-bit word, even, then their total
  ///         at the segment count, once the batch ran.
  const core::Buffer& slot_offsets() const noexcept { return slot_offsets_; }

 private:
  StepLayout() = default;

  // Borrowed (must outlive this).
  core::Device* device_ = nullptr;
  core::Allocator* allocator_ = nullptr;
  std::uint32_t max_workgroup_count_x_ = 0;
  VkDeviceSize max_storage_buffer_range_ = 0;

  // Declared before pool_, so the pool is destroyed first.
  core::ComputeKernel kernel_;
  core::DescriptorPool pool_;

  // Device-local, retained.
  core::Buffer starts_;        // N + 1, of the bound
  core::Buffer slot_offsets_;  // segment count + 1, of the bound
  core::Buffer totals_;        // starts_[N] and slot_offsets_[S]
};

}  // namespace volumetric_kit::recon::codec::detail
