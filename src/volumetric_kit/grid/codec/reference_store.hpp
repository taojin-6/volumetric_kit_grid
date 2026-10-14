// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file reference_store.hpp
/// @brief The frame a P-frame is coded or decoded against, kept where the
///        transforms read and write it, and the apply that makes each
///        decoded frame from it.
///
/// Internal (under src/, never installed). The frame's quantized
/// coefficients and masks, and its block list, stay on the device, in the
/// transform's layout; the host keeps its @ref FrameReference alone. A
/// reference is never rebuilt from a grid: the grid holds dequantized SDF,
/// and a residual adds to the q the reference was coded with.

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

namespace volumetric_kit::recon::codec::detail {

/// @brief Where the next frame's blocks go, as @ref ReferenceStore::reserve
///        last sized them: the store's spare, and the apply's sources. The
///        decoder's merge writes the list and the sources
///        (@ref BlockLists::record_merge) and its apply the masks and
///        coefficients; the encoder copies its sorted list into the list and
///        its forward writes the masks and coefficients
///        (@ref DctTransform::record_forward).
struct FrameBuffers {
  const core::Buffer* list = nullptr;     ///< `volume::BlockIndex` a block.
  const core::Buffer* sources = nullptr;  ///< An @ref ApplySource a block.
  const core::Buffer* masks = nullptr;    ///< `kMaskWordsPerBlock` words.
  /// @ref coefficient_words "coefficient_words(K)" words a block.
  const core::Buffer* coefficients = nullptr;
};

/// @brief The held reference frame and the frame being coded or decoded, in
///        two slots of device memory that swap at each @ref commit.
///
/// The current slot holds the reference: each block's 16 mask words and
/// @ref coefficient_words(K) coefficient words, and its block list, in the
/// reference's order. The spare receives the next frame (@ref frame_buffers)
/// and becomes the current slot only at @ref commit. So until a frame
/// commits, the reference stays as it was, whatever was recorded or failed.
///
/// A decoder's frame comes from its merge and @ref record_apply
/// (`inter_apply.comp`); an intra frame takes the same apply, from its
/// payload alone, so every decoded frame lands in the store by one path. An
/// encoder's comes from its sort and forward transform.
///
/// Each slot takes `80 + 4 * coefficient_words(K)` bytes a block, and the
/// sources `8`. Every buffer is device-local and retained, growing as
/// @ref reserve asks (`ensure_device_scratch`).
///
/// @warning The `Device` and `Allocator` passed to @ref create must
///          outlive this object. Not thread-safe: one frame at a time.
class VR_CODEC_API ReferenceStore {
 public:
  /// @brief Build the apply kernel and the store's first, smallest buffers.
  /// @param max_blocks_per_dispatch  Blocks per apply dispatch, as
  ///        @ref DctTransformConfig::max_blocks_per_dispatch: 0 means the
  ///        device's `maxComputeWorkGroupCount[0]`, which caps it either way.
  /// @return The store, holding no reference; or a pipeline, pool or
  ///         allocation failure.
  static core::Result<std::unique_ptr<ReferenceStore>> create(
      core::Device& device, core::Allocator& allocator,
      std::uint32_t max_blocks_per_dispatch = 0);

  ~ReferenceStore() = default;
  ReferenceStore(const ReferenceStore&) = delete;
  ReferenceStore& operator=(const ReferenceStore&) = delete;
  ReferenceStore(ReferenceStore&&) = delete;
  ReferenceStore& operator=(ReferenceStore&&) = delete;

  /// @return The held reference, for @ref parse_frame; null until the first
  ///         @ref commit.
  const FrameReference* held() const noexcept {
    return held_ ? &reference_ : nullptr;
  }

  /// @return The held reference's block list, `volume::BlockIndex` each,
  ///         `ptr` 0, its coordinates strictly increasing: its M entries'
  ///         places, which the merge searches. With none held, M is 0.
  const core::Buffer& reference_list() const noexcept {
    return slots_[current_].list;
  }

  /// @return The spare and the sources, as @ref reserve last sized them.
  FrameBuffers frame_buffers() const noexcept {
    const Slot& spare = slots_[current_ ^ 1u];
    return FrameBuffers{&spare.list, &sources_, &spare.masks,
                        &spare.coefficients};
  }

  /// @brief Make the spare hold a frame of @p n blocks of @p k
  ///        coefficients before anything is recorded: what lets a decoder
  ///        refuse before it touches the grid, and gives an encoder's forward
  ///        its output.
  /// @return OK; `Status::Code::InvalidArgument` for a buffer past
  ///         `maxStorageBufferRange`; or an allocation's failure. The held
  ///         reference is untouched either way.
  core::Status reserve(std::uint32_t n, std::uint32_t k);

  /// @brief Record the apply of a decoded frame of @p n blocks into the
  ///        spare, from the sources at @ref frame_buffers, which the caller
  ///        recorded the writing of into @p batch before this.
  ///
  /// A source past what is bound -- a reference entry when @p payload holds
  /// none, or past its M; a payload entry past its count -- is read as
  /// absent and counted, and @ref check_apply refuses the frame after the
  /// submit.
  /// @param batch    The batch to record into; submitted by the caller.
  /// @param payload  The frame's entries by stream position, where the
  ///                 device frame reader decoded them: a P-frame's M must be
  ///                 the held reference's, at its K.
  /// @param n        The frame's blocks.
  /// @param stage    Optional scope the apply's device span records into.
  /// @return The spare, valid until the next call, as the inverse reads it
  ///         (no reference entries); `Status::Code::InvalidArgument` for
  ///         more blocks than @ref reserve made room for, no payload
  ///         buffers, or a P-frame of another M or K than the held
  ///         reference's.
  core::Result<ResidentBlocks> record_apply(
      core::CommandBatch& batch, const ResidentBlocks& payload, std::uint32_t n,
      core::GpuStageScope* stage = nullptr);

  /// @return After the last @ref record_apply's batch has run, OK, or
  ///         `Status::Code::InvalidArgument` for the blocks whose source was
  ///         past what was bound.
  core::Status check_apply() const;

  /// @brief Hold the spare as the reference: the only call that changes
  ///        @ref held and @ref reference_list.
  /// @pre The batch that made the spare's frame -- the last
  ///      @ref record_apply's, or an encoder's forward -- was submitted and
  ///      succeeded, and the spare's list holds the frame's
  ///      `reference.header.block_count` blocks.
  void commit(const FrameReference& reference);

 private:
  ReferenceStore() = default;

  /// A frame's coefficients and masks, in the transform's layout, and its
  /// block list.
  struct Slot {
    core::Buffer masks;
    core::Buffer coefficients;
    core::Buffer list;
  };

  // Borrowed (must outlive this).
  core::Device* device_ = nullptr;
  core::Allocator* allocator_ = nullptr;
  std::uint32_t max_workgroup_count_x_ = 0;
  // The blocks one apply dispatch covers: create's cap, at most the above.
  std::uint32_t blocks_per_dispatch_ = 0;
  VkDeviceSize max_storage_buffer_range_ = 0;

  // Declared before pool_, so the pool is destroyed first.
  core::ComputeKernel apply_kernel_;
  core::DescriptorPool pool_;

  // slots_[current_] holds the reference; the other is the spare.
  Slot slots_[2];
  std::uint32_t current_ = 0;
  core::Buffer sources_;  // the apply's, an ApplySource a block
  core::Buffer faults_;   // the apply's out-of-range sources, a u32

  bool held_ = false;
  FrameReference reference_;
  // What the last apply's batch read back of faults_.
  std::uint32_t faults_host_ = 0;
};

}  // namespace volumetric_kit::recon::codec::detail
