// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file reference_store.hpp
/// @brief The frame a P-frame is coded or decoded against, kept where the
///        transforms read and write it: the decoder's apply that makes each
///        decoded frame from it, and the encoder's diff that codes each
///        P-frame against it.
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
#include "volumetric_kit/recon/core/dispatch.hpp"
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
/// encoder's comes from its sort and forward transform, intra or P alike,
/// and a P-frame's entries are coded from it against the reference by
/// @ref record_diff (`inter_diff.comp`). Every frame either side commits is
/// the q and masks the next one is coded or decoded against.
///
/// Each slot takes `80 + 4 * coefficient_words(K)` bytes a block, and the
/// sources `8`; the diff takes `64 + 4 * coefficient_words(K)` an entry it
/// codes (a reference entry, and a block of the frame), and 16 more a block
/// of the frame and 8 a reference entry. Every buffer is device-local and
/// retained, growing as @ref reserve and @ref record_diff ask
/// (`ensure_device_scratch`).
///
/// @warning The `Device` and `Allocator` passed to @ref create must
///          outlive this object. Not thread-safe: one frame at a time.
class VR_CODEC_API ReferenceStore {
 public:
  /// @brief Build the apply's and the diff's kernels and the store's first,
  ///        smallest buffers.
  /// @return The store, holding no reference; or a pipeline, pool or
  ///         allocation failure.
  static core::Result<std::unique_ptr<ReferenceStore>> create(
      core::Device& device, core::Allocator& allocator);

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

  /// @return The held reference as the inverse reads it: its list, masks
  ///         and coefficients, block count and K, valid until the next
  ///         @ref commit; no blocks with none held. What `Decoder::rebuild`
  ///         reconstructs.
  ResidentBlocks reference_blocks() const noexcept {
    ResidentBlocks out;
    if (!held_) return out;
    const Slot& reference = slots_[current_];
    out.list = &reference.list;
    out.masks = &reference.masks;
    out.coefficients = &reference.coefficients;
    out.count = reference_.header.block_count;
    out.coefficient_count = reference_.header.params.coefficient_count;
    return out;
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
  ///        recorded the writing of into @p batch before this, over
  ///        @p slices: an invocation a block, sized and gated on the device.
  ///
  /// A source past what is bound -- a reference entry when @p payload holds
  /// none, or past its M; a payload entry past its count -- is read as
  /// absent and counted in @ref faults, which @ref check_faults words.
  /// @param batch    The batch to record into; submitted by the caller.
  /// @param payload  The frame's entries by stream position, where the
  ///                 device frame reader decoded them: a P-frame's M must be
  ///                 the held reference's, at its K.
  /// @param n        The frame's blocks.
  /// @param slices   The apply's slices, a group a block, for @p n blocks.
  /// @param stage    Optional scope the apply's device span records into.
  /// @return The spare, valid until the next call, as the inverse reads it
  ///         (no reference entries); `Status::Code::InvalidArgument` for
  ///         more blocks than @ref reserve made room for, no payload
  ///         buffers, or a P-frame of another M or K than the held
  ///         reference's.
  core::Result<ResidentBlocks> record_apply(
      core::CommandBatch& batch, const ResidentBlocks& payload, std::uint32_t n,
      const recon::detail::IndirectSlices& slices,
      core::GpuStageScope* stage = nullptr);

  /// @return The blocks the last @ref record_apply found a source past what
  ///         was bound for, a u32, once its batch has run.
  const core::Buffer& faults() const noexcept { return faults_; }

  /// @return OK for no faults, or `Status::Code::InvalidArgument` for
  ///         @p faults blocks whose source was past what was bound.
  static core::Status check_faults(std::uint32_t faults);

  /// @brief Record a P-frame's entries against the held reference, coded
  ///        from the frame of @p n blocks the spare holds once the commands
  ///        @p batch already holds have run (an encoder's sorted list,
  ///        copied in, and its forward), into this store's own buffers.
  ///
  /// `inter_diff.comp` gives each reference entry its mode -- remove when
  /// the frame lacks its block, skip when the block is unchanged, keep or
  /// zero otherwise -- and the payload keep and zero carry; then the
  /// survivor ranks; then each block the reference lacks its place among
  /// the new blocks, its coordinate and its payload (the kernel has the
  /// rules and the price that picks zero over keep). The frame's blocks must
  /// be strictly increasing, as the encoder's sort leaves them. Reads
  /// nothing back: only the device knows how many blocks are new.
  /// @param batch  The batch to record into; submitted by the caller.
  /// @param n      The frame's blocks.
  /// @param k      Their K, the held reference's.
  /// @param stage  As @ref record_apply.
  /// @return The P-frame's entries by stream position once @p batch has
  ///         run, valid until the next call, bounded
  ///         (@ref ResidentBlocks::bounded): M + @p n of them, of which the
  ///         device frame writer, counting in the same batch, takes off the
  ///         `ranks[M]` survivors to leave M + A. Or
  ///         `Status::Code::InvalidArgument` for no reference held, a K
  ///         other than its, more blocks than @ref reserve made room for,
  ///         more than 2^32 - 1 entries, or a buffer past
  ///         `maxStorageBufferRange`; otherwise a buffer failure.
  core::Result<ResidentBlocks> record_diff(
      core::CommandBatch& batch, std::uint32_t n, std::uint32_t k,
      core::GpuStageScope* stage = nullptr);

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
  VkDeviceSize max_storage_buffer_range_ = 0;

  // Declared before pool_, so the pool is destroyed first.
  core::ComputeKernel apply_kernel_;
  core::ComputeKernel diff_kernel_;
  core::ComputeKernel ranks_kernel_;  // survivor_ranks.comp, between passes
  core::DescriptorPool pool_;

  // slots_[current_] holds the reference; the other is the spare.
  Slot slots_[2];
  std::uint32_t current_ = 0;
  core::Buffer sources_;  // the apply's, an ApplySource a block
  core::Buffer faults_;   // the apply's out-of-range sources, a u32
  // The diff's P-frame entries: masks and coefficients by stream position,
  // a mode a reference entry, the survivor ranks (M + 1), the new blocks'
  // list, and the ranks' two words (survivor_ranks.comp's verdict layout,
  // the survivor count second), which nothing reads back.
  core::Buffer diff_masks_;
  core::Buffer diff_coefficients_;
  core::Buffer modes_;
  core::Buffer ranks_;
  core::Buffer added_;
  core::Buffer survivors_;

  bool held_ = false;
  FrameReference reference_;
};

}  // namespace volumetric_kit::recon::codec::detail
