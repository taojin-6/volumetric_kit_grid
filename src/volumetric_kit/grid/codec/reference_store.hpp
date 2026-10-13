// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file reference_store.hpp
/// @brief The frame a P-frame decodes against, kept where the inverse
///        transform reads it, and the apply that makes each decoded frame
///        from it.
///
/// Internal (under src/, never installed). The decoded frame's quantized
/// coefficients and masks stay on the device, in the transform's layout;
/// the host keeps its @ref FrameReference and its block coordinates. A
/// reference is never rebuilt from a grid: the grid holds dequantized SDF,
/// and a residual adds to the q the reference decoded to.

#include <cstdint>
#include <memory>
#include <vector>

#include "bitstream.hpp"
#include "dct_blocks.hpp"
#include "dct_transform.hpp"
#include "volumetric_kit/core/base/result.hpp"
#include "volumetric_kit/core/vulkan/buffer.hpp"
#include "volumetric_kit/core/vulkan/compute_kernel.hpp"
#include "volumetric_kit/core/vulkan/descriptor.hpp"
#include "volumetric_kit/recon/codec/export.hpp"
#include "volumetric_kit/recon/core/fwd.hpp"
#include "volumetric_kit/recon/core/math/vector_types.hpp"

namespace volumetric_kit::recon::codec::detail {

/// @brief The held reference frame and the frame being decoded, in two slots
///        of device memory that swap at each @ref commit.
///
/// The current slot holds the reference: each block's 16 mask words and
/// `(K + 1) / 2` coefficient words, in the reference's order. The spare
/// receives the next frame from @ref record_apply (`inter_apply.comp`), with
/// that frame's block list beside it, and becomes the current slot only at
/// @ref commit. So until a decode commits, the reference stays as it was,
/// whatever was recorded or failed. An intra frame takes the same apply,
/// from its payload alone, so every decoded frame lands in the store by one
/// path.
///
/// Each slot takes `64 + 4 * ((K + 1) / 2)` bytes a block, and the list,
/// the sources and, on the host path, the upload `16`, `8` and
/// `64 + 4 * ((K + 1) / 2)`. Every buffer is device-local and retained,
/// growing as @ref reserve asks (`ensure_device_scratch`).
///
/// @warning The `Device` and `Allocator` passed to @ref create must
///          outlive this object. Not thread-safe: one frame at a time.
class VR_CODEC_API ReferenceStore {
 public:
  /// @brief Build the apply kernel and the store's first, smallest buffers.
  /// @return The store, holding no reference; or a pipeline, pool or
  ///         allocation failure.
  static core::Result<std::unique_ptr<ReferenceStore>> create(
      core::Device& device, core::Allocator& allocator);

  ~ReferenceStore() = default;
  ReferenceStore(const ReferenceStore&) = delete;
  ReferenceStore& operator=(const ReferenceStore&) = delete;

  /// @return The held reference, for @ref parse_frame; null until the first
  ///         @ref commit.
  const FrameReference* held() const noexcept {
    return held_ ? &reference_ : nullptr;
  }

  /// @return The held reference's block coordinates, strictly increasing:
  ///         its M entries' places, for @ref merge_frame. Empty with none
  ///         held.
  const std::vector<Vec3i>& coords() const noexcept { return coords_; }

  /// @brief Make the spare hold a frame of @p n blocks of @p k coefficients,
  ///        and the upload @p uploaded entries, before anything is recorded:
  ///        what lets a decoder refuse before it touches the grid.
  /// @param uploaded  The entries @ref record_upload will upload; 0 when the
  ///                  device decoded the payload.
  /// @return OK; `Status::Code::InvalidArgument` for a buffer past
  ///         `maxStorageBufferRange`; or an allocation's failure. The held
  ///         reference is untouched either way.
  core::Status reserve(std::uint32_t n, std::uint32_t k,
                       std::uint32_t uploaded);

  /// @brief Upload the entries of a host-decoded frame that @p sources name,
  ///        in the order they name them, and renumber each source's payload
  ///        to its place in the upload: skip and remove entries, which have
  ///        no payload, never cross the bus.
  /// @param batch    The batch to record into; submitted by the caller.
  /// @param entries  @ref read_frame's payload by stream position.
  /// @param sources  @ref merge_frame's; each payload a stream position of
  ///                 @p entries, rewritten to an upload index.
  /// @return The upload, for @ref record_apply (no list, which the store
  ///         keeps); `Status::Code::InvalidArgument` for more entries than
  ///         @ref reserve made room for, or a source past @p entries.
  core::Result<ResidentBlocks> record_upload(core::CommandBatch& batch,
                                             const DctBlocks& entries,
                                             std::vector<ApplySource>& sources);

  /// @brief Record the apply of a decoded frame into the spare.
  /// @param batch    The batch to record into; submitted by the caller.
  /// @param payload  Where the frame's payload sits: the device frame
  ///                 reader's entries, or @ref record_upload's.
  /// @param sources  Each block's, in frame order (@ref merge_frame); an
  ///                 intra frame's are `{kNoSource, i}`.
  /// @param coords   The blocks' coordinates, the same order: uploaded as
  ///                 the spare's list.
  /// @param stage    Optional scope the apply's device span records into.
  /// @return The spare, valid until the next call, as the inverse reads it
  ///         (no reference entries); `Status::Code::InvalidArgument` for
  ///         sources and coordinates of different lengths, more blocks than
  ///         @ref reserve made room for, a source past the payload or the
  ///         held reference, or one that reads a reference of another K.
  core::Result<ResidentBlocks> record_apply(
      core::CommandBatch& batch, const ResidentBlocks& payload,
      const std::vector<ApplySource>& sources, const std::vector<Vec3i>& coords,
      core::GpuStageScope* stage = nullptr);

  /// @brief Hold the spare as the reference: the only call that changes
  ///        @ref held and @ref coords.
  /// @pre The last @ref record_apply's batch was submitted and succeeded,
  ///      and @p coords is its `coords`, `reference.header.block_count`
  ///      long.
  void commit(const FrameReference& reference, std::vector<Vec3i>&& coords);

 private:
  ReferenceStore() = default;

  /// A frame's coefficients and masks, in the transform's layout.
  struct Slot {
    core::Buffer masks;
    core::Buffer coefficients;
  };

  // Borrowed (must outlive this).
  core::Device* device_ = nullptr;
  core::Allocator* allocator_ = nullptr;
  std::uint32_t max_workgroup_count_x_ = 0;
  VkDeviceSize max_storage_buffer_range_ = 0;

  // Declared before pool_, so the pool is destroyed first.
  core::ComputeKernel apply_kernel_;
  core::DescriptorPool pool_;

  // slots_[current_] holds the reference; the other is the spare.
  Slot slots_[2];
  std::uint32_t current_ = 0;
  core::Buffer list_;     // the spare's blocks, volume::BlockIndex each
  core::Buffer sources_;  // the apply's, an ApplySource a block
  Slot upload_;           // a host-decoded payload, compacted

  bool held_ = false;
  FrameReference reference_;
  std::vector<Vec3i> coords_;
};

}  // namespace volumetric_kit::recon::codec::detail
