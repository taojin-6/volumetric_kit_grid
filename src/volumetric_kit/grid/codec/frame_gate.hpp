// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file frame_gate.hpp
/// @brief The decoder's judgement on the device: the gate a recorded frame's
///        grid writes run behind, their slices, and the results the host
///        reads once the frame's batch has run.
///
/// Internal (under src/, never installed). It carries `VR_CODEC_API` so the
/// tier's tests reach it in a shared-library build.

#include <cstdint>
#include <memory>

#include "volumetric_kit/core/base/result.hpp"
#include "volumetric_kit/core/vulkan/allocator.hpp"
#include "volumetric_kit/core/vulkan/buffer.hpp"
#include "volumetric_kit/core/vulkan/compute_kernel.hpp"
#include "volumetric_kit/core/vulkan/descriptor.hpp"
#include "volumetric_kit/core/vulkan/host_ring.hpp"
#include "volumetric_kit/recon/codec/export.hpp"
#include "volumetric_kit/recon/core/dispatch.hpp"
#include "volumetric_kit/recon/core/fwd.hpp"

namespace volumetric_kit::recon::codec::detail {

/// @brief What a frame's batch leaves in the decoder's results slot, written
///        whole by `frame_results.comp`: every count `Decoder::finish` and
///        `Decoder::rebuild` judge a frame by.
struct DecodeResults {
  std::uint32_t failed;       ///< The first failing segment (kNoVerdict).
  std::uint32_t survivors;    ///< A P-frame's surviving reference entries.
  std::uint32_t collision_q;  ///< Row 13's new block (kNoSource for none).
  std::uint32_t collision_e;  ///< ... and its reference entry.
  std::uint32_t gone;         ///< The grid blocks the frame lacks.
  std::uint32_t missing;      ///< The frame blocks the grid lacks.
  std::uint32_t faults;       ///< The apply's sources past what was bound.
  std::uint32_t rejected;     ///< The inverse's entries the grid lacks.
  std::uint32_t exact;        ///< The deblock's masked blocks.
  std::uint32_t gate;         ///< 1 when every step succeeded.
};
static_assert(sizeof(DecodeResults) == 4 * 10,
              "DecodeResults is ten words, as frame_results.comp writes it");

/// @brief What @ref FrameGate::record_gate judges.
struct GateCheck {
  bool rebuild = false;         ///< A rebuild's grid, not a decode's frame.
  bool inter = false;           ///< A P-frame.
  std::uint32_t blocks = 0;     ///< N.
  std::uint32_t new_count = 0;  ///< A.
};

/// @brief The words the gate and the results read: each piece's own, left
///        on the device by the frame's batch.
struct ResultSources {
  const core::Buffer* verdict = nullptr;      ///< The reader's two words.
  const core::Buffer* collision = nullptr;    ///< The merge's two words.
  const core::Buffer* diff_counts = nullptr;  ///< The diff's two words.
  const core::Buffer* faults = nullptr;       ///< The apply's word.
  const core::Buffer* rejected = nullptr;     ///< The inverse's word.
  const core::Buffer* exact = nullptr;        ///< The deblock's word.
};

/// @brief The slices of a frame's apply, inverse and deblock: a workgroup a
///        block, and the deblock flags' 64 blocks a workgroup.
struct FrameSlices {
  recon::detail::IndirectSlices blocks;
  recon::detail::IndirectSlices flags;
};

/// @brief The decoder's gate, its slices and its results.
///
/// The gate is one word (@ref gate): `frame_gate.comp` writes it from the
/// frame's verdict -- rows 11 to 13 for a decode, the grid's blocks against
/// the frame's for a rebuild -- the map's rounds close it on their own
/// failures, and recon's count-to-slices kernel sizes the apply, the inverse
/// and the deblock from N behind it (@ref record_slices), so a frame refused
/// on the device writes no voxel and no stamp. `frame_results.comp` gathers
/// every count into the decoder's results slot (@ref record_results).
///
/// @warning The `Device` and `Allocator` passed to @ref create must outlive
///          this object. Not thread-safe.
class VR_CODEC_API FrameGate {
 public:
  /// @brief Build the three kernels and the gate word.
  /// @return The gate, or a pipeline, pool or allocation failure.
  static core::Result<std::unique_ptr<FrameGate>> create(
      core::Device& device, core::Allocator& allocator);

  ~FrameGate() = default;
  FrameGate(const FrameGate&) = delete;
  FrameGate& operator=(const FrameGate&) = delete;
  FrameGate(FrameGate&&) = delete;
  FrameGate& operator=(FrameGate&&) = delete;

  /// @return The gate word, a u32 at offset 0.
  const core::Buffer& gate() const noexcept { return gate_; }

  /// @brief Record the gate's judgement of @p check from @p sources' verdict,
  ///        collision and diff counts into @p batch, after what wrote them.
  core::Status record_gate(core::CommandBatch& batch,
                           const ResultSources& sources, const GateCheck& check,
                           core::GpuStageScope* stage = nullptr);

  /// @brief Record the sizing of a frame's @p n blocks behind the gate.
  /// @return The slices, valid until the next call; or a buffer failure.
  core::Result<FrameSlices> record_slices(core::CommandBatch& batch,
                                          std::uint32_t n,
                                          core::GpuStageScope* stage = nullptr);

  /// @brief Record the gather of @p sources and the gate into @p slot, last.
  core::Status record_results(core::CommandBatch& batch,
                              const core::HostSlot& slot,
                              const ResultSources& sources,
                              core::GpuStageScope* stage = nullptr);

 private:
  FrameGate() = default;

  // Borrowed (must outlive this).
  core::Device* device_ = nullptr;
  core::Allocator* allocator_ = nullptr;
  std::uint32_t max_workgroup_count_x_ = 0;

  // Declared before pool_, so the pool is destroyed first.
  core::ComputeKernel gate_kernel_;
  core::ComputeKernel args_kernel_;
  core::ComputeKernel results_kernel_;
  core::DescriptorPool pool_;
  core::Buffer gate_;
  core::Buffer slices_;
};

}  // namespace volumetric_kit::recon::codec::detail
