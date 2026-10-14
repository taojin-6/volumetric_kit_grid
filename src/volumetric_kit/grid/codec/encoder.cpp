// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include "volumetric_kit/recon/codec/encoder.hpp"

#include <string>
#include <utility>
#include <vector>

#include "bitstream.hpp"
#include "block_lists.hpp"
#include "dct_transform.hpp"
#include "device_frame_writer.hpp"
#include "frame_hash.hpp"
#include "reference_store.hpp"
#include "volumetric_kit/core/vulkan/allocator.hpp"
#include "volumetric_kit/core/vulkan/command_batch.hpp"
#include "volumetric_kit/core/vulkan/device.hpp"

namespace volumetric_kit::recon::codec {
namespace {

// The public default restates the frame's, which the installed header cannot
// name; this is what keeps the two one number.
static_assert(EncoderConfig{}.segment_size == detail::kDefaultSegmentSize,
              "EncoderConfig's default segment size is the frame's");

}  // namespace

Encoder::Encoder() = default;
Encoder::~Encoder() = default;

Encoder::Encoder(Encoder&& other) noexcept = default;
Encoder& Encoder::operator=(Encoder&& other) noexcept = default;

bool Encoder::valid() const noexcept { return transform_ != nullptr; }

core::Result<Encoder> Encoder::create(core::Device& device,
                                      core::Allocator& allocator,
                                      const EncoderConfig& config) {
  VKC_TRY(config.params.validate());
  VKC_TRY(detail::check_segment_size("Encoder", "segment_size",
                                     config.segment_size));
  Encoder e;
  e.config_ = config;
  e.device_ = &device;
  e.allocator_ = &allocator;
  VKC_ASSIGN(e.transform_, detail::DctTransform::create(device, allocator));
  VKC_ASSIGN(e.lists_, detail::BlockLists::create(device, allocator));
  VKC_ASSIGN(e.writer_, detail::DeviceFrameWriter::create(device, allocator));
  VKC_ASSIGN(e.store_, detail::ReferenceStore::create(device, allocator));
  VKC_ASSIGN(e.gpu_timer_, core::GpuTimer::create(device));
  return e;
}

core::Result<std::vector<std::uint8_t>> Encoder::encode(
    volume::VoxelBlockGrid& grid, core::StageMetrics* metrics) {
  // Before any refusal, so a refused call still costs its row.
  core::GpuStageScope stage(metrics, gpu_timer_, "codec encode");
  if (!valid()) {
    return core::Status::invalid_argument(
        "Encoder::encode: moved-from encoder");
  }
  if (!grid.valid()) {
    return core::Status::invalid_argument("Encoder::encode: moved-from grid");
  }

  // The active set, on the device and inside this stage, so its row is a
  // breakdown ("  ..active set"). The map hands back the list an extract just
  // compacted while it still holds, and keeps this one for the extract after,
  // so encoding between two costs neither of them a compaction.
  VKC_ASSIGN(const volume::DeviceBlockList list,
             grid.map().compact_active_blocks_on_device(metrics));
  // Only the blocks with an observed voxel are coded, found on the device
  // before the transform: a fused band is mostly never integrated, and
  // transforming those blocks only to drop them read every one back. Only
  // their count crosses.
  std::uint32_t observed = 0;
  {
    core::StageScope row(metrics, "  ..observed");
    VKC_ASSIGN(observed, transform_->observed(grid, list, &stage));
  }

  // Then, in one batch, into the order the frame requires -- the hash
  // table's order is whatever the allocate atomics left, so sorting is also
  // what makes the same content give the same bytes -- and through the
  // forward, into the reference store's spare: the frame the decoder will
  // hold, which this one holds too once it returns the frame. The same
  // batch counts its symbols: only the counts cross before the coding.
  const volume::VoxelGridParams& gp = grid.grid();
  const CodecParams& params = config_.params;
  std::uint32_t rejected = 0;
  detail::ResidentBlocks entries;
  {
    core::StageScope forward(metrics, "  ..forward");
    VKC_TRY(store_->reserve(observed, params.coefficient_count));
    const detail::FrameBuffers frame = store_->frame_buffers();
    core::CommandBatch batch(*device_, *allocator_);
    VKC_ASSIGN(const core::Buffer* sorted,
               lists_->record_sort(batch, transform_->observed_list(), observed,
                                   &stage));
    VKC_TRY(batch.copy(*sorted, 0, *frame.list, 0,
                       VkDeviceSize(observed) * sizeof(volume::BlockIndex),
                       &stage));
    VKC_ASSIGN(entries,
               transform_->record_forward(
                   batch, grid, *frame.list, observed, params, *frame.masks,
                   *frame.coefficients, rejected, &stage));
    VKC_TRY(writer_->record_count(batch, entries, config_.segment_size,
                                  detail::kDefaultUpdateSize, &stage));
    VKC_TRY(batch.submit());
    VKC_TRY(
        detail::DctTransform::check_rejected("forward", rejected, observed));
  }
  std::vector<std::uint8_t> bytes;
  {
    core::StageScope entropy(metrics, "  ..rans encode");
    VKC_ASSIGN(bytes, writer_->finish(entries, gp.voxel_size, gp.trunc_dist,
                                      params, /*reference=*/nullptr, &stage));
  }
  // Only a frame returned is held. It holds every block the forward made,
  // which the spare holds in frame order.
  VKC_ASSIGN(const detail::FrameHeader header,
             detail::read_frame_header(bytes.data(), bytes.size()));
  if (header.block_count != observed) {
    return core::Status::invalid_argument(
        "Encoder::encode: the frame holds " +
        std::to_string(header.block_count) + " blocks of the " +
        std::to_string(observed) + " the forward made (a bug)");
  }
  store_->commit(detail::FrameReference{
      detail::frame_hash(bytes.data(), bytes.size()), header});
  return bytes;
}

}  // namespace volumetric_kit::recon::codec
