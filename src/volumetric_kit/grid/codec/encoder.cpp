// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include "volumetric_kit/recon/codec/encoder.hpp"

#include <utility>
#include <vector>

#include "bitstream.hpp"
#include "dct_transform.hpp"
#include "device_frame_writer.hpp"
#include "entropy_choice.hpp"
#include "volumetric_kit/core/vulkan/allocator.hpp"
#include "volumetric_kit/core/vulkan/command_batch.hpp"
#include "volumetric_kit/core/vulkan/device.hpp"
#include "volumetric_kit/recon/volume/hash_types.hpp"

namespace volumetric_kit::recon::codec {
namespace {

// The public default restates the writer's, which the installed header cannot
// name; this is what keeps the two one number.
static_assert(EncoderConfig{}.segment_size == detail::kDefaultSegmentSize,
              "EncoderConfig's default segment size is the writer's");

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
  if (config.segment_size == 0) {
    return core::Status::invalid_argument(
        "Encoder: segment_size must be at least 1");
  }
  Encoder e;
  e.config_ = config;
  e.device_ = &device;
  e.allocator_ = &allocator;
  VKC_ASSIGN(e.transform_, detail::DctTransform::create(device, allocator));
  // kAuto builds the writer at its first device frame, which a small scene
  // never reaches.
  if (config.entropy == EntropyCoding::kDevice) VKC_TRY(e.ensure_writer());
  VKC_ASSIGN(e.gpu_timer_, core::GpuTimer::create(device));
  return e;
}

core::Status Encoder::ensure_writer() {
  if (writer_ != nullptr) return {};
  // A build that failed fails again, so it is tried once.
  if (!writer_failure_.ok()) return writer_failure_;
  core::Result<std::unique_ptr<detail::DeviceFrameWriter>> writer =
      detail::DeviceFrameWriter::create(*device_, *allocator_);
  if (!writer.ok()) return writer_failure_ = writer.status();
  writer_ = std::move(writer).value();
  return {};
}

core::Result<std::vector<std::uint8_t>> Encoder::encode_on_device(
    volume::VoxelBlockGrid& grid, const std::vector<volume::BlockIndex>& blocks,
    core::StageMetrics* metrics, core::GpuStageScope& stage) {
  // The forward output stays on the device; the same batch counts its
  // symbols, so only the counts cross before the coding.
  std::uint32_t rejected = 0;
  detail::ResidentBlocks resident;
  {
    core::StageScope forward(metrics, "  ..forward");
    core::CommandBatch batch(*device_, *allocator_);
    VKC_ASSIGN(resident,
               transform_->record_forward(batch, grid, grid.block_list(blocks),
                                          config_.params, rejected, &stage));
    VKC_TRY(
        writer_->record_count(batch, resident, config_.segment_size, &stage));
    VKC_TRY(batch.submit());
    VKC_TRY(detail::DctTransform::check_rejected("forward", rejected,
                                                 resident.count));
  }
  core::StageScope entropy(metrics, "  ..rans encode");
  return writer_->finish(resident, grid.grid().voxel_size,
                         grid.grid().trunc_dist, config_.params, &stage);
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
  // breakdown ("  ..active set"). The map hands back the list a fuse just
  // compacted while it still holds, and keeps this one for the extract after,
  // so encoding between the two costs neither of them a compaction.
  VKC_ASSIGN(const volume::DeviceBlockList list,
             grid.map().compact_active_blocks_on_device(metrics));
  // Only the blocks with an observed voxel are coded, found on the device
  // before the transform: a fused band is mostly never integrated, and
  // transforming those blocks only to drop them read every one back.
  std::vector<volume::BlockIndex> blocks;
  {
    core::StageScope observed(metrics, "  ..observed");
    VKC_ASSIGN(blocks, transform_->observed(grid, list, &stage));
  }
  // Then into the order the frame requires. The hash table's order is
  // whatever the allocate atomics left, so sorting is also what makes the
  // same content give the same bytes.
  {
    core::StageScope sort(metrics, "  ..sort");
    detail::sort_by_coord(blocks);
  }

  const std::uint64_t segments =
      detail::frame_segment_count(blocks.size(), config_.segment_size);
  // kAuto codes on the host whatever the device cannot: a frame it cannot
  // hold, or every frame once its kernels failed to build. The host refuses
  // a bad input again, with its own message.
  if (detail::codes_on_device(config_.entropy, segments,
                              kMinDeviceEncodeSegments) &&
      ensure_writer().ok()) {
    core::Result<std::vector<std::uint8_t>> frame =
        encode_on_device(grid, blocks, metrics, stage);
    if (frame.ok() || !detail::retry_on_host(config_.entropy, frame.status())) {
      return frame;
    }
  }

  detail::IntraFrame frame;
  frame.voxel_size = grid.grid().voxel_size;
  {
    core::StageScope forward(metrics, "  ..forward");
    VKC_TRY(transform_->forward(grid, grid.block_list(blocks), config_.params,
                                frame.blocks, &stage));
  }
  frame.coords.reserve(blocks.size());
  for (const volume::BlockIndex& b : blocks) {
    frame.coords.push_back(b.coord);
  }

  core::StageScope entropy(metrics, "  ..rans encode");
  detail::FrameWriteOptions options;
  options.segment_size = config_.segment_size;
  return detail::write_intra_frame(frame, options);
}

}  // namespace volumetric_kit::recon::codec
