// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include "volumetric_kit/recon/codec/encoder.hpp"

#include <utility>
#include <vector>

#include "bitstream.hpp"
#include "dct_transform.hpp"
#include "device_frame_writer.hpp"
#include "entropy_choice.hpp"
#include "volumetric_kit/recon/core/allocator.hpp"
#include "volumetric_kit/recon/core/command_batch.hpp"
#include "volumetric_kit/recon/core/device.hpp"
#include "volumetric_kit/recon/volume/hash_types.hpp"

namespace volumetric_kit::recon::codec {
namespace {

// The public default restates the writer's, which the installed header cannot
// name; this is what keeps the two one number.
static_assert(EncoderConfig{}.segment_size == detail::kDefaultSegmentSize,
              "EncoderConfig's default segment size is the writer's");

// What a moved-from encoder reports: no coefficients and no segments, a
// configuration that fails validation just as the encoder fails valid().
EncoderConfig empty_config() {
  EncoderConfig c;
  c.params.coefficient_count = 0;
  c.params.quantization_scale = 0.0f;
  c.params.quantization_weights.fill(0.0f);
  c.segment_size = 0;
  return c;
}

}  // namespace

Encoder::Encoder() = default;
Encoder::~Encoder() = default;

Encoder::Encoder(Encoder&& other) noexcept
    : config_(std::exchange(other.config_, empty_config())),
      device_(std::exchange(other.device_, nullptr)),
      allocator_(std::exchange(other.allocator_, nullptr)),
      transform_(std::move(other.transform_)),
      writer_(std::move(other.writer_)),
      writer_failure_(std::exchange(other.writer_failure_, Status{})),
      gpu_timer_(std::move(other.gpu_timer_)) {}

Encoder& Encoder::operator=(Encoder&& other) noexcept {
  if (this != &other) {
    config_ = std::exchange(other.config_, empty_config());
    device_ = std::exchange(other.device_, nullptr);
    allocator_ = std::exchange(other.allocator_, nullptr);
    transform_ = std::move(other.transform_);
    writer_ = std::move(other.writer_);
    writer_failure_ = std::exchange(other.writer_failure_, Status{});
    gpu_timer_ = std::move(other.gpu_timer_);
  }
  return *this;
}

bool Encoder::valid() const noexcept { return transform_ != nullptr; }

Result<Encoder> Encoder::create(Device& device, Allocator& allocator,
                                const EncoderConfig& config) {
  VR_TRY(config.params.validate());
  if (config.segment_size == 0) {
    return Status::invalid_argument("Encoder: segment_size must be at least 1");
  }
  Encoder e;
  e.config_ = config;
  e.device_ = &device;
  e.allocator_ = &allocator;
  VR_ASSIGN(e.transform_, detail::DctTransform::create(device, allocator));
  // kAuto builds the writer at its first device frame, which a small scene
  // never reaches.
  if (config.entropy == EntropyCoding::kDevice) VR_TRY(e.ensure_writer());
  VR_ASSIGN(e.gpu_timer_, GpuTimer::create(device));
  return e;
}

Status Encoder::ensure_writer() {
  if (writer_ != nullptr) return {};
  // A build that failed fails again, so it is tried once.
  if (!writer_failure_.ok()) return writer_failure_;
  Result<std::unique_ptr<detail::DeviceFrameWriter>> writer =
      detail::DeviceFrameWriter::create(*device_, *allocator_);
  if (!writer.ok()) return writer_failure_ = writer.status();
  writer_ = std::move(writer).value();
  return {};
}

Result<std::vector<std::uint8_t>> Encoder::encode_on_device(
    volume::VoxelBlockGrid& grid, const std::vector<volume::BlockIndex>& blocks,
    StageMetrics* metrics, GpuStageScope& stage) {
  // The forward output stays on the device; the same batch counts its
  // symbols, so only the counts cross before the coding.
  std::uint32_t rejected = 0;
  detail::ResidentBlocks resident;
  {
    StageScope forward(metrics, "  ..forward");
    CommandBatch batch(*device_, *allocator_);
    VR_ASSIGN(resident,
              transform_->record_forward(batch, grid, grid.block_list(blocks),
                                         config_.params, rejected, &stage));
    VR_TRY(
        writer_->record_count(batch, resident, config_.segment_size, &stage));
    VR_TRY(batch.submit());
    VR_TRY(detail::DctTransform::check_rejected("forward", rejected,
                                                resident.count));
  }
  StageScope entropy(metrics, "  ..rans encode");
  return writer_->finish(resident, grid.grid().voxel_size,
                         grid.grid().trunc_dist, config_.params, &stage);
}

Result<std::vector<std::uint8_t>> Encoder::encode(volume::VoxelBlockGrid& grid,
                                                  StageMetrics* metrics) {
  // Before any refusal, so a refused call still costs its row.
  GpuStageScope stage(metrics, gpu_timer_, "codec encode");
  if (!valid()) {
    return Status::invalid_argument("Encoder::encode: moved-from encoder");
  }
  if (!grid.valid()) {
    return Status::invalid_argument("Encoder::encode: moved-from grid");
  }

  // The active set, on the device and inside this stage, so its row is a
  // breakdown ("  ..active set"). The map hands back the list a fuse just
  // compacted while it still holds, and keeps this one for the extract after,
  // so encoding between the two costs neither of them a compaction.
  VR_ASSIGN(const volume::DeviceBlockList list,
            grid.map().compact_active_blocks_on_device(metrics));
  // Only the blocks with an observed voxel are coded, found on the device
  // before the transform: a fused band is mostly never integrated, and
  // transforming those blocks only to drop them read every one back.
  std::vector<volume::BlockIndex> blocks;
  {
    StageScope observed(metrics, "  ..observed");
    VR_ASSIGN(blocks, transform_->observed(grid, list, &stage));
  }
  // Then into the order the frame requires. The hash table's order is
  // whatever the allocate atomics left, so sorting is also what makes the
  // same content give the same bytes.
  {
    StageScope sort(metrics, "  ..sort");
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
    Result<std::vector<std::uint8_t>> frame =
        encode_on_device(grid, blocks, metrics, stage);
    if (frame.ok() || !detail::retry_on_host(config_.entropy, frame.status())) {
      return frame;
    }
  }

  detail::IntraFrame frame;
  frame.voxel_size = grid.grid().voxel_size;
  {
    StageScope forward(metrics, "  ..forward");
    VR_TRY(transform_->forward(grid, grid.block_list(blocks), config_.params,
                               frame.blocks, &stage));
  }
  frame.coords.reserve(blocks.size());
  for (const volume::BlockIndex& b : blocks) {
    frame.coords.push_back(b.coord);
  }

  StageScope entropy(metrics, "  ..rans encode");
  detail::FrameWriteOptions options;
  options.segment_size = config_.segment_size;
  return detail::write_intra_frame(frame, options);
}

}  // namespace volumetric_kit::recon::codec
