// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include "volumetric_kit/recon/codec/encoder.hpp"

#include <algorithm>
#include <cstddef>
#include <utility>

#include "bitstream.hpp"
#include "dct_transform.hpp"
#include "volumetric_kit/recon/core/allocator.hpp"
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
  c.params.dc_step = 0.0f;
  c.params.ac_step = 0.0f;
  c.segment_size = 0;
  return c;
}

}  // namespace

Encoder::Encoder() = default;
Encoder::~Encoder() = default;

Encoder::Encoder(Encoder&& other) noexcept
    : config_(std::exchange(other.config_, empty_config())),
      transform_(std::move(other.transform_)),
      gpu_timer_(std::move(other.gpu_timer_)) {}

Encoder& Encoder::operator=(Encoder&& other) noexcept {
  if (this != &other) {
    config_ = std::exchange(other.config_, empty_config());
    transform_ = std::move(other.transform_);
    gpu_timer_ = std::move(other.gpu_timer_);
  }
  return *this;
}

bool Encoder::valid() const noexcept {
  return transform_ != nullptr && transform_->valid();
}

Result<Encoder> Encoder::create(Device& device, Allocator& allocator,
                                const EncoderConfig& config) {
  VR_TRY(config.params.validate());
  if (config.segment_size == 0) {
    return Status::invalid_argument("Encoder: segment_size must be at least 1");
  }
  Encoder e;
  e.config_ = config;
  VR_ASSIGN(detail::DctTransform transform,
            detail::DctTransform::create(device, allocator));
  e.transform_ = std::make_unique<detail::DctTransform>(std::move(transform));
  VR_ASSIGN(e.gpu_timer_, GpuTimer::create(device));
  return e;
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

  // The active set, compacted inside this stage so its row is a breakdown
  // ("  ..active set"), then put in the order the frame requires. The hash
  // table's order is whatever the allocate atomics left, so sorting is also
  // what makes the same content give the same bytes.
  VR_ASSIGN(std::vector<volume::BlockIndex> active,
            grid.map().compact_active_blocks(metrics));
  {
    StageScope sort(metrics, "  ..sort");
    detail::sort_by_coord(active);
  }

  // Transform all of them, then keep the blocks with at least one observed
  // voxel. The mask is what says which, and it comes out of the transform, so
  // the filter runs after it rather than before -- in place, sliding each kept
  // block down over the dropped ones, so the readback is never held twice.
  detail::IntraFrame frame;
  frame.voxel_size = grid.grid().voxel_size;
  {
    StageScope forward(metrics, "  ..forward");
    detail::DctBlocks all;
    VR_TRY(transform_->forward(grid, grid.block_list(active), config_.params,
                               all, &stage));
    const std::size_t k = config_.params.coefficient_count;
    frame.coords.reserve(active.size());
    std::size_t kept = 0;
    for (std::size_t i = 0; i < active.size(); ++i) {
      const auto mask = all.masks.begin() +
                        static_cast<std::ptrdiff_t>(i * kMaskWordsPerBlock);
      const bool observed =
          std::any_of(mask, mask + kMaskWordsPerBlock,
                      [](std::uint32_t w) { return w != 0u; });
      if (!observed) {
        continue;
      }
      if (kept != i) {  // kept < i: the destination is a block already read
        std::copy(mask, mask + kMaskWordsPerBlock,
                  all.masks.begin() +
                      static_cast<std::ptrdiff_t>(kept * kMaskWordsPerBlock));
        const auto coeffs =
            all.coefficients.begin() + static_cast<std::ptrdiff_t>(i * k);
        std::copy(
            coeffs, coeffs + static_cast<std::ptrdiff_t>(k),
            all.coefficients.begin() + static_cast<std::ptrdiff_t>(kept * k));
      }
      frame.coords.push_back(active[i].coord);
      ++kept;
    }
    all.masks.resize(kept * kMaskWordsPerBlock);
    all.coefficients.resize(kept * k);
    frame.blocks = std::move(all);
  }

  StageScope entropy(metrics, "  ..rans encode");
  detail::FrameWriteOptions options;
  options.segment_size = config_.segment_size;
  return detail::write_intra_frame(frame, options);
}

}  // namespace volumetric_kit::recon::codec
