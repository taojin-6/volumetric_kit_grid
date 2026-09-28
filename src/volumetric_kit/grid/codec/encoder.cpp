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

Encoder::Encoder() = default;
Encoder::~Encoder() = default;
Encoder::Encoder(Encoder&& other) noexcept = default;
Encoder& Encoder::operator=(Encoder&& other) noexcept = default;

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
    std::sort(active.begin(), active.end(),
              [](const volume::BlockIndex& a, const volume::BlockIndex& b) {
                return detail::coord_less(a.coord, b.coord);
              });
  }

  // Transform all of them, then keep the blocks with at least one observed
  // voxel. The mask is what says which, and it comes out of the transform, so
  // the filter runs after it rather than before.
  detail::IntraFrame frame;
  frame.voxel_size = grid.grid().voxel_size;
  {
    StageScope transform(metrics, "  ..transform");
    detail::DctBlocks all;
    VR_TRY(transform_->forward(grid, grid.block_list(active), config_.params,
                               all, &stage));
    const std::size_t k = config_.params.coefficient_count;
    frame.blocks.params = all.params;
    frame.blocks.trunc_dist = all.trunc_dist;
    frame.coords.reserve(active.size());
    frame.blocks.coefficients.reserve(all.coefficients.size());
    frame.blocks.masks.reserve(all.masks.size());
    for (std::size_t i = 0; i < active.size(); ++i) {
      const auto mask = all.masks.begin() +
                        static_cast<std::ptrdiff_t>(i * kMaskWordsPerBlock);
      const bool observed =
          std::any_of(mask, mask + kMaskWordsPerBlock,
                      [](std::uint32_t w) { return w != 0u; });
      if (!observed) {
        continue;
      }
      frame.coords.push_back(active[i].coord);
      frame.blocks.masks.insert(frame.blocks.masks.end(), mask,
                                mask + kMaskWordsPerBlock);
      const auto coeffs =
          all.coefficients.begin() + static_cast<std::ptrdiff_t>(i * k);
      frame.blocks.coefficients.insert(frame.blocks.coefficients.end(), coeffs,
                                       coeffs + static_cast<std::ptrdiff_t>(k));
    }
  }

  StageScope entropy(metrics, "  ..entropy");
  detail::FrameWriteOptions options;
  options.segment_size = config_.segment_size;
  return detail::write_intra_frame(frame, options);
}

}  // namespace volumetric_kit::recon::codec
