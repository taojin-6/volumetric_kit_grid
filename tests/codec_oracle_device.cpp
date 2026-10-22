// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include "codec_oracle_device.hpp"

#include <algorithm>
#include <limits>
#include <utility>

#include "codec_frames.hpp"
#include "codec_oracle_encode.hpp"
#include "grid_readback.hpp"
#include "volumetric_kit/core/vulkan/command_batch.hpp"
#include "volumetric_kit/core/vulkan/compute_util.hpp"

namespace codec_oracle {

std::vector<std::uint32_t> pack(const DctBlocks& b, std::uint16_t pad) {
  const std::uint32_t k = b.params.coefficient_count;
  const std::uint32_t words = d::coefficient_words(k);
  const std::size_t n = b.masks.size() / codec::kMaskWordsPerBlock;
  std::vector<std::uint32_t> out(n * words, 0);
  for (std::size_t i = 0; i < n; ++i) {
    for (std::uint32_t j = 0; j < words * 2; ++j) {
      const std::uint16_t v =
          j < k ? static_cast<std::uint16_t>(b.coefficients[i * k + j]) : pad;
      out[i * words + j / 2] |= std::uint32_t(v) << (16 * (j & 1));
    }
  }
  return out;
}

vkc::Result<Resident> resident(const vr_test::Gpu& gpu,
                               const DctBlocks& entries,
                               const std::vector<d::BlockMode>& modes,
                               const std::vector<vr::Vec3i>& coords,
                               std::uint16_t pad,
                               std::optional<std::uint32_t> tail) {
  const auto upload =
      [&gpu](const void* bytes,
             VkDeviceSize size) -> vkc::Result<std::unique_ptr<vkc::Buffer>> {
    VKC_ASSIGN(vkc::Buffer buffer, vr_test::upload_device_buffer(
                                       gpu.device, gpu.allocator, bytes, size));
    return std::make_unique<vkc::Buffer>(std::move(buffer));
  };
  // The bound's garbage past the frame's entries, which no count may read.
  const auto m = static_cast<std::uint32_t>(modes.size());
  DctBlocks bound;
  std::vector<vr::Vec3i> bound_coords;
  std::vector<std::uint32_t> ranks;
  if (tail) {
    codec_frames::Lcg rng{0x7a11u + *tail};
    const std::uint32_t k = entries.params.coefficient_count;
    bound = entries;
    bound_coords = coords;
    for (std::uint32_t i = 0; i < *tail; ++i) {
      for (std::uint32_t w = 0; w < codec::kMaskWordsPerBlock; ++w) {
        bound.masks.push_back(rng.next());
      }
      for (std::uint32_t j = 0; j < k; ++j) {
        bound.coefficients.push_back(
            j == i % k ? std::numeric_limits<std::int16_t>::min()
                       : static_cast<std::int16_t>(rng.next()));
      }
      bound_coords.emplace_back(int(rng.next()), int(rng.next()),
                                int(rng.next()));
    }
    ranks.resize(std::size_t(m) + 1);
    for (std::uint32_t& word : ranks) word = rng.next();
    ranks[m] = *tail;
  }
  const DctBlocks& all = tail ? bound : entries;
  const std::vector<vr::Vec3i>& all_coords = tail ? bound_coords : coords;
  Resident r;
  r.blocks.count =
      static_cast<std::uint32_t>(all.masks.size() / codec::kMaskWordsPerBlock);
  r.blocks.reference_count = m;
  r.blocks.coefficient_count = all.params.coefficient_count;
  if (r.blocks.count != 0) {
    const std::vector<std::uint32_t> packed = pack(all, pad);
    VKC_ASSIGN(r.masks, upload(all.masks.data(),
                               all.masks.size() * sizeof(std::uint32_t)));
    VKC_ASSIGN(r.coefficients,
               upload(packed.data(), packed.size() * sizeof(std::uint32_t)));
    r.blocks.masks = r.masks.get();
    r.blocks.coefficients = r.coefficients.get();
  }
  if (!all_coords.empty()) {
    std::vector<vol::BlockIndex> list(all_coords.size());
    for (std::size_t q = 0; q < all_coords.size(); ++q) {
      list[q].coord = all_coords[q];
    }
    VKC_ASSIGN(r.list,
               upload(list.data(), list.size() * sizeof(vol::BlockIndex)));
    r.blocks.list = r.list.get();
  }
  if (!modes.empty()) {
    std::vector<std::uint32_t> words(modes.size());
    for (std::size_t e = 0; e < modes.size(); ++e) {
      words[e] = static_cast<std::uint32_t>(modes[e]);
    }
    VKC_ASSIGN(r.modes,
               upload(words.data(), words.size() * sizeof(std::uint32_t)));
    r.blocks.modes = r.modes.get();
  }
  if (tail) {
    VKC_ASSIGN(r.ranks,
               upload(ranks.data(), ranks.size() * sizeof(std::uint32_t)));
    r.blocks.ranks = r.ranks.get();
    r.blocks.bounded = true;
  }
  return r;
}

vkc::Status forward(const vr_test::Gpu& gpu, d::DctTransform& t,
                    const vol::VoxelBlockGrid& grid,
                    const std::vector<vol::BlockIndex>& blocks,
                    const codec::CodecParams& params, DctBlocks& out) {
  out = DctBlocks{};
  const std::uint32_t k = params.coefficient_count;
  const std::uint32_t words = d::coefficient_words(k);
  const auto count = static_cast<std::uint32_t>(blocks.size());
  // The list on the device, as the encoder's sort leaves it; none for an
  // empty one, which the transform takes as it is.
  vkc::Buffer list;
  if (count != 0) {
    VKC_ASSIGN(list, vr_test::upload_device_buffer(
                         gpu.device, gpu.allocator, blocks.data(),
                         count * sizeof(vol::BlockIndex)));
  }
  // Where the output goes, as the encoder's reference store holds it; none
  // for params the transform refuses first, or no blocks.
  vkc::Buffer mask_out;
  vkc::Buffer coefficient_out;
  if (params.validate().ok() && count != 0) {
    VKC_ASSIGN(mask_out, vkc::device_storage_buffer(
                             gpu.allocator, VkDeviceSize(count) *
                                                codec::kMaskWordsPerBlock *
                                                sizeof(std::uint32_t)));
    VKC_ASSIGN(coefficient_out, vkc::device_storage_buffer(
                                    gpu.allocator, VkDeviceSize(count) * words *
                                                       sizeof(std::uint32_t)));
  }
  std::uint32_t rejected = 0;
  vkc::CommandBatch batch(gpu.device, gpu.allocator);
  VKC_ASSIGN(const d::ResidentBlocks resident,
             t.record_forward(batch, grid, list, count, params, mask_out,
                              coefficient_out, rejected));
  const std::size_t n = resident.count;
  std::vector<std::uint32_t> packed(n * words);
  std::vector<std::uint32_t> masks(n * codec::kMaskWordsPerBlock);
  if (n != 0) {
    VKC_TRY(batch.readback(*resident.coefficients, 0,
                           packed.size() * sizeof(std::uint32_t),
                           packed.data()));
    VKC_TRY(batch.readback(*resident.masks, 0,
                           masks.size() * sizeof(std::uint32_t), masks.data()));
    VKC_TRY(batch.submit());
    VKC_TRY(
        d::DctTransform::check_rejected("forward", rejected, resident.count));
  }
  out.params = params;
  out.trunc_dist = grid.grid().trunc_dist;
  out.coefficients.resize(n * k);
  for (std::size_t i = 0; i < n; ++i) {
    for (std::uint32_t j = 0; j < k; ++j) {
      out.coefficients[i * k + j] = static_cast<std::int16_t>(
          packed[i * words + j / 2] >> (16 * (j & 1)));
    }
  }
  out.masks = std::move(masks);
  return {};
}

vkc::Status inverse(const vr_test::Gpu& gpu, d::DctTransform& t,
                    vol::VoxelBlockGrid& grid,
                    const std::vector<vol::BlockIndex>& blocks,
                    const DctBlocks& in) {
  const std::size_t n = blocks.size();
  if (in.coefficients.size() != n * in.params.coefficient_count ||
      in.masks.size() != n * codec::kMaskWordsPerBlock) {
    return vkc::Status::invalid_argument(
        "codec_oracle::inverse: the coefficients and masks are not the "
        "list's");
  }
  if (in.trunc_dist != grid.grid().trunc_dist) {
    return vkc::Status::invalid_argument(
        "codec_oracle::inverse: the coefficients were made against another "
        "trunc_dist than the grid's");
  }
  VKC_ASSIGN(Resident r, resident(gpu, in));
  // The list as given, ptrs included, which the inverse must never read.
  if (n != 0) {
    VKC_ASSIGN(vkc::Buffer list, vr_test::upload_device_buffer(
                                     gpu.device, gpu.allocator, blocks.data(),
                                     n * sizeof(vol::BlockIndex)));
    r.list = std::make_unique<vkc::Buffer>(std::move(list));
    r.blocks.list = r.list.get();
  }
  vkc::CommandBatch batch(gpu.device, gpu.allocator);
  return t.inverse(batch, grid, r.blocks, in.params);
}

vkc::Result<std::vector<std::uint8_t>> encode(
    const vr_test::Gpu& gpu, d::DctTransform& t, vol::VoxelBlockGrid& grid,
    const codec::EncoderConfig& config) {
  VKC_ASSIGN(const std::vector<vol::BlockIndex> active,
             grid.map().compact_active_blocks());
  VKC_ASSIGN(const vr_test::AttributeSpan<float> weight,
             vr_test::read_span<float>(gpu.device, gpu.allocator, grid,
                                       "weight", active));
  std::vector<vol::BlockIndex> observed;
  for (const vol::BlockIndex& b : active) {
    const auto first = weight.values.begin() +
                       std::ptrdiff_t(std::size_t(b.ptr) - weight.first);
    if (std::any_of(first, first + codec::kVoxelsPerBlock,
                    [](float w) { return w >= vol::kObservedWeight; })) {
      observed.push_back(b);
    }
  }
  std::sort(observed.begin(), observed.end(),
            [](const vol::BlockIndex& a, const vol::BlockIndex& b) {
              return d::coord_less(a.coord, b.coord);
            });
  Frame frame;
  frame.voxel_size = grid.grid().voxel_size;
  VKC_TRY(forward(gpu, t, grid, observed, config.params, frame.blocks));
  for (const vol::BlockIndex& b : observed) frame.coords.push_back(b.coord);
  FrameWriteOptions options;
  options.segment_size = config.segment_size;
  return write_frame(frame, options);
}

vkc::Result<std::vector<std::uint8_t>> encode(
    const vr_test::Gpu& gpu, vol::VoxelBlockGrid& grid,
    const codec::EncoderConfig& config) {
  VKC_ASSIGN(const std::unique_ptr<d::DctTransform> t,
             d::DctTransform::create(gpu.device, gpu.allocator));
  return encode(gpu, *t, grid, config);
}

}  // namespace codec_oracle
