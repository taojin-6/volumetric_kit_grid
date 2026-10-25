// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include "codec_oracle_device.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <map>
#include <optional>
#include <utility>

#include "codec_frames.hpp"
#include "codec_oracle_encode.hpp"
#include "grid_readback.hpp"
#include "host_slices.hpp"
#include "volumetric_kit/core/vulkan/command_batch.hpp"
#include "volumetric_kit/core/vulkan/compute_util.hpp"
#include "volumetric_kit/core/vulkan/gpu_timer.hpp"

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
                    const codec::CodecParams& params, DctBlocks& out,
                    const vkc::Buffer* cut) {
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
  // The count as the encoder's observed pass leaves it, a device word.
  VKC_ASSIGN(const vkc::Buffer word,
             vr_test::upload_device_buffer(gpu.device, gpu.allocator, &count,
                                           sizeof(count)));
  std::uint32_t rejected = 0;
  vkc::CommandBatch batch(gpu.device, gpu.allocator);
  if (params.validate().ok()) VKC_TRY(t.record_steps(batch, params));
  VKC_ASSIGN(
      const d::ResidentBlocks resident,
      t.record_forward(batch, grid, list, vol::DeviceCount{&word, 0, count},
                       params, mask_out, coefficient_out, cut));
  const std::size_t n = resident.count;
  std::vector<std::uint32_t> packed(n * words);
  std::vector<std::uint32_t> masks(n * codec::kMaskWordsPerBlock);
  if (n != 0) {
    VKC_TRY(batch.readback(*resident.coefficients, 0,
                           packed.size() * sizeof(std::uint32_t),
                           packed.data()));
    VKC_TRY(batch.readback(*resident.masks, 0,
                           masks.size() * sizeof(std::uint32_t), masks.data()));
    VKC_TRY(batch.readback(t.rejected(), 0, sizeof(rejected), &rejected));
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
  VKC_ASSIGN(const vr_test::HostSlices slices,
             vr_test::host_slices(gpu, r.blocks.count, 1));
  vkc::CommandBatch batch(gpu.device, gpu.allocator);
  VKC_TRY(t.record_steps(batch, in.params));
  VKC_TRY(t.record_inverse(batch, grid, r.blocks, in.params, slices.slices()));
  VKC_TRY(batch.submit());
  VKC_ASSIGN(const std::vector<std::uint32_t> rejected,
             vr_test::read_back<std::uint32_t>(gpu.device, gpu.allocator,
                                               t.rejected(), 1));
  return d::DctTransform::check_rejected("inverse", rejected[0],
                                         r.blocks.count);
}

namespace {

// A cell's corner c (bit 0 x, 1 y, 2 z) of the cell anchored at local
// (x, y, z): the octant of the 2x2x2 neighbourhood holding it, and its local
// index there.
struct Corner {
  int octant;
  std::uint32_t local;
};

Corner corner(int x, int y, int z, int c) {
  const int cx = x + (c & 1), cy = y + ((c >> 1) & 1), cz = z + ((c >> 2) & 1);
  return {(cx >> 3) | ((cy >> 3) << 1) | ((cz >> 3) << 2),
          std::uint32_t((cx & 7) + 8 * (cy & 7) + 64 * (cz & 7))};
}

// The coordinate @p o steps past @p c, wrapping as the device's int addition
// does at int32's extremes.
vr::Vec3i step(const vr::Vec3i& c, int o) {
  const auto wrap = [](std::int32_t v, int d) {
    return static_cast<std::int32_t>(static_cast<std::uint32_t>(v) +
                                     static_cast<std::uint32_t>(d));
  };
  return vr::Vec3i(wrap(c.x, o & 1), wrap(c.y, (o >> 1) & 1),
                   wrap(c.z, (o >> 2) & 1));
}

}  // namespace

vkc::Result<SurfaceSet> surface_set(const vr_test::Gpu& gpu,
                                    vol::VoxelBlockGrid& grid) {
  VKC_ASSIGN(const std::vector<vol::BlockIndex> active,
             grid.map().compact_active_blocks());
  VKC_ASSIGN(
      const std::vector<float> tsdf,
      vr_test::read_attribute<float>(gpu.device, gpu.allocator, grid, "tsdf"));
  VKC_ASSIGN(const std::vector<float> weight,
             vr_test::read_attribute<float>(gpu.device, gpu.allocator, grid,
                                            "weight"));
  const auto less = [](const vr::Vec3i& a, const vr::Vec3i& b) {
    return d::coord_less(a, b);
  };
  std::map<vr::Vec3i, std::int64_t, decltype(less)> slot_of(less);
  for (const vol::BlockIndex& b : active) slot_of[b.coord] = b.ptr;
  // The rule's distances, metres: past sqrt(3) voxels a corner of a crossing
  // cannot lie, and within one voxel a corner keeps its cell.
  const float voxel = grid.grid().voxel_size;
  const float tau = std::fma(std::sqrt(3.0f), voxel, 1e-6f);
  const float near = voxel;
  const std::size_t voxels =
      std::size_t(grid.grid().num_blocks) * codec::kVoxelsPerBlock;
  std::vector<std::uint8_t> cut(voxels, 0);
  std::vector<std::uint8_t> across(voxels, 0);
  std::map<std::int64_t, std::pair<bool, bool>> kept;  // slot: sign, near

  // One walk over every cell anchored in an active block: the corners'
  // voxel slots, or false when a corner is unallocated or unobserved -- or,
  // with @p uncut, cut.
  const auto cells = [&](bool uncut, const auto& visit) {
    for (const vol::BlockIndex& b : active) {
      std::int64_t nb[8];
      for (int o = 0; o < 8; ++o) {
        const auto it = slot_of.find(step(b.coord, o));
        nb[o] = it == slot_of.end() ? -1 : it->second;
      }
      for (int z = 0; z < 8; ++z) {
        for (int y = 0; y < 8; ++y) {
          for (int x = 0; x < 8; ++x) {
            std::size_t at[8];
            int octant[8];
            bool ok = true;
            for (int c = 0; c < 8 && ok; ++c) {
              const Corner k = corner(x, y, z, c);
              ok = nb[k.octant] >= 0;
              if (!ok) break;
              at[c] = std::size_t(nb[k.octant]) + k.local;
              octant[c] = k.octant;
              ok = weight[at[c]] >= vol::kObservedWeight &&
                   !(uncut && cut[at[c]] != 0);
            }
            if (!ok) continue;
            int inside = 0;
            for (int c = 0; c < 8; ++c) inside += tsdf[at[c]] < 0.0f ? 1 : 0;
            visit(at, octant, nb, inside != 0 && inside != 8);
          }
        }
      }
    }
  };
  // The cut, from the field as it stands: no cut changes another cell.
  cells(false, [&](const std::size_t* at, const int* octant,
                   const std::int64_t*, bool flips) {
    bool far = false;
    for (int c = 0; c < 8; ++c) far = far || std::fabs(tsdf[at[c]]) > tau;
    if (!flips || !far) return;
    for (int c = 0; c < 8; ++c) {
      if (std::fabs(tsdf[at[c]]) <= tau) continue;
      cut[at[c]] = 1;
      if (octant[c] != 0) across[at[c]] = 1;
    }
  });
  // Then every block holding a corner of a cell with no corner cut, whose
  // sign changes or that has a corner within one voxel of zero.
  cells(true, [&](const std::size_t* at, const int* octant,
                  const std::int64_t* nb, bool flips) {
    bool close = false;
    for (int c = 0; c < 8; ++c) close = close || std::fabs(tsdf[at[c]]) < near;
    if (!flips && !close) return;
    for (int c = 0; c < 8; ++c) {
      std::pair<bool, bool>& k = kept[nb[octant[c]]];
      k.first = k.first || flips;
      k.second = true;
    }
  });

  SurfaceSet out;
  out.cut.assign(voxels / 32, 0u);
  for (std::size_t v = 0; v < voxels; ++v) {
    if (cut[v] == 0) continue;
    out.cut[v / 32] |= 1u << (v % 32);
    ++out.cut_voxels;
    out.cut_across_faces += across[v];
  }
  for (const vol::BlockIndex& b : active) {
    const auto first = weight.begin() + std::ptrdiff_t(b.ptr);
    if (std::any_of(first, first + codec::kVoxelsPerBlock,
                    [](float w) { return w >= vol::kObservedWeight; })) {
      ++out.observed_blocks;
    }
    const auto k = kept.find(b.ptr);
    if (k == kept.end()) continue;
    out.blocks.push_back(b);
    out.near_only_blocks += k->second.first ? 0 : 1;
  }
  std::sort(out.blocks.begin(), out.blocks.end(),
            [](const vol::BlockIndex& a, const vol::BlockIndex& b) {
              return d::coord_less(a.coord, b.coord);
            });
  return out;
}

vkc::Result<std::vector<std::uint8_t>> write_counted(
    const vr_test::Gpu& gpu, d::DeviceFrameWriter& writer,
    const d::ResidentBlocks& counted, float voxel_size, float trunc_dist,
    const codec::CodecParams& params, const d::FrameReference* reference) {
  vkc::CommandBatch batch(gpu.device, gpu.allocator);
  VKC_TRY(writer.record_code(batch, counted, voxel_size, trunc_dist, params,
                             reference));
  VKC_TRY(batch.submit());
  VKC_ASSIGN(vkc::GpuTimer timer, vkc::GpuTimer::create(gpu.device));
  VKC_ASSIGN(const codec::EncodedFrame frame, writer.finish(nullptr, timer));
  std::vector<std::uint8_t> bytes(frame.head.begin(), frame.head.end());
  bytes.insert(bytes.end(), frame.payload, frame.payload + frame.payload_size);
  return bytes;
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
  std::optional<vkc::Buffer> cut;
  if (config.true_sdf) {
    VKC_ASSIGN(SurfaceSet set, surface_set(gpu, grid));
    observed = std::move(set.blocks);
    VKC_ASSIGN(cut, vr_test::upload_device_buffer(
                        gpu.device, gpu.allocator, set.cut.data(),
                        set.cut.size() * sizeof(std::uint32_t)));
  } else {
    for (const vol::BlockIndex& b : active) {
      const auto first = weight.values.begin() +
                         std::ptrdiff_t(std::size_t(b.ptr) - weight.first);
      if (std::any_of(first, first + codec::kVoxelsPerBlock,
                      [](float w) { return w >= vol::kObservedWeight; })) {
        observed.push_back(b);
      }
    }
  }
  std::sort(observed.begin(), observed.end(),
            [](const vol::BlockIndex& a, const vol::BlockIndex& b) {
              return d::coord_less(a.coord, b.coord);
            });
  Frame frame;
  frame.voxel_size = grid.grid().voxel_size;
  VKC_TRY(forward(gpu, t, grid, observed, config.params, frame.blocks,
                  cut ? &*cut : nullptr));
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
