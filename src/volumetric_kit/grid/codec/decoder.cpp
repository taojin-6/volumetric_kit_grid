// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include "volumetric_kit/recon/codec/decoder.hpp"

#include <algorithm>
#include <string>
#include <utility>
#include <vector>

#include "bitstream.hpp"
#include "dct_transform.hpp"
#include "volumetric_kit/recon/core/allocator.hpp"
#include "volumetric_kit/recon/core/device.hpp"
#include "volumetric_kit/recon/volume/hash_types.hpp"

namespace volumetric_kit::recon::codec {
namespace {

// Allocation rounds before giving up on a residue that is not a capacity
// limit. VoxelHashMap::allocate already re-dispatches internally; what can
// survive that is lock contention between blocks hashing to one bucket, and a
// second call over the same list (present blocks are left alone) clears it.
constexpr int kAllocRounds = 4;

Status fail(const std::string& why) {
  return Status::invalid_argument("Decoder::decode: " + why);
}

void sort_by_coord(std::vector<volume::BlockIndex>& blocks) {
  std::sort(blocks.begin(), blocks.end(),
            [](const volume::BlockIndex& a, const volume::BlockIndex& b) {
              return detail::coord_less(a.coord, b.coord);
            });
}

}  // namespace

Result<FrameInfo> read_frame_info(const std::uint8_t* data, std::size_t size) {
  VR_ASSIGN(const detail::FrameHeader header,
            detail::read_frame_header(data, size));
  FrameInfo info;
  info.voxel_size = header.voxel_size;
  info.trunc_dist = header.trunc_dist;
  info.block_count = header.block_count;
  info.params = header.params;
  return info;
}

Decoder::Decoder() = default;
Decoder::~Decoder() = default;
Decoder::Decoder(Decoder&& other) noexcept = default;
Decoder& Decoder::operator=(Decoder&& other) noexcept = default;

bool Decoder::valid() const noexcept {
  return transform_ != nullptr && transform_->valid();
}

Result<Decoder> Decoder::create(Device& device, Allocator& allocator) {
  Decoder d;
  VR_ASSIGN(detail::DctTransform transform,
            detail::DctTransform::create(device, allocator));
  d.transform_ = std::make_unique<detail::DctTransform>(std::move(transform));
  VR_ASSIGN(d.gpu_timer_, GpuTimer::create(device));
  return d;
}

Status Decoder::decode(const std::uint8_t* data, std::size_t size,
                       volume::VoxelBlockGrid& grid, StageMetrics* metrics) {
  // Before any refusal, so a refused call still costs its row.
  GpuStageScope stage(metrics, gpu_timer_, "codec decode");
  if (!valid()) {
    return fail("moved-from decoder");
  }
  if (!grid.valid()) {
    return fail("moved-from grid");
  }

  // --- Everything checkable before the grid is touched. -----------------
  const volume::VoxelGridParams& gp = grid.grid();
  if (gp.block_size != kBlockSize) {
    return fail(
        "the codec decodes into 8^3 blocks, and this grid's "
        "block_size is " +
        std::to_string(gp.block_size));
  }
  for (const char* name : {"tsdf", "weight"}) {
    VR_ASSIGN(const volume::AttributeView view, grid.attribute(name));
    if (view.element_size != sizeof(float)) {
      return fail(std::string("the grid's ") + name + " is not a 4-byte float");
    }
  }
  detail::IntraFrame frame;
  {
    // The whole frame, entropy-decoded: a corrupt one is refused here, with
    // the grid untouched. The heap bounds the block count the reader will
    // allocate for, so a frame the grid cannot hold is refused here too.
    StageScope entropy(metrics, "  ..entropy");
    VR_ASSIGN(frame,
              detail::read_intra_frame(
                  data, size, static_cast<std::uint32_t>(gp.num_blocks)));
  }
  // Exact, not approximate: the coordinates are in this grid's voxels, and
  // the coefficients are fractions of its band. A frame from a grid of other
  // geometry would decode into the wrong place at the wrong scale and say OK.
  if (frame.voxel_size != gp.voxel_size ||
      frame.blocks.trunc_dist != gp.trunc_dist) {
    return fail("the frame's geometry (voxel_size " +
                std::to_string(frame.voxel_size) + ", trunc_dist " +
                std::to_string(frame.blocks.trunc_dist) +
                ") is not this grid's (" + std::to_string(gp.voxel_size) +
                ", " + std::to_string(gp.trunc_dist) +
                "); build the grid from read_frame_info");
  }

  // --- Make the grid's block set the frame's. -----------------------------
  // A merge of two sorted lists: what the grid holds that the frame lacks is
  // removed (VoxelBlockGrid::remove zeroes it, so a later reuse of its slot
  // starts fresh), and what the frame holds that the grid lacks is allocated.
  // Blocks in both are kept, and their tsdf/weight are rewritten in full
  // below, so nothing of the previous frame survives in them.
  VR_ASSIGN(std::vector<volume::BlockIndex> current,
            grid.map().compact_active_blocks(metrics));
  {
    StageScope apply(metrics, "  ..apply");
    sort_by_coord(current);
    std::vector<volume::BlockIndex> gone;
    std::vector<volume::BlockIndex> missing;
    std::size_t i = 0;
    std::size_t j = 0;
    while (i < current.size() || j < frame.coords.size()) {
      if (j == frame.coords.size() ||
          (i < current.size() &&
           detail::coord_less(current[i].coord, frame.coords[j]))) {
        gone.push_back(current[i++]);
      } else if (i == current.size() ||
                 detail::coord_less(frame.coords[j], current[i].coord)) {
        volume::BlockIndex b{};
        b.coord = frame.coords[j++];
        missing.push_back(b);
      } else {
        ++i;
        ++j;
      }
    }
    if (!gone.empty()) {
      // A non-zero count is capacity the heap leaked, not a block left
      // behind: the blocks are out of the table either way.
      Result<std::uint32_t> removed =
          grid.remove(gone.data(), static_cast<std::uint32_t>(gone.size()));
      if (!removed.ok()) {
        return removed.status();
      }
    }
    for (int round = 0; !missing.empty(); ++round) {
      volume::AllocFailures failures;
      VR_ASSIGN(const std::uint32_t failed,
                grid.map().allocate(missing.data(),
                                    static_cast<std::uint32_t>(missing.size()),
                                    &failures));
      if (failed == 0) {
        break;
      }
      if (failures.capacity_limited() || round + 1 == kAllocRounds) {
        return Status::out_of_memory(
            "Decoder::decode: the grid's hash table cannot place " +
            std::to_string(failed) + " of the frame's " +
            std::to_string(frame.coords.size()) +
            " blocks; grow it with VoxelBlockGrid::resize and decode again "
            "(the grid now holds part of the frame)");
      }
    }
  }

  // --- The frame's blocks, in the frame's order, reconstructed. -----------
  VR_ASSIGN(std::vector<volume::BlockIndex> blocks,
            grid.map().compact_active_blocks(metrics));
  {
    StageScope apply(metrics, "  ..apply");
    sort_by_coord(blocks);
    // Both are strictly increasing, so equal sizes and equal coordinates mean
    // the grid holds exactly the frame. Anything else is a grid changed under
    // this call.
    const bool exact =
        blocks.size() == frame.coords.size() &&
        std::equal(blocks.begin(), blocks.end(), frame.coords.begin(),
                   [](const volume::BlockIndex& b, const Vec3i& c) {
                     return b.coord == c;
                   });
    if (!exact) {
      return fail(
          "the grid does not hold exactly the frame's blocks after "
          "applying it (was it modified during the call?)");
    }
  }
  StageScope transform(metrics, "  ..transform");
  return transform_->inverse(grid, grid.block_list(blocks), frame.blocks,
                             &stage);
}

}  // namespace volumetric_kit::recon::codec
