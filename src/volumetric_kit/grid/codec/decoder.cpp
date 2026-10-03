// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include "volumetric_kit/recon/codec/decoder.hpp"

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

// Rounds of removing, and of allocating, before giving up on a residue that is
// not a capacity limit. Both calls already re-dispatch internally; what can
// survive that is lock contention between blocks hashing to one bucket, and
// another call over the same list clears it (a block already removed, or
// already present, is left alone).
constexpr int kRounds = 4;

Status fail(const std::string& why) {
  return Status::invalid_argument("Decoder::decode: " + why);
}

// Every round's residue lost bucket-lock races over a table with room: not a
// capacity limit, so the answer is to decode again, and a resize would double
// every attribute array for nothing (see volume::AllocFailures).
Status contended(const char* doing, std::uint32_t blocks) {
  return Status::io_error("Decoder::decode: " + std::to_string(blocks) +
                          " blocks lost bucket-lock races in every round of " +
                          doing +
                          ", over a table with room for them; decode again "
                          "(a resize would not help)");
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
  // Those two are all a frame carries, so any other attribute of a block kept
  // across frames would hold the previous frame's values under this one's
  // geometry, with nothing to say which blocks those are.
  if (grid.attribute_count() != 2) {
    return fail("the grid declares " + std::to_string(grid.attribute_count()) +
                " attributes; a frame carries tsdf and weight alone, so a grid "
                "it decodes into must declare only those");
  }
  // Exact, not approximate: the coordinates are in this grid's voxels, and
  // the coefficients are fractions of its band. A frame from a grid of other
  // geometry would decode into the wrong place at the wrong scale and say OK.
  // Read off the header, before the frame is decoded, so such a frame is
  // never mistaken for one the grid is too small for.
  VR_ASSIGN(const detail::FrameHeader header,
            detail::read_frame_header(data, size));
  if (header.voxel_size != gp.voxel_size ||
      header.trunc_dist != gp.trunc_dist) {
    return fail("the frame's geometry (voxel_size " +
                std::to_string(header.voxel_size) + ", trunc_dist " +
                std::to_string(header.trunc_dist) + ") is not this grid's (" +
                std::to_string(gp.voxel_size) + ", " +
                std::to_string(gp.trunc_dist) +
                "); build the grid from read_frame_info");
  }
  detail::IntraFrame frame;
  {
    // The whole frame, entropy-decoded: a corrupt one is refused here, with
    // the grid untouched. The heap bounds the block count the reader will
    // allocate for, so a frame the grid cannot hold is refused here too.
    // TODO(codec): decode on the device, one invocation per segment as the
    // encoder codes, writing the coefficients where the inverse reads them
    // (the 2026-10-03 decision).
    StageScope read(metrics, "  ..rans decode");
    Result<detail::IntraFrame> r = detail::read_intra_frame(
        data, size, static_cast<std::uint32_t>(gp.num_blocks));
    if (!r.ok()) {
      if (r.status().domain() == Status::Code::OutOfMemory) {
        return Status::out_of_memory(
            "Decoder::decode: " + r.status().message() +
            " (the grid's num_blocks); grow it with VoxelBlockGrid::resize "
            "and decode again");
      }
      return r.status();
    }
    frame = std::move(r).value();
  }
  // Nor can the inverse's maxStorageBufferRange checks refuse once the grid
  // starts to change. The frame fits the heap, so every buffer it binds --
  // K <= 512 coefficients and 16 mask words and one list entry per block, 20
  // bytes of hash entry per slot -- is at most the 4-byte tsdf array, which
  // VoxelBlockGrid::create / resize bound to that limit and attribute()
  // confirmed above covers the live grid.

  // --- Make the grid's block set the frame's. -----------------------------
  // A merge of two sorted lists: what the grid holds that the frame lacks is
  // removed (VoxelBlockGrid::remove zeroes it, so a later reuse of its slot
  // starts fresh), and what the frame holds that the grid lacks is allocated.
  // Blocks in both are kept in their slots, and their tsdf/weight are
  // rewritten in full below, so nothing of the previous frame survives in
  // them.
  VR_ASSIGN(std::vector<volume::BlockIndex> current,
            grid.map().compact_active_blocks(metrics));
  {
    StageScope apply(metrics, "  ..apply");
    detail::sort_by_coord(current);
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
    for (int round = 0; !gone.empty(); ++round) {
      volume::AllocFailures failures;
      VR_ASSIGN(
          const std::uint32_t failed,
          grid.remove(gone.data(), static_cast<std::uint32_t>(gone.size()),
                      &failures));
      if (failed == 0) {
        break;
      }
      // Out of the table but refused by the heap: the heap only takes back
      // what it handed out, so this is a grid whose accounting is already
      // broken, and every later frame would find it a block shorter. Not
      // IoError, which says decode again: no retry mends it.
      if (failures.terminal != 0) {
        return fail("the grid's free heap refused " +
                    std::to_string(failures.terminal) +
                    " removed blocks; its block accounting is broken, which "
                    "neither decoding again nor a resize mends");
      }
      if (round + 1 == kRounds) {
        return contended("removing", failed);
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
      if (failures.capacity_limited()) {
        return Status::out_of_memory(
            "Decoder::decode: the grid's hash table cannot place " +
            std::to_string(failed) + " of the frame's " +
            std::to_string(frame.coords.size()) +
            " blocks; grow it with VoxelBlockGrid::resize and decode again");
      }
      if (round + 1 == kRounds) {
        return contended("allocating", failed);
      }
    }
  }

  // --- The frame's blocks, in the frame's order, reconstructed. -----------
  // By coordinate alone: the inverse finds each block in the hash table, so
  // the slots the allocation drew need not be read back, and a frame block
  // the grid does not hold -- a grid changed under this call -- is refused
  // there rather than written.
  std::vector<volume::BlockIndex> blocks(frame.coords.size());
  for (std::size_t b = 0; b < blocks.size(); ++b) {
    blocks[b].coord = frame.coords[b];
  }
  StageScope inverse(metrics, "  ..inverse");
  return transform_->inverse(grid, grid.block_list(blocks), frame.blocks,
                             &stage);
}

}  // namespace volumetric_kit::recon::codec
