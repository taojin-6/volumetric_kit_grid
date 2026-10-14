// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include "volumetric_kit/recon/codec/decoder.hpp"

#include <string>
#include <utility>
#include <vector>

#include "bitstream.hpp"
#include "dct_transform.hpp"
#include "device_frame_reader.hpp"
#include "frame_hash.hpp"
#include "reference_store.hpp"
#include "volumetric_kit/core/vulkan/allocator.hpp"
#include "volumetric_kit/core/vulkan/command_batch.hpp"
#include "volumetric_kit/core/vulkan/device.hpp"
#include "volumetric_kit/recon/volume/hash_types.hpp"

namespace volumetric_kit::recon::codec {
namespace {

// Rounds of removing, and of allocating, before giving up on a residue that is
// not a capacity limit. Both calls already re-dispatch internally; what can
// survive that is lock contention between blocks hashing to one bucket, and
// another call over the same list clears it (a block already removed, or
// already present, is left alone).
constexpr int kRounds = 4;

core::Status fail(const std::string& why) {
  return core::Status::invalid_argument("Decoder::decode: " + why);
}

// Every round's residue lost bucket-lock races over a table with room: not a
// capacity limit, so the answer is to decode again, and a resize would double
// every attribute array for nothing (see volume::AllocFailures).
core::Status contended(const char* doing, std::uint32_t blocks) {
  return core::Status::io_error(
      "Decoder::decode: " + std::to_string(blocks) +
      " blocks lost bucket-lock races in every round of " + doing +
      ", over a table with room for them; decode again "
      "(a resize would not help)");
}

}  // namespace

core::Result<FrameInfo> read_frame_info(const std::uint8_t* data,
                                        std::size_t size) {
  VKC_ASSIGN(const detail::FrameHeader header,
             detail::read_frame_header(data, size));
  FrameInfo info;
  info.type = header.type;
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

bool Decoder::valid() const noexcept { return transform_ != nullptr; }

core::Result<Decoder> Decoder::create(core::Device& device,
                                      core::Allocator& allocator) {
  Decoder d;
  d.device_ = &device;
  d.allocator_ = &allocator;
  VKC_ASSIGN(d.transform_, detail::DctTransform::create(device, allocator));
  VKC_ASSIGN(d.reader_, detail::DeviceFrameReader::create(device, allocator));
  VKC_ASSIGN(d.store_, detail::ReferenceStore::create(device, allocator));
  VKC_ASSIGN(d.gpu_timer_, core::GpuTimer::create(device));
  return d;
}

core::Status Decoder::decode(const std::uint8_t* data, std::size_t size,
                             volume::VoxelBlockGrid& grid,
                             core::StageMetrics* metrics) {
  // Before any refusal, so a refused call still costs its row.
  core::GpuStageScope stage(metrics, gpu_timer_, "codec decode");
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
    VKC_ASSIGN(const volume::AttributeView view, grid.attribute(name));
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
  VKC_ASSIGN(const detail::FrameHeader header,
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
  const bool inter = header.type == FrameType::kInter;
  // The frame's new blocks, and its payload by stream position where the
  // device decoded it; a P-frame's modes stay the reader's.
  std::vector<Vec3i> added;
  detail::ResidentBlocks payload;
  {
    // The whole frame, entropy-decoded, a P-frame against the reference
    // held: a corrupt one, or one on another reference, is refused here,
    // with the grid untouched. The heap bounds the block count the reader
    // will allocate for, so a frame the grid cannot hold is refused here too;
    // a frame the device cannot hold is refused as the reader refuses it.
    core::StageScope read(metrics, "  ..rans decode");
    core::Result<detail::ParsedFrame> parsed = detail::parse_frame(
        data, size, static_cast<std::uint32_t>(gp.num_blocks), store_->held());
    if (!parsed.ok()) {
      const core::Status& s = parsed.status();
      if (s.domain() != core::Status::Code::OutOfMemory) return s;
      return core::Status::out_of_memory(
          "Decoder::decode: " + s.message() +
          " (the grid's num_blocks); grow it with VoxelBlockGrid::resize and "
          "decode again");
    }
    core::CommandBatch batch(*device_, *allocator_);
    VKC_ASSIGN(payload, reader_->record_decode(batch, parsed.value(), &stage));
    VKC_TRY(batch.submit());
    VKC_TRY(reader_->check());
    added.reserve(reader_->blocks().size());
    for (const volume::BlockIndex& b : reader_->blocks()) {
      added.push_back(b.coord);
    }
  }

  // --- The frame's blocks, and the grid's block set made the frame's. ----
  VKC_ASSIGN(std::vector<volume::BlockIndex> current,
             grid.map().compact_active_blocks(metrics));
  detail::MergedFrame frame;
  {
    core::StageScope apply(metrics, "  ..apply");
    // The frame's blocks in frame order, and where each comes from: a
    // P-frame's surviving reference blocks merged with its new ones, an
    // intra frame's from its payload alone. The last refusal, a new block on
    // a reference block, is made here, before the grid changes.
    const std::vector<Vec3i> no_reference;
    VKC_ASSIGN(frame,
               detail::merge_frame(inter ? store_->coords() : no_reference,
                                   reader_->modes(), added));
    // Then room on the device for its N blocks, before the grid changes:
    // the store's spare, list and sources. Each, like every buffer the
    // inverse binds -- K <= 512 coefficients and 16 mask words a block, 20
    // bytes of hash entry a slot -- is at most the 4-byte tsdf array once
    // the frame fits the heap, and VoxelBlockGrid::create / resize bound
    // that array to maxStorageBufferRange (attribute() confirmed it covers
    // the live grid), so no range check refuses once the grid starts to
    // change. A P-frame's reader bound its M + A entries itself: a fresh
    // grid sized from N does not bound M.
    VKC_TRY(store_->reserve(static_cast<std::uint32_t>(frame.coords.size()),
                            header.params.coefficient_count));

    // A merge of two sorted lists: what the grid holds that the frame lacks
    // is removed (VoxelBlockGrid::remove zeroes it, so a later reuse of its
    // slot starts fresh), and what the frame holds that the grid lacks is
    // allocated. Blocks in both are kept in their slots, and their
    // tsdf/weight are rewritten in full below, so nothing of the previous
    // frame survives in them. TODO(codec): merge on the device, so neither
    // list crosses the bus (the 2026-10-13 rule).
    detail::sort_by_coord(current);
    std::vector<volume::BlockIndex> gone;
    std::vector<volume::BlockIndex> missing;
    VKC_TRY(detail::merge_by_coord(
        current.size(),
        [&](std::size_t i) -> const Vec3i& { return current[i].coord; },
        frame.coords.size(),
        [&](std::size_t j) -> const Vec3i& { return frame.coords[j]; },
        [&](std::size_t i, std::size_t j) {
          if (j == detail::kAbsent) gone.push_back(current[i]);
          if (i == detail::kAbsent) {
            missing.push_back(volume::BlockIndex{frame.coords[j], 0});
          }
          return core::Status{};
        }));
    for (int round = 0; !gone.empty(); ++round) {
      volume::AllocFailures failures;
      VKC_ASSIGN(
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
      VKC_ASSIGN(const std::uint32_t failed,
                 grid.map().allocate(missing.data(),
                                     static_cast<std::uint32_t>(missing.size()),
                                     &failures));
      if (failed == 0) {
        break;
      }
      if (failures.capacity_limited()) {
        return core::Status::out_of_memory(
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

  // --- The frame's blocks, made and reconstructed in one submit. ---------
  // Each from the reference and the payload, into the store's spare; then
  // by coordinate alone: the inverse finds each block in the hash table, so
  // the slots the allocation drew need not be read back, and a frame block
  // the grid does not hold -- a grid changed under this call -- is refused
  // there rather than written.
  {
    core::StageScope inverse(metrics, "  ..inverse");
    core::CommandBatch batch(*device_, *allocator_);
    VKC_ASSIGN(const detail::ResidentBlocks blocks,
               store_->record_apply(batch, payload, frame.sources, frame.coords,
                                    &stage));
    VKC_TRY(transform_->inverse(batch, grid, blocks, header.params, &stage));
  }
  // Only a decode that succeeded moves the reference.
  store_->commit(detail::FrameReference{detail::frame_hash(data, size), header},
                 std::move(frame.coords));
  return {};
}

}  // namespace volumetric_kit::recon::codec
