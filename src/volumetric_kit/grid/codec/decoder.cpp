// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include "volumetric_kit/recon/codec/decoder.hpp"

#include <string>
#include <utility>

#include "bitstream.hpp"
#include "block_lists.hpp"
#include "dct_transform.hpp"
#include "deblock.hpp"
#include "device_frame_reader.hpp"
#include "frame_hash.hpp"
#include "reference_store.hpp"
#include "volumetric_kit/core/vulkan/allocator.hpp"
#include "volumetric_kit/core/vulkan/command_batch.hpp"
#include "volumetric_kit/core/vulkan/device.hpp"

namespace volumetric_kit::recon::codec {
namespace {

// Rounds of removing, and of allocating, before giving up on a residue that is
// not a capacity limit. Both calls already re-dispatch internally; what can
// survive that is lock contention between blocks hashing to one bucket, and
// another call over the same list clears it (a block already removed, or
// already present, is left alone).
constexpr int kRounds = 4;

core::Status fail(const std::string& why, const char* op = "decode") {
  return core::Status::invalid_argument(std::string("Decoder::") + op + ": " +
                                        why);
}

// The grid checks a decode and a rebuild make before anything is recorded:
// 8^3 blocks, and a 4-byte float tsdf and weight and no other attribute.
core::Status check_grid(const volume::VoxelBlockGrid& grid, const char* op) {
  if (!grid.valid()) {
    return fail("moved-from grid", op);
  }
  const volume::VoxelGridParams& gp = grid.grid();
  if (gp.block_size != kBlockSize) {
    return fail(
        "the codec decodes into 8^3 blocks, and this grid's "
        "block_size is " +
            std::to_string(gp.block_size),
        op);
  }
  for (const char* name : {"tsdf", "weight"}) {
    VKC_ASSIGN(const volume::AttributeView view, grid.attribute(name));
    if (view.element_size != sizeof(float)) {
      return fail(std::string("the grid's ") + name + " is not a 4-byte float",
                  op);
    }
  }
  // Those two are all a frame carries, so any other attribute of a block kept
  // across frames would hold the previous frame's values under this one's
  // geometry, with nothing to say which blocks those are.
  if (grid.attribute_count() != 2) {
    return fail("the grid declares " + std::to_string(grid.attribute_count()) +
                    " attributes; a frame carries tsdf and weight alone, so a "
                    "grid it decodes into must declare only those",
                op);
  }
  return {};
}

// The frame's geometry against the grid's, exactly, not approximately: the
// coordinates are in this grid's voxels, and the coefficients are fractions
// of its band. A frame from a grid of other geometry would decode into the
// wrong place at the wrong scale and say OK.
core::Status check_geometry(const volume::VoxelBlockGrid& grid,
                            const detail::FrameHeader& header, const char* op) {
  const volume::VoxelGridParams& gp = grid.grid();
  if (header.voxel_size != gp.voxel_size ||
      header.trunc_dist != gp.trunc_dist) {
    return fail("the frame's geometry (voxel_size " +
                    std::to_string(header.voxel_size) + ", trunc_dist " +
                    std::to_string(header.trunc_dist) +
                    ") is not this grid's (" + std::to_string(gp.voxel_size) +
                    ", " + std::to_string(gp.trunc_dist) +
                    "); build the grid from read_frame_info",
                op);
  }
  return {};
}

// What the deblock runs on a frame of @p header, and reports: the rule's
// filter at the frame's T, or none when @p on is false.
struct FrameDeblock {
  DeblockInfo info;
  detail::DeblockPasses passes;
};

FrameDeblock frame_deblock(const detail::FrameHeader& header, bool on) {
  FrameDeblock out;
  const float t = detail::step_scale(header.params);
  out.info.step = t * header.trunc_dist;
  if (!on) return out;
  const double t_mm = double(t) * double(header.trunc_dist) * 1e3;
  const detail::DeblockRow* row =
      detail::deblock_rule(header.params.coefficient_count, t_mm);
  if (row == nullptr) return out;
  out.info.filter = row->filter;
  out.passes = detail::deblock_passes(*row, t);
  return out;
}

// The frame's blocks reconstructed from @p blocks into @p grid and deblocked
// under @p passes, recorded after what @p batch holds and submitted with it,
// the inverse's reject count read back last.
core::Status reconstruct(core::CommandBatch& batch,
                         detail::DctTransform& transform,
                         detail::Deblock& deblock, volume::VoxelBlockGrid& grid,
                         const detail::ResidentBlocks& blocks,
                         const CodecParams& params,
                         const detail::DeblockPasses& passes,
                         core::GpuStageScope* stage) {
  VKC_TRY(transform.record_inverse(batch, grid, blocks, params, stage));
  VKC_TRY(deblock.record(batch, grid, blocks, passes, stage));
  std::uint32_t rejected = 0;
  VKC_TRY(transform.record_rejected(batch, rejected));
  VKC_TRY(batch.submit());
  return detail::DctTransform::check_rejected("inverse", rejected,
                                              blocks.count);
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
  VKC_ASSIGN(d.lists_, detail::BlockLists::create(device, allocator));
  VKC_ASSIGN(d.deblock_, detail::Deblock::create(device, allocator));
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

  // --- Everything checkable before the grid is touched. -----------------
  VKC_TRY(check_grid(grid, "decode"));
  // The geometry, read off the header before the frame is decoded, so a
  // frame of another geometry is never mistaken for one the grid is too
  // small for.
  VKC_ASSIGN(const detail::FrameHeader header,
             detail::read_frame_header(data, size));
  VKC_TRY(check_geometry(grid, header, "decode"));
  const volume::VoxelGridParams& gp = grid.grid();
  const std::uint32_t n = header.block_count;
  const FrameDeblock filter = frame_deblock(header, deblock_on_);
  // The frame's payload by stream position, where the device decoded it.
  detail::ResidentBlocks payload;
  {
    // The whole frame, entropy-decoded, a P-frame against the reference
    // held, merged into frame order and diffed against the grid, in one
    // submit: a corrupt frame, or one on another reference, is refused here,
    // with the grid and the reference untouched. The heap bounds the block
    // count the reader will allocate for, so a frame the grid cannot hold is
    // refused here too; a frame the device cannot hold is refused as the
    // reader refuses it.
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
    // Room on the device for its N blocks before anything is recorded: the
    // store's spare, list and sources, the diff's lists, and the deblock's
    // snapshot. Each, like every buffer the inverse and the deblock bind --
    // K <= 512 coefficients, 16 mask words and 296 floats of snapshot a
    // block, 20 bytes of hash entry a slot, 16 bytes of gone list a slot --
    // is at most the 4-byte tsdf array once the frame fits the heap, and
    // VoxelBlockGrid::create / resize bound that array to
    // maxStorageBufferRange (attribute() confirmed it covers the live grid),
    // so no range check refuses a frame the parse passed: only a device out
    // of memory can, which is no refusal of the frame. A P-frame's reader
    // bounds its M + A entries itself: a fresh grid sized from N does not
    // bound M.
    VKC_TRY(store_->reserve(n, header.params.coefficient_count));
    VKC_TRY(lists_->reserve(
        n, static_cast<std::uint32_t>(gp.num_buckets * gp.bucket_size)));
    VKC_TRY(deblock_->reserve(n, filter.passes));
    core::CommandBatch batch(*device_, *allocator_);
    VKC_ASSIGN(payload, reader_->record_decode(batch, parsed.value(), &stage));
    const detail::FrameBuffers frame = store_->frame_buffers();
    VKC_TRY(lists_->record_merge(batch, payload, store_->reference_list(),
                                 frame, n, &stage));
    VKC_TRY(lists_->record_diff(batch, grid, *frame.list, n, &stage));
    VKC_TRY(batch.submit());
    // Rows 11 and 12, then 13, the last refusal, made before the grid
    // changes.
    VKC_TRY(reader_->check());
    const detail::BlockLists::Verdict found = lists_->verdict();
    if (found.collision_q != detail::kNoSource) {
      return detail::check_new_block(found.collision_q, found.collision_e);
    }
  }

  // --- The grid's block set made the frame's. ---------------------------
  {
    core::StageScope apply(metrics, "  ..apply");
    // What the grid holds that the frame lacks is removed
    // (VoxelBlockGrid::remove zeroes it, so a later reuse of its slot starts
    // fresh), and what the frame holds that the grid lacks is allocated,
    // each from the diff's device list. Blocks in both are kept in their
    // slots, and their tsdf/weight are rewritten in full below, so nothing of
    // the previous frame survives in them.
    const detail::BlockLists::Verdict found = lists_->verdict();
    for (int round = 0; found.gone != 0; ++round) {
      volume::AllocFailures failures;
      VKC_ASSIGN(const std::uint32_t failed,
                 grid.remove(lists_->gone(), found.gone, &failures));
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
    for (int round = 0; found.missing != 0; ++round) {
      volume::AllocFailures failures;
      VKC_ASSIGN(
          const std::uint32_t failed,
          grid.map().allocate(lists_->missing(), found.missing, &failures));
      if (failed == 0) {
        break;
      }
      if (failures.capacity_limited()) {
        return core::Status::out_of_memory(
            "Decoder::decode: the grid's hash table cannot place " +
            std::to_string(failed) + " of the frame's " + std::to_string(n) +
            " blocks; grow it with VoxelBlockGrid::resize and decode again");
      }
      if (round + 1 == kRounds) {
        return contended("allocating", failed);
      }
    }
  }

  // --- The frame's blocks, made, reconstructed and deblocked in one
  // submit. Each from the reference and the payload, into the store's
  // spare; then by coordinate alone: the inverse finds each block in the
  // hash table, so the slots the allocation drew need not be read back, and
  // a frame block the grid does not hold -- a grid changed under this call
  // -- is refused there rather than written.
  {
    core::StageScope inverse(metrics, "  ..inverse");
    core::CommandBatch batch(*device_, *allocator_);
    VKC_ASSIGN(const detail::ResidentBlocks blocks,
               store_->record_apply(batch, payload, n, &stage));
    VKC_TRY(reconstruct(batch, *transform_, *deblock_, grid, blocks,
                        header.params, filter.passes, &stage));
    VKC_TRY(store_->check_apply());
  }
  // Only a decode that succeeded moves the reference.
  store_->commit(
      detail::FrameReference{detail::frame_hash(data, size), header});
  last_deblock_ = filter.info;
  return {};
}

core::Status Decoder::rebuild(volume::VoxelBlockGrid& grid,
                              core::StageMetrics* metrics) {
  core::GpuStageScope stage(metrics, gpu_timer_, "codec decode");
  if (!valid()) {
    return fail("moved-from decoder", "rebuild");
  }
  const detail::FrameReference* held = store_->held();
  if (held == nullptr) {
    return core::Status::not_found(
        "Decoder::rebuild: no frame decoded yet; decode one first");
  }
  VKC_TRY(check_grid(grid, "rebuild"));
  const detail::FrameHeader& header = held->header;
  VKC_TRY(check_geometry(grid, header, "rebuild"));
  const volume::VoxelGridParams& gp = grid.grid();
  const std::uint32_t n = header.block_count;
  const FrameDeblock filter = frame_deblock(header, deblock_on_);
  VKC_TRY(lists_->reserve(
      n, static_cast<std::uint32_t>(gp.num_buckets * gp.bucket_size)));
  VKC_TRY(deblock_->reserve(n, filter.passes));
  {
    // The grid's blocks against the frame's, before anything is written: a
    // block the frame lacks would be read as a neighbour.
    core::CommandBatch batch(*device_, *allocator_);
    VKC_TRY(
        lists_->record_diff(batch, grid, store_->reference_list(), n, &stage));
    VKC_TRY(batch.submit());
    const detail::BlockLists::Verdict found = lists_->verdict();
    if (found.gone != 0 || found.missing != 0) {
      return fail("the grid holds " + std::to_string(found.gone) +
                      " blocks the frame lacks and lacks " +
                      std::to_string(found.missing) +
                      " it holds; decode the frame into it instead",
                  "rebuild");
    }
  }
  core::CommandBatch batch(*device_, *allocator_);
  VKC_TRY(reconstruct(batch, *transform_, *deblock_, grid,
                      store_->reference_blocks(), header.params, filter.passes,
                      &stage));
  last_deblock_ = filter.info;
  return {};
}

}  // namespace volumetric_kit::recon::codec
