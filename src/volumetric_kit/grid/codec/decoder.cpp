// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include "volumetric_kit/recon/codec/decoder.hpp"

#include <algorithm>
#include <cstring>
#include <string>
#include <utility>

#include "bitstream.hpp"
#include "block_lists.hpp"
#include "dct_transform.hpp"
#include "deblock.hpp"
#include "device_frame_reader.hpp"
#include "frame_gate.hpp"
#include "frame_hash.hpp"
#include "reference_store.hpp"
#include "volumetric_kit/core/base/check.hpp"
#include "volumetric_kit/core/vulkan/allocator.hpp"
#include "volumetric_kit/core/vulkan/command_batch.hpp"
#include "volumetric_kit/core/vulkan/device.hpp"

namespace volumetric_kit::recon::codec {
namespace detail {

// What finish needs of a recorded frame: the reference it becomes, its new
// blocks, what the deblock ran on it, where its rounds and its results land,
// and the grid it was recorded into.
struct RecordedFrame {
  std::uint64_t serial = 0;
  FrameReference reference;
  std::uint32_t new_count = 0;
  DeblockInfo deblock;
  bool filtered = false;
  core::HostSlot results;
  volume::RecordedRounds removed;
  volume::RecordedRounds allocated;
  const volume::VoxelBlockGrid* grid = nullptr;
};

}  // namespace detail

namespace {

// The zeroed bytes after a record in its input slot: the word holding the
// payload's last byte, which the decode reads whole, stays inside the
// binding.
constexpr std::size_t kSlack = 8;
// The input slots' first size, about a rafa2 I-frame's; a larger record
// grows the slot it claims.
constexpr VkDeviceSize kFirstInputBytes = VkDeviceSize{64} << 10;

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
// filter at the frame's T, or none when @p on is false; finish and rebuild
// add the blocks it kept.
struct FrameDeblock {
  DeblockInfo info;
  detail::DeblockPasses passes;
  bool runs() const noexcept { return passes.linfit || passes.gauss; }
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

// What the deblock did to a frame of @p n blocks: the filter @p info names,
// and the blocks it left as decoded, those holding an unobserved voxel (the
// 2026-10-21 decision's 2026-10-23 amendment), or every block when no filter
// @p ran.
DeblockInfo deblock_ran(const DeblockInfo& info, bool ran, std::uint32_t n,
                        std::uint32_t exact) {
  DeblockInfo out = info;
  out.block_count = n;
  out.exact_blocks = ran ? exact : n;
  return out;
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

detail::DecodeResults read_results(const core::HostSlot& slot) {
  detail::DecodeResults r;
  std::memcpy(&r, slot.data(), sizeof(r));
  return r;
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
  VKC_ASSIGN(d.gate_, detail::FrameGate::create(device, allocator));
  // Two slots each: one a recorded frame holds until its batch has resolved,
  // and one the host fills meanwhile.
  core::HostRingDesc inputs;
  inputs.slot_bytes = kFirstInputBytes;
  inputs.name = "codec.decode_input";
  VKC_ASSIGN(d.inputs_, core::HostRing::create(device, allocator, inputs));
  core::HostRingDesc results;
  results.slot_bytes = sizeof(detail::DecodeResults);
  results.name = "codec.decode_results";
  VKC_ASSIGN(d.results_, core::HostRing::create(device, allocator, results));
  VKC_ASSIGN(d.gpu_timer_, core::GpuTimer::create(device));
  return d;
}

core::Result<core::HostSlot> Decoder::claim_input(std::size_t bytes) {
  if (!valid()) {
    return fail("moved-from decoder", "claim_input");
  }
  if (bytes == 0) {
    return fail("a record of 0 bytes", "claim_input");
  }
  return inputs_->claim(VkDeviceSize(bytes) + kSlack);
}

core::Status Decoder::decode(const std::uint8_t* data, std::size_t size,
                             volume::VoxelBlockGrid& grid,
                             core::StageMetrics* metrics) {
  if (!valid()) {
    core::GpuStageScope stage(metrics, gpu_timer_, "codec decode");
    return fail("moved-from decoder");
  }
  // No record is the header's refusal, as an empty frame always was.
  if (data == nullptr || size == 0) {
    core::GpuStageScope stage(metrics, gpu_timer_, "codec decode");
    return detail::read_frame_header(data, size).status();
  }
  core::Result<core::HostSlot> slot = claim_input(size);
  if (!slot.ok()) {
    core::GpuStageScope stage(metrics, gpu_timer_, "codec decode");
    return slot.status();
  }
  std::memcpy(slot->data(), data, size);
  return decode(*slot, size, grid, metrics);
}

core::Status Decoder::decode(const core::HostSlot& slot, std::size_t size,
                             volume::VoxelBlockGrid& grid,
                             core::StageMetrics* metrics) {
  // Before any refusal, so a refused call still costs its row; its wait,
  // and the device halves the batch hands back, land in it too.
  core::GpuStageScope stage(metrics, gpu_timer_, "codec decode");
  if (!valid()) {
    return fail("moved-from decoder");
  }
  core::CommandBatch batch(*device_, *allocator_);
  VKC_ASSIGN(const PendingDecode pending,
             record_frame(batch, slot, size, grid, metrics));
  const core::Status ran = [&]() -> core::Status {
    VKC_ASSIGN(core::PendingBatch done, batch.submit_async({}, {}));
    VKC_TRY(done.wait());
    if (metrics != nullptr) metrics->merge_device(done.device_rows());
    return {};
  }();
  if (!ran.ok()) {
    abandon(pending);
    return ran;
  }
  return finish(pending, grid);
}

core::Result<PendingDecode> Decoder::record(core::CommandBatch& batch,
                                            const core::HostSlot& slot,
                                            std::size_t size,
                                            volume::VoxelBlockGrid& grid,
                                            core::StageMetrics* metrics) {
  core::GpuStageScope stage(metrics, gpu_timer_, "codec decode");
  return record_frame(batch, slot, size, grid, metrics);
}

core::Result<PendingDecode> Decoder::record_frame(core::CommandBatch& batch,
                                                  const core::HostSlot& slot,
                                                  std::size_t size,
                                                  volume::VoxelBlockGrid& grid,
                                                  core::StageMetrics* metrics) {
  if (!valid()) {
    return fail("moved-from decoder", "record");
  }

  // --- Everything checkable before anything is recorded. ----------------
  VKC_TRY(check_grid(grid, "decode"));
  if (!slot.valid() || size > slot.capacity() ||
      slot.capacity() - size < kSlack) {
    return fail("the slot holds " + std::to_string(slot.capacity()) +
                    " bytes, fewer than the record's " + std::to_string(size) +
                    " and " + std::to_string(kSlack) + " of slack",
                "record");
  }
  std::uint8_t* const data = slot.data();
  // The geometry, read off the header before the frame is parsed, so a
  // frame of another geometry is never mistaken for one the grid is too
  // small for.
  VKC_ASSIGN(const detail::FrameHeader header,
             detail::read_frame_header(data, size));
  VKC_TRY(check_geometry(grid, header, "decode"));
  const volume::VoxelGridParams& gp = grid.grid();
  const std::uint32_t n = header.block_count;
  const FrameDeblock filter = frame_deblock(header, deblock_on_);
  // The whole frame, parsed in place, a P-frame against the reference held:
  // a corrupt frame, or one on another reference, is refused here, with the
  // grid and the reference untouched. The heap bounds the block count the
  // reader will allocate for, so a frame the grid cannot hold is refused here
  // too; a frame the device cannot hold is refused as the reader refuses it.
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
  // flags and snapshot. Each, like every buffer the inverse and the
  // deblock bind -- K <= 512 coefficients, 16 mask words and 296 floats of
  // snapshot a block, 20 bytes of hash entry a slot, 16 bytes of gone list
  // and 4 of flag a slot --
  // is at most the 4-byte tsdf array once the frame fits the heap, and
  // VoxelBlockGrid::create / resize bound that array to
  // maxStorageBufferRange (attribute() confirmed it covers the live grid),
  // so no range check refuses a frame the parse passed: only a device out
  // of memory can, which is no refusal of the frame. A P-frame's reader
  // bounds its M + A entries itself: a fresh grid sized from N does not
  // bound M.
  const auto slots =
      static_cast<std::uint32_t>(gp.num_buckets * gp.bucket_size);
  VKC_TRY(store_->reserve(n, header.params.coefficient_count));
  VKC_TRY(lists_->reserve(n, slots));
  VKC_TRY(deblock_->reserve(n, static_cast<std::uint32_t>(gp.num_blocks),
                            filter.passes));
  // A record left unfinished is superseded. Its batch has resolved (the
  // @pre), so a results slot is free.
  pending_.reset();
  core::Result<core::HostSlot> results =
      results_->claim(sizeof(detail::DecodeResults));
  VKC_CHECK(results.ok(),
            "Decoder::record: the last record's batch has resolved, so a "
            "results slot is free");
  auto recorded = std::make_unique<detail::RecordedFrame>();
  recorded->reference =
      detail::FrameReference{detail::frame_hash(data, size), header};
  recorded->new_count = parsed->new_count;
  recorded->deblock = filter.info;
  recorded->filtered = filter.runs();
  recorded->results = std::move(results).value();
  recorded->grid = &grid;
  // Stands in for the zero pad a staged payload had.
  std::memset(data + size, 0, kSlack);

  // --- The whole decode, into the caller's batch. Every update first: the
  // steps here, the tables and segment words in the reader's recording,
  // ahead of its first dispatch, so MoltenVK opens no blit encoder mid-batch.
  VKC_ASSIGN(const VkBuffer input, batch.input(slot, 0, size + kSlack));
  VKC_TRY(transform_->record_steps(batch, header.params));
  const detail::ResultSources words{
      &reader_->verdict(), &lists_->collision(),    &lists_->diff_counts(),
      &store_->faults(),   &transform_->rejected(), &deblock_->exact()};
  detail::ResidentBlocks payload;
  {
    // The frame entropy-decoded, a P-frame's survivors merged with its new
    // blocks into frame order, and the grid's blocks diffed against it.
    core::GpuStageScope read(metrics, gpu_timer_, "  ..rans decode");
    VKC_ASSIGN(payload, reader_->record_decode(
                            batch, parsed.value(), input, size + kSlack,
                            VkDeviceSize(parsed->payload - data), &read));
    const detail::FrameBuffers frame = store_->frame_buffers();
    VKC_TRY(lists_->record_merge(batch, payload, store_->reference_list(),
                                 frame, n, &read));
    VKC_TRY(lists_->record_diff(batch, grid, *frame.list, n, &read));
  }
  const volume::DeviceCount gate{&gate_->gate(), 0, 1};
  {
    // Rows 11 to 13 judged on the device, then the grid's block set made the
    // frame's behind that gate: what the grid holds that the frame lacks is
    // removed (record_remove zeroes it, so a later reuse of its slot starts
    // fresh), and what the frame holds that the grid lacks is allocated, each
    // from the diff's list and count. Blocks in both are kept in their slots,
    // and their tsdf/weight are rewritten in full below, so nothing of the
    // previous frame survives in them. Each kind's rounds close the gate on
    // their own failure.
    core::GpuStageScope apply(metrics, gpu_timer_, "  ..apply");
    VKC_TRY(gate_->record_gate(
        batch, words,
        detail::GateCheck{false, header.type == FrameType::kInter, n,
                          parsed->new_count},
        &apply));
    VKC_ASSIGN(recorded->removed,
               grid.record_remove(
                   batch, lists_->gone(),
                   volume::DeviceCount{&lists_->diff_counts(), 0, slots}, &gate,
                   &apply));
    VKC_ASSIGN(recorded->allocated,
               grid.map().record_allocate(
                   batch, lists_->missing(),
                   volume::DeviceCount{&lists_->diff_counts(),
                                       sizeof(std::uint32_t), n},
                   &gate, &apply));
  }
  {
    // The frame's blocks, made, reconstructed and deblocked behind the gate:
    // each from the reference and the payload into the store's spare, then
    // by coordinate alone -- the inverse finds each block in the hash table,
    // so the slots the allocation drew need not be read back, and a frame
    // block the grid does not hold, a grid changed under this call, is
    // counted there rather than written. Then every count into the results
    // slot.
    core::GpuStageScope inverse(metrics, gpu_timer_, "  ..inverse");
    VKC_ASSIGN(const detail::FrameSlices slices,
               gate_->record_slices(batch, n, &inverse));
    VKC_ASSIGN(
        const detail::ResidentBlocks blocks,
        store_->record_apply(batch, payload, n, slices.blocks, &inverse));
    VKC_TRY(transform_->record_inverse(batch, grid, blocks, header.params,
                                       slices.blocks, &inverse));
    VKC_TRY(deblock_->record(batch, grid, blocks, filter.passes, slices.blocks,
                             slices.flags, &inverse));
    VKC_TRY(gate_->record_results(batch, recorded->results, words, &inverse));
  }
  recorded->serial = ++serial_;
  pending_ = std::move(recorded);
  return PendingDecode{pending_->serial, n, gate, slot, size};
}

core::Status Decoder::finish(const PendingDecode& pending,
                             volume::VoxelBlockGrid& grid) {
  if (!valid()) {
    return fail("moved-from decoder", "finish");
  }
  if (pending_ == nullptr || pending.serial == 0 ||
      pending.serial != pending_->serial) {
    return fail(
        "the pending frame is not this decoder's last record (finished, "
        "abandoned or superseded)",
        "finish");
  }
  if (&grid != pending_->grid) {
    return fail("the grid is not the one the frame was recorded into",
                "finish");
  }
  const std::unique_ptr<detail::RecordedFrame> frame = std::move(pending_);
  VKC_ASSIGN(const volume::AllocFailures removed,
             grid.map().settle(frame->removed));
  VKC_ASSIGN(const volume::AllocFailures allocated,
             grid.map().settle(frame->allocated));
  last_rounds_ = std::max(removed.rounds, allocated.rounds);
  const detail::DecodeResults r = read_results(frame->results);
  const detail::FrameHeader& header = frame->reference.header;
  const std::uint32_t n = header.block_count;
  // Rows 11 and 12, then 13, the last refusal, which the gate made before
  // the grid changed.
  VKC_TRY(detail::DeviceFrameReader::judge(header.type, n, frame->new_count,
                                           r.failed, r.survivors));
  if (r.collision_q != detail::kNoSource) {
    return detail::check_new_block(r.collision_q, r.collision_e);
  }
  // Out of the table but refused by the heap: the heap only takes back what
  // it handed out, so this is a grid whose accounting is already broken, and
  // every later frame would find it a block shorter. Not IoError, which says
  // decode again: no retry mends it.
  if (removed.terminal != 0) {
    return fail("the grid's free heap refused " +
                std::to_string(removed.terminal) +
                " removed blocks; its block accounting is broken, which "
                "neither decoding again nor a resize mends");
  }
  if (removed.total != 0) {
    return contended("removing", removed.total);
  }
  if (allocated.capacity_limited()) {
    return core::Status::out_of_memory(
        "Decoder::decode: the grid's hash table cannot place " +
        std::to_string(allocated.total) + " of the frame's " +
        std::to_string(n) +
        " blocks; grow it with VoxelBlockGrid::resize and decode again");
  }
  if (allocated.total != 0) {
    return contended("allocating", allocated.total);
  }
  VKC_TRY(detail::ReferenceStore::check_faults(r.faults));
  VKC_TRY(detail::DctTransform::check_rejected("inverse", r.rejected, n));
  // Only a frame that succeeded moves the reference.
  store_->commit(frame->reference);
  last_deblock_ = deblock_ran(frame->deblock, frame->filtered, n, r.exact);
  return {};
}

void Decoder::abandon(const PendingDecode& pending) noexcept {
  if (pending_ != nullptr && pending.serial != 0 &&
      pending.serial == pending_->serial) {
    pending_.reset();
  }
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
  VKC_TRY(deblock_->reserve(n, static_cast<std::uint32_t>(gp.num_blocks),
                            filter.passes));
  // A slot of its own: a frame recorded and unfinished may hold the other.
  core::Result<core::HostSlot> results =
      results_->claim(sizeof(detail::DecodeResults));
  VKC_CHECK(results.ok(),
            "Decoder::rebuild: no batch of this decoder is in flight, so a "
            "results slot is free");
  // One batch: the grid's blocks against the frame's -- a block the frame
  // lacks would be read as a neighbour -- then the inverse and the deblock
  // behind that gate, so nothing is written to a grid that fails it, and the
  // results.
  const detail::ResultSources words{
      &reader_->verdict(), &lists_->collision(),    &lists_->diff_counts(),
      &store_->faults(),   &transform_->rejected(), &deblock_->exact()};
  const detail::ResidentBlocks blocks = store_->reference_blocks();
  core::CommandBatch batch(*device_, *allocator_);
  VKC_TRY(transform_->record_steps(batch, header.params));
  VKC_TRY(
      lists_->record_diff(batch, grid, store_->reference_list(), n, &stage));
  VKC_TRY(gate_->record_gate(batch, words, detail::GateCheck{true, false, n, 0},
                             &stage));
  VKC_ASSIGN(const detail::FrameSlices slices,
             gate_->record_slices(batch, n, &stage));
  VKC_TRY(transform_->record_inverse(batch, grid, blocks, header.params,
                                     slices.blocks, &stage));
  VKC_TRY(deblock_->record(batch, grid, blocks, filter.passes, slices.blocks,
                           slices.flags, &stage));
  VKC_TRY(gate_->record_results(batch, *results, words, &stage));
  VKC_TRY(batch.submit());
  const detail::DecodeResults r = read_results(*results);
  if (r.gate == 0) {
    return fail("the grid holds " + std::to_string(r.gone) +
                    " blocks the frame lacks and lacks " +
                    std::to_string(r.missing) +
                    " it holds; decode the frame into it instead",
                "rebuild");
  }
  VKC_TRY(detail::DctTransform::check_rejected("inverse", r.rejected, n));
  last_deblock_ = deblock_ran(filter.info, filter.runs(), n, r.exact);
  return {};
}

}  // namespace volumetric_kit::recon::codec
