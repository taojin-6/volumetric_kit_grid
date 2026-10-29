// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include "volumetric_kit/grid/codec/encoder.hpp"

#include <atomic>
#include <new>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "bitstream.hpp"
#include "block_lists.hpp"
#include "dct_transform.hpp"
#include "device_frame_writer.hpp"
#include "frame_hash.hpp"
#include "reference_store.hpp"
#include "volumetric_kit/core/base/exception_status.hpp"
#include "volumetric_kit/core/vulkan/allocator.hpp"
#include "volumetric_kit/core/vulkan/command_batch.hpp"
#include "volumetric_kit/core/vulkan/compute_util.hpp"
#include "volumetric_kit/core/vulkan/device.hpp"

namespace volumetric_kit::grid::codec {
namespace detail {

// What the halves need of a recorded frame: the reference a P-frame is
// coded against, the grid's geometry, the entries the front counted, the
// blocks it found, and how far the frame has come.
struct RecordedEncode {
  std::uint64_t serial = 0;
  FrameType type = FrameType::kIntra;
  FrameReference reference;
  float voxel_size = 0.0f;
  float trunc_dist = 0.0f;
  ResidentBlocks entries;
  std::uint32_t blocks = 0;
  bool back = false;
};

}  // namespace detail

namespace {

// The public default restates the frame's, which the installed header cannot
// name; this is what keeps the pair one number.
static_assert(EncoderConfig{}.segment_size == detail::kDefaultSegmentSize,
              "EncoderConfig's default segment size is the frame's");

// A record's serial, unique in the process, so a token names one encoder's
// record and no other's.
std::uint64_t next_serial() noexcept {
  static std::atomic<std::uint64_t> counter{0};
  return counter.fetch_add(1, std::memory_order_relaxed) + 1;
}

core::Status fail(const char* op, const std::string& why) {
  return core::Status::invalid_argument(std::string("Encoder::") + op + ": " +
                                        why);
}

}  // namespace

core::Status EncoderConfig::validate() const {
  VKC_TRY(params.validate());
  VKC_TRY(detail::check_segment_size("EncoderConfig", "segment_size",
                                     segment_size));
  VKC_TRY(detail::check_segment_size("EncoderConfig", "update_size",
                                     resolved_update_size()));
  if (intra_period == 0) {
    return core::Status::invalid_argument(
        "EncoderConfig: intra_period 0; 1 makes every frame an I-frame");
  }
  return {};
}

Encoder::Encoder() = default;
Encoder::~Encoder() = default;

Encoder::Encoder(Encoder&& other) noexcept = default;
Encoder& Encoder::operator=(Encoder&& other) noexcept = default;

bool Encoder::valid() const noexcept { return transform_ != nullptr; }

void Encoder::force_intra() noexcept { force_intra_ = true; }

core::Result<Encoder> Encoder::create(core::Device& device,
                                      core::Allocator& allocator,
                                      const EncoderConfig& config) {
  VKC_TRY(config.validate());
  Encoder e;
  e.config_ = config;
  e.config_.update_size = config.resolved_update_size();
  e.device_ = &device;
  e.allocator_ = &allocator;
  VKC_ASSIGN(e.transform_, detail::DctTransform::create(device, allocator));
  VKC_ASSIGN(e.lists_, detail::BlockLists::create(device, allocator));
  VKC_ASSIGN(e.writer_, detail::DeviceFrameWriter::create(device, allocator));
  VKC_ASSIGN(e.store_, detail::ReferenceStore::create(device, allocator));
  VKC_ASSIGN(e.list_count_,
             core::device_storage_buffer(allocator, sizeof(std::uint32_t)));
  device.set_object_name(VK_OBJECT_TYPE_BUFFER,
                         core::debug_object_handle(e.list_count_.handle()),
                         "codec.encode_list_count");
  VKC_ASSIGN(e.gpu_timer_, core::GpuTimer::create(device));
  return e;
}

core::Result<std::vector<std::uint8_t>> Encoder::encode(
    const volume::VoxelBlockGrid& grid, core::StageMetrics* metrics) try {
  // Before any refusal, so a refused call still costs its row; both
  // batches' waits, and the device halves they hand back, land in it too.
  // Its row is placed now, ahead of its breakdowns', which the batches'
  // device rows add while it is open.
  core::GpuStageScope stage(metrics, gpu_timer_, "codec encode");
  if (metrics != nullptr) metrics->add_cpu("codec encode", 0.0);
  // One half's batch, submitted and waited on, its device rows kept.
  const auto run = [&](core::CommandBatch& batch) -> core::Status {
    VKC_ASSIGN(core::PendingBatch done, batch.submit_async({}, {}));
    VKC_TRY(done.wait());
    if (metrics != nullptr) metrics->merge_device(done.device_rows());
    return {};
  };
  core::CommandBatch front_batch(*device_, *allocator_);
  VKC_ASSIGN(const PendingEncode pending, front(front_batch, grid, metrics));
  core::Status ran = run(front_batch);
  core::CommandBatch back_batch(*device_, *allocator_);
  if (ran.ok()) ran = back(back_batch, pending, metrics);
  if (ran.ok()) ran = run(back_batch);
  if (!ran.ok()) {
    abandon(pending);
    return ran;
  }
  std::vector<std::uint8_t> bytes;
  const auto frame = finish_frame(pending, metrics, &bytes);
  if (!frame.ok()) return frame.status();
  return bytes;
} catch (const std::bad_alloc&) {
  return core::exception_status("Encoder::encode",
                                core::Status::Code::OutOfMemory);
}

core::Result<PendingEncode> Encoder::record_front(
    core::CommandBatch& batch, const volume::VoxelBlockGrid& grid,
    core::StageMetrics* metrics) try {
  // As encode(): the stage's row placed ahead of its breakdowns'.
  core::GpuStageScope stage(metrics, gpu_timer_, "codec encode");
  if (metrics != nullptr) metrics->add_cpu("codec encode", 0.0);
  return front(batch, grid, metrics);
} catch (const std::bad_alloc&) {
  return core::exception_status("Encoder::record_front",
                                core::Status::Code::OutOfMemory);
}

core::Status Encoder::record_back(core::CommandBatch& batch,
                                  const PendingEncode& pending,
                                  core::StageMetrics* metrics) try {
  core::GpuStageScope stage(metrics, gpu_timer_, "codec encode");
  return back(batch, pending, metrics);
} catch (const std::bad_alloc&) {
  return core::exception_status("Encoder::record_back",
                                core::Status::Code::OutOfMemory);
}

core::Result<EncodedFrame> Encoder::finish(const PendingEncode& pending,
                                           core::StageMetrics* metrics) try {
  core::GpuStageScope stage(metrics, gpu_timer_, "codec encode");
  return finish_frame(pending, metrics);
} catch (const std::bad_alloc&) {
  return core::exception_status("Encoder::finish",
                                core::Status::Code::OutOfMemory);
}

void Encoder::abandon(const PendingEncode& pending) noexcept {
  if (!holds(pending)) return;
  pending_.reset();
  writer_->drop();
}

bool Encoder::holds(const PendingEncode& pending) const noexcept {
  return pending_ != nullptr && pending.serial != 0 &&
         pending.serial == pending_->serial;
}

core::Result<PendingEncode> Encoder::front(core::CommandBatch& batch,
                                           const volume::VoxelBlockGrid& grid,
                                           core::StageMetrics* metrics) {
  if (!valid()) return fail("encode", "moved-from encoder");
  if (!grid.valid()) return fail("encode", "moved-from grid");
  const volume::VoxelGridParams& gp = grid.grid();
  const CodecParams& params = config_.params;
  // Everything refusable before anything is recorded: the grid as the
  // transform takes it, and the geometry the header carries.
  VKC_TRY(transform_->check_grid(grid));
  VKC_TRY(detail::check_coding(gp.voxel_size, gp.trunc_dist, params));

  // Every frame is held against the frame returned last, intra or P, when
  // the two can be compared: the encoder's params never change, so of the
  // coding parameters only the grid's geometry can differ, and an empty
  // reference holds nothing. A P-frame codes against it too, unless the
  // rules on encode() make this an I-frame. Whether the grid has a block to
  // code is the device's to say: a frame of none is an I-frame of none
  // (back), which costs no remove for every reference entry.
  const detail::FrameReference* held = store_->held();
  if (held != nullptr &&
      (held->header.block_count == 0 ||
       detail::coding_difference(held->header, gp.voxel_size, gp.trunc_dist,
                                 params) != nullptr)) {
    held = nullptr;
  }
  const detail::FrameReference* reference =
      force_intra_ || gop_position_ == 0 ? nullptr : held;
  // Made before an unfinished record is superseded, so a host allocation
  // failure leaves that token finishable.
  auto recorded = std::make_unique<detail::RecordedEncode>();
  recorded->type = reference != nullptr ? FrameType::kInter : FrameType::kIntra;
  if (reference != nullptr) recorded->reference = *reference;
  recorded->voxel_size = gp.voxel_size;
  recorded->trunc_dist = gp.trunc_dist;
  pending_.reset();
  writer_->drop();

  // Every step is sized for the grid's block slots, the most the active
  // set, and so the frame, can hold; the device finds the counts.
  const auto bound = static_cast<std::uint32_t>(gp.num_blocks);
  const std::uint32_t k = params.coefficient_count;
  VKC_TRY(store_->reserve(bound, k));
  // The steps the forward reads, inline, ahead of this half's dispatches --
  // not the batch's first in a sender's batch, whose prep and fusion come
  // before -- until a frame has finished with them: they never change, so a
  // warm encoder records no update here.
  if (!steps_held_) VKC_TRY(transform_->record_steps(batch, params));

  // Each breakdown's host half is the stage's own, so they record into a
  // scratch row set and keep their device halves alone, which the batch's
  // device rows hand back by name. Each one's row is placed in the caller's
  // rows as it opens, its host half 0, so a caller sharing the batch tells
  // the encode's device rows by the rows it holds.
  core::StageMetrics parts;
  core::StageMetrics* const breakdowns = metrics != nullptr ? &parts : nullptr;
  const auto place = [metrics](const char* name) {
    if (metrics != nullptr) metrics->add_cpu(name, 0.0);
  };
  // The active set: the map's list while it still holds, its count set
  // here, or one compacted into this batch ("  ..encode set": apart from a
  // fuse's "  ..active set" timed into the same rows), its count left on
  // the device.
  const core::Buffer* list = nullptr;
  volume::DeviceCount list_count{&list_count_, 0, bound};
  if (const std::optional<volume::DeviceBlockList> held =
          grid.map().held_device_list()) {
    VKC_TRY(batch.fill(list_count_, 0, sizeof(std::uint32_t), held->count));
    list = held->buffer;
  } else {
    core::GpuStageScope compaction(breakdowns, gpu_timer_, "  ..encode set");
    place("  ..encode set");
    VKC_ASSIGN(const volume::RecordedBlockList active,
               grid.map().record_compaction(batch, nullptr, &compaction));
    list = active.buffer;
    list_count.buffer = active.count;
  }
  // Only the blocks with an observed voxel are coded -- with true_sdf, only
  // those holding surface, the cut set left beside them -- found on the
  // device before the transform: a fused band is mostly never integrated.
  {
    core::GpuStageScope row(breakdowns, gpu_timer_, "  ..observed");
    place("  ..observed");
    VKC_TRY(transform_->record_observed(batch, grid, *list, list_count,
                                        config_.true_sdf, &row));
  }
  // Then into the order the frame requires -- the hash table's order is
  // whatever the allocate atomics left, so sorting is also what makes the
  // same content give the same bytes -- the sort's last level writing the
  // reference store's spare list, and through the forward into the spare,
  // held against the reference (the one-step hold, encoder.hpp): the frame
  // the decoder will hold, which this one holds too once it returns the
  // frame. A P-frame's entries are coded against the reference there too,
  // bounded at M plus the frame's blocks; the count takes the survivors off
  // on the device. Either frame's symbols are counted, laid out and gathered
  // for the host in the same batch.
  // TODO(codec): trim a P-frame's host overhead at small R, where its encode
  // is still slower than intra's (DECISIONS, the 2026-10-17 freeze).
  {
    core::GpuStageScope forward(breakdowns, gpu_timer_, "  ..forward");
    place("  ..forward");
    const detail::FrameBuffers frame = store_->frame_buffers();
    const volume::DeviceCount observed = transform_->observed_count();
    VKC_TRY(lists_->record_sort(batch, transform_->observed_list(), observed,
                                *frame.list, &forward));
    VKC_ASSIGN(
        recorded->entries,
        transform_->record_forward(
            batch, grid, *frame.list, observed, params, *frame.masks,
            *frame.coefficients,
            config_.true_sdf ? &transform_->cut_set() : nullptr, &forward));
    if (held != nullptr) {
      VKC_TRY(store_->record_hold(batch, observed, k, &forward));
    }
    if (reference != nullptr) {
      VKC_ASSIGN(recorded->entries,
                 store_->record_diff(batch, observed, k, &forward));
    }
    VKC_TRY(writer_->record_count(
        batch, recorded->entries, config_.segment_size, config_.update_size,
        &forward, &observed,
        detail::TransformWords{&transform_->observed_counts(),
                               &transform_->rejected()}));
  }
  recorded->serial = next_serial();
  pending_ = std::move(recorded);
  return PendingEncode{pending_->serial, pending_->type};
}

core::Status Encoder::back(core::CommandBatch& batch,
                           const PendingEncode& pending,
                           core::StageMetrics* metrics) {
  if (!holds(pending)) {
    return fail("record_back",
                "the pending frame is not this encoder's last record "
                "(finished, abandoned, superseded, or another encoder's)");
  }
  detail::RecordedEncode& set = *pending_;
  if (set.back) {
    return fail("record_back", "the frame's back half is already recorded");
  }
  // The front's refusals: entries the transform found no block for, which a
  // grid changed under the batch would leave.
  const detail::FrontResults r = writer_->front();
  const std::uint32_t bound = set.entries.count - set.entries.reference_count;
  VKC_TRY(detail::DctTransform::check_rejected("observed", r.observed_rejected,
                                               bound));
  VKC_TRY(detail::DctTransform::check_rejected("forward", r.forward_rejected,
                                               r.blocks));
  set.blocks = r.blocks;
  if (r.blocks == 0) {
    // No block to code: an I-frame of none, the smaller, with no back batch.
    VKC_TRY(
        writer_->record_none(set.voxel_size, set.trunc_dist, config_.params));
    set.type = FrameType::kIntra;
    set.back = true;
    return {};
  }
  core::StageMetrics parts;
  core::GpuStageScope entropy(metrics != nullptr ? &parts : nullptr, gpu_timer_,
                              "  ..rans encode");
  VKC_TRY(writer_->record_code(
      batch, set.entries, set.voxel_size, set.trunc_dist, config_.params,
      set.type == FrameType::kInter ? &set.reference : nullptr, metrics,
      &entropy));
  set.back = true;
  return {};
}

core::Result<EncodedFrame> Encoder::finish_frame(
    const PendingEncode& pending, core::StageMetrics* metrics,
    std::vector<std::uint8_t>* bytes) {
  if (!holds(pending)) {
    return fail("finish",
                "the pending frame is not this encoder's last record "
                "(finished, abandoned, superseded, or another encoder's)");
  }
  if (!pending_->back) {
    return fail("finish", "the frame's back half is not recorded");
  }
  const std::unique_ptr<detail::RecordedEncode> set = std::move(pending_);
  VKC_ASSIGN(EncodedFrame frame, writer_->finish(metrics, gpu_timer_));
  // Only a frame returned becomes the reference, and counts toward the
  // period. It holds every block the forward made -- a P-frame's survivors
  // and new blocks, an I-frame's all -- which the spare holds in frame order,
  // so the next P-frame's M is the spare's list.
  VKC_ASSIGN(const detail::FrameHeader header,
             detail::read_frame_header(frame.head.data(), frame.head.size()));
  if (header.block_count != set->blocks) {
    return fail("encode", "the frame holds " +
                              std::to_string(header.block_count) +
                              " blocks of the " + std::to_string(set->blocks) +
                              " the forward made (a bug)");
  }
  // encode() returns contiguous bytes, so its allocation must succeed before
  // this frame replaces the reference or consumes the period/force request.
  if (bytes != nullptr) {
    bytes->reserve(frame.size());
    bytes->assign(frame.head.begin(), frame.head.end());
    if (frame.payload_size != 0) {
      bytes->insert(bytes->end(), frame.payload,
                    frame.payload + frame.payload_size);
    }
  }
  store_->commit(detail::FrameReference{
      detail::frame_hash(frame.head.data(), frame.head.size(), frame.payload,
                         frame.payload_size),
      header});
  gop_position_ = (frame.type == FrameType::kInter ? gop_position_ : 0) + 1;
  if (gop_position_ == config_.intra_period) gop_position_ = 0;
  force_intra_ = false;
  steps_held_ = true;  // its front batch ran, writing them
  return frame;
}

}  // namespace volumetric_kit::grid::codec
