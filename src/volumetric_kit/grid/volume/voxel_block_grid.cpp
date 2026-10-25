// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include "volumetric_kit/recon/volume/voxel_block_grid.hpp"

#include <algorithm>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include "volumetric_kit/core/vulkan/command_batch.hpp"
#include "volumetric_kit/core/vulkan/compute_util.hpp"
#include "volumetric_kit/core/vulkan/device.hpp"
#include "volumetric_kit/core/vulkan/vulkan.hpp"
#include "volumetric_kit/recon/core/dispatch.hpp"

#include "table_sizes.hpp"

#include "block_stamp_comp.spv.hpp"
#include "block_zero_comp.spv.hpp"

namespace volumetric_kit::recon::volume {
namespace {

// Both kernels run a workgroup a listed block, item_base the first block a
// dispatch covers (recon::detail::dispatch_items).
struct StampPush {
  std::uint32_t item_base;
  std::uint32_t voxels_per_block;
  std::uint32_t tick;
  std::uint32_t max_age;
  float observed_weight;
};

struct ZeroPush {
  std::uint32_t item_base;
  std::uint32_t voxels_per_block;
  std::uint32_t words_per_block;
  std::uint32_t num_words;
  std::int32_t num_buckets;
  std::int32_t bucket_size;
  std::int32_t max_chain;
};

// The per-voxel count backing every attribute: one element per voxel across the
// whole block pool. Grid dimensions are validated positive by
// VoxelHashMap::create (VoxelGridParams::validate), so this is well-defined.
std::uint64_t voxel_count(const VoxelGridParams& grid) {
  return static_cast<std::uint64_t>(grid.num_blocks) *
         static_cast<std::uint64_t>(grid.voxels_per_block);
}

}  // namespace

core::Result<VoxelBlockGrid> VoxelBlockGrid::create(core::Device& device,
                                                    core::Allocator& allocator,
                                                    const VoxelGridParams& grid,
                                                    const AttributeSpec* attrs,
                                                    std::size_t attr_count) {
  if (attr_count > 0 && attrs == nullptr) {
    return core::Status::invalid_argument(
        "VoxelBlockGrid::create: attrs is null");
  }

  // Validate every spec up front, before building the map or allocating any
  // buffer, so a malformed spec (empty name, zero size, duplicate) is rejected
  // without the (potentially multi-GB) allocations the rest of create() does.
  for (std::size_t i = 0; i < attr_count; ++i) {
    const AttributeSpec& spec = attrs[i];
    if (spec.name.empty()) {
      return core::Status::invalid_argument(
          "VoxelBlockGrid::create: attribute name must be non-empty");
    }
    if (spec.element_size == 0) {
      return core::Status::invalid_argument(
          "VoxelBlockGrid::create: attribute element_size must be positive");
    }
    // Whole words a block, so the zero kernel can clear one block alone.
    if (static_cast<std::uint64_t>(grid.voxels_per_block) * spec.element_size %
            4 !=
        0) {
      return core::Status::invalid_argument(
          "VoxelBlockGrid::create: an attribute's block must fill whole 4-byte "
          "words (voxels_per_block * element_size a multiple of 4)");
    }
    for (std::size_t j = 0; j < i; ++j) {
      if (attrs[j].name == spec.name) {
        return core::Status::invalid_argument(
            "VoxelBlockGrid::create: duplicate attribute name");
      }
    }
  }

  const VkDeviceSize max_range = core::max_storage_buffer_range(device);

  // Check every attribute against the binding limit before building the map or
  // allocating anything, for the same reason the spec validation above runs
  // first: these are the largest allocations in the repo, and rejecting one
  // after the others have been made wastes gigabytes. Bound here rather than at
  // the consumers because this is where the size is chosen -- tsdf integration
  // and meshing bind these VK_WHOLE_SIZE, so what they could check is the same
  // number decided here.
  const std::uint64_t elements = voxel_count(grid);
  for (std::size_t i = 0; i < attr_count; ++i) {
    const auto bytes = static_cast<VkDeviceSize>(elements) *
                       static_cast<VkDeviceSize>(attrs[i].element_size);
    VKC_TRY(core::check_storage_buffer_range(
        "VoxelBlockGrid::create: attribute array", bytes, max_range));
  }

  VKC_ASSIGN(VoxelHashMap map, VoxelHashMap::create(device, allocator, grid));
  VoxelBlockGrid vbg(std::move(map), &device, &allocator, max_range);

  // Device-local, and zeroed on the device, so a fresh attribute reads as
  // all-zero without the host ever touching it.
  vbg.attributes_.reserve(attr_count);
  core::CommandBatch zero(device, allocator);
  for (std::size_t i = 0; i < attr_count; ++i) {
    const AttributeSpec& spec = attrs[i];
    const auto bytes = static_cast<VkDeviceSize>(elements) *
                       static_cast<VkDeviceSize>(spec.element_size);
    VKC_ASSIGN(core::Buffer buffer,
               core::device_storage_buffer(allocator, bytes));
    VKC_TRY(zero.zero(buffer, 0, bytes));
    vbg.attributes_.push_back(Attribute{std::string(spec.name),
                                        spec.element_size, std::move(buffer)});
  }
  VKC_TRY(zero.submit());
  vbg.name_attribute_buffers();
  if (!vbg.attributes_.empty()) {
    VKC_TRY(vbg.build_block_kernels());
  }
  return vbg;
}

void VoxelBlockGrid::name_attribute_buffers() const noexcept {
  if (device_ == nullptr) {
    return;
  }
  for (const Attribute& attr : attributes_) {
    // c_str() rather than the string_view the spec was declared with: the
    // driver reads a null-terminated string, and Attribute::name is the owned
    // copy made at create() precisely so one outlives the declaration.
    device_->set_object_name(VK_OBJECT_TYPE_BUFFER,
                             core::debug_object_handle(attr.buffer.handle()),
                             attr.name.c_str());
  }
}

core::Result<AttributeView> VoxelBlockGrid::attribute(
    std::string_view name) const {
  if (!valid()) {
    return core::Status::invalid_argument(
        "VoxelBlockGrid::attribute: moved-from grid");
  }
  for (const Attribute& attr : attributes_) {
    if (attr.name == name) {
      // element_count is derived from the buffer this view carries, not the
      // live grid, so it always matches attr.buffer and never inflates past it.
      const std::uint64_t count = attr.buffer.size() / attr.element_size;
      // ...and *because* it describes the buffer rather than the grid, it is
      // also what makes the two comparable. Every consumer that binds an
      // attribute comes through here, so this is the one place a desync can be
      // caught -- and it has to be caught, because both consumers bind
      // VK_WHOLE_SIZE over the stale, smaller buffer while the kernels address
      // it by BlockIndex::ptr derived from the grown grid, and
      // robustBufferAccess is enabled nowhere. Only VoxelHashMap::resize
      // reached through map() can produce this; VoxelBlockGrid::resize grows
      // both sides together.
      if (count < voxel_count(map_.grid())) {
        return core::Status::invalid_argument(
            "VoxelBlockGrid::attribute: the attribute array is smaller than "
            "the live grid's num_blocks * voxels_per_block -- the map was "
            "resized through map() instead of VoxelBlockGrid::resize");
      }
      return AttributeView{&attr.buffer, attr.element_size, count};
    }
  }
  return core::Status::invalid_argument(
      "VoxelBlockGrid::attribute: no attribute of that name");
}

bool VoxelBlockGrid::has_attribute(std::string_view name) const noexcept {
  for (const Attribute& attr : attributes_) {
    if (attr.name == name) {
      return true;
    }
  }
  return false;
}

core::Status VoxelBlockGrid::resize(std::int32_t new_num_buckets) {
  if (!valid()) {
    return core::Status::invalid_argument(
        "VoxelBlockGrid::resize: moved-from grid");
  }
  // A value copy, not a reference: map_.resize() below mutates map_'s grid in
  // place, so a reference would start reporting the grown dimensions mid-way.
  const VoxelGridParams grid = map_.grid();
  if (new_num_buckets <= grid.num_buckets) {
    return core::Status::invalid_argument(
        "VoxelBlockGrid::resize: new_num_buckets must exceed the current "
        "count");
  }

  // Validate the grown grid FIRST, against the same contract map_.resize will
  // apply. It runs that check too -- but only after this function has already
  // built every enlarged attribute buffer beside the live ones, which at the
  // examples' defaults is gigabytes committed and then thrown away to report an
  // invalid_argument the parameters could have been rejected on.
  //
  // This overlaps the maxStorageBufferRange guard below, and on real hardware
  // that guard fires first -- an attribute array large enough to overflow the
  // block-pointer bound is at least 8 GB, far past any device's binding limit.
  // Kept anyway because it says something the other cannot: the grown grid is
  // illegal *arithmetically*, on a machine of any size and a device of any
  // limit, and it reports that rather than a limit the caller might read as
  // "this hardware is too small".
  VoxelGridParams grown_grid = grid;
  grown_grid.num_buckets = new_num_buckets;
  const std::int64_t grown_blocks =
      static_cast<std::int64_t>(new_num_buckets) * grid.bucket_size;
  if (grown_blocks > std::numeric_limits<std::int32_t>::max()) {
    return core::Status::invalid_argument(
        "VoxelBlockGrid::resize: new_num_buckets * bucket_size overflows a "
        "signed 32-bit num_blocks");
  }
  grown_grid.num_blocks = static_cast<std::int32_t>(grown_blocks);
  VKC_TRY(grown_grid.validate());

  // The grown per-voxel count: new num_blocks = bucket_size * new_num_buckets
  // (the invariant VoxelHashMap::create validates), one attribute element per
  // voxel of the pool.
  const std::uint64_t new_elements =
      static_cast<std::uint64_t>(grid.bucket_size) *
      static_cast<std::uint64_t>(new_num_buckets) *
      static_cast<std::uint64_t>(grid.voxels_per_block);

  // Build each grown attribute buffer off to the side: copy the old contents
  // into the head (bounded by the source buffer's own size, so it can never
  // over-read even if map_ and the attributes were ever out of lockstep) and
  // zero only the grown tail. map_.resize below preserves every block's index
  // (BlockIndex::ptr < the old count), so the copied data stays correctly
  // addressed and the tail stays zero for future blocks. Commit only once the
  // map resize succeeds -- and map_.resize is itself all-or-nothing -- so a
  // failure leaves the grid untouched.
  // Every doubling takes the arrays further past the binding limit, so the
  // grown size is checked exactly as the initial one is -- before any of the
  // (potentially multi-GB) allocations below.
  for (const Attribute& attr : attributes_) {
    const auto new_bytes = static_cast<VkDeviceSize>(new_elements) *
                           static_cast<VkDeviceSize>(attr.element_size);
    VKC_TRY(core::check_storage_buffer_range(
        "VoxelBlockGrid::resize: grown attribute array", new_bytes,
        max_storage_buffer_range_));
  }

  // One batch for every attribute: each old array copied into its grown one
  // on the device, the tail zeroed there too.
  std::vector<core::Buffer> grown;
  grown.reserve(attributes_.size());
  core::CommandBatch batch(*device_, *allocator_);
  for (const Attribute& attr : attributes_) {
    const auto new_bytes = static_cast<VkDeviceSize>(new_elements) *
                           static_cast<VkDeviceSize>(attr.element_size);
    VKC_ASSIGN(core::Buffer buffer,
               core::device_storage_buffer(*allocator_, new_bytes));
    const VkDeviceSize old_bytes = attr.buffer.size();
    VKC_TRY(batch.copy(attr.buffer, 0, buffer, 0, old_bytes));
    VKC_TRY(batch.zero(buffer, old_bytes, new_bytes - old_bytes));
    grown.push_back(std::move(buffer));
  }
  VKC_TRY(batch.submit());

  VKC_TRY(map_.resize(new_num_buckets));
  for (std::size_t i = 0; i < attributes_.size(); ++i) {
    attributes_[i].buffer = std::move(grown[i]);
  }
  // Every attribute now sits on a fresh handle, and a debug-utils name lives on
  // the handle -- so without this a capture taken after the first grow shows
  // exactly the arrays a bandwidth question is about as unnamed.
  name_attribute_buffers();
  return {};
}

std::int32_t VoxelBlockGrid::max_num_buckets() const noexcept {
  if (!valid()) return 0;
  const VoxelGridParams& grid = map_.grid();
  // validate(): num_blocks, and num_blocks * voxels_per_block (a block
  // pointer), each within int32.
  constexpr std::uint64_t kInt32Max = std::numeric_limits<std::int32_t>::max();
  std::uint64_t blocks = kInt32Max / std::uint64_t(grid.voxels_per_block);
  // resize(): every attribute array within one binding.
  for (const Attribute& attr : attributes_) {
    blocks = std::min<std::uint64_t>(
        blocks, max_storage_buffer_range_ /
                    (std::uint64_t(grid.voxels_per_block) * attr.element_size));
  }
  return static_cast<std::int32_t>(blocks / std::uint64_t(grid.bucket_size));
}

std::uint64_t VoxelBlockGrid::bytes_at(
    std::int32_t num_buckets) const noexcept {
  if (!valid() || num_buckets <= 0) return 0;
  const VoxelGridParams& grid = map_.grid();
  const auto buckets = std::uint64_t(num_buckets);
  const auto bucket_size = std::uint64_t(grid.bucket_size);
  const std::uint64_t voxels =
      buckets * bucket_size * std::uint64_t(grid.voxels_per_block);
  std::uint64_t bytes = table_sizes(buckets, bucket_size).total();
  for (const Attribute& attr : attributes_) {
    bytes += voxels * attr.element_size;
  }
  return bytes;
}

core::Result<std::uint32_t> VoxelBlockGrid::remove(const core::Buffer& coords,
                                                   std::uint32_t count,
                                                   AllocFailures* out_failures,
                                                   core::GpuStageScope* stage) {
  if (!valid()) {
    return core::Status::invalid_argument(
        "VoxelBlockGrid::remove: moved-from grid");
  }
  if (count == 0) {
    return std::uint32_t{0};
  }
  // Before anything is zeroed or freed, so a refused input changes nothing.
  const VkDeviceSize bytes = VkDeviceSize(count) * sizeof(BlockIndex);
  VKC_TRY(core::StorageInput(coords).check("VoxelBlockGrid::remove: the coords",
                                           bytes));
  VKC_TRY(core::check_storage_buffer_range("VoxelBlockGrid::remove: the coords",
                                           bytes, max_storage_buffer_range_));
  return zero_and_remove(coords, count, out_failures, stage);
}

core::Result<std::uint32_t> VoxelBlockGrid::zero_and_remove(
    const core::Buffer& list, std::uint32_t count, AllocFailures* out_failures,
    core::GpuStageScope* stage) {
  // The topology epoch moves inside VoxelHashMap::remove -- where the index is
  // actually freed -- so there is nothing to bump here, and no way for the raw
  // map() path to skip it (see VoxelHashMap::topology_epoch).
  if (attributes_.empty()) {
    return map_.remove(list, count, out_failures, {}, stage);
  }
  // The zero kernel finds each coord's block in the table itself, so this
  // costs the count, not the grid. It rides the map's first recording, ahead
  // of the delete in the same submit, so every block is zeroed before its
  // index is freed.
  return map_.remove(
      list, count, out_failures,
      [&](core::CommandBatch& batch) {
        return record_zero(batch, list, count, nullptr, nullptr, stage);
      },
      stage);
}

core::Result<RecordedRounds> VoxelBlockGrid::record_remove(
    core::CommandBatch& batch, const core::Buffer& coords,
    const DeviceCount& count, const DeviceCount* gate,
    core::GpuStageScope* stage) {
  if (!valid()) {
    return core::Status::invalid_argument(
        "VoxelBlockGrid::record_remove: moved-from grid");
  }
  if (attributes_.empty()) {
    return map_.record_remove(batch, coords, count, gate, stage);
  }
  // Zeroed in the map's recording once nothing more can refuse it, ahead of
  // its rounds, so a refused recording zeroes nothing.
  return map_.record_remove(
      batch, coords, count, gate, stage, [&](core::CommandBatch& b) {
        return record_zero(b, coords, count.bound, &count, gate, stage);
      });
}

core::Status VoxelBlockGrid::clear() {
  if (!valid()) {
    return core::Status::invalid_argument(
        "VoxelBlockGrid::clear: moved-from grid");
  }
  // Zero first, then free: a failure between the two leaves every block
  // still owning its index, at worst reading as unobserved. Freeing first
  // would hand the indices back over the old surface if the zeroing then
  // failed, and the LIFO heap would re-draw them onto it at full weight.
  core::CommandBatch batch(*device_, *allocator_);
  for (const Attribute& attr : attributes_) {
    VKC_TRY(batch.zero(attr.buffer, 0, attr.buffer.size()));
  }
  VKC_TRY(batch.submit());
  return map_.clear();  // moves the topology epoch; see topology_epoch()
}

core::Status VoxelBlockGrid::prepare_block_pass() {
  if (!valid()) {
    return core::Status::invalid_argument("VoxelBlockGrid: moved-from grid");
  }
  VKC_ASSIGN(const AttributeView weight, attribute("weight"));
  if (weight.element_size != sizeof(float)) {
    return core::Status::invalid_argument(
        "VoxelBlockGrid: the block pass needs a float weight attribute");
  }
  return {};
}

core::Status VoxelBlockGrid::build_block_kernels() {
  VKC_ASSIGN(core::Buffer count,
             core::device_storage_buffer(*allocator_, sizeof(std::uint32_t)));
  VkPushConstantRange stamp_push{};
  stamp_push.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  stamp_push.size = sizeof(StampPush);
  VkPushConstantRange zero_push = stamp_push;
  zero_push.size = sizeof(ZeroPush);
  core::ComputeKernel stamp;
  core::ComputeKernel zero;
  core::ComputeKernel zero_args;
  core::ComputeKernel stamp_args;
  core::KernelSetBuilder kb(*device_);
  VKC_TRY(kb.add(stamp, "block_stamp", vr_block_stamp_comp_spv,
                 vr_block_stamp_comp_spv_size, 5, &stamp_push));
  VKC_TRY(kb.add(zero, "block_zero", vr_block_zero_comp_spv,
                 vr_block_zero_comp_spv_size, 3, &zero_push));
  VKC_TRY(recon::detail::add_dispatch_args(kb, zero_args, "block_zero_args"));
  VKC_TRY(recon::detail::add_dispatch_args(kb, stamp_args, "block_stamp_args"));
  VKC_ASSIGN(core::DescriptorPool pool, kb.build());
  core::KernelSets sets;
  VKC_TRY(sets.reserve(*device_, zero,
                       static_cast<std::uint32_t>(attributes_.size())));
  VkPhysicalDeviceProperties props{};
  vkGetPhysicalDeviceProperties(device_->physical_device(), &props);
  device_->set_object_name(VK_OBJECT_TYPE_BUFFER,
                           core::debug_object_handle(count.handle()),
                           "grid.stale_count");
  stale_count_ = std::move(count);
  block_pool_ = std::move(pool);
  zero_kernel_ = std::move(zero);
  zero_sets_ = std::move(sets);
  zero_args_kernel_ = std::move(zero_args);
  stamp_args_kernel_ = std::move(stamp_args);
  max_workgroup_count_x_ = props.limits.maxComputeWorkGroupCount[0];
  stamp_kernel_ = std::move(stamp);
  return {};
}

core::Status VoxelBlockGrid::prepare_free_pass(
    std::uint32_t max_age, const core::StageMetrics* metrics) {
  if (max_age == 0) {
    return core::Status::invalid_argument(
        "VoxelBlockGrid::free_stale_blocks: max_age must be at least 1");
  }
  VKC_TRY(prepare_block_pass());
  // The first timed pass makes the timer, so a grid nobody times holds no
  // query pool: a device layered on Metal backs each with one of its few
  // counter sample buffers.
  if (metrics != nullptr && !gpu_timer_.valid()) {
    VKC_ASSIGN(gpu_timer_, core::GpuTimer::create(*device_));
  }
  return {};
}

void VoxelBlockGrid::place_free_pass_rows(core::StageMetrics* metrics) {
  // The stage's row and its compaction's, placed as the pass opens, so a
  // batch's later stages, whose device halves come back with theirs, do not
  // come between them.
  if (metrics == nullptr) return;
  metrics->add_cpu("block stamps", 0.0);
  metrics->add_cpu("  ..stale set", 0.0);
}

core::Result<RecordedRounds> VoxelBlockGrid::record_stale(
    core::CommandBatch& batch, std::uint32_t max_age,
    core::GpuStageScope& stage, std::uint32_t* initial_count) {
  VKC_ASSIGN(const AttributeView weight, attribute("weight"));
  // Every block slot's room, since only the device knows how many are stale;
  // every batch this grid recorded into before has resolved (the @pre).
  const auto slots = static_cast<std::uint32_t>(map_.grid().num_blocks);
  const VkDeviceSize list_bytes = VkDeviceSize(slots) * sizeof(BlockIndex);
  VKC_TRY(core::ensure_device_scratch(*device_, *allocator_, stale_list_,
                                      list_bytes, max_storage_buffer_range_,
                                      "grid.stale_list"));
  // The active set, on the device: the compaction's own breakdown, its
  // device half alone, as the stage's row holds its host time.
  core::StageMetrics parts;
  RecordedBlockList active;
  {
    core::GpuStageScope compaction(stage.timer() != nullptr ? &parts : nullptr,
                                   gpu_timer_, "  ..stale set");
    VKC_ASSIGN(active, map_.record_compaction(batch, nullptr, &compaction));
  }
  // The stamp pass, a group an active block, sized from the compaction's
  // count; it lists the stale blocks and counts them.
  recon::detail::IndirectSlices groups;
  VKC_TRY(recon::detail::record_dispatch_args(
      batch, *device_, *allocator_, stamp_args_kernel_, stamp_slices_,
      "grid.stamp_slices",
      recon::detail::SlicedCount{active.count, 0, slots, 1}, 1, &groups, 0,
      nullptr, max_workgroup_count_x_, &stage));
  const core::DescriptorSet& set = stamp_kernel_.set;
  set.write_storage_buffer(0, weight.buffer->handle(), 0, VK_WHOLE_SIZE);
  set.write_storage_buffer(1, active.buffer->handle(), 0, list_bytes);
  set.write_storage_buffer(2, map_.stamps_buffer().handle(), 0, VK_WHOLE_SIZE);
  set.write_storage_buffer(3, stale_list_.handle(), 0, list_bytes);
  set.write_storage_buffer(4, stale_count_.handle(), 0, VK_WHOLE_SIZE);
  const StampPush push{0,
                       static_cast<std::uint32_t>(map_.grid().voxels_per_block),
                       map_.tick(), max_age, kObservedWeight};
  VKC_TRY(batch.fill(stale_count_, 0, sizeof(std::uint32_t), 0u));
  VKC_TRY(recon::detail::dispatch_slices(batch, stamp_kernel_, push, groups,
                                         &stage));
  // Each stale block zeroed, then freed, from the list and count the pass
  // left: the removal needs no host coordinate or count.
  VKC_ASSIGN(
      RecordedRounds removed,
      record_remove(batch, stale_list_, DeviceCount{&stale_count_, 0, slots},
                    nullptr, &stage));
  if (initial_count != nullptr) {
    // The removal leaves the compaction's count unchanged. Read it beside
    // the result, since an earlier unsettled recording can leave the host's
    // occupancy stale in either direction.
    VKC_TRY(batch.readback(*active.count, 0, sizeof(*initial_count),
                           initial_count));
  }
  return removed;
}

core::Result<RecordedRounds> VoxelBlockGrid::record_free_stale(
    core::CommandBatch& batch, std::uint32_t max_age,
    core::StageMetrics* metrics) {
  VKC_TRY(prepare_free_pass(max_age, metrics));
  core::GpuStageScope stage(metrics, gpu_timer_, "block stamps");
  place_free_pass_rows(metrics);
  return record_stale(batch, max_age, stage);
}

core::Result<std::uint32_t> VoxelBlockGrid::free_stale_blocks(
    std::uint32_t max_age, core::StageMetrics* metrics) {
  VKC_TRY(prepare_free_pass(max_age, metrics));
  // The row spans the wait too, and takes the batch's device halves after.
  core::GpuStageScope stage(metrics, gpu_timer_, "block stamps");
  place_free_pass_rows(metrics);
  std::uint32_t held = 0;
  core::CommandBatch batch(*device_, *allocator_);
  VKC_ASSIGN(const RecordedRounds removed,
             record_stale(batch, max_age, stage, &held));
  VKC_ASSIGN(core::PendingBatch pending, batch.submit_async({}, {}));
  VKC_TRY(pending.wait());
  if (metrics != nullptr) metrics->merge_device(pending.device_rows());
  VKC_TRY(map_.settle(removed).status());
  return held - std::min(held, map_.allocated_blocks());
}

core::Status VoxelBlockGrid::record_zero(core::CommandBatch& batch,
                                         const core::Buffer& list,
                                         std::uint32_t count,
                                         const DeviceCount* words,
                                         const DeviceCount* gate,
                                         core::GpuStageScope* stage) {
  const VoxelGridParams& gp = map_.grid();
  const auto voxels_per_block = static_cast<std::uint32_t>(gp.voxels_per_block);
  const VkDeviceSize list_bytes = VkDeviceSize(count) * sizeof(BlockIndex);
  const VkDeviceSize entries_bytes = map_.entries_buffer_size();
  VKC_TRY(core::check_storage_buffer_range("VoxelBlockGrid: the hash entries",
                                           entries_bytes,
                                           max_storage_buffer_range_));
  // A workgroup a block, as many as the device's count says behind its gate.
  recon::detail::IndirectSlices slices;
  if (words != nullptr) {
    VKC_TRY(recon::detail::record_dispatch_args(
        batch, *device_, *allocator_, zero_args_kernel_, zero_slices_,
        "grid.zero_slices",
        recon::detail::SlicedCount{words->buffer, words->offset, count, 1,
                                   gate != nullptr ? gate->buffer : nullptr,
                                   gate != nullptr ? gate->offset : 0},
        1, &slices, 0, nullptr, max_workgroup_count_x_, stage));
  }
  for (std::size_t i = 0; i < attributes_.size(); ++i) {
    const Attribute& attr = attributes_[i];
    const core::DescriptorSet& set = zero_sets_[i];
    set.write_storage_buffer(0, attr.buffer.handle(), 0, VK_WHOLE_SIZE);
    set.write_storage_buffer(1, list.handle(), 0, list_bytes);
    set.write_storage_buffer(2, map_.entries_buffer(), 0, entries_bytes);
    const ZeroPush push{0,
                        voxels_per_block,
                        voxels_per_block * attr.element_size / 4,
                        static_cast<std::uint32_t>(attr.buffer.size() / 4),
                        gp.num_buckets,
                        gp.bucket_size,
                        gp.max_chain};
    if (words != nullptr) {
      // An indirect dispatch binds the kernel's own set.
      zero_kernel_.set = set;
      VKC_TRY(recon::detail::dispatch_slices(batch, zero_kernel_, push, slices,
                                             stage));
    } else {
      VKC_TRY(recon::detail::dispatch_items(batch, zero_kernel_, push, count, 1,
                                            max_workgroup_count_x_, stage,
                                            &set));
    }
  }
  return {};
}

}  // namespace volumetric_kit::recon::volume
