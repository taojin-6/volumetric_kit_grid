// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include "volumetric_kit/recon/volume/voxel_block_grid.hpp"

#include <algorithm>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include "volumetric_kit/recon/core/command_batch.hpp"
#include "volumetric_kit/recon/core/compute_util.hpp"
#include "volumetric_kit/recon/core/device.hpp"
#include "volumetric_kit/recon/core/vulkan.hpp"

#include "block_stamp_comp.spv.hpp"
#include "block_zero_comp.spv.hpp"

namespace volumetric_kit::recon::volume {
namespace {

struct StampPush {
  std::uint32_t base;
  std::uint32_t voxels_per_block;
  std::uint32_t tick;
  std::uint32_t max_age;
  float observed_weight;
};

struct ZeroPush {
  std::uint32_t base;
  std::uint32_t voxels_per_block;
  std::uint32_t words_per_block;
  std::uint32_t num_words;
};

// One workgroup per listed block, in dispatches of at most the device's
// maxComputeWorkGroupCount[0], each starting at push.base.
template <typename Push>
Status dispatch_per_block(CommandBatch& batch, const ComputeKernel& kernel,
                          const DescriptorSet& set, Push push,
                          std::uint32_t count, std::uint32_t max_groups,
                          GpuStageScope* stage) {
  for (std::uint32_t base = 0; base < count;) {
    const std::uint32_t groups = std::min(max_groups, count - base);
    push.base = base;
    VR_TRY(batch.dispatch(kernel, set, &push, sizeof(push), groups, max_groups,
                          stage));
    base += groups;
  }
  return {};
}

// The per-voxel count backing every attribute: one element per voxel across the
// whole block pool. Grid dimensions are validated positive by
// VoxelHashMap::create (VoxelGridParams::validate), so this is well-defined.
std::uint64_t voxel_count(const VoxelGridParams& grid) {
  return static_cast<std::uint64_t>(grid.num_blocks) *
         static_cast<std::uint64_t>(grid.voxels_per_block);
}

}  // namespace

Result<VoxelBlockGrid> VoxelBlockGrid::create(Device& device,
                                              Allocator& allocator,
                                              const VoxelGridParams& grid,
                                              const AttributeSpec* attrs,
                                              std::size_t attr_count) {
  if (attr_count > 0 && attrs == nullptr) {
    return Status::invalid_argument("VoxelBlockGrid::create: attrs is null");
  }

  // Validate every spec up front, before building the map or allocating any
  // buffer, so a malformed spec (empty name, zero size, duplicate) is rejected
  // without the (potentially multi-GB) allocations the rest of create() does.
  for (std::size_t i = 0; i < attr_count; ++i) {
    const AttributeSpec& spec = attrs[i];
    if (spec.name.empty()) {
      return Status::invalid_argument(
          "VoxelBlockGrid::create: attribute name must be non-empty");
    }
    if (spec.element_size == 0) {
      return Status::invalid_argument(
          "VoxelBlockGrid::create: attribute element_size must be positive");
    }
    for (std::size_t j = 0; j < i; ++j) {
      if (attrs[j].name == spec.name) {
        return Status::invalid_argument(
            "VoxelBlockGrid::create: duplicate attribute name");
      }
    }
  }

  const VkDeviceSize max_range =
      max_storage_buffer_range(device.physical_device());

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
    VR_TRY(check_storage_buffer_range("VoxelBlockGrid::create: attribute array",
                                      bytes, max_range));
  }

  VR_ASSIGN(VoxelHashMap map, VoxelHashMap::create(device, allocator, grid));
  VoxelBlockGrid vbg(std::move(map), &device, &allocator, max_range);

  // Device-local, and zeroed on the device, so a fresh attribute reads as
  // all-zero without the host ever touching it.
  vbg.attributes_.reserve(attr_count);
  CommandBatch zero(device, allocator);
  for (std::size_t i = 0; i < attr_count; ++i) {
    const AttributeSpec& spec = attrs[i];
    const auto bytes = static_cast<VkDeviceSize>(elements) *
                       static_cast<VkDeviceSize>(spec.element_size);
    VR_ASSIGN(Buffer buffer, device_storage_buffer(allocator, bytes));
    VR_TRY(zero.zero(buffer, 0, bytes));
    vbg.attributes_.push_back(Attribute{std::string(spec.name),
                                        spec.element_size, std::move(buffer)});
  }
  VR_TRY(zero.submit());
  vbg.name_attribute_buffers();
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
                             debug_object_handle(attr.buffer.handle()),
                             attr.name.c_str());
  }
}

Result<AttributeView> VoxelBlockGrid::attribute(std::string_view name) const {
  if (!valid()) {
    return Status::invalid_argument(
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
        return Status::invalid_argument(
            "VoxelBlockGrid::attribute: the attribute array is smaller than "
            "the live grid's num_blocks * voxels_per_block -- the map was "
            "resized through map() instead of VoxelBlockGrid::resize");
      }
      return AttributeView{&attr.buffer, attr.element_size, count};
    }
  }
  return Status::invalid_argument(
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

Status VoxelBlockGrid::resize(std::int32_t new_num_buckets) {
  if (!valid()) {
    return Status::invalid_argument("VoxelBlockGrid::resize: moved-from grid");
  }
  // A value copy, not a reference: map_.resize() below mutates map_'s grid in
  // place, so a reference would start reporting the grown dimensions mid-way.
  const VoxelGridParams grid = map_.grid();
  if (new_num_buckets <= grid.num_buckets) {
    return Status::invalid_argument(
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
    return Status::invalid_argument(
        "VoxelBlockGrid::resize: new_num_buckets * bucket_size overflows a "
        "signed 32-bit num_blocks");
  }
  grown_grid.num_blocks = static_cast<std::int32_t>(grown_blocks);
  VR_TRY(grown_grid.validate());

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
    VR_TRY(check_storage_buffer_range(
        "VoxelBlockGrid::resize: grown attribute array", new_bytes,
        max_storage_buffer_range_));
  }

  // One batch for every attribute: each old array copied into its grown one
  // on the device, the tail zeroed there too.
  std::vector<Buffer> grown;
  grown.reserve(attributes_.size());
  CommandBatch batch(*device_, *allocator_);
  for (const Attribute& attr : attributes_) {
    const auto new_bytes = static_cast<VkDeviceSize>(new_elements) *
                           static_cast<VkDeviceSize>(attr.element_size);
    VR_ASSIGN(Buffer buffer, device_storage_buffer(*allocator_, new_bytes));
    const VkDeviceSize old_bytes = attr.buffer.size();
    VR_TRY(batch.copy(attr.buffer, 0, buffer, 0, old_bytes));
    VR_TRY(batch.zero(buffer, old_bytes, new_bytes - old_bytes));
    grown.push_back(std::move(buffer));
  }
  VR_TRY(batch.submit());

  VR_TRY(map_.resize(new_num_buckets));
  for (std::size_t i = 0; i < attributes_.size(); ++i) {
    attributes_[i].buffer = std::move(grown[i]);
  }
  // Every attribute now sits on a fresh handle, and a debug-utils name lives on
  // the handle -- so without this a capture taken after the first grow shows
  // exactly the arrays a bandwidth question is about as unnamed.
  name_attribute_buffers();
  return {};
}

Result<std::uint32_t> VoxelBlockGrid::remove(const BlockIndex* coords,
                                             std::uint32_t count,
                                             AllocFailures* out_failures) {
  if (!valid()) {
    return Status::invalid_argument("VoxelBlockGrid::remove: moved-from grid");
  }
  if (count == 0) {
    return std::uint32_t{0};
  }
  if (coords == nullptr) {
    return Status::invalid_argument("VoxelBlockGrid::remove: coords is null");
  }

  // Resolve each coord to the block pointer it currently holds. The map keys
  // coords to pointers on the device, and the compacted active set is the
  // existing way to read that mapping out -- no new kernel, and the dispatch is
  // quiescent between calls, so the snapshot is exact. Only needed when the
  // grid actually carries attributes.
  if (!attributes_.empty()) {
    VR_ASSIGN(std::vector<BlockIndex> active, map_.compact_active_blocks());
    // Sorted once, so each coord is a binary search: a scan of the whole
    // active set per coord made removing k of n blocks O(k * n), which a
    // decoder removing thousands of blocks a frame cannot afford.
    const auto less = [](const Vec3i& a, const Vec3i& b) {
      if (a.z != b.z) return a.z < b.z;
      if (a.y != b.y) return a.y < b.y;
      return a.x < b.x;
    };
    std::sort(active.begin(), active.end(),
              [&](const BlockIndex& a, const BlockIndex& b) {
                return less(a.coord, b.coord);
              });
    std::vector<std::uint64_t> firsts;  // each found block's first voxel
    for (std::uint32_t i = 0; i < count; ++i) {
      const auto block =
          std::lower_bound(active.begin(), active.end(), coords[i].coord,
                           [&](const BlockIndex& b, const Vec3i& c) {
                             return less(b.coord, c);
                           });
      if (block == active.end() || block->coord != coords[i].coord) {
        continue;
      }
      firsts.push_back(static_cast<std::uint64_t>(block->ptr));
    }
    VR_TRY(zero_blocks(std::move(firsts)));
  }

  // The topology epoch moves inside VoxelHashMap::remove -- where the index is
  // actually freed -- so there is nothing to bump here, and no way for the raw
  // map() path to skip it (see VoxelHashMap::topology_epoch).
  return map_.remove(coords, count, out_failures);
}

Status VoxelBlockGrid::clear() {
  if (!valid()) {
    return Status::invalid_argument("VoxelBlockGrid::clear: moved-from grid");
  }
  // Zero first, then free: a failure between the two leaves every block
  // still owning its index, at worst reading as unobserved. Freeing first
  // would hand the indices back over the old surface if the zeroing then
  // failed, and the LIFO heap would re-draw them onto it at full weight.
  CommandBatch batch(*device_, *allocator_);
  for (const Attribute& attr : attributes_) {
    VR_TRY(batch.zero(attr.buffer, 0, attr.buffer.size()));
  }
  VR_TRY(batch.submit());
  return map_.clear();  // moves the topology epoch; see topology_epoch()
}

Status VoxelBlockGrid::prepare_block_pass() {
  if (!valid()) {
    return Status::invalid_argument("VoxelBlockGrid: moved-from grid");
  }
  VR_ASSIGN(const AttributeView weight, attribute("weight"));
  if (weight.element_size != sizeof(float)) {
    return Status::invalid_argument(
        "VoxelBlockGrid: the block pass needs a float weight attribute");
  }
  if (stamp_kernel_.valid()) return {};
  // Built aside and committed at the end, the stamp kernel last, since it is
  // what says the pass is built.
  VR_ASSIGN(Buffer count,
            device_storage_buffer(*allocator_, sizeof(std::uint32_t)));
  VkPushConstantRange stamp_push{};
  stamp_push.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  stamp_push.size = sizeof(StampPush);
  VkPushConstantRange zero_push = stamp_push;
  zero_push.size = sizeof(ZeroPush);
  ComputeKernel stamp;
  ComputeKernel zero;
  KernelSetBuilder kb(*device_);
  VR_TRY(kb.add(stamp, "block_stamp", vr_block_stamp_comp_spv,
                vr_block_stamp_comp_spv_size, 5, &stamp_push));
  VR_TRY(kb.add(zero, "block_zero", vr_block_zero_comp_spv,
                vr_block_zero_comp_spv_size, 2, &zero_push));
  VR_ASSIGN(DescriptorPool pool, kb.build());
  KernelSets sets;
  VR_TRY(sets.reserve(*device_, zero,
                      static_cast<std::uint32_t>(attributes_.size())));
  VR_ASSIGN(GpuTimer timer, GpuTimer::create(*device_));
  VkPhysicalDeviceProperties props{};
  vkGetPhysicalDeviceProperties(device_->physical_device(), &props);
  device_->set_object_name(VK_OBJECT_TYPE_BUFFER,
                           debug_object_handle(count.handle()),
                           "grid.stale_count");
  stale_count_ = std::move(count);
  block_pool_ = std::move(pool);
  zero_kernel_ = std::move(zero);
  zero_sets_ = std::move(sets);
  gpu_timer_ = std::move(timer);
  max_workgroup_count_x_ = props.limits.maxComputeWorkGroupCount[0];
  stamp_kernel_ = std::move(stamp);
  return {};
}

Result<std::uint32_t> VoxelBlockGrid::block_pass(std::uint32_t max_age,
                                                 GpuStageScope& stage,
                                                 StageMetrics* metrics) {
  VR_ASSIGN(const AttributeView weight, attribute("weight"));
  VR_ASSIGN(const DeviceBlockList active,
            map_.compact_active_blocks_on_device(metrics));
  if (active.count == 0) return std::uint32_t{0};
  const VkDeviceSize list_bytes =
      VkDeviceSize(active.count) * sizeof(BlockIndex);
  if (stale_list_.size() < list_bytes) {
    VR_ASSIGN(stale_list_, device_storage_buffer(*allocator_, list_bytes));
    device_->set_object_name(VK_OBJECT_TYPE_BUFFER,
                             debug_object_handle(stale_list_.handle()),
                             "grid.stale_list");
  }
  const DescriptorSet& set = stamp_kernel_.set;
  set.write_storage_buffer(0, weight.buffer->handle(), 0, VK_WHOLE_SIZE);
  set.write_storage_buffer(1, active.buffer->handle(), 0, list_bytes);
  set.write_storage_buffer(2, map_.stamps_buffer().handle(), 0, VK_WHOLE_SIZE);
  set.write_storage_buffer(3, stale_list_.handle(), 0, VK_WHOLE_SIZE);
  set.write_storage_buffer(4, stale_count_.handle(), 0, VK_WHOLE_SIZE);
  const StampPush push{0,
                       static_cast<std::uint32_t>(map_.grid().voxels_per_block),
                       map_.tick(), max_age, kObservedWeight};
  std::uint32_t stale = 0;
  CommandBatch batch(*device_, *allocator_);
  VR_TRY(batch.fill(stale_count_, 0, sizeof(stale), 0u));
  VR_TRY(dispatch_per_block(batch, stamp_kernel_, set, push, active.count,
                            max_workgroup_count_x_, &stage));
  VR_TRY(batch.readback(stale_count_, 0, sizeof(stale), &stale));
  VR_TRY(batch.submit());
  return std::min(stale, active.count);
}

Status VoxelBlockGrid::stamp_blocks(StageMetrics* metrics) {
  VR_TRY(prepare_block_pass());
  GpuStageScope stage(metrics, gpu_timer_, "block stamps");
  return block_pass(0, stage, metrics).status();
}

Result<std::uint32_t> VoxelBlockGrid::free_stale_blocks(std::uint32_t max_age,
                                                        StageMetrics* metrics) {
  if (max_age == 0) {
    return Status::invalid_argument(
        "VoxelBlockGrid::free_stale_blocks: max_age must be at least 1");
  }
  VR_TRY(prepare_block_pass());
  const auto voxels_per_block =
      static_cast<std::uint32_t>(map_.grid().voxels_per_block);
  for (const Attribute& attr : attributes_) {
    if (std::uint64_t{voxels_per_block} * attr.element_size % 4 != 0) {
      return Status::invalid_argument(
          "VoxelBlockGrid::free_stale_blocks: attribute '" + attr.name +
          "' does not fill whole 4-byte words a block");
    }
  }
  GpuStageScope stage(metrics, gpu_timer_, "block stamps");
  VR_ASSIGN(const std::uint32_t stale, block_pass(max_age, stage, metrics));
  if (stale == 0) return std::uint32_t{0};
  // Zero the blocks where the pass listed them, a dispatch an attribute, and
  // read the list back for remove, which takes host coordinates.
  const VkDeviceSize list_bytes = VkDeviceSize(stale) * sizeof(BlockIndex);
  std::vector<BlockIndex> blocks(stale);
  CommandBatch batch(*device_, *allocator_);
  for (std::size_t i = 0; i < attributes_.size(); ++i) {
    const Attribute& attr = attributes_[i];
    const DescriptorSet& set = zero_sets_[i];
    set.write_storage_buffer(0, attr.buffer.handle(), 0, VK_WHOLE_SIZE);
    set.write_storage_buffer(1, stale_list_.handle(), 0, list_bytes);
    const ZeroPush push{0, voxels_per_block,
                        voxels_per_block * attr.element_size / 4,
                        static_cast<std::uint32_t>(attr.buffer.size() / 4)};
    VR_TRY(dispatch_per_block(batch, zero_kernel_, set, push, stale,
                              max_workgroup_count_x_, &stage));
  }
  VR_TRY(batch.readback(stale_list_, 0, list_bytes, blocks.data()));
  VR_TRY(batch.submit());
  VR_ASSIGN(const std::uint32_t failed, map_.remove(blocks.data(), stale));
  return stale - std::min(failed, stale);
}

Status VoxelBlockGrid::zero_blocks(std::vector<std::uint64_t> firsts) {
  // Zero each block's slice of every attribute on the device, in one batch.
  // Sorted and merged, so blocks the LIFO heap handed out side by side cost
  // one fill, not one each; and attribute by attribute, so each array's
  // fills rise through it and share one barrier (see CommandBatch).
  // TODO(volume): zero through block_zero.comp, as free_stale_blocks does, if
  // removing thousands of scattered blocks becomes common: MoltenVK runs each
  // fill as a dispatch of its own (2026-10-01).
  const auto voxels_per_block =
      static_cast<std::uint64_t>(map_.grid().voxels_per_block);
  std::sort(firsts.begin(), firsts.end());
  firsts.erase(std::unique(firsts.begin(), firsts.end()), firsts.end());
  CommandBatch batch(*device_, *allocator_);
  for (const Attribute& attr : attributes_) {
    for (std::size_t i = 0; i < firsts.size();) {
      std::size_t j = i + 1;
      while (j < firsts.size() &&
             firsts[j] == firsts[j - 1] + voxels_per_block) {
        ++j;
      }
      const std::uint64_t offset = firsts[i] * attr.element_size;
      const std::uint64_t bytes =
          (j - i) * voxels_per_block * attr.element_size;
      i = j;
      if (offset + bytes > attr.buffer.size()) {
        continue;  // an out-of-lockstep array; attribute() reports it
      }
      VR_TRY(batch.zero(attr.buffer, offset, bytes));
    }
  }
  return batch.submit();
}

Status VoxelBlockGrid::check_block_list(const BlockList& blocks,
                                        const char* who) const {
  if (blocks.blocks == nullptr && blocks.count != 0) {
    return Status::invalid_argument(
        std::string(who) + ": the block list is null with a non-zero count");
  }
  // Exempt when empty: a default-constructed list's epoch is 0, which
  // next_topology_epoch() never returns, so the comparison could only fail.
  if (blocks.count != 0 && blocks.epoch != topology_epoch()) {
    return Status::invalid_argument(
        std::string(who) +
        ": the block list was compacted against topology epoch " +
        std::to_string(blocks.epoch) + ", but this grid is now at " +
        std::to_string(topology_epoch()) +
        " (a remove()/clear() since then has re-used its block pointers)");
  }
  if (blocks.count > static_cast<std::uint32_t>(grid().num_blocks)) {
    return Status::invalid_argument(
        std::string(who) + ": the block list holds " +
        std::to_string(blocks.count) +
        " blocks, more than this grid's block heap (" +
        std::to_string(grid().num_blocks) + ")");
  }
  return {};
}

}  // namespace volumetric_kit::recon::volume
