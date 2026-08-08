// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include "volumetric_kit/recon/volume/voxel_block_grid.hpp"

#include <cstring>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include "volumetric_kit/recon/core/compute_util.hpp"
#include "volumetric_kit/recon/core/device.hpp"
#include "volumetric_kit/recon/core/vulkan.hpp"

namespace volumetric_kit::recon::volume {
namespace {

// The per-voxel count backing every attribute: one element per voxel across the
// whole block pool. Grid dimensions are validated positive by
// VoxelHashMap::create (VoxelGridParams::validate), so this is well-defined.
std::uint64_t voxel_count(const VoxelGridParams& grid) {
  return static_cast<std::uint64_t>(grid.num_blocks) *
         static_cast<std::uint64_t>(grid.voxels_per_block);
}

// A host-visible, host-mapped storage buffer of the given byte size,
// uninitialised (the caller fills or zeroes it). resize() copies old attribute
// contents into the head and zeroes only the grown tail through this, avoiding
// a redundant full-buffer zero-then-overwrite.
// TODO(volume): host-visible for this slice; a device-local + staging path is a
// follow-up perf pass, matching the hash-map buffers.
Result<Buffer> raw_attribute_buffer(Allocator& allocator, VkDeviceSize bytes) {
  BufferDesc desc;
  desc.size = bytes;
  desc.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
  desc.memory = MemoryUsage::HostVisible;
  desc.mapped = true;
  desc.host_access = HostAccess::Random;
  return allocator.create_buffer(desc);
}

// The same, fully zeroed so a freshly-created attribute reads as all-zero.
Result<Buffer> attribute_buffer(Allocator& allocator, VkDeviceSize bytes) {
  VR_ASSIGN(Buffer buffer, raw_attribute_buffer(allocator, bytes));
  std::memset(buffer.mapped(), 0, static_cast<std::size_t>(bytes));
  return buffer;
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
  VoxelBlockGrid vbg(std::move(map), &allocator, max_range);

  vbg.attributes_.reserve(attr_count);
  for (std::size_t i = 0; i < attr_count; ++i) {
    const AttributeSpec& spec = attrs[i];
    const auto bytes = static_cast<VkDeviceSize>(elements) *
                       static_cast<VkDeviceSize>(spec.element_size);
    VR_ASSIGN(Buffer buffer, attribute_buffer(allocator, bytes));
    vbg.attributes_.push_back(Attribute{std::string(spec.name),
                                        spec.element_size, std::move(buffer)});
  }
  return vbg;
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

  std::vector<Buffer> grown;
  grown.reserve(attributes_.size());
  for (const Attribute& attr : attributes_) {
    const auto new_bytes = static_cast<VkDeviceSize>(new_elements) *
                           static_cast<VkDeviceSize>(attr.element_size);
    VR_ASSIGN(Buffer buffer, raw_attribute_buffer(*allocator_, new_bytes));
    const auto old_bytes = static_cast<std::size_t>(attr.buffer.size());
    auto* dst = static_cast<std::uint8_t*>(buffer.mapped());
    std::memcpy(dst, attr.buffer.mapped(), old_bytes);
    std::memset(dst + old_bytes, 0,
                static_cast<std::size_t>(new_bytes) - old_bytes);
    grown.push_back(std::move(buffer));
  }

  VR_TRY(map_.resize(new_num_buckets));
  for (std::size_t i = 0; i < attributes_.size(); ++i) {
    attributes_[i].buffer = std::move(grown[i]);
  }
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
    const auto voxels_per_block =
        static_cast<std::uint64_t>(map_.grid().voxels_per_block);
    for (std::uint32_t i = 0; i < count; ++i) {
      for (const BlockIndex& block : active) {
        if (block.coord != coords[i].coord) {
          continue;
        }
        // Zero this block's slice of every attribute. The buffers are
        // host-visible and mapped, and remove() is synchronous, so this is a
        // plain memset rather than a fill dispatch.
        const auto first = static_cast<std::uint64_t>(block.ptr);
        for (Attribute& attr : attributes_) {
          const std::uint64_t offset = first * attr.element_size;
          const std::uint64_t bytes = voxels_per_block * attr.element_size;
          if (offset + bytes > attr.buffer.size()) {
            continue;  // an out-of-lockstep array; attribute() reports it
          }
          std::memset(static_cast<std::uint8_t*>(attr.buffer.mapped()) + offset,
                      0, static_cast<std::size_t>(bytes));
        }
        break;
      }
    }
  }

  return map_.remove(coords, count, out_failures);
}

Status VoxelBlockGrid::clear() {
  if (!valid()) {
    return Status::invalid_argument("VoxelBlockGrid::clear: moved-from grid");
  }
  // Order matters only in that both must happen; the map clear is the one that
  // can fail, so run it first and leave the attributes untouched if it does --
  // zeroed attributes under a still-populated table would read as fused blocks
  // that lost their data.
  VR_TRY(map_.clear());
  for (Attribute& attr : attributes_) {
    std::memset(attr.buffer.mapped(), 0,
                static_cast<std::size_t>(attr.buffer.size()));
  }
  return {};
}

}  // namespace volumetric_kit::recon::volume
