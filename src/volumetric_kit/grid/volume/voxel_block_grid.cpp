// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include "volumetric_kit/recon/volume/voxel_block_grid.hpp"

#include <cstring>
#include <string>
#include <utility>
#include <vector>

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

  VR_ASSIGN(VoxelHashMap map, VoxelHashMap::create(device, allocator, grid));
  VoxelBlockGrid vbg(std::move(map), &allocator);

  const std::uint64_t elements = voxel_count(grid);
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
      // live grid, so it always matches attr.buffer and never inflates past it
      // if the map is later resized (see VoxelBlockGrid::map()).
      return AttributeView{&attr.buffer, attr.element_size,
                           attr.buffer.size() / attr.element_size};
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

}  // namespace volumetric_kit::recon::volume
