// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include "volumetric_kit/recon/volume/voxel_block_grid.hpp"

#include <cstring>
#include <string>
#include <utility>

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

// A host-visible, host-mapped storage buffer of the given byte size, zeroed so
// a freshly-created attribute reads as all-zero.
// TODO(volume): host-visible for this slice; a device-local + staging path is a
// follow-up perf pass, matching the hash-map buffers.
Result<Buffer> attribute_buffer(Allocator& allocator, VkDeviceSize bytes) {
  BufferDesc desc;
  desc.size = bytes;
  desc.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
  desc.memory = MemoryUsage::HostVisible;
  desc.mapped = true;
  desc.host_access = HostAccess::Random;
  VR_ASSIGN(Buffer buffer, allocator.create_buffer(desc));
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
  VoxelBlockGrid vbg(std::move(map));

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

}  // namespace volumetric_kit::recon::volume
