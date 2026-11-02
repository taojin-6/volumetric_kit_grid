// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include "volumetric_kit/grid/volume/ray_caster.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

#include "volumetric_kit/core/vulkan/compute_util.hpp"
#include "volumetric_kit/core/vulkan/device.hpp"
#include "volumetric_kit/core/vulkan/vulkan.hpp"
#include "volumetric_kit/grid/base/device_requirements.hpp"

#include "hash_map_internal.hpp"
#include "ray_cast_comp.spv.hpp"

namespace volumetric_kit::grid::volume {
namespace {

constexpr std::uint32_t kLocalSize = 64;  // ray_cast.comp's local_size_x
constexpr std::uint32_t kBindings = 7;
constexpr VkDeviceSize kCameraBytes = sizeof(DepthPinhole);
constexpr VkDeviceSize kDepthBytes = sizeof(float);
constexpr VkDeviceSize kNormalBytes = 3 * sizeof(float);

// Mirrors Push in ray_cast.comp, byte for byte under scalar layout.
struct CastPush {
  VoxelGridParams grid;
  float free_step;
  float min_step_voxels;
  float min_weight;
  std::uint32_t max_steps;
  std::uint32_t allow[8];
  std::uint32_t width;
  std::uint32_t height;
  std::uint32_t count;
  std::uint32_t use_attribute;
};
static_assert(sizeof(CastPush) == 96, "CastPush layout drift");
static_assert(offsetof(CastPush, allow) == 48, "CastPush layout drift");
static_assert(offsetof(CastPush, width) == 80, "CastPush layout drift");

core::Status check_config(const RayCasterConfig& c) {
  if (!(c.free_step > 0.0f && c.free_step <= 1.0f)) {
    return core::Status::invalid_argument(
        "RayCaster::create: free_step must lie in (0, 1]");
  }
  if (!(c.min_step_voxels > 0.0f && std::isfinite(c.min_step_voxels))) {
    return core::Status::invalid_argument(
        "RayCaster::create: min_step_voxels must be positive and finite");
  }
  if (c.max_steps == 0) {
    return core::Status::invalid_argument(
        "RayCaster::create: max_steps must be at least 1");
  }
  if (c.casts_per_batch == 0) {
    return core::Status::invalid_argument(
        "RayCaster::create: casts_per_batch must be at least 1");
  }
  return {};
}

// A grid attribute of `bytes` a voxel, by name.
core::Result<AttributeView> attribute_of(const VoxelBlockGrid& grid,
                                         std::string_view name,
                                         std::uint32_t bytes) {
  if (!grid.has_attribute(name)) {
    return core::Status::invalid_argument(
        "RayCaster::record_cast: the grid has no attribute '" +
        std::string(name) + "'");
  }
  VKC_ASSIGN(const AttributeView view, grid.attribute(name));
  if (view.element_size != bytes) {
    return core::Status::invalid_argument(
        "RayCaster::record_cast: attribute '" + std::string(name) + "' is " +
        std::to_string(view.element_size) + " bytes a voxel, not " +
        std::to_string(bytes));
  }
  return view;
}

}  // namespace

RayCaster::RayCaster(core::Device& device, core::Allocator& allocator,
                     const RayCasterConfig& config) noexcept
    : device_(device), allocator_(allocator), config_(config) {}

RayCaster::~RayCaster() = default;

core::Result<std::unique_ptr<RayCaster>> RayCaster::create(
    core::Device& device, core::Allocator& allocator,
    const RayCasterConfig& config) {
  VKC_TRY(check_config(config));
  VKC_TRY(check_device_requirements(device, "RayCaster::create"));
  std::unique_ptr<RayCaster> caster(new RayCaster(device, allocator, config));
  VkPushConstantRange push{};
  push.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  push.offset = 0;
  push.size = sizeof(CastPush);
  core::KernelSetBuilder kb(device);
  VKC_TRY(kb.add(caster->kernel_, "ray_cast", vgrid_ray_cast_comp_spv,
                 vgrid_ray_cast_comp_spv_size, kBindings, &push));
  VKC_ASSIGN(caster->pool_, kb.build());
  VKC_TRY(
      caster->sets_.reserve(device, caster->kernel_, config.casts_per_batch));
  VkPhysicalDeviceProperties props{};
  vkGetPhysicalDeviceProperties(device.physical_device(), &props);
  caster->max_groups_ = props.limits.maxComputeWorkGroupCount[0];
  caster->max_range_ = core::max_storage_buffer_range(device);
  return caster;
}

std::uint64_t RayCaster::max_rays() const noexcept {
  // One thread a ray in a 1-D dispatch, its 32-bit index, and the normals,
  // the larger output, in one binding.
  return std::min({std::uint64_t(max_groups_) * kLocalSize,
                   std::uint64_t(max_range_ / kNormalBytes),
                   std::uint64_t(UINT32_MAX)});
}

core::Result<ModelView> RayCaster::make_view(std::uint32_t width,
                                             std::uint32_t height,
                                             std::uint32_t views) {
  if (width == 0 || height == 0 || views == 0) {
    return core::Status::invalid_argument(
        "RayCaster::make_view: width, height and views must be positive");
  }
  const std::uint64_t rays = std::uint64_t(width) * height * views;
  if (rays > max_rays()) {
    return core::Status::invalid_argument(
        "RayCaster::make_view: " + std::to_string(rays) +
        " rays, more than one dispatch holds on this device (" +
        std::to_string(max_rays()) + ")");
  }
  ModelView out;
  out.width = width;
  out.height = height;
  out.views = views;
  VKC_ASSIGN(out.depth,
             core::device_storage_buffer(allocator_, rays * kDepthBytes));
  VKC_ASSIGN(out.normal,
             core::device_storage_buffer(allocator_, rays * kNormalBytes));
  // So a GPU capture shows what the later kernels read.
  device_.set_object_name(VK_OBJECT_TYPE_BUFFER,
                          core::debug_object_handle(out.depth.handle()),
                          "model_view.depth");
  device_.set_object_name(VK_OBJECT_TYPE_BUFFER,
                          core::debug_object_handle(out.normal.handle()),
                          "model_view.normal");
  return out;
}

core::Status RayCaster::record_cast(core::CommandBatch& batch,
                                    const VoxelBlockGrid& grid,
                                    const core::Buffer& cameras,
                                    std::uint32_t count,
                                    const CastFilter& filter, ModelView& out,
                                    core::GpuStageScope* stage) {
  if (count == 0 || count > out.views) {
    return core::Status::invalid_argument(
        "RayCaster::record_cast: count " + std::to_string(count) +
        " is not 1 to the view's " + std::to_string(out.views));
  }
  // At a floor of 0 or below, a voxel never integrated (weight 0, tsdf 0)
  // would answer as a surface.
  if (!(filter.min_weight > 0.0f) || !std::isfinite(filter.min_weight)) {
    return core::Status::invalid_argument(
        "RayCaster::record_cast: the filter's min_weight must be positive "
        "and finite");
  }
  const std::uint64_t pixels = std::uint64_t(out.width) * out.height;
  const std::uint64_t view_rays = pixels * out.views;
  const std::uint64_t rays = pixels * count;
  if (pixels == 0 || rays > max_rays()) {
    return core::Status::invalid_argument(
        "RayCaster::record_cast: " + std::to_string(rays) +
        " rays, not 1 to what one dispatch holds (" +
        std::to_string(max_rays()) + ")");
  }
  const VkBufferUsageFlags storage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
  if (out.depth.size() < view_rays * kDepthBytes ||
      out.normal.size() < view_rays * kNormalBytes ||
      (out.depth.usage() & storage) == 0 ||
      (out.normal.usage() & storage) == 0) {
    return core::Status::invalid_argument(
        "RayCaster::record_cast: the view's depth and normal must be storage "
        "buffers of at least 4 and 12 bytes a pixel of its views");
  }
  if (cameras.size() < count * kCameraBytes ||
      (cameras.usage() & storage) == 0) {
    return core::Status::invalid_argument(
        "RayCaster::record_cast: cameras must be a storage buffer of at least "
        "96 bytes a camera");
  }
  VKC_ASSIGN(const AttributeView tsdf,
             attribute_of(grid, kTsdfAttribute, sizeof(float)));
  VKC_ASSIGN(const AttributeView weight,
             attribute_of(grid, kWeightAttribute, sizeof(float)));
  // With no attribute named, the weight array fills its binding, unread:
  // the core enables no null descriptors.
  AttributeView source = weight;
  if (!filter.attribute.empty()) {
    VKC_ASSIGN(source, attribute_of(grid, filter.attribute, 1));
  }
  const VoxelHashMap& map = grid.map();
  const VkDeviceSize entries_bytes =
      VoxelHashMap::Internal::entries_buffer_size(map);
  VKC_TRY(core::check_storage_buffer_range(
      "RayCaster::record_cast: hash entries", entries_bytes, max_range_));

  // Every binding, every cast: the grid's buffers move with a resize, and a
  // recast in one batch binds another set.
  const core::DescriptorSet& set = sets_[next_set_];
  next_set_ = (next_set_ + 1) % config_.casts_per_batch;
  set.write_storage_buffer(0, VoxelHashMap::Internal::entries_buffer(map), 0,
                           entries_bytes);
  set.write_storage_buffer(1, tsdf.buffer->handle(), 0, VK_WHOLE_SIZE);
  set.write_storage_buffer(2, weight.buffer->handle(), 0, VK_WHOLE_SIZE);
  set.write_storage_buffer(3, source.buffer->handle(), 0, VK_WHOLE_SIZE);
  set.write_storage_buffer(4, cameras.handle(), 0, count * kCameraBytes);
  set.write_storage_buffer(5, out.depth.handle(), 0, rays * kDepthBytes);
  set.write_storage_buffer(6, out.normal.handle(), 0, rays * kNormalBytes);

  CastPush push{};
  push.grid = grid.grid();
  push.free_step = config_.free_step;
  push.min_step_voxels = config_.min_step_voxels;
  push.min_weight = filter.min_weight;
  push.max_steps = config_.max_steps;
  std::copy(filter.allow.begin(), filter.allow.end(), push.allow);
  push.width = out.width;
  push.height = out.height;
  push.count = count;
  push.use_attribute = filter.attribute.empty() ? 0u : 1u;
  const std::uint32_t groups =
      core::group_count(static_cast<std::uint32_t>(rays), kLocalSize);
  return batch.dispatch(kernel_, set, &push, sizeof(push), groups, max_groups_,
                        stage);
}

}  // namespace volumetric_kit::grid::volume
