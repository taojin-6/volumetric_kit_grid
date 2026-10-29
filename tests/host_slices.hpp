// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

// Indirect slices whose sizes the host writes, for a test that records a pass
// the decoder sizes on the device (recon::detail::IndirectSlices). Reaches
// the internal core/dispatch.hpp by src/.

#include <cstdint>
#include <vector>

#include "volumetric_kit/core/base/result.hpp"
#include "volumetric_kit/core/testing/buffer_readback.hpp"
#include "volumetric_kit/core/vulkan/buffer.hpp"
#include "volumetric_kit/core/vulkan/compute_util.hpp"
#include "volumetric_kit/core/vulkan/vulkan.hpp"
#include "volumetric_kit/recon/core/dispatch.hpp"

namespace vr_test {

namespace vkc = volumetric_kit::core;

// `items` items at `lanes` a group, in slices as the device writes them.
struct HostSlices {
  vkc::Buffer buffer;
  std::uint32_t dispatches = 0;
  std::uint32_t per_dispatch = 0;
  std::uint32_t lanes = 1;

  volumetric_kit::recon::detail::IndirectSlices slices() const {
    return {&buffer, 0, dispatches, per_dispatch, lanes};
  }
};

inline vkc::Result<HostSlices> host_slices(const vkc::test::Gpu& gpu,
                                           std::uint32_t items,
                                           std::uint32_t lanes) {
  namespace rd = volumetric_kit::recon::detail;
  VkPhysicalDeviceProperties props{};
  vkGetPhysicalDeviceProperties(gpu.device.physical_device(), &props);
  HostSlices out;
  out.lanes = lanes;
  out.per_dispatch =
      rd::groups_per_dispatch(props.limits.maxComputeWorkGroupCount[0]);
  out.dispatches = rd::slice_count(items, lanes, out.per_dispatch);
  const std::uint32_t groups = (items + lanes - 1) / lanes;
  std::vector<VkDispatchIndirectCommand> commands(out.dispatches);
  for (std::uint32_t d = 0; d < out.dispatches; ++d) {
    const std::uint32_t begin = d * out.per_dispatch;
    commands[d] = {
        groups > begin ? std::min(groups - begin, out.per_dispatch) : 0u, 1u,
        1u};
  }
  const VkDeviceSize bytes =
      commands.size() * sizeof(VkDispatchIndirectCommand);
  VKC_ASSIGN(out.buffer,
             vkc::device_storage_buffer(gpu.allocator, bytes,
                                        VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT));
  VKC_TRY(
      vkc::test::write_back(gpu.device, gpu.allocator, out.buffer, commands));
  return out;
}

}  // namespace vr_test
