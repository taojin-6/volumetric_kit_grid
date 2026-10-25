// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file dispatch.hpp
/// @brief A loop that records a 1-D dispatch of items whose group count can
///        pass the device's limit, its test seam, and the same split for a
///        pass whose size a kernel finds on the device.
///
/// Internal (under src/, never installed). Vulkan guarantees only 65535
/// groups along x (`maxComputeWorkGroupCount[0]`), which a room's blocks
/// already pass, so such a dispatch is recorded as several rather than
/// refused. The seam carries `VR_CORE_API` so a test reaches it in a
/// shared-library build too.

#include <algorithm>
#include <cstdint>

#include "volumetric_kit/core/base/result.hpp"
#include "volumetric_kit/core/vulkan/allocator.hpp"
#include "volumetric_kit/core/vulkan/buffer.hpp"
#include "volumetric_kit/core/vulkan/command_batch.hpp"
#include "volumetric_kit/core/vulkan/compute_kernel.hpp"
#include "volumetric_kit/core/vulkan/descriptor.hpp"
#include "volumetric_kit/core/vulkan/device.hpp"
#include "volumetric_kit/recon/core/export.hpp"

namespace volumetric_kit::recon::detail {

/// @return The groups one dispatch of @ref dispatch_items holds:
///         @p max_groups, the device's `maxComputeWorkGroupCount[0]`, or the
///         cap @ref set_test_groups_per_dispatch set, if lower.
VR_CORE_API std::uint32_t groups_per_dispatch(std::uint32_t max_groups);

/// @brief For tests only: cap @ref groups_per_dispatch at @p groups, so a
///        small input takes the split path a device's limit gives a large
///        one; 0 lifts the cap. Process-wide.
VR_CORE_API void set_test_groups_per_dispatch(std::uint32_t groups);

/// Record @p kernel over @p items invocations of @p lanes a workgroup, in as
/// many dispatches of @ref groups_per_dispatch groups as it takes; each pushes
/// @p push with the first item it covers as `item_base`, and binds @p set, or
/// the kernel's own set when it is null. The batch's barrier between
/// dispatches orders each after the one before.
template <typename Push>
core::Status dispatch_items(core::CommandBatch& batch,
                            const core::ComputeKernel& kernel, Push push,
                            std::uint64_t items, std::uint32_t lanes,
                            std::uint32_t max_groups,
                            core::GpuStageScope* stage,
                            const core::DescriptorSet* set = nullptr) {
  const std::uint64_t per_dispatch =
      std::uint64_t(groups_per_dispatch(max_groups)) * lanes;
  for (std::uint64_t base = 0; base < items; base += per_dispatch) {
    push.item_base = static_cast<std::uint32_t>(base);
    const std::uint64_t count =
        std::min<std::uint64_t>(per_dispatch, items - base);
    const auto groups = static_cast<std::uint32_t>((count + lanes - 1) / lanes);
    VKC_TRY(set != nullptr ? batch.dispatch(kernel, *set, &push, sizeof(push),
                                            groups, max_groups, stage)
                           : batch.dispatch(kernel, &push, sizeof(push), groups,
                                            max_groups, stage));
  }
  return {};
}

/// The indirect slices of one pass whose size a kernel wrote on the device
/// (`core/shaders/dispatch_slices.glsl`): @ref dispatches
/// `VkDispatchIndirectCommand`s from @ref offset of @ref buffer, each of up
/// to @ref per_dispatch groups of @ref lanes items, split as
/// @ref dispatch_items splits a pass the host knows.
struct IndirectSlices {
  const core::Buffer* buffer = nullptr;
  VkDeviceSize offset = 0;
  std::uint32_t dispatches = 0;
  std::uint32_t per_dispatch = 0;
  std::uint32_t lanes = 1;
};

/// @return The slices a pass of at most @p max_items items, @p lanes a group,
///         takes at @p per_dispatch groups a slice: at least 1.
VR_CORE_API std::uint32_t slice_count(std::uint64_t max_items,
                                      std::uint32_t lanes,
                                      std::uint32_t per_dispatch);

/// Record @p kernel over @p slices, each pushing @p push with its first item
/// as `item_base`, binding the kernel's own set.
template <typename Push>
core::Status dispatch_slices(core::CommandBatch& batch,
                             const core::ComputeKernel& kernel, Push push,
                             const IndirectSlices& slices,
                             core::GpuStageScope* stage) {
  for (std::uint32_t d = 0; d < slices.dispatches; ++d) {
    push.item_base = d * slices.per_dispatch * slices.lanes;
    VKC_TRY(batch.dispatch_indirect(
        kernel, &push, sizeof(push), *slices.buffer,
        slices.offset + VkDeviceSize(d) * sizeof(VkDispatchIndirectCommand),
        stage));
  }
  return {};
}

/// @brief What @ref record_dispatch_args sizes: a count -- the word at
///        @ref offset of @ref buffer, read as at most @ref bound, or
///        @ref bound itself when @ref buffer is null -- of entries of
///        @ref multiplier items each, and the gate that makes it 0 -- the
///        word at @ref gate_offset of @ref gate -- unless @ref gate is null.
struct SlicedCount {
  const core::Buffer* buffer = nullptr;
  VkDeviceSize offset = 0;
  std::uint32_t bound = 0;
  std::uint32_t multiplier = 1;
  const core::Buffer* gate = nullptr;
  VkDeviceSize gate_offset = 0;
};

/// Add recon's one count-to-indirect-slices kernel
/// (`core/shaders/dispatch_args.comp`) to @p kb as @p kernel, named @p name.
/// Each owner builds an instance of its own: a kernel and the buffer of the
/// slices it writes (@ref record_dispatch_args).
VR_CORE_API core::Status add_dispatch_args(core::KernelSetBuilder& kb,
                                           core::ComputeKernel& kernel,
                                           const char* name);

/// Record into @p batch the sizing of passes of @p count at @p lanes items a
/// group into @p slices, and at @p second_lanes (0 for none) into @p second,
/// so a count on the device, or a gate there, sizes them without reaching the
/// host: @p kernel (from @ref add_dispatch_args) writes them into @p buffer,
/// grown to the slices @p count's bound needs and named @p name. Every batch
/// that read @p buffer before must have resolved.
/// @return OK; `Status::Code::InvalidArgument` for a misaligned word or a
///         bound past 32-bit items; or a buffer failure.
VR_CORE_API core::Status record_dispatch_args(
    core::CommandBatch& batch, core::Device& device, core::Allocator& allocator,
    core::ComputeKernel& kernel, core::Buffer& buffer, const char* name,
    const SlicedCount& count, std::uint32_t lanes, IndirectSlices* slices,
    std::uint32_t second_lanes, IndirectSlices* second,
    std::uint32_t max_groups, core::GpuStageScope* stage);

}  // namespace volumetric_kit::recon::detail
