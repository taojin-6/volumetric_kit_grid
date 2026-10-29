// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file device_requirements.hpp
/// @brief What grid's kernels require of a device.
///
/// The device is volumetric_kit_core's (`core/vulkan/device.hpp`). What grid
/// adds is @ref device_requirements, the one statement of what grid's kernels
/// need, which a standalone caller passes to `Instance::select_physical_device`
/// and `Device::create`, and an embedder merges with its own (`merge`, or a
/// `SharedDeviceConfig`) before building a shared device.
///
/// @code
/// const core::DeviceRequirements reqs = device_requirements();
/// VKC_ASSIGN(core::PhysicalDeviceInfo gpu,
///            instance.select_physical_device(reqs));
/// VKC_ASSIGN(core::Device device, core::Device::create(instance, gpu, reqs));
/// @endcode

#include "volumetric_kit/core/base/result.hpp"
#include "volumetric_kit/core/vulkan/device.hpp"
#include "volumetric_kit/core/vulkan/device_requirements.hpp"

namespace volumetric_kit::grid {

/// @brief What grid's kernels require of a device.
///
/// A compute queue with timeline semaphores (the core's defaults) and
/// `scalarBlockLayout`, the buffer ABI every grid shader declares
/// (`layout(scalar)`; DECISIONS.md, 2026-07-05).
///
/// Pass these to every `Device::create` and `Device::adopt` grid runs on.
/// The core's default requirements leave `scalarBlockLayout` off, and a
/// grid class that builds kernels refuses such a device
/// (@ref check_device_requirements).
///
/// @code
/// VKC_ASSIGN(core::Device device,
///            core::Device::create(instance, gpu, device_requirements()));
/// @endcode
/// @return The requirements; a caller may add to them before creating.
inline core::DeviceRequirements device_requirements() {
  core::DeviceRequirements reqs;
  reqs.scalar_block_layout = true;
  return reqs;
}

/// @brief Refuse a device that did not enable what grid's kernels need.
///
/// Called first by every grid `create` that builds kernels: on a device
/// without `scalarBlockLayout` the kernels would build and run without an
/// error, reading their buffers at the wrong offsets.
///
/// @code
/// VKC_TRY(check_device_requirements(device, "MarchingCubes::create"));
/// @endcode
/// @param device  The device the kernels are for.
/// @param who     The caller, prefixed to a refusal.
/// @return OK; or `Device::check_enabled`'s `Status::Code::Unsupported`
///         naming what @ref device_requirements needs and the device did not
///         enable.
inline core::Status check_device_requirements(const core::Device& device,
                                              const char* who) {
  return device.check_enabled(device_requirements()).with_context(who);
}

}  // namespace volumetric_kit::grid
