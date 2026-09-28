// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

// A grid attribute as a host vector, and back, for a test: the arrays are
// device-local, so a test edits a copy and writes it back.

#include <string_view>
#include <vector>

#include "buffer_readback.hpp"
#include "volumetric_kit/recon/volume/voxel_block_grid.hpp"

namespace vr_test {

// The whole of attribute `name`, as elements of T.
template <typename T>
vr::Result<std::vector<T>> read_attribute(
    const vr::Device& device, vr::Allocator& allocator,
    const volumetric_kit::recon::volume::VoxelBlockGrid& grid,
    std::string_view name) {
  VR_ASSIGN(const auto view, grid.attribute(name));
  return read_back<T>(device, allocator, *view.buffer,
                      view.buffer->size() / sizeof(T));
}

// Write `data` over the start of attribute `name`.
template <typename T>
vr::Status write_attribute(
    const vr::Device& device, vr::Allocator& allocator,
    const volumetric_kit::recon::volume::VoxelBlockGrid& grid,
    std::string_view name, const std::vector<T>& data) {
  VR_ASSIGN(const auto view, grid.attribute(name));
  return write_back(device, allocator, *view.buffer, data);
}

}  // namespace vr_test
