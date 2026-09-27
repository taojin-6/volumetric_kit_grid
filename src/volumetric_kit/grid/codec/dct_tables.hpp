// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file dct_tables.hpp
/// @brief The two constant tables the 8^3 DCT is defined by: the orthonormal
///        1-D DCT-II basis and the 3-D zigzag coefficient order.
///
/// Internal (under src/, never installed). Built on the host and uploaded once,
/// rather than evaluated in the shader, so every device transforms against the
/// same bits: a GPU `cos()` is only as accurate as its vendor makes it, and the
/// codec's per-device reproducibility (the 2026-09-26 decision) should not
/// depend on that. The tests build their double-precision reference from the
/// same definitions, so the kernel and the oracle cannot disagree about *what*
/// the transform is -- only about how precisely it was computed.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <tuple>

#include "volumetric_kit/recon/codec/codec_params.hpp"

namespace volumetric_kit::recon::codec::detail {

/// One block edge, unsigned: the length of every line the separable transform
/// runs along.
inline constexpr std::uint32_t kEdge = static_cast<std::uint32_t>(kBlockSize);

/// A voxel's linear index inside a block is `x + kStrideY * y + kStrideZ * z`
/// -- the order the volume tier lays a block's voxels out in
/// (`BlockIndex::ptr + local`).
inline constexpr std::uint32_t kStrideY = kEdge;
inline constexpr std::uint32_t kStrideZ = kEdge * kEdge;
static_assert(kStrideZ * kEdge == kVoxelsPerBlock, "a block is kEdge^3");

/// Entries in the 1-D DCT matrix, `kEdge x kEdge`.
inline constexpr std::uint32_t kBasisSize = kEdge * kEdge;

/// @return The linear index of voxel `(x, y, z)` inside a block.
constexpr std::uint32_t voxel_index(std::uint32_t x, std::uint32_t y,
                                    std::uint32_t z) {
  return x + y * kStrideY + z * kStrideZ;
}

/// @brief The orthonormal 1-D DCT-II matrix, row-major `basis[k * kEdge + n]`.
///
/// With `N = kEdge`, `basis[k][n] = a(k) * cos(pi * k * (2n + 1) / (2N))`,
/// `a(0) = sqrt(1/N)` and `a(k) = sqrt(2/N)`. Orthonormal, so the forward
/// transform is `X[k] = sum_n basis[k][n] * x[n]`, the inverse is its
/// transpose, and the 3-D transform is this applied along x, then y, then z.
/// Evaluated in double.
/// @tparam T  `double` for the tests' reference, `float` for the upload.
template <typename T>
std::array<T, kBasisSize> dct_basis() {
  constexpr double kPi = 3.14159265358979323846;
  constexpr double n_points = kEdge;
  std::array<T, kBasisSize> basis{};
  for (std::uint32_t k = 0; k < kEdge; ++k) {
    const double alpha =
        k == 0 ? std::sqrt(1.0 / n_points) : std::sqrt(2.0 / n_points);
    for (std::uint32_t n = 0; n < kEdge; ++n) {
      basis[k * kEdge + n] = static_cast<T>(
          alpha * std::cos(kPi * k * (2.0 * n + 1.0) / (2.0 * n_points)));
    }
  }
  return basis;
}

/// @brief The 3-D zigzag order: `order[j]` is the voxel index (see
///        @ref voxel_index) of the `j`-th coefficient kept.
///
/// Sorted by total frequency `x + y + z`, ties broken lexicographically by
/// `(x, y, z)`. That rule reproduces the prior engine's hand-written 512-entry
/// table exactly (checked when it was ported), and is written as a rule so the
/// order is reviewable rather than a wall of literals.
inline std::array<std::uint32_t, kVoxelsPerBlock> zigzag_order() {
  std::array<std::tuple<std::uint32_t, std::uint32_t, std::uint32_t>,
             kVoxelsPerBlock>
      cells{};
  std::size_t i = 0;
  for (std::uint32_t x = 0; x < kEdge; ++x) {
    for (std::uint32_t y = 0; y < kEdge; ++y) {
      for (std::uint32_t z = 0; z < kEdge; ++z) {
        cells[i++] = {x, y, z};
      }
    }
  }
  std::sort(cells.begin(), cells.end(), [](const auto& a, const auto& b) {
    const auto [ax, ay, az] = a;
    const auto [bx, by, bz] = b;
    return std::make_tuple(ax + ay + az, ax, ay, az) <
           std::make_tuple(bx + by + bz, bx, by, bz);
  });
  std::array<std::uint32_t, kVoxelsPerBlock> order{};
  for (std::size_t j = 0; j < cells.size(); ++j) {
    const auto [x, y, z] = cells[j];
    order[j] = voxel_index(x, y, z);
  }
  return order;
}

}  // namespace volumetric_kit::recon::codec::detail
