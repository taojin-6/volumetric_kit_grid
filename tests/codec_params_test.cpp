// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// Host-only tests for the codec tier: CodecParams validation, and the two
// tables the DCT is defined by. CPU-only, so these always run.
//
// The tables get independent checks because the GPU test's reference is BUILT
// from them: a wrong zigzag order or basis would be reproduced by both sides of
// that comparison and pass it. So the basis is checked for orthonormality and
// known values, and the zigzag order for being a permutation that climbs in
// total frequency and matches the prior engine's table, all 512 entries.

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>

#include "dct_tables.hpp"
#include "volumetric_kit/recon/codec/codec_params.hpp"

namespace codec = volumetric_kit::recon::codec;
using codec::kVoxelsPerBlock;
using codec::detail::kBasisSize;
using codec::detail::kEdge;

#define CHECK(cond)                                                        \
  do {                                                                     \
    if (!(cond)) {                                                         \
      std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
      return 1;                                                            \
    }                                                                      \
  } while (0)

namespace {

int params_validate_case() {
  codec::CodecParams p;
  CHECK(p.validate().ok());  // the defaults pass

  p.coefficient_count = 0;
  CHECK(!p.validate().ok());
  p.coefficient_count = 513;
  CHECK(!p.validate().ok());
  p.coefficient_count = 1;
  CHECK(p.validate().ok());
  p.coefficient_count = kVoxelsPerBlock;
  CHECK(p.validate().ok());

  // The floor is inclusive, and it is what keeps the clamp from engaging:
  // the largest possible coefficient, sqrt(kVoxelsPerBlock), divided by the
  // floor step rounds to no more than the clamp.
  p.dc_step = codec::kMinStep;
  p.ac_step = codec::kMinStep;
  CHECK(p.validate().ok());
  CHECK(std::nearbyint(std::sqrt(double(kVoxelsPerBlock)) /
                       double(codec::kMinStep)) <=
        double(codec::kMaxQuantizedMagnitude));
  p.dc_step = codec::kMinStep * 0.99f;
  CHECK(!p.validate().ok());
  p.dc_step = 0.25f;
  p.ac_step = 0.0f;
  CHECK(!p.validate().ok());
  p.ac_step = -0.05f;
  CHECK(!p.validate().ok());
  p.ac_step = std::numeric_limits<float>::quiet_NaN();
  CHECK(!p.validate().ok());
  p.ac_step = std::numeric_limits<float>::infinity();
  CHECK(!p.validate().ok());
  return 0;
}

int basis_case() {
  const std::array<double, kBasisSize> b = codec::detail::dct_basis<double>();
  // Orthonormal: B * B^T = I.
  for (std::uint32_t i = 0; i < kEdge; ++i) {
    for (std::uint32_t j = 0; j < kEdge; ++j) {
      double dot = 0.0;
      for (std::uint32_t n = 0; n < kEdge; ++n) {
        dot += b[i * kEdge + n] * b[j * kEdge + n];
      }
      CHECK(std::fabs(dot - (i == j ? 1.0 : 0.0)) < 1e-12);
    }
  }
  // Known values, written out for the 8-point transform rather than derived
  // from the formula under test: the DC row is flat at sqrt(1/8), and row 1
  // starts at sqrt(2/8) * cos(pi/16).
  static_assert(kEdge == 8, "the known values below are the 8-point DCT's");
  for (std::uint32_t n = 0; n < kEdge; ++n) {
    CHECK(std::fabs(b[n] - std::sqrt(1.0 / 8.0)) < 1e-12);
  }
  CHECK(std::fabs(b[kEdge] - 0.5 * std::cos(3.14159265358979323846 / 16.0)) <
        1e-12);
  // The float upload is the double table rounded, nothing more.
  const std::array<float, kBasisSize> f = codec::detail::dct_basis<float>();
  for (std::size_t i = 0; i < kBasisSize; ++i) {
    CHECK(f[i] == static_cast<float>(b[i]));
  }
  return 0;
}

int zigzag_case() {
  const auto order = codec::detail::zigzag_order();
  // A permutation of the 512 voxel indices.
  std::array<bool, kVoxelsPerBlock> seen{};
  for (std::uint32_t v : order) {
    CHECK(v < kVoxelsPerBlock);
    CHECK(!seen[v]);
    seen[v] = true;
  }
  // Climbing in total frequency x + y + z, so a prefix of K keeps the lowest.
  int last_sum = -1;
  for (std::uint32_t v : order) {
    const int sum = int(v % kEdge) +
                    int((v / codec::detail::kStrideY) % kEdge) +
                    int(v / codec::detail::kStrideZ);
    CHECK(sum >= last_sum);
    last_sum = sum;
  }
  // The prior engine's table (zigzag_table.cuh), its first 20 entries as
  // written there as (x, y, z) -- total frequencies 0 through 3 (1 + 3 + 6 +
  // 10), readable here; the hash below pins the other 492.
  const int prior[20][3] = {{0, 0, 0}, {0, 0, 1}, {0, 1, 0}, {1, 0, 0},
                            {0, 0, 2}, {0, 1, 1}, {0, 2, 0}, {1, 0, 1},
                            {1, 1, 0}, {2, 0, 0}, {0, 0, 3}, {0, 1, 2},
                            {0, 2, 1}, {0, 3, 0}, {1, 0, 2}, {1, 1, 1},
                            {1, 2, 0}, {2, 0, 1}, {2, 1, 0}, {3, 0, 0}};
  for (int j = 0; j < 20; ++j) {
    CHECK(order[static_cast<std::size_t>(j)] ==
          codec::detail::voxel_index(std::uint32_t(prior[j][0]),
                                     std::uint32_t(prior[j][1]),
                                     std::uint32_t(prior[j][2])));
  }
  CHECK(order[kVoxelsPerBlock - 1] ==
        codec::detail::voxel_index(kEdge - 1, kEdge - 1, kEdge - 1));
  // The whole table: FNV-1a over its 512 voxel indices (x + 8y + 64z), one
  // index per step, computed from zigzag_table.cuh's kZigZagOrder when the
  // order was ported. Any entry out of place changes it.
  std::uint32_t hash = 2166136261u;
  for (std::uint32_t v : order) {
    hash = (hash ^ v) * 16777619u;
  }
  CHECK(hash == 0x26fa5e49u);
  return 0;
}

}  // namespace

int main() {
  if (params_validate_case() != 0) return 1;
  if (basis_case() != 0) return 1;
  if (zigzag_case() != 0) return 1;
  std::printf("codec params + tables: OK\n");
  return 0;
}
