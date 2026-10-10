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
#include "fnv1a.hpp"
#include "test_check.hpp"
#include "volumetric_kit/recon/codec/codec_params.hpp"

namespace codec = volumetric_kit::recon::codec;
using codec::kVoxelsPerBlock;
using codec::detail::kBasisSize;
using codec::detail::kEdge;

namespace {

int params_validate_case() {
  codec::CodecParams p;
  CHECK(p.validate().ok());
  CHECK(p.quantization_scale == 0.2f);
  for (float weight : p.quantization_weights) CHECK(weight == 1.0f);
  for (std::uint32_t k : {0u, 513u}) {
    p.coefficient_count = k;
    CHECK(!p.validate().ok());
  }
  for (std::uint32_t k : {1u, kVoxelsPerBlock}) {
    p.coefficient_count = k;
    CHECK(p.validate().ok());
  }

  // The literal is the formula's float, and both effective-step bounds are
  // inclusive. Even the largest possible coefficient fits the int16 clamp.
  CHECK(codec::kMinStep ==
        static_cast<float>(std::sqrt(double(kVoxelsPerBlock)) /
                           codec::kMaxQuantizedMagnitude));
  CHECK(std::nearbyint(std::sqrt(double(kVoxelsPerBlock)) /
                       double(codec::kMinStep)) <=
        double(codec::kMaxQuantizedMagnitude));
  CHECK(std::sqrt(double(kVoxelsPerBlock)) < 0.5 * double(codec::kMaxStep));
  for (float step : {codec::kMinStep, codec::kMaxStep}) {
    p.quantization_scale = step;
    CHECK(p.validate().ok());
  }
  for (float scale :
       {0.0f, -1.0f, std::numeric_limits<float>::quiet_NaN(),
        std::numeric_limits<float>::infinity(), codec::kMinStep * 0.99f,
        std::nextafter(codec::kMaxStep, 128.0f), 1e38f}) {
    p.quantization_scale = scale;
    CHECK(!p.validate().ok());
  }

  // Validate the kept table entries; no special DC rule exists. Bad products
  // include overflow and underflow, and a subnormal factor is refused.
  p.coefficient_count = kVoxelsPerBlock;
  p.quantization_scale = 0.2f;
  for (std::size_t i : {std::size_t(0), std::size_t(1), std::size_t(8),
                        std::size_t(64), std::size_t(511)}) {
    for (float weight : {0.0f, -1.0f, std::numeric_limits<float>::quiet_NaN(),
                         std::numeric_limits<float>::infinity(),
                         std::numeric_limits<float>::denorm_min(), 1e38f}) {
      p.quantization_weights[i] = weight;
      CHECK(!p.validate().ok());
    }
    p.quantization_weights[i] = 1.0f;
  }
  p.quantization_scale = 1e30f;
  p.quantization_weights.fill(1e30f);
  CHECK(!p.validate().ok());  // finite factors, infinite product
  p.quantization_scale = 1e-30f;
  p.quantization_weights.fill(1e-30f);
  CHECK(!p.validate().ok());  // finite factors, zero product

  // A subnormal factor is refused even when the product is a normal step, so
  // validity cannot depend on a host's denormals-as-zero mode.
  p.quantization_scale = 1e-40f;
  p.quantization_weights.fill(1e38f);
  CHECK(!p.validate().ok());
  p.quantization_scale = 1e38f;
  p.quantization_weights.fill(1e-40f);
  CHECK(!p.validate().ok());

  // Only kept weights count: K = 1 keeps DC alone.
  p.coefficient_count = 1;
  p.quantization_scale = 0.2f;
  p.quantization_weights.fill(1.0f);
  p.quantization_weights[511] = 0.0f;
  CHECK(p.validate().ok());
  p.quantization_weights[0] = 0.0f;
  CHECK(!p.validate().ok());
  p.quantization_weights.fill(1.0f);
  p.coefficient_count = kVoxelsPerBlock;

  // Only the product is bounded: rescaling a table is permitted, and DC
  // weight 1 is a preset convention rather than an API restriction.
  p.quantization_scale = 128.0f;
  p.quantization_weights.fill(0.25f);
  CHECK(p.validate().ok());
  p.quantization_scale = codec::kMinStep * 0.5f;
  p.quantization_weights.fill(2.0f);
  CHECK(p.validate().ok());
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
  CHECK(vr_test::fnv1a(order) == 0x26fa5e49u);
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
