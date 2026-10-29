// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// The tests' spatial-hash oracle (volume_hash_oracle.hpp): bucket range, locked
// known values (one per prime, one that combines them), and a weak distribution
// sanity check.
// Pure host math -- no device -- so it always runs.

#include <cstdint>
#include <cstdio>

#include "volume_hash_oracle.hpp"
#include "volumetric_kit/core/testing/test_check.hpp"
#include "volumetric_kit/grid/base/math/vector_types.hpp"

namespace vgrid = volumetric_kit::grid;

int main() {
  constexpr std::int32_t kBuckets = 30000;

  // --- Range: always in [0, num_buckets), including negative coordinates.
  for (int x = -50; x <= 50; ++x) {
    for (int y = -5; y <= 5; ++y) {
      const std::uint32_t bucket =
          vgrid_test::hash_bucket(vgrid::Vec3i(x, y, x - y), kBuckets);
      CHECK(bucket < static_cast<std::uint32_t>(kBuckets));
    }
  }

  // --- Known values: hard literals that pin each prime independently
  // (N = 30000). A self-referential `== kHashPrimeX % N` moves with an edited
  // constant and catches nothing; these fail on a typo in any prime -- or a
  // Y<->Z swap. With two axes zero, each case isolates one prime:
  //   hash(1,0,0) = kHashPrimeX % N = 73856093 % 30000 = 26093
  //   hash(0,1,0) = kHashPrimeY % N = 19349669 % 30000 = 29669
  //   hash(0,0,1) = kHashPrimeZ % N = 83492791 % 30000 =  2791
  CHECK(vgrid_test::hash_bucket(vgrid::Vec3i(0, 0, 0), kBuckets) == 0u);
  CHECK(vgrid_test::hash_bucket(vgrid::Vec3i(1, 0, 0), kBuckets) == 26093u);
  CHECK(vgrid_test::hash_bucket(vgrid::Vec3i(0, 1, 0), kBuckets) == 29669u);
  CHECK(vgrid_test::hash_bucket(vgrid::Vec3i(0, 0, 1), kBuckets) == 2791u);
  // All three axes pin how the products combine: their XOR is 28855119, so
  // 25119; a sum would give 28553.
  CHECK(vgrid_test::hash_bucket(vgrid::Vec3i(1, 1, 1), kBuckets) == 25119u);

  // --- Weak distribution sanity: a diagonal line of coordinates must not all
  // collide into one bucket.
  const std::uint32_t origin =
      vgrid_test::hash_bucket(vgrid::Vec3i(0, 0, 0), kBuckets);
  bool any_different = false;
  for (int i = 1; i < 64; ++i) {
    if (vgrid_test::hash_bucket(vgrid::Vec3i(i, i, i), kBuckets) != origin) {
      any_different = true;
      break;
    }
  }
  CHECK(any_different);

  std::printf("grid volume hash test passed\n");
  return 0;
}
