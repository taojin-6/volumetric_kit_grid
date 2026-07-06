// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// Spatial-hash tests: determinism, bucket range, three locked known values (one
// per prime), and a weak distribution sanity check. Pure host math -- no device
// -- so it always runs.

#include <cstdint>
#include <cstdio>

#include "volumetric_kit/recon/core/math/vector_types.hpp"
#include "volumetric_kit/recon/volume/hash.hpp"

namespace vr = volumetric_kit::recon;
namespace vol = volumetric_kit::recon::volume;

#define CHECK(cond)                                                        \
  do {                                                                     \
    if (!(cond)) {                                                         \
      std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
      return 1;                                                            \
    }                                                                      \
  } while (0)

int main() {
  constexpr std::int32_t kBuckets = 30000;

  // --- Determinism: same coordinate -> same bucket.
  const vr::Vec3i coord(12, -7, 3);
  CHECK(vol::hash_bucket(coord, kBuckets) == vol::hash_bucket(coord, kBuckets));

  // --- Range: always in [0, num_buckets), including negative coordinates.
  for (int x = -50; x <= 50; ++x) {
    for (int y = -5; y <= 5; ++y) {
      const std::uint32_t bucket =
          vol::hash_bucket(vr::Vec3i(x, y, x - y), kBuckets);
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
  CHECK(vol::hash_bucket(vr::Vec3i(0, 0, 0), kBuckets) == 0u);
  CHECK(vol::hash_bucket(vr::Vec3i(1, 0, 0), kBuckets) == 26093u);
  CHECK(vol::hash_bucket(vr::Vec3i(0, 1, 0), kBuckets) == 29669u);
  CHECK(vol::hash_bucket(vr::Vec3i(0, 0, 1), kBuckets) == 2791u);

  // --- Weak distribution sanity: a diagonal line of coordinates must not all
  // collide into one bucket.
  const std::uint32_t origin = vol::hash_bucket(vr::Vec3i(0, 0, 0), kBuckets);
  bool any_different = false;
  for (int i = 1; i < 64; ++i) {
    if (vol::hash_bucket(vr::Vec3i(i, i, i), kBuckets) != origin) {
      any_different = true;
      break;
    }
  }
  CHECK(any_different);

  std::printf("recon volume hash test passed\n");
  return 0;
}
