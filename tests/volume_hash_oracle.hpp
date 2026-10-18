// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

// The tests' host oracle of the volume tier's spatial hash, `computeHashPos`
// in volume/shaders/hash_common.glsl, which alone places a block in the table.
// recon_volume_hash pins it against literals, and chain_refusals
// (device_blocks.hpp) has the device confirm the buckets a fixture picks with
// it.

#include <cstdint>

#include "volumetric_kit/recon/core/math/vector_types.hpp"

namespace vr_test {

// The large primes of the Teschner et al. spatial hash. Unsigned so the
// coordinate multiply wraps with defined behaviour.
inline constexpr std::uint32_t kHashPrimeX = 73856093u;
inline constexpr std::uint32_t kHashPrimeY = 19349669u;
inline constexpr std::uint32_t kHashPrimeZ = 83492791u;

// The bucket in [0, num_buckets) that `block` hashes to: the XOR of prime
// products. Deliberately all-unsigned: the block coordinate can be negative,
// and signed integer overflow is undefined, so each axis is cast to uint32_t
// before the multiply (a straight reinterpretation of the bits, as the GLSL
// does). `num_buckets` > 0, as VoxelGridParams::validate requires; 0 would
// divide by zero.
inline std::uint32_t hash_bucket(volumetric_kit::recon::Vec3i block,
                                 std::int32_t num_buckets) {
  const std::uint32_t hash =
      (static_cast<std::uint32_t>(block.x) * kHashPrimeX) ^
      (static_cast<std::uint32_t>(block.y) * kHashPrimeY) ^
      (static_cast<std::uint32_t>(block.z) * kHashPrimeZ);
  return hash % static_cast<std::uint32_t>(num_buckets);
}

}  // namespace vr_test
