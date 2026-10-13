// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include "frame_hash.hpp"

namespace volumetric_kit::recon::codec::detail {
namespace {

constexpr std::uint64_t kPrime1 = 0x9E3779B185EBCA87ull;
constexpr std::uint64_t kPrime2 = 0xC2B2AE3D27D4EB4Full;
constexpr std::uint64_t kPrime3 = 0x165667B19E3779F9ull;
constexpr std::uint64_t kPrime4 = 0x85EBCA77C2B2AE63ull;
constexpr std::uint64_t kPrime5 = 0x27D4EB2F165667C5ull;

std::uint64_t rotl(std::uint64_t x, int r) {
  return (x << r) | (x >> (64 - r));
}

// Little-endian, a byte at a time.
std::uint64_t read64(const std::uint8_t* p) {
  std::uint64_t v = 0;
  for (int i = 7; i >= 0; --i) v = (v << 8) | p[i];
  return v;
}

std::uint64_t read32(const std::uint8_t* p) {
  return std::uint64_t(p[0]) | std::uint64_t(p[1]) << 8 |
         std::uint64_t(p[2]) << 16 | std::uint64_t(p[3]) << 24;
}

std::uint64_t lane_round(std::uint64_t acc, std::uint64_t lane) {
  return rotl(acc + lane * kPrime2, 31) * kPrime1;
}

std::uint64_t merge_lane(std::uint64_t acc, std::uint64_t lane) {
  return (acc ^ lane_round(0, lane)) * kPrime1 + kPrime4;
}

}  // namespace

std::uint64_t frame_hash(const std::uint8_t* data, std::size_t size) {
  const std::uint8_t* p = data;
  const std::uint8_t* const end = data + size;
  std::uint64_t acc = 0;
  if (size >= 32) {
    // Four lanes over each 32-byte stripe.
    std::uint64_t v1 = kPrime1 + kPrime2;
    std::uint64_t v2 = kPrime2;
    std::uint64_t v3 = 0;
    std::uint64_t v4 = 0 - kPrime1;
    for (; end - p >= 32; p += 32) {
      v1 = lane_round(v1, read64(p));
      v2 = lane_round(v2, read64(p + 8));
      v3 = lane_round(v3, read64(p + 16));
      v4 = lane_round(v4, read64(p + 24));
    }
    acc = rotl(v1, 1) + rotl(v2, 7) + rotl(v3, 12) + rotl(v4, 18);
    acc = merge_lane(acc, v1);
    acc = merge_lane(acc, v2);
    acc = merge_lane(acc, v3);
    acc = merge_lane(acc, v4);
  } else {
    acc = kPrime5;
  }
  acc += std::uint64_t(size);
  for (; end - p >= 8; p += 8) {
    acc = rotl(acc ^ lane_round(0, read64(p)), 27) * kPrime1 + kPrime4;
  }
  if (end - p >= 4) {
    acc = rotl(acc ^ (read32(p) * kPrime1), 23) * kPrime2 + kPrime3;
    p += 4;
  }
  for (; p < end; ++p) {
    acc = rotl(acc ^ (*p * kPrime5), 11) * kPrime1;
  }
  // The avalanche.
  acc ^= acc >> 33;
  acc *= kPrime2;
  acc ^= acc >> 29;
  acc *= kPrime3;
  acc ^= acc >> 32;
  return acc;
}

}  // namespace volumetric_kit::recon::codec::detail
