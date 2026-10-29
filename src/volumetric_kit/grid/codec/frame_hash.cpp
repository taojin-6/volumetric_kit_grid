// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include "frame_hash.hpp"

#include <algorithm>

namespace volumetric_kit::grid::codec::detail {
namespace {

constexpr std::uint64_t kPrime1 = 0x9E3779B185EBCA87ull;
constexpr std::uint64_t kPrime2 = 0xC2B2AE3D27D4EB4Full;
constexpr std::uint64_t kPrime3 = 0x165667B19E3779F9ull;
constexpr std::uint64_t kPrime4 = 0x85EBCA77C2B2AE63ull;
constexpr std::uint64_t kPrime5 = 0x27D4EB2F165667C5ull;
constexpr std::size_t kStripe = 32;

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

// XXH64 over bytes that arrive in runs: the four lanes take each 32-byte
// stripe whichever runs it spans, and the bytes short of a stripe wait.
class Xxh64 {
 public:
  void update(const std::uint8_t* p, std::size_t size) {
    total_ += size;
    if (buffered_ != 0) {
      const std::size_t take = std::min(size, kStripe - buffered_);
      std::copy(p, p + take, buffer_ + buffered_);
      buffered_ += take;
      p += take;
      size -= take;
      if (buffered_ < kStripe) return;
      stripe(buffer_);
      buffered_ = 0;
    }
    for (; size >= kStripe; p += kStripe, size -= kStripe) stripe(p);
    std::copy(p, p + size, buffer_);
    buffered_ = size;
  }

  std::uint64_t digest() const {
    std::uint64_t acc = 0;
    if (total_ >= kStripe) {
      acc = rotl(v_[0], 1) + rotl(v_[1], 7) + rotl(v_[2], 12) + rotl(v_[3], 18);
      for (const std::uint64_t v : v_) acc = merge_lane(acc, v);
    } else {
      acc = kPrime5;
    }
    acc += total_;
    const std::uint8_t* p = buffer_;
    const std::uint8_t* const end = buffer_ + buffered_;
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

 private:
  void stripe(const std::uint8_t* p) {
    for (int lane = 0; lane < 4; ++lane) {
      v_[lane] = lane_round(v_[lane], read64(p + 8 * lane));
    }
  }

  std::uint64_t v_[4] = {kPrime1 + kPrime2, kPrime2, 0, 0 - kPrime1};
  std::uint8_t buffer_[kStripe] = {};
  std::size_t buffered_ = 0;
  std::uint64_t total_ = 0;
};

}  // namespace

std::uint64_t frame_hash(const std::uint8_t* data, std::size_t size) {
  return frame_hash(data, size, nullptr, 0);
}

std::uint64_t frame_hash(const std::uint8_t* head, std::size_t head_size,
                         const std::uint8_t* payload,
                         std::size_t payload_size) {
  Xxh64 h;
  if (head_size != 0) h.update(head, head_size);
  if (payload_size != 0) h.update(payload, payload_size);
  return h.digest();
}

}  // namespace volumetric_kit::grid::codec::detail
