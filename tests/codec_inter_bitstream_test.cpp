// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// Host-only tests for P-frames in the v3 frame: the reference hash's known
// answers. CPU-only, so these always run.

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "frame_hash.hpp"
#include "test_check.hpp"

namespace d = volumetric_kit::recon::codec::detail;

namespace {

using Bytes = std::vector<std::uint8_t>;

// --- The reference hash. ----------------------------------------------------

// xxHash's own sanity input (cli/xsum_sanity_check.c): each byte the top
// byte of a generator that starts at PRIME32_1 and multiplies by PRIME64_1.
Bytes xxhash_sanity_buffer(std::size_t n) {
  Bytes out(n);
  std::uint64_t gen = 2654435761u;
  for (std::uint8_t& b : out) {
    b = static_cast<std::uint8_t>(gen >> 56);
    gen *= 11400714785074694797ull;
  }
  return out;
}

std::uint64_t hash_of(const std::string& s) {
  return d::frame_hash(reinterpret_cast<const std::uint8_t*>(s.data()),
                       s.size());
}

// Known answers of XXH64, seed 0: the short inputs, and xxHash's own vectors
// over its sanity buffer, 222 bytes of which run the four-lane loop six
// times and then every tail (8, 4 and 1 bytes). Checked against the
// reference implementation (python-xxhash 4.0.1, xxHash 0.8.3).
int hash_case() {
  CHECK(d::frame_hash(nullptr, 0) == 0xEF46DB3751D8E999ull);
  CHECK(hash_of("") == 0xEF46DB3751D8E999ull);
  CHECK(hash_of("a") == 0xD24EC4F1A98C6E5Bull);
  CHECK(hash_of("abc") == 0x44BC2CF5AD770999ull);
  CHECK(hash_of("The quick brown fox jumps over the lazy dog") ==
        0x0B242D361FDA71BCull);
  const Bytes sanity = xxhash_sanity_buffer(222);
  CHECK(d::frame_hash(sanity.data(), 14) == 0x8282DCC4994E35C8ull);
  CHECK(d::frame_hash(sanity.data(), 222) == 0xB641AE8CB691C174ull);
  // Bytes read with shifts: any alignment gives the same value.
  Bytes shifted(sanity.size() + 1);
  std::memcpy(shifted.data() + 1, sanity.data(), sanity.size());
  CHECK(d::frame_hash(shifted.data() + 1, 222) == 0xB641AE8CB691C174ull);
  return 0;
}

}  // namespace

int main() {
  if (hash_case() != 0) return 1;
  std::printf("codec inter bitstream: OK\n");
  return 0;
}
