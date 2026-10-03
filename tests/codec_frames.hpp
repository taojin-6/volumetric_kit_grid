// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

// Synthetic intra frames for the codec's frame tests: the host writer's
// round trips and the device writer's byte comparison draw the same frames.

#include <cstddef>
#include <cstdint>
#include <set>

#include "bitstream.hpp"
#include "volumetric_kit/recon/codec/codec_params.hpp"

namespace codec_frames {

namespace vr = volumetric_kit::recon;
namespace codec = volumetric_kit::recon::codec;
namespace d = volumetric_kit::recon::codec::detail;

struct Lcg {
  std::uint64_t state;
  std::uint32_t next() {
    state = state * 6364136223846793005ull + 1442695040888963407ull;
    return static_cast<std::uint32_t>(state >> 32);
  }
  std::uint32_t below(std::uint32_t n) { return next() % n; }
};

// A partial mask shaped like a fused band's edge: the voxels on one side of a
// plane through the block, so its planes and lines repeat, run full or empty,
// or cut across -- every symbol the mask code has.
inline void slab_mask(Lcg& rng, std::uint32_t* mask) {
  const int a = int(rng.below(7)) - 3;
  const int b = int(rng.below(7)) - 3;
  const int c = int(rng.below(7)) - 3;
  const int d = int(rng.below(40)) - 20;
  for (std::uint32_t w = 0; w < codec::kMaskWordsPerBlock; ++w) {
    mask[w] = 0;
  }
  for (std::uint32_t v = 0; v < codec::kVoxelsPerBlock; ++v) {
    const int x = int(v % 8), y = int(v / 8 % 8), z = int(v / 64);
    if (a * x + b * y + c * z < d) {
      mask[v / 32] |= 1u << (v % 32);
    }
  }
}

// A frame shaped like a real one: blocks in runs along x with jumps between
// them, masks mostly full with some empty and some partial -- noise, or a
// slab's edge -- coefficients small and sparser at higher indices, and the
// odd extreme value.
inline d::IntraFrame make_frame(std::size_t n, std::uint32_t k,
                                std::uint64_t seed) {
  Lcg rng{seed};
  auto less = [](const vr::Vec3i& a, const vr::Vec3i& b) {
    return d::coord_less(a, b);
  };
  std::set<vr::Vec3i, decltype(less)> coords(less);
  while (coords.size() < n) {
    // One draw per statement: the order a function's arguments are evaluated
    // in is unspecified (GCC and Clang disagree), and a fixture must be the
    // same frame on every compiler.
    const int sx = int(rng.below(200)) - 100;
    const int sy = int(rng.below(60)) - 30;
    const int sz = int(rng.below(60)) - 30;
    const vr::Vec3i start(sx, sy, sz);
    const std::uint32_t run = 1 + rng.below(12);
    for (std::uint32_t i = 0; i < run && coords.size() < n; ++i) {
      coords.insert(vr::Vec3i(start.x + int(i), start.y, start.z));
    }
  }
  d::IntraFrame f;
  f.voxel_size = 0.005f;
  f.coords.assign(coords.begin(), coords.end());
  f.blocks.params.coefficient_count = k;
  f.blocks.trunc_dist = 0.04f;
  for (std::size_t i = 0; i < n; ++i) {
    const std::uint32_t cls = rng.below(10);
    std::uint32_t mask[codec::kMaskWordsPerBlock];
    if (cls < 8) {
      for (std::uint32_t& w : mask) {
        w = cls < 6 ? ~0u : cls < 7 ? 0u : rng.next();
      }
    } else {
      slab_mask(rng, mask);
    }
    f.blocks.masks.insert(f.blocks.masks.end(), mask,
                          mask + codec::kMaskWordsPerBlock);
    for (std::uint32_t j = 0; j < k; ++j) {
      std::int32_t v = 0;
      const std::uint32_t spread = 1 + 64 / (1 + j);
      if (rng.below(1 + j / 4) == 0) {
        v = int(rng.below(2 * spread + 1)) - int(spread);
      }
      if (rng.below(500) == 0) {
        v = rng.below(2) == 0 ? codec::kMaxQuantizedMagnitude
                              : -codec::kMaxQuantizedMagnitude;
      }
      f.blocks.coefficients.push_back(static_cast<std::int16_t>(v));
    }
  }
  return f;
}

}  // namespace codec_frames
