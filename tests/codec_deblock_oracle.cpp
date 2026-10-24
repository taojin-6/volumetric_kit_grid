// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include "codec_deblock_oracle.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <map>
#include <tuple>
#include <vector>

namespace codec_deblock_oracle {
namespace {

namespace vr = volumetric_kit::recon;
namespace vol = volumetric_kit::recon::volume;

constexpr int kB = 8;
constexpr std::size_t kV = 512;
constexpr float kSaturated = 1.0f - 1e-5f;  // the study's kSaturated

// The voxel at coordinate t along `axis` and (u, w) along the other two, in
// increasing order (the study's line_local).
std::size_t line_local(int axis, int t, int u, int w) {
  switch (axis) {
    case 0:
      return std::size_t(t + kB * (u + kB * w));
    case 1:
      return std::size_t(u + kB * (t + kB * w));
    default:
      return std::size_t(u + kB * (w + kB * t));
  }
}

// Blocks by coordinate.
class Blocks {
 public:
  explicit Blocks(const std::vector<vr::Vec3i>& coords) {
    for (std::size_t i = 0; i < coords.size(); ++i) {
      at_[key(coords[i])] = std::int64_t(i);
    }
  }
  // The block one step along `axis` (s = -1 or 1) from @p c, or -1.
  std::int64_t find(const vr::Vec3i& c, int axis, int s) const {
    vr::Vec3i n = c;
    (axis == 0 ? n.x : axis == 1 ? n.y : n.z) += s;
    const auto it = at_.find(key(n));
    return it == at_.end() ? -1 : it->second;
  }

 private:
  static std::tuple<int, int, int> key(const vr::Vec3i& c) {
    return {c.x, c.y, c.z};
  }
  std::map<std::tuple<int, int, int>, std::int64_t> at_;
};

bool same_bits(float a, float b) { return std::memcmp(&a, &b, sizeof a) == 0; }

// The study's extrapolate: the least-squares line through (i + 0.5, v_i),
// i < m, at 0.
double extrapolate(const double* v, int m) {
  double tm = 0, vm = 0;
  for (int i = 0; i < m; ++i) {
    tm += i + 0.5;
    vm += v[i];
  }
  tm /= m;
  vm /= m;
  double num = 0, den = 0;
  for (int i = 0; i < m; ++i) {
    const double dt = (i + 0.5) - tm;
    num += dt * (v[i] - vm);
    den += dt * dt;
  }
  return vm - num / den * tm;
}

// The study's tent: voxel i of n a side takes (n - i - 0.5) / (2n).
double tent(int i, int n) { return (n - i - 0.5) / (2.0 * n); }

// One axis of the line fit (the study's line_pass and its linfit filter),
// across the faces whose blocks are both @p filtered.
void line_pass(codec_fixture::Snapshot& g, const Blocks& blocks,
               const std::vector<std::uint8_t>& filtered, int axis, double clip,
               float trunc) {
  constexpr int kN = 2, kM = 3;
  const codec_fixture::Snapshot src = g;  // the pass's snapshot
  for (std::size_t b = 0; b < src.coords.size(); ++b) {
    const std::int64_t nb = blocks.find(src.coords[b], axis, 1);
    if (nb < 0 || !filtered[b] || !filtered[std::size_t(nb)]) continue;
    for (int w = 0; w < kB; ++w) {
      for (int u = 0; u < kB; ++u) {
        std::size_t pi[kM], qi[kM];
        float pf[kM], qf[kM];
        double p[kM], q[kM];
        bool ok = true;
        for (int i = 0; i < kM; ++i) {
          pi[i] = b * kV + line_local(axis, kB - 1 - i, u, w);
          qi[i] = std::size_t(nb) * kV + line_local(axis, i, u, w);
          ok = ok && src.weight[pi[i]] >= vol::kObservedWeight &&
               src.weight[qi[i]] >= vol::kObservedWeight;
          pf[i] = src.tsdf[pi[i]] / trunc;
          qf[i] = src.tsdf[qi[i]] / trunc;
          p[i] = pf[i];
          q[i] = qf[i];
        }
        if (!ok) continue;
        bool saturated = false;
        for (int i = 0; i < kM; ++i) {
          saturated = saturated || std::fabs(pf[i]) >= kSaturated ||
                      std::fabs(qf[i]) >= kSaturated;
        }
        if (saturated) continue;
        const double delta =
            std::clamp(extrapolate(q, kM) - extrapolate(p, kM), -clip, clip);
        for (int i = 0; i < kN; ++i) {
          const double d = delta * tent(i, kN);
          if (d == 0) continue;
          const float np = float(std::clamp(p[i] + d, -1.0, 1.0));
          const float nq = float(std::clamp(q[i] - d, -1.0, 1.0));
          if (!same_bits(np, pf[i])) g.tsdf[pi[i]] = np * trunc;
          if (!same_bits(nq, qf[i])) g.tsdf[qi[i]] = nq * trunc;
        }
      }
    }
  }
}

// The Gaussian's three passes (the study's gauss_pass, r 1), on the field in
// float, over the @p filtered blocks' voxels, the others kept as they are;
// then the voxels whose bits changed written back in metres.
void gauss(codec_fixture::Snapshot& g, const Blocks& blocks,
           const std::vector<std::uint8_t>& filtered, double g1, double clip,
           float trunc) {
  const std::size_t n = g.tsdf.size();
  std::vector<std::uint8_t> obs(n);
  std::vector<float> cur(n, 0.0f);
  for (std::size_t i = 0; i < n; ++i) {
    obs[i] = g.weight[i] >= vol::kObservedWeight ? 1 : 0;
    if (obs[i]) cur[i] = g.tsdf[i] / trunc;
  }
  const std::vector<float> in = cur;
  for (int axis = 0; axis < 3; ++axis) {
    std::vector<float> dst = cur;
    for (std::size_t b = 0; b < g.coords.size(); ++b) {
      if (!filtered[b]) continue;
      const std::int64_t nbs[3] = {blocks.find(g.coords[b], axis, -1),
                                   std::int64_t(b),
                                   blocks.find(g.coords[b], axis, 1)};
      for (int w = 0; w < kB; ++w) {
        for (int u = 0; u < kB; ++u) {
          for (int t = 0; t < kB; ++t) {
            const std::size_t self = b * kV + line_local(axis, t, u, w);
            if (!obs[self]) continue;
            double sw = 0, sv = 0;
            for (int k = -1; k <= 1; ++k) {
              const int tt = t + k;
              const int o = tt < 0 ? -1 : (tt >= kB ? 1 : 0);
              if (nbs[o + 1] < 0) continue;
              const std::size_t at = std::size_t(nbs[o + 1]) * kV +
                                     line_local(axis, tt - kB * o, u, w);
              if (!obs[at]) continue;
              const double gk = k == 0 ? 1.0 : g1;
              sw += gk;
              sv += gk * double(cur[at]);
            }
            const double v = cur[self];
            const double out = v + std::clamp(sv / sw - v, -clip, clip);
            dst[self] = float(std::clamp(out, -1.0, 1.0));
          }
        }
      }
    }
    cur.swap(dst);
  }
  for (std::size_t i = 0; i < n; ++i) {
    if (obs[i] && !same_bits(cur[i], in[i])) g.tsdf[i] = cur[i] * trunc;
  }
}

}  // namespace

float study_step_scale(
    const volumetric_kit::recon::codec::CodecParams& params) {
  const double phi0 = std::sqrt(1.0 / 8.0);
  const double phi1 =
      std::sqrt(2.0 / 8.0) * std::cos(3.14159265358979323846 / 16.0);
  double t = params.quantization_scale * params.quantization_weights[0] * phi0 *
             phi0 * phi0;
  const std::uint32_t s1[3] = {64u, 8u, 1u};
  for (std::uint32_t j = 0; j < 3; ++j) {
    if (params.coefficient_count < 2 + j) break;
    t = std::max(t, double(params.quantization_scale) *
                        params.quantization_weights[s1[j]] * phi1 * phi0 *
                        phi0);
  }
  return float(t);
}

bool masked(const codec_fixture::Snapshot& grid, std::size_t b) {
  for (std::size_t l = 0; l < kV; ++l) {
    if (grid.weight[b * kV + l] < vol::kObservedWeight) return true;
  }
  return false;
}

void apply(codec_fixture::Snapshot& grid, const d::DeblockPasses& passes,
           float trunc_dist, bool fallback) {
  const Blocks blocks(grid.coords);
  std::vector<std::uint8_t> filtered(grid.coords.size(), 1);
  for (std::size_t b = 0; fallback && b < grid.coords.size(); ++b) {
    filtered[b] = masked(grid, b) ? 0 : 1;
  }
  if (passes.linfit) {
    for (int axis = 0; axis < 3; ++axis) {
      line_pass(grid, blocks, filtered, axis, double(passes.line_clip),
                trunc_dist);
    }
  }
  if (passes.gauss) {
    gauss(grid, blocks, filtered, double(passes.gauss_g1),
          double(passes.gauss_clip), trunc_dist);
  }
}

}  // namespace codec_deblock_oracle
