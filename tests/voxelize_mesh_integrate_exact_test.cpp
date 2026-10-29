// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// GPU test: MeshIntegrator's early exit is exact. Its kernel rules triangles
// out by lower bounds and stops scanning a voxel's bin early, and every voxel
// must still write, bit for bit, what measuring its whole bin writes. The
// reference is that brute force (shaders/mesh_integrate_reference.comp): one
// thread per voxel over every triangle of the voxel's bin, through the same
// closest point (triangle_common.glsl), over bins rebuilt by replaying the
// binning's work items through its own GLSL (triangle_candidates.glsl and
// hash_lookup.glsl).
//
// The fixtures aim where an early exit goes wrong: faces tied at a shared
// vertex with opposite windings, where the lowest index must sign the voxel,
// down to a voxel whose distance underflows to zero at every face; duplicated
// triangles wound both ways; parallel sheets closer than the bounds' margin,
// and just past it, so a bound lands just inside the limit; vertices on voxel
// nodes; slivers; triangles metres long, and a sphere hundreds of metres from
// the origin, where rounding grows with the coordinates; bins past the 2,048
// entries the kernel sorts; 2-, 8- and 16-voxel blocks; signed and shell, in
// 8- and 16-voxel bands. Skips where no device is present.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "test_meshes.hpp"
#include "volumetric_kit/core/testing/test_check.hpp"
#include "volumetric_kit/core/vulkan/allocator.hpp"
#include "volumetric_kit/core/vulkan/compute_kernel.hpp"
#include "volumetric_kit/core/vulkan/compute_util.hpp"
#include "volumetric_kit/core/vulkan/device.hpp"
#include "volumetric_kit/core/vulkan/vulkan.hpp"
#include "volumetric_kit/recon/core/math/vector_types.hpp"
#include "volumetric_kit/recon/volume/hash_map_internal.hpp"
#include "volumetric_kit/recon/volume/hash_types.hpp"
#include "volumetric_kit/recon/volume/triangle_candidates.hpp"
#include "volumetric_kit/recon/volume/voxel_block_grid.hpp"
#include "volumetric_kit/recon/volume/voxel_grid.hpp"
#include "volumetric_kit/recon/volume/voxel_hash_map.hpp"
#include "volumetric_kit/recon/voxelize/mesh_integrator.hpp"

#include "grid_readback.hpp"
#include "mesh_integrate_reference_comp.spv.hpp"
#include "volumetric_kit/core/testing/gpu_test.hpp"
#include "volumetric_kit/recon/core/device_requirements.hpp"

namespace vr = volumetric_kit::recon;
namespace vkc = volumetric_kit::core;
namespace vol = volumetric_kit::recon::volume;
namespace vx = volumetric_kit::recon::voxelize;

namespace {

using test_meshes::Mesh;
using vr::Vec3f;

// mesh_integrate.comp's VR_MAX_SORTED: a bin past it has an unsorted tail.
constexpr std::uint32_t kKernelSorted = 2048;
constexpr float kVoxel = 0.005f;

// ---- The reference --------------------------------------------------------

// Mirrors the reference kernel's RefBlock and push constants (scalar).
struct RefBlock {
  vr::Vec3i coord;
  std::uint32_t begin;
  std::uint32_t count;
};
struct RefPush {
  vol::VoxelGridParams grid;
  std::uint32_t tri_count;
  std::uint32_t block_count;
  std::uint32_t pass;
  std::uint32_t mode;
  float shell;
};
static_assert(sizeof(RefBlock) == 20, "RefBlock layout drift");
static_assert(sizeof(RefPush) == 52, "RefPush layout drift");
constexpr std::uint32_t kPassBins = 0;
constexpr std::uint32_t kPassMeasure = 1;

struct Reference {
  vkc::ComputeKernel kernel;
  vkc::DescriptorPool pool;
};

// What the brute force writes for `blocks`, block by block: vpb tsdf values
// then vpb weights each, and the largest bin it measured.
struct RefResult {
  std::vector<float> tsdf;
  std::vector<float> weight;
  std::uint32_t entries = 0;
  std::uint32_t largest_bin = 0;
};

vkc::Result<RefResult> run_reference(const vkc::test::Gpu& gpu, Reference& ref,
                                     vol::VoxelBlockGrid& grid, const Mesh& m,
                                     const std::vector<vol::BlockIndex>& blocks,
                                     const vx::MeshSdfParams& params) {
  const vol::VoxelGridParams& g = grid.grid();
  const std::uint32_t nt = m.triangle_count();
  const auto nb = static_cast<std::uint32_t>(blocks.size());
  const auto vpb = static_cast<std::uint32_t>(g.voxels_per_block);
  const std::uint32_t max_groups =
      gpu.device.caps().limits().maxComputeWorkGroupCount[0];
  const std::uint64_t pairs = std::uint64_t{nb} * nt;
  const std::uint64_t voxels = std::uint64_t{nb} * vpb;
  VKC_ASSIGN(const std::vector<std::uint32_t> offsets,
             vol::triangle_candidate_offsets(g, m.v.data(), m.vertex_count(),
                                             m.i.data(), nt, "reference"));
  const std::uint32_t items = offsets[nt];
  if (pairs > 0xFFFFFFFFull || items == 0 ||
      std::uint64_t{items} + 255 > 0xFFFFFFFFull ||
      (std::uint64_t{items} + 255) / 256 > max_groups ||
      (voxels + 255) / 256 > max_groups) {
    return vkc::Status::invalid_argument("fixture too large for one dispatch");
  }
  std::vector<RefBlock> ref_blocks(nb);
  std::vector<std::int32_t> slot_block(std::size_t(g.num_blocks), -1);
  for (std::uint32_t b = 0; b < nb; ++b) {
    ref_blocks[b] = {blocks[b].coord, 0, 0};
    slot_block[std::size_t(blocks[b].ptr) / vpb] = std::int32_t(b);
  }
  VKC_ASSIGN(vkc::Buffer slots, vkc::test::upload_device_buffer(
                                    gpu.device, gpu.allocator,
                                    slot_block.data(), slot_block.size() * 4));
  VKC_ASSIGN(vkc::Buffer vertices,
             vkc::test::upload_device_buffer(gpu.device, gpu.allocator,
                                             m.v.data(), m.v.size() * 12));
  VKC_ASSIGN(vkc::Buffer indices,
             vkc::test::upload_device_buffer(gpu.device, gpu.allocator,
                                             m.i.data(), m.i.size() * 4));
  VKC_ASSIGN(vkc::Buffer offsets_buf, vkc::test::upload_device_buffer(
                                          gpu.device, gpu.allocator,
                                          offsets.data(), offsets.size() * 4));
  const std::vector<std::uint32_t> zeros((pairs + 31) / 32, 0u);
  VKC_ASSIGN(vkc::Buffer bits,
             vkc::test::upload_device_buffer(gpu.device, gpu.allocator,
                                             zeros.data(), zeros.size() * 4));
  VKC_ASSIGN(vkc::Buffer blocks_buf,
             vkc::test::upload_device_buffer(gpu.device, gpu.allocator,
                                             ref_blocks.data(), nb * 20));
  VKC_ASSIGN(vkc::Buffer tsdf, vkc::device_storage_buffer(
                                   gpu.allocator, VkDeviceSize(voxels) * 4));
  VKC_ASSIGN(vkc::Buffer weight, vkc::device_storage_buffer(
                                     gpu.allocator, VkDeviceSize(voxels) * 4));
  const auto bind = [&](std::uint32_t binding, const vkc::Buffer& buffer) {
    ref.kernel.set.write_storage_buffer(binding, buffer.handle(), 0,
                                        VK_WHOLE_SIZE);
  };
  bind(0, vertices);
  bind(1, indices);
  bind(2, offsets_buf);
  bind(3, blocks_buf);
  bind(4, bits);
  bind(5, tsdf);
  bind(6, weight);
  ref.kernel.set.write_storage_buffer(
      7, vol::VoxelHashMap::Internal::entries_buffer(grid.map()), 0,
      vol::VoxelHashMap::Internal::entries_buffer_size(grid.map()));
  bind(8, slots);
  const float shell = params.mode == vx::MeshSdfMode::Shell
                          ? params.shell_voxels * g.voxel_size
                          : 0.0f;
  const auto mode = static_cast<std::uint32_t>(params.mode);
  RefPush push{g, nt, nb, kPassBins, mode, shell};
  VKC_TRY(vkc::dispatch(gpu.device, ref.kernel, &push, sizeof(push),
                        vkc::group_count(items, 256), max_groups));

  // The bins, each in triangle order.
  VKC_ASSIGN(const std::vector<std::uint32_t> got_bits,
             vkc::test::read_back<std::uint32_t>(gpu.device, gpu.allocator,
                                                 bits, zeros.size()));
  RefResult out;
  std::vector<std::uint32_t> lists;
  for (std::uint32_t b = 0; b < nb; ++b) {
    ref_blocks[b].begin = static_cast<std::uint32_t>(lists.size());
    for (std::uint32_t t = 0; t < nt; ++t) {
      const std::uint64_t k = std::uint64_t{b} * nt + t;
      if ((got_bits[k / 32] >> (k % 32)) & 1u) {
        lists.push_back(t);
      }
    }
    ref_blocks[b].count =
        static_cast<std::uint32_t>(lists.size()) - ref_blocks[b].begin;
    out.largest_bin = std::max(out.largest_bin, ref_blocks[b].count);
  }
  out.entries = static_cast<std::uint32_t>(lists.size());
  if (lists.empty()) lists.push_back(0);
  VKC_ASSIGN(vkc::Buffer lists_buf,
             vkc::test::upload_device_buffer(gpu.device, gpu.allocator,
                                             lists.data(), lists.size() * 4));
  VKC_TRY(
      vkc::test::write_back(gpu.device, gpu.allocator, blocks_buf, ref_blocks));
  bind(4, lists_buf);
  push.pass = kPassMeasure;
  VKC_TRY(vkc::dispatch(gpu.device, ref.kernel, &push, sizeof(push),
                        vkc::group_count(std::uint32_t(voxels), 256),
                        max_groups));
  VKC_ASSIGN(out.tsdf, vkc::test::read_back<float>(gpu.device, gpu.allocator,
                                                   tsdf, voxels));
  VKC_ASSIGN(out.weight, vkc::test::read_back<float>(gpu.device, gpu.allocator,
                                                     weight, voxels));
  return out;
}

// ---- Fixtures --------------------------------------------------------------

struct Rng {
  std::uint64_t s;
  explicit Rng(std::uint64_t seed) : s(seed * 0x9E3779B97F4A7C15ull + 1) {}
  double u() {
    s ^= s << 13;
    s ^= s >> 7;
    s ^= s << 17;
    return double(s >> 11) * (1.0 / 9007199254740992.0);
  }
  double r(double lo, double hi) { return lo + (hi - lo) * u(); }
  Vec3f v(double lo, double hi) {
    return Vec3f(float(r(lo, hi)), float(r(lo, hi)), float(r(lo, hi)));
  }
};

void add_triangle(Mesh& m, Vec3f a, Vec3f b, Vec3f c) {
  const auto base = m.vertex_count();
  m.v.insert(m.v.end(), {a, b, c});
  m.i.insert(m.i.end(), {base, base + 1, base + 2});
}

void append(Mesh& to, const Mesh& from) {
  const auto base = to.vertex_count();
  to.v.insert(to.v.end(), from.v.begin(), from.v.end());
  for (std::uint32_t k : from.i) to.i.push_back(base + k);
}

// Rotations about x, then y, then z, in double.
Vec3f rotate(Vec3f p, double ax, double ay, double az) {
  double x = p.x, y = p.y, z = p.z;
  double c = std::cos(ax), s = std::sin(ax);
  double t = c * y - s * z;
  z = s * y + c * z;
  y = t;
  c = std::cos(ay);
  s = std::sin(ay);
  t = c * x + s * z;
  z = -s * x + c * z;
  x = t;
  c = std::cos(az);
  s = std::sin(az);
  t = c * x - s * y;
  y = s * x + c * y;
  x = t;
  return Vec3f(float(x), float(y), float(z));
}

// A fan of k faces around `apex`, its rim a circle of radius r at height h
// along `axis` (an x, y or z unit vector), starting at angle a0. Face f is
// (apex, rim f, rim f+1), or the other way round where flip(f) says so.
template <class Flip>
Mesh fan(Vec3f apex, int axis, float h, float r, int k, double a0, Flip flip) {
  Mesh m;
  m.v.push_back(apex);
  for (int f = 0; f < k; ++f) {
    const double a = a0 + 2.0 * 3.14159265358979323846 * f / k;
    float e[3] = {0.0f, 0.0f, 0.0f};
    e[axis] = h;
    e[(axis + 1) % 3] = float(r * std::cos(a));
    e[(axis + 2) % 3] = float(r * std::sin(a));
    m.v.push_back(apex + Vec3f(e[0], e[1], e[2]));
  }
  for (int f = 0; f < k; ++f) {
    const auto b = static_cast<std::uint32_t>(1 + f);
    const auto c = static_cast<std::uint32_t>(1 + (f + 1) % k);
    if (flip(f)) {
      m.i.insert(m.i.end(), {0u, c, b});
    } else {
      m.i.insert(m.i.end(), {0u, b, c});
    }
  }
  return m;
}

// An icosphere, wound outward.
Mesh icosphere(Vec3f centre, float radius, int subdivisions) {
  const double t = (1.0 + std::sqrt(5.0)) / 2.0;
  std::vector<std::array<double, 3>> p = {{-1, t, 0},  {1, t, 0},   {-1, -t, 0},
                                          {1, -t, 0},  {0, -1, t},  {0, 1, t},
                                          {0, -1, -t}, {0, 1, -t},  {t, 0, -1},
                                          {t, 0, 1},   {-t, 0, -1}, {-t, 0, 1}};
  std::vector<std::array<std::uint32_t, 3>> f = {
      {0, 11, 5}, {0, 5, 1},  {0, 1, 7},   {0, 7, 10}, {0, 10, 11},
      {1, 5, 9},  {5, 11, 4}, {11, 10, 2}, {10, 7, 6}, {7, 1, 8},
      {3, 9, 4},  {3, 4, 2},  {3, 2, 6},   {3, 6, 8},  {3, 8, 9},
      {4, 9, 5},  {2, 4, 11}, {6, 2, 10},  {8, 6, 7},  {9, 8, 1}};
  const auto unit = [](std::array<double, 3> q) {
    const double l = std::sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2]);
    return std::array<double, 3>{q[0] / l, q[1] / l, q[2] / l};
  };
  for (auto& q : p) q = unit(q);
  for (int s = 0; s < subdivisions; ++s) {
    std::map<std::pair<std::uint32_t, std::uint32_t>, std::uint32_t> mids;
    const auto mid = [&](std::uint32_t a, std::uint32_t b) {
      const auto key = std::make_pair(std::min(a, b), std::max(a, b));
      const auto it = mids.find(key);
      if (it != mids.end()) return it->second;
      p.push_back(
          unit({p[a][0] + p[b][0], p[a][1] + p[b][1], p[a][2] + p[b][2]}));
      return mids[key] = static_cast<std::uint32_t>(p.size() - 1);
    };
    std::vector<std::array<std::uint32_t, 3>> next;
    for (const auto& q : f) {
      const std::uint32_t a = mid(q[0], q[1]), b = mid(q[1], q[2]),
                          c = mid(q[2], q[0]);
      next.push_back({q[0], a, c});
      next.push_back({q[1], b, a});
      next.push_back({q[2], c, b});
      next.push_back({a, b, c});
    }
    f = std::move(next);
  }
  Mesh m;
  for (const auto& q : p) {
    m.v.push_back(centre + Vec3f(float(radius * q[0]), float(radius * q[1]),
                                 float(radius * q[2])));
  }
  for (const auto& q : f) m.i.insert(m.i.end(), {q[0], q[1], q[2]});
  return m;
}

struct Fixture {
  std::string name;
  Mesh mesh;
  vx::MeshSdfMode mode = vx::MeshSdfMode::Signed;
  int block_size = 8;
  float trunc = 0.08f;  // a 16-voxel band
  // Check the bins reach past the kernel's sort.
  bool past_sorted = false;
  // Check the voxel at the origin holds -0, observed.
  bool minus_zero_at_origin = false;
};

std::vector<Fixture> fixtures() {
  std::vector<Fixture> out;

  // Faces tied at their shared apex, a voxel node, with alternating windings:
  // past the apex the lowest index signs the voxel. Then the apex off the
  // lattice.
  for (const Vec3f off : {Vec3f(0.0f), Vec3f(0.0013f, -0.0007f, 0.0021f)}) {
    Fixture f;
    f.name = off.x == 0.0f ? "fan at a node" : "fan off the nodes";
    f.mesh = fan(Vec3f(20 * kVoxel, -12 * kVoxel, 7 * kVoxel) + off, 2, -0.03f,
                 0.05f, 48, 0.0, [](int k) { return k % 2 == 1; });
    out.push_back(std::move(f));
  }
  // The apex 1e-30 m from the voxel at the origin: that voxel's distance
  // underflows to zero at every face. Face 0 alone is wound so that it signs
  // the voxel -0, and it lies opposite the block's centre, so the kernel
  // sorts it behind the first chunk.
  {
    Fixture f;
    f.name = "fan at zero distance";
    const double step = 2.0 * 3.14159265358979323846 / 128;
    f.mesh = fan(Vec3f(1e-30f, 0.0f, 0.0f), 0, 0.03f, 0.05f, 128,
                 1.25 * 3.14159265358979323846 - 0.5 * step,
                 [](int k) { return k != 0; });
    f.minus_zero_at_origin = true;
    out.push_back(std::move(f));
  }
  // Each triangle four times: reversed, and its corners rotated -- exact ties
  // with opposite signs.
  {
    Fixture f;
    f.name = "duplicates";
    Rng r(7);
    for (int t = 0; t < 300; ++t) {
      const Vec3f c = r.v(-0.15, 0.15);
      const Vec3f a = c + r.v(-0.02, 0.02);
      const Vec3f b = c + r.v(-0.02, 0.02);
      const Vec3f d = c + r.v(-0.02, 0.02);
      add_triangle(f.mesh, a, b, d);
      add_triangle(f.mesh, a, d, b);
      add_triangle(f.mesh, b, d, a);
      add_triangle(f.mesh, d, b, a);
    }
    out.push_back(std::move(f));
  }
  // Two tilted, divided sheets wound opposite ways, delta apart: coincident,
  // inside the bounds' margin (about 16 um here), just past it and well past.
  for (const double delta : {0.0, 1e-5, 2e-5, 1e-4}) {
    Fixture f;
    char name[64];
    std::snprintf(name, sizeof name, "sheets %g um apart", delta * 1e6);
    f.name = name;
    const int n = 12;
    const float half = 0.12f;
    for (int s = 0; s < 2; ++s) {
      for (int y = 0; y < n; ++y) {
        for (int x = 0; x < n; ++x) {
          const auto at = [&](int ix, int iy) {
            const Vec3f q(-half + 2 * half * float(ix) / n,
                          -half + 2 * half * float(iy) / n,
                          s == 0 ? 0.0f : float(delta));
            return rotate(q, 0.31, -0.17, 0.53) +
                   Vec3f(0.021f, -0.013f, 0.008f);
          };
          const Vec3f a = at(x, y), b = at(x + 1, y), c = at(x + 1, y + 1),
                      d = at(x, y + 1);
          if (s == 0) {
            add_triangle(f.mesh, a, b, c);
            add_triangle(f.mesh, a, c, d);
          } else {
            add_triangle(f.mesh, a, c, b);
            add_triangle(f.mesh, a, d, c);
          }
        }
      }
    }
    out.push_back(std::move(f));
  }
  // Corners on voxel nodes, wound at random: zero distances and exact ties.
  const auto lattice = [](int span, std::uint64_t seed) {
    Mesh m;
    Rng r(seed);
    for (int t = 0; t < 3000; ++t) {
      const int bx = int(r.r(-span, span)), by = int(r.r(-span, span)),
                bz = int(r.r(-span, span));
      const auto node = [&]() {
        return Vec3f(float(bx + int(r.r(-4, 5))) * kVoxel,
                     float(by + int(r.r(-4, 5))) * kVoxel,
                     float(bz + int(r.r(-4, 5))) * kVoxel);
      };
      const Vec3f a = node(), b = node(), c = node();
      const Vec3f n = vr::cross(b - a, c - a);
      if (!(vr::dot(n, n) > 0.0f)) continue;
      if (r.u() < 0.5) {
        add_triangle(m, a, b, c);
      } else {
        add_triangle(m, a, c, b);
      }
    }
    return m;
  };
  {
    Fixture f;
    f.name = "lattice soup";
    f.mesh = lattice(12, 12);
    out.push_back(std::move(f));
    Fixture g;
    g.name = "lattice soup, 8-voxel band";
    g.mesh = lattice(40, 11);
    g.trunc = 0.04f;
    out.push_back(std::move(g));
  }
  // Slivers and needles: near-collinear corners, aspect ratios to 1e-7.
  {
    Fixture f;
    f.name = "slivers";
    Rng r(23);
    for (int t = 0; t < 2500; ++t) {
      const Vec3f a = r.v(-0.12, 0.12);
      const Vec3f d = r.v(-1, 1);
      const Vec3f e = r.v(-1, 1);
      const float len = float(r.r(0.002, 0.08));
      const double aspect = std::pow(10.0, -r.r(1, 7));
      const Vec3f b = a + d * (len / vr::length(d));
      const Vec3f c = a + d * (float(r.r(0, 1)) * len / vr::length(d)) +
                      e * float(aspect * len / vr::length(e));
      const Vec3f n = vr::cross(b - a, c - a);
      if (!(vr::dot(n, n) > 0.0f)) continue;
      if (r.u() < 0.5) {
        add_triangle(f.mesh, a, b, c);
      } else {
        add_triangle(f.mesh, c, b, a);
      }
    }
    out.push_back(std::move(f));
  }
  // Triangles metres long through the blocks near the origin, each with a
  // twin 10 um off it wound the other way: the closest point's rounding
  // grows with the corners, far from the block.
  for (const float length : {2.0f, 20.0f}) {
    Fixture f;
    f.name = std::to_string(int(length)) + " m triangles";
    f.trunc = 0.04f;
    for (int s = 0; s < 2; ++s) {
      const float dz = s ? 1e-5f : 0.0f;
      const Vec3f a = rotate(Vec3f(-length / 2, -0.004f, dz), 7e-4, -1.1e-3, 0);
      const Vec3f b = rotate(Vec3f(length / 2, -0.003f, dz), 7e-4, -1.1e-3, 0);
      const Vec3f c = rotate(Vec3f(0.013f, 0.011f, dz), 7e-4, -1.1e-3, 0);
      if (s) {
        add_triangle(f.mesh, a, c, b);
      } else {
        add_triangle(f.mesh, a, b, c);
      }
    }
    out.push_back(std::move(f));
  }
  // Hundreds of metres from the origin, signed and shell.
  for (const vx::MeshSdfMode mode :
       {vx::MeshSdfMode::Signed, vx::MeshSdfMode::Shell}) {
    Fixture f;
    f.name =
        mode == vx::MeshSdfMode::Signed ? "far sphere" : "far sphere, shell";
    f.mesh = icosphere(Vec3f(312.37f, -207.11f, 151.93f), 0.21f, 3);
    f.mode = mode;
    out.push_back(std::move(f));
  }
  // Other block sizes: row-order voxels, and 1 round or 16.
  {
    Fixture f;
    f.name = "sphere, 2-voxel blocks";
    f.mesh = icosphere(Vec3f(0.0137f, -0.021f, 0.0093f), 0.05f, 3);
    f.block_size = 2;
    f.trunc = 0.02f;
    out.push_back(std::move(f));
    Fixture g;
    g.name = "sphere, 16-voxel blocks";
    g.mesh = icosphere(Vec3f(0.0137f, -0.021f, 0.0093f), 0.15f, 3);
    g.block_size = 16;
    out.push_back(std::move(g));
  }
  // A small dense sphere: every bin holds every triangle, past the sort.
  {
    Fixture f;
    f.name = "dense sphere";
    f.mesh = icosphere(Vec3f(0.0031f, 0.0017f, -0.0029f), 0.03f, 4);
    f.past_sorted = true;
    out.push_back(std::move(f));
  }
  // Shell: the first fan and the duplicates together.
  {
    Fixture f;
    f.name = "fan and duplicates, shell";
    f.mode = vx::MeshSdfMode::Shell;
    append(f.mesh, out[0].mesh);
    append(f.mesh, out[3].mesh);
    out.push_back(std::move(f));
  }
  return out;
}

// A grid for `f`'s block size and band: 8,192 blocks, or 2^22 voxels where
// that is fewer. The 20 m triangles take 5,741 8-voxel blocks.
vkc::Result<vol::VoxelBlockGrid> make_grid(const vkc::test::Gpu& gpu,
                                           const Fixture& f) {
  vol::VoxelGridParams gp{};
  gp.voxel_size = kVoxel;
  gp.block_size = f.block_size;
  gp.voxels_per_block = f.block_size * f.block_size * f.block_size;
  gp.trunc_dist = f.trunc;
  gp.bucket_size = 8;
  gp.num_buckets = std::min(1024, (1 << 22) / gp.voxels_per_block / 8);
  gp.num_blocks = gp.bucket_size * gp.num_buckets;
  gp.max_chain = 128;
  const vol::AttributeSpec attrs[] = {{vol::kTsdfAttribute, sizeof(float)},
                                      {vol::kWeightAttribute, sizeof(float)}};
  return vol::VoxelBlockGrid::create(gpu.device, gpu.allocator, gp, attrs, 2);
}

// Convert `f` into `grid`, cleared, run the reference over the blocks
// written, and compare every voxel's bits.
int check_fixture(vkc::test::GpuContext& gpu, vx::MeshIntegrator& integ,
                  Reference& ref, vol::VoxelBlockGrid& grid, const Fixture& f) {
  const vol::VoxelGridParams& gp = grid.grid();
  const Mesh& m = f.mesh;
  CHECK(grid.clear().ok());
  vkc::Result<std::uint32_t> failed = grid.map().allocate_from_triangles(
      m.v.data(), m.vertex_count(), m.i.data(), m.triangle_count());
  CHECK(failed.ok() && failed.value() == 0);
  const vx::MeshSdfParams params{f.mode, 1.5f};
  vkc::Result<vx::MeshIntegrateStats> stats =
      integ.integrate(grid, m.v.data(), m.vertex_count(), m.i.data(),
                      m.triangle_count(), params);
  if (!stats.ok()) {
    std::fprintf(stderr, "%s: %s\n", f.name.c_str(),
                 stats.status().message().c_str());
    return 1;
  }
  vkc::Result<std::vector<vol::BlockIndex>> active_r =
      grid.map().compact_active_blocks();
  CHECK(active_r.ok());
  const std::vector<vol::BlockIndex>& active = active_r.value();
  CHECK(stats.value().blocks == active.size());

  vkc::Result<RefResult> want_r =
      run_reference(gpu, ref, grid, m, active, params);
  if (!want_r.ok()) {
    std::fprintf(stderr, "%s: reference: %s\n", f.name.c_str(),
                 want_r.status().message().c_str());
    return 1;
  }
  const RefResult& want = want_r.value();
  // The reference's bins are the integrator's.
  if (want.entries != stats.value().bin_entries) {
    std::fprintf(stderr,
                 "%s: the reference bins %u entries, the integrator %u\n",
                 f.name.c_str(), want.entries, stats.value().bin_entries);
    return 1;
  }
  if (f.past_sorted) CHECK(want.largest_bin > kKernelSorted);

  // Both block by block, in `active`'s order.
  const std::vector<float> tsdf =
      vr_test::read_blocks<float>(gpu, grid, vol::kTsdfAttribute, active)
          .value();
  const std::vector<float> weight =
      vr_test::read_blocks<float>(gpu, grid, vol::kWeightAttribute, active)
          .value();
  const auto vpb = static_cast<std::size_t>(gp.voxels_per_block);
  std::size_t differ = 0;
  std::size_t observed = 0;
  for (std::size_t b = 0; b < active.size(); ++b) {
    for (std::size_t k = 0; k < vpb; ++k) {
      const std::size_t at = b * vpb + k;
      observed += want.weight[at] != 0.0f ? 1 : 0;
      if (std::memcmp(&tsdf[at], &want.tsdf[at], 4) == 0 &&
          std::memcmp(&weight[at], &want.weight[at], 4) == 0) {
        continue;
      }
      if (differ++ < 4) {
        const int bs = gp.block_size;
        std::fprintf(stderr,
                     "FAIL %s: block (%d, %d, %d) voxel (%zu, %zu, %zu): "
                     "tsdf %.9g weight %g, the whole bin gives %.9g, %g\n",
                     f.name.c_str(), active[b].coord.x, active[b].coord.y,
                     active[b].coord.z, k % bs, (k / bs) % bs, k / (bs * bs),
                     double(tsdf[at]), double(weight[at]),
                     double(want.tsdf[at]), double(want.weight[at]));
      }
    }
  }
  if (f.minus_zero_at_origin) {
    bool found = false;
    for (std::size_t b = 0; b < active.size(); ++b) {
      if (active[b].coord == vr::Vec3i(0)) {
        found = true;
        CHECK(std::signbit(tsdf[b * vpb]));
        CHECK(tsdf[b * vpb] == 0.0f);
        CHECK(weight[b * vpb] == 1.0f);
      }
    }
    CHECK(found);
  }
  std::printf(
      "%-28s %5u triangles %5zu blocks %8u bin entries (largest %5u) "
      "%8zu observed: %zu differ\n",
      f.name.c_str(), m.triangle_count(), active.size(), want.entries,
      want.largest_bin, observed, differ);
  CHECK(observed > 0);
  return differ == 0 ? 0 : 1;
}

int gpu_main(vkc::test::GpuContext& gpu) {
  Reference ref;
  {
    VkPushConstantRange range{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(RefPush)};
    vkc::KernelSetBuilder builder(gpu.device);
    CHECK(builder
              .add(ref.kernel, "mesh_integrate_reference",
                   vr_mesh_integrate_reference_comp_spv,
                   vr_mesh_integrate_reference_comp_spv_size, 9, &range)
              .ok());
    vkc::Result<vkc::DescriptorPool> pool = builder.build();
    CHECK(pool.ok());
    ref.pool = std::move(pool).value();
  }
  vkc::Result<vx::MeshIntegrator> integ_r =
      vx::MeshIntegrator::create(gpu.device, gpu.allocator);
  CHECK(integ_r.ok());
  vx::MeshIntegrator integ = std::move(integ_r).value();

  // One grid per block size and band, cleared for each fixture.
  std::map<std::pair<int, float>, vol::VoxelBlockGrid> grids;
  int failures = 0;
  for (const Fixture& f : fixtures()) {
    const std::pair<int, float> shape{f.block_size, f.trunc};
    auto it = grids.find(shape);
    if (it == grids.end()) {
      vkc::Result<vol::VoxelBlockGrid> made = make_grid(gpu, f);
      CHECK(made.ok());
      it = grids.emplace(shape, std::move(made).value()).first;
    }
    failures += check_fixture(gpu, integ, ref, it->second, f);
  }
  CHECK(failures == 0);
  std::printf("voxelize_mesh_integrate_exact: OK\n");
  return 0;
}

}  // namespace

int main() {
  return vkc::test::run_on_gpu(vr::device_requirements(), gpu_main);
}
