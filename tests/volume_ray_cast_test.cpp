// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// volume::RayCaster over analytic scenes whose tsdf, weight and 1-byte
// attributes the test writes (grid_readback.hpp's writer). A plane hits
// everywhere at its exact depth and normal; a sphere within a tenth of a
// voxel and 2 degrees, missing where its analytic view misses but within a
// pixel of the silhouette. A ray misses that starts behind a surface, meets
// a back band from unobserved space, or whose surface lies outside its depth
// range, and views from the count up keep their bytes. An unallocated span
// costs a step a block. The allow set and the weight floor pick the voxels
// that answer, a trilinear sample only when its 8 corners do. A K-view cast
// equals K single casts bit for bit; a recast reads cameras its batch moved;
// a grown grid is cast with its new blocks; the ring of sets bounds a
// batch's casts; and the refusals. VGRID_RAY_CAST_TIMING=1 adds a timing
// case, which prints and asserts nothing. Skips without a device.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string_view>
#include <vector>

#include "volumetric_kit/core/base/result.hpp"
#include "volumetric_kit/core/base/stage_metrics.hpp"
#include "volumetric_kit/core/testing/buffer_readback.hpp"
#include "volumetric_kit/core/testing/gpu_test.hpp"
#include "volumetric_kit/core/testing/test_check.hpp"
#include "volumetric_kit/core/vulkan/allocator.hpp"
#include "volumetric_kit/core/vulkan/command_batch.hpp"
#include "volumetric_kit/core/vulkan/compute_util.hpp"
#include "volumetric_kit/core/vulkan/device.hpp"
#include "volumetric_kit/core/vulkan/gpu_timer.hpp"
#include "volumetric_kit/grid/base/device_requirements.hpp"
#include "volumetric_kit/grid/base/math/vector_types.hpp"
#include "volumetric_kit/grid/testing/device_blocks.hpp"
#include "volumetric_kit/grid/testing/grid_readback.hpp"
#include "volumetric_kit/grid/testing/sphere_scene.hpp"
#include "volumetric_kit/grid/volume/ray_caster.hpp"
#include "volumetric_kit/grid/volume/voxel_block_grid.hpp"

namespace vgrid = volumetric_kit::grid;
namespace vkc = volumetric_kit::core;
namespace vol = volumetric_kit::grid::volume;

namespace {

using vgrid::Mat4f;
using vgrid::Vec3f;
using vgrid::Vec3i;
using vgrid::Vec4f;

// The test grids' two 1-byte attributes: a square's index, and one mark.
constexpr std::string_view kSource = "source";
constexpr std::string_view kMark = "mark";

vol::VoxelGridParams grid_params(float voxel, float trunc,
                                 std::int32_t buckets) {
  vol::VoxelGridParams g{};
  g.voxel_size = voxel;
  g.block_size = 8;
  g.voxels_per_block = 512;
  g.trunc_dist = trunc;
  g.bucket_size = 8;
  g.num_buckets = buckets;
  g.num_blocks = 8 * buckets;
  g.max_chain = 64;
  return g;
}

vkc::Result<vol::VoxelBlockGrid> make_grid(const vkc::test::Gpu& gpu,
                                           const vol::VoxelGridParams& g) {
  const vol::AttributeSpec attrs[] = {{vol::kTsdfAttribute, 4},
                                      {vol::kWeightAttribute, 4},
                                      {kSource, 1},
                                      {kMark, 1}};
  return vol::VoxelBlockGrid::create(gpu.device, gpu.allocator, g, attrs, 4);
}

// What a test writes at a voxel.
struct Voxel {
  float tsdf = 0.0f;
  float weight = 0.0f;
  std::uint8_t source = 0;
  std::uint8_t mark = 0;
};

// A fusion's voxel at signed distance d: clamped to the band and observed
// to its back edge, as an integrator leaves it.
Voxel fused(float d, float trunc, float weight = 1.0f) {
  Voxel v;
  v.tsdf = std::clamp(d, -trunc, trunc);
  v.weight = d >= -trunc ? weight : 0.0f;
  return v;
}

// The blocks of the world box [lo, hi] (block-aligned or not) holding a voxel
// within the band of `sdf`, an SDF of slope at most 1: those whose centre is
// within the band and the block's half diagonal.
template <typename Sdf>
std::vector<vol::BlockIndex> band_blocks(const vol::VoxelGridParams& g,
                                         Vec3f lo, Vec3f hi, Sdf sdf) {
  const float extent = g.voxel_size * static_cast<float>(g.block_size);
  const Vec3i b0(glm::floor(lo / extent));
  const Vec3i b1(glm::floor(hi / extent));
  const float reach = g.trunc_dist + 0.5f * std::sqrt(3.0f) * extent;
  std::vector<vol::BlockIndex> out;
  for (int z = b0.z; z <= b1.z; ++z) {
    for (int y = b0.y; y <= b1.y; ++y) {
      for (int x = b0.x; x <= b1.x; ++x) {
        const Vec3f centre =
            (Vec3f(Vec3i(x, y, z) * g.block_size) + 3.5f) * g.voxel_size;
        if (std::fabs(sdf(centre)) <= reach) {
          vol::BlockIndex b{};
          b.coord = Vec3i(x, y, z);
          out.push_back(b);
        }
      }
    }
  }
  return out;
}

// Allocate `add`, then write every active block's voxels from `field`
// (world centre, voxel coordinate) -> Voxel; a slot no block holds is zero.
template <typename Field>
vkc::Status fill(const vkc::test::Gpu& gpu, vol::VoxelBlockGrid& grid,
                 const std::vector<vol::BlockIndex>& add, Field field) {
  if (!add.empty()) {
    VKC_ASSIGN(const std::uint32_t failed,
               vgrid_test::allocate_blocks(gpu, grid.map(), add));
    if (failed != 0) {
      return vkc::Status::invalid_argument("fill: blocks refused");
    }
  }
  VKC_ASSIGN(const std::vector<vol::BlockIndex> active,
             grid.map().compact_active_blocks());
  VKC_ASSIGN(const vol::AttributeView view,
             grid.attribute(vol::kTsdfAttribute));
  const auto n = static_cast<std::size_t>(view.element_count);
  std::vector<float> tsdf(n, 0.0f);
  std::vector<float> weight(n, 0.0f);
  std::vector<std::uint8_t> source(n, 0);
  std::vector<std::uint8_t> mark(n, 0);
  const int bs = grid.grid().block_size;
  for (const vol::BlockIndex& b : active) {
    for (int k = 0; k < grid.grid().voxels_per_block; ++k) {
      const Vec3i voxel =
          b.coord * bs + Vec3i(k % bs, (k / bs) % bs, k / (bs * bs));
      const Voxel v = field(Vec3f(voxel) * grid.grid().voxel_size, voxel);
      const std::size_t i = static_cast<std::size_t>(b.ptr) + k;
      tsdf[i] = v.tsdf;
      weight[i] = v.weight;
      source[i] = v.source;
      mark[i] = v.mark;
    }
  }
  VKC_TRY(vgrid_test::write_attribute(gpu.device, gpu.allocator, grid,
                                      vol::kTsdfAttribute, tsdf));
  VKC_TRY(vgrid_test::write_attribute(gpu.device, gpu.allocator, grid,
                                      vol::kWeightAttribute, weight));
  VKC_TRY(vgrid_test::write_attribute(gpu.device, gpu.allocator, grid, kSource,
                                      source));
  return vgrid_test::write_attribute(gpu.device, gpu.allocator, grid, kMark,
                                     mark);
}

// A camera at `eye` looking along +Z of `pose`, its principal point at the
// image centre.
vol::DepthPinhole pinhole(std::uint32_t w, std::uint32_t h, float f,
                          const Mat4f& pose = Mat4f(1.0f)) {
  vol::DepthPinhole c{};
  c.fx = f;
  c.fy = f;
  c.cx = 0.5f * static_cast<float>(w - 1);
  c.cy = 0.5f * static_cast<float>(h - 1);
  c.min_depth = 0.1f;
  c.max_depth = 5.0f;
  c.width = w;
  c.height = h;
  c.cam_to_world = pose;
  return c;
}

// A pose at `eye` looking at `target`, y down as OpenCV's.
Mat4f look_at(Vec3f eye, Vec3f target) {
  const Vec3f z = glm::normalize(target - eye);
  const Vec3f x = glm::normalize(glm::cross(Vec3f(0, 1, 0), z));
  const Vec3f y = glm::cross(z, x);
  return Mat4f(Vec4f(x, 0), Vec4f(y, 0), Vec4f(z, 0), Vec4f(eye, 1));
}

vkc::Result<vkc::Buffer> camera_buffer(
    const vkc::test::Gpu& gpu, const std::vector<vol::DepthPinhole>& cams) {
  return vkc::test::upload_device_buffer(
      gpu.device, gpu.allocator, cams.data(),
      cams.size() * sizeof(vol::DepthPinhole));
}

// A view's contents on the host.
struct Image {
  std::vector<float> depth;
  std::vector<float> normal;  // three a pixel
};

vkc::Result<Image> read_view(const vkc::test::Gpu& gpu,
                             const vol::ModelView& view) {
  const std::size_t n = std::size_t(view.width) * view.height * view.views;
  Image out;
  VKC_ASSIGN(out.depth, vkc::test::read_back<float>(gpu.device, gpu.allocator,
                                                    view.depth, n));
  VKC_ASSIGN(out.normal, vkc::test::read_back<float>(gpu.device, gpu.allocator,
                                                     view.normal, 3 * n));
  return out;
}

// Cast `cams` into a fresh stacked view in one batch, and read it back.
vkc::Result<Image> cast(const vkc::test::Gpu& gpu, vol::RayCaster& caster,
                        const vol::VoxelBlockGrid& grid,
                        const std::vector<vol::DepthPinhole>& cams,
                        std::uint32_t w, std::uint32_t h,
                        const vol::CastFilter& filter = {}) {
  VKC_ASSIGN(const vkc::Buffer buf, camera_buffer(gpu, cams));
  const auto count = static_cast<std::uint32_t>(cams.size());
  VKC_ASSIGN(vol::ModelView view, caster.make_view(w, h, count));
  vkc::CommandBatch batch(gpu.device, gpu.allocator);
  VKC_TRY(caster.record_cast(batch, grid, buf, count, filter, view));
  VKC_TRY(batch.submit());
  return read_view(gpu, view);
}

std::size_t hits(const std::vector<float>& depth) {
  return static_cast<std::size_t>(
      std::count_if(depth.begin(), depth.end(), [](float d) { return d > 0; }));
}

// A plane at depth z facing a camera at the origin, over x and y in
// [lo, hi] metres, at 1 cm voxels and a 4 cm band.
constexpr float kVoxel = 0.01f;
constexpr float kTrunc = 0.04f;

vkc::Status plane(const vkc::test::Gpu& gpu, vol::VoxelBlockGrid& grid, float z,
                  float lo, float hi, float weight = 1.0f) {
  const auto sdf = [z](Vec3f p) { return z - p.z; };
  const float t = grid.grid().trunc_dist;
  return fill(gpu, grid,
              band_blocks(grid.grid(), Vec3f(lo, lo, z - 2 * t),
                          Vec3f(hi, hi, z + 2 * t), sdf),
              [&](Vec3f p, Vec3i) { return fused(sdf(p), t, weight); });
}

// 1. Shapes. A plane's trilinear field is its own SDF, so its depth and
// normal are exact but for rounding. The sphere's are within a tenth of a
// voxel and 2 degrees, and it misses where its analytic view misses, on the
// caster's rays through integer pixel coordinates; within a pixel of the
// silhouette, where the rays graze it, nothing is held.
int shapes(vkc::test::GpuContext& gpu, vol::RayCaster& caster) {
  {
    auto grid = make_grid(gpu, grid_params(kVoxel, kTrunc, 512));
    CHECK(grid.ok() && plane(gpu, *grid, 1.0f, -0.8f, 0.8f).ok());
    auto img = cast(gpu, caster, *grid, {pinhole(256, 192, 200.0f)}, 256, 192);
    CHECK(img.ok());
    float worst_depth = 0.0f;
    float worst_normal = 0.0f;
    for (std::size_t i = 0; i < img->depth.size(); ++i) {
      CHECK(img->depth[i] > 0.0f);
      worst_depth = std::max(worst_depth, std::fabs(img->depth[i] - 1.0f));
      const Vec3f n(img->normal[3 * i], img->normal[3 * i + 1],
                    img->normal[3 * i + 2]);
      worst_normal =
          std::max(worst_normal, glm::length(n - Vec3f(0.0f, 0.0f, -1.0f)));
    }
    std::printf("  plane: depth within %.2e m, normal within %.2e\n",
                double(worst_depth), double(worst_normal));
    CHECK(worst_depth <= 1e-4f && worst_normal <= 1e-4f);
  }
  {
    const float voxel = 0.005f;
    const float trunc = 0.02f;
    const float r = vgrid_test::kSphereRadius;
    auto grid = make_grid(gpu, grid_params(voxel, trunc, 1024));
    const auto sdf = [r](Vec3f p) { return glm::length(p) - r; };
    CHECK(grid.ok());
    CHECK(fill(gpu, *grid,
               band_blocks(grid->grid(), Vec3f(-0.35f), Vec3f(0.35f), sdf),
               [&](Vec3f p, Vec3i) { return fused(sdf(p), trunc); })
              .ok());
    const std::uint32_t w = vgrid_test::kSphereWidth;
    const std::uint32_t h = vgrid_test::kSphereHeight;
    double worst_depth = 0.0;
    double worst_angle = 0.0;
    std::size_t compared = 0;
    for (int c = 0; c < 3; ++c) {
      const vol::DepthPinhole cam =
          vgrid_test::sphere_view<vol::DepthPinhole>(c).cam;
      auto img = cast(gpu, caster, *grid, {cam}, w, h);
      CHECK(img.ok());
      // The analytic view on the caster's rays.
      const vgrid::Vec3d eye(cam.cam_to_world[3]);
      const vgrid::Vec3d ax(cam.cam_to_world[0]);
      const vgrid::Vec3d ay(cam.cam_to_world[1]);
      const vgrid::Vec3d az(cam.cam_to_world[2]);
      std::vector<double> depth(std::size_t(w) * h, 0.0);
      std::vector<vgrid::Vec3d> normal(depth.size());
      for (std::uint32_t v = 0; v < h; ++v) {
        for (std::uint32_t u = 0; u < w; ++u) {
          const vgrid::Vec3d ray =
              glm::normalize((double(u) - cam.cx) / cam.fx * ax +
                             (double(v) - cam.cy) / cam.fy * ay + az);
          const double b = glm::dot(eye, ray);
          const double disc = b * b - glm::dot(eye, eye) + double(r) * r;
          if (disc <= 0.0) continue;
          const double t = -b - std::sqrt(disc);
          depth[v * w + u] = t * glm::dot(ray, az);
          normal[v * w + u] = glm::normalize(eye + t * ray);
        }
      }
      for (std::uint32_t v = 0; v < h; ++v) {
        for (std::uint32_t u = 0; u < w; ++u) {
          const std::size_t i = v * w + u;
          const bool want = depth[i] > 0.0;
          bool rim = false;
          for (int dv = -1; dv <= 1; ++dv) {
            for (int du = -1; du <= 1; ++du) {
              const int uu = std::clamp(int(u) + du, 0, int(w) - 1);
              const int vv = std::clamp(int(v) + dv, 0, int(h) - 1);
              rim = rim || ((depth[std::size_t(vv) * w + uu] > 0.0) != want);
            }
          }
          // Within a pixel of the silhouette a ray grazes the sphere, and
          // neither its hit nor its depth is held.
          if (rim) continue;
          CHECK((img->depth[i] > 0.0f) == want);
          if (!want) continue;
          ++compared;
          worst_depth =
              std::max(worst_depth, std::fabs(img->depth[i] - depth[i]));
          const vgrid::Vec3d n(img->normal[3 * i], img->normal[3 * i + 1],
                               img->normal[3 * i + 2]);
          worst_angle = std::max(
              worst_angle,
              std::acos(std::clamp(glm::dot(n, normal[i]), -1.0, 1.0)));
        }
      }
    }
    worst_angle *= 180.0 / 3.14159265358979;
    std::printf(
        "  sphere: %zu hits, depth within %.3f voxel, normal within "
        "%.2f deg\n",
        compared, worst_depth / voxel, worst_angle);
    CHECK(compared > 3000);
    CHECK(worst_depth <= 0.1 * voxel && worst_angle <= 2.0);
  }
  return 0;
}

// 2. Misses: a ray that starts behind the surface, one that meets a negative
// slab behind an unobserved front, and one whose surface lies outside its
// depth range. Views from the count up keep their bytes.
int misses(vkc::test::GpuContext& gpu, vol::RayCaster& caster) {
  auto grid = make_grid(gpu, grid_params(kVoxel, kTrunc, 512));
  CHECK(grid.ok() && plane(gpu, *grid, 1.0f, -0.4f, 0.4f).ok());
  const auto none = [&](vol::DepthPinhole cam) -> bool {
    auto img = cast(gpu, caster, *grid, {cam}, 64, 48);
    return img.ok() && hits(img->depth) == 0;
  };
  vol::DepthPinhole cam = pinhole(64, 48, 100.0f);
  {
    auto img = cast(gpu, caster, *grid, {cam}, 64, 48);
    CHECK(img.ok() && hits(img->depth) == img->depth.size());
  }
  cam.min_depth = 1.02f;  // inside the negative half of the band
  CHECK(none(cam));
  cam.min_depth = 1.2f;  // past the surface and its band
  CHECK(none(cam));
  cam.min_depth = 0.1f;
  cam.max_depth = 0.95f;  // short of the band
  CHECK(none(cam));
  cam.max_depth = 0.995f;  // samples the band, the surface past it
  CHECK(none(cam));
  cam.max_depth = 5.0f;

  // A negative slab behind a front of weight 0: a back band entered from
  // unobserved space.
  {
    auto slab = make_grid(gpu, grid_params(kVoxel, kTrunc, 512));
    const auto sdf = [](Vec3f p) { return 1.0f - p.z; };
    CHECK(slab.ok());
    CHECK(fill(gpu, *slab,
               band_blocks(slab->grid(), Vec3f(-0.4f, -0.4f, 0.9f),
                           Vec3f(0.4f, 0.4f, 1.1f), sdf),
               [&](Vec3f p, Vec3i) {
                 Voxel v = fused(sdf(p), kTrunc);
                 if (sdf(p) > 0.0f) v.weight = 0.0f;
                 return v;
               })
              .ok());
    auto img = cast(gpu, caster, *slab, {cam}, 64, 48);
    CHECK(img.ok() && hits(img->depth) == 0);
  }

  // Views from the count up keep their sentinel bytes.
  {
    auto view = caster.make_view(64, 48, 3);
    auto cams = camera_buffer(gpu, {cam, cam});
    CHECK(view.ok() && cams.ok());
    const VkDeviceSize pixels = 64 * 48;
    const std::uint32_t sentinel = 0x7FC0BEEFu;
    vkc::CommandBatch batch(gpu.device, gpu.allocator);
    CHECK(batch.fill(view->depth, 0, view->depth.size(), sentinel).ok());
    CHECK(batch.fill(view->normal, 0, view->normal.size(), sentinel).ok());
    CHECK(caster.record_cast(batch, *grid, *cams, 2, {}, *view).ok());
    CHECK(batch.submit().ok());
    auto img = read_view(gpu, *view);
    CHECK(img.ok());
    for (std::size_t i = 0; i < img->depth.size(); ++i) {
      std::uint32_t d;
      std::memcpy(&d, &img->depth[i], 4);
      CHECK((d == sentinel) == (i >= 2 * pixels));
      for (int k = 0; k < 3; ++k) {
        std::uint32_t n;
        std::memcpy(&n, &img->normal[3 * i + k], 4);
        CHECK((n == sentinel) == (i >= 2 * pixels));
      }
    }
  }
  return 0;
}

// 3. Block skip: 3 m of unallocated space before a plane, at 1 cm voxels and
// a 4 cm band, within 64 steps. Free steps of 0.8 bands would need 94.
int block_skip(vkc::test::GpuContext& gpu) {
  vol::RayCasterConfig config;
  config.max_steps = 64;
  auto caster = vol::RayCaster::create(gpu.device, gpu.allocator, config);
  auto grid = make_grid(gpu, grid_params(kVoxel, kTrunc, 256));
  CHECK(caster.ok() && grid.ok());
  CHECK(plane(gpu, *grid, 3.2f, -0.32f, 0.32f).ok());
  auto img = cast(gpu, **caster, *grid, {pinhole(32, 32, 200.0f)}, 32, 32);
  CHECK(img.ok() && hits(img->depth) == img->depth.size());
  for (const float d : img->depth) CHECK(std::fabs(d - 3.2f) <= 1e-4f);
  return 0;
}

// 4. The allow set and the weight floor, on a plane in 16 x 16 squares of 8
// x 8 voxels, the `source` byte of a square its index, the camera square on
// to it with 4 pixels a voxel. A pixel hits only when the 4 voxel columns of
// its hit's trilinear sample are allowed, and does hit when its hit lies 1.5
// voxels inside an allowed square.
constexpr float kSquaresZ = 4.003f;
constexpr std::uint32_t kSquaresSide = 480;

int square_of(int column) { return (column + 64) / 8; }

std::uint8_t byte_at(Vec3i v) {
  return static_cast<std::uint8_t>(square_of(v.y) * 16 + square_of(v.x));
}

// Pixel (u, v)'s hit on the squares' plane, in voxel units.
vgrid::Vec2f squares_hit(std::uint32_t u, std::uint32_t v) {
  const float c = 0.5f * (kSquaresSide - 1);
  return {(float(u) - c) * 0.25f, (float(v) - c) * 0.25f};
}

// Whether the trilinear sample at hit `g` has every column allowed by
// `allowed(byte)`; and whether `g` lies 1.5 voxels inside an allowed square.
template <typename Allowed>
void classify(vgrid::Vec2f g, Allowed allowed, bool& may, bool& must) {
  const int x0 = int(std::floor(g.x));
  const int y0 = int(std::floor(g.y));
  may = true;
  for (int k = 0; k < 4; ++k) {
    may = may && allowed(byte_at(Vec3i(x0 + (k & 1), y0 + (k >> 1), 0)));
  }
  const auto inside = [](float s) {
    const float in = s + 64.0f - 8.0f * std::floor((s + 64.0f) / 8.0f);
    return in >= 1.5f && in <= 7.0f - 1.5f;
  };
  must = inside(g.x) && inside(g.y) && allowed(byte_at(Vec3i(x0, y0, 0)));
}

int allow_set(vkc::test::GpuContext& gpu, vol::RayCaster& caster) {
  auto grid = make_grid(gpu, grid_params(kVoxel, kTrunc, 256));
  CHECK(grid.ok());
  const auto sdf = [](Vec3f p) { return kSquaresZ - p.z; };
  // The mark: one voxel at the surface, (0, 0, 400), in front of it.
  const Vec3i marked(0, 0, 400);
  CHECK(fill(gpu, *grid,
             band_blocks(grid->grid(), Vec3f(-0.64f, -0.64f, 3.9f),
                         Vec3f(0.63f, 0.63f, 4.1f), sdf),
             [&](Vec3f p, Vec3i v) {
               Voxel out = fused(sdf(p), kTrunc);
               out.source = byte_at(v);
               out.mark = v == marked ? 1 : 0;
               return out;
             })
            .ok());
  // 1601.2 px at 4.003 m: 0.25 voxel a pixel, so no hit lies within an
  // eighth of a voxel of a column's edge.
  const vol::DepthPinhole cam = pinhole(kSquaresSide, kSquaresSide, 1601.2f);
  const auto run = [&](const vol::CastFilter& filter, auto allowed,
                       std::size_t& hit_count) -> int {
    auto img =
        cast(gpu, caster, *grid, {cam}, kSquaresSide, kSquaresSide, filter);
    CHECK(img.ok());
    std::array<std::size_t, 256> per{};
    hit_count = 0;
    for (std::uint32_t v = 0; v < kSquaresSide; ++v) {
      for (std::uint32_t u = 0; u < kSquaresSide; ++u) {
        const vgrid::Vec2f g = squares_hit(u, v);
        bool may = false;
        bool must = false;
        classify(g, allowed, may, must);
        const float d = img->depth[v * kSquaresSide + u];
        if (d > 0.0f) {
          CHECK(may);
          CHECK(std::fabs(d - kSquaresZ) <= 1e-3f);
          ++per[byte_at(Vec3i(int(std::floor(g.x)), int(std::floor(g.y)), 0))];
          ++hit_count;
        }
        if (must) CHECK(d > 0.0f);
      }
    }
    for (int b = 0; b < 256; ++b) {
      CHECK((per[std::size_t(b)] > 0) == allowed(std::uint8_t(b)));
    }
    return 0;
  };
  std::size_t count = 0;

  // The default allows every byte: every pixel hits.
  {
    vol::CastFilter all;
    all.attribute = kSource;
    CHECK(run(all, [](std::uint8_t) { return true; }, count) == 0);
    CHECK(count == std::size_t(kSquaresSide) * kSquaresSide);
  }
  // Even bytes only.
  {
    vol::CastFilter even;
    even.attribute = kSource;
    even.allow.fill(0x55555555u);
    CHECK(run(even, [](std::uint8_t b) { return b % 2 == 0; }, count) == 0);
  }
  // One byte, at a word's first and last bit.
  for (const int one : {0, 31, 32, 255}) {
    vol::CastFilter only;
    only.attribute = kSource;
    only.allow.fill(0u);
    only.allow[std::size_t(one) >> 5] = 1u << (one & 31);
    CHECK(run(only, [one](std::uint8_t b) { return b == one; }, count) == 0);
    CHECK(count > 0);
  }
  // The marked voxel excluded: exactly the pixels whose hit's sample holds
  // it miss, the 8 x 8 whose columns' floors are -1 or 0.
  {
    vol::CastFilter unmarked;
    unmarked.attribute = kMark;
    unmarked.allow[0] = ~2u;
    auto img =
        cast(gpu, caster, *grid, {cam}, kSquaresSide, kSquaresSide, unmarked);
    CHECK(img.ok());
    std::size_t missed = 0;
    for (std::uint32_t v = 0; v < kSquaresSide; ++v) {
      for (std::uint32_t u = 0; u < kSquaresSide; ++u) {
        const vgrid::Vec2f g = squares_hit(u, v);
        const int x0 = int(std::floor(g.x));
        const int y0 = int(std::floor(g.y));
        const bool holds = (x0 == -1 || x0 == 0) && (y0 == -1 || y0 == 0);
        const bool hit = img->depth[v * kSquaresSide + u] > 0.0f;
        CHECK(hit != holds);
        missed += hit ? 0 : 1;
      }
    }
    CHECK(missed == 64);
  }
  // The weight floor answers at >=: weight 1 against 2 misses, weight 2
  // hits.
  {
    vol::CastFilter floor2;
    floor2.min_weight = 2.0f;
    auto img =
        cast(gpu, caster, *grid, {cam}, kSquaresSide, kSquaresSide, floor2);
    CHECK(img.ok() && hits(img->depth) == 0);
    auto w = vgrid_test::read_attribute<float>(gpu.device, gpu.allocator, *grid,
                                               vol::kWeightAttribute);
    CHECK(w.ok());
    for (float& x : w.value()) x *= 2.0f;
    CHECK(vgrid_test::write_attribute(gpu.device, gpu.allocator, *grid,
                                      vol::kWeightAttribute, w.value())
              .ok());
    img = cast(gpu, caster, *grid, {cam}, kSquaresSide, kSquaresSide, floor2);
    CHECK(img.ok() && hits(img->depth) == img->depth.size());
  }
  return 0;
}

// The sphere's grid at 5 mm, for the casts that compare views.
vkc::Result<vol::VoxelBlockGrid> sphere_grid(const vkc::test::Gpu& gpu) {
  const float r = vgrid_test::kSphereRadius;
  VKC_ASSIGN(vol::VoxelBlockGrid grid,
             make_grid(gpu, grid_params(0.005f, 0.02f, 1024)));
  const auto sdf = [r](Vec3f p) { return glm::length(p) - r; };
  VKC_TRY(fill(gpu, grid,
               band_blocks(grid.grid(), Vec3f(-0.35f), Vec3f(0.35f), sdf),
               [&](Vec3f p, Vec3i) { return fused(sdf(p), 0.02f); }));
  return grid;
}

bool same_bytes(const std::vector<float>& a, std::size_t a_at,
                const std::vector<float>& b, std::size_t n) {
  return a.size() >= a_at + n && b.size() >= n &&
         std::memcmp(a.data() + a_at, b.data(), n * sizeof(float)) == 0;
}

// 5. Five cameras of different intrinsics and poses cast once as a 5-view
// cast and again as five 1-view casts in another batch, each from a camera
// buffer of its own, equal bit for bit.
int stacked(vkc::test::GpuContext& gpu, const vol::VoxelBlockGrid& grid) {
  vol::RayCasterConfig config;
  config.casts_per_batch = 5;
  auto caster = vol::RayCaster::create(gpu.device, gpu.allocator, config);
  CHECK(caster.ok());
  const std::uint32_t w = vgrid_test::kSphereWidth;
  const std::uint32_t h = vgrid_test::kSphereHeight;
  std::vector<vol::DepthPinhole> cams;
  for (int i = 0; i < 5; ++i) {
    vol::DepthPinhole c = vgrid_test::sphere_view<vol::DepthPinhole>(i).cam;
    c.fx = 130.0f + 8.0f * float(i);
    c.fy = 135.0f + 5.0f * float(i);
    c.cx = 79.5f + float(i);
    c.cy = 59.5f - float(i);
    cams.push_back(c);
  }
  auto stack = cast(gpu, **caster, grid, cams, w, h);
  CHECK(stack.ok());
  std::vector<vkc::Buffer> bufs;
  std::vector<vol::ModelView> views;
  vkc::CommandBatch batch(gpu.device, gpu.allocator);
  for (const vol::DepthPinhole& c : cams) {
    auto buf = camera_buffer(gpu, {c});
    auto view = (*caster)->make_view(w, h);
    CHECK(buf.ok() && view.ok());
    bufs.push_back(std::move(buf).value());
    views.push_back(std::move(view).value());
  }
  for (std::size_t i = 0; i < cams.size(); ++i) {
    CHECK((*caster)->record_cast(batch, grid, bufs[i], 1, {}, views[i]).ok());
  }
  CHECK(batch.submit().ok());
  const std::size_t pixels = std::size_t(w) * h;
  for (std::size_t i = 0; i < cams.size(); ++i) {
    auto one = read_view(gpu, views[i]);
    CHECK(one.ok());
    CHECK(hits(one->depth) > 1000);
    CHECK(same_bytes(stack->depth, i * pixels, one->depth, pixels));
    CHECK(same_bytes(stack->normal, 3 * i * pixels, one->normal, 3 * pixels));
  }
  return 0;
}

// 6. One batch casts at pose A, uploads pose B over the same camera buffer,
// and casts into a second view: each view equals a lone cast at its pose.
int recast(vkc::test::GpuContext& gpu, vol::RayCaster& caster,
           const vol::VoxelBlockGrid& grid) {
  const std::uint32_t w = vgrid_test::kSphereWidth;
  const std::uint32_t h = vgrid_test::kSphereHeight;
  const vol::DepthPinhole a = vgrid_test::sphere_view<vol::DepthPinhole>(0).cam;
  const vol::DepthPinhole b = vgrid_test::sphere_view<vol::DepthPinhole>(1).cam;
  auto buf = camera_buffer(gpu, {a});
  auto first = caster.make_view(w, h);
  auto second = caster.make_view(w, h);
  CHECK(buf.ok() && first.ok() && second.ok());
  vkc::CommandBatch batch(gpu.device, gpu.allocator);
  CHECK(caster.record_cast(batch, grid, *buf, 1, {}, *first).ok());
  CHECK(batch.upload(*buf, 0, &b, sizeof(b)).ok());
  CHECK(caster.record_cast(batch, grid, *buf, 1, {}, *second).ok());
  CHECK(batch.submit().ok());
  auto got_a = read_view(gpu, *first);
  auto got_b = read_view(gpu, *second);
  auto lone_a = cast(gpu, caster, grid, {a}, w, h);
  auto lone_b = cast(gpu, caster, grid, {b}, w, h);
  CHECK(got_a.ok() && got_b.ok() && lone_a.ok() && lone_b.ok());
  const std::size_t n = std::size_t(w) * h;
  CHECK(hits(got_b->depth) > 1000);
  CHECK(same_bytes(got_a->depth, 0, lone_a->depth, n));
  CHECK(same_bytes(got_b->depth, 0, lone_b->depth, n));
  CHECK(same_bytes(got_b->normal, 0, lone_b->normal, 3 * n));
  CHECK(!same_bytes(got_a->depth, 0, got_b->depth, n));
  return 0;
}

// 7. A grid grown between two batches is cast with its new blocks: a plane
// at 1 m left of x = 0, every set of the caster used once, then a resize, a
// second plane at 1.5 m to the right, and a cast in a new batch through the
// first set again.
int grown(vkc::test::GpuContext& gpu, vol::RayCaster& caster) {
  auto grid = make_grid(gpu, grid_params(kVoxel, kTrunc, 256));
  CHECK(grid.ok());
  const auto sdf = [](Vec3f p) { return (p.x < 0.0f ? 1.0f : 1.5f) - p.z; };
  const auto field = [&](Vec3f p, Vec3i) { return fused(sdf(p), kTrunc); };
  const auto left = [](Vec3f p) { return 1.0f - p.z; };
  CHECK(fill(gpu, *grid,
             band_blocks(grid->grid(), Vec3f(-0.8f, -0.8f, 0.9f),
                         Vec3f(-0.01f, 0.8f, 1.1f), left),
             field)
            .ok());
  const std::uint32_t w = 64;
  const std::uint32_t h = 48;
  const vol::DepthPinhole cam = pinhole(w, h, 50.0f);
  auto buf = camera_buffer(gpu, {cam});
  auto view = caster.make_view(w, h);
  CHECK(buf.ok() && view.ok());
  // Pixel u's side: its ray at 1 m and at 1.5 m clear of x = 0 by 3 voxels.
  const auto side = [&](std::uint32_t u) {
    const float x = (float(u) - cam.cx) / cam.fx;
    return x < -0.03f ? -1 : (x > 0.02f ? 1 : 0);
  };
  const auto check = [&](bool right_hits) -> int {
    auto img = read_view(gpu, *view);
    CHECK(img.ok());
    for (std::uint32_t v = 0; v < h; ++v) {
      for (std::uint32_t u = 0; u < w; ++u) {
        const float d = img->depth[v * w + u];
        if (side(u) < 0) CHECK(std::fabs(d - 1.0f) <= 1e-4f);
        if (side(u) > 0 && right_hits) CHECK(std::fabs(d - 1.5f) <= 1e-4f);
        if (side(u) > 0 && !right_hits) CHECK(d == 0.0f);
      }
    }
    return 0;
  };
  for (int k = 0; k < 3; ++k) {  // every set of the ring
    vkc::CommandBatch batch(gpu.device, gpu.allocator);
    CHECK(caster.record_cast(batch, *grid, *buf, 1, {}, *view).ok());
    CHECK(batch.submit().ok());
    CHECK(check(false) == 0);
  }
  const auto right = [](Vec3f p) { return 1.5f - p.z; };
  CHECK(grid->resize(512).ok());
  CHECK(fill(gpu, *grid,
             band_blocks(grid->grid(), Vec3f(0.0f, -0.8f, 1.4f),
                         Vec3f(1.2f, 0.8f, 1.6f), right),
             field)
            .ok());
  vkc::CommandBatch batch(gpu.device, gpu.allocator);
  CHECK(caster.record_cast(batch, *grid, *buf, 1, {}, *view).ok());
  CHECK(batch.submit().ok());
  CHECK(check(true) == 0);
  return 0;
}

// 8. The ring: with casts_per_batch 3, three casts in one batch submit, and a
// fourth makes the submit refuse the batch.
int ring(vkc::test::GpuContext& gpu, const vol::VoxelBlockGrid& grid) {
  auto caster = vol::RayCaster::create(gpu.device, gpu.allocator);
  auto buf = camera_buffer(gpu, {pinhole(16, 16, 20.0f)});
  CHECK(caster.ok() && buf.ok());
  auto view = (*caster)->make_view(16, 16);
  CHECK(view.ok());
  for (int casts = 3; casts <= 4; ++casts) {
    vkc::CommandBatch batch(gpu.device, gpu.allocator);
    for (int k = 0; k < casts; ++k) {
      CHECK((*caster)->record_cast(batch, grid, *buf, 1, {}, *view).ok());
    }
    const vkc::Status submitted = batch.submit();
    CHECK(submitted.ok() == (casts == 3));
    if (casts == 4) {
      CHECK(submitted.domain() == vkc::Status::Code::InvalidArgument);
    }
  }
  return 0;
}

bool refused(const vkc::Status& s) {
  return s.domain() == vkc::Status::Code::InvalidArgument;
}

// 9. Refusals, each before anything is recorded.
int refusals(vkc::test::GpuContext& gpu, vol::RayCaster& caster,
             const vol::VoxelBlockGrid& grid) {
  const auto config_refused = [&](vol::RayCasterConfig c) {
    return refused(
        vol::RayCaster::create(gpu.device, gpu.allocator, c).status());
  };
  vol::RayCasterConfig c;
  c.free_step = 0.0f;
  CHECK(config_refused(c));
  c.free_step = 1.5f;
  CHECK(config_refused(c));
  c = {};
  c.min_step_voxels = 0.0f;
  CHECK(config_refused(c));
  c = {};
  c.max_steps = 0;
  CHECK(config_refused(c));
  c = {};
  c.casts_per_batch = 0;
  CHECK(config_refused(c));
  c = {};
  c.free_step = 1.0f;
  CHECK(vol::RayCaster::create(gpu.device, gpu.allocator, c).ok());

  CHECK(refused(caster.make_view(0, 16).status()));
  CHECK(refused(caster.make_view(16, 0).status()));
  CHECK(refused(caster.make_view(16, 16, 0).status()));
  CHECK(refused(caster.make_view(65536, 65536).status()));

  const vol::DepthPinhole cam = pinhole(16, 16, 20.0f);
  auto cams = camera_buffer(gpu, {cam, cam});
  auto one_cam = camera_buffer(gpu, {cam});
  auto view = caster.make_view(16, 16, 2);
  CHECK(cams.ok() && one_cam.ok() && view.ok());
  vkc::CommandBatch batch(gpu.device, gpu.allocator);
  const auto rec = [&](const vol::VoxelBlockGrid& g, const vkc::Buffer& b,
                       std::uint32_t n, const vol::CastFilter& f,
                       vol::ModelView& out) {
    return caster.record_cast(batch, g, b, n, f, out);
  };
  CHECK(rec(grid, *cams, 2, {}, *view).ok());
  CHECK(refused(rec(grid, *cams, 0, {}, *view)));
  CHECK(refused(rec(grid, *cams, 3, {}, *view)));
  CHECK(refused(rec(grid, *one_cam, 2, {}, *view)));
  {
    vkc::BufferDesc desc;
    desc.size = 2 * sizeof(vol::DepthPinhole);
    desc.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    auto plain = gpu.allocator.create_buffer(desc);
    CHECK(plain.ok());
    CHECK(refused(rec(grid, *plain, 2, {}, *view)));
  }
  {
    // Outputs smaller than their views, and more rays than one dispatch.
    auto small = caster.make_view(16, 16, 1);
    CHECK(small.ok());
    vol::ModelView short_depth;
    short_depth.width = 16;
    short_depth.height = 16;
    short_depth.views = 2;
    short_depth.depth = std::move(small->depth);
    short_depth.normal = std::move(view->normal);
    CHECK(refused(rec(grid, *cams, 1, {}, short_depth)));
    view->normal = std::move(short_depth.normal);
    auto small2 = caster.make_view(16, 16, 1);
    CHECK(small2.ok());
    vol::ModelView short_normal;
    short_normal.width = 16;
    short_normal.height = 16;
    short_normal.views = 2;
    short_normal.depth = std::move(view->depth);
    short_normal.normal = std::move(small2->normal);
    CHECK(refused(rec(grid, *cams, 1, {}, short_normal)));
    view->depth = std::move(short_normal.depth);
    vol::ModelView huge;
    huge.width = 65536;
    huge.height = 65536;
    huge.views = 1;
    huge.depth = std::move(view->depth);
    huge.normal = std::move(view->normal);
    CHECK(refused(rec(grid, *cams, 1, {}, huge)));
    view->depth = std::move(huge.depth);
    view->normal = std::move(huge.normal);
    CHECK(rec(grid, *cams, 2, {}, *view).ok());
  }
  // Grids without tsdf or weight, and filter attributes missing or 4 bytes.
  for (const std::string_view keep :
       {vol::kTsdfAttribute, vol::kWeightAttribute}) {
    const vol::AttributeSpec spec[] = {{keep, 4}};
    auto lone = vol::VoxelBlockGrid::create(gpu.device, gpu.allocator,
                                            grid.grid(), spec, 1);
    CHECK(lone.ok());
    CHECK(refused(rec(*lone, *cams, 1, {}, *view)));
  }
  vol::CastFilter missing;
  missing.attribute = "nope";
  CHECK(refused(rec(grid, *cams, 1, missing, *view)));
  vol::CastFilter wide;
  wide.attribute = vol::kWeightAttribute;
  CHECK(refused(rec(grid, *cams, 1, wide, *view)));
  CHECK(batch.submit().ok());
  return 0;
}

// The room for the timing case: inside a 4 x 2.5 x 4 m box, a sphere of
// 0.4 m on the floor's side, at 1 cm voxels and a 4 cm band.
float room_sdf(Vec3f p) {
  const Vec3f half(2.0f, 1.25f, 2.0f);
  const Vec3f in = half - glm::abs(p);
  const float wall = std::min({in.x, in.y, in.z});
  return std::min(wall, glm::length(p - Vec3f(0.6f, 0.85f, 0.9f)) - 0.4f);
}

double quantile(std::vector<double> v, double q) {
  std::sort(v.begin(), v.end());
  return v[std::min(v.size() - 1, std::size_t(q * double(v.size())))];
}

// 10. The timing case: K = 1, and K = 20 stacked against 20 single casts, at
// 256 x 192 from cameras inside the room, 100 repeats each; device ms from
// the GpuStageScope rows. Prints only.
int timing(vkc::test::GpuContext& gpu) {
  vol::RayCasterConfig config;
  config.casts_per_batch = 20;
  auto caster = vol::RayCaster::create(gpu.device, gpu.allocator, config);
  auto grid = make_grid(gpu, grid_params(kVoxel, kTrunc, 8192));
  auto timer = vkc::GpuTimer::create(gpu.device);
  CHECK(caster.ok() && grid.ok() && timer.ok());
  const std::vector<vol::BlockIndex> blocks =
      band_blocks(grid->grid(), Vec3f(-2.1f, -1.35f, -2.1f),
                  Vec3f(2.1f, 1.35f, 2.1f), room_sdf);
  CHECK(fill(gpu, *grid, blocks, [](Vec3f p, Vec3i) {
          return fused(room_sdf(p), kTrunc);
        }).ok());
  const std::uint32_t w = 256;
  const std::uint32_t h = 192;
  std::vector<vol::DepthPinhole> cams;
  for (int i = 0; i < 20; ++i) {
    const float a = 0.314159f * float(i);
    const Vec3f eye(0.7f * std::cos(a), -0.3f + 0.03f * float(i),
                    0.7f * std::sin(a));
    const Vec3f target =
        eye + Vec3f(std::cos(a + 2.0f), 0.2f, std::sin(a + 2.0f));
    cams.push_back(pinhole(w, h, 212.0f, look_at(eye, target)));
  }
  auto stack_buf = camera_buffer(gpu, cams);
  auto stack = (*caster)->make_view(w, h, 20);
  CHECK(stack_buf.ok() && stack.ok());
  std::vector<vkc::Buffer> one_bufs;
  std::vector<vol::ModelView> ones;
  for (const vol::DepthPinhole& c : cams) {
    auto b = camera_buffer(gpu, {c});
    auto v = (*caster)->make_view(w, h);
    CHECK(b.ok() && v.ok());
    one_bufs.push_back(std::move(b).value());
    ones.push_back(std::move(v).value());
  }
  // One case: its casts in one batch, `repeats` times after 5 to warm.
  const auto measure = [&](int casts_in_batch, std::uint32_t count,
                           std::vector<double>& ms) -> int {
    for (int r = 0; r < 105; ++r) {
      vkc::StageMetrics rows;
      {
        vkc::GpuStageScope stage(&rows, timer.value(), "cast");
        vkc::CommandBatch batch(gpu.device, gpu.allocator);
        for (int k = 0; k < casts_in_batch; ++k) {
          const bool stacked_case = count > 1;
          CHECK((*caster)
                    ->record_cast(
                        batch, *grid,
                        stacked_case ? *stack_buf : one_bufs[std::size_t(k)],
                        count, {}, stacked_case ? *stack : ones[std::size_t(k)],
                        &stage)
                    .ok());
        }
        CHECK(batch.submit().ok());
      }
      CHECK(rows.rows().size() == 1);
      if (rows.rows()[0].device != vkc::DeviceTime::Measured) {
        std::puts("  timing: this device does not time");
        return 0;
      }
      if (r >= 5) ms.push_back(rows.rows()[0].gpu_ms);
    }
    return 0;
  };
  std::vector<double> k1;
  std::vector<double> k20;
  std::vector<double> singles;
  CHECK(measure(1, 1, k1) == 0);
  CHECK(measure(1, 20, k20) == 0);
  CHECK(measure(20, 1, singles) == 0);
  if (k1.empty()) return 0;
  auto img = read_view(gpu, *stack);
  CHECK(img.ok());
  std::printf(
      "  timing (%s; room %zu blocks, 1 cm voxels, %ux%u, %zu of %zu rays "
      "hit):\n"
      "    K = 1:             p50 %.3f ms, p95 %.3f ms\n"
      "    K = 20 stacked:    p50 %.3f ms, p95 %.3f ms\n"
      "    20 single casts:   p50 %.3f ms, p95 %.3f ms\n",
      gpu.device.caps().properties().deviceName, blocks.size(), w, h,
      hits(img->depth), img->depth.size(), quantile(k1, 0.5),
      quantile(k1, 0.95), quantile(k20, 0.5), quantile(k20, 0.95),
      quantile(singles, 0.5), quantile(singles, 0.95));
  return 0;
}

int gpu_main(vkc::test::GpuContext& gpu) {
  auto caster = vol::RayCaster::create(gpu.device, gpu.allocator);
  CHECK(caster.ok());
  CHECK(shapes(gpu, **caster) == 0);
  CHECK(misses(gpu, **caster) == 0);
  CHECK(block_skip(gpu) == 0);
  CHECK(allow_set(gpu, **caster) == 0);
  {
    auto sphere = sphere_grid(gpu);
    CHECK(sphere.ok());
    CHECK(stacked(gpu, *sphere) == 0);
    CHECK(recast(gpu, **caster, *sphere) == 0);
    CHECK(ring(gpu, *sphere) == 0);
    CHECK(refusals(gpu, **caster, *sphere) == 0);
  }
  CHECK(grown(gpu, **caster) == 0);
  const char* timed = std::getenv("VGRID_RAY_CAST_TIMING");
  if (timed != nullptr && std::strcmp(timed, "1") == 0) {
    CHECK(timing(gpu) == 0);
  }
  std::puts("volume_ray_cast: OK");
  return 0;
}

}  // namespace

int main() {
  return vkc::test::run_on_gpu(vgrid::device_requirements(), gpu_main);
}
