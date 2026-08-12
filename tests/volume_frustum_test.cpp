// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// GPU test for frustum-culled compaction: allocate blocks at known world
// positions and verify compact_active_blocks_in_frustum keeps exactly the
// in-view ones while plain compact_active_blocks keeps all. Two scenarios:
// (1) an identity-pose camera with edge cases -- an edge-straddling block kept
// only by the ~10% side-plane widening, and one just inside the far plane;
// (2) a NON-identity pose (rotated + translated) so the plane->world transform
// (inverseTranspose(cam_to_world)) is actually exercised, not a no-op. Expected
// sets are derived from the camera geometry, independent of
// make_frustum_planes, and each result is pinned by count + distinct heap ptrs.
// Runs on the real driver (MoltenVK on Apple, the NVIDIA ICD on Linux CI).
// Exits 0 (skip) where no device is present.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <set>
#include <tuple>
#include <utility>
#include <vector>

#include <glm/ext/matrix_clip_space.hpp>
#include <glm/ext/matrix_transform.hpp>

#include "volumetric_kit/recon/core/allocator.hpp"
#include "volumetric_kit/recon/core/device.hpp"
#include "volumetric_kit/recon/core/instance.hpp"
#include "volumetric_kit/recon/core/math/vector_types.hpp"
#include "volumetric_kit/recon/core/vulkan.hpp"
#include "volumetric_kit/recon/volume/frustum.hpp"
#include "volumetric_kit/recon/volume/hash_types.hpp"
#include "volumetric_kit/recon/volume/voxel_grid.hpp"
#include "volumetric_kit/recon/volume/voxel_hash_map.hpp"

namespace vr = volumetric_kit::recon;
namespace vol = volumetric_kit::recon::volume;

#define CHECK(cond)                                                        \
  do {                                                                     \
    if (!(cond)) {                                                         \
      std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
      return 1;                                                            \
    }                                                                      \
  } while (0)

namespace {

using Coord = std::tuple<int, int, int>;

std::set<Coord> to_set(const std::vector<vol::BlockIndex>& blocks) {
  std::set<Coord> s;
  for (const vol::BlockIndex& b : blocks) {
    s.insert({b.coord.x, b.coord.y, b.coord.z});
  }
  return s;
}

// Assert a frustum result matches `want` exactly: the same coord set AND the
// same count (no missing/duplicate survivor) AND pairwise-distinct valid heap
// ptrs -- a duplicate append or double-pop in the frustum kernel would give the
// same coord set but a larger size / a repeated ptr. Returns 1 (fail) via
// CHECK.
int check_result(const std::vector<vol::BlockIndex>& blocks,
                 const std::set<Coord>& want) {
  CHECK(blocks.size() == want.size());
  std::set<int> ptrs;
  for (const vol::BlockIndex& b : blocks) {
    CHECK(b.ptr >= 0);
    ptrs.insert(b.ptr);
  }
  CHECK(ptrs.size() == blocks.size());
  CHECK(to_set(blocks) == want);
  return 0;
}

}  // namespace

int main() {
  vr::Result<vr::Instance> instance = vr::Instance::create({});
  if (!instance) {
    std::fprintf(stderr, "no Vulkan instance (%s); skipping\n",
                 instance.status().message().c_str());
    return 0;
  }
  vr::Result<VkPhysicalDevice> gpu = instance.value().select_physical_device();
  if (!gpu) {
    std::fprintf(stderr, "no compute-capable device (%s); skipping\n",
                 gpu.status().message().c_str());
    return 0;
  }
  vr::Result<vr::Device> device =
      vr::Device::create(instance.value().handle(), gpu.value(), {});
  if (!device) {
    std::fprintf(stderr, "device create failed: %s\n",
                 device.status().message().c_str());
    return 1;
  }
  vr::Result<vr::Allocator> allocator =
      vr::Allocator::create(instance.value().handle(), device.value());
  if (!allocator) {
    std::fprintf(stderr, "allocator create failed: %s\n",
                 allocator.status().message().c_str());
    return 1;
  }

  vol::VoxelGridParams grid{};
  grid.voxel_size = 0.005f;
  grid.block_size = 8;
  grid.voxels_per_block = 512;
  grid.trunc_dist = 0.04f;
  grid.bucket_size = 8;
  grid.num_buckets = 1024;
  grid.num_blocks = 8192;
  grid.max_chain = 128;

  vr::Result<vol::VoxelHashMap> map_result =
      vol::VoxelHashMap::create(device.value(), allocator.value(), grid);
  if (!map_result) {
    std::fprintf(stderr, "VoxelHashMap::create failed: %s\n",
                 map_result.status().message().c_str());
    return 1;
  }
  vol::VoxelHashMap map = std::move(map_result).value();

  // Identity-pose camera at the origin looking down +Z (fx=fy=100, cx=cy=50,
  // 100x100, near 0.1, far 5). A block's world min-corner is coord * block_size
  // * voxel_size = coord * 0.04 m. Two of the six sit at deliberate edge cases:
  // kEdge is just PAST the true right image edge, kept only by the ~10%
  // side-plane widening (remove/reverse the widening and it culls); kNearFar is
  // just INSIDE the far plane (world z 4.8 < far 5), pinning the far clip.
  const vr::Vec3i kInside(0, 0, 25);    // world (0,0,1.0): dead centre
  const vr::Vec3i kEdge(14, 0, 25);     // world (0.56,0,1.0): past true edge
  const vr::Vec3i kNearFar(0, 0, 120);  // world (0,0,4.8): just inside far
  const vr::Vec3i kBehind(0, 0, -25);   // world (0,0,-1.0): behind the camera
  const vr::Vec3i kFar(0, 0, 200);      // world (0,0,8.0): beyond far
  const vr::Vec3i kSide(100, 0, 25);    // world (4.0,0,1.0): outside the fov
  std::vector<vol::BlockIndex> coords;
  for (const vr::Vec3i& c : {kInside, kEdge, kNearFar, kBehind, kFar, kSide}) {
    vol::BlockIndex b{};
    b.coord = c;
    coords.push_back(b);
  }
  vr::Result<std::uint32_t> fail =
      map.allocate(coords.data(), static_cast<std::uint32_t>(coords.size()));
  CHECK(fail.ok() && fail.value() == 0);

  // Plain compaction: all six blocks are active.
  vr::Result<std::vector<vol::BlockIndex>> all = map.compact_active_blocks();
  CHECK(all.ok());
  CHECK(all.value().size() == 6);

  // Frustum: the three in-view blocks survive (dead-centre, edge-straddling,
  // near-far); behind / far / lateral cull. check_result also pins the exact
  // count + distinct ptrs, so a duplicate append in the frustum kernel fails.
  const std::set<Coord> want = {{0, 0, 25}, {14, 0, 25}, {0, 0, 120}};
  const vol::FrustumPlanes planes = vol::make_frustum_planes(
      100.0f, 100.0f, 50.0f, 50.0f, 100, 100, 0.1f, 5.0f, vr::Mat4f(1.0f));
  vr::Result<std::vector<vol::BlockIndex>> visible =
      map.compact_active_blocks_in_frustum(planes);
  CHECK(visible.ok());
  if (check_result(visible.value(), want) != 0) return 1;

  // The DepthCameraParams convenience derives the same frustum (min/max_depth
  // as near/far) and culls identically.
  vr::DepthCameraParams cam{};
  cam.fx = 100.0f;
  cam.fy = 100.0f;
  cam.cx = 50.0f;
  cam.cy = 50.0f;
  cam.min_depth = 0.1f;
  cam.max_depth = 5.0f;
  cam.width = 100;
  cam.height = 100;
  cam.cam_to_world = vr::Mat4f(1.0f);
  vr::Result<std::vector<vol::BlockIndex>> visible_cam =
      map.compact_active_blocks_in_frustum(cam);
  CHECK(visible_cam.ok());
  if (check_result(visible_cam.value(), want) != 0) return 1;

  // --- Posed camera ---------------------------------------------------------
  // Repeat with a NON-identity pose so the plane->world transform
  // (inverseTranspose(cam_to_world)) is actually exercised on-device -- under
  // identity it is a no-op, so a transpose/translation bug in
  // make_frustum_planes would ship green. Camera at world (1,2,3) looking down
  // world +X (90 deg about Y). `want` is derived from the camera geometry (its
  // world position + view axis), NOT from make_frustum_planes.
  CHECK(map.clear().ok());
  // Column-major: col0 (cam +X -> world -Z), col1 (cam +Y -> world +Y), col2
  // (cam +Z view -> world +X), col3 = camera world position (1,2,3).
  const vr::Mat4f pose(0.0f, 0.0f, -1.0f, 0.0f,  // column 0
                       0.0f, 1.0f, 0.0f, 0.0f,   // column 1
                       1.0f, 0.0f, 0.0f, 0.0f,   // column 2
                       1.0f, 2.0f, 3.0f, 1.0f);  // column 3 (camera position)
  // Camera at (1,2,3) looking +X. kFront -> world (3,2,3) is 2 m dead ahead
  // (cam-space (0,0,2)); the others are behind / beyond far / far off-axis.
  const vr::Vec3i kFront(75, 50, 75);     // cam-space (0,0,2): dead centre
  const vr::Vec3i kPBehind(-25, 50, 75);  // cam-space depth -2: behind
  const vr::Vec3i kPFar(225, 50, 75);     // cam-space depth 8: beyond far
  const vr::Vec3i kPSide(75, 50, -175);   // cam-space (10,0,2): u ~ 550
  std::vector<vol::BlockIndex> pcoords;
  for (const vr::Vec3i& c : {kFront, kPBehind, kPFar, kPSide}) {
    vol::BlockIndex b{};
    b.coord = c;
    pcoords.push_back(b);
  }
  vr::Result<std::uint32_t> pfail =
      map.allocate(pcoords.data(), static_cast<std::uint32_t>(pcoords.size()));
  CHECK(pfail.ok() && pfail.value() == 0);

  vr::DepthCameraParams pcam = cam;
  pcam.cam_to_world = pose;
  vr::Result<std::vector<vol::BlockIndex>> pvisible =
      map.compact_active_blocks_in_frustum(pcam);
  CHECK(pvisible.ok());
  const std::set<Coord> pwant = {{75, 50, 75}};
  if (check_result(pvisible.value(), pwant) != 0) return 1;

  // --- Render camera (view-projection matrix) -------------------------------
  // The same six blocks and the same frustum geometry, reached through the
  // matrix overload a *renderer's* camera has instead of pixel intrinsics.
  // fovy = 2*atan(cy/fy) = 2*atan(0.5) at aspect 1 reproduces the fx=fy=100,
  // cx=cy=50, 100x100 pinhole above.
  //
  // RH_ZO explicitly, not glm::perspective: that one follows
  // GLM_FORCE_DEPTH_ZERO_TO_ONE, which recon does not define (it draws
  // nothing), so it would hand this a GL-convention [-1,1] matrix and the test
  // would then assert the very confusion the overload's @warning is about.
  CHECK(map.clear().ok());
  vr::Result<std::uint32_t> vfail =
      map.allocate(coords.data(), static_cast<std::uint32_t>(coords.size()));
  CHECK(vfail.ok() && vfail.value() == 0);
  // lookAtRH from the origin toward world +Z with up = world -Y puts camera
  // right on world +X and camera down on world +Y: the OpenCV basis the pinhole
  // overload assumes, so the two describe the same frustum in the same place.
  const vr::Mat4f view_proj =
      glm::perspectiveRH_ZO(2.0f * std::atan(0.5f), 1.0f, 0.1f, 5.0f) *
      glm::lookAtRH(vr::Vec3f(0.0f, 0.0f, 0.0f), vr::Vec3f(0.0f, 0.0f, 1.0f),
                    vr::Vec3f(0.0f, -1.0f, 0.0f));

  // Exact, so kEdge is CULLED here -- it survives the pinhole overload only on
  // that one's built-in ~10% side widening, and this is what pins the
  // difference. A near-plane convention slip ([-1,1] read as [0,1]) also lands
  // here: it clips at the eye rather than at 0.1, which keeps kBehind.
  const std::set<Coord> vp_want = {{0, 0, 25}, {0, 0, 120}};
  vr::Result<std::vector<vol::BlockIndex>> vp_visible =
      map.compact_active_blocks_in_frustum(vol::make_frustum_planes(view_proj));
  CHECK(vp_visible.ok());
  if (check_result(vp_visible.value(), vp_want) != 0) return 1;

  // The margin is a distance in metres, which is the whole reason it is
  // normalized in: kEdge's box sits ~0.035 m outside the exact right plane, so
  // 0.1 m recovers it and nothing else -- kBehind (~1.06 m out), kFar (~3.0 m)
  // and kSide (~3.1 m) stay culled. A margin applied before normalizing, or one
  // folded into the focal lengths, would not scale this way.
  const std::set<Coord> margin_want = {{0, 0, 25}, {14, 0, 25}, {0, 0, 120}};
  vr::Result<std::vector<vol::BlockIndex>> vp_margin =
      map.compact_active_blocks_in_frustum(
          vol::make_frustum_planes(view_proj, 0.1f));
  CHECK(vp_margin.ok());
  if (check_result(vp_margin.value(), margin_want) != 0) return 1;

  std::printf(
      "recon volume frustum test passed: identity view kept %zu/6 (widening + "
      "near-far edges), posed view kept %zu/4 (pose transform exercised), "
      "view_proj kept %zu/6 exact and %zu/6 at a 0.1 m margin\n",
      want.size(), pwant.size(), vp_want.size(), margin_want.size());
  return 0;
}
