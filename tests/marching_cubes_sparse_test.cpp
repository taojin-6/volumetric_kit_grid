// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// GPU test for marching-cubes extraction straight off a sparse VoxelBlockGrid.
// Writes an analytic sphere signed-distance field into a REAL block grid that
// spans a 6x6x6 grid of blocks, so the surface crosses interior block
// boundaries and extraction must sample corners from neighbouring blocks. The
// decisive check is equivalence with the dense path: the same sphere at the
// same resolution meshes the same cells with the same corner values, so the two
// triangle counts must match exactly -- a wrong neighbour octant or lookup
// index would corrupt every boundary cell and diverge. Also verifies the sphere
// shape (radius, outward normals, winding), cross-block colour interpolation,
// and the empty / argument-validation / moved-from paths. Exits 0 (skip) where
// no device is present.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <utility>
#include <vector>

#include "volumetric_kit/recon/core/allocator.hpp"
#include "volumetric_kit/recon/core/device.hpp"
#include "volumetric_kit/recon/core/instance.hpp"
#include "volumetric_kit/recon/core/math/vector_types.hpp"
#include "volumetric_kit/recon/core/vulkan.hpp"
#include "volumetric_kit/recon/mesh/marching_cubes.hpp"
#include "volumetric_kit/recon/mesh/mesh.hpp"
#include "volumetric_kit/recon/volume/hash_types.hpp"
#include "volumetric_kit/recon/volume/voxel_block_grid.hpp"
#include "volumetric_kit/recon/volume/voxel_grid.hpp"

namespace vr = volumetric_kit::recon;
namespace vol = volumetric_kit::recon::volume;
namespace mesh = volumetric_kit::recon::mesh;

#define CHECK(cond)                                                        \
  do {                                                                     \
    if (!(cond)) {                                                         \
      std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
      return 1;                                                            \
    }                                                                      \
  } while (0)

namespace {

constexpr int kBlock = 8;             // voxels per block edge
constexpr int kBlocks = 6;            // blocks per axis
constexpr int kN = kBlock * kBlocks;  // voxels per axis (48)
constexpr float kH = 0.05f;           // metres between voxels
constexpr float kRadius = 0.7f;       // sphere radius (well inside)
constexpr float kSpan = static_cast<float>(kN - 1) * kH;  // grid world extent

// Sphere centre, shared by the dense and sparse fields (node convention: voxel
// (x,y,z) sits at world (x,y,z)*kH, so both grids sample identical positions).
vr::Vec3f sphere_center() {
  const float c = static_cast<float>(kN - 1) * 0.5f * kH;
  return vr::Vec3f(c, c, c);
}

float sphere_sdf(vr::Vec3f world) {
  return vr::length(world - sphere_center()) - kRadius;
}

// A linear gradient colour: world position normalized to [0,1] per axis. Linear
// interpolation of a linear field is exact, so a vertex's colour must match the
// gradient at that vertex's own position to within u8 quantization.
vr::Vec3f grad_color(vr::Vec3f p) {
  auto clamp01 = [](float v) {
    return v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v);
  };
  return vr::Vec3f(clamp01(p.x / kSpan), clamp01(p.y / kSpan),
                   clamp01(p.z / kSpan));
}

std::uint32_t pack_rgb(vr::Vec3f c) {
  // RGB in the low three bytes, alpha 0xFF in the high byte -- matching the
  // tsdf integrator's packUnorm4x8(vec4(rgb, 1.0)). An observed colour is then
  // always non-zero, so 0 uniquely means "colour unobserved", the sentinel the
  // sparse kernel falls back to white on.
  return static_cast<std::uint32_t>(c.x * 255.0f + 0.5f) |
         (static_cast<std::uint32_t>(c.y * 255.0f + 0.5f) << 8) |
         (static_cast<std::uint32_t>(c.z * 255.0f + 0.5f) << 16) |
         (0xFFu << 24);
}

// The dense reference field over the same kN^3 samples (x-fastest).
std::vector<vol::Voxel> make_dense_sphere() {
  std::vector<vol::Voxel> samples(static_cast<std::size_t>(kN) * kN * kN);
  for (int z = 0; z < kN; ++z) {
    for (int y = 0; y < kN; ++y) {
      for (int x = 0; x < kN; ++x) {
        const vr::Vec3f p(static_cast<float>(x) * kH,
                          static_cast<float>(y) * kH,
                          static_cast<float>(z) * kH);
        vol::Voxel& v =
            samples[static_cast<std::size_t>(x + kN * (y + kN * z))];
        v.sdf = sphere_sdf(p);
        v.weight = 1.0f;
      }
    }
  }
  return samples;
}

vol::VoxelGridParams sphere_grid_params() {
  vol::VoxelGridParams grid{};
  grid.voxel_size = kH;
  grid.block_size = kBlock;
  grid.voxels_per_block = kBlock * kBlock * kBlock;
  grid.trunc_dist = 0.04f;  // unused by meshing; must pass validate()
  grid.bucket_size = 8;
  grid.num_buckets = 128;
  grid.num_blocks = 1024;  // = bucket_size * num_buckets; >> 216 active blocks
  grid.max_chain = 128;
  return grid;
}

// Allocate the full kBlocks^3 cube of blocks, then write the sphere SDF (at the
// given @p weight) and, when requested, the gradient colour into every voxel of
// every allocated block, addressed by the compacted BlockIndex::ptr + local. A
// colour attribute left unwritten (with_color = false on a grid that carries
// one) stays zero -- the integrator's "colour unobserved" sentinel. Returns
// false on any device error.
bool fill_sphere_grid(vol::VoxelBlockGrid& g, bool with_color,
                      float weight_value = 1.0f) {
  std::vector<vol::BlockIndex> blocks;
  for (int cz = 0; cz < kBlocks; ++cz) {
    for (int cy = 0; cy < kBlocks; ++cy) {
      for (int cx = 0; cx < kBlocks; ++cx) {
        vol::BlockIndex b{};
        b.coord = vr::Vec3i(cx, cy, cz);
        blocks.push_back(b);
      }
    }
  }
  vr::Result<std::uint32_t> failed = g.map().allocate(
      blocks.data(), static_cast<std::uint32_t>(blocks.size()));
  if (!failed || failed.value() != 0) {
    return false;
  }
  vr::Result<std::vector<vol::BlockIndex>> active =
      g.map().compact_active_blocks();
  if (!active || active.value().size() != blocks.size()) {
    return false;
  }

  vr::Result<vol::AttributeView> tsdf = g.attribute("tsdf");
  vr::Result<vol::AttributeView> weight = g.attribute("weight");
  if (!tsdf || !weight) {
    return false;
  }
  auto* tptr = static_cast<float*>(tsdf.value().buffer->mapped());
  auto* wptr = static_cast<float*>(weight.value().buffer->mapped());
  std::uint32_t* cptr = nullptr;
  if (with_color) {
    vr::Result<vol::AttributeView> color = g.attribute("color");
    if (!color) {
      return false;
    }
    cptr = static_cast<std::uint32_t*>(color.value().buffer->mapped());
  }

  for (const vol::BlockIndex& b : active.value()) {
    for (int lz = 0; lz < kBlock; ++lz) {
      for (int ly = 0; ly < kBlock; ++ly) {
        for (int lx = 0; lx < kBlock; ++lx) {
          const int local = lx + kBlock * (ly + kBlock * lz);
          const vr::Vec3i voxel = b.coord * kBlock + vr::Vec3i(lx, ly, lz);
          const vr::Vec3f world = vr::Vec3f(voxel) * kH;
          const auto idx = static_cast<std::size_t>(b.ptr) + local;
          tptr[idx] = sphere_sdf(world);
          wptr[idx] = weight_value;
          if (cptr != nullptr) {
            cptr[idx] = pack_rgb(grad_color(world));
          }
        }
      }
    }
  }
  return true;
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

  vr::Result<mesh::MarchingCubes> mc_result =
      mesh::MarchingCubes::create(device.value(), allocator.value());
  if (!mc_result) {
    std::fprintf(stderr, "MarchingCubes::create failed: %s\n",
                 mc_result.status().message().c_str());
    return 1;
  }
  mesh::MarchingCubes extractor = std::move(mc_result).value();

  const vol::VoxelGridParams gp = sphere_grid_params();
  const vol::AttributeSpec attrs[] = {{"tsdf", sizeof(float)},
                                      {"weight", sizeof(float)}};

  // --- Sparse extraction of a multi-block sphere -----------------------------
  vr::Result<vol::VoxelBlockGrid> grid_result = vol::VoxelBlockGrid::create(
      device.value(), allocator.value(), gp, attrs, 2);
  CHECK(grid_result.ok());
  vol::VoxelBlockGrid grid = std::move(grid_result).value();
  CHECK(fill_sphere_grid(grid, /*with_color=*/false));

  vr::Result<mesh::Mesh> sparse_result = extractor.extract(grid, 0.0f);
  CHECK(sparse_result.ok());
  mesh::Mesh sphere = std::move(sparse_result).value();
  CHECK(!sphere.empty());
  CHECK(sphere.indices.size() == sphere.vertices.size());
  CHECK(sphere.indices.size() % 3 == 0);
  CHECK(sphere.triangle_count() > 500);

  const vr::Vec3f center = sphere_center();

  // Every vertex lies on the sphere; every normal is unit and points outward.
  double radius_sum = 0.0;
  double outward_sum = 0.0;
  for (const mesh::Vertex& v : sphere.vertices) {
    const vr::Vec3f d = v.position - center;
    const float r = vr::length(d);
    CHECK(std::fabs(r - kRadius) < 1.5f * kH);  // MC accuracy ~ one voxel
    CHECK(std::fabs(vr::length(v.normal) - 1.0f) < 1e-3f);
    CHECK(vr::dot(vr::normalize(d), v.normal) > 0.5f);  // never inward
    radius_sum += r;
    outward_sum += vr::dot(vr::normalize(d), v.normal);
  }
  const auto nv = static_cast<double>(sphere.vertices.size());
  CHECK(std::fabs(radius_sum / nv - kRadius) < 0.25 * kH);  // no radial bias
  CHECK(outward_sum / nv > 0.9);

  // Winding agrees with the gradient normal, per face (a boundary seam or a
  // flipped table would drop this well below 1).
  int face_count = 0;
  int face_outward = 0;
  for (std::size_t t = 0; t + 2 < sphere.vertices.size(); t += 3) {
    const vr::Vec3f face = vr::cross(
        sphere.vertices[t + 1].position - sphere.vertices[t].position,
        sphere.vertices[t + 2].position - sphere.vertices[t].position);
    if (vr::length(face) > 1e-8f) {
      if (vr::dot(vr::normalize(face), sphere.vertices[t].normal) > 0.0f) {
        ++face_outward;
      }
      ++face_count;
    }
  }
  CHECK(face_count > 0);
  const double outward_ratio =
      static_cast<double>(face_outward) / static_cast<double>(face_count);
  CHECK(outward_ratio > 0.95);

  // No colour input -> opaque-white vertices, uv0 at the sentinel.
  for (const mesh::Vertex& v : sphere.vertices) {
    CHECK(v.color.x == 1.0f && v.color.y == 1.0f && v.color.z == 1.0f &&
          v.color.w == 1.0f);
    CHECK(v.uv0.x < 0.0f && v.uv0.y < 0.0f);
  }

  // --- Equivalence with the dense path (the cross-block correctness proof) ---
  // The dense grid samples the identical kN^3 field at the identical positions,
  // so it meshes the identical cell set: base voxels [0, kN-2]^3 (the sparse
  // +face cells at voxel kN-1 reach the unallocated block kN and are skipped,
  // exactly the cells the dense grid also lacks). Same cells, same corner
  // values -> identical triangle count. A mis-indexed neighbour lookup would
  // corrupt every interior-boundary cell and break this equality.
  const std::vector<vol::Voxel> dense = make_dense_sphere();
  mesh::DenseGrid dense_grid;
  dense_grid.dims = vr::Vec3i(kN, kN, kN);
  dense_grid.voxel_size = kH;
  dense_grid.origin = vr::Vec3f(0.0f, 0.0f, 0.0f);
  vr::Result<mesh::Mesh> dense_result =
      extractor.extract(dense.data(), dense.size(), dense_grid, 0.0f);
  CHECK(dense_result.ok());
  const mesh::Mesh dense_mesh = std::move(dense_result).value();
  CHECK(sphere.triangle_count() == dense_mesh.triangle_count());

  // --- Cross-block colour interpolation --------------------------------------
  // A grid carrying a packed-RGB colour attribute set to the linear gradient:
  // each vertex's colour must match grad_color at that vertex's own position
  // (linear interp of a linear field is exact), to within u8 quantization --
  // including vertices on triangles whose corners came from neighbouring
  // blocks.
  const vol::AttributeSpec cattrs[] = {{"tsdf", sizeof(float)},
                                       {"weight", sizeof(float)},
                                       {"color", sizeof(std::uint32_t)}};
  vr::Result<vol::VoxelBlockGrid> cgrid_result = vol::VoxelBlockGrid::create(
      device.value(), allocator.value(), gp, cattrs, 3);
  CHECK(cgrid_result.ok());
  vol::VoxelBlockGrid cgrid = std::move(cgrid_result).value();
  CHECK(fill_sphere_grid(cgrid, /*with_color=*/true));

  vr::Result<mesh::Mesh> colored_result = extractor.extract(cgrid, 0.0f);
  CHECK(colored_result.ok());
  const mesh::Mesh colored = std::move(colored_result).value();
  CHECK(!colored.empty());
  CHECK(colored.triangle_count() == dense_mesh.triangle_count());
  for (const mesh::Vertex& v : colored.vertices) {
    const vr::Vec3f expected = grad_color(v.position);
    CHECK(std::fabs(v.color.x - expected.x) < 0.005f);  // ~2.5x the u8 floor
    CHECK(std::fabs(v.color.y - expected.y) < 0.005f);
    CHECK(std::fabs(v.color.z - expected.z) < 0.005f);
    CHECK(v.color.w == 1.0f);
    CHECK(v.uv0.x < 0.0f && v.uv0.y < 0.0f);  // still the sentinel
  }

  // --- Weight gate: sub-threshold weight drops every cell --------------------
  // The kernel skips any cell touching a voxel whose weight is at/below the
  // unintegrated threshold. Fill the same sphere SDF but with weight 0 (the
  // never-integrated state -- e.g. an allocated-but-unfused truncation-band
  // block): the field still crosses the iso, yet every cell is gated out, so
  // the mesh is empty. Proves the gate fires on-device (a dropped or mis-signed
  // gate would mesh the sphere here); the sphere fill above only ever exercises
  // the pass side.
  vr::Result<vol::VoxelBlockGrid> zw_result = vol::VoxelBlockGrid::create(
      device.value(), allocator.value(), gp, attrs, 2);
  CHECK(zw_result.ok());
  vol::VoxelBlockGrid zw_grid = std::move(zw_result).value();
  CHECK(fill_sphere_grid(zw_grid, /*with_color=*/false, /*weight=*/0.0f));
  vr::Result<mesh::Mesh> zw_mesh = extractor.extract(zw_grid, 0.0f);
  CHECK(zw_mesh.ok());
  CHECK(std::move(zw_mesh).value().empty());

  // --- Colour sentinel: unobserved colour is white, not black ----------------
  // A grid that carries a colour attribute whose colour was never written (all
  // zero -- the integrator's "colour unobserved" sentinel, e.g. a separate
  // colour camera that never saw these voxels) meshes with has_color enabled,
  // yet every corner reads the 0 sentinel. Each vertex must fall back to opaque
  // white rather than be dragged toward black (which is what an unguarded
  // unpack of the 0 sentinel would produce). Geometry is unaffected, so the
  // triangle count still matches the dense path.
  vr::Result<vol::VoxelBlockGrid> sgrid_result = vol::VoxelBlockGrid::create(
      device.value(), allocator.value(), gp, cattrs, 3);
  CHECK(sgrid_result.ok());
  vol::VoxelBlockGrid sgrid = std::move(sgrid_result).value();
  CHECK(fill_sphere_grid(sgrid, /*with_color=*/false));  // colour left at 0
  vr::Result<mesh::Mesh> sentinel_result = extractor.extract(sgrid, 0.0f);
  CHECK(sentinel_result.ok());
  const mesh::Mesh sentinel = std::move(sentinel_result).value();
  CHECK(!sentinel.empty());
  CHECK(sentinel.triangle_count() == dense_mesh.triangle_count());
  for (const mesh::Vertex& v : sentinel.vertices) {
    CHECK(v.color.x == 1.0f && v.color.y == 1.0f && v.color.z == 1.0f &&
          v.color.w == 1.0f);
  }

  // --- Empty map -> empty mesh -----------------------------------------------
  // A grid with no allocated blocks has no active set, so nothing meshes.
  vr::Result<vol::VoxelBlockGrid> empty_result = vol::VoxelBlockGrid::create(
      device.value(), allocator.value(), gp, attrs, 2);
  CHECK(empty_result.ok());
  vol::VoxelBlockGrid empty_grid = std::move(empty_result).value();
  vr::Result<mesh::Mesh> empty_mesh = extractor.extract(empty_grid, 0.0f);
  CHECK(empty_mesh.ok());
  CHECK(std::move(empty_mesh).value().empty());

  // --- Vertex-arena growth policy --------------------------------------------
  // The arena is retained between extracts and only ever grown, so its size is
  // part of the extractor's contract, not an implementation detail: it must
  // always cover the call's worst-case capacity, must grow when a call needs
  // more than the last one, and must NOT shrink back when a later call needs
  // less (that reuse is the whole point -- reallocating this buffer was ~90% of
  // a sparse extract). ExtractTimings::arena_bytes reports what the extractor
  // is holding, which is what makes the policy checkable from outside.
  //
  // Checked here rather than on the dense path because the sizes must be
  // observed, not inferred: the worst-case capacity is ~5 triangles per cell
  // while a sphere emits a small fraction of that, so an undersized arena still
  // holds every emitted triangle and comparing meshes would NOT catch a broken
  // growth policy.
  vr::Result<mesh::MarchingCubes> arena_result =
      mesh::MarchingCubes::create(device.value(), allocator.value());
  CHECK(arena_result.ok());
  mesh::MarchingCubes arena_mc = std::move(arena_result).value();

  // One allocated block: the smallest non-empty active set. (An empty one
  // returns early without sizing anything, so it cannot anchor this.)
  vr::Result<vol::VoxelBlockGrid> one_result = vol::VoxelBlockGrid::create(
      device.value(), allocator.value(), gp, attrs, 2);
  CHECK(one_result.ok());
  vol::VoxelBlockGrid one_grid = std::move(one_result).value();
  vol::BlockIndex single_block{};
  single_block.coord = vr::Vec3i(0, 0, 0);
  vr::Result<std::uint32_t> single_failed =
      one_grid.map().allocate(&single_block, 1);
  CHECK(single_failed.ok());
  CHECK(single_failed.value() == 0);

  // Bytes the call's own worst case needs, independent of what is held.
  const auto needed_bytes = [](const mesh::ExtractTimings& t) {
    return static_cast<std::uint64_t>(t.triangle_capacity) * 3 *
           sizeof(mesh::Vertex);
  };

  mesh::ExtractTimings small_timings;
  CHECK(arena_mc.extract(one_grid, 0.0f, &small_timings).ok());
  CHECK(small_timings.triangle_capacity > 0);
  CHECK(small_timings.arena_bytes >= needed_bytes(small_timings));

  // The sphere grid needs a far larger capacity: the grow path.
  mesh::ExtractTimings big_timings;
  CHECK(arena_mc.extract(grid, 0.0f, &big_timings).ok());
  CHECK(big_timings.triangle_capacity > small_timings.triangle_capacity);
  CHECK(big_timings.arena_bytes >= needed_bytes(big_timings));
  CHECK(big_timings.arena_bytes > small_timings.arena_bytes);

  // Back to the one-block grid: the oversized arena is reused as-is, neither
  // reallocated nor shrunk.
  mesh::ExtractTimings reuse_timings;
  CHECK(arena_mc.extract(one_grid, 0.0f, &reuse_timings).ok());
  CHECK(reuse_timings.triangle_capacity == small_timings.triangle_capacity);
  CHECK(reuse_timings.arena_bytes == big_timings.arena_bytes);

  // --- Argument validation ---------------------------------------------------
  // A grid missing the tsdf/weight attributes is rejected (a bare grid here).
  vr::Result<vol::VoxelBlockGrid> bare_result = vol::VoxelBlockGrid::create(
      device.value(), allocator.value(), gp, nullptr, 0);
  CHECK(bare_result.ok());
  vol::VoxelBlockGrid bare_grid = std::move(bare_result).value();
  CHECK(!extractor.extract(bare_grid, 0.0f).ok());

  // A moved-from grid is rejected.
  vol::VoxelBlockGrid moved_grid = std::move(empty_grid);
  CHECK(!extractor.extract(empty_grid, 0.0f).ok());

  // --- Move-only extractor ---------------------------------------------------
  mesh::MarchingCubes moved = std::move(extractor);
  CHECK(!extractor.valid());
  CHECK(moved.valid());
  mesh::MarchingCubes* alias = &moved;
  moved = std::move(*alias);  // self-move: intact
  CHECK(moved.valid());
  vr::Result<mesh::Mesh> reextract = moved.extract(grid, 0.0f);
  CHECK(reextract.ok());
  CHECK(!std::move(reextract).value().empty());

  std::printf(
      "recon mesh sparse marching-cubes test passed: meshed a sphere across "
      "%d^3 blocks (%zu triangles), matched the dense path exactly, and "
      "verified cross-block colour on-device\n",
      kBlocks, sphere.triangle_count());
  return 0;
}
