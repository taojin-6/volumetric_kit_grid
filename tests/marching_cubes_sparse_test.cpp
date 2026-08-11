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
// the vertex-arena growth policy, the refit-and-re-run path an undersized
// arena takes -- over one block and over a run of 27, where the arena boundary
// falls inside one block's span and past others entirely -- that a block's
// triangles land CONTIGUOUSLY in the arena on both paths, that the block size
// is an allocation detail the surface does not depend on (the same field at
// block_size 8 and 16 meshes identically), and the empty /
// argument-validation / moved-from paths. Exits 0 (skip) where no device is
// present.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <map>
#include <set>
#include <utility>
#include <vector>

#include "volumetric_kit/recon/core/allocator.hpp"
#include "volumetric_kit/recon/core/color_space.hpp"
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

// Allocate a @p span cubed run of blocks and fill every voxel with a
// sign-alternating field, so every cell has mixed corner signs and emits
// several triangles: ~1400 triangles per block of 8, against the ~64 per block
// a first extract plans for. That gap is what makes the refit-and-re-run path
// fire deterministically rather than by numeric accident. Returns false on any
// device error.
//
// The checkerboard is keyed on the GLOBAL voxel coordinate rather than the
// block-local one, so it is continuous across block boundaries. That costs
// nothing at span 1 (block (0,0,0)'s local and global coordinates agree) and
// buys two things: a multi-block run meshes its interior seams like any real
// field, and the SAME field can be written into grids with different block
// sizes and must extract to the same surface -- which is what pins the kernel's
// per-block cell cache, whose only effect is supposed to be speed.
//
// Reads the block size from the grid rather than assuming kBlock, for that
// second use.
bool fill_dense_blocks(vol::VoxelBlockGrid& g, int span) {
  const int bs = g.grid().block_size;
  const float h = g.grid().voxel_size;
  std::vector<vol::BlockIndex> blocks;
  for (int cz = 0; cz < span; ++cz) {
    for (int cy = 0; cy < span; ++cy) {
      for (int cx = 0; cx < span; ++cx) {
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
  for (const vol::BlockIndex& blk : active.value()) {
    for (int lz = 0; lz < bs; ++lz) {
      for (int ly = 0; ly < bs; ++ly) {
        for (int lx = 0; lx < bs; ++lx) {
          const int local = lx + bs * (ly + bs * lz);
          const vr::Vec3i voxel = blk.coord * bs + vr::Vec3i(lx, ly, lz);
          const auto idx = static_cast<std::size_t>(blk.ptr) + local;
          tptr[idx] =
              ((voxel.x + voxel.y + voxel.z) % 2 == 0) ? -0.5f * h : 0.5f * h;
          wptr[idx] = 1.0f;
        }
      }
    }
  }
  return true;
}

// How a mesh's triangles are laid out in the arena relative to the blocks that
// produced them: how many distinct blocks it spans, and how many times the
// owning block CHANGES walking the index buffer in order. Per-block grouping
// gives EXACTLY `distinct - 1` transitions -- spans are contiguous and
// disjoint, so the owner changes once per span boundary and nowhere else --
// while full interleaving gives nearly one per triangle.
//
// Exactly, not approximately, because attributing a triangle by its CENTROID is
// exact here. The three vertices lie on the edges of one marching-cubes cell,
// so the centroid lies in that cell's closed cube; the cell's base voxel
// belongs to the block and the cube reaches at most to the block's far
// boundary, so the only centroid that could land in the NEXT block is one
// exactly on that boundary -- which takes all three vertices on the cell's far
// face, i.e. three sign changes around a 4-cycle, and sign changes around a
// cycle come in pairs. Two on the far face is the parity-legal maximum, and
// that centroid still sits a third of a voxel short of the boundary, which no
// float rounding closes.
//
// The owning block is keyed by its three coordinates rather than a packed
// integer: a packed key is injective only while coordinates stay non-negative
// and small, and a collision would lower BOTH numbers -- loosening the
// assertion at exactly the moment attribution stopped distinguishing blocks.
struct BlockLayout {
  std::size_t triangles = 0;
  std::size_t distinct = 0;
  std::size_t transitions = 0;
};

BlockLayout block_layout(const mesh::Mesh& m, int block_size,
                         float voxel_size) {
  const float block_span = static_cast<float>(block_size) * voxel_size;
  std::vector<std::array<long long, 3>> owner;
  owner.reserve(m.indices.size() / 3);
  for (std::size_t t = 0; t + 2 < m.indices.size(); t += 3) {
    vr::Vec3f c(0.0f, 0.0f, 0.0f);
    for (int k = 0; k < 3; ++k) {
      c = c + m.vertices[m.indices[t + k]].position;
    }
    c = c / 3.0f;
    owner.push_back({static_cast<long long>(std::floor(c.x / block_span)),
                     static_cast<long long>(std::floor(c.y / block_span)),
                     static_cast<long long>(std::floor(c.z / block_span))});
  }
  BlockLayout out;
  out.triangles = owner.size();
  out.distinct =
      std::set<std::array<long long, 3>>(owner.begin(), owner.end()).size();
  for (std::size_t i = 1; i < owner.size(); ++i) {
    if (owner[i] != owner[i - 1]) {
      ++out.transitions;
    }
  }
  return out;
}

// The same block attribution, applied to the VERTEX arena: which vertex indices
// each block's triangles reference.
//
// Needed because BlockLayout walks the index run and is therefore blind to
// where the vertices went. The sharing kernel reserves TWO ranges per block,
// and putting its vertex claims back on the global counter -- leaving the
// triangle span exactly as it is -- interleaves the arena again while every
// transition count above still passes.
//
// A block's own cells are the only ones that can reference the vertices it
// created: an in-block edge is looked up in this block's table, and one whose
// owner lies outside is duplicated locally. Marching cubes puts every crossed
// edge in its own cell's triangle row, so every vertex a block creates is
// referenced by a triangle of that block, and the set below is exactly the
// range the block reserved.
struct VertexSpans {
  std::size_t blocks = 0;
  std::size_t gaps = 0;      // blocks whose indices are not one dense range
  std::size_t overlaps = 0;  // ranges intersecting the one before them
  std::size_t covered = 0;   // vertices lying inside some block's range
};

VertexSpans vertex_spans(const mesh::Mesh& m, int block_size,
                         float voxel_size) {
  const float block_span = static_cast<float>(block_size) * voxel_size;
  std::map<std::array<long long, 3>, std::set<std::uint32_t>> used;
  for (std::size_t t = 0; t + 2 < m.indices.size(); t += 3) {
    vr::Vec3f c(0.0f, 0.0f, 0.0f);
    for (int k = 0; k < 3; ++k) {
      c = c + m.vertices[m.indices[t + k]].position;
    }
    c = c / 3.0f;
    std::set<std::uint32_t>& block =
        used[{static_cast<long long>(std::floor(c.x / block_span)),
              static_cast<long long>(std::floor(c.y / block_span)),
              static_cast<long long>(std::floor(c.z / block_span))}];
    for (int k = 0; k < 3; ++k) {
      block.insert(m.indices[t + k]);
    }
  }

  VertexSpans out;
  out.blocks = used.size();
  std::vector<std::pair<std::uint32_t, std::uint32_t>> ranges;
  ranges.reserve(used.size());
  for (const auto& entry : used) {
    const std::uint32_t lo = *entry.second.begin();
    const std::uint32_t hi = *entry.second.rbegin();
    const std::size_t extent = static_cast<std::size_t>(hi - lo) + 1;
    if (entry.second.size() != extent) ++out.gaps;
    out.covered += extent;
    ranges.push_back({lo, hi});
  }
  std::sort(ranges.begin(), ranges.end());
  for (std::size_t i = 1; i < ranges.size(); ++i) {
    if (ranges[i].first <= ranges[i - 1].second) ++out.overlaps;
  }
  return out;
}

// A mesh's triangles as a sorted, directly comparable list of their nine
// position floats. Marching cubes appends through an atomic, so two runs over
// the same field emit the same triangles in a different ORDER; each triangle's
// arithmetic is independent of that order, so the floats themselves are
// bit-identical and the sorted lists must match exactly.
// Resolved through the INDEX buffer, which is what makes this comparison a
// statement about the surface rather than about allocation order. It used to
// read consecutive vertex triples, valid only while every triangle owned three
// private vertices emitted in order; the kernel now claims vertices and
// triangles from two independent atomics. It is also what lets this same
// helper state the vertex-sharing invariant, where the triangle list must be
// unchanged while the vertex array behind it shrinks ~4x.
std::vector<std::array<float, 9>> canonical_triangles(const mesh::Mesh& m) {
  std::vector<std::array<float, 9>> tris;
  tris.reserve(m.indices.size() / 3);
  for (std::size_t i = 0; i + 2 < m.indices.size(); i += 3) {
    std::array<float, 9> t{};
    for (std::size_t k = 0; k < 3; ++k) {
      const mesh::Vertex& v = m.vertices[m.indices[i + k]];
      t[k * 3 + 0] = v.position.x;
      t[k * 3 + 1] = v.position.y;
      t[k * 3 + 2] = v.position.z;
    }
    tris.push_back(t);
  }
  std::sort(tris.begin(), tris.end());
  return tris;
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
    // The tangent placeholder, through the *sparse* kernel's own Vertex mirror
    // (see the dense test for why an unwritten slot is not merely zero).
    CHECK(v.tangent.x == 1.0f && v.tangent.y == 0.0f && v.tangent.z == 0.0f &&
          v.tangent.w == 1.0f);
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

  // --- A block's triangles are CONTIGUOUS in the arena -----------------------
  //
  // Stage 2's actual deliverable, and nothing else here can see it: the golden
  // sparse-vs-dense equivalence below compares triangles as a SET, so it passes
  // identically whether a block's triangles are grouped or scattered the length
  // of the arena. Under the per-triangle append they interleaved with every
  // other block in flight, and per-block ranges are the precondition for
  // meshing only the blocks a fuse changed.
  //
  // Asserted EXACTLY -- see block_layout for why centroid attribution admits no
  // slack here, and why a loose bound would be the wrong instrument: the
  // failure this guards against (a span overrun, or a block emitting through
  // two spans) moves the count by one, not by an order of magnitude.
  {
    const BlockLayout layout = block_layout(sphere, kBlock, kH);
    // The sphere must actually straddle many blocks, or "grouped" is vacuous.
    CHECK(layout.distinct >= 20);
    // ... and there must be many more triangles than blocks, or the assertion
    // below is satisfied by arithmetic rather than by grouping.
    CHECK(layout.triangles > 4 * layout.distinct);
    CHECK(layout.transitions == layout.distinct - 1);
  }

  // Winding agrees with the gradient normal, per face (a boundary seam or a
  // flipped table would drop this well below 1).
  int face_count = 0;
  int face_outward = 0;
  // Walked through the INDEX buffer, not as consecutive vertex triples. That
  // shorthand was valid only while the run was the identity 0,1,2,...; the
  // kernel now allocates vertices and triangles through two independent
  // atomics, so vertex order is not triangle order. Reading the indices is what
  // keeps this assertion about winding rather than about allocation order --
  // and it is what lets the same check keep its teeth once vertices are shared.
  for (std::size_t t = 0; t + 2 < sphere.indices.size(); t += 3) {
    const mesh::Vertex& a = sphere.vertices[sphere.indices[t + 0]];
    const mesh::Vertex& b = sphere.vertices[sphere.indices[t + 1]];
    const mesh::Vertex& c = sphere.vertices[sphere.indices[t + 2]];
    const vr::Vec3f face =
        vr::cross(b.position - a.position, c.position - a.position);
    if (vr::length(face) > 1e-8f) {
      if (vr::dot(vr::normalize(face), a.normal) > 0.0f) {
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

  // --- The counter's UNITS, pinned against the fixture's analytic area -------
  // Everything above compares the kernel with itself: dense and sparse share
  // mcEmitCell, so a mistake in what the append atomic counts moves both sides
  // equally and every equality still holds. Measured, not assumed -- reverting
  // `atomicAdd(index_count, kIndicesPerTriangle) / kIndicesPerTriangle` to a
  // plain `atomicAdd(..., 1u)` left this whole suite green.
  //
  // The command's indexCount is now the draw's, so counting triangles instead
  // of indices makes the host's `/ kIndicesPerTriangle` report a THIRD of the
  // surface -- a mesh that still passes every per-vertex check, because the
  // triangles it does report are correct. What it cannot do is still cover the
  // sphere: total triangle area is an analytic property of the fixture
  // (4*pi*r^2), independent of anything the counter says, so a 3x undercount
  // fails by a mile. The tolerance is wide because marching cubes chords a
  // curved surface and the field is sampled at kH -- it is sized to catch a
  // factor of three, not to measure discretisation error.
  // Through the indices, for the reason given at the winding check above: a
  // vertex triple is a triangle only under an identity index run, which the
  // kernel no longer produces. Keeping this assertion honest matters more than
  // most -- it is the one that pins the counter's UNITS against a quantity no
  // counter participates in, so it must measure the real surface.
  double area = 0.0;
  for (std::size_t i = 0; i + 2 < sphere.indices.size(); i += 3) {
    const vr::Vec3f& a = sphere.vertices[sphere.indices[i + 0]].position;
    const vr::Vec3f& b = sphere.vertices[sphere.indices[i + 1]].position;
    const vr::Vec3f& c = sphere.vertices[sphere.indices[i + 2]].position;
    area += 0.5 * static_cast<double>(vr::length(vr::cross(b - a, c - a)));
  }
  const double analytic_area = 4.0 * 3.14159265358979323846 *
                               static_cast<double>(kRadius) *
                               static_cast<double>(kRadius);
  CHECK(area > 0.80 * analytic_area);
  CHECK(area < 1.20 * analytic_area);

  // --- The collision chain the on-device probe walks
  // -------------------------- Everything above meshes through
  // sphere_grid_params(), whose 8-entry buckets over 128 buckets never
  // overflow: the 216 block coords reach a maximum bucket occupancy of 4, so
  // allocate_in_overflow is never entered and NOT ONE entry is reachable only
  // through a chain. That leaves the chain half of vrFindBlockPtr -- the
  // subtlest code in the kernel, and the half that has no host-side counterpart
  // to have been proven against -- executed by nothing. Measured, not assumed:
  // replacing the chain walk with `return -1;` left this whole suite green, and
  // every other suite too.
  //
  // Production is the overflow case (VoxelGridParams::defaults() is 50x30000
  // with max_chain 128, and HashDiagnostics::overflow_count exists precisely
  // because buckets fill), so mesh the SAME field through a table shaped to
  // spill: 2-entry buckets make the last slot of each bucket its chain anchor,
  // so a bucket holding two coords already pushes one into the chain. The block
  // set, the field and therefore the surface are identical -- only the table
  // geometry differs -- so the mesh must match the reference triangle for
  // triangle. Under the mutation above it comes back short and returns
  // Status::ok, which is the failure mode this exists to catch: surface
  // silently missing at block seams.
  vol::VoxelGridParams chained_gp = sphere_grid_params();
  chained_gp.bucket_size = 2;
  chained_gp.num_buckets = 512;  // num_blocks unchanged at 1024
  vr::Result<vol::VoxelBlockGrid> chained_result = vol::VoxelBlockGrid::create(
      device.value(), allocator.value(), chained_gp, attrs, 2);
  CHECK(chained_result.ok());
  vol::VoxelBlockGrid chained_grid = std::move(chained_result).value();
  CHECK(fill_sphere_grid(chained_grid, /*with_color=*/false));

  // The fixture only tests what it exercises, so assert that it spills before
  // trusting what it proves -- otherwise a later change to the hash or to these
  // numbers turns this case vacuous without failing.
  vr::Result<vol::HashDiagnostics> chained_diag =
      chained_grid.map().diagnostics();
  CHECK(chained_diag.ok());
  CHECK(chained_diag.value().overflow_count > 0);
  CHECK(chained_diag.value().max_chain_length > 0);

  vr::Result<mesh::Mesh> chained_mesh_result =
      extractor.extract(chained_grid, 0.0f);
  CHECK(chained_mesh_result.ok());
  const mesh::Mesh chained_mesh = std::move(chained_mesh_result).value();
  CHECK(chained_mesh.triangle_count() == sphere.triangle_count());
  CHECK(canonical_triangles(chained_mesh) == canonical_triangles(sphere));

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
    // Compared in ENCODED space: `Vertex::color` is linear working values while
    // the gradient was written as canonical-encoded 8-bit, and inverting the
    // vertex keeps the u8-floor tolerance meaningful (one code is a flat 1/255
    // encoded, but ~0.0089 in linear near white). See the dense test for the
    // full note.
    const vr::Vec3f expected = grad_color(v.position);
    const vr::Vec3f encoded = vr::linear_to_srgb(vr::Vec3f(v.color));
    CHECK(std::fabs(encoded.x - expected.x) < 0.005f);  // ~2.5x the u8 floor
    CHECK(std::fabs(encoded.y - expected.y) < 0.005f);
    CHECK(std::fabs(encoded.z - expected.z) < 0.005f);
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
  // reallocated nor shrunk, and the dispatch runs at its full capacity (so it
  // never drops a triangle the arena had room for).
  mesh::ExtractTimings reuse_timings;
  CHECK(arena_mc.extract(one_grid, 0.0f, &reuse_timings).ok());
  CHECK(reuse_timings.triangle_capacity == big_timings.triangle_capacity);
  CHECK(reuse_timings.arena_bytes == big_timings.arena_bytes);
  CHECK(reuse_timings.dispatches == 1);  // it already fits: no refit

  // --- Refit and re-run when the planned capacity undershoots ----------------
  // The arena is fitted to the surface, so a call CAN plan too small -- and the
  // recovery is the load-bearing part of that trade. It rests on the kernel
  // counting every triangle the field produces, dropped ones included, so an
  // undersized arena reports the exact size it needed; a kernel that stopped at
  // the first drop would report a lower bound, the refit would still be too
  // small, and the retry would overflow again (an out_of_memory below).
  //
  // Forced by density, not by a tuned number: one block of a sign-alternating
  // field emits ~1400 triangles where a first extract plans ~64 per block, so
  // the first call MUST refit and re-run. The second call over the same grid
  // then plans from the density the first one measured, so it must not.
  vr::Result<mesh::MarchingCubes> refit_result =
      mesh::MarchingCubes::create(device.value(), allocator.value());
  CHECK(refit_result.ok());
  mesh::MarchingCubes refit_mc = std::move(refit_result).value();

  vr::Result<vol::VoxelBlockGrid> dense_block_result =
      vol::VoxelBlockGrid::create(device.value(), allocator.value(), gp, attrs,
                                  2);
  CHECK(dense_block_result.ok());
  vol::VoxelBlockGrid dense_block = std::move(dense_block_result).value();
  CHECK(fill_dense_blocks(dense_block, 1));

  mesh::ExtractTimings refit_timings;
  vr::Result<mesh::Mesh> refit_mesh_result =
      refit_mc.extract(dense_block, 0.0f, &refit_timings);
  CHECK(refit_mesh_result.ok());
  const mesh::Mesh refit_mesh = std::move(refit_mesh_result).value();
  CHECK(refit_timings.dispatches == 2);  // planned short, measured, re-ran
  CHECK(refit_timings.emitted_triangles > refit_timings.active_blocks * 64);
  CHECK(refit_mesh.triangle_count() == refit_timings.emitted_triangles);
  // The refitted arena holds the whole surface, with the growth headroom on
  // top of the measured count.
  CHECK(refit_timings.triangle_capacity >= refit_timings.emitted_triangles);
  CHECK(refit_timings.arena_bytes >= needed_bytes(refit_timings));

  // Same grid again: the measured density now plans it in ONE dispatch, and
  // the geometry is identical triangle-for-triangle -- so the retried mesh lost
  // nothing and duplicated nothing, which a truncated or double-counted first
  // pass would both break.
  mesh::ExtractTimings settled_timings;
  vr::Result<mesh::Mesh> settled_result =
      refit_mc.extract(dense_block, 0.0f, &settled_timings);
  CHECK(settled_result.ok());
  const mesh::Mesh settled_mesh = std::move(settled_result).value();
  CHECK(settled_timings.dispatches == 1);
  CHECK(settled_timings.emitted_triangles == refit_timings.emitted_triangles);
  CHECK(canonical_triangles(refit_mesh) == canonical_triangles(settled_mesh));

  // --- ...and the same overflow across MANY blocks ---------------------------
  //
  // A distinct path since the kernel began reserving one span per block, and
  // the single-block fixture above cannot reach it. With one block every drop
  // is a suffix of one span, which is what the per-triangle append did
  // everywhere. With 27 blocks the arena boundary falls in three different
  // places at once: blocks whose span fits entirely, ONE whose span straddles
  // `capacity` (its first triangles written, its last dropped), and blocks
  // whose span begins wholly past it -- the case the kernel now rejects up
  // front rather than one triangle at a time.
  //
  // The demand is what makes it deterministic: 27 blocks x ~1400 triangles
  // against the ~64 per block a first extract plans, so every one of those
  // three cases is occupied rather than hoped for.
  vr::Result<vol::VoxelBlockGrid> dense_run_result =
      vol::VoxelBlockGrid::create(device.value(), allocator.value(), gp, attrs,
                                  2);
  CHECK(dense_run_result.ok());
  vol::VoxelBlockGrid dense_run = std::move(dense_run_result).value();
  CHECK(fill_dense_blocks(dense_run, 3));

  vr::Result<mesh::MarchingCubes> run_result =
      mesh::MarchingCubes::create(device.value(), allocator.value());
  CHECK(run_result.ok());
  mesh::MarchingCubes run_mc = std::move(run_result).value();

  mesh::ExtractTimings run_timings;
  vr::Result<mesh::Mesh> run_mesh_result =
      run_mc.extract(dense_run, 0.0f, &run_timings);
  CHECK(run_mesh_result.ok());
  const mesh::Mesh run_mesh = std::move(run_mesh_result).value();
  CHECK(run_timings.active_blocks == 27);
  CHECK(run_timings.dispatches == 2);  // planned short, measured, re-ran
  CHECK(run_timings.emitted_triangles > run_timings.active_blocks * 64);
  // Nothing lost and nothing duplicated by the overflow: the retry emitted
  // exactly the count the overflowing pass reported, which is the whole
  // contract that pass exists to uphold.
  CHECK(run_mesh.triangle_count() == run_timings.emitted_triangles);
  CHECK(run_timings.triangle_capacity >= run_timings.emitted_triangles);

  // The same field again, now planned in one dispatch, must be the same
  // surface triangle-for-triangle -- so a span dropped at the boundary was
  // dropped, not misplaced into a neighbouring block's.
  mesh::ExtractTimings run_settled_timings;
  vr::Result<mesh::Mesh> run_settled_result =
      run_mc.extract(dense_run, 0.0f, &run_settled_timings);
  CHECK(run_settled_result.ok());
  const mesh::Mesh run_settled = std::move(run_settled_result).value();
  CHECK(run_settled_timings.dispatches == 1);
  CHECK(canonical_triangles(run_mesh) == canonical_triangles(run_settled));

  // And contiguity survives a refit: the assertion above runs only on a clean
  // first extract, where no span was ever truncated. Same exact figure here --
  // `distinct - 1`, against the ~48 000 transitions full interleaving would
  // give at this triangle count -- because truncating a span at the arena
  // boundary shortens it without splitting it.
  {
    const BlockLayout layout = block_layout(run_settled, kBlock, kH);
    CHECK(layout.distinct >= 20);
    CHECK(layout.triangles > 4 * layout.distinct);
    CHECK(layout.transitions == layout.distinct - 1);
  }

  // Reusing one ExtractTimings across calls must not accumulate: every field is
  // overwritten, so the second call's spans are its own. (dispatch_ms and
  // readback_ms sum over a call's attempts internally, which is exactly why
  // they have to start from zero.)
  mesh::ExtractTimings reused_stats = refit_timings;
  CHECK(refit_mc.extract(dense_block, 0.0f, &reused_stats).ok());
  CHECK(reused_stats.dispatches == 1);
  CHECK(reused_stats.emitted_triangles == refit_timings.emitted_triangles);

  // --- Vertex sharing --------------------------------------------------------
  // Nothing above reaches MarchingCubesConfig::share_vertices: it selects a
  // second compiled kernel, so with no test constructing that config the whole
  // owned-edge pass, its barrier, the shared-slot lookup, the block-face
  // duplicate path and the host's separate vertex budget were dead to ctest --
  // `bool sharing = false;` left every suite green.
  mesh::MarchingCubesConfig share_config;
  share_config.share_vertices = true;
  vr::Result<mesh::MarchingCubes> share_result = mesh::MarchingCubes::create(
      device.value(), allocator.value(), share_config);
  CHECK(share_result.ok());
  mesh::MarchingCubes share_mc = std::move(share_result).value();

  mesh::ExtractTimings share_timings;
  vr::Result<mesh::Mesh> share_mesh_result =
      share_mc.extract(grid, 0.0f, &share_timings);
  CHECK(share_mesh_result.ok());
  const mesh::Mesh share_mesh = std::move(share_mesh_result).value();

  // THE invariant, and the one worth having: sharing moves the vertex count and
  // nothing else. Compared through the index buffer on exact position floats,
  // so it is a statement about the surface -- and it is what catches a
  // traversal-direction mismatch between the two emitters. kEdgeToVert lists
  // edges 2/3/6/7 max-corner-first, so an owner cell and the neighbour that
  // duplicates the same edge walk it in OPPOSITE directions unless
  // mcEdgeVertex orders the endpoints canonically; `mix` and the near-tangent
  // guard are both direction-dependent, and under the guard the two land at
  // opposite ends of the edge -- a whole voxel apart.
  CHECK(share_mesh.triangle_count() == sphere.triangle_count());
  CHECK(canonical_triangles(share_mesh) == canonical_triangles(sphere));

  // The unshared mesh is three private vertices per triangle; the shared one is
  // materially fewer. Asserted as a ratio rather than a fixed number so it
  // states the property (in-block sharing, ~4x) instead of pinning this
  // fixture's arithmetic -- but tightly enough that a kernel which shared
  // nothing, or shared only within a cell, would fail.
  CHECK(sphere.vertices.size() == sphere.indices.size());  // 3 per triangle
  CHECK(share_mesh.vertices.size() * 2 < sphere.vertices.size());

  // ... and this kernel's triangles are per-block contiguous too, which is what
  // lets a dirty-only dispatch describe a block's output as a range. It was NOT
  // true here until now: the default kernel reserved a span per block while
  // this one still appended per triangle through the global counter, so
  // `share_vertices` was the one path incremental extraction could not use.
  //
  // Exactly `distinct - 1` transitions, the same bound the unshared path is
  // held to -- spans are contiguous and disjoint, so the owner changes once per
  // boundary and nowhere else.
  //
  // And the arena the same way, which the transition count above cannot see:
  // each block's vertices are one dense range, the ranges do not overlap, and
  // together they tile the whole arena.
  {
    const BlockLayout layout = block_layout(share_mesh, kBlock, kH);
    CHECK(layout.distinct >= 20);
    CHECK(layout.triangles > 4 * layout.distinct);
    CHECK(layout.transitions == layout.distinct - 1);

    const VertexSpans spans = vertex_spans(share_mesh, kBlock, kH);
    CHECK(spans.blocks == layout.distinct);
    CHECK(spans.gaps == 0);
    CHECK(spans.overlaps == 0);
    CHECK(spans.covered == share_mesh.vertices.size());
  }
  CHECK(share_timings.emitted_vertices == share_mesh.vertices.size());
  CHECK(share_timings.emitted_triangles == share_mesh.triangle_count());
  // Every index addresses a live vertex. With sharing this is a real check
  // rather than a restatement: the run is written by the kernel from a vertex
  // counter independent of the triangle counter.
  for (std::uint32_t i : share_mesh.indices) {
    CHECK(i < share_mesh.vertices.size());
  }
  // At least one vertex really is shared -- i.e. some index appears more than
  // once. Implied by the count ratio above, but stated directly because it is
  // the property, and a mesh of unique-but-fewer vertices would be a different
  // bug.
  {
    std::vector<std::uint32_t> uses(share_mesh.vertices.size(), 0);
    for (std::uint32_t i : share_mesh.indices) ++uses[i];
    std::size_t multi = 0;
    for (std::uint32_t u : uses) {
      CHECK(u > 0);  // no vertex emitted and then referenced by nothing
      if (u > 1) ++multi;
    }
    CHECK(multi > share_mesh.vertices.size() / 2);
  }

  // The device view says so, which is what texture::ProjectiveTexturer refuses
  // on -- per-triangle visibility cannot be written to a per-vertex uv0 that
  // several triangles share.
  vr::Result<mesh::DeviceMesh> share_device =
      share_mc.extract_device(grid, 0.0f);
  CHECK(share_device.ok());
  CHECK(share_device.value().shares_vertices);
  CHECK(share_device.value().vertex_count == share_timings.emitted_vertices);
  CHECK(share_mc.download(share_device.value()).ok());

  vr::Result<mesh::DeviceMesh> plain_device =
      extractor.extract_device(grid, 0.0f);
  CHECK(plain_device.ok());
  CHECK(!plain_device.value().shares_vertices);
  CHECK(plain_device.value().vertex_count ==
        plain_device.value().triangle_count * 3);

  // A dropped vertex must be counted ONCE, by the cell that owns its edge, and
  // not re-claimed by every cell that references it. Forced through the same
  // sign-alternating block that forces the triangle refit: the seed budget is
  // ~45 vertices per block against ~1000 the field needs, so the first dispatch
  // drops nearly all of them.
  //
  // The assertion is a MAGNITUDE, because the shape passes either way: the
  // second dispatch has room, so its counter is the true total whatever the
  // first one reported. What the first one reports is what the arena is REFIT
  // to, and the arena is grow-only -- so re-claiming per reference reports
  // ~3 vertices per triangle instead of the true ~0.75 and pins an arena ~6x
  // the surface, which is more than not sharing at all.
  vr::Result<mesh::MarchingCubes> share_refit_result =
      mesh::MarchingCubes::create(device.value(), allocator.value(),
                                  share_config);
  CHECK(share_refit_result.ok());
  mesh::MarchingCubes share_refit_mc = std::move(share_refit_result).value();
  mesh::ExtractTimings share_refit_timings;
  vr::Result<mesh::Mesh> share_refit_result_mesh =
      share_refit_mc.extract(dense_block, 0.0f, &share_refit_timings);
  CHECK(share_refit_result_mesh.ok());
  const mesh::Mesh share_refit_mesh =
      std::move(share_refit_result_mesh).value();
  CHECK(share_refit_timings.dispatches ==
        2);  // planned short, measured, re-ran
  CHECK(share_refit_timings.emitted_vertices > 0);
  CHECK(share_refit_timings.vertex_capacity >=
        share_refit_timings.emitted_vertices);
  CHECK(share_refit_timings.vertex_capacity <=
        2 * share_refit_timings.emitted_vertices);
  // And the MESH the refit returned, not just its magnitudes. This fixture is
  // the only one that reaches the duplicate count at all -- the sphere's edge
  // owners are all valid, so its `else` branch never fires -- and a mesh nobody
  // looks at made both of that branch's guards deletable with the suite green.
  // The unshared extract of the same field is the oracle: an owner miscounted
  // is a vertex slot over-consumed, which the reservation bound turns into
  // dropped geometry rather than a write into the next block's range.
  //
  // Bounds FIRST, then the geometry: a triangle the kernel counted but declined
  // to write leaves its three index slots holding whatever the VMA block last
  // did, and `canonical_triangles` resolves every index through the vertex
  // array. Checked the other way round, the diagnosis is a bus error inside a
  // test helper rather than a named line.
  for (std::uint32_t i : share_refit_mesh.indices) {
    CHECK(i < share_refit_mesh.vertices.size());
  }
  CHECK(canonical_triangles(share_refit_mesh) ==
        canonical_triangles(settled_mesh));

  // --- ...and the same, shared, across MANY blocks ---------------------------
  //
  // `dense_block` is ONE block, so a cursor that overran its reservation has no
  // next block to land in: unobservable there in principle, whichever way the
  // count is wrong. Twenty-seven of them, refitting, is where both ranges have
  // a neighbour to run into.
  vr::Result<mesh::MarchingCubes> share_run_result =
      mesh::MarchingCubes::create(device.value(), allocator.value(),
                                  share_config);
  CHECK(share_run_result.ok());
  mesh::MarchingCubes share_run_mc = std::move(share_run_result).value();

  mesh::ExtractTimings share_run_timings;
  vr::Result<mesh::Mesh> share_run_result_mesh =
      share_run_mc.extract(dense_run, 0.0f, &share_run_timings);
  CHECK(share_run_result_mesh.ok());
  const mesh::Mesh share_run_mesh = std::move(share_run_result_mesh).value();
  CHECK(share_run_timings.active_blocks == 27);
  CHECK(share_run_timings.dispatches == 2);  // planned short, measured, re-ran
  for (std::uint32_t i : share_run_mesh.indices) {
    CHECK(i < share_run_mesh.vertices.size());
  }
  CHECK(canonical_triangles(share_run_mesh) ==
        canonical_triangles(run_settled));
  {
    const BlockLayout layout = block_layout(share_run_mesh, kBlock, kH);
    CHECK(layout.distinct >= 20);
    CHECK(layout.transitions == layout.distinct - 1);

    const VertexSpans spans = vertex_spans(share_run_mesh, kBlock, kH);
    CHECK(spans.blocks == layout.distinct);
    CHECK(spans.gaps == 0);
    CHECK(spans.overlaps == 0);
    CHECK(spans.covered == share_run_mesh.vertices.size());
  }

  // Both ranges survive the refit: the same field planned in one dispatch is
  // the same surface, and still one range each per block.
  mesh::ExtractTimings share_run_settled_timings;
  vr::Result<mesh::Mesh> share_run_settled_result =
      share_run_mc.extract(dense_run, 0.0f, &share_run_settled_timings);
  CHECK(share_run_settled_result.ok());
  const mesh::Mesh share_run_settled =
      std::move(share_run_settled_result).value();
  CHECK(share_run_settled_timings.dispatches == 1);
  CHECK(canonical_triangles(share_run_settled) ==
        canonical_triangles(share_run_mesh));
  {
    const VertexSpans spans = vertex_spans(share_run_settled, kBlock, kH);
    CHECK(spans.gaps == 0);
    CHECK(spans.overlaps == 0);
    CHECK(spans.covered == share_run_settled.vertices.size());
  }

  // Growing one output buffer must not resize the other. They used to be
  // released and reallocated together -- justified by arena_capacity() coming
  // off the ARENA, which stopped being true when sharing moved it to the index
  // run -- and the coupling then grew the buffer that FITTED to 1.5x what it
  // already held, compounding on every such event: numerically the ring runaway
  // the slot-independence decision exists to prevent, one buffer over.
  //
  // Reachable only where the two budgets are out of proportion, which is what
  // sharing makes them: this extractor holds ~0.75 vertices per triangle, and
  // the DENSE overload (which shares nothing, and shares this extractor's
  // slot) asks for exactly 3. Sized from the measured capacities so the case is
  // constructed rather than hoped for -- the arena must be short and the index
  // run must not.
  mesh::ExtractTimings before_dense;
  CHECK(share_mc.extract(grid, 0.0f, &before_dense).ok());
  CHECK(before_dense.vertex_capacity > 0 && before_dense.triangle_capacity > 0);
  {
    // Triangles whose 3-per-triangle vertices overflow the held arena while
    // fitting the held index run: anything strictly between the two bounds.
    const std::uint64_t lower = before_dense.vertex_capacity / 3;
    const std::uint64_t upper = before_dense.triangle_capacity;
    CHECK(lower < upper);  // sharing is what makes this window exist
    const std::uint64_t target_tris = (lower + upper) / 2;
    int dims = 2;
    while (static_cast<std::uint64_t>(dims - 1) * (dims - 1) * (dims - 1) * 5 <
           target_tris) {
      ++dims;
    }
    const std::uint64_t dense_tris =
        static_cast<std::uint64_t>(dims - 1) * (dims - 1) * (dims - 1) * 5;
    // Non-vacuous by construction, and checked rather than assumed: if the
    // capacities ever drift out of this window the test fails loudly instead of
    // quietly proving nothing.
    CHECK(dense_tris * 3 > before_dense.vertex_capacity);
    CHECK(dense_tris <= before_dense.triangle_capacity);

    std::vector<vol::Voxel> sub(static_cast<std::size_t>(dims) * dims * dims);
    for (int z = 0; z < dims; ++z) {
      for (int y = 0; y < dims; ++y) {
        for (int x = 0; x < dims; ++x) {
          const vr::Vec3f p(static_cast<float>(x) * kH,
                            static_cast<float>(y) * kH,
                            static_cast<float>(z) * kH);
          vol::Voxel& v =
              sub[static_cast<std::size_t>(x + dims * (y + dims * z))];
          v.sdf = sphere_sdf(p);
          v.weight = 1.0f;
        }
      }
    }
    mesh::DenseGrid sub_grid;
    sub_grid.dims = vr::Vec3i(dims, dims, dims);
    sub_grid.voxel_size = kH;
    sub_grid.origin = vr::Vec3f(0.0f, 0.0f, 0.0f);
    CHECK(share_mc.extract(sub.data(), sub.size(), sub_grid, 0.0f).ok());

    mesh::ExtractTimings after_dense;
    CHECK(share_mc.extract(grid, 0.0f, &after_dense).ok());
    // The arena grew (the dense call needed three vertices per triangle)...
    CHECK(after_dense.vertex_capacity > before_dense.vertex_capacity);
    // ...and the index run, which already fitted, was left exactly alone.
    CHECK(after_dense.triangle_capacity == before_dense.triangle_capacity);
  }

  // A block this kernel's compile-time cell table cannot index is refused up
  // front, not meshed wrong: the shared array is sized for block_size 8, and an
  // out-of-bounds threadgroup write is invisible to every validation layer and
  // reads back as a plausible mesh.
  vol::VoxelGridParams big_block_gp = sphere_grid_params();
  big_block_gp.block_size = 16;
  big_block_gp.voxels_per_block = 16 * 16 * 16;  // 4096 > the kernel's 1024
  big_block_gp.num_buckets = 32;
  big_block_gp.num_blocks = 256;  // = bucket_size * num_buckets
  vr::Result<vol::VoxelBlockGrid> big_block_result =
      vol::VoxelBlockGrid::create(device.value(), allocator.value(),
                                  big_block_gp, attrs, 2);
  CHECK(big_block_result.ok());
  vol::VoxelBlockGrid big_block_grid = std::move(big_block_result).value();
  // FILLED, not merely allocated. An allocated-but-unintegrated block has
  // weight 0 at every corner, so every cell is rejected at the gather and the
  // block emits nothing -- which means the default kernel returns at its
  // "nothing reserved" check and the uncached tail this grid exists to reach
  // never runs. With the field written, cells 1024..4095 fall past the four
  // slots the kernel's per-cell cache holds and take the second full gather
  // instead of the cheap register rejection, and only then is the uncached
  // branch exercised at all.
  CHECK(fill_dense_blocks(big_block_grid, 1));
  CHECK(!share_mc.extract(big_block_grid, 0.0f).ok());
  // The same grid is fine without sharing -- the refusal is the kernel's table,
  // not the block size.
  mesh::ExtractTimings big_timings_16;
  vr::Result<mesh::Mesh> big_mesh_16_result =
      arena_mc.extract(big_block_grid, 0.0f, &big_timings_16);
  CHECK(big_mesh_16_result.ok());
  const mesh::Mesh big_mesh_16 = std::move(big_mesh_16_result).value();
  CHECK(big_timings_16.active_blocks == 1);
  CHECK(big_mesh_16.triangle_count() > 0);
  // The shortfall is REPORTED rather than refused: correct mesh, ~1.8 gathers
  // per cell instead of ~1.1, and nothing else could tell a caller so.
  CHECK(big_timings_16.uncached_cells_per_block == 4096 - 1024);

  // The block size is an allocation detail, so the SAME field divided into
  // eight blocks of 8 must extract to the same surface. This is what pins the
  // per-cell cache as an optimisation: it changes which cells are rejected
  // cheaply and which are gathered twice, and must change nothing else. The
  // outer +face cells reach a corner at voxel 16, unallocated under either
  // division, so both meshes stop in the same place.
  vol::VoxelGridParams split_gp = sphere_grid_params();
  split_gp.num_buckets = 32;
  split_gp.num_blocks = 256;
  vr::Result<vol::VoxelBlockGrid> split_result = vol::VoxelBlockGrid::create(
      device.value(), allocator.value(), split_gp, attrs, 2);
  CHECK(split_result.ok());
  vol::VoxelBlockGrid split_grid = std::move(split_result).value();
  CHECK(fill_dense_blocks(split_grid, 2));  // 2x2x2 blocks of 8 = the same 16^3
  mesh::ExtractTimings split_timings;
  vr::Result<mesh::Mesh> split_mesh_result =
      arena_mc.extract(split_grid, 0.0f, &split_timings);
  CHECK(split_mesh_result.ok());
  const mesh::Mesh split_mesh = std::move(split_mesh_result).value();
  CHECK(split_timings.active_blocks == 8);
  CHECK(split_timings.uncached_cells_per_block == 0);  // 512 cells, all cached
  CHECK(canonical_triangles(big_mesh_16) == canonical_triangles(split_mesh));

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
