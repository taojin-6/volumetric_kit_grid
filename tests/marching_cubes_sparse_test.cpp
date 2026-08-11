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
#include "volumetric_kit/recon/core/compute_util.hpp"
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
                      float weight_value = 1.0f, float radius = kRadius) {
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
          tptr[idx] = vr::length(world - sphere_center()) - radius;
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

// Does the published span table exactly describe @p m?
//
// This is what makes the table verifiable at all rather than state nothing
// reads. Four properties, and the last two are the ones with teeth:
//   1. the counts sum to the mesh,
//   2. the triangle ranges PARTITION it -- disjoint, and covering [0, total)
//      with no gap,
//   3. every triangle inside a block's range actually belongs to that block,
//      by the same centroid attribution block_layout uses,
//   4. every vertex those triangles reference lies in the same span's VERTEX
//      range.
// (1) and (2) would both hold if the kernel published plausible arithmetic that
// pointed at the wrong geometry; (3) is what ties a span to its block, and (4)
// is the only thing anywhere that reads BlockSpan::vertex_base.
bool spans_describe(const mesh::MarchingCubes& mc, const mesh::Mesh& m,
                    const std::vector<vol::BlockIndex>& active, int block_size,
                    float voxel_size) {
  const auto vpb =
      static_cast<std::uint32_t>(block_size * block_size * block_size);
  const mesh::BlockSpan* spans = mc.block_spans();
  if (spans == nullptr || m.indices.empty()) {
    return false;  // nothing published, or nothing to describe
  }
  const float block_span = static_cast<float>(block_size) * voxel_size;

  std::vector<bool> tri_seen(m.indices.size() / 3, false);
  std::uint64_t tri_total = 0;
  std::uint64_t vert_total = 0;
  for (const vol::BlockIndex& b : active) {
    const std::uint32_t slot = static_cast<std::uint32_t>(b.ptr) / vpb;
    if (slot >= mc.block_span_capacity()) {
      return false;
    }
    const mesh::BlockSpan sp = spans[slot];
    tri_total += sp.triangle_count;
    vert_total += sp.vertex_count;
    for (std::uint32_t i = 0; i < sp.triangle_count; ++i) {
      // Widened BEFORE the addition, not after. Both operands are uint32, so
      // `sp.triangle_base + i` is evaluated at unsigned int rank and converted
      // only then: a base near UINT32_MAX would wrap to a small `t` and pass
      // the range test below, turning a corrupt span into a mis-attribution
      // report that reads like a kernel bug.
      const std::size_t t = static_cast<std::size_t>(sp.triangle_base) + i;
      if (t >= tri_seen.size() || tri_seen[t]) {
        return false;  // out of range, or two blocks claim the same triangle
      }
      tri_seen[t] = true;
      vr::Vec3f c(0.0f, 0.0f, 0.0f);
      for (int k = 0; k < 3; ++k) {
        // The index values are device output too, and under share_vertices they
        // are written by the kernel rather than being the host's identity run
        // -- so they are checked here rather than trusted. Without this the
        // dereference below is the first thing an out-of-range index touches,
        // and it aborts inside this helper under ASan, pointing the reader at
        // the span table instead of at the index run.
        const std::uint32_t vi = m.indices[t * 3 + static_cast<std::size_t>(k)];
        if (vi >= m.vertices.size()) {
          return false;
        }
        // ... and it lies in the VERTEX range the same span claims. This is the
        // only thing that reads vertex_base at all: the counts summing and the
        // triangle ranges tiling both hold with vertex_base left at 0 for every
        // block, and under share_vertices the vertex range is independently
        // reserved, so it is exactly where a second atomic can drift from the
        // first. A block's own cells are the only ones that can reference the
        // vertices it created -- an in-block edge resolves in this block's
        // table, one owned outside is duplicated locally.
        if (vi < sp.vertex_base || vi - sp.vertex_base >= sp.vertex_count) {
          return false;
        }
        c = c + m.vertices[vi].position;
      }
      c = c / 3.0f;
      if (static_cast<long long>(std::floor(c.x / block_span)) != b.coord.x ||
          static_cast<long long>(std::floor(c.y / block_span)) != b.coord.y ||
          static_cast<long long>(std::floor(c.z / block_span)) != b.coord.z) {
        return false;  // the span points at another block's geometry
      }
    }
  }
  // Every triangle is claimed exactly once: `tri_seen` refuses a second claim
  // above, so a count equal to the total means the ranges tile it. A separate
  // sweep for unseen entries would be dead code -- N distinct claims over N
  // slots leaves none.
  return tri_total == tri_seen.size() && vert_total == m.vertices.size();
}

// Slots block_span_valid() reports live for @p g -- what the "exactly the
// active set, and nothing else" assertions are made against. Swept over the
// whole capacity rather than over the active set, because the interesting
// failure is a slot OUTSIDE it answering true.
std::size_t count_valid_spans(const mesh::MarchingCubes& mc,
                              const vol::VoxelBlockGrid& g) {
  std::size_t n = 0;
  for (std::uint32_t i = 0; i < mc.block_span_capacity(); ++i) {
    if (mc.block_span_valid(g, i)) {
      ++n;
    }
  }
  return n;
}

// Drop the zero-area triangles an incremental extract leaves behind.
//
// Retiring a range writes three identical vertices rather than compacting the
// index run -- see the kernel's phase four -- so a block that shrank or
// relocated leaves degenerate triangles in the arena. They are culled before
// rasterisation and they are not geometry, but `download` copies the arena, so
// a comparison against a full extract has to ignore them. Returns the count
// dropped, because "none were left" is itself worth asserting: it says the
// retire pass never ran.
std::size_t drop_degenerate(std::vector<std::array<float, 9>>& tris) {
  const std::size_t before = tris.size();
  tris.erase(std::remove_if(tris.begin(), tris.end(),
                            [](const std::array<float, 9>& t) {
                              for (int i = 0; i < 3; ++i) {
                                if (t[i] != t[3 + i] || t[i] != t[6 + i]) {
                                  return false;
                                }
                              }
                              return true;
                            }),
             tris.end());
  return before - tris.size();
}

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

  // The main extractor asks for the span table; most of the fixtures below do
  // not, which is the point -- track_block_spans is off by default and the
  // suite exercises both sides of that.
  mesh::MarchingCubesConfig spans_config;
  spans_config.track_block_spans = true;
  vr::Result<mesh::MarchingCubes> mc_result = mesh::MarchingCubes::create(
      device.value(), allocator.value(), spans_config);
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

  // The published span table describes that layout exactly -- the mapping the
  // reservation computes, which is not derivable on the host because the atomic
  // hands spans out in workgroup arrival order rather than block order.
  {
    vr::Result<std::vector<vol::BlockIndex>> active =
        grid.map().compact_active_blocks();
    CHECK(active.ok());
    CHECK(spans_describe(extractor, sphere, active.value(), kBlock, kH));
    // The table describes the extract that wrote it, and says which one that
    // was. A consumer holding a DeviceMesh compares this against its own
    // generation -- there is one table for the whole ring, so above one slot
    // this is what keeps a span from being read against another slot's arena.
    CHECK(extractor.block_spans_generation() != 0);
    CHECK(extractor.block_span_capacity() >=
          static_cast<std::uint32_t>(active.value().size()));

    // The anchor. A span is keyed by block SLOT, which names a block only
    // against a particular grid and topology epoch -- the heap is LIFO, so
    // after a remove() a reused slot names a DIFFERENT block and its span would
    // read as that block's geometry under Status::ok.
    const auto vpb = static_cast<std::uint32_t>(kBlock * kBlock * kBlock);
    const std::uint32_t some_slot =
        static_cast<std::uint32_t>(active.value().front().ptr) / vpb;
    CHECK(extractor.block_span_valid(grid, some_slot));
    // EXACTLY the active blocks are valid -- no slot the extract never meshed
    // reports a span, and none it did meshed is missing one. (Which slots those
    // are is not guessable: the block heap is LIFO, so `ptr` is handed out from
    // the top and the last slot is an allocated block, not a free one.)
    //
    // On a first extract this is a sanity check and not much more: the stamps
    // start empty, so it holds for any implementation that stamps the active
    // set at all. The version of it that constrains anything is below, and the
    // one that constrains the most is in the anchor block at the end of this
    // file, where the table already carries a stamp for a slot the current
    // extract did not mesh.
    CHECK(count_valid_spans(extractor, grid) == active.value().size());

    // A SECOND extract republishes it. Everything above ran against an
    // extractor's first and only extract, which is the one call where a table
    // that is never cleared and a table that is correctly rewritten look
    // identical.
    const std::uint64_t first_gen = extractor.block_spans_generation();
    vr::Result<mesh::Mesh> again_result = extractor.extract(grid, 0.0f);
    CHECK(again_result.ok());
    const mesh::Mesh again = std::move(again_result).value();
    CHECK(spans_describe(extractor, again, active.value(), kBlock, kH));
    CHECK(extractor.block_spans_generation() > first_gen);
    // ... and the count again, now that the stamps are non-empty going in. A
    // stamp is keyed to the extract that wrote it, so the previous call's
    // stamps have to stop counting as this call's -- the property the check
    // above could not see.
    CHECK(count_valid_spans(extractor, grid) == active.value().size());

    // A DENSE extract on the same extractor claims the same slot and can
    // reallocate the same arena, so it retires the table rather than leaving
    // spans that name geometry which is no longer there. The draw command gets
    // the same treatment for the same reason.
    const std::vector<vol::Voxel> tiny(8, vol::Voxel{-1.0f, 1.0f});
    mesh::DenseGrid tiny_grid;
    tiny_grid.dims = vr::Vec3i(2, 2, 2);
    tiny_grid.voxel_size = kH;
    tiny_grid.origin = vr::Vec3f(0.0f, 0.0f, 0.0f);
    CHECK(extractor.extract(tiny.data(), tiny.size(), tiny_grid, 0.0f).ok());
    CHECK(extractor.block_spans() == nullptr);
    CHECK(extractor.block_spans_generation() == 0);
    // The two halves of the question agree: retiring the table has to retire
    // every slot with it, or a caller gets a slot reported live beside a null
    // pointer. The per-slot stamps are untouched here -- it is the whole-table
    // test that closes this.
    CHECK(!extractor.block_span_valid(grid, some_slot));

    // ... and the next sparse extract brings it back, so retiring the table is
    // not a one-way door.
    vr::Result<mesh::Mesh> revived = extractor.extract(grid, 0.0f);
    CHECK(revived.ok());
    CHECK(
        spans_describe(extractor, revived.value(), active.value(), kBlock, kH));
    CHECK(extractor.block_span_valid(grid, some_slot));
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
  // An empty extract publishes no table either. It returns before any dispatch,
  // so the spans still standing are the previous extract's -- describing a mesh
  // this call did not hand out, against a slot it has already claimed.
  CHECK(extractor.block_spans() == nullptr);
  CHECK(extractor.block_spans_generation() == 0);

  // --- track_block_spans is off by default -----------------------------------
  // The table is sized by the GRID (num_blocks * 16, which is 24 MB at
  // VoxelGridParams::defaults and doubles with every resize), so a caller who
  // never reads it must not pay for it -- the bargain
  // TsdfIntegratorConfig::track_dirty_blocks strikes for the same table shape.
  // Asserted through arena_bytes, which is what makes "costs nothing" a
  // measurable claim rather than a comment: it counts the span table, so an
  // ungated allocation would show up here.
  // Two FRESH extractors differing in nothing but the flag, each run once over
  // the same grid. Comparing `extractor` against a new one would compare two
  // different extract histories -- the arenas are grow-only, so the one that
  // has meshed more is larger for reasons that have nothing to do with the
  // table -- and the difference here has to be attributable to the flag alone.
  {
    mesh::MarchingCubesConfig gated_config;
    gated_config.track_block_spans = true;
    vr::Result<mesh::MarchingCubes> gated_result = mesh::MarchingCubes::create(
        device.value(), allocator.value(), gated_config);
    CHECK(gated_result.ok());
    mesh::MarchingCubes gated = std::move(gated_result).value();
    vr::Result<mesh::MarchingCubes> ungated_result =
        mesh::MarchingCubes::create(device.value(), allocator.value());
    CHECK(ungated_result.ok());
    mesh::MarchingCubes ungated = std::move(ungated_result).value();

    mesh::ExtractTimings gated_timings;
    mesh::ExtractTimings ungated_timings;
    vr::Result<mesh::Mesh> gated_mesh =
        gated.extract(grid, 0.0f, &gated_timings);
    vr::Result<mesh::Mesh> ungated_mesh =
        ungated.extract(grid, 0.0f, &ungated_timings);
    CHECK(gated_mesh.ok());
    CHECK(ungated_mesh.ok());
    // The same surface either way: the kernel skipping the store changes
    // nothing it emits, which is what makes the flag a pure opt-out.
    CHECK(ungated_mesh.value().triangle_count() == sphere.triangle_count());
    CHECK(canonical_triangles(ungated_mesh.value()) ==
          canonical_triangles(gated_mesh.value()));

    CHECK(ungated.block_spans() == nullptr);
    CHECK(ungated.block_span_capacity() == 0);
    CHECK(ungated.block_spans_generation() == 0);
    CHECK(gated.block_span_capacity() ==
          static_cast<std::uint32_t>(gp.num_blocks));

    // EXACTLY the table AND the host-side stamps beside it, and no more: same
    // grid, same surface, same plan, so every other resident byte is identical
    // and the whole difference is what the flag allocates. An equality rather
    // than a bound, because that is the claim -- and because arena_bytes
    // silently omitting a component (it counted only the arenas and index runs,
    // and later the table but not the stamps) is the defect this figure keeps
    // attracting. Both terms, so leaving either out fails here rather than
    // under-reporting the feature by a third in a caller's profile.
    CHECK(gated_timings.arena_bytes ==
          ungated_timings.arena_bytes +
              static_cast<std::uint64_t>(gp.num_blocks) *
                  (sizeof(mesh::BlockSpan) + sizeof(std::uint64_t)));
  }

  // --- The span table survives a grid GROW -----------------------------------
  // A VoxelHashMap::resize raises num_blocks, so the table has to grow with it;
  // and because resize PRESERVES each block's index (which is why
  // topology_epoch deliberately does not move across one), a slot means the
  // same block on both sides. The grow therefore carries the old spans forward
  // and zeroes only the new tail, exactly as
  // TsdfIntegrator::prepare_dirty_flags does for the sibling slot-keyed table
  // -- replacing it wholesale would discard every span on the one event the
  // volume tier guarantees they survive.
  {
    vr::Result<vol::VoxelBlockGrid> grow_grid_result =
        vol::VoxelBlockGrid::create(device.value(), allocator.value(), gp,
                                    attrs, 2);
    CHECK(grow_grid_result.ok());
    vol::VoxelBlockGrid grow_grid = std::move(grow_grid_result).value();
    CHECK(fill_sphere_grid(grow_grid, /*with_color=*/false));

    mesh::MarchingCubesConfig grow_config;
    grow_config.track_block_spans = true;
    vr::Result<mesh::MarchingCubes> grow_result = mesh::MarchingCubes::create(
        device.value(), allocator.value(), grow_config);
    CHECK(grow_result.ok());
    mesh::MarchingCubes grow_mc = std::move(grow_result).value();

    vr::Result<mesh::Mesh> before = grow_mc.extract(grow_grid, 0.0f);
    CHECK(before.ok());
    CHECK(grow_mc.block_span_capacity() ==
          static_cast<std::uint32_t>(gp.num_blocks));
    vr::Result<std::vector<vol::BlockIndex>> active_before =
        grow_grid.map().compact_active_blocks();
    CHECK(active_before.ok());
    CHECK(spans_describe(grow_mc, before.value(), active_before.value(), kBlock,
                         kH));

    const std::uint64_t epoch_before_grow = grow_grid.topology_epoch();
    CHECK(grow_grid.resize(gp.num_buckets * 2).ok());
    // The anchor must NOT break here, and that is the one mutation for which
    // that is true: a resize frees no index, so the token holds and the spans
    // carried forward keep describing the blocks they were written for. If this
    // moved, the grow path below would be dead code.
    CHECK(grow_grid.topology_epoch() == epoch_before_grow);
    // Same blocks, same slots -- the resize preserves indices, so this is the
    // property that makes carrying the table forward meaningful rather than
    // merely harmless.
    vr::Result<std::vector<vol::BlockIndex>> active_after =
        grow_grid.map().compact_active_blocks();
    CHECK(active_after.ok());
    CHECK(active_after.value().size() == active_before.value().size());

    vr::Result<mesh::Mesh> after = grow_mc.extract(grow_grid, 0.0f);
    CHECK(after.ok());
    CHECK(grow_mc.block_span_capacity() ==
          static_cast<std::uint32_t>(gp.num_blocks) * 2);
    CHECK(spans_describe(grow_mc, after.value(), active_after.value(), kBlock,
                         kH));
    // The grown tail is ZEROED, not whatever VMA handed over: only blocks in an
    // extract's active set are written, and every other entry is published as
    // readable. An empty span is a truthful "this block owns no geometry"; a
    // driver-garbage base indexes the arena anywhere.
    const mesh::BlockSpan* grown = grow_mc.block_spans();
    CHECK(grown != nullptr);
    std::size_t nonempty = 0;
    for (std::uint32_t i = 0; i < grow_mc.block_span_capacity(); ++i) {
      if (grown[i].triangle_count != 0 || grown[i].vertex_count != 0 ||
          grown[i].triangle_base != 0 || grown[i].vertex_base != 0) {
        ++nonempty;
      }
    }
    // Every non-empty entry is an active block's; the whole tail past them is
    // zero. (One active block can legitimately mesh to nothing and record an
    // empty span, so this is a bound rather than an equality.)
    CHECK(nonempty <= active_after.value().size());
  }

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
  share_config.track_block_spans = true;
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

    // ... and its published table describes it, with vertex counts that are NOT
    // three per triangle -- the case the default path cannot exercise. The
    // ranges above are inferred from the mesh; this is what the kernel claims.
    vr::Result<std::vector<vol::BlockIndex>> active =
        grid.map().compact_active_blocks();
    CHECK(active.ok());
    CHECK(spans_describe(share_mc, share_mesh, active.value(), kBlock, kH));
    CHECK(share_mesh.vertices.size() != share_mesh.indices.size());
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

  // The device view says so, which is how a consumer sizes against a shared
  // mesh without inspecting the buffers -- vertex_count is no longer
  // `3 * triangle_count`. It is not an incompatibility: texture::
  // ProjectiveTexturer decides visibility per vertex and textures this mesh
  // like any other (see texture_device_mesh_test).
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
  // The source must be left EMPTY, not merely invalid: this class's
  // rule-of-zero moves are correct only while nothing caches a copy of an owned
  // member's state, and the span table is the newest place that could go wrong.
  // A block_span_capacity() tracked in a uint32 beside the buffer would survive
  // the move and report a live capacity next to the null pointer below -- the
  // exact shape a caller sizing its loop from the capacity walks off the end
  // of.
  CHECK(extractor.block_span_capacity() > 0);
  mesh::MarchingCubes moved = std::move(extractor);
  CHECK(!extractor.valid());
  CHECK(extractor.block_spans() == nullptr);
  CHECK(extractor.block_span_capacity() == 0);
  CHECK(moved.valid());
  CHECK(moved.block_span_capacity() > 0);
  mesh::MarchingCubes* alias = &moved;
  moved = std::move(*alias);  // self-move: intact
  CHECK(moved.valid());
  CHECK(moved.block_span_capacity() > 0);
  vr::Result<mesh::Mesh> reextract = moved.extract(grid, 0.0f);
  CHECK(reextract.ok());
  CHECK(!std::move(reextract).value().empty());

  // --- The span anchor breaks on a topology change ---------------------------
  //
  // LAST, because it mutates the shared grid: remove() takes a block out of the
  // active set, so every extract after it meshes a different surface. Placed
  // earlier, this failed the sharing path's triangle-count equivalence -- but
  // only SOMETIMES, because the victim is whichever block compaction happened
  // to put last and removing one that carries no surface changes nothing. A
  // test that fails on some runs is worse than one that fails on all of them.
  //
  // Checked against the GRID rather than against the last extract: between a
  // remove() and the next extract the stamps are still set, so a query trusting
  // them alone would call a stale span live.
  //
  // Through EITHER remove: the token lives on VoxelHashMap, which is where an
  // index is actually freed, so the raw map() path moves it exactly as the
  // grid's wrapper does. It used to live one tier up and be bumped only by the
  // wrapper, which left the raw path defeating this anchor in silence; both are
  // exercised below.
  {
    // Its OWN grid and extractor. Everything above has been moved from, had
    // blocks removed, or been pointed at another grid by now -- and a span
    // table describing the last grid it saw is the anchor working, not a
    // wrinkle to route around.
    vr::Result<vol::VoxelBlockGrid> anchor_grid_result =
        vol::VoxelBlockGrid::create(device.value(), allocator.value(), gp,
                                    attrs, 2);
    CHECK(anchor_grid_result.ok());
    vol::VoxelBlockGrid anchor_grid = std::move(anchor_grid_result).value();
    CHECK(fill_sphere_grid(anchor_grid, /*with_color=*/false));

    mesh::MarchingCubesConfig anchor_config;
    anchor_config.track_block_spans = true;
    vr::Result<mesh::MarchingCubes> anchor_mc_result =
        mesh::MarchingCubes::create(device.value(), allocator.value(),
                                    anchor_config);
    CHECK(anchor_mc_result.ok());
    mesh::MarchingCubes anchor_mc = std::move(anchor_mc_result).value();
    CHECK(anchor_mc.extract(anchor_grid, 0.0f).ok());

    vr::Result<std::vector<vol::BlockIndex>> live =
        anchor_grid.map().compact_active_blocks();
    CHECK(live.ok());
    CHECK(!live.value().empty());
    const auto vpb = static_cast<std::uint32_t>(kBlock * kBlock * kBlock);
    const std::uint32_t slot =
        static_cast<std::uint32_t>(live.value().front().ptr) / vpb;
    CHECK(anchor_mc.block_span_valid(anchor_grid, slot));
    CHECK(count_valid_spans(anchor_mc, anchor_grid) == live.value().size());

    vol::BlockIndex victim = live.value().back();
    const std::uint32_t victim_slot =
        static_cast<std::uint32_t>(victim.ptr) / vpb;
    vr::Result<std::uint32_t> removed = anchor_grid.remove(&victim, 1);
    CHECK(removed.ok());
    // The removal ACTUALLY happened. remove() moves the epoch unconditionally
    // -- a partial removal has still freed slots -- so without this the anchor
    // check below would pass over a no-op, and the whole block would be
    // asserting that a token moves rather than that a stale span goes dead.
    CHECK(removed.value() == 0);
    CHECK(!anchor_mc.block_span_valid(anchor_grid, slot));
    CHECK(count_valid_spans(anchor_mc, anchor_grid) == 0);

    // --- Re-extract on the changed topology ---------------------------------
    //
    // The path that matters, and the one a query before the next extract cannot
    // reach: the table is re-anchored to the new topology and republished, so
    // the stamps standing from the FIRST extract are now sitting under a table
    // that describes a different active set. Every one of them has to stop
    // counting -- and the freed slot in particular, whose span still names an
    // arena range this extract has handed to some other block.
    vr::Result<std::vector<vol::BlockIndex>> after_remove =
        anchor_grid.map().compact_active_blocks();
    CHECK(after_remove.ok());
    CHECK(after_remove.value().size() == live.value().size() - 1);
    CHECK(anchor_mc.extract(anchor_grid, 0.0f).ok());
    CHECK(anchor_mc.block_span_valid(anchor_grid, slot));  // re-meshed
    CHECK(!anchor_mc.block_span_valid(anchor_grid, victim_slot));
    // Exactly the surviving blocks, counted over the WHOLE capacity: a stamp
    // that survives its extract shows up here as one too many, which is what a
    // "has this slot ever been meshed" test reports and what a table nothing
    // retires would report for the rest of the run.
    CHECK(count_valid_spans(anchor_mc, anchor_grid) ==
          after_remove.value().size());

    // --- The same extractor, pointed at a SECOND grid ------------------------
    //
    // The other way the anchor re-arms, and the one that leaves the most stale
    // state behind: the slot numbers coincide (both grids allocate the same
    // blocks from the same fresh LIFO heap), so nothing about a slot's VALUE
    // distinguishes the two tables. Only the token does.
    vr::Result<vol::VoxelBlockGrid> other_result = vol::VoxelBlockGrid::create(
        device.value(), allocator.value(), gp, attrs, 2);
    CHECK(other_result.ok());
    vol::VoxelBlockGrid other_grid = std::move(other_result).value();
    CHECK(fill_sphere_grid(other_grid, /*with_color=*/false));
    // Two fresh grids never share a token, even at the same topology and even
    // if one is built in storage the other has vacated -- it is drawn from a
    // process-wide counter, which is what closes the ABA a grid pointer cannot
    // see.
    CHECK(other_grid.topology_epoch() != anchor_grid.topology_epoch());

    CHECK(anchor_mc.extract(other_grid, 0.0f).ok());
    vr::Result<std::vector<vol::BlockIndex>> other_live =
        other_grid.map().compact_active_blocks();
    CHECK(other_live.ok());
    CHECK(count_valid_spans(anchor_mc, other_grid) ==
          other_live.value().size());
    // ... and the first grid is now wholly unrepresented, including the slot it
    // had just re-meshed. The table is about one grid at a time.
    CHECK(count_valid_spans(anchor_mc, anchor_grid) == 0);

    // --- A moved-from grid is not the grid ----------------------------------
    //
    // The token rides with the map into the destination, so both objects answer
    // topology_epoch() the same -- and the corpse owns no blocks. Without a
    // valid() check the answer inverts: the moved-from grid reports its slots
    // live and the object holding them reports nothing.
    vol::VoxelBlockGrid taken = std::move(other_grid);
    CHECK(count_valid_spans(anchor_mc, other_grid) == 0);
    CHECK(count_valid_spans(anchor_mc, taken) == other_live.value().size());
  }

  // --- Incremental extraction: the skip is observable, not inferred ----------
  //
  // Comparing an incremental extract against a full one over the SAME field
  // proves nothing: a pass that silently fell back to full returns the
  // identical mesh, and so does one that skipped correctly. So the field is
  // CHANGED under the extractor between passes, with the flags still saying
  // nothing moved.
  //
  //   flags all zero, field changed  -> every block takes the early return, so
  //     the mesh must still be the OLD surface. A fallback to full, or a dirty
  //     test that reads the wrong way, returns the new one and fails here.
  //   flags MIXED, same new field    -> the case the feature actually runs in,
  //     and the only one where the two halves can disagree: a clean block has
  //     to keep its range while a dirty neighbour relocates past it. Under a
  //     uniform flag array every block's old range is being rewritten anyway,
  //     so a kernel that ignored `s_neighbour` entirely -- or retired the wrong
  //     range -- passes both of the other two unchanged.
  //   flags all one, same new field  -> every block re-meshes into the range it
  //     already owns, so in-place reuse, the span read and the retire pass all
  //     run, and the mesh must now be the NEW surface.
  //
  // Compared as triangle sets, since a re-mesh may reorder within a block. Run
  // for BOTH kernels: sharing is what the only device consumer uses, and it is
  // the one whose retire pass touches indices instead of vertices.
  for (int share_pass = 0; share_pass < 2; ++share_pass) {
    vr::Result<vol::VoxelBlockGrid> inc_grid_result =
        vol::VoxelBlockGrid::create(device.value(), allocator.value(), gp,
                                    attrs, 2);
    CHECK(inc_grid_result.ok());
    vol::VoxelBlockGrid inc_grid = std::move(inc_grid_result).value();
    CHECK(fill_sphere_grid(inc_grid, /*with_color=*/false));

    mesh::MarchingCubesConfig inc_config;
    inc_config.track_block_spans =
        true;  // what an incremental pass re-meshes against
    // The point of the loop: pass 0 is the default emitter, pass 1 the sharing
    // one. Without this the two iterations are the same kernel run twice.
    inc_config.share_vertices = share_pass == 1;
    vr::Result<mesh::MarchingCubes> inc_result = mesh::MarchingCubes::create(
        device.value(), allocator.value(), inc_config);
    CHECK(inc_result.ok());
    mesh::MarchingCubes inc_mc = std::move(inc_result).value();

    // The first extract can only be full -- there is no watermark yet -- and it
    // is what establishes the spans and the arena the next one reuses.
    vr::Result<mesh::Mesh> first = inc_mc.extract(inc_grid, 0.0f);
    CHECK(first.ok());
    const std::vector<std::array<float, 9>> old_surface =
        canonical_triangles(first.value());
    CHECK(!old_surface.empty());

    // Each block's triangle count under the OLD field, off the table that first
    // extract published. Kept so the mixed pass below can flag a block whose
    // geometry demonstrably MOVED rather than one that merely has some.
    std::vector<std::uint32_t> old_counts(inc_mc.block_span_capacity(), 0);
    {
      const mesh::BlockSpan* spans = inc_mc.block_spans();
      CHECK(spans != nullptr);
      for (std::uint32_t slot = 0; slot < inc_mc.block_span_capacity();
           ++slot) {
        if (inc_mc.block_span_valid(inc_grid, slot)) {
          old_counts[slot] = spans[slot].triangle_count;
        }
      }
    }

    // A visibly different sphere, written straight into the same blocks.
    const float kGrown = kRadius * 1.15f;
    CHECK(fill_sphere_grid(inc_grid, /*with_color=*/false, 1.0f, kGrown));

    // What a full extract of the NEW field gives, taken now so the mixed pass
    // below can be checked against both surfaces. A separate extractor, so
    // taking it does not disturb the arena `inc_mc` is carrying across passes.
    std::vector<std::array<float, 9>> new_surface;
    // The one block the mixed pass flags: one that carried surface before and
    // emits a DIFFERENT number of triangles now, so re-meshing it is guaranteed
    // to change the output.
    //
    // Chosen by what moved, never by position in the slot list, and that is not
    // fussiness: a block's slot is handed out by the allocator's atomics, so
    // coord -> slot varies run to run, and "the middle slot with any surface"
    // picks a different BLOCK each time -- sometimes one the growth barely
    // touches, whose re-mesh then produces nothing the old surface did not
    // already contain. That is an 8%-flaky assertion, and it fails for a reason
    // that has nothing to do with what it is testing.
    std::uint32_t flag_slot = 0;
    bool have_flag_slot = false;
    {
      vr::Result<mesh::MarchingCubes> ref_result = mesh::MarchingCubes::create(
          device.value(), allocator.value(), inc_config);
      CHECK(ref_result.ok());
      mesh::MarchingCubes ref_mc = std::move(ref_result).value();
      vr::Result<mesh::Mesh> grown = ref_mc.extract(inc_grid, 0.0f);
      CHECK(grown.ok());
      new_surface = canonical_triangles(grown.value());
      CHECK(drop_degenerate(new_surface) == 0);  // a full extract retires none
      CHECK(new_surface != old_surface);         // the field really did change

      // Same grid, so the same slots: a slot names a block, not an extractor.
      const mesh::BlockSpan* new_spans = ref_mc.block_spans();
      CHECK(new_spans != nullptr);
      for (std::uint32_t slot = 0; slot < old_counts.size(); ++slot) {
        if (old_counts[slot] > 0 && ref_mc.block_span_valid(inc_grid, slot) &&
            new_spans[slot].triangle_count != old_counts[slot]) {
          flag_slot = slot;
          have_flag_slot = true;
          break;
        }
      }
    }
    CHECK(have_flag_slot);

    // One flag per block slot, as the tsdf tier publishes them. Built here
    // rather than by fusing: the contract is a buffer, and this is a test of
    // the extractor rather than of the integrator.
    const auto slots = static_cast<std::uint32_t>(gp.num_blocks);
    vr::Result<vr::Buffer> flags_result = vr::storage_buffer(
        allocator.value(),
        static_cast<VkDeviceSize>(slots) * sizeof(std::uint32_t),
        vr::HostAccess::SequentialWrite);
    CHECK(flags_result.ok());
    vr::Buffer flags = std::move(flags_result).value();
    auto* flag_ptr = static_cast<std::uint32_t*>(flags.mapped());

    // The epoch travels with the flags, so every pass below names the grid it
    // is meshing. Omitting it is not a compile error -- it is an aggregate
    // field -- so a pass built without it would silently take the full-extract
    // fallback and every assertion about skipping would quietly stop testing
    // anything. Hence the ExtractTimings::incremental check on each.
    const mesh::DirtyBlocks dirty_blocks{flags.handle(), slots,
                                         inc_grid.topology_epoch()};

    for (std::uint32_t i = 0; i < slots; ++i) flag_ptr[i] = 0u;
    mesh::ExtractTimings clean_rt{};
    vr::Result<mesh::DeviceMesh> clean = inc_mc.extract_device_incremental(
        inc_grid, 0.0f, dirty_blocks, &clean_rt);
    CHECK(clean.ok());
    CHECK(clean_rt.incremental);           // not the fallback
    CHECK(clean_rt.remeshed_blocks == 0);  // and nothing was re-meshed
    vr::Result<mesh::Mesh> clean_host = inc_mc.download(clean.value());
    CHECK(clean_host.ok());
    CHECK(canonical_triangles(clean_host.value()) == old_surface);

    // --- The mixed pass -------------------------------------------------
    //
    // One flagged block, dilated on-device into the up-to-eight blocks whose
    // `+{0,1}^3` neighbourhood contains it. So a handful of blocks relocate or
    // shrink while every other block keeps the range it already owns, in an
    // arena being appended to at the same time -- which is the only
    // configuration where keeping and re-meshing can disagree.
    for (std::uint32_t i = 0; i < slots; ++i) flag_ptr[i] = 0u;
    flag_ptr[flag_slot] = 1u;
    mesh::ExtractTimings mixed_rt{};
    vr::Result<mesh::DeviceMesh> mixed = inc_mc.extract_device_incremental(
        inc_grid, 0.0f, dirty_blocks, &mixed_rt);
    CHECK(mixed.ok());
    CHECK(mixed_rt.incremental);
    // Genuinely mixed, on the kernel's own count: some blocks re-meshed, and
    // not all of them. Without this the flag pattern could dilate to everything
    // (or to nothing) and the assertions below would still pass, describing a
    // uniform pass by another name.
    CHECK(mixed_rt.remeshed_blocks > 0);
    CHECK(mixed_rt.remeshed_blocks < mixed_rt.active_blocks);
    vr::Result<mesh::Mesh> mixed_host = inc_mc.download(mixed.value());
    CHECK(mixed_host.ok());
    std::vector<std::array<float, 9>> mixed_tris =
        canonical_triangles(mixed_host.value());
    // Retirement runs on this pass too: the re-meshed blocks relocate or shrink
    // and give their old ranges back, in the middle of an arena whose other
    // blocks must be untouched.
    CHECK(drop_degenerate(mixed_tris) > 0);
    {
      // Every surviving triangle came from one of the two surfaces, and BOTH
      // are represented. A pass that quietly re-meshed everything would be all
      // new; one that skipped everything would be all old; one that corrupted a
      // clean block's range while a neighbour relocated past it would hold a
      // triangle from neither.
      const std::set<std::array<float, 9>> old_set(old_surface.begin(),
                                                   old_surface.end());
      const std::set<std::array<float, 9>> new_set(new_surface.begin(),
                                                   new_surface.end());
      std::size_t kept = 0;
      std::size_t remeshed = 0;
      for (const std::array<float, 9>& t : mixed_tris) {
        const bool in_old = old_set.count(t) != 0;
        const bool in_new = new_set.count(t) != 0;
        CHECK(in_old || in_new);  // invented nothing
        if (in_old && !in_new) ++kept;
        if (in_new && !in_old) ++remeshed;
      }
      CHECK(kept > 0);      // clean blocks really did keep their triangles
      CHECK(remeshed > 0);  // and the flagged neighbourhood really did redo its
    }

    // --- And then all of it ----------------------------------------------
    for (std::uint32_t i = 0; i < slots; ++i) flag_ptr[i] = 1u;
    mesh::ExtractTimings dirty_rt{};
    vr::Result<mesh::DeviceMesh> dirty = inc_mc.extract_device_incremental(
        inc_grid, 0.0f, dirty_blocks, &dirty_rt);
    CHECK(dirty.ok());
    CHECK(dirty_rt.incremental);
    // Every block, and the kernel's own count says so rather than the flags.
    CHECK(dirty_rt.remeshed_blocks == dirty_rt.active_blocks);
    vr::Result<mesh::Mesh> dirty_host = inc_mc.download(dirty.value());
    CHECK(dirty_host.ok());
    std::vector<std::array<float, 9>> all_dirty =
        canonical_triangles(dirty_host.value());
    // The grown sphere makes some blocks outgrow the range they held and others
    // shrink inside it, so both halves of the retire pass run -- and leave the
    // degenerates this drops. Asserting some were dropped is what keeps that
    // pass covered rather than merely compiled.
    CHECK(drop_degenerate(all_dirty) > 0);
    CHECK(!all_dirty.empty());

    // Exactly the surface a full extract of the new field gives -- so
    // relocation, in-place reuse and retirement together lose and invent
    // nothing, and the mixed pass before it left an arena the next pass could
    // build on. That last part is what the mixed case adds: a corrupted clean
    // range survives into here.
    CHECK(all_dirty == new_surface);

    // --- The fallbacks, each proved by a field the skip would hide ---------
    //
    // Every one of these is checked by asserting BOTH halves: that the pass
    // reported itself full, and that the mesh is the CURRENT field rather than
    // the arena's previous contents. The second half is what makes the first
    // mean anything -- a clause that silently stopped holding would leave a
    // pass calling itself incremental while publishing a stale surface, which
    // is exactly the shape of the defects this covers.
    //
    // The field is changed once more, back toward the original radius, so a
    // pass that wrongly skipped would return the grown sphere and be caught.
    CHECK(fill_sphere_grid(inc_grid, /*with_color=*/false, 1.0f, kRadius));
    std::vector<std::array<float, 9>> shrunk_surface;
    {
      vr::Result<mesh::MarchingCubes> ref_result = mesh::MarchingCubes::create(
          device.value(), allocator.value(), inc_config);
      CHECK(ref_result.ok());
      vr::Result<mesh::Mesh> shrunk =
          std::move(ref_result).value().extract(inc_grid, 0.0f);
      CHECK(shrunk.ok());
      shrunk_surface = canonical_triangles(shrunk.value());
      CHECK(shrunk_surface != new_surface);
    }
    // All-zero flags throughout, so an incremental pass returns the arena's
    // previous contents and a full one returns the shrunk sphere. The two are
    // distinguishable, which is the whole point.
    for (std::uint32_t i = 0; i < slots; ++i) flag_ptr[i] = 0u;

    // (a) Flags the integrator will not vouch for. dirty_flags_buffer() returns
    //     null and dirty_epoch() returns 0 on every staleness this tier can
    //     see, so both are refusals a caller can pass through verbatim.
    for (const mesh::DirtyBlocks& refused :
         {mesh::DirtyBlocks{VK_NULL_HANDLE, slots, inc_grid.topology_epoch()},
          mesh::DirtyBlocks{flags.handle(), slots, 0},
          mesh::DirtyBlocks{flags.handle(), 0, inc_grid.topology_epoch()}}) {
      mesh::ExtractTimings rt{};
      vr::Result<mesh::DeviceMesh> dm =
          inc_mc.extract_device_incremental(inc_grid, 0.0f, refused, &rt);
      CHECK(dm.ok());
      CHECK(!rt.incremental);
      vr::Result<mesh::Mesh> host = inc_mc.download(dm.value());
      CHECK(host.ok());
      CHECK(canonical_triangles(host.value()) == shrunk_surface);
    }

    // (b) A DENSE extract in between. It claims the same slot and overwrites
    //     the same arena with a kernel that knows nothing about blocks, so the
    //     watermark it leaves behind names triangles that are gone -- and the
    //     span table it retires is not what catches that, since a dense extract
    //     moves no topology epoch and bumps no span serial.
    {
      const std::vector<vol::Voxel> tiny(8, vol::Voxel{-1.0f, 1.0f});
      mesh::DenseGrid tiny_grid;
      tiny_grid.dims = vr::Vec3i(2, 2, 2);
      tiny_grid.voxel_size = kH;
      tiny_grid.origin = vr::Vec3f(0.0f, 0.0f, 0.0f);
      CHECK(inc_mc.extract(tiny.data(), tiny.size(), tiny_grid, 0.0f).ok());

      mesh::ExtractTimings rt{};
      vr::Result<mesh::DeviceMesh> dm =
          inc_mc.extract_device_incremental(inc_grid, 0.0f, dirty_blocks, &rt);
      CHECK(dm.ok());
      CHECK(!rt.incremental);
      vr::Result<mesh::Mesh> host = inc_mc.download(dm.value());
      CHECK(host.ok());
      CHECK(canonical_triangles(host.value()) == shrunk_surface);
    }

    // (c) A topology change. remove() re-draws the grid's epoch and puts the
    //     freed indices back on the LIFO list, so a slot now names a different
    //     block and BOTH the flags and the spans describe geometry that is
    //     gone. The re-anchor that ensure_block_spans does on the way past is
    //     what made this look sound: comparing the table's anchor to the grid
    //     AFTER re-anchoring it compares a value with itself.
    {
      vol::BlockIndex corner{};
      corner.coord = vr::Vec3i(0, 0, 0);
      vr::Result<std::uint32_t> removed = inc_grid.remove(&corner, 1);
      CHECK(removed.ok());

      // Built AFTER the remove, so the flags name the grid's current epoch and
      // only the ARENA's anchor is stale. Passing the pre-remove epoch would
      // fail on clause (a) instead and never reach the one under test.
      const mesh::DirtyBlocks post_remove{flags.handle(), slots,
                                          inc_grid.topology_epoch()};
      mesh::ExtractTimings rt{};
      vr::Result<mesh::DeviceMesh> dm =
          inc_mc.extract_device_incremental(inc_grid, 0.0f, post_remove, &rt);
      CHECK(dm.ok());
      CHECK(!rt.incremental);
    }
  }

  std::printf(
      "recon mesh sparse marching-cubes test passed: meshed a sphere across "
      "%d^3 blocks (%zu triangles), matched the dense path exactly, and "
      "verified cross-block colour on-device\n",
      kBlocks, sphere.triangle_count());
  return 0;
}
