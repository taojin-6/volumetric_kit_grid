// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// GPU test for an extract recorded into a caller's batch
// (MarchingCubes::record_extract, finish and abandon): the mesh is
// extract_device's, sharing on and off; a surface past the plan refits, its
// mesh ready at a later value, its row a breakdown of the finish's; a closed
// gate gives no mesh and its slot back; an empty set meshes nothing; abandoned
// and dropped extracts never exhaust the slots; a batch that skipped the
// signal, or a pending superseded, is refused, and another extractor's token
// is not abandoned; `ready` names a value reached, and outlives a move of the
// extractor; a steady extract costs one submit and no staging buffer,
// recorded or blocking, after a grid change too; and a grid the sharing
// kernel cannot hold is refused with nothing recorded. Skips with no device.

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "volumetric_kit/core/base/stage_metrics.hpp"
#include "volumetric_kit/core/testing/test_check.hpp"
#include "volumetric_kit/core/vulkan/allocator.hpp"
#include "volumetric_kit/core/vulkan/command_batch.hpp"
#include "volumetric_kit/core/vulkan/device.hpp"
#include "volumetric_kit/core/vulkan/sync.hpp"
#include "volumetric_kit/core/vulkan/vulkan.hpp"
#include "volumetric_kit/grid/base/math/vector_types.hpp"
#include "volumetric_kit/grid/mesh/marching_cubes.hpp"
#include "volumetric_kit/grid/mesh/mesh.hpp"
#include "volumetric_kit/grid/testing/device_blocks.hpp"
#include "volumetric_kit/grid/testing/grid_readback.hpp"
#include "volumetric_kit/grid/testing/stage_rows.hpp"
#include "volumetric_kit/grid/volume/hash_types.hpp"
#include "volumetric_kit/grid/volume/voxel_block_grid.hpp"
#include "volumetric_kit/grid/volume/voxel_grid.hpp"

#include "volumetric_kit/core/testing/gpu_test.hpp"
#include "volumetric_kit/grid/base/device_requirements.hpp"

namespace vgrid = volumetric_kit::grid;
namespace vkc = volumetric_kit::core;
namespace vol = volumetric_kit::grid::volume;
namespace mesh = volumetric_kit::grid::mesh;
using C = vkc::Status::Code;

namespace {

constexpr int kBlocks = 4;       // blocks per axis
constexpr float kH = 0.05f;      // metres between voxels
constexpr float kRadius = 0.5f;  // the sphere's

vol::VoxelGridParams grid_params(int block_size = 8) {
  vol::VoxelGridParams g{};
  g.voxel_size = kH;
  g.block_size = block_size;
  g.voxels_per_block = block_size * block_size * block_size;
  g.trunc_dist = 0.1f;
  g.bucket_size = 8;
  g.num_buckets = 64;
  g.num_blocks = 8 * 64;
  g.max_chain = 64;
  return g;
}

// A grid of kBlocks^3 blocks holding a sphere's distance at weight 1, its
// centre mid-grid: a surface crossing many block faces.
vkc::Result<vol::VoxelBlockGrid> sphere_grid(vkc::test::Gpu& gpu,
                                             int blocks = kBlocks) {
  const vol::AttributeSpec attrs[] = {{vol::kTsdfAttribute, 4},
                                      {vol::kWeightAttribute, 4}};
  VKC_ASSIGN(vol::VoxelBlockGrid grid,
             vol::VoxelBlockGrid::create(gpu.device, gpu.allocator,
                                         grid_params(), attrs, 2));
  std::vector<vol::BlockIndex> coords;
  for (int i = 0; i < blocks * blocks * blocks; ++i) {
    vol::BlockIndex b{};
    b.coord =
        vgrid::Vec3i(i % blocks, (i / blocks) % blocks, i / (blocks * blocks));
    coords.push_back(b);
  }
  if (coords.empty()) return grid;
  VKC_ASSIGN(const std::uint32_t failed,
             vgrid_test::allocate_blocks(gpu, grid.map(), coords));
  if (failed != 0) return vkc::Status::io_error("blocks left unallocated");
  VKC_ASSIGN(const std::vector<vol::BlockIndex> active,
             grid.map().compact_active_blocks());
  VKC_ASSIGN(std::vector<float> tsdf,
             vgrid_test::read_attribute<float>(gpu.device, gpu.allocator, grid,
                                               vol::kTsdfAttribute));
  std::vector<float> weight(tsdf.size(), 0.0f);
  const float c = 0.5f * float(blocks * 8 - 1) * kH;
  for (const vol::BlockIndex& b : active) {
    for (int v = 0; v < 512; ++v) {
      const vgrid::Vec3i voxel =
          b.coord * 8 + vgrid::Vec3i(v % 8, (v / 8) % 8, v / 64);
      const vgrid::Vec3f p = vgrid::Vec3f(voxel) * kH - vgrid::Vec3f(c, c, c);
      tsdf[std::size_t(b.ptr) + v] = vgrid::length(p) - kRadius;
      weight[std::size_t(b.ptr) + v] = 1.0f;
    }
  }
  VKC_TRY(vgrid_test::write_attribute(gpu.device, gpu.allocator, grid,
                                      vol::kTsdfAttribute, tsdf));
  VKC_TRY(vgrid_test::write_attribute(gpu.device, gpu.allocator, grid,
                                      vol::kWeightAttribute, weight));
  return grid;
}

// A mesh's triangles as sorted position nonuples, through its indices: two
// extracts of one field emit the same triangles in another order.
std::vector<std::array<float, 9>> triangles(const mesh::Mesh& m) {
  std::vector<std::array<float, 9>> out;
  for (std::size_t i = 0; i + 2 < m.indices.size(); i += 3) {
    std::array<float, 9> t{};
    for (std::size_t k = 0; k < 3; ++k) {
      const vgrid::Vec3f p = m.vertices[m.indices[i + k]].position;
      t[3 * k] = p.x;
      t[3 * k + 1] = p.y;
      t[3 * k + 2] = p.z;
    }
    out.push_back(t);
  }
  std::sort(out.begin(), out.end());
  return out;
}

using vgrid_test::find_row;

vkc::Result<mesh::MarchingCubes> make_mc(vkc::test::Gpu& gpu,
                                         bool share = false,
                                         std::uint32_t slots = 1) {
  mesh::MarchingCubesConfig config;
  config.share_vertices = share;
  config.slot_count = slots;
  return mesh::MarchingCubes::create(gpu.device, gpu.allocator, config);
}

// One recorded extract in a batch of its own, which sets the signal (or, with
// @p skip_signal, does not), finished.
vkc::Result<std::optional<mesh::DeviceMesh>> recorded(
    vkc::test::Gpu& gpu, mesh::MarchingCubes& mc, vol::VoxelBlockGrid& grid,
    std::uint32_t expected, const vol::DeviceCount* gate = nullptr,
    mesh::ExtractTimings* timings = nullptr, bool skip_signal = false) {
  vkc::CommandBatch batch(gpu.device, gpu.allocator);
  VKC_ASSIGN(const mesh::PendingExtract pending,
             mc.record_extract(batch, grid, expected, gate));
  std::vector<vkc::TimelinePoint> signals;
  if (!skip_signal) signals.push_back(pending.signal);
  VKC_ASSIGN(vkc::PendingBatch ran, batch.submit_async({}, signals));
  VKC_TRY(ran.wait());
  return mc.finish(pending, grid, timings);
}

// --- The mesh.
// ----------------------------------------------------------------

// The recorded extract meshes what extract_device meshes, sharing on and off,
// and hands it over stamped with the batch's signal, reached.
int same_mesh_case(vkc::test::Gpu& gpu) {
  vkc::Result<vol::VoxelBlockGrid> grid = sphere_grid(gpu);
  CHECK(grid.ok());
  const std::uint32_t n = grid->map().allocated_blocks();
  for (const bool share : {false, true}) {
    vkc::Result<mesh::MarchingCubes> a = make_mc(gpu, share);
    vkc::Result<mesh::MarchingCubes> b = make_mc(gpu, share);
    CHECK(a.ok() && b.ok());
    vkc::Result<mesh::DeviceMesh> want = a->extract_device(*grid);
    CHECK(want.ok() && want->triangle_count > 100);
    vkc::CommandBatch batch(gpu.device, gpu.allocator);
    vkc::Result<mesh::PendingExtract> pending =
        b->record_extract(batch, *grid, n);
    CHECK(pending.ok() && pending->generation != 0);
    CHECK(pending->signal.semaphore != nullptr && pending->signal.value != 0);
    vkc::Result<vkc::PendingBatch> ran =
        batch.submit_async({}, {pending->signal});
    CHECK(ran.ok() && ran->wait().ok());
    vkc::Result<std::optional<mesh::DeviceMesh>> got =
        b->finish(*pending, *grid);
    CHECK(got.ok() && got->has_value());
    const mesh::DeviceMesh& m = **got;
    CHECK(m.generation == pending->generation && m.is_current());
    CHECK(m.ready.semaphore == pending->signal.semaphore &&
          m.ready.value == pending->signal.value);
    CHECK(m.ready.semaphore->value().value() >= m.ready.value);
    CHECK(m.shares_vertices == share);
    CHECK(m.triangle_count == want->triangle_count &&
          m.vertex_count == want->vertex_count);
    vkc::Result<mesh::Mesh> wm = a->download(*want);
    vkc::Result<mesh::Mesh> gm = b->download(m);
    CHECK(wm.ok() && gm.ok() && triangles(*wm) == triangles(*gm));
    // A blocking extract stamps its mesh ready too.
    CHECK(want->ready.semaphore != nullptr &&
          want->ready.semaphore->value().value() >= want->ready.value);
  }
  return 0;
}

// A plan for one block on a fresh extractor cannot hold the sphere: finish
// grows the slot and meshes again in a batch of its own, whose later value
// the mesh is ready at; the surface is the same.
int refit_case(vkc::test::Gpu& gpu) {
  vkc::Result<vol::VoxelBlockGrid> grid = sphere_grid(gpu);
  vkc::Result<mesh::MarchingCubes> a = make_mc(gpu);
  vkc::Result<mesh::MarchingCubes> b = make_mc(gpu);
  CHECK(grid.ok() && a.ok() && b.ok());
  vkc::Result<mesh::DeviceMesh> want = a->extract_device(*grid);
  CHECK(want.ok());
  vkc::CommandBatch batch(gpu.device, gpu.allocator);
  vkc::Result<mesh::PendingExtract> pending =
      b->record_extract(batch, *grid, 1);
  CHECK(pending.ok());
  vkc::Result<vkc::PendingBatch> ran =
      batch.submit_async({}, {pending->signal});
  CHECK(ran.ok() && ran->wait().ok());
  mesh::ExtractTimings timings;
  vkc::StageMetrics m;
  vkc::Result<std::optional<mesh::DeviceMesh>> got =
      b->finish(*pending, *grid, &timings, &m);
  CHECK(got.ok() && got->has_value());
  CHECK(timings.dispatches == 2);
  // The refit's row is a breakdown of the call's "marching cubes".
  CHECK(find_row(m, "  ..refit") != nullptr);
  CHECK(m.rows().front().name == std::string("marching cubes"));
  CHECK(vgrid_test::breakdowns_nest(m));
  CHECK((*got)->ready.value > pending->signal.value);
  CHECK((*got)->ready.semaphore->value().value() >= (*got)->ready.value);
  CHECK((*got)->triangle_count == want->triangle_count);
  vkc::Result<mesh::Mesh> wm = a->download(*want);
  vkc::Result<mesh::Mesh> gm = b->download(**got);
  CHECK(wm.ok() && gm.ok() && triangles(*wm) == triangles(*gm));
  // The density it measured plans the next one right: no refit.
  mesh::ExtractTimings next;
  vkc::Result<std::optional<mesh::DeviceMesh>> again =
      recorded(gpu, *b, *grid, grid->map().allocated_blocks(), nullptr, &next);
  CHECK(again.ok() && again->has_value() && next.dispatches == 1);
  return 0;
}

// --- Gates, empties and the slots. -------------------------------------------

// A closed gate meshes nothing and finishes with no mesh, the slot back for
// the next extract; an empty grid meshes an empty mesh.
int gate_case(vkc::test::Gpu& gpu) {
  vkc::Result<vol::VoxelBlockGrid> grid = sphere_grid(gpu);
  vkc::Result<mesh::MarchingCubes> mc = make_mc(gpu);
  CHECK(grid.ok() && mc.ok());
  const std::uint32_t zero = 0;
  vkc::Result<vkc::Buffer> gate =
      vkc::test::upload_device_buffer(gpu.device, gpu.allocator, &zero, 4);
  CHECK(gate.ok());
  const vol::DeviceCount closed{&*gate, 0, 1};
  const std::uint32_t n = grid->map().allocated_blocks();
  for (int i = 0; i < 3; ++i) {
    vkc::Result<std::optional<mesh::DeviceMesh>> none =
        recorded(gpu, *mc, *grid, n, &closed);
    CHECK(none.ok() && !none->has_value());
  }
  vkc::Result<std::optional<mesh::DeviceMesh>> some =
      recorded(gpu, *mc, *grid, n);
  CHECK(some.ok() && some->has_value() && (*some)->triangle_count > 100);

  vkc::Result<vol::VoxelBlockGrid> empty = sphere_grid(gpu, 0);
  CHECK(empty.ok());
  vkc::Result<std::optional<mesh::DeviceMesh>> nothing =
      recorded(gpu, *mc, *empty, 0);
  CHECK(nothing.ok() && nothing->has_value() && (*nothing)->empty());
  return 0;
}

// Empty subsets need no list buffer, both before the first extract and after
// a populated slot. The draw command must also be empty when that slot is
// reused.
int empty_subset_case(vkc::test::Gpu& gpu) {
  auto grid = sphere_grid(gpu);
  CHECK(grid.ok());
  for (const bool share : {false, true}) {
    auto mc = make_mc(gpu, share);
    CHECK(mc.ok());
    for (int frame = 0; frame < 3; ++frame) {
      mesh::ExtractTimings timings;
      const vol::DeviceBlockList empty{};
      auto got = mc->extract_device(*grid, 0.0f, empty, &timings);
      CHECK(got.ok() && got->empty() && got->is_current());
      CHECK(got->vertex_count == 0 && timings.active_blocks == 0);
      CHECK(got->ready.semaphore != nullptr &&
            got->ready.semaphore->value().value() >= got->ready.value);
      VkDrawIndexedIndirectCommand draw{};
      vkc::Buffer indirect(
          got->indirect, sizeof(draw), got->indirect_usage, got->sharing_mode,
          nullptr, [] {}, std::nullopt);
      vkc::CommandBatch batch(gpu.device, gpu.allocator);
      CHECK(batch.readback(indirect, 0, sizeof(draw), &draw).ok());
      CHECK(batch.submit().ok());
      CHECK(draw.indexCount == 0);
      auto host = mc->download(*got);
      CHECK(host.ok() && host->vertices.empty() && host->indices.empty());
      auto populated = mc->extract_device(*grid);
      CHECK(populated.ok() && populated->triangle_count > 100);
    }
  }
  return 0;
}

// Refusing a gate must neither invalidate a published mesh nor record work.
// An offset word in a valid storage buffer remains usable after each refusal.
int gate_metadata_case(vkc::test::Gpu& gpu) {
  auto grid = sphere_grid(gpu);
  auto mc = make_mc(gpu);
  CHECK(grid.ok() && mc.ok());
  const std::uint32_t words[] = {0, 1};
  auto storage = vkc::test::upload_device_buffer(gpu.device, gpu.allocator,
                                                 words, sizeof(words));
  vkc::BufferDesc desc;
  desc.size = sizeof(words);
  desc.usage =
      VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
  auto transfer = gpu.allocator.create_buffer(desc);
  desc.size = 7;
  desc.usage |= VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
  auto truncated = gpu.allocator.create_buffer(desc);
  CHECK(storage.ok() && transfer.ok() && truncated.ok());
  vkc::Buffer empty;
  vkc::Buffer unknown(
      storage->handle(), storage->size(), storage->usage(),
      storage->sharing_mode(), nullptr, [] {}, std::nullopt);
  auto memory = storage->memory_info();
  CHECK(memory.has_value());
  memory->properties &= ~VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
  vkc::Buffer nonlocal(
      storage->handle(), storage->size(), storage->usage(),
      storage->sharing_mode(), nullptr, [] {}, memory);
  vkc::Buffer oversized(
      storage->handle(), vkc::max_storage_buffer_range(gpu.device) + 4,
      storage->usage(), storage->sharing_mode(), nullptr, [] {},
      storage->memory_info());
  const vol::DeviceCount invalid[] = {
      {nullptr, 0, 1},
      {&empty, 0, 1},
      {&*transfer, 0, 1},
      {&*truncated, 4, 1},
      {&*storage, 1, 1},
      {&*storage, 8, 1},
      {&*storage, std::numeric_limits<VkDeviceSize>::max() - 3, 1},
      {&unknown, 0, 1},
      {&nonlocal, 0, 1},
      {&oversized, 0, 1}};
  for (const auto& gate : invalid) {
    auto held = mc->extract_device(*grid);
    CHECK(held.ok() && held->triangle_count > 100);
    const auto before = gpu.device.counters();
    vkc::CommandBatch batch(gpu.device, gpu.allocator);
    auto refused =
        mc->record_extract(batch, *grid, grid->map().allocated_blocks(), &gate);
    CHECK(!refused.ok() && refused.status().domain() == C::InvalidArgument);
    CHECK(held->is_current());
    CHECK(batch.submit().ok());
    CHECK(gpu.device.counters().dispatches == before.dispatches);
    const vol::DeviceCount open{&*storage, 4, 1};
    auto next =
        recorded(gpu, *mc, *grid, grid->map().allocated_blocks(), &open);
    CHECK(next.ok() && next->has_value());
    CHECK((*next)->triangle_count == held->triangle_count);
  }
  return 0;
}

// Abandoned extracts, and tokens dropped unfinished, give their slots back:
// many more than slot_count, with nothing released, all record. A pending
// superseded by the next record, or whose batch skipped the signal, is
// refused.
int slots_case(vkc::test::Gpu& gpu) {
  vkc::Result<vol::VoxelBlockGrid> grid = sphere_grid(gpu);
  vkc::Result<mesh::MarchingCubes> mc = make_mc(gpu, false, 2);
  CHECK(grid.ok() && mc.ok());
  const std::uint32_t n = grid->map().allocated_blocks();
  for (int i = 0; i < 6; ++i) {
    vkc::CommandBatch batch(gpu.device, gpu.allocator);
    vkc::Result<mesh::PendingExtract> pending =
        mc->record_extract(batch, *grid, n);
    CHECK(pending.ok());
    vkc::Result<vkc::PendingBatch> ran =
        batch.submit_async({}, {pending->signal});
    CHECK(ran.ok() && ran->wait().ok());
    if (i % 2 != 0) continue;  // dropped
    mc->abandon(*pending);
    CHECK(mc->finish(*pending, *grid).status().domain() == C::InvalidArgument);
  }
  vkc::Result<std::optional<mesh::DeviceMesh>> held =
      recorded(gpu, *mc, *grid, n);
  CHECK(held.ok() && held->has_value());

  // Superseded: the first pending is no longer the extractor's.
  mesh::PendingExtract first;
  {
    vkc::CommandBatch batch(gpu.device, gpu.allocator);
    vkc::Result<mesh::PendingExtract> p = mc->record_extract(batch, *grid, n);
    CHECK(p.ok());
    first = *p;
    vkc::Result<vkc::PendingBatch> ran = batch.submit_async({}, {p->signal});
    CHECK(ran.ok() && ran->wait().ok());
  }
  mc->release_through(held->value().generation);
  vkc::Result<std::optional<mesh::DeviceMesh>> second =
      recorded(gpu, *mc, *grid, n);
  CHECK(second.ok() && second->has_value());
  CHECK(mc->finish(first, *grid).status().domain() == C::InvalidArgument);
  mc->abandon(first);  // ignored

  // A batch that did not set the signal: refused once the grace is over.
  mc->release_through(second->value().generation);
  vkc::Result<std::optional<mesh::DeviceMesh>> unsignalled =
      recorded(gpu, *mc, *grid, n, nullptr, nullptr, true);
  CHECK(unsignalled.status().domain() == C::InvalidArgument);
  CHECK(unsignalled.status().message().find("signal") != std::string::npos);
  vkc::Result<std::optional<mesh::DeviceMesh>> after =
      recorded(gpu, *mc, *grid, n);
  CHECK(after.ok() && after->has_value());

  // Another extractor's token of the same generation is not this one's.
  vkc::Result<mesh::MarchingCubes> mine = make_mc(gpu);
  vkc::Result<mesh::MarchingCubes> other = make_mc(gpu);
  CHECK(mine.ok() && other.ok());
  mesh::PendingExtract tokens[2];
  for (int i = 0; i < 2; ++i) {
    vkc::CommandBatch batch(gpu.device, gpu.allocator);
    vkc::Result<mesh::PendingExtract> p =
        (i == 0 ? *mine : *other).record_extract(batch, *grid, n);
    CHECK(p.ok());
    tokens[i] = *p;
    vkc::Result<vkc::PendingBatch> ran = batch.submit_async({}, {p->signal});
    CHECK(ran.ok() && ran->wait().ok());
  }
  CHECK(tokens[0].generation == tokens[1].generation);
  mine->abandon(tokens[1]);  // ignored
  vkc::Result<std::optional<mesh::DeviceMesh>> kept =
      mine->finish(tokens[0], *grid);
  CHECK(kept.ok() && kept->has_value());
  return 0;
}

// --- Lifetime, cost and refusals.
// ---------------------------------------------

// The timeline lives with the extractor's state, so a mesh's ready point
// stays good when the extractor moves.
int move_case(vkc::test::Gpu& gpu) {
  vkc::Result<vol::VoxelBlockGrid> grid = sphere_grid(gpu);
  vkc::Result<mesh::MarchingCubes> mc = make_mc(gpu);
  CHECK(grid.ok() && mc.ok());
  vkc::Result<std::optional<mesh::DeviceMesh>> got =
      recorded(gpu, *mc, *grid, grid->map().allocated_blocks());
  CHECK(got.ok() && got->has_value());
  const mesh::DeviceMesh m = **got;
  mesh::MarchingCubes moved = std::move(*mc);
  CHECK(m.ready.semaphore->wait(m.ready.value, 0).ok());
  vkc::Result<mesh::Mesh> host = moved.download(m);
  CHECK(host.ok() && host->triangle_count() == m.triangle_count);
  return 0;
}

// Once warm, an extract costs one submit and no staging buffer: recorded;
// blocking after a grid change, its compaction in that submit; and blocking
// while the map's last list holds, which it binds, compacting nothing and
// leaving the list holding.
int counters_case(vkc::test::Gpu& gpu) {
  vkc::Result<vol::VoxelBlockGrid> grid = sphere_grid(gpu);
  vkc::Result<mesh::MarchingCubes> mc = make_mc(gpu);
  CHECK(grid.ok() && mc.ok());
  const std::uint32_t n = grid->map().allocated_blocks();
  for (int frame = 0; frame < 3; ++frame) {
    const vkc::DeviceCounters before = gpu.device.counters();
    vkc::Result<std::optional<mesh::DeviceMesh>> got =
        recorded(gpu, *mc, *grid, n);
    const vkc::DeviceCounters after = gpu.device.counters();
    CHECK(got.ok() && got->has_value());
    if (frame == 0) continue;  // the first makes the results ring and slot
    CHECK(after.submits - before.submits == 1);
    CHECK(after.staging_buffers - before.staging_buffers == 0);
  }
  vkc::Result<mesh::MarchingCubes> blocking = make_mc(gpu);
  CHECK(blocking.ok());
  for (int frame = 0; frame < 4; ++frame) {
    // An unobserved block far from the sphere: the grid changes, its mesh
    // does not.
    vol::BlockIndex extra{};
    extra.coord = vgrid::Vec3i(20 + frame, 0, 0);
    CHECK(vgrid_test::allocate_blocks(gpu, grid->map(), {extra}).value() == 0);
    CHECK(!grid->map().held_device_list().has_value());
    vkc::StageMetrics m;
    mesh::ExtractTimings t;
    const vkc::DeviceCounters before = gpu.device.counters();
    vkc::Result<mesh::DeviceMesh> got =
        blocking->extract_device(*grid, 0.0f, &t, &m);
    const vkc::DeviceCounters after = gpu.device.counters();
    CHECK(got.ok() && got->triangle_count > 100);
    CHECK(find_row(m, "  ..compact") != nullptr);
    CHECK(t.compact_ms > 0.0 && t.descriptor_ms > 0.0 && t.dispatch_ms > 0.0);
    if (frame == 0) continue;
    CHECK(after.submits - before.submits == 1);
    CHECK(after.staging_buffers - before.staging_buffers == 0);
    CHECK(after.host_waits - before.host_waits == 1);
  }
  vkc::Result<vol::DeviceBlockList> held =
      grid->map().compact_active_blocks_on_device();
  CHECK(held.ok());
  for (int frame = 0; frame < 2; ++frame) {
    vkc::StageMetrics m;
    mesh::ExtractTimings t;
    const vkc::DeviceCounters before = gpu.device.counters();
    vkc::Result<mesh::DeviceMesh> got =
        blocking->extract_device(*grid, 0.0f, &t, &m);
    const vkc::DeviceCounters after = gpu.device.counters();
    CHECK(got.ok() && got->triangle_count > 100);
    CHECK(find_row(m, "  ..compact") == nullptr && t.compact_ms == 0.0);
    CHECK(t.descriptor_ms > 0.0);
    CHECK(after.submits - before.submits == 1);
    CHECK(after.staging_buffers - before.staging_buffers == 0);
    CHECK(grid->map().check_device_block_list(*held, "test").ok());
  }
  return 0;
}

// A grid whose blocks the sharing kernel's table cannot hold (16^3 voxels) is
// refused before anything is recorded or claimed.
int refusal_case(vkc::test::Gpu& gpu) {
  const vol::AttributeSpec attrs[] = {{vol::kTsdfAttribute, 4},
                                      {vol::kWeightAttribute, 4}};
  vkc::Result<vol::VoxelBlockGrid> big = vol::VoxelBlockGrid::create(
      gpu.device, gpu.allocator, grid_params(16), attrs, 2);
  vkc::Result<mesh::MarchingCubes> mc = make_mc(gpu, true);
  CHECK(big.ok() && mc.ok());
  vkc::CommandBatch batch(gpu.device, gpu.allocator);
  CHECK(mc->record_extract(batch, *big, 1).status().domain() ==
        C::InvalidArgument);
  CHECK(batch.submit().ok());
  vkc::Result<vol::VoxelBlockGrid> grid = sphere_grid(gpu);
  CHECK(grid.ok());
  vkc::Result<std::optional<mesh::DeviceMesh>> fine =
      recorded(gpu, *mc, *grid, grid->map().allocated_blocks());
  CHECK(fine.ok() && fine->has_value());
  return 0;
}

int gpu_main(vkc::test::GpuContext& gpu) {
  if (same_mesh_case(gpu) != 0) return 1;
  if (refit_case(gpu) != 0) return 1;
  if (gate_case(gpu) != 0) return 1;
  if (gate_metadata_case(gpu) != 0) return 1;
  if (empty_subset_case(gpu) != 0) return 1;
  if (slots_case(gpu) != 0) return 1;
  if (move_case(gpu) != 0) return 1;
  if (counters_case(gpu) != 0) return 1;
  if (refusal_case(gpu) != 0) return 1;
  std::printf("grid mesh recorded extract test passed\n");
  return 0;
}

}  // namespace

int main() {
  return vkc::test::run_on_gpu(vgrid::device_requirements(), gpu_main);
}
