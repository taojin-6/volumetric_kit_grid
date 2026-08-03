// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// GPU test for MarchingCubesConfig -- the usage bits a *consumer* of the mesh
// declares, OR-ed onto STORAGE_BUFFER for the vertex arena and the index run.
//
// The feature has no in-tree consumer (recon itself only ever needs storage),
// so without this test the one behavioural line -- `desc.usage | extra_usage`
// -- would never execute with a non-zero value in any CI leg, and dropping
// `config_` from the grow path would pass the whole suite. What makes the
// assertion possible is that the flags come back: Vulkan cannot be asked what a
// VkBuffer was created with, so core::Buffer records its usage() and DeviceMesh
// carries vertex_usage / index_usage.
//
// A grow is exercised as well as the first allocation. Today they are the same
// call site -- ensure_output_buffers early-returns on the steady state and
// otherwise reallocates -- so this is not a second code path but a guard on the
// property config_ claims: a reallocated arena carries the same usage. The test
// forces a real grow (a small grid, then a much larger one) and asserts the
// arena actually grew, so the case cannot pass vacuously by never reallocating.
// Exits 0 (skip) where no device is present.

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

constexpr int kBlock = 8;    // voxels per block edge
constexpr float kH = 0.05f;  // metres between voxels

vol::VoxelGridParams grid_params() {
  vol::VoxelGridParams gp{};
  gp.voxel_size = kH;
  gp.block_size = kBlock;
  gp.voxels_per_block = kBlock * kBlock * kBlock;
  gp.trunc_dist = 0.04f;  // unused by meshing; must pass validate()
  gp.bucket_size = 8;
  gp.num_buckets = 128;
  gp.num_blocks = 1024;  // = bucket_size * num_buckets; >> 216 active blocks
  gp.max_chain = 128;
  return gp;
}

// Fill every block of a `blocks`^3 cube with a sphere SDF centred in it, so the
// surface crosses the volume and marching cubes emits a triangle count that
// scales with the block count -- which is what lets the second extract outgrow
// the arena the first one sized.
bool fill_sphere(vol::VoxelBlockGrid& g, int blocks) {
  const int n = kBlock * blocks;
  const float centre = static_cast<float>(n - 1) * 0.5f * kH;
  const float radius = static_cast<float>(n - 1) * 0.35f * kH;

  std::vector<vol::BlockIndex> coords;
  for (int cz = 0; cz < blocks; ++cz) {
    for (int cy = 0; cy < blocks; ++cy) {
      for (int cx = 0; cx < blocks; ++cx) {
        vol::BlockIndex b{};
        b.coord = vr::Vec3i(cx, cy, cz);
        coords.push_back(b);
      }
    }
  }
  vr::Result<std::uint32_t> failed = g.map().allocate(
      coords.data(), static_cast<std::uint32_t>(coords.size()));
  if (!failed || failed.value() != 0) {
    return false;
  }
  vr::Result<std::vector<vol::BlockIndex>> active =
      g.map().compact_active_blocks();
  if (!active || active.value().size() != coords.size()) {
    return false;
  }

  vr::Result<vol::AttributeView> tsdf = g.attribute("tsdf");
  vr::Result<vol::AttributeView> weight = g.attribute("weight");
  if (!tsdf || !weight) {
    return false;
  }
  auto* tptr = static_cast<float*>(tsdf.value().buffer->mapped());
  auto* wptr = static_cast<float*>(weight.value().buffer->mapped());

  for (const vol::BlockIndex& b : active.value()) {
    for (int lz = 0; lz < kBlock; ++lz) {
      for (int ly = 0; ly < kBlock; ++ly) {
        for (int lx = 0; lx < kBlock; ++lx) {
          const int local = lx + kBlock * (ly + kBlock * lz);
          const float x =
              static_cast<float>(b.coord.x * kBlock + lx) * kH - centre;
          const float y =
              static_cast<float>(b.coord.y * kBlock + ly) * kH - centre;
          const float z =
              static_cast<float>(b.coord.z * kBlock + lz) * kH - centre;
          const auto idx = static_cast<std::size_t>(b.ptr) + local;
          tptr[idx] = std::sqrt(x * x + y * y + z * z) - radius;
          wptr[idx] = 1.0f;
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

  const vol::VoxelGridParams gp = grid_params();
  const vol::AttributeSpec attrs[] = {{"tsdf", sizeof(float)},
                                      {"weight", sizeof(float)}};

  // --- A configured extractor carries the consumer's bits --------------------
  mesh::MarchingCubesConfig config;
  config.extra_vertex_usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
  config.extra_index_usage = VK_BUFFER_USAGE_INDEX_BUFFER_BIT;

  vr::Result<mesh::MarchingCubes> mc_result =
      mesh::MarchingCubes::create(device.value(), allocator.value(), config);
  CHECK(mc_result.ok());
  mesh::MarchingCubes extractor = std::move(mc_result).value();

  vr::Result<vol::VoxelBlockGrid> small_result = vol::VoxelBlockGrid::create(
      device.value(), allocator.value(), gp, attrs, 2);
  CHECK(small_result.ok());
  vol::VoxelBlockGrid small = std::move(small_result).value();
  CHECK(fill_sphere(small, 2));

  mesh::ExtractTimings first{};
  vr::Result<mesh::DeviceMesh> small_mesh =
      extractor.extract_device(small, 0.0f, &first);
  CHECK(small_mesh.ok());
  CHECK(!small_mesh.value().empty());

  // Both bits present, and STORAGE_BUFFER never displaced -- the kernel still
  // binds these as SSBOs, so the consumer's usage is added, not substituted.
  CHECK((small_mesh.value().vertex_usage & VK_BUFFER_USAGE_VERTEX_BUFFER_BIT) !=
        0);
  CHECK((small_mesh.value().vertex_usage &
         VK_BUFFER_USAGE_STORAGE_BUFFER_BIT) != 0);
  CHECK((small_mesh.value().index_usage & VK_BUFFER_USAGE_INDEX_BUFFER_BIT) !=
        0);
  CHECK((small_mesh.value().index_usage & VK_BUFFER_USAGE_STORAGE_BUFFER_BIT) !=
        0);
  // Not the other way round: an index bit on the vertex arena would mean the
  // two config fields were crossed.
  CHECK((small_mesh.value().vertex_usage & VK_BUFFER_USAGE_INDEX_BUFFER_BIT) ==
        0);
  CHECK((small_mesh.value().index_usage & VK_BUFFER_USAGE_VERTEX_BUFFER_BIT) ==
        0);

  // --- ...and keeps them across an arena grow --------------------------------
  // The arena is destroyed and rebuilt here, so the flags have to be reapplied
  // rather than surviving in the old allocation.
  vr::Result<vol::VoxelBlockGrid> big_result = vol::VoxelBlockGrid::create(
      device.value(), allocator.value(), gp, attrs, 2);
  CHECK(big_result.ok());
  vol::VoxelBlockGrid big = std::move(big_result).value();
  CHECK(fill_sphere(big, 6));

  mesh::ExtractTimings second{};
  vr::Result<mesh::DeviceMesh> big_mesh =
      extractor.extract_device(big, 0.0f, &second);
  CHECK(big_mesh.ok());
  CHECK(!big_mesh.value().empty());
  // The grow actually happened -- otherwise this case proves nothing.
  CHECK(second.arena_bytes > first.arena_bytes);
  CHECK((big_mesh.value().vertex_usage & VK_BUFFER_USAGE_VERTEX_BUFFER_BIT) !=
        0);
  CHECK((big_mesh.value().index_usage & VK_BUFFER_USAGE_INDEX_BUFFER_BIT) != 0);

  // --- The default is exactly STORAGE_BUFFER ---------------------------------
  // A recon-only consumer pays for nothing it does not use.
  vr::Result<mesh::MarchingCubes> plain_result =
      mesh::MarchingCubes::create(device.value(), allocator.value());
  CHECK(plain_result.ok());
  mesh::MarchingCubes plain = std::move(plain_result).value();

  vr::Result<mesh::DeviceMesh> plain_mesh = plain.extract_device(small, 0.0f);
  CHECK(plain_mesh.ok());
  CHECK(plain_mesh.value().vertex_usage == VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
  CHECK(plain_mesh.value().index_usage == VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

  // --- An unsupported bit is rejected where the caller supplied it -----------
  // Device::create never enables bufferDeviceAddress, so this would otherwise
  // trip a VMA assert inside the first extract's arena grow rather than
  // reporting cleanly from create().
  mesh::MarchingCubesConfig bad_vertex;
  bad_vertex.extra_vertex_usage = VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
  vr::Result<mesh::MarchingCubes> bad_vertex_result =
      mesh::MarchingCubes::create(device.value(), allocator.value(),
                                  bad_vertex);
  CHECK(!bad_vertex_result.ok());
  CHECK(bad_vertex_result.status().domain() ==
        vr::Status::Code::InvalidArgument);

  // The index field is guarded too, not just the vertex one.
  mesh::MarchingCubesConfig bad_index;
  bad_index.extra_index_usage = VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
  vr::Result<mesh::MarchingCubes> bad_index_result =
      mesh::MarchingCubes::create(device.value(), allocator.value(), bad_index);
  CHECK(!bad_index_result.ok());
  CHECK(bad_index_result.status().domain() ==
        vr::Status::Code::InvalidArgument);

  // --- slot_count: the output ring ------------------------------------------
  // A count outside 1..kMaxSlots is refused at create rather than indexing off
  // the end of the slot array on the first extract.
  for (std::uint32_t bad_count : {0u, 9u}) {
    mesh::MarchingCubesConfig bad_slots;
    bad_slots.slot_count = bad_count;
    vr::Result<mesh::MarchingCubes> bad_slots_result =
        mesh::MarchingCubes::create(device.value(), allocator.value(),
                                    bad_slots);
    CHECK(!bad_slots_result.ok());
    CHECK(bad_slots_result.status().domain() ==
          vr::Status::Code::InvalidArgument);
  }

  // Two slots, and the property the whole feature exists for: consecutive
  // extracts land in *different* buffers, so a consumer still reading
  // generation N is not overwritten by N+1. Comparing the VkBuffer handles is
  // the assertion -- with one slot they are necessarily equal, which is exactly
  // the hazard.
  mesh::MarchingCubesConfig ringed;
  ringed.slot_count = 2;
  vr::Result<mesh::MarchingCubes> ring_result =
      mesh::MarchingCubes::create(device.value(), allocator.value(), ringed);
  CHECK(ring_result.ok());
  mesh::MarchingCubes ring = std::move(ring_result).value();

  vr::Result<mesh::DeviceMesh> gen1 = ring.extract_device(small, 0.0f);
  CHECK(gen1.ok());
  CHECK(gen1.value().valid());

  // Nothing released yet, so the only other slot is free and this succeeds.
  vr::Result<mesh::DeviceMesh> gen2 = ring.extract_device(small, 0.0f);
  CHECK(gen2.ok());
  CHECK(gen2.value().valid());
  CHECK(gen2.value().generation != gen1.value().generation);
  CHECK(gen2.value().vertices != gen1.value().vertices);
  CHECK(gen2.value().indices != gen1.value().indices);

  // Now both slots are outstanding. A third extract would have to overwrite the
  // one holding gen1, which the consumer has not finished with -- refused,
  // rather than corrupting a live read.
  vr::Result<mesh::DeviceMesh> gen3 = ring.extract_device(small, 0.0f);
  CHECK(!gen3.ok());
  CHECK(gen3.status().domain() == vr::Status::Code::InvalidArgument);

  // ...and the refusal costs the consumer nothing. This is the whole point of
  // refusing: gen2's slot was not touched, so gen2 must still be downloadable.
  // It is not automatic -- the slot claim has to happen *before* the extract
  // bumps generation_, or download()'s currency check retires the very mesh the
  // refusal was protecting. Moving claim_output_slot() back after the bump
  // fails here, and nothing else in the suite notices.
  vr::Result<mesh::Mesh> gen2_after_refusal = ring.download(gen2.value());
  CHECK(gen2_after_refusal.ok());

  // Releasing gen1 frees its slot and the next extract proceeds. What is
  // asserted is that it did not land on the slot still being read: gen2 is
  // outstanding, so gen4 must not share its buffer.
  //
  // Deliberately not "gen4 reuses gen1's buffer". The ring comes round to the
  // same *slot*, but a slot may grow when it gets there -- the capacity plan
  // moves as the extractor measures triangles per block -- and a grown slot is
  // a new allocation. Pinning the handle would have been testing the growth
  // policy while claiming to test the ring.
  ring.release_through(gen1.value().generation);
  vr::Result<mesh::DeviceMesh> gen4 = ring.extract_device(small, 0.0f);
  CHECK(gen4.ok());
  CHECK(gen4.value().vertices != gen2.value().vertices);

  // Releasing an older generation than the newest reported must not un-release
  // anything: the slot holding gen2 is still outstanding, so this stays
  // refused.
  ring.release_through(0);
  vr::Result<mesh::DeviceMesh> gen5 = ring.extract_device(small, 0.0f);
  CHECK(!gen5.ok());

  // Self-move leaves the ring intact, like every other member. This is not
  // hypothetical: the first cut held the slots in a std::vector, whose
  // self-move-assignment is valid-but-unspecified and empties under libc++, so
  // the extractor passed valid() and then indexed nothing.
  mesh::MarchingCubes* ring_alias = &ring;
  ring = std::move(*ring_alias);
  CHECK(ring.valid());
  ring.release_through(gen4.value().generation);
  vr::Result<mesh::DeviceMesh> after_self_move =
      ring.extract_device(small, 0.0f);
  CHECK(after_self_move.ok());
  CHECK(after_self_move.value().valid());

  std::fprintf(stderr, "marching_cubes_config: OK\n");
  return 0;
}
