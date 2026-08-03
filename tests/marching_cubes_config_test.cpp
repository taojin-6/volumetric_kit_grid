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
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <utility>
#include <vector>

#include "volumetric_kit/recon/core/allocator.hpp"
#include "volumetric_kit/recon/core/buffer.hpp"
#include "volumetric_kit/recon/core/compute_util.hpp"
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
  // The command is unconditional, not config-gated: a recon-only consumer still
  // gets one. Asserted on the *default* extractor specifically, because every
  // other command assertion here runs through a configured one -- gating the
  // buffer on config would otherwise leave the suite green while a default
  // DeviceMesh carried a null handle.
  CHECK(plain_mesh.value().indirect != VK_NULL_HANDLE);
  CHECK(plain_mesh.value().indirect_usage ==
        (VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
         VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT));
  // Default config names no families, so all three stay EXCLUSIVE -- bit for
  // bit what this tier allocated before any of it existed.
  CHECK(plain_mesh.value().sharing_mode == VK_SHARING_MODE_EXCLUSIVE);

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

  // And the indirect field: it takes consumer usage the same way, so leaving it
  // out of the guard would let the bit through on the one buffer nothing else
  // checks.
  mesh::MarchingCubesConfig bad_indirect;
  bad_indirect.extra_indirect_usage = VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
  vr::Result<mesh::MarchingCubes> bad_indirect_result =
      mesh::MarchingCubes::create(device.value(), allocator.value(),
                                  bad_indirect);
  CHECK(!bad_indirect_result.ok());
  CHECK(bad_indirect_result.status().domain() ==
        vr::Status::Code::InvalidArgument);

  // A queue-family list longer than the fixed array is refused at create, not
  // at the first arena grow several frames into a scan.
  mesh::MarchingCubesConfig bad_families;
  bad_families.queue_family_count = vr::BufferDesc::kMaxQueueFamilies + 1;
  vr::Result<mesh::MarchingCubes> bad_families_result =
      mesh::MarchingCubes::create(device.value(), allocator.value(),
                                  bad_families);
  CHECK(!bad_families_result.ok());
  CHECK(bad_families_result.status().domain() ==
        vr::Status::Code::InvalidArgument);

  // --- Queue families reach every output buffer ------------------------------
  // Naming two distinct families must give CONCURRENT on all three, or a
  // renderer on the second family reads a buffer its queue does not own. Only
  // runs where the device actually exposes two families -- on a single-family
  // driver (lavapipe, both Linux CI legs) two indices would collapse to one and
  // EXCLUSIVE is the correct answer, which is asserted instead.
  {
    std::uint32_t family_count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(device.value().physical_device(),
                                             &family_count, nullptr);
    mesh::MarchingCubesConfig shared_config;
    shared_config.queue_families[0] = 0;
    shared_config.queue_families[1] = family_count > 1 ? 1 : 0;
    shared_config.queue_family_count = 2;
    vr::Result<mesh::MarchingCubes> shared_result = mesh::MarchingCubes::create(
        device.value(), allocator.value(), shared_config);
    CHECK(shared_result.ok());
    mesh::MarchingCubes shared = std::move(shared_result).value();
    vr::Result<mesh::DeviceMesh> shared_mesh =
        shared.extract_device(small, 0.0f);
    CHECK(shared_mesh.ok());
    const VkSharingMode expected = family_count > 1 ? VK_SHARING_MODE_CONCURRENT
                                                    : VK_SHARING_MODE_EXCLUSIVE;
    CHECK(shared_mesh.value().sharing_mode == expected);
  }

  // --- The indirect draw command --------------------------------------------
  // Present, and carrying the usage a vkCmdDrawIndexedIndirect needs beside the
  // STORAGE_BUFFER the kernel counts through -- the same verify-don't-assume
  // reason the vertex and index usage come back.
  //
  // Asserted on `big_mesh`, NOT `small_mesh`: the big extract ran on the same
  // extractor at the default slot_count == 1 and grew the arena (checked
  // above), so small_mesh's vertex and index handles have already been through
  // vmaDestroyBuffer. Reading a superseded view is the very thing download()
  // refuses, and a test that did it would start dereferencing freed buffers the
  // moment the command joins the grow path.
  CHECK(big_mesh.value().indirect != VK_NULL_HANDLE);
  CHECK((big_mesh.value().indirect_usage &
         VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT) != 0);
  CHECK((big_mesh.value().indirect_usage &
         VK_BUFFER_USAGE_STORAGE_BUFFER_BIT) != 0);
  // valid() covers all three buffers, so a null command cannot pass for a
  // drawable mesh.
  CHECK(big_mesh.value().valid());

  // The command's *contents*, which is the half nothing pinned before: a
  // consumer drawing indirectly reads these five fields and never consults
  // triangle_count, so an indexCount in the wrong units renders a third of the
  // mesh with every other assertion in this suite still green (verified by
  // mutation: `emitted * kIndicesPerTriangle` -> `emitted` passed everything).
  //
  // Read back through a copy rather than a map, because the test holds only a
  // VkBuffer. That makes this a genuine consumer of extra_indirect_usage -- the
  // TRANSFER_SRC bit below is the config knob under test, so the readback and
  // the flag prove each other.
  {
    mesh::MarchingCubesConfig cmd_config;
    cmd_config.extra_indirect_usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    vr::Result<mesh::MarchingCubes> cmd_result = mesh::MarchingCubes::create(
        device.value(), allocator.value(), cmd_config);
    CHECK(cmd_result.ok());
    mesh::MarchingCubes cmd_extractor = std::move(cmd_result).value();

    vr::Result<mesh::DeviceMesh> cmd_mesh =
        cmd_extractor.extract_device(small, 0.0f);
    CHECK(cmd_mesh.ok());
    CHECK(!cmd_mesh.value().empty());
    CHECK((cmd_mesh.value().indirect_usage &
           VK_BUFFER_USAGE_TRANSFER_SRC_BIT) != 0);

    vr::Result<vr::Buffer> staging = vr::storage_buffer(
        allocator.value(), sizeof(VkDrawIndexedIndirectCommand),
        vr::HostAccess::Random, VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    CHECK(staging.ok());

    const VkBuffer src = cmd_mesh.value().indirect;
    const VkBuffer dst = staging.value().handle();
    CHECK(device.value()
              .submit_single_time([src, dst](VkCommandBuffer cb) {
                VkBufferCopy region{};
                region.size = sizeof(VkDrawIndexedIndirectCommand);
                vkCmdCopyBuffer(cb, src, dst, 1, &region);
              })
              .ok());

    VkDrawIndexedIndirectCommand cmd{};
    std::memcpy(&cmd, staging.value().mapped(), sizeof(cmd));
    // The units: three indices per triangle, which is what makes the kernel's
    // append atomic the command itself.
    CHECK(cmd.indexCount == cmd_mesh.value().triangle_count * 3);
    CHECK(cmd.indexCount == cmd_mesh.value().vertex_count);
    // The four fields that make it *drawable* rather than just a number. They
    // have no effect until something issues a draw, and nothing in recon can --
    // which is exactly why they need pinning here instead of downstream.
    CHECK(cmd.instanceCount == 1);
    CHECK(cmd.firstIndex == 0);
    CHECK(cmd.vertexOffset == 0);
    CHECK(cmd.firstInstance == 0);
  }

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
  // The command rings with them. It is per-mesh, not per-extractor: a consumer
  // drawing gen1 indirectly reads this buffer for the duration of its frame, so
  // sharing one would hand gen2's count to gen1's draw.
  CHECK(gen2.value().indirect != gen1.value().indirect);

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

  // The arena must not ratchet as the ring turns.
  //
  // plan_capacity used to floor its estimate at what the arena already held.
  // Read against the slot just written while a *different* slot was about to be
  // grown, that floor compounds 1.5x per extract -- geometrically, with the
  // measured triangle density never getting a say. On an iPad Pro it reached a
  // 1.1 GB arena for 36904 triangles (0.59% full) before vkAllocateMemory
  // refused and the device was lost.
  //
  // Same grid every time, so a correct extractor settles and stays there. Two
  // assertions, because each is blind to something the other catches: flatness
  // holds perfectly under a revert to worst-case sizing (which is flat, just
  // ~19x too big), and the magnitude bound holds under any drift slow enough to
  // stay inside it. The 1.5x compound this fixes trips both; the checks below
  // were each confirmed to fail against a re-introduction of it.
  {
    constexpr std::uint32_t kRatchetSlots = 3;
    // Four turns of the ring: two to let the seeded plan settle, two to hold it
    // flat. Derived from the slot count, not hardcoded beside it, so changing
    // one does not silently make the assertions below vacuous.
    constexpr int kRatchetExtracts = 4 * static_cast<int>(kRatchetSlots);

    mesh::MarchingCubesConfig ratchet_config;
    ratchet_config.slot_count = kRatchetSlots;
    vr::Result<mesh::MarchingCubes> ratchet_result =
        mesh::MarchingCubes::create(device.value(), allocator.value(),
                                    ratchet_config);
    CHECK(ratchet_result.ok());
    mesh::MarchingCubes ratchet = std::move(ratchet_result).value();

    std::uint64_t bytes[kRatchetExtracts] = {};
    for (int i = 0; i < kRatchetExtracts; ++i) {
      mesh::ExtractTimings t{};
      vr::Result<mesh::DeviceMesh> m = ratchet.extract_device(small, 0.0f, &t);
      CHECK(m.ok());
      // Released immediately: this is testing the plan, not the ring's ability
      // to refuse, and an exhausted ring would end the loop early.
      ratchet.release_through(m.value().generation);
      // The sum over every slot, not the one this extract wrote -- which is
      // what makes it the extractor's real resident cost and what makes the
      // flatness check below meaningful across the ring.
      bytes[i] = t.arena_bytes;
      CHECK(t.emitted_triangles > 0);

      // (1) What the extractor holds tracks the *surface*, and it accounts for
      // every slot. Measured here: 1052 triangles of surface, 911232 bytes of
      // arena once all three slots are sized -- 4.5x, which is the 1.5x growth
      // headroom times the three slots. Both bounds have a 1.5x margin or
      // better, and each catches a different mutation.
      //
      // Above 9x: a revert to `return worst_case;`. This fixture's ceiling is 5
      // triangles per cell over 8 blocks of 512 voxels, ~19x its actual surface
      // *per slot* -- and perfectly flat, so (2) cannot see it.
      //
      // Below 3x: reporting one slot instead of the ring (`arena().size()`),
      // which lands at ~1.5x and would make this whole loop blind to two thirds
      // of the memory it exists to bound. Started once every slot has been
      // sized, since before that the sum is genuinely smaller.
      const std::uint64_t surface_bytes =
          static_cast<std::uint64_t>(t.emitted_triangles) *
          mesh::kIndicesPerTriangle * sizeof(mesh::Vertex);
      CHECK(bytes[i] <= surface_bytes * 3 * kRatchetSlots);
      if (i >= static_cast<int>(kRatchetSlots)) {
        CHECK(bytes[i] >= surface_bytes * kRatchetSlots);
      }

      // (2) Flat. Every slot has been sized by the end of the first turn and
      // the grid never changes, so from the second turn on no slot may grow
      // again. Started at the second turn rather than the first so a settling
      // plan is not mistaken for a drift. This is the assertion that survives a
      // change of fixture: a compound gentle enough to stay inside (1) for
      // twelve extracts still cannot be flat.
      if (i >= 2 * static_cast<int>(kRatchetSlots)) {
        CHECK(bytes[i] == bytes[i - 1]);
      }
    }
    CHECK(bytes[kRatchetExtracts - 1] > 0);
  }

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

  // --- The host extract overloads do not starve the ring ---------------------
  // They claim and stamp a slot like any extract, but return Result<Mesh> --
  // no generation, so a host-only caller cannot release what it never saw, and
  // download() does not release either. Left alone, slot_count calls exhaust
  // the ring and every extract after that fails permanently, quoting
  // generations the API never handed out. The overloads therefore release
  // their own slot once the host copy is taken.
  //
  // Deterministic, not racy: with two slots this bricks on call three. Deleting
  // either release_through() in the host overloads fails this loop.
  {
    mesh::MarchingCubesConfig host_ring;
    host_ring.slot_count = 2;
    vr::Result<mesh::MarchingCubes> host_result = mesh::MarchingCubes::create(
        device.value(), allocator.value(), host_ring);
    CHECK(host_result.ok());
    mesh::MarchingCubes host = std::move(host_result).value();

    for (int i = 0; i < 5; ++i) {
      vr::Result<mesh::Mesh> host_mesh = host.extract(small, 0.0f);
      CHECK(host_mesh.ok());
      CHECK(!host_mesh.value().vertices.empty());
    }

    // The dense overload shares the ring and mixes with the sparse one, so it
    // has to release too.
    const int dense_dim = 8;
    std::vector<vol::Voxel> samples(
        static_cast<std::size_t>(dense_dim * dense_dim * dense_dim));
    for (int z = 0; z < dense_dim; ++z) {
      for (int y = 0; y < dense_dim; ++y) {
        for (int x = 0; x < dense_dim; ++x) {
          const float dx = static_cast<float>(x) - 3.5f;
          const float dy = static_cast<float>(y) - 3.5f;
          const float dz = static_cast<float>(z) - 3.5f;
          vol::Voxel& v = samples[static_cast<std::size_t>(
              (z * dense_dim + y) * dense_dim + x)];
          v.sdf = std::sqrt(dx * dx + dy * dy + dz * dz) - 2.0f;
          v.weight = 1.0f;
        }
      }
    }
    mesh::DenseGrid dense_grid;
    dense_grid.dims = vr::Vec3i{dense_dim, dense_dim, dense_dim};
    dense_grid.voxel_size = 1.0f;
    dense_grid.origin = vr::Vec3f{0.0f, 0.0f, 0.0f};
    for (int i = 0; i < 5; ++i) {
      vr::Result<mesh::Mesh> dense_mesh =
          host.extract(samples.data(), samples.size(), dense_grid, 0.0f);
      CHECK(dense_mesh.ok());
    }

    // ...and they must give back *their own* slot, not everything below it.
    //
    // Handing the slot back is required (a Result<Mesh> carries no generation,
    // so a host-only caller could not release it), but doing it with
    // release_through -- the *consumer's* high-water mark -- also retires every
    // older slot. Mixed with extract_device on the same extractor, that frees a
    // slot the consumer is still drawing out of, and the next grow runs
    // vmaDestroyBuffer under a live draw: undefined behaviour with validation
    // off, which is the shipping configuration and the only one on iOS. The
    // loops above cannot see it -- they use a dedicated extractor and hand out
    // no DeviceMesh at all.
    for (int overload = 0; overload < 2; ++overload) {
      mesh::MarchingCubesConfig mixed_config;
      mixed_config.slot_count = 2;
      vr::Result<mesh::MarchingCubes> mixed_result =
          mesh::MarchingCubes::create(device.value(), allocator.value(),
                                      mixed_config);
      CHECK(mixed_result.ok());
      mesh::MarchingCubes mixed = std::move(mixed_result).value();

      // Outstanding for the rest of the block, and never released: this stands
      // in for a renderer with the mesh bound in an in-flight draw.
      vr::Result<mesh::DeviceMesh> live = mixed.extract_device(small, 0.0f);
      CHECK(live.ok());

      // A host extract takes the *other* slot and gives it straight back. Both
      // overloads share the ring, so both are run through this.
      if (overload == 0) {
        vr::Result<mesh::Mesh> host_copy = mixed.extract(small, 0.0f);
        CHECK(host_copy.ok());
      } else {
        vr::Result<mesh::Mesh> host_copy =
            mixed.extract(samples.data(), samples.size(), dense_grid, 0.0f);
        CHECK(host_copy.ok());
      }

      // So a slot is free for this one -- and it must be the freed one, never
      // the one `live` names.
      vr::Result<mesh::DeviceMesh> next = mixed.extract_device(small, 0.0f);
      CHECK(next.ok());
      CHECK(next.value().vertices != live.value().vertices);

      // Both slots are outstanding again, so the ring refuses. This is the
      // handle-independent half: with release_through in the host overload,
      // `live`'s generation would have been marked released, `next` would have
      // landed on `live`'s slot, and the other slot would still be free here --
      // so this would succeed.
      vr::Result<mesh::DeviceMesh> refused = mixed.extract_device(small, 0.0f);
      CHECK(!refused.ok());
    }
  }

  // --- An empty extract is a first-class result ------------------------------
  // A grid with no active blocks meshes nothing, and that path returns without
  // ever reaching a dispatch. It still has to behave like every other extract:
  // name its command, claim its own slot, and leave a consumer able to draw
  // (nothing) without special-casing.
  {
    vr::Result<vol::VoxelBlockGrid> empty_result = vol::VoxelBlockGrid::create(
        device.value(), allocator.value(), gp, attrs, 2);
    CHECK(empty_result.ok());
    vol::VoxelBlockGrid empty_grid =
        std::move(empty_result).value();  // unfilled

    mesh::MarchingCubesConfig empty_config;
    empty_config.slot_count = 2;
    vr::Result<mesh::MarchingCubes> empty_ex = mesh::MarchingCubes::create(
        device.value(), allocator.value(), empty_config);
    CHECK(empty_ex.ok());
    mesh::MarchingCubes ex = std::move(empty_ex).value();

    // A real mesh first, so the empty extract that follows has a live previous
    // generation to *not* disturb.
    vr::Result<mesh::DeviceMesh> real = ex.extract_device(small, 0.0f);
    CHECK(real.ok());
    CHECK(!real.value().empty());

    vr::Result<mesh::DeviceMesh> none = ex.extract_device(empty_grid, 0.0f);
    CHECK(none.ok());
    CHECK(none.value().empty());
    CHECK(none.value().triangle_count == 0);
    // The command is named and zeroed, not left null and not left holding the
    // previous extract's count -- a consumer drawing indirectly reads it and
    // never looks at triangle_count, so either would silently redraw the
    // previous mesh.
    CHECK(none.value().indirect != VK_NULL_HANDLE);
    CHECK(none.value().indirect != real.value().indirect);
    // It took a slot of its own. That is what keeps release_through honest:
    // released_through_ is a high-water mark, so an empty generation that
    // shared -- or skipped -- a slot would mark the live mesh's slot released
    // the moment the consumer retired the empty one, and the next grow would
    // free an arena under a live draw.
    CHECK(none.value().generation != real.value().generation);
    ex.release_through(none.value().generation);
    // Both slots are now spoken for (real is still outstanding by contract, but
    // release_through is a high-water mark, so this releases it too) -- the
    // point being simply that the empty extract consumed a slot rather than
    // vanishing, which the generation inequality above already shows.
    vr::Result<mesh::Mesh> none_host = ex.download(none.value());
    CHECK(none_host.ok());
    CHECK(none_host.value().vertices.empty());
  }

  std::fprintf(stderr, "marching_cubes_config: OK\n");
  return 0;
}
