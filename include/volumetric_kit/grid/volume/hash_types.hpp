// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file volume/hash_types.hpp
/// @brief POD storage layouts for the sparse voxel hash map.
///
/// These are the on-device storage layouts shared by the CPU, the Vulkan/GLSL
/// compute shaders, and the optional CUDA accelerator. They are deliberately
/// plain structs with `static_assert`'d sizes/offsets so the host and device
/// views agree byte-for-byte -- a layout drift between the host allocator and a
/// device kernel is a silent corruption bug, so it is made a compile error.
///
/// The layout here is the C/CUDA layout (a bare `Vec3i` packs to 12 B, so
/// `HashEntry` is 20 B with `pos` at offset 4). The GLSL side reads these via
/// **scalar block layout** (`GL_EXT_scalar_block_layout`; Vulkan 1.2 core,
/// MoltenVK-supported), under which the shader struct is byte-identical to this
/// one. A naive `std430` block does *not* match -- `std430` 16-byte-aligns a
/// three-component vector, placing `pos` at offset 16 and spanning 32 B. The
/// `static_assert`s below guard only the host side; the shader keeps its
/// `layout(scalar)` definition in lockstep (see the gotchas in CLAUDE.md).
///
/// The sparse-hashing scheme (a hash table of block coordinates into a heap of
/// fixed-size voxel blocks) keeps memory proportional to the observed surface
/// rather than the bounding volume. The prior engine's per-voxel neural
/// "feature"/triplane channels are intentionally absent here (see the exclusion
/// policy in CLAUDE.md): this carries only SDF, weight, and optional color.

#include <cstddef>
#include <cstdint>

#include "volumetric_kit/recon/core/device_macros.hpp"
#include "volumetric_kit/recon/core/math/vector_types.hpp"

namespace volumetric_kit::recon::volume {

/// Hash-table entry -- one per slot in the table.
struct HashEntry {
  std::int32_t ptr;     ///< Voxel-block pointer, or an allocated/free flag.
  Vec3i pos;            ///< Block coordinate (signed, 32-bit per axis).
  std::int32_t offset;  ///< Collision offset for linked-list chaining.
};
static_assert(sizeof(HashEntry) == 20, "HashEntry must be 20 bytes");
static_assert(offsetof(HashEntry, ptr) == 0, "HashEntry layout drift");
static_assert(offsetof(HashEntry, pos) == 4, "HashEntry layout drift");
static_assert(offsetof(HashEntry, offset) == 16, "HashEntry layout drift");

/// Active-block reference used in compacted block lists.
struct BlockIndex {
  Vec3i coord;       ///< Voxel-block coordinate.
  std::int32_t ptr;  ///< Offset into the voxel-block array.
};
static_assert(sizeof(BlockIndex) == 16, "BlockIndex must be 16 bytes");
static_assert(offsetof(BlockIndex, coord) == 0, "BlockIndex layout drift");
static_assert(offsetof(BlockIndex, ptr) == 12, "BlockIndex layout drift");

/// Lightweight, non-owning view of a compacted block array.
struct BlockList {
  const BlockIndex* blocks = nullptr;
  std::int32_t count = 0;
};

/// Per-voxel signed-distance payload.
struct Voxel {
  float sdf;     ///< Signed distance (meters).
  float weight;  ///< Integration weight (clamped to a configured maximum).

  /// @return The stored signed distance.
  VR_DEVICE_HOST float get_sdf() const { return sdf; }
  /// Set the signed distance.
  VR_DEVICE_HOST void set_sdf(float value) { sdf = value; }
};
static_assert(sizeof(Voxel) == 8, "Voxel must be 8 bytes");
static_assert(offsetof(Voxel, sdf) == 0, "Voxel layout drift");
static_assert(offsetof(Voxel, weight) == 4, "Voxel layout drift");

/// Unified, layout-decoupled view of a hash table's per-voxel data.
///
/// Integrators and mesh extraction read SDF (and optional color) through this
/// rather than reaching into @ref HashTable internals, so new per-voxel
/// channels can be added without touching every consumer.
struct VoxelData {
  Voxel* sdf_blocks = nullptr;     ///< SDF + weight (non-null once allocated).
  Vec3u8* color_blocks = nullptr;  ///< Optional per-voxel RGB (nullptr = off).
};

/// Device-side voxel hash table: the block index plus its backing storage.
struct HashTable {
  HashEntry* hash_entries = nullptr;      ///< Hash-table slots.
  Voxel* sdf_blocks = nullptr;            ///< Voxel blocks (SDF + weight).
  Vec3u8* color_blocks = nullptr;         ///< Optional per-voxel color.
  std::uint32_t* heap = nullptr;          ///< Free-block allocation stack.
  std::uint32_t* heap_counter = nullptr;  ///< Heap top (atomic on device).
  std::int32_t* bucket_mutex = nullptr;   ///< Per-bucket spin locks.

  /// @return A @ref VoxelData view of this table's voxel storage.
  VR_DEVICE_HOST VoxelData voxel_data() const {
    return VoxelData{sdf_blocks, color_blocks};
  }
};

}  // namespace volumetric_kit::recon::volume
