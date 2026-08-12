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

/// @brief A compacted list of active blocks -- the subset of a grid the next
///        pass should run over -- borrowed from whoever compacted it.
///
/// Non-owning: it names the caller's storage, typically the `std::vector` from
/// @ref VoxelHashMap::compact_active_blocks or
/// @ref VoxelHashMap::compact_active_blocks_in_frustum, and a consumer reads it
/// for the duration of the call it is passed to and does not retain it.
///
/// @ref epoch is what makes the list safe to carry away from the map that
/// produced it. A @ref BlockIndex::ptr addresses per-voxel attribute storage
/// directly and the block heap is LIFO, so a `remove()` / `clear()` between the
/// compaction and the pass that consumes this hands the same `ptr` to a
/// *different* block -- leaving a list that still typechecks, still indexes in
/// range, and names geometry that is gone. Prefer
/// @ref VoxelBlockGrid::block_list, which stamps the epoch off the grid that
/// owns the blocks so the two cannot be mispaired; a consumer compares it
/// against the grid it is handed rather than trusting the two to have been
/// fetched together.
///
/// @warning Non-owning in @ref blocks *and* unversioned in @ref count: nothing
///          here notices a vector that was reassigned, reallocated by a
///          `push_back`, or refilled shorter while this list still names its
///          old length -- and the epoch cannot catch any of them, since
///          allocate and resize deliberately leave the topology token alone.
///          Rebuild the list beside every change to the storage it names, do
///          not cache one across frames.
///          TODO(volume): an owning `CompactedBlocks { std::vector<BlockIndex>;
///          epoch; view() }` returned straight from the compaction entry points
///          would make all three unrepresentable rather than documented.
struct BlockList {
  /// The compacted blocks. Null only when @ref count is 0.
  const BlockIndex* blocks = nullptr;
  /// How many blocks @ref blocks addresses. Zero is a legal empty set (a
  /// camera looking at nothing), not an error -- and an empty list names no
  /// block, so it is exempt from the @ref epoch check a consumer makes. That is
  /// what lets a default-constructed `BlockList{}` mean "nothing visible"
  /// rather than being refused for carrying an epoch (0) no live grid has.
  std::uint32_t count = 0;
  /// The @ref VoxelBlockGrid::topology_epoch the list was compacted at.
  /// Meaningless, and unchecked, when @ref count is 0.
  std::uint64_t epoch = 0;
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
