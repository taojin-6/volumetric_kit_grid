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
/// `layout(scalar)` definition in lockstep (see the gotchas in DESIGN.md).
///
/// The sparse-hashing scheme (a hash table of block coordinates into a heap of
/// fixed-size voxel blocks) keeps memory proportional to the observed surface
/// rather than the bounding volume. The prior engine's per-voxel neural
/// "feature"/triplane channels are intentionally absent here (see the exclusion
/// policy in AGENTS.md): this carries only SDF, weight, and optional color.

#include <cstddef>
#include <cstdint>

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

/// @return `true` if block coordinate @p a sorts before @p b: by z, then y,
///         then x. The one order every sorted block list here uses, the
///         codec's frames included; `vrCoordLess` in
///         `volume/shaders/coord_order.glsl` mirrors it for kernels.
inline bool coord_less(const Vec3i& a, const Vec3i& b) noexcept {
  if (a.z != b.z) return a.z < b.z;
  if (a.y != b.y) return a.y < b.y;
  return a.x < b.x;
}

/// @brief One block slot's stamps: ticks of the map's clock
///        (@ref VoxelHashMap::tick), each written by the pass that knows its
///        fact, so a consumer reads a block's age against a tick of its own
///        rather than a flag someone must reset.
///
/// Indexed by block slot, `BlockIndex::ptr / voxels_per_block`. A slot's
/// record is zeroed when its block is freed, and 0 is no tick. Mirrored by
/// `volume/shaders/block_stamp.glsl`.
struct BlockStamp {
  /// The last tick an allocation asked for the block, whether it inserted the
  /// block or found it there.
  std::uint32_t requested = 0;
  /// The last tick a @ref VoxelBlockGrid::free_stale_blocks pass found any of
  /// the block's voxels holding weight; 0 until one does.
  std::uint32_t weighted = 0;
  /// The last tick a pass that writes voxels changed any of the block's:
  /// `tsdf::TsdfIntegrator` and the codec's decoder where a value changed,
  /// `tsdf::MeshIntegrator` on every block it writes. 0 until one does.
  std::uint32_t changed = 0;
};
static_assert(sizeof(BlockStamp) == 12, "BlockStamp must be 12 bytes");
static_assert(offsetof(BlockStamp, weighted) == 4, "BlockStamp layout drift");
static_assert(offsetof(BlockStamp, changed) == 8, "BlockStamp layout drift");

/// @brief Whether tick @p a is later than tick @p b, modulo 2^32: right while
///        the two are under 2^31 ticks apart, across the clock's wrap too.
constexpr bool tick_after(std::uint32_t a, std::uint32_t b) noexcept {
  return static_cast<std::int32_t>(a - b) > 0;
}

}  // namespace volumetric_kit::recon::volume
