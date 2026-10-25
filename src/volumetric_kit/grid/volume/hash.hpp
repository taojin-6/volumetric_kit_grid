// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file volume/hash.hpp
/// @brief The device hash table's entry layout and fail-tally slots.
///
/// Internal (under src/, never installed): the map sizes its table by
/// @ref HashEntry and splits its retry tally by these slots, and its kernels
/// read both through `volume/shaders/hash_common.glsl`, which mirrors them.

#include <cstddef>
#include <cstdint>

#include "volumetric_kit/recon/core/math/vector_types.hpp"

namespace volumetric_kit::recon::volume {

/// Hash-table entry -- one per slot in the table. `hash_common.glsl`'s
/// `HashEntry` is its device half, byte-identical under scalar block layout
/// (the 2026-07-05 ABI; DESIGN.md's gotchas say why not `std430`).
struct HashEntry {
  std::int32_t ptr;     ///< A block pointer, or `hash_common.glsl`'s
                        ///< `kFreeEntry` / `kLockEntry`.
  Vec3i pos;            ///< Block coordinate (signed, 32-bit per axis).
  std::int32_t offset;  ///< Collision offset for linked-list chaining.
};
static_assert(sizeof(HashEntry) == 20, "HashEntry must be 20 bytes");
static_assert(offsetof(HashEntry, ptr) == 0, "HashEntry layout drift");
static_assert(offsetof(HashEntry, pos) == 4, "HashEntry layout drift");
static_assert(offsetof(HashEntry, offset) == 16, "HashEntry layout drift");

/// @name Fail-count slots
/// The layout of the per-dispatch failure tally the allocate and delete kernels
/// share, mirrored from `volume/shaders/hash_common.glsl`.
///
/// Every slot but @ref kFailTerminal is **retryable**: the host re-dispatches
/// while the total keeps dropping, because the element that failed is still
/// unprocessed and so restates itself next round. @ref kFailTerminal does not,
/// and the split is what keeps the reported count honest -- a delete whose heap
/// append fails has already cleared its table entry, so the next round finds
/// the coord absent and counts nothing. Accumulated across rounds rather than
/// read from the last one, or a permanently leaked block index would be
/// reported as a clean delete.
///
/// Retryable is not the same as *resolvable*: @ref kFailChain, @ref kFailHeap
/// and @ref kFailTable restate themselves honestly every round and are answered
/// by @ref VoxelHashMap::resize, not by another dispatch. Only @ref kFailLock
/// can clear on its own.
/// @{
inline constexpr std::uint32_t kFailTotal = 0;     ///< Retryable total.
inline constexpr std::uint32_t kFailLock = 1;      ///< Lock contention.
inline constexpr std::uint32_t kFailChain = 2;     ///< Collision chain full.
inline constexpr std::uint32_t kFailHeap = 3;      ///< Block heap (too) low.
inline constexpr std::uint32_t kFailTerminal = 4;  ///< Non-retryable.
/// No free non-anchor slot anywhere in the table: the overflow scan swept every
/// entry and every one it may use was occupied. Proof of exhaustion rather than
/// a heuristic -- the scan is deliberately uncapped, and one that skipped a
/// free-looking slot because another thread held its bucket reports
/// @ref kFailLock instead, so this reason never stands in for contention.
inline constexpr std::uint32_t kFailTable = 5;
inline constexpr std::uint32_t kFailSlots = 6;  ///< Slots in the tally.
/// @}

/// @brief What a recorded removal's or allocation's rounds leave in its
///        result slot: `hash_rounds.comp`'s `RoundsResult`, written whole by
///        its gather, and @ref VoxelHashMap::settle's input.
struct RoundsResult {
  std::uint32_t total;      ///< The last round's retryable failures.
  std::uint32_t lock;       ///< Of them, lost bucket-lock races.
  std::uint32_t chain;      ///< Collision chain full.
  std::uint32_t heap;       ///< Block heap (too) low.
  std::uint32_t table;      ///< No free non-anchor slot.
  std::uint32_t terminal;   ///< Non-retryable, over every round.
  std::uint32_t rounds;     ///< Rounds run.
  std::uint32_t heap_free;  ///< The heap counter after them.
};
static_assert(sizeof(RoundsResult) == 4 * 8,
              "RoundsResult is eight words, as hash_rounds.comp writes it");

}  // namespace volumetric_kit::recon::volume
