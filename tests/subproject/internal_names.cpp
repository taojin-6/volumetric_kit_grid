// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// recon_subproject_headers compiles this in the sibling that fetches recon and
// never runs it. The hash table's internals stay out of the public headers:
// volume/hash.hpp and volume/hash_map_internal.hpp are not on the include path
// recon's targets give a consumer; every public volume header, and every
// public recon header that includes one, compiles without them; none of them
// declares one of the names below in the volume namespace; and VoxelHashMap
// hands out no hash-entry array (its Internal is defined only in
// hash_map_internal.hpp).

#if __has_include("volumetric_kit/recon/volume/hash.hpp")
#error "volume/hash.hpp is public, but the hash table's layout is internal"
#endif
#if __has_include("volumetric_kit/recon/volume/hash_map_internal.hpp")
#error "volume/hash_map_internal.hpp is public, but the entries are internal"
#endif

#include <type_traits>
#include <utility>

// Those headers, as subproject/CMakeLists.txt finds them in recon's include/.
#include "vr_volume_headers.hpp"

// An alias is ill-formed after any other declaration of its name: a class or
// its forward declaration, a variable, a function or another alias.
namespace volumetric_kit::recon::volume {
struct InternalNameProbe {};
using HashEntry = InternalNameProbe;
using hash_bucket = InternalNameProbe;
using kFreeEntry = InternalNameProbe;
using kLockEntry = InternalNameProbe;
using kNoOffset = InternalNameProbe;
using kHashPrimeX = InternalNameProbe;
using kHashPrimeY = InternalNameProbe;
using kHashPrimeZ = InternalNameProbe;
using kFailTotal = InternalNameProbe;
using kFailLock = InternalNameProbe;
using kFailChain = InternalNameProbe;
using kFailHeap = InternalNameProbe;
using kFailTerminal = InternalNameProbe;
using kFailTable = InternalNameProbe;
using kFailSlots = InternalNameProbe;
using entries_buffer = InternalNameProbe;
using entries_buffer_size = InternalNameProbe;

// A consumer can name neither VoxelHashMap::entries_buffer() nor
// entries_buffer_size(), and VoxelHashMap::Internal is incomplete.
template <typename T, typename = void>
struct HasEntriesBuffer : std::false_type {};
template <typename T>
struct HasEntriesBuffer<
    T, std::void_t<decltype(std::declval<const T&>().entries_buffer())>>
    : std::true_type {};
template <typename T, typename = void>
struct HasEntriesBufferSize : std::false_type {};
template <typename T>
struct HasEntriesBufferSize<
    T, std::void_t<decltype(std::declval<const T&>().entries_buffer_size())>>
    : std::true_type {};
template <typename T, typename = void>
struct IsComplete : std::false_type {};
template <typename T>
struct IsComplete<T, std::void_t<decltype(sizeof(T))>> : std::true_type {};
static_assert(!HasEntriesBuffer<VoxelHashMap>::value,
              "VoxelHashMap::entries_buffer() is public");
static_assert(!HasEntriesBufferSize<VoxelHashMap>::value,
              "VoxelHashMap::entries_buffer_size() is public");
static_assert(!IsComplete<VoxelHashMap::Internal>::value,
              "VoxelHashMap::Internal is defined in a public header");
}  // namespace volumetric_kit::recon::volume
