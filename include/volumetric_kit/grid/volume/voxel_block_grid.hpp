// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file volume/voxel_block_grid.hpp
/// @brief Sparse voxel block grid: a @ref VoxelHashMap block index plus a set
/// of
///        independently-allocated, named per-voxel attribute arrays (SoA).

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "volumetric_kit/recon/core/allocator.hpp"
#include "volumetric_kit/recon/core/buffer.hpp"
#include "volumetric_kit/recon/core/result.hpp"
#include "volumetric_kit/recon/core/vulkan.hpp"
#include "volumetric_kit/recon/volume/export.hpp"
#include "volumetric_kit/recon/volume/voxel_grid.hpp"
#include "volumetric_kit/recon/volume/voxel_hash_map.hpp"

namespace volumetric_kit::recon {
class Device;
}

namespace volumetric_kit::recon::volume {

/// @brief Declares one per-voxel attribute array to allocate: a name and the
///        byte size of its per-voxel element.
///
/// The element size folds dtype and channel count into a single stride (a
/// 32-bit float SDF is 4, an RGB @ref Vec3u8 colour is 3), so attributes of
/// different shapes are each allocated as their own device buffer
/// (structure-of-arrays), not interleaved.
struct AttributeSpec {
  std::string_view name;       ///< Attribute name, e.g. "tsdf", "weight".
  std::uint32_t element_size;  ///< Bytes per voxel (e.g. 4 for a float).
};

/// @brief Non-owning view of one attribute's backing store.
///
/// @ref buffer is the device buffer to bind to a compute kernel (or read/write
/// through @ref Buffer::mapped for this host-visible slice); it holds
/// @ref element_count voxels of @ref element_size bytes each. Re-fetch the view
/// (do not cache @ref buffer or its handle) across a move **or a
/// @ref VoxelBlockGrid::resize** of the owning grid -- resize replaces every
/// attribute buffer, so a held view or handle dangles.
struct AttributeView {
  const Buffer* buffer = nullptr;   ///< The attribute's device buffer.
  std::uint32_t element_size = 0;   ///< Bytes per voxel.
  std::uint64_t element_count = 0;  ///< Voxels (num_blocks * voxels_per_block).
};

/// @brief A sparse voxel block grid: the @ref VoxelHashMap block index plus a
///        set of named, independently-allocated per-voxel attribute arrays.
///
/// Mirrors Open3D's `VoxelBlockGrid`: the hash map keys a block *index*, and
/// each attribute ("tsdf", "weight", "color", ...) is its own flat array of
/// `num_blocks * voxels_per_block` elements, addressed by the same
/// `BlockIndex::ptr + local` the allocate kernels compute. Attributes are
/// declared up front and allocated independently (structure-of-arrays), so a
/// consumer materialises only the channels it needs -- a grid with no
/// attributes costs no per-voxel memory. The TSDF / colour integrators and mesh
/// extraction bind the attribute buffers they read or write.
///
/// @warning The @ref Device and @ref Allocator passed to @ref create must
///          outlive this object; it stores references to them (through the
///          owned
///          @ref VoxelHashMap, and the allocator backs every attribute buffer).
class VR_VOLUME_API VoxelBlockGrid {
 public:
  /// @brief Create the grid: build the block index, then allocate and zero each
  ///        declared attribute array.
  /// @param device     The compute device (must outlive this object).
  /// @param allocator  The allocator its buffers come from (must outlive this).
  /// @param grid       The grid resolution + hash-table shape.
  /// @param attrs      The attributes to allocate (may be null iff @p
  /// attr_count
  ///                   is 0).
  /// @param attr_count How many @p attrs.
  /// @return The grid, or a non-OK @ref Status: whatever @ref
  ///         VoxelHashMap::create returns; @ref Status::Code::InvalidArgument
  ///         for a null list, an empty name, a zero element size, or a
  ///         duplicate name; or an allocation failure.
  static Result<VoxelBlockGrid> create(Device& device, Allocator& allocator,
                                       const VoxelGridParams& grid,
                                       const AttributeSpec* attrs,
                                       std::size_t attr_count);

  // The owned VoxelHashMap and each attribute Buffer self-reset on move, so the
  // destructor and move-construct are defaulted and a moved-from grid is left
  // empty (valid() == false). Move-assignment is hand-written for the self-move
  // guard: a defaulted memberwise version would run
  // `attributes_ = std::move(attributes_)`, and std::vector
  // self-move-assignment frees its storage -- destroying every attribute Buffer
  // while map_ (whose members self-guard) survives, leaving valid() == true
  // with the attributes gone. The guard below keeps a self-assigned grid
  // intact.
  //
  // Being hand-written, it must name EVERY member, and a forgotten one is
  // silent: the topology epoch used to be a member here and was not assigned,
  // so a move-assigned grid took on another grid's blocks while still reporting
  // the destination's old epoch -- and every slot-keyed cache anchored to it
  // stayed "valid" across the swap. It now lives in map_ (see topology_epoch),
  // which is moved, so that particular member cannot be dropped again; the
  // obligation for anything added below still stands.
  ~VoxelBlockGrid() = default;
  VoxelBlockGrid(VoxelBlockGrid&&) noexcept = default;
  VoxelBlockGrid& operator=(VoxelBlockGrid&& other) noexcept {
    if (this != &other) {
      map_ = std::move(other.map_);
      attributes_ = std::move(other.attributes_);
      max_storage_buffer_range_ = other.max_storage_buffer_range_;
      allocator_ = other.allocator_;
    }
    return *this;
  }
  VoxelBlockGrid(const VoxelBlockGrid&) = delete;
  VoxelBlockGrid& operator=(const VoxelBlockGrid&) = delete;

  /// @brief The composed block index, for allocation / compaction.
  ///
  /// @warning Two of this handle's operations invalidate state only this class
  ///          can keep consistent, so prefer the wrappers here:
  ///          - @ref VoxelHashMap::resize grows the table (preserving block
  ///            indices) but leaves the attribute arrays at their old
  ///            `num_blocks * voxels_per_block`, so a block allocated into the
  ///            grown capacity addresses past them. Use @ref resize, which
  ///            grows both. Reaching it through this handle anyway is caught
  ///            rather than silently tolerated: @ref attribute then **refuses**
  ///            (see there), so the desync surfaces as a clean @ref Status at
  ///            the next bind instead of an out-of-bounds device write.
  ///          - @ref VoxelHashMap::remove frees a block index without clearing
  ///            the per-voxel data it addressed, and the free heap is LIFO, so
  ///            the next allocation resurrects it. Use @ref remove. This one
  ///            cannot be detected after the fact -- the stale data is
  ///            indistinguishable from fused data -- which is why it is
  ///            wrapped rather than checked. It does move @ref topology_epoch,
  ///            though, so a slot-keyed *cache* is invalidated either way; it
  ///            is the per-voxel attribute data, and only that, which the
  ///            wrapper exists to clear.
  /// @return The block index.
  VoxelHashMap& map() noexcept { return map_; }
  /// @overload
  const VoxelHashMap& map() const noexcept { return map_; }

  /// @return The grid + hash-table parameters this grid was built with.
  const VoxelGridParams& grid() const noexcept { return map_.grid(); }

  /// @brief A token identifying this grid's current block-index assignment; it
  ///        changes whenever a block stops being live.
  ///
  /// @ref VoxelHashMap::topology_epoch, which is where it lives and where the
  /// contract is stated. Forwarded rather than duplicated so that the raw path
  /// cannot dodge it: @ref VoxelHashMap::remove and @ref VoxelHashMap::clear
  /// reached through @ref map() move this exactly as the wrappers here do,
  /// where a counter owned by this class was moved only by the wrappers and
  /// left every anchor built on it defeatable in silence.
  ///
  /// @ref resize deliberately does **not** move it: it preserves every block's
  /// index, so a slot-keyed cache stays correct across a grow.
  /// @return The token; compare for equality only -- it counts nothing.
  std::uint64_t topology_epoch() const noexcept {
    return map_.topology_epoch();
  }

  /// @brief Look up an attribute's backing store by name.
  ///
  /// Also the one place the attribute arrays are checked against the live grid,
  /// because every consumer that binds one passes through here. An array that
  /// no longer covers `num_blocks * voxels_per_block` -- which is what calling
  /// @ref VoxelHashMap::resize through @ref map() leaves behind -- is refused,
  /// so the mismatch becomes a clean @ref Status at the binding site rather
  /// than a kernel indexing past the end of a buffer bound `VK_WHOLE_SIZE`.
  /// @param name  The attribute name (as declared at @ref create).
  /// @return A view of the attribute, or @ref Status::Code::InvalidArgument if
  ///         no attribute of that name was declared (or the grid is
  ///         moved-from), or if the array no longer covers the live grid.
  Result<AttributeView> attribute(std::string_view name) const;

  /// @brief Remove voxel blocks at the given block coordinates, clearing the
  ///        per-voxel attribute data they held.
  ///
  /// @ref VoxelHashMap::remove with the half it cannot do: the block index does
  /// not know what attributes a consumer declared, so it returns a freed index
  /// to the heap with its attribute range untouched. The heap is LIFO, so the
  /// next allocation re-draws that index onto the same range and the removed
  /// surface's `tsdf` / `weight` / `color` resurrect under the new geometry --
  /// at full fused weight, which also bypasses the integrator's
  /// first-observation-assigns colour gate. This zeroes each removed block's
  /// range first, matching the state @ref create leaves a fresh array in.
  ///
  /// The ranges are resolved from a snapshot of the active set, so a coord that
  /// is not currently allocated costs nothing and clears nothing.
  /// @param coords  The block coordinates to remove (only `coord` is read).
  /// @param count   How many.
  /// @param out_failures  Optional: forwarded to @ref VoxelHashMap::remove.
  /// @return What @ref VoxelHashMap::remove returns, or a non-OK @ref Status if
  ///         the grid is moved-from, @p coords is null, or the snapshot fails.
  Result<std::uint32_t> remove(const BlockIndex* coords, std::uint32_t count,
                               AllocFailures* out_failures = nullptr);

  /// @brief Empty the grid: clear the block index and zero every attribute.
  ///
  /// @ref VoxelHashMap::clear returns every block to the heap, so every
  /// attribute range is about to be re-drawn; zeroing them here is what keeps
  /// "a freshly allocated block reads as zero" true after a clear, exactly as
  /// it is after @ref create.
  /// @return OK, or a non-OK @ref Status if the grid is moved-from or the
  ///         underlying @ref VoxelHashMap::clear fails.
  Status clear();

  /// @return `true` if an attribute of @p name was declared.
  bool has_attribute(std::string_view name) const noexcept;

  /// @brief Grow the grid to @p new_num_buckets buckets, preserving every
  ///        block's per-voxel attribute data.
  ///
  /// Grows each attribute array to the new `num_blocks * voxels_per_block` (new
  /// capacity zero-filled, existing contents copied), then rehashes the map
  /// with
  /// @ref VoxelHashMap::resize, which **preserves each block's index** -- so
  /// the data a block held (addressed by @ref BlockIndex::ptr) stays valid at
  /// the same offset. All-or-nothing: the enlarged buffers are built and filled
  /// before the map resize and committed only once it succeeds, so an
  /// allocation failure leaves the grid untouched.
  /// @param new_num_buckets  The new bucket count (> the current @ref grid).
  /// @return OK on success, or a non-OK @ref Status: @ref
  ///         Status::Code::InvalidArgument for a moved-from grid or a
  ///         non-growing count; an allocation failure; or whatever @ref
  ///         VoxelHashMap::resize returns (e.g. @ref Status::Code::OutOfMemory
  ///         on a rehash overflow).
  Status resize(std::int32_t new_num_buckets);

  /// @return `true` if this owns a live grid (`false` when moved-from).
  bool valid() const noexcept { return map_.valid(); }

 private:
  /// Construct from an already-built block index + the allocator its attribute
  /// buffers come from (borrowed; must outlive the grid). Attributes are added
  /// by @ref create. (VoxelHashMap has no public default ctor, so the grid is
  /// built map-first rather than default-then-assign.)
  VoxelBlockGrid(VoxelHashMap map, Allocator* allocator,
                 VkDeviceSize max_storage_buffer_range)
      : map_(std::move(map)),
        max_storage_buffer_range_(max_storage_buffer_range),
        allocator_(allocator) {}

  /// One named attribute array: its declared name + element size + the buffer.
  struct Attribute {
    std::string name;
    std::uint32_t element_size = 0;
    Buffer buffer;
  };

  VoxelHashMap map_;
  std::vector<Attribute> attributes_;
  // The device's maxStorageBufferRange, read once at create(). An attribute
  // array is the largest buffer this repo allocates and is bound whole, so it
  // is the one most likely to exceed what a single binding may cover -- at the
  // examples' defaults it is already 2x Vulkan's guaranteed minimum.
  VkDeviceSize max_storage_buffer_range_ = 0;
  // Borrowed (must outlive this): backs every attribute buffer, including those
  // grown by resize(). A moved-from grid keeps the pointer but reports
  // valid() == false through map_, so it is never dereferenced.
  Allocator* allocator_ = nullptr;
};

}  // namespace volumetric_kit::recon::volume
