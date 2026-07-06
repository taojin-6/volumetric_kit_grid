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
  ~VoxelBlockGrid() = default;
  VoxelBlockGrid(VoxelBlockGrid&&) noexcept = default;
  VoxelBlockGrid& operator=(VoxelBlockGrid&& other) noexcept {
    if (this != &other) {
      map_ = std::move(other.map_);
      attributes_ = std::move(other.attributes_);
      allocator_ = other.allocator_;
    }
    return *this;
  }
  VoxelBlockGrid(const VoxelBlockGrid&) = delete;
  VoxelBlockGrid& operator=(const VoxelBlockGrid&) = delete;

  /// @brief The composed block index, for allocation / compaction.
  ///
  /// @warning To grow a grid that carries attributes, call @ref resize, which
  ///          grows the attribute arrays *and* rehashes the map preserving
  ///          block indices. Calling @ref VoxelHashMap::resize through this
  ///          handle grows the table (preserving indices) but leaves the
  ///          attribute arrays at their old `num_blocks * voxels_per_block`
  ///          size, so a block allocated into the grown capacity would address
  ///          past them.
  ///          @ref attribute reports the buffer-derived capacity (not the live
  ///          grid), so a downstream bounds check stays sound.
  /// @return The block index.
  VoxelHashMap& map() noexcept { return map_; }
  /// @overload
  const VoxelHashMap& map() const noexcept { return map_; }

  /// @return The grid + hash-table parameters this grid was built with.
  const VoxelGridParams& grid() const noexcept { return map_.grid(); }

  /// @brief Look up an attribute's backing store by name.
  /// @param name  The attribute name (as declared at @ref create).
  /// @return A view of the attribute, or @ref Status::Code::InvalidArgument if
  ///         no attribute of that name was declared (or the grid is
  ///         moved-from).
  Result<AttributeView> attribute(std::string_view name) const;

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
  VoxelBlockGrid(VoxelHashMap map, Allocator* allocator)
      : map_(std::move(map)), allocator_(allocator) {}

  /// One named attribute array: its declared name + element size + the buffer.
  struct Attribute {
    std::string name;
    std::uint32_t element_size = 0;
    Buffer buffer;
  };

  VoxelHashMap map_;
  std::vector<Attribute> attributes_;
  // Borrowed (must outlive this): backs every attribute buffer, including those
  // grown by resize(). A moved-from grid keeps the pointer but reports
  // valid() == false through map_, so it is never dereferenced.
  Allocator* allocator_ = nullptr;
};

}  // namespace volumetric_kit::recon::volume
