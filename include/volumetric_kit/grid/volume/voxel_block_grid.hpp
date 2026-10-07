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

#include "volumetric_kit/core/base/result.hpp"
#include "volumetric_kit/core/base/stage_metrics.hpp"
#include "volumetric_kit/core/vulkan/allocator.hpp"
#include "volumetric_kit/core/vulkan/buffer.hpp"
#include "volumetric_kit/core/vulkan/compute_kernel.hpp"
#include "volumetric_kit/core/vulkan/descriptor.hpp"
#include "volumetric_kit/core/vulkan/gpu_timer.hpp"
#include "volumetric_kit/core/vulkan/vulkan.hpp"
#include "volumetric_kit/recon/core/fwd.hpp"
#include "volumetric_kit/recon/volume/export.hpp"
#include "volumetric_kit/recon/volume/voxel_grid.hpp"
#include "volumetric_kit/recon/volume/voxel_hash_map.hpp"

namespace volumetric_kit::recon::volume {

/// @brief Declares one per-voxel attribute array to allocate: a name and the
///        byte size of its per-voxel element.
///
/// The element size folds dtype and channel count into a single stride (a
/// 32-bit float SDF is 4, an 8-bit RGB colour is 3), so attributes of
/// different shapes are each allocated as their own device buffer
/// (structure-of-arrays), not interleaved.
struct AttributeSpec {
  std::string_view name;       ///< Attribute name, e.g. "tsdf", "weight".
  std::uint32_t element_size;  ///< Bytes per voxel (e.g. 4 for a float).
};

/// @brief Non-owning view of one attribute's backing store.
///
/// @ref buffer is the device-local buffer to bind to a compute kernel (the
/// host reaches it through a `CommandBatch`); it holds
/// @ref element_count voxels of @ref element_size bytes each. Re-fetch the view
/// (do not cache @ref buffer or its handle) across a move **or a
/// @ref VoxelBlockGrid::resize** of the owning grid -- resize replaces every
/// attribute buffer, so a held view or handle dangles.
struct AttributeView {
  const core::Buffer* buffer = nullptr;  ///< The attribute's device buffer.
  std::uint32_t element_size = 0;        ///< Bytes per voxel.
  std::uint64_t element_count = 0;  ///< Voxels (num_blocks * voxels_per_block).
};

/// @brief The `weight` at or above which a voxel counts as observed.
///
/// Small and positive, so a never-integrated voxel (weight 0) is excluded while
/// any genuine integration counts. Defined here, the one tier both readers
/// link, because the two must agree: the mesher skips any cell with a corner
/// below it, and the codec's observed mask marks exactly the voxels at or above
/// it -- a mask that disagreed would decode holes, or voxels the mesher would
/// have skipped.
inline constexpr float kObservedWeight = 1e-6f;

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
/// @warning The `Device` and `Allocator` passed to @ref create must
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
  /// @return The grid, or a non-OK `Status`: whatever @ref
  ///         VoxelHashMap::create returns; `Status::Code::InvalidArgument`
  ///         for a null list, an empty name, a zero element size, an
  ///         attribute whose block is not whole 4-byte words
  ///         (`voxels_per_block * element_size` not a multiple of 4), or a
  ///         duplicate name; or an allocation failure.
  static core::Result<VoxelBlockGrid> create(core::Device& device,
                                             core::Allocator& allocator,
                                             const VoxelGridParams& grid,
                                             const AttributeSpec* attrs,
                                             std::size_t attr_count);

  // Move construction only. The owned VoxelHashMap and each attribute Buffer
  // self-reset on move, so a moved-from grid is empty (valid() == false). A
  // defaulted move assignment would self-move `attributes_`, and std::vector
  // self-move frees every attribute buffer while the map survives; a
  // hand-written one must name every member, and once missed one. Replace a
  // grid with std::optional::emplace instead.
  ~VoxelBlockGrid() = default;
  VoxelBlockGrid(VoxelBlockGrid&&) noexcept = default;
  VoxelBlockGrid& operator=(VoxelBlockGrid&&) = delete;
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
  ///            (see there), so the desync surfaces as a clean `Status` at
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

  /// @brief Anchor a compacted block list to this grid, ready to pass to a
  ///        consumer that runs over a subset.
  ///
  /// The pairing @ref BlockList documents as a discipline, made a call instead:
  /// the pointer, the count and the epoch all come from one expression, so the
  /// epoch cannot be fetched from a different grid, read a frame early, or
  /// simply forgotten -- and a forgotten one is not a compile error, since
  /// @ref BlockList::epoch has a default that no live grid ever equals.
  ///
  /// @warning Borrowing, not owning: @p blocks must outlive the returned list
  ///          and must not be reallocated under it. Call this beside the
  ///          consumer rather than caching the result.
  /// @param blocks  A compacted active set, typically straight from
  ///                @ref VoxelHashMap::compact_active_blocks.
  /// @return The list, stamped with this grid's current @ref topology_epoch.
  BlockList block_list(const std::vector<BlockIndex>& blocks) const noexcept {
    return BlockList{blocks.data(), static_cast<std::uint32_t>(blocks.size()),
                     topology_epoch()};
  }
  /// Deleted: the list would outlive the vector it borrows, so
  /// `block_list(map().compact_active_blocks().value())` dangles by the next
  /// statement. Name the compaction first, then anchor it.
  BlockList block_list(std::vector<BlockIndex>&&) const = delete;

  /// @brief Check a caller-supplied @ref BlockList against this grid, for a
  ///        consumer about to upload it and index attribute storage with it.
  ///
  /// Refuses a list that is null with a count; one holding more blocks than the
  /// heap, which the epoch is blind to, since a vector re-compacted shorter
  /// under a cached list keeps its old count through every allocate and
  /// resize, and the upload would read past the caller's array; and one
  /// compacted at another @ref topology_epoch, whose in-range ptrs a
  /// `remove()` / `clear()` has handed to different blocks. An empty list names
  /// no block, so it passes whatever its epoch: that is what lets a
  /// default-constructed `BlockList{}` mean "nothing".
  ///
  /// Not checked: that each entry is a block this grid handed out. That is
  /// O(count) per call, and every consumer answers it on the device instead.
  /// @param blocks  The list.
  /// @param who     Prefixes the error message (the consumer's entry point).
  /// @return OK, or `Status::Code::InvalidArgument` naming the refusal.
  core::Status check_block_list(const BlockList& blocks, const char* who) const;

  /// @brief Look up an attribute's backing store by name.
  ///
  /// Also the one place the attribute arrays are checked against the live grid,
  /// because every consumer that binds one passes through here. An array that
  /// no longer covers `num_blocks * voxels_per_block` -- which is what calling
  /// @ref VoxelHashMap::resize through @ref map() leaves behind -- is refused,
  /// so the mismatch becomes a clean `Status` at the binding site rather
  /// than a kernel indexing past the end of a buffer bound `VK_WHOLE_SIZE`.
  ///
  /// A write through the view stamps no block `changed` (@ref BlockStamp),
  /// unlike the library's own writers, so a consumer of the stamps does not
  /// see it.
  /// @param name  The attribute name (as declared at @ref create).
  /// @return A view of the attribute, or `Status::Code::InvalidArgument` if
  ///         no attribute of that name was declared (or the grid is
  ///         moved-from), or if the array no longer covers the live grid.
  core::Result<AttributeView> attribute(std::string_view name) const;

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
  /// Each coord is found in the hash table on the device, by the kernel that
  /// zeroes its block, so the cost is the count's, not the grid's, and a
  /// coord that is not currently allocated costs nothing and clears nothing.
  /// The blocks are zeroed before the
  /// map's remove runs, so a block that call leaves in the table
  /// (@ref AllocFailures::lock) is left zeroed, reading as a freshly
  /// allocated block until a later call removes it.
  /// @param coords  The block coordinates to remove (only `coord` is read).
  /// @param count   How many.
  /// @param out_failures  Optional: forwarded to @ref VoxelHashMap::remove.
  /// @return What @ref VoxelHashMap::remove returns, or a non-OK `Status` if
  ///         the grid is moved-from, @p coords is null, or the zeroing fails,
  ///         which comes before any index is freed.
  core::Result<std::uint32_t> remove(const BlockIndex* coords,
                                     std::uint32_t count,
                                     AllocFailures* out_failures = nullptr);

  /// @brief Empty the grid: clear the block index and zero every attribute.
  ///
  /// @ref VoxelHashMap::clear returns every block to the heap, so every
  /// attribute range is about to be re-drawn; zeroing them here is what keeps
  /// "a freshly allocated block reads as zero" true after a clear, exactly as
  /// it is after @ref create. The zeroing runs first, so a failure never frees
  /// an index over stale data.
  /// @return OK, or a non-OK `Status` if the grid is moved-from, the
  ///         zeroing fails, or the underlying @ref VoxelHashMap::clear fails.
  core::Status clear();

  /// @return `true` if an attribute of @p name was declared.
  bool has_attribute(std::string_view name) const noexcept;

  /// @return How many attributes were declared at @ref create (0 when
  ///         moved-from) -- what lets a consumer that writes only some of them
  ///         refuse a grid whose others it would leave stale.
  std::size_t attribute_count() const noexcept { return attributes_.size(); }

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
  /// @return OK on success, or a non-OK `Status`: @ref
  ///         Status::Code::InvalidArgument for a moved-from grid or a
  ///         non-growing count; an allocation failure; or whatever @ref
  ///         VoxelHashMap::resize returns (e.g. `Status::Code::OutOfMemory`
  ///         on a rehash overflow).
  core::Status resize(std::int32_t new_num_buckets);

  /// @return `true` if this owns a live grid (`false` when moved-from).
  bool valid() const noexcept { return map_.valid(); }

  /// @brief Stamp `weighted` with the map's tick (@ref VoxelHashMap::tick) on
  ///        every active block holding an observed voxel (`weight >=
  ///        kObservedWeight`, as the mesher and the codec read it), then free
  ///        every active block that has been
  ///        neither asked for by an allocation nor found holding weight for
  ///        more than @p max_age ticks (@ref remove).
  ///
  /// So a block the allocator still asks for stays, holding weight or not --
  /// the band around the surface the cameras see -- and so does one still
  /// holding weight, seen or not; what goes is space a surface has left.
  /// *More* than: the fuse after an allocation advances the clock, so a block
  /// the last set asked for is a tick old once that set is fused. A rig that
  /// fuses a set a tick therefore frees a block @p max_age sets after it was
  /// last asked for. Freeing moves the map's topology epoch, as any
  /// @ref remove does.
  /// Ages are differences of ticks, so they hold across the clock's wrap.
  /// @param max_age  Ticks a block may be neither and stay; at least 1.
  /// The stamping is one workgroup per active block, reading its `weight`
  /// attribute, so this is meant for every few ticks rather than every fuse.
  /// Its kernels are built on the first call.
  /// @param metrics  Optional rows: a `"block stamps"` row with both halves,
  ///                 spanning the frees too, over the compaction's
  ///                 `"  ..active set"`.
  /// @return The blocks freed; `Status::Code::InvalidArgument` for a
  ///         @p max_age of 0, a moved-from grid or one without a `float`
  ///         `weight` attribute; or what @ref VoxelHashMap::remove refuses.
  core::Result<std::uint32_t> free_stale_blocks(
      std::uint32_t max_age, core::StageMetrics* metrics = nullptr);

 private:
  /// Construct from an already-built block index + the allocator its attribute
  /// buffers come from (borrowed; must outlive the grid). Attributes are added
  /// by @ref create. (VoxelHashMap has no public default ctor, so the grid is
  /// built map-first rather than default-then-assign.)
  VoxelBlockGrid(VoxelHashMap map, core::Device* device,
                 core::Allocator* allocator,
                 VkDeviceSize max_storage_buffer_range)
      : map_(std::move(map)),
        max_storage_buffer_range_(max_storage_buffer_range),
        device_(device),
        allocator_(allocator) {}

  /// Attach each attribute array's declared name to its buffer, so a GPU
  /// capture shows "tsdf" rather than a raw handle.
  ///
  /// Called by @ref create and again by @ref resize, which replaces every
  /// attribute buffer -- a name lives on the handle, so a grown array is
  /// anonymous until it is renamed. A no-op where the device resolved no
  /// debug-utils entry points.
  void name_attribute_buffers() const noexcept;

  /// One named attribute array: its declared name + element size + the buffer.
  struct Attribute {
    std::string name;
    std::uint32_t element_size = 0;
    core::Buffer buffer;
  };

  // Build the block pass's kernels, timer and count on its first call, so a
  // grid that never runs one never pays for them; the pass also checks for a
  // float weight, which the zeroing alone does not need.
  core::Status prepare_block_pass();
  core::Status prepare_block_kernels();
  // Record the zero kernel over the first `count` entries of `list`, an
  // attribute a dispatch, each block found by its coord.
  core::Status record_zero(core::CommandBatch& batch, const core::Buffer& list,
                           std::uint32_t count, core::GpuStageScope* stage);
  // Upload `coords` and zero their blocks, in one batch.
  core::Status zero_listed_blocks(const BlockIndex* coords,
                                  std::uint32_t count);
  // The block pass: stamps, and with a max_age the stale blocks listed in
  // stale_list_. Returns how many.
  core::Result<std::uint32_t> block_pass(std::uint32_t max_age,
                                         core::GpuStageScope& stage,
                                         core::StageMetrics* metrics);

  VoxelHashMap map_;
  std::vector<Attribute> attributes_;
  // The block pass's kernels (their pool first, so it outlives the sets): the
  // stamp pass, and the zeroing of the stale blocks, a set an attribute. Then
  // its device timer, and its stale list and count; the list grows to the
  // most active blocks a pass has seen.
  core::DescriptorPool block_pool_;
  core::ComputeKernel stamp_kernel_;
  core::ComputeKernel zero_kernel_;
  core::KernelSets zero_sets_;
  core::GpuTimer gpu_timer_;
  core::Buffer stale_list_;
  core::Buffer stale_count_;
  std::uint32_t max_workgroup_count_x_ = 0;
  // The device's maxStorageBufferRange, read once at create(). An attribute
  // array is the largest buffer this repo allocates and is bound whole, so it
  // is the one most likely to exceed what a single binding may cover -- at the
  // examples' defaults it is already 2x Vulkan's guaranteed minimum.
  VkDeviceSize max_storage_buffer_range_ = 0;
  // Borrowed (must outlive this), and held only to name the attribute buffers
  // for a GPU capture -- the grid dispatches through map_, which carries its
  // own. Like allocator_, a moved-from grid keeps the pointer but reports
  // valid() == false through map_, so it is never dereferenced.
  core::Device* device_ = nullptr;
  // Borrowed (must outlive this): backs every attribute buffer, including those
  // grown by resize(). A moved-from grid keeps the pointer but reports
  // valid() == false through map_, so it is never dereferenced.
  core::Allocator* allocator_ = nullptr;
};

}  // namespace volumetric_kit::recon::volume
