// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file mesh/mesh_exchange.hpp
/// @brief The handoff of a `MarchingCubes` ring's meshes from the thread that
///        extracts them to the thread that draws them.

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <optional>
#include <utility>
#include <vector>

#include "volumetric_kit/core/base/check.hpp"
#include "volumetric_kit/core/vulkan/vulkan.hpp"
#include "volumetric_kit/recon/mesh/device_mesh.hpp"

namespace volumetric_kit::recon::mesh {

/// @brief How the consumer of a @ref MeshExchange reads a mesh: the frames it
///        keeps in flight, and what it binds the buffers as.
struct MeshExchangeConfig {
  /// Frames the consumer keeps in flight, each with a slot of its own in
  /// @ref MeshExchange::begin_frame. The producer's ring needs one slot more
  /// (`MarchingCubesConfig::slot_count`).
  std::uint32_t frames_in_flight = 2;
  /// Whether the consumer reads the buffers from another queue family than
  /// the producer's. Reading an EXCLUSIVE buffer from a family that does not
  /// own it is undefined, so such a consumer needs
  /// `VK_SHARING_MODE_CONCURRENT`.
  bool cross_family = false;
  /// The usage each buffer must carry. The defaults are an indexed indirect
  /// draw's.
  VkBufferUsageFlags vertex_usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
  /// See @ref vertex_usage.
  VkBufferUsageFlags index_usage = VK_BUFFER_USAGE_INDEX_BUFFER_BIT;
  /// See @ref vertex_usage.
  VkBufferUsageFlags indirect_usage = VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT;
};

/// @brief Why @p mesh cannot be bound as @p config asks, or null if it can.
///
/// Checked rather than assumed: Vulkan cannot be asked what a `VkBuffer` was
/// created with, so the mesh carries its usage and sharing mode, and binding a
/// buffer without the usage is undefined with the validation layers off.
///
/// Not for an empty mesh: an extract that meshed nothing may name null
/// buffers, and draws nothing.
inline const char* unbindable_reason(
    const DeviceMesh& mesh, const MeshExchangeConfig& config) noexcept {
  if (config.cross_family && mesh.sharing_mode != VK_SHARING_MODE_CONCURRENT) {
    return "its buffers are EXCLUSIVE to the producer's queue family, and the "
           "consumer reads them from another";
  }
  if (!mesh.valid()) return "a buffer handle is null";
  if ((mesh.vertex_usage & config.vertex_usage) != config.vertex_usage ||
      (mesh.index_usage & config.index_usage) != config.index_usage ||
      (mesh.indirect_usage & config.indirect_usage) != config.indirect_usage) {
    return "a buffer lacks the usage the consumer binds it with";
  }
  return nullptr;
}

/// @brief What @ref MeshExchange::begin_frame did with the mesh it took.
enum class ExchangeOutcome {
  kNone,       ///< Nothing was taken or parked: the live mesh is unchanged.
  kCommitted,  ///< A mesh became the live one (an empty one included).
  kParked,     ///< The consumer could not commit it yet; retried next frame.
  /// It cannot be bound (@ref MeshExchange::refused says why). Latched: no
  /// later mesh is taken.
  kRefused,
};

/// @brief The handoff of `MarchingCubes::extract_device`'s meshes, each with a
///        @p Payload that must be drawn with it, from the thread that
///        extracts to a renderer drawing them in place.
///
/// A ring of `MarchingCubesConfig::slot_count` slots lets the renderer draw
/// one generation while the next is extracted, and the renderer says, through
/// `MarchingCubes::release_through`, which generations its frames have
/// finished reading. That mark is a single monotone value, which makes the
/// handoff's rules:
///
/// - **The release is computed as a frame retires, then the newest mesh is
///   taken, under one lock.** Taking frees the producer to extract again, so
///   a mark computed after the take lets it extract on a stale mark and be
///   refused a slot.
/// - **What is released is everything below the oldest generation a frame in
///   flight or the live mesh holds**, read as a set: one generation is drawn
///   by several consecutive frames, and an extract may free the buffers of
///   any slot the mark covers. Before anything is committed, everything taken
///   is released, so takes that are never drawn do not drain the ring.
/// - **A taken mesh the consumer cannot commit yet is parked**, retried every
///   frame, and covered by no mark until it is committed. A newer mesh is not
///   taken meanwhile, so at most one is held uncommitted.
/// - **An empty mesh is committed as "draw nothing"**, without the bindable
///   check: it holds a slot like any other, and may name null buffers.
/// - **A mesh that cannot be bound latches** the exchange: nothing more is
///   taken, so the producer stops extracting rather than walking the ring
///   through undrawable generations.
/// - **The producer applies the release, on its own thread**, before the
///   extract it makes room for: `MarchingCubes::release_through` is not
///   synchronized against the extracting thread.
/// - **The producer publishes every extract** and does not publish over a
///   mesh not yet taken, except to supersede it.
///
/// The payload is what the mesh's `Vertex::uv0` index into, such as the
/// pixels of the camera that textured it: it is published, taken, parked and
/// committed with the mesh, so a mesh is never drawn against another mesh's
/// atlas.
///
/// The producer half (@ref release_and_may_publish, @ref publish,
/// @ref wait_collected) belongs to one thread, the consumer half
/// (@ref begin_frame, @ref live, @ref refused) to another.
///
/// @tparam Payload  A movable value published with each mesh.
template <typename Payload>
class MeshExchange {
 public:
  /// @brief An exchange for a consumer reading as @p config says.
  /// @param config  Its `frames_in_flight` must be at least 1.
  explicit MeshExchange(const MeshExchangeConfig& config = {})
      : config_(config), frame_generations_(config.frames_in_flight, 0) {
    VKC_CHECK(config.frames_in_flight > 0,
              "MeshExchange: frames_in_flight must be at least 1");
  }

  MeshExchange(const MeshExchange&) = delete;
  MeshExchange& operator=(const MeshExchange&) = delete;
  MeshExchange(MeshExchange&&) = delete;
  MeshExchange& operator=(MeshExchange&&) = delete;
  ~MeshExchange() = default;

  /// @brief Producer: apply the consumer's release mark to @p ring, then say
  ///        whether a mesh published now would supersede one not yet taken.
  ///
  /// Call it before every extract, on the extracting thread.
  /// @param ring  The extractor, or anything with
  ///              `release_through(std::uint64_t)`.
  /// @return `true` when no published mesh is waiting to be taken.
  template <typename Ring>
  bool release_and_may_publish(Ring& ring) {
    std::uint64_t mark = 0;
    bool uncollected = false;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      mark = release_mark_;
      uncollected = pending_.has_value();
    }
    if (mark != 0) ring.release_through(mark);
    return !uncollected;
  }

  /// @brief Producer: hand @p mesh and its @p payload to the consumer.
  ///
  /// Publish every extract, an empty one included: it holds a ring slot that
  /// only the consumer's release can free. A mesh not yet taken is
  /// superseded, and its slot is released with the generations before the
  /// next one committed.
  /// @param mesh  An extract's `DeviceMesh`; generation 0 is never one.
  void publish(const DeviceMesh& mesh, Payload payload) {
    VKC_CHECK(mesh.generation != 0,
              "MeshExchange::publish: a mesh has a nonzero generation");
    std::lock_guard<std::mutex> lock(mutex_);
    pending_.emplace(Entry{mesh, std::move(payload)});
  }

  /// @brief Producer: wait until the consumer has taken the published mesh.
  /// @param timeout  The longest wait.
  /// @param cancel   Ends the wait early once true, looked at every few
  ///                 milliseconds.
  /// @return `true` when no published mesh is waiting to be taken.
  bool wait_collected(std::chrono::milliseconds timeout,
                      const std::atomic<bool>& cancel) {
    const auto deadline = std::chrono::steady_clock::now() + timeout;
    std::unique_lock<std::mutex> lock(mutex_);
    while (pending_.has_value()) {
      const auto now = std::chrono::steady_clock::now();
      if (cancel.load() || now >= deadline) return false;
      collected_.wait_until(lock, std::min(deadline, now + kCancelPoll));
    }
    return true;
  }

  /// @brief Consumer: retire @p slot's last frame, release what no frame
  ///        holds, take the newest mesh, and commit it with @p commit.
  ///
  /// Call it once per frame, once the frame that last used @p slot has
  /// completed (after the renderer's fence wait for that slot), whether or
  /// not the frame will draw: the release does not depend on drawing. It then
  /// holds @ref live for @p slot until @p slot is retired again.
  ///
  /// A mesh taken now or parked earlier is committed as follows: an empty one
  /// at once; one that cannot be bound is refused, which latches; any other
  /// once @p commit accepts it, and parked when it does not.
  ///
  /// @param slot    The frame-in-flight slot, below `frames_in_flight`.
  /// @param commit  `bool(const DeviceMesh&, Payload&)`: make what the mesh
  ///                is drawn with (its atlas, from the payload, which it may
  ///                move from) and return `true`, or return `false` to keep
  ///                the mesh parked and the live one drawn.
  /// @return What happened to the mesh taken or parked.
  template <typename Commit>
  ExchangeOutcome begin_frame(std::uint32_t slot, Commit&& commit) {
    VKC_CHECK(slot < frame_generations_.size(),
              "MeshExchange::begin_frame: slot out of range");
    bool took = false;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      release_mark_ = retire(slot);
      if (pending_.has_value() && refused_ == nullptr && !parked_.has_value()) {
        parked_ = std::move(pending_);
        pending_.reset();
        newest_taken_ = parked_->mesh.generation;
        took = true;
      }
    }
    if (took) collected_.notify_all();
    ExchangeOutcome outcome = ExchangeOutcome::kNone;
    if (parked_.has_value()) {
      if (parked_->mesh.empty()) {
        live_ = parked_->mesh;
        outcome = ExchangeOutcome::kCommitted;
      } else if (const char* why = unbindable_reason(parked_->mesh, config_)) {
        refused_ = why;
        outcome = ExchangeOutcome::kRefused;
      } else if (commit(std::as_const(parked_->mesh), parked_->payload)) {
        live_ = parked_->mesh;
        outcome = ExchangeOutcome::kCommitted;
      } else {
        outcome = ExchangeOutcome::kParked;
      }
      if (outcome != ExchangeOutcome::kParked) parked_.reset();
    }
    frame_generations_[slot] = live_.generation;
    return outcome;
  }

  /// @brief Consumer: the committed mesh, generation 0 before the first.
  ///
  /// Draw it only when `valid()` and not `empty()`.
  const DeviceMesh& live() const noexcept { return live_; }

  /// @brief Consumer: why a mesh was refused, or null while none was.
  const char* refused() const noexcept { return refused_; }

 private:
  struct Entry {
    DeviceMesh mesh;
    Payload payload;
  };

  static constexpr std::chrono::milliseconds kCancelPoll{2};

  // Clears `slot` and returns the mark: below the oldest generation a frame
  // in flight holds, else below the live one, else, before any commit,
  // through the newest taken; never through a parked one.
  std::uint64_t retire(std::uint32_t slot) {
    frame_generations_[slot] = 0;
    std::uint64_t oldest = 0;
    for (const std::uint64_t g : frame_generations_) {
      if (g != 0 && (oldest == 0 || g < oldest)) oldest = g;
    }
    if (oldest == 0) oldest = live_.generation;
    std::uint64_t mark = oldest != 0 ? oldest - 1 : newest_taken_;
    if (parked_.has_value()) {
      mark = std::min(mark, parked_->mesh.generation - 1);
    }
    return mark;
  }

  const MeshExchangeConfig config_;

  // Shared, under mutex_.
  std::mutex mutex_;
  std::condition_variable collected_;
  std::optional<Entry> pending_;
  std::uint64_t release_mark_ = 0;

  // The consumer's own.
  std::vector<std::uint64_t> frame_generations_;
  std::optional<Entry> parked_;
  DeviceMesh live_;
  std::uint64_t newest_taken_ = 0;
  const char* refused_ = nullptr;
};

}  // namespace volumetric_kit::recon::mesh
