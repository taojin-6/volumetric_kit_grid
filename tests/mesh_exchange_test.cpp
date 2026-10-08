// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// mesh::MeshExchange against a host model of the MarchingCubes ring: the
// newest mesh is taken, after the frame's release mark; the commit runs
// beside the producer and sees only its own mesh's payload; a generation is
// released once no frame in flight or live mesh holds it; a parked take is
// never released, so no extract reclaims its slot; an empty mesh with null
// buffers is committed as "draw nothing" and the exchange keeps going; an
// unbindable mesh latches; the producer's bounded wait; generations at the top
// of the range; and a seeded run of producer and consumer steps, failures
// included, in which no slot the consumer holds is ever reclaimed and the ring
// never stalls. Host only: no device.

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <memory>
#include <optional>
#include <random>
#include <thread>
#include <vector>

#include "volumetric_kit/core/vulkan/vulkan.hpp"
#include "volumetric_kit/recon/mesh/device_mesh.hpp"
#include "volumetric_kit/recon/mesh/mesh_exchange.hpp"

namespace rmesh = volumetric_kit::recon::mesh;

#define CHECK(cond)                                                        \
  do {                                                                     \
    if (!(cond)) {                                                         \
      std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
      return 1;                                                            \
    }                                                                      \
  } while (0)

namespace {

using rmesh::ExchangeOutcome;

VkBuffer fake_buffer(std::uint64_t value) {
  static_assert(sizeof(VkBuffer) == sizeof(value), "64-bit handles");
  VkBuffer buffer = VK_NULL_HANDLE;
  std::memcpy(&buffer, &value, sizeof(buffer));
  return buffer;
}

// What a viewer's extractor hands out: drawable buffers shared across two
// families, or for an empty extract on a slot never sized, null buffers.
rmesh::DeviceMesh make_mesh(std::uint64_t generation, std::uint32_t triangles) {
  rmesh::DeviceMesh mesh;
  mesh.generation = generation;
  mesh.triangle_count = triangles;
  mesh.vertex_count = 3 * triangles;
  if (triangles != 0) {
    // Never null, at any generation.
    mesh.vertices = fake_buffer((generation << 2) | 1);
    mesh.indices = fake_buffer((generation << 2) | 2);
    mesh.indirect = fake_buffer((generation << 2) | 3);
  }
  mesh.vertex_usage =
      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
  mesh.index_usage =
      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT;
  mesh.indirect_usage =
      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT;
  mesh.sharing_mode = VK_SHARING_MODE_CONCURRENT;
  return mesh;
}

// MarchingCubes' ring on the host: an extract claims, round-robin from the
// slot after the last claimed, a slot whose generation the consumer has
// released, and stamps it with the next generation; one that fails after
// claiming moves the cursor and spends the generation, stamping nothing.
class FakeRing {
 public:
  explicit FakeRing(std::size_t slots, std::uint64_t first_generation = 1)
      : slots_(slots, 0), generation_(first_generation - 1) {}

  void release_through(std::uint64_t generation) {
    released_ = std::max(released_, generation);
  }

  // The extract's mesh, or none when every slot is outstanding or it
  // failed after claiming.
  std::optional<rmesh::DeviceMesh> extract(std::uint32_t triangles,
                                           bool fail = false) {
    for (std::size_t i = 1; i <= slots_.size(); ++i) {
      const std::size_t candidate = (cursor_ + i) % slots_.size();
      if (slots_[candidate] > released_) continue;
      cursor_ = candidate;
      ++generation_;
      if (fail) return std::nullopt;
      slots_[candidate] = generation_;
      return make_mesh(generation_, triangles);
    }
    ++refusals_;
    return std::nullopt;
  }

  // Whether `generation`'s buffers are still in its slot.
  bool holds(std::uint64_t generation) const {
    return std::find(slots_.begin(), slots_.end(), generation) != slots_.end();
  }

  std::uint64_t last_generation() const { return generation_; }
  std::uint64_t released() const { return released_; }
  std::size_t refusals() const { return refusals_; }

 private:
  std::vector<std::uint64_t> slots_;
  std::uint64_t generation_ = 0;
  std::uint64_t released_ = 0;
  std::size_t cursor_ = 0;
  std::size_t refusals_ = 0;
};

// The payload a test publishes: the generation it belongs to, so a commit can
// check that the mesh and its payload stayed one value. Move-only, as an
// atlas is.
using Tag = std::unique_ptr<std::uint64_t>;

Tag tag(const rmesh::DeviceMesh& mesh) {
  return std::make_unique<std::uint64_t>(mesh.generation);
}

bool tag_matches(const rmesh::DeviceMesh& mesh, const Tag& payload) {
  return payload != nullptr && *payload == mesh.generation;
}

rmesh::MeshExchangeConfig two_frames() {
  rmesh::MeshExchangeConfig config;
  config.frames_in_flight = 2;
  config.cross_family = true;
  return config;
}

int test_unbindable_reason() {
  rmesh::MeshExchangeConfig config = two_frames();
  const rmesh::DeviceMesh good = make_mesh(1, 4);
  CHECK(rmesh::unbindable_reason(good, config) == nullptr);

  rmesh::DeviceMesh exclusive = good;
  exclusive.sharing_mode = VK_SHARING_MODE_EXCLUSIVE;
  CHECK(rmesh::unbindable_reason(exclusive, config) != nullptr);
  config.cross_family = false;
  CHECK(rmesh::unbindable_reason(exclusive, config) == nullptr);

  rmesh::DeviceMesh no_vertex_bit = good;
  no_vertex_bit.vertex_usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
  CHECK(rmesh::unbindable_reason(no_vertex_bit, config) != nullptr);
  rmesh::DeviceMesh no_index_bit = good;
  no_index_bit.index_usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
  CHECK(rmesh::unbindable_reason(no_index_bit, config) != nullptr);
  rmesh::DeviceMesh no_indirect_bit = good;
  no_indirect_bit.indirect_usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
  CHECK(rmesh::unbindable_reason(no_indirect_bit, config) != nullptr);
  rmesh::DeviceMesh null_indirect = good;
  null_indirect.indirect = VK_NULL_HANDLE;
  CHECK(rmesh::unbindable_reason(null_indirect, config) != nullptr);

  // The usage asked for is the consumer's: a consumer reading the vertices
  // as storage needs that bit, not the draw's.
  config.vertex_usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
  CHECK(rmesh::unbindable_reason(no_vertex_bit, config) == nullptr);
  config.vertex_usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
  CHECK(rmesh::unbindable_reason(good, config) != nullptr);
  return 0;
}

// The newest mesh is taken; one not yet taken holds the producer back until
// it is, and a mesh published over it supersedes it.
int test_publish_take_order() {
  rmesh::MeshExchange<Tag> exchange(two_frames());
  FakeRing ring(3);
  int commits = 0;
  auto commit = [&](const rmesh::DeviceMesh& mesh, Tag& payload) {
    ++commits;
    return tag_matches(mesh, payload);
  };

  CHECK(exchange.begin_frame(0, commit) == ExchangeOutcome::kNone);
  CHECK(exchange.live().generation == 0);
  CHECK(exchange.release_and_may_publish(ring));

  const auto g1 = ring.extract(4);
  CHECK(g1.has_value());
  exchange.publish(*g1, tag(*g1));
  CHECK(!exchange.release_and_may_publish(ring));
  CHECK(exchange.begin_frame(1, commit) == ExchangeOutcome::kCommitted);
  CHECK(exchange.live().generation == g1->generation);
  CHECK(exchange.release_and_may_publish(ring));

  // Two published before a frame: the second supersedes the first.
  const auto g2 = ring.extract(4);
  exchange.publish(*g2, tag(*g2));
  const auto g3 = ring.extract(5);
  CHECK(g3.has_value());
  exchange.publish(*g3, tag(*g3));
  CHECK(exchange.begin_frame(0, commit) == ExchangeOutcome::kCommitted);
  CHECK(exchange.live().generation == g3->generation);
  CHECK(exchange.live().triangle_count == 5);
  CHECK(commits == 2);

  // Nothing new: the live mesh stays, and the commit is not asked.
  CHECK(exchange.begin_frame(1, commit) == ExchangeOutcome::kNone);
  CHECK(exchange.live().generation == g3->generation);
  CHECK(commits == 2);
  return 0;
}

// The commit runs outside the lock, after the take has freed the producer: a
// mesh published while it runs does not change the payload it sees, and is
// the next take. This is how a consumer pairs what it shows with the mesh it
// draws, rather than reading the producer's newest state after the commit.
int test_commit_runs_beside_the_producer() {
  rmesh::MeshExchange<Tag> exchange(two_frames());
  FakeRing ring(3);
  const auto g1 = ring.extract(4);
  exchange.publish(*g1, tag(*g1));

  std::thread producer;
  std::atomic<bool> published{false};
  std::optional<rmesh::DeviceMesh> g2;
  bool published_during_commit = false;
  bool paired = false;
  auto commit_g1 = [&](const rmesh::DeviceMesh& mesh, Tag& payload) {
    producer = std::thread([&] {
      if (exchange.release_and_may_publish(ring)) {
        g2 = ring.extract(5);
        exchange.publish(*g2, tag(*g2));
        published.store(true);
      }
    });
    // Bounded, so a commit run under the lock fails here instead of hanging:
    // the producer then waits for begin_frame to return.
    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (!published.load() && std::chrono::steady_clock::now() < deadline) {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    published_during_commit = published.load();
    paired = tag_matches(mesh, payload);
    return true;
  };
  CHECK(exchange.begin_frame(0, commit_g1) == ExchangeOutcome::kCommitted);
  producer.join();
  CHECK(published_during_commit);
  CHECK(paired);
  CHECK(exchange.live().generation == g1->generation);

  CHECK(g2.has_value());
  std::uint64_t committed = 0;
  auto commit = [&](const rmesh::DeviceMesh& mesh, Tag& payload) {
    committed = tag_matches(mesh, payload) ? *payload : 0;
    return true;
  };
  CHECK(exchange.begin_frame(1, commit) == ExchangeOutcome::kCommitted);
  CHECK(committed == g2->generation);
  CHECK(exchange.live().generation == g2->generation);
  return 0;
}

// A generation is released once every frame that drew it has retired, and
// the producer sees the mark of the frame that took a mesh as soon as it may
// publish again.
int test_release_mark() {
  rmesh::MeshExchange<Tag> exchange(two_frames());
  FakeRing ring(3);
  auto commit = [](const rmesh::DeviceMesh& mesh, Tag& payload) {
    return tag_matches(mesh, payload);
  };
  const auto g1 = ring.extract(4);
  exchange.publish(*g1, tag(*g1));
  CHECK(exchange.begin_frame(0, commit) == ExchangeOutcome::kCommitted);
  CHECK(exchange.begin_frame(1, commit) == ExchangeOutcome::kNone);
  CHECK(exchange.release_and_may_publish(ring));
  CHECK(ring.released() == 0);  // both frames in flight draw g1

  const auto g2 = ring.extract(4);
  exchange.publish(*g2, tag(*g2));
  CHECK(exchange.begin_frame(0, commit) == ExchangeOutcome::kCommitted);
  CHECK(exchange.release_and_may_publish(ring));
  CHECK(ring.released() == 0);  // slot 1's frame still draws g1
  CHECK(exchange.begin_frame(1, commit) == ExchangeOutcome::kNone);
  CHECK(exchange.release_and_may_publish(ring));
  CHECK(ring.released() == g1->generation);

  // Fusion outrunning the renderer: the producer extracts the moment a take
  // frees it, so it runs on the mark of the frame that took. A three-slot
  // ring under two frames in flight never refuses it.
  for (int frame = 0; frame < 200; ++frame) {
    if (exchange.release_and_may_publish(ring)) {
      const auto mesh = ring.extract(4);
      CHECK(mesh.has_value());
      exchange.publish(*mesh, tag(*mesh));
    }
    CHECK(exchange.begin_frame(static_cast<std::uint32_t>(frame % 2), commit) ==
          ExchangeOutcome::kCommitted);
  }
  CHECK(ring.refusals() == 0);
  return 0;
}

// The first mesh's atlas cannot be made, so it is parked, and in the next
// frame nothing is in flight and nothing committed. The mark must still stop
// below it: an extract that fails after claiming moves the ring's cursor, and
// an extract forced over an uncollected mesh (a viewer's final one) would
// otherwise claim the parked slot, which the consumer then draws.
int test_parked_take_is_never_released() {
  rmesh::MeshExchange<Tag> exchange(two_frames());
  FakeRing ring(3);
  bool atlas_ready = false;
  auto commit = [&](const rmesh::DeviceMesh& mesh, Tag& payload) {
    return atlas_ready && tag_matches(mesh, payload);
  };
  const auto g1 = ring.extract(4);
  exchange.publish(*g1, tag(*g1));
  CHECK(exchange.begin_frame(0, commit) == ExchangeOutcome::kParked);
  CHECK(exchange.begin_frame(1, commit) == ExchangeOutcome::kParked);
  CHECK(exchange.live().generation == 0);
  CHECK(exchange.release_and_may_publish(ring));
  CHECK(ring.released() < g1->generation);

  CHECK(!ring.extract(4, /*fail=*/true).has_value());
  const auto g3 = ring.extract(4);
  CHECK(g3.has_value());
  exchange.publish(*g3, tag(*g3));
  CHECK(exchange.begin_frame(0, commit) == ExchangeOutcome::kParked);
  CHECK(!exchange.release_and_may_publish(ring));
  ring.extract(4);  // forced, as a final extract is
  CHECK(ring.holds(g1->generation));

  // Committed at last, it is the parked mesh that is drawn, intact, and the
  // newer one is taken next.
  atlas_ready = true;
  CHECK(exchange.begin_frame(1, commit) == ExchangeOutcome::kCommitted);
  CHECK(exchange.live().generation == g1->generation);
  CHECK(ring.holds(g1->generation));
  CHECK(exchange.begin_frame(0, commit) == ExchangeOutcome::kCommitted);
  CHECK(exchange.live().generation == g3->generation);
  return 0;
}

// An extract that meshed nothing, on a slot never sized, names null buffers.
// It is committed as "draw nothing" without the bindable check or the commit,
// and the exchange goes on taking meshes.
int test_empty_mesh_draws_nothing() {
  rmesh::MeshExchange<Tag> exchange(two_frames());
  FakeRing ring(3);
  int commits = 0;
  auto commit = [&](const rmesh::DeviceMesh& mesh, Tag& payload) {
    ++commits;
    return tag_matches(mesh, payload);
  };
  auto empty = ring.extract(0);
  CHECK(empty.has_value() && empty->empty() && !empty->valid());
  empty->sharing_mode = VK_SHARING_MODE_EXCLUSIVE;  // unchecked when empty
  exchange.publish(*empty, nullptr);
  CHECK(exchange.begin_frame(0, commit) == ExchangeOutcome::kCommitted);
  CHECK(commits == 0);
  CHECK(exchange.refused() == nullptr);
  CHECK(exchange.live().generation == empty->generation);
  CHECK(exchange.live().empty());

  CHECK(exchange.release_and_may_publish(ring));
  const auto g2 = ring.extract(4);
  CHECK(g2.has_value());
  exchange.publish(*g2, tag(*g2));
  CHECK(exchange.begin_frame(1, commit) == ExchangeOutcome::kCommitted);
  CHECK(commits == 1);
  CHECK(exchange.live().generation == g2->generation);

  // The empty mesh's slot is released like any other.
  CHECK(exchange.begin_frame(0, commit) == ExchangeOutcome::kNone);
  CHECK(exchange.release_and_may_publish(ring));
  CHECK(ring.released() == empty->generation);
  return 0;
}

// A mesh that cannot be bound is refused once and latches: nothing more is
// taken, so the producer is held back rather than walking the ring.
int test_unbindable_mesh_latches() {
  rmesh::MeshExchange<Tag> exchange(two_frames());
  FakeRing ring(3);
  auto commit = [](const rmesh::DeviceMesh& mesh, Tag& payload) {
    return tag_matches(mesh, payload);
  };
  auto g1 = ring.extract(4);
  g1->sharing_mode = VK_SHARING_MODE_EXCLUSIVE;
  exchange.publish(*g1, tag(*g1));
  CHECK(exchange.begin_frame(0, commit) == ExchangeOutcome::kRefused);
  CHECK(exchange.refused() != nullptr);
  CHECK(exchange.live().generation == 0);

  CHECK(exchange.release_and_may_publish(ring));
  const auto g2 = ring.extract(4);
  exchange.publish(*g2, tag(*g2));
  CHECK(exchange.begin_frame(1, commit) == ExchangeOutcome::kNone);
  CHECK(exchange.begin_frame(0, commit) == ExchangeOutcome::kNone);
  CHECK(!exchange.release_and_may_publish(ring));
  CHECK(exchange.live().generation == 0);
  return 0;
}

// The producer's bounded wait: done at once with nothing published, done
// once a consumer thread takes, and given up on time-out or cancel.
int test_wait_collected() {
  using std::chrono::milliseconds;
  rmesh::MeshExchange<Tag> exchange(two_frames());
  auto commit = [](const rmesh::DeviceMesh& mesh, Tag& payload) {
    return tag_matches(mesh, payload);
  };
  std::atomic<bool> cancel{false};
  CHECK(exchange.wait_collected(milliseconds(0), cancel));

  exchange.publish(make_mesh(1, 4), std::make_unique<std::uint64_t>(1));
  CHECK(!exchange.wait_collected(milliseconds(10), cancel));
  cancel.store(true);
  const auto start = std::chrono::steady_clock::now();
  CHECK(!exchange.wait_collected(milliseconds(60000), cancel));
  CHECK(std::chrono::steady_clock::now() - start < milliseconds(10000));
  cancel.store(false);

  std::thread consumer([&]() {
    std::this_thread::sleep_for(milliseconds(20));
    exchange.begin_frame(0, commit);
  });
  const bool collected = exchange.wait_collected(milliseconds(60000), cancel);
  consumer.join();
  CHECK(collected);
  CHECK(exchange.live().generation == 1);
  return 0;
}

// Generations at the top of the range: a parked one caps the mark one below
// it, and the marks follow the frames in flight up to the last generation,
// with nothing wrapping.
int test_generations_at_top_of_range() {
  constexpr std::uint64_t kMax = std::numeric_limits<std::uint64_t>::max();
  rmesh::MeshExchange<Tag> exchange(two_frames());
  FakeRing ring(3, kMax - 3);
  bool atlas_ready = false;
  auto commit = [&](const rmesh::DeviceMesh& mesh, Tag& payload) {
    return atlas_ready && tag_matches(mesh, payload);
  };
  const auto first = ring.extract(4);
  CHECK(first.has_value() && first->generation == kMax - 3);
  exchange.publish(*first, tag(*first));
  CHECK(exchange.begin_frame(0, commit) == ExchangeOutcome::kParked);
  CHECK(exchange.begin_frame(1, commit) == ExchangeOutcome::kParked);
  CHECK(exchange.release_and_may_publish(ring));
  CHECK(ring.released() == kMax - 4);

  atlas_ready = true;
  std::uint32_t frame = 0;
  for (; exchange.live().generation != kMax; ++frame) {
    CHECK(frame < 16);
    if (exchange.release_and_may_publish(ring) &&
        ring.last_generation() != kMax) {
      const auto mesh = ring.extract(4);
      CHECK(mesh.has_value());
      exchange.publish(*mesh, tag(*mesh));
    }
    exchange.begin_frame(frame % 2, commit);
  }
  exchange.begin_frame(frame % 2, commit);
  exchange.release_and_may_publish(ring);
  CHECK(ring.released() == kMax - 1);
  CHECK(ring.refusals() == 0);
  return 0;
}

// A seeded run of producer and consumer steps with every failure the
// handoff meets: extracts that fail after claiming or mesh nothing, atlases
// that fail, frames skipped (a minimized window) and extracts forced over an
// uncollected mesh. After every step, every generation the consumer holds --
// drawn by a frame in flight, live, or parked -- is still in its slot, and
// every commit pairs a mesh with its own payload. Then, with the failures
// gone, the producer keeps publishing: the ring never stalls.
int test_seeded_run() {
  constexpr std::uint32_t kFrames = 2;
  for (std::uint32_t seed = 1; seed <= 500; ++seed) {
    std::mt19937 random(seed);
    auto chance = [&](int percent) {
      return std::uniform_int_distribution<int>(0, 99)(random) < percent;
    };
    rmesh::MeshExchangeConfig config = two_frames();
    config.frames_in_flight = kFrames;
    rmesh::MeshExchange<Tag> exchange(config);
    FakeRing ring(kFrames + 1);
    std::vector<std::uint64_t> drawn(kFrames, 0);
    std::uint64_t parked = 0;
    bool healthy = false;
    bool mismatched = false;
    auto commit = [&](const rmesh::DeviceMesh& mesh, Tag& payload) {
      mismatched = mismatched || !tag_matches(mesh, payload);
      if (!healthy && chance(30)) {
        parked = mesh.generation;
        return false;
      }
      parked = 0;
      return true;
    };
    std::uint32_t frame = 0;
    std::size_t published_healthy = 0;
    for (int step = 0; step < 600; ++step) {
      if (step == 400) healthy = true;
      if (chance(50)) {
        const bool may = exchange.release_and_may_publish(ring);
        if (may || (!healthy && chance(10))) {
          const bool fail = !healthy && chance(10);
          const std::uint32_t triangles = !healthy && chance(10) ? 0 : 4;
          const auto mesh = ring.extract(triangles, fail);
          if (mesh) {
            exchange.publish(*mesh, tag(*mesh));
            published_healthy += healthy ? 1 : 0;
          }
        }
      } else if (healthy || !chance(10)) {
        const std::uint32_t slot = frame++ % kFrames;
        const ExchangeOutcome outcome = exchange.begin_frame(slot, commit);
        CHECK(outcome != ExchangeOutcome::kRefused);
        if (outcome == ExchangeOutcome::kCommitted) parked = 0;
        drawn[slot] = exchange.live().generation;
      }
      CHECK(!mismatched);
      for (const std::uint64_t g : drawn) CHECK(g == 0 || ring.holds(g));
      CHECK(parked == 0 || ring.holds(parked));
    }
    CHECK(published_healthy > 20);
  }
  return 0;
}

}  // namespace

int main() {
  if (const int rc = test_unbindable_reason()) return rc;
  if (const int rc = test_publish_take_order()) return rc;
  if (const int rc = test_commit_runs_beside_the_producer()) return rc;
  if (const int rc = test_release_mark()) return rc;
  if (const int rc = test_parked_take_is_never_released()) return rc;
  if (const int rc = test_empty_mesh_draws_nothing()) return rc;
  if (const int rc = test_unbindable_mesh_latches()) return rc;
  if (const int rc = test_wait_collected()) return rc;
  if (const int rc = test_generations_at_top_of_range()) return rc;
  if (const int rc = test_seeded_run()) return rc;
  std::puts("mesh_exchange: OK");
  return 0;
}
