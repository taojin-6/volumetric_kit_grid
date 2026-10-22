// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// StreamServer under load, for ThreadSanitizer (CI runs it under TSan on the
// Mac): a publisher at full rate, keys every 30 records, while 4 viewers
// connect, read for a while and leave, over and over, more than max_viewers
// at once at times; each connection's records are whole, rising, from a key,
// and a key after every jump. Then the server is destroyed while the viewers
// are mid-chunk, within 1 s.

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <random>
#include <thread>
#include <vector>

#include "networking_loopback.hpp"
#include "test_check.hpp"

namespace vkc = volumetric_kit::core;
namespace networking = volumetric_kit::recon::networking;
namespace stream = volumetric_kit::recon::stream;
namespace net = vr_test::net;

namespace {

using net::Bytes;
using net::Clock;
using namespace std::chrono_literals;

constexpr std::uint64_t kGop = 30;

std::size_t size_of(std::uint64_t seq) {
  return seq % kGop == 0 ? 64 << 10 : 4096 + (seq * 7919) % 8192;
}

// What the viewers found, over every connection.
struct Tally {
  std::atomic<std::uint64_t> connections{0};
  std::atomic<std::uint64_t> records{0};
  std::atomic<std::uint64_t> jumps{0};
  std::atomic<bool> broken{false};
};

// One viewer: connect, check the records it gets for a while, leave; again,
// until `stop`. The last connection stays until the server goes.
void viewer(std::uint16_t port, std::uint32_t seed,
            const std::atomic<bool>& stop, Tally& tally) {
  std::mt19937 rng(seed);
  while (!stop) {
    auto client = net::connect(port);
    if (client == nullptr) {
      tally.broken = true;
      return;
    }
    const auto stay = std::chrono::milliseconds(
        std::uniform_int_distribution<int>(20, 200)(rng));
    const Clock::time_point leave = Clock::now() + stay;
    bool any = false;
    std::uint64_t last = 0;
    stream::Record record;
    Bytes payload;
    while (!stop && Clock::now() < leave) {
      const vkc::Result<bool> got = client->next(10ms, &record, &payload);
      // Past max_viewers a connection is closed at once; try again soon. Any
      // other error is a chunk that failed a check.
      if (!got.ok()) {
        if (got.status().domain() != vkc::Status::Code::IoError) {
          std::fprintf(stderr, "FAIL viewer: %s\n",
                       got.status().message().c_str());
          tally.broken = true;
          return;
        }
        std::this_thread::sleep_for(5ms);
        break;
      }
      if (!*got) continue;
      const std::uint64_t seq = record.info.seq;
      if (record.info.key != (seq % kGop == 0) ||
          payload != net::record_payload(seq, size_of(seq)) ||
          (!any && !record.info.key) || (any && seq <= last) ||
          (any && seq != last + 1 && !record.info.key)) {
        std::fprintf(stderr, "FAIL viewer: seq %llu after %llu (any %d)\n",
                     static_cast<unsigned long long>(seq),
                     static_cast<unsigned long long>(last), any);
        tally.broken = true;
        return;
      }
      if (any && seq != last + 1) ++tally.jumps;
      any = true;
      last = seq;
      ++tally.records;
    }
    ++tally.connections;
  }
}

int stress_case() {
  networking::ServerOptions options;
  options.max_viewers = 3;  // the 4 viewers are past it at times
  auto server = net::serve(net::geometry_session(), options);
  CHECK(server != nullptr);
  std::atomic<bool> stop_viewers{false};
  Tally tally;
  std::vector<std::thread> viewers;
  for (std::uint32_t v = 0; v < 4; ++v) {
    viewers.emplace_back(viewer, server->port(), 100 + v,
                         std::cref(stop_viewers), std::ref(tally));
  }
  // Checked once the viewers are joined.
  std::uint64_t seq = 0;
  bool published = true;
  const Clock::time_point end = Clock::now() + 1500ms;
  while (published && Clock::now() < end) {
    const Bytes payload = net::record_payload(seq, size_of(seq));
    published = server
                    ->publish(net::record_info(seq, seq % kGop == 0),
                              payload.data(), payload.size())
                    .ok();
    ++seq;
  }
  // Destroy it under the viewers, with records still to send.
  const Clock::time_point start = Clock::now();
  server.reset();
  const auto took = Clock::now() - start;
  stop_viewers = true;
  for (std::thread& t : viewers) t.join();
  CHECK(published);
  CHECK(took < 1s);
  std::printf(
      "stress: %llu published; %llu connections, %llu records, %llu jumps\n",
      static_cast<unsigned long long>(seq),
      static_cast<unsigned long long>(tally.connections.load()),
      static_cast<unsigned long long>(tally.records.load()),
      static_cast<unsigned long long>(tally.jumps.load()));
  CHECK(!tally.broken);
  CHECK(tally.connections > 4);
  CHECK(tally.records > 0);
  return 0;
}

// Viewers join one after another while every record published is a key, so
// the log changes under every join: each gets the session, then keys only,
// rising.
int join_under_keys_case() {
  auto server = net::serve(net::geometry_session());
  CHECK(server != nullptr);
  std::atomic<bool> stop{false};
  std::atomic<bool> published{true};
  std::thread publisher([&] {
    const Bytes payload = net::record_payload(0, 64);
    for (std::uint64_t seq = 0; !stop && published; ++seq) {
      published = server
                      ->publish(net::record_info(seq, true), payload.data(),
                                payload.size())
                      .ok();
    }
  });
  bool ok = true;
  for (int i = 0; i < 50 && ok; ++i) {
    auto client = net::connect(server->port());
    ok = client != nullptr;
    std::uint64_t last = 0;
    for (int r = 0; r < 3 && ok; ++r) {
      net::Got got;
      ok = net::next_within(*client, &got) && got.record.info.key &&
           (r == 0 || got.record.info.seq > last);
      last = got.record.info.seq;
    }
  }
  stop = true;
  publisher.join();
  CHECK(published);
  CHECK(ok);
  return 0;
}

}  // namespace

int main() {
  if (stress_case() != 0) return 1;
  if (join_under_keys_case() != 0) return 1;
  std::puts("networking stress: OK");
  return 0;
}
