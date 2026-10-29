// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// StreamServer on loopback (127.0.0.1, port 0) when a record's copy fails, an
// allocation failure injected there: without replay_gop, publish returns
// OutOfMemory and has counted the viewer it sends back in key_wanted before
// it returns; a key lost so counts the viewer waiting for it again; the next
// key lets it in, with no record between. Apart from the server's other
// tests, which stay free of the replaced operator new, so they link under
// ThreadSanitizer on Linux too.

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>

#include "networking_loopback.hpp"
#include "volumetric_kit/core/testing/test_allocation_failure.hpp"
#include "volumetric_kit/core/testing/test_check.hpp"

namespace vkc = volumetric_kit::core;
namespace networking = volumetric_kit::recon::networking;
namespace net = vr_test::net;

namespace {

using Code = vkc::Status::Code;
using net::Bytes;
using net::Got;
using namespace std::chrono_literals;

int lost_copy_case() {
  constexpr std::size_t kSize = 64 << 10;
  networking::ServerOptions options;
  options.replay_gop = false;
  auto server = net::serve(net::geometry_session(), options);
  CHECK(server != nullptr);
  auto client = net::connect(server->port());
  CHECK(client != nullptr && net::wait_session(*client));
  const auto publish = [&](std::uint64_t seq, bool key) {
    const Bytes payload = net::record_payload(seq, kSize);
    return server
        ->publish(net::record_info(seq, key), payload.data(), payload.size())
        .domain();
  };
  Got got;
  CHECK(publish(0, true) == Code::Ok);
  CHECK(net::next_within(*client, &got) && got.record.info.seq == 0);
  // A publish's largest allocation is its chunk's copy, the same at every
  // record of this size.
  auto& failure = vkc::test::allocation_failure;
  failure = {};
  failure.measure = true;
  const Code measured = publish(1, false);
  failure.measure = false;
  const std::size_t copy = failure.largest;
  const void* const site = failure.largest_site;
  failure = {};
  CHECK(measured == Code::Ok && copy > kSize);
  CHECK(net::next_within(*client, &got) && got.record.info.seq == 1);
  const auto lost = [&](std::uint64_t seq, bool key) {
    failure.bytes = copy;
    failure.site = site;
    failure.armed = true;
    const Code code = publish(seq, key);
    const bool injected = failure.injected;
    failure = {};
    return injected && code == Code::OutOfMemory;
  };
  CHECK(server->key_wanted() == 1);
  CHECK(lost(2, false));
  CHECK(server->key_wanted() == 2);
  CHECK(lost(3, true));
  CHECK(server->key_wanted() == 3);
  CHECK(publish(4, false) == Code::Ok);
  CHECK(!net::next_within(*client, &got, 200ms));
  CHECK(publish(5, true) == Code::Ok);
  CHECK(net::next_within(*client, &got) && got.record.info.seq == 5);
  CHECK(server->key_wanted() == 3);
  return 0;
}

}  // namespace

int main() {
  if (lost_copy_case() != 0) return 1;
  std::puts("networking server oom: OK");
  return 0;
}
