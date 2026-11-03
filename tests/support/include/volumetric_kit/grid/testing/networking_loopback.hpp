// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

// What the networking tests share: a server and a client on loopback, the
// records a client reads and the viewers a server serves, beside the core's
// raw sockets on 127.0.0.1 port 0 (core/testing/net_loopback.hpp), named
// here too, to play a sender or a viewer with any bytes.

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "volumetric_kit/core/testing/net_loopback.hpp"
#include "volumetric_kit/grid/networking/stream_client.hpp"
#include "volumetric_kit/grid/networking/stream_server.hpp"
#include "volumetric_kit/grid/stream/chunks.hpp"
#include "volumetric_kit/grid/testing/stream_files.hpp"

namespace vgrid_test::net {

namespace networking = volumetric_kit::grid::networking;
namespace stream = volumetric_kit::grid::stream;
namespace vkc = volumetric_kit::core;
using Bytes = std::vector<std::uint8_t>;
using Clock = std::chrono::steady_clock;
using namespace std::chrono_literals;

// The core's raw loopback sockets, as net::bind_loopback and so on.
using vkc::test::accept_within;
using vkc::test::bind_loopback;
using vkc::test::closed_port;
using vkc::test::connect_loopback;
using vkc::test::local_port;
using vkc::test::loopback;
using vkc::test::LoopbackListener;
using vkc::test::read_upto;
using vkc::test::send_bytes;

// A one-track Geometry session, as a sender of the codec's frames has.
inline stream::Session geometry_session() {
  stream::Session s;
  s.created_ns = 1760000000000000001ull;
  s.writer = "networking test";
  stream::Track track;
  track.kind = stream::TrackKind::Geometry;
  track.rate = {30, 1};
  s.tracks.push_back(track);
  return s;
}

// The bytes of `session`'s chunk, as a viewer receives them first.
inline Bytes session_chunk(const stream::Session& session) {
  vkc::Result<stream::ChunkEncoder> encoder =
      stream::ChunkEncoder::create(session);
  return encoder.ok() ? encoder->session_chunk() : Bytes{};
}

// A server on 127.0.0.1, a free port unless one is given, or null with the
// reason on stderr.
inline std::unique_ptr<networking::StreamServer> serve(
    const stream::Session& session, networking::ServerOptions options = {}) {
  options.host = "127.0.0.1";
  if (options.port == networking::kDefaultPort) options.port = 0;
  vkc::Result<std::unique_ptr<networking::StreamServer>> server =
      networking::StreamServer::start(session, options);
  if (!server.ok()) {
    std::fprintf(stderr, "start: %s\n", server.status().message().c_str());
    return nullptr;
  }
  return std::move(server).value();
}

inline std::unique_ptr<networking::StreamClient> connect(
    std::uint16_t port, const networking::ClientOptions& options = {},
    const std::string& host = "127.0.0.1") {
  vkc::Result<std::unique_ptr<networking::StreamClient>> client =
      networking::StreamClient::connect(host, port, options);
  if (!client.ok()) {
    std::fprintf(stderr, "connect: %s\n", client.status().message().c_str());
    return nullptr;
  }
  return std::move(client).value();
}

// The reason in `message`, an error of a client of 127.0.0.1:`port`, which
// reads "StreamClient: 127.0.0.1:<port>: <reason>" with the address in it
// once (stream_client.hpp); when it does not, the message marked so that it
// matches no reason.
inline std::string client_reason(const std::string& message,
                                 std::uint16_t port) {
  const std::string address = "127.0.0.1:" + std::to_string(port);
  const std::string prefix = "StreamClient: " + address + ": ";
  if (message.compare(0, prefix.size(), prefix) != 0 ||
      message.find(address, prefix.size()) != std::string::npos) {
    return "(not a client's) " + message;
  }
  return message.substr(prefix.size());
}

// Read until the client has its session, within `timeout`, before anything
// is published: a record read here would be lost, and fails it.
inline bool wait_session(networking::StreamClient& client,
                         std::chrono::milliseconds timeout = 5s) {
  const Clock::time_point end = Clock::now() + timeout;
  stream::Record record;
  Bytes payload;
  while (client.session() == nullptr && Clock::now() < end) {
    const vkc::Result<bool> got = client.next(10ms, &record, &payload);
    if (!got.ok()) {
      std::fprintf(stderr, "wait_session: %s\n",
                   got.status().message().c_str());
      return false;
    }
    if (*got) return false;
  }
  return client.session() != nullptr;
}

// A record as a client read it.
struct Got {
  stream::Record record;
  Bytes payload;
};

// The next record within `timeout`, or none (the reason on stderr for a
// failure).
inline bool next_within(networking::StreamClient& client, Got* got,
                        std::chrono::milliseconds timeout = 5s) {
  const Clock::time_point end = Clock::now() + timeout;
  for (;;) {
    const vkc::Result<bool> r = client.next(10ms, &got->record, &got->payload);
    if (!r.ok()) {
      std::fprintf(stderr, "next: %s\n", r.status().message().c_str());
      return false;
    }
    if (*r) return true;
    if (Clock::now() >= end) return false;
  }
}

// Whether the client of 127.0.0.1:`port` reads the sender's end after its
// session, within `timeout`, with no record before it (another error on
// stderr).
inline bool wait_end(networking::StreamClient& client, std::uint16_t port,
                     std::chrono::milliseconds timeout = 5s) {
  const Clock::time_point end = Clock::now() + timeout;
  stream::Record record;
  Bytes payload;
  for (;;) {
    const vkc::Result<bool> got = client.next(10ms, &record, &payload);
    if (!got.ok()) {
      if (got.status().domain() == vkc::Status::Code::IoError &&
          client_reason(got.status().message(), port) ==
              "the sender closed the connection") {
        return true;
      }
      std::fprintf(stderr, "wait_end: %s\n", got.status().message().c_str());
      return false;
    }
    if (*got || Clock::now() >= end) return false;
  }
}

// Whether `server` serves `n` viewers within 5 s: it sees a viewer's end
// within a poll slice.
inline bool serves_within(const networking::StreamServer& server,
                          std::size_t n) {
  const Clock::time_point end = Clock::now() + 5s;
  while (server.viewers() != n) {
    if (Clock::now() >= end) return false;
    std::this_thread::sleep_for(5ms);
  }
  return true;
}

// A record's seq and key, with a payload made from its seq.
inline stream::RecordInfo record_info(std::uint64_t seq, bool key) {
  return vrst::info(0, key, 1'000'000 * seq, seq);
}
inline Bytes record_payload(std::uint64_t seq, std::size_t size) {
  return vrst::pattern(size, seq);
}

}  // namespace vgrid_test::net
