// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

// What the networking tests share: raw sockets on 127.0.0.1 port 0 (never
// 0.0.0.0, which could raise a firewall prompt) to play a sender or a viewer
// with any bytes, a server and a client on loopback, and the records a client
// reads.

#include <arpa/inet.h>
#include <netinet/in.h>
#include <poll.h>
#include <sys/socket.h>
#include <sys/types.h>  // IWYU pragma: keep (ssize_t on Linux)
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "stream_files.hpp"
#include "volumetric_kit/core/base/unique_fd.hpp"
#include "volumetric_kit/recon/networking/stream_client.hpp"
#include "volumetric_kit/recon/networking/stream_server.hpp"
#include "volumetric_kit/recon/stream/chunks.hpp"

namespace vr_test::net {

namespace networking = volumetric_kit::recon::networking;
namespace stream = volumetric_kit::recon::stream;
namespace vkc = volumetric_kit::core;
using Bytes = std::vector<std::uint8_t>;
using Clock = std::chrono::steady_clock;
using namespace std::chrono_literals;

inline sockaddr_in loopback(std::uint16_t port) {
  sockaddr_in a{};
  a.sin_family = AF_INET;
  a.sin_port = htons(port);
  a.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  return a;
}

inline std::uint16_t local_port(int fd) {
  sockaddr_in a{};
  socklen_t size = sizeof a;
  if (::getsockname(fd, reinterpret_cast<sockaddr*>(&a), &size) != 0) return 0;
  return ntohs(a.sin_port);
}

// A blocking socket listening on 127.0.0.1, on a free port.
struct Listener {
  vkc::UniqueFd fd;
  std::uint16_t port = 0;
};
inline Listener bind_loopback() {
  Listener l;
  l.fd.reset(::socket(AF_INET, SOCK_STREAM, 0));
  const sockaddr_in a = loopback(0);
  if (!l.fd.valid() ||
      ::bind(l.fd.get(), reinterpret_cast<const sockaddr*>(&a), sizeof a) !=
          0 ||
      ::listen(l.fd.get(), 8) != 0) {
    std::perror("bind_loopback");
    return {};
  }
  l.port = local_port(l.fd.get());
  return l;
}

// A port on 127.0.0.1 that was free a moment ago, its listener closed, so a
// connect to it is refused. (One bound and not listening is no use: macOS
// drops a SYN to it, where Linux refuses it.)
inline std::uint16_t closed_port() { return bind_loopback().port; }

// The next connection on `listener` within `timeout`, or none.
inline vkc::UniqueFd accept_within(int listener,
                                   std::chrono::milliseconds timeout) {
  pollfd p{listener, POLLIN, 0};
  if (::poll(&p, 1, static_cast<int>(timeout.count())) != 1) return {};
  return vkc::UniqueFd(::accept(listener, nullptr, nullptr));
}

// A blocking connection to 127.0.0.1:`port`, its receive buffer
// `receive_buffer` bytes when nonzero.
inline vkc::UniqueFd connect_loopback(std::uint16_t port,
                                      int receive_buffer = 0) {
  vkc::UniqueFd fd(::socket(AF_INET, SOCK_STREAM, 0));
  if (!fd.valid()) return {};
  if (receive_buffer > 0) {
    ::setsockopt(fd.get(), SOL_SOCKET, SO_RCVBUF, &receive_buffer,
                 sizeof receive_buffer);
  }
  const sockaddr_in a = loopback(port);
  if (::connect(fd.get(), reinterpret_cast<const sockaddr*>(&a), sizeof a) !=
      0) {
    return {};
  }
  return fd;
}

inline bool send_bytes(int fd, const Bytes& bytes) {
  std::size_t done = 0;
  while (done < bytes.size()) {
    const ssize_t n = ::send(fd, bytes.data() + done, bytes.size() - done, 0);
    if (n <= 0) return false;
    done += static_cast<std::size_t>(n);
  }
  return true;
}

// Read from `fd` until `out` holds `size` bytes, the peer's end (`*eof`), or
// `timeout`.
inline void read_upto(int fd, std::size_t size,
                      std::chrono::milliseconds timeout, Bytes* out,
                      bool* eof = nullptr) {
  const Clock::time_point end = Clock::now() + timeout;
  std::uint8_t buffer[65536];
  if (eof != nullptr) *eof = false;
  while (out->size() < size) {
    const auto left = std::chrono::duration_cast<std::chrono::milliseconds>(
        end - Clock::now());
    if (left.count() <= 0) return;
    pollfd p{fd, POLLIN, 0};
    if (::poll(&p, 1, static_cast<int>(left.count())) != 1) continue;
    const std::size_t want = std::min(sizeof buffer, size - out->size());
    const ssize_t n = ::recv(fd, buffer, want, 0);
    if (n <= 0) {
      if (eof != nullptr) *eof = true;
      return;
    }
    out->insert(out->end(), buffer, buffer + n);
  }
}

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

// Whether the client reads the sender's end, IoError, within `timeout`, with
// no record before it.
inline bool wait_end(networking::StreamClient& client,
                     std::chrono::milliseconds timeout = 5s) {
  const Clock::time_point end = Clock::now() + timeout;
  stream::Record record;
  Bytes payload;
  for (;;) {
    const vkc::Result<bool> got = client.next(10ms, &record, &payload);
    if (!got.ok()) return got.status().domain() == vkc::Status::Code::IoError;
    if (*got || Clock::now() >= end) return false;
  }
}

// A record's seq and key, with a payload made from its seq.
inline stream::RecordInfo record_info(std::uint64_t seq, bool key) {
  return vrst::info(0, key, 1'000'000 * seq, seq);
}
inline Bytes record_payload(std::uint64_t seq, std::size_t size) {
  return vrst::pattern(size, seq);
}

}  // namespace vr_test::net
