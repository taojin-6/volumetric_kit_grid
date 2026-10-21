// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

// The BSD socket calls the server and the client make, each failure a Status
// naming the caller, what failed and the system's reason. Every socket is
// non-blocking and close-on-exec, set explicitly, accepted ones included
// (Linux does not inherit the flags, and macOS has no accept4), and never
// raises SIGPIPE: SO_NOSIGPIPE on Apple, MSG_NOSIGNAL elsewhere, chosen by
// platform since Apple's SDKs define MSG_NOSIGNAL too. Every call is retried
// on EINTR: an application's signal handler, as fuse_orbbec's for Ctrl+C,
// interrupts poll, which is never restarted.

#include <netinet/in.h>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include "volumetric_kit/core/base/result.hpp"
#include "volumetric_kit/core/base/unique_fd.hpp"

namespace volumetric_kit::recon::networking::detail {

// The longest a thread waits on a socket at a time, so it sees a stop within
// it with no shutdown semantics that differ by OS.
inline constexpr std::chrono::milliseconds kPollSlice{20};
// What a sender's socket may hold unsent before poll stops calling it
// writable (TCP_NOTSENT_LOWAT): a viewer takes its next record only below it.
inline constexpr int kSendLowWater = 128 << 10;
// A client's receive buffer, set before the connect so the window scale fits
// it: what a reader that falls behind can hide from the sender's drop rule.
inline constexpr int kReceiveBuffer = 256 << 10;

// `from` + `limit`, saturating: a limit past the clock's end, as
// milliseconds::max() ("never"), gives that end; one of 0 or less, `from`.
inline std::chrono::steady_clock::time_point after(
    std::chrono::steady_clock::time_point from,
    std::chrono::milliseconds limit) {
  using Clock = std::chrono::steady_clock;
  if (limit <= std::chrono::milliseconds{0}) return from;
  // Compared in milliseconds: the limit in the clock's nanoseconds may not
  // fit.
  if (limit >= std::chrono::floor<std::chrono::milliseconds>(
                   Clock::time_point::max() - from)) {
    return Clock::time_point::max();
  }
  return from + limit;
}

// An IPv4 socket address. A StreamServer listens on IPv4 alone, so a client
// tries no other family: an IPv6 address could only be refused, or stall a
// connect where IPv6 is broken.
struct Address {
  sockaddr_in in{};
};

// `host`, a numeric IPv4 address ("192.168.21.217"), with `port`; false for
// anything else.
bool parse_ipv4(const std::string& host, std::uint16_t port, Address* out);

// Every IPv4 address `host` resolves to for TCP (getaddrinfo), with `port`.
// Blocks for as long as the resolver takes. NotFound for a name that does not
// resolve; IoError for a lookup that fails otherwise; `api` names the host.
core::Result<std::vector<Address>> resolve(std::string_view api,
                                           const std::string& host,
                                           std::uint16_t port);

// "a.b.c.d:port".
std::string to_string(const Address& address);

// A socket listening on `address` (SO_REUSEADDR, so a restarted sender takes
// its port back at once). IoError for an address that cannot be bound.
core::Result<core::UniqueFd> listen_on(std::string_view api,
                                       const Address& address);

// The port `fd` is bound to.
core::Result<std::uint16_t> bound_port(std::string_view api, int fd);

// The next connection waiting on `listener`, set up to send (TCP_NODELAY,
// TCP_NOTSENT_LOWAT at kSendLowWater), and its peer; an empty fd when none
// is waiting, or one failed before it was taken or cannot be set up, which
// is closed.
core::Result<core::UniqueFd> accept_next(std::string_view api, int listener,
                                         Address* peer);

// A socket, its receive buffer kReceiveBuffer, connecting to `address`
// without waiting: `*connected` says whether it already has, else poll it
// writable and call finish_connect. `api` names the address.
core::Result<core::UniqueFd> start_connect(std::string_view api,
                                           const Address& address,
                                           bool* connected);

// The outcome of the connect start_connect began on `fd`, once it polls
// writable.
core::Status finish_connect(std::string_view api, int fd);

// Wait up to `timeout` for `events` on `fd` (poll): the events that came, 0
// when none did.
core::Result<short> wait_for(std::string_view api, int fd, short events,
                             std::chrono::milliseconds timeout);

// Send what `fd` takes now of `size` bytes at `data`: the bytes sent, 0 when
// it takes none.
core::Result<std::size_t> send_some(std::string_view api, int fd,
                                    const std::uint8_t* data, std::size_t size);

// The bytes sent on `fd` that its peer has not acknowledged, unsent ones
// included: SO_NWRITE on Apple, SIOCOUTQ elsewhere.
core::Result<std::size_t> unacknowledged(std::string_view api, int fd);

// What one receive found.
struct Received {
  std::size_t bytes = 0;  // read; 0 when none had come
  bool closed = false;    // the peer closed its side
};

// Receive what has come on `fd`, up to `size` bytes, without waiting.
core::Result<Received> receive_some(std::string_view api, int fd,
                                    std::uint8_t* out, std::size_t size);

}  // namespace volumetric_kit::recon::networking::detail
