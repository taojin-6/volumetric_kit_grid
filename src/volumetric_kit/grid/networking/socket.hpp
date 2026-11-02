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

namespace volumetric_kit::grid::networking::detail {

// The longest a thread waits on a socket at a time, so it sees a stop within
// it with no shutdown semantics that differ by OS.
inline constexpr std::chrono::milliseconds kPollSlice{20};
// What a sender's socket may hold unsent before poll stops calling it
// writable (TCP_NOTSENT_LOWAT): a viewer takes its next record only below it.
inline constexpr int kSendLowWater = 128 << 10;
// A client's receive buffer, set before the connect so the window scale fits
// it: the most a stream carries a round trip, and what a reader that falls
// behind can hide from the sender's drop rule. Sized for the rig's stream
// over the VPN (DECISIONS.md, 2026-10-21 networking).
inline constexpr int kReceiveBuffer = 2 << 20;
// The congestion control a sender's socket to a peer that is not loopback
// asks for, on Linux (TCP_CONGESTION): at a long round trip CUBIC's window
// collapses on loss and decays while a stream idles between frames; BBR's
// does neither (DECISIONS.md, 2026-11-01 networking).
inline constexpr std::string_view kCongestion = "bbr";

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

// Whether `address` is IPv4 loopback, 127.0.0.0/8.
bool is_loopback(const Address& address);

// `host`, one host's numeric IPv4 address, with `port`: InvalidArgument
// "<api>: <reason>" for a host that is empty, not numeric IPv4, or the
// wildcard 0.0.0.0, which names every interface to listen on and no host to
// connect to. Each reason a sentence an address entry can show.
core::Result<Address> one_host(std::string_view api, const std::string& host,
                               std::uint16_t port);

// Whether `allowed`, names separated by white space as
// /proc/sys/net/ipv4/tcp_allowed_congestion_control holds them, lists `name`
// as a whole word.
bool lists(std::string_view allowed, std::string_view name);

// Make `fd` close-on-exec and non-blocking, and on Apple never raise
// SIGPIPE, as every socket of the tier is.
core::Status prepare(std::string_view api, int fd);

// A socket listening on `address` (SO_REUSEADDR, so a restarted sender takes
// its port back at once), its receive buffer `receive_buffer` bytes when
// positive, set before the listen so that each socket it accepts inherits
// it and the window scale fits it. IoError for an address that cannot be
// bound.
core::Result<core::UniqueFd> listen_on(std::string_view api,
                                       const Address& address,
                                       int receive_buffer = 0);

// The port `fd` is bound to.
core::Result<std::uint16_t> bound_port(std::string_view api, int fd);

// The peer of `fd`, which must be a connected IPv4 stream socket:
// InvalidArgument for one that is not a socket, not a stream socket or not
// IPv4; IoError for one not connected, or reset already.
core::Status peer_of(std::string_view api, int fd, Address* peer);

// Set `fd`, connected to `peer`, up to send: TCP_NODELAY, TCP_NOTSENT_LOWAT
// at kSendLowWater, and on Linux prefer_congestion for a peer that is not
// loopback. The accepted viewers' and the added ones' one setup; IoError for
// a socket that cannot be set up, as one its peer reset.
core::Status set_up_sender(std::string_view api, int fd, const Address& peer);

// The next connection waiting on `listener`, prepared, and its peer; an
// empty fd when none is waiting, or one failed before it was taken, cannot
// be prepared or has no peer left (reset as it waited), which is closed.
// IoError (OutOfMemory for the kernel's
// memory) for an accept that fails otherwise, as out of descriptors: Linux
// keeps that connection queued, and macOS drops it.
core::Result<core::UniqueFd> accept_one(std::string_view api, int listener,
                                        Address* peer);

// accept_one's connection, set up to send (set_up_sender); one that cannot
// be is closed, and an empty fd returned.
core::Result<core::UniqueFd> accept_next(std::string_view api, int listener,
                                         Address* peer);

#ifdef __linux__
// Set `fd` to kCongestion if `allowed`, the text of
// tcp_allowed_congestion_control, lists it: whether it did. Linux limits a
// process without CAP_NET_ADMIN to the algorithms listed there; consulting
// the list here makes one with it, as root, behave as every other.
// Otherwise, or if the kernel refuses, `fd` keeps the kernel's default, and
// the first such socket of a process says so at Info, naming `api`.
bool prefer_congestion(std::string_view api, int fd, std::string_view allowed);
#endif

// A socket, its receive buffer `receive_buffer` bytes when positive (a
// client's kReceiveBuffer by default), connecting to `address` without
// waiting: `*connected` says whether it already has, else poll it writable
// and call finish_connect. `api` names the address.
core::Result<core::UniqueFd> start_connect(std::string_view api,
                                           const Address& address,
                                           bool* connected,
                                           int receive_buffer = kReceiveBuffer);

// The outcome of the connect start_connect began on `fd`, once it polls
// writable.
core::Status finish_connect(std::string_view api, int fd);

// A socket connected to `address` within `limit`, which is positive, its
// receive buffer the kernel's: start_connect, a wait, then finish_connect.
// IoError "<api>: cannot connect: <the system's reason>", or "<api>: cannot
// connect within N ms" past `limit`.
core::Result<core::UniqueFd> connect_within(std::string_view api,
                                            const Address& address,
                                            std::chrono::milliseconds limit);

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

}  // namespace volumetric_kit::grid::networking::detail
