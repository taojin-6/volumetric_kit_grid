// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file networking/dial.hpp
/// @brief A connection made the other way round: the sender dials its
///        reader, which listens, so the sender listens on nothing.

#include <chrono>
#include <cstdint>
#include <memory>
#include <string>

#include "volumetric_kit/core/base/result.hpp"
#include "volumetric_kit/core/base/unique_fd.hpp"
#include "volumetric_kit/grid/networking/export.hpp"

namespace volumetric_kit::grid::networking {

/// @brief Connect to a reader's @ref Listener, for a sender to serve the
///        socket through @ref StreamServer::add_viewer, which sets it up to
///        send.
///
/// Blocks the calling thread until the connect ends or @p limit passes. The
/// socket is non-blocking and close-on-exec, as every socket of this tier.
/// Each refusal's reason, after `"dial: "`, is a sentence an address entry
/// can show as it is: the checks are the ones such an entry makes.
///
/// @code
/// VKC_ASSIGN(core::UniqueFd fd, networking::dial("10.8.0.1", port, 2s));
/// VKC_TRY(server->add_viewer(std::move(fd)));
/// @endcode
/// @param host   The reader's numeric IPv4 address.
/// @param port   Its port.
/// @param limit  The longest to wait for the connect; positive.
/// @return The connected socket; `Status::Code::InvalidArgument` for a host
///         that is empty, not numeric IPv4 or the wildcard 0.0.0.0, port 0,
///         or a @p limit of 0 or less (`"dial: <reason>"`);
///         `Status::Code::IoError` for a connect refused or failed
///         (`"dial: <host>:<port>: cannot connect: …"`, the system's words)
///         or not made within @p limit (`"… cannot connect within N ms"`);
///         `Status::Code::OutOfMemory` when the kernel has no memory for the
///         socket.
VGRID_NETWORKING_API core::Result<core::UniqueFd> dial(
    const std::string& host, std::uint16_t port,
    std::chrono::milliseconds limit);

/// @brief Listens on one numeric IPv4 address for senders that dial
///        (@ref dial), for a reader to read each connection with
///        @ref StreamClient::adopt.
///
/// The address names one interface. A wildcard would listen on every one,
/// and whatever reaches the port can offer a session, so it is refused.
/// Each accepted socket has a client's 2 MiB receive buffer, set on the
/// listening socket so that the window scale fits it. Linux holds it to
/// `net.core.rmem_max`, as it does a client's.
///
/// Errors read `"Listener: <host>:<port>: <reason>"`; @ref create's
/// refusals read `"Listener::create: <reason>"`.
///
/// @code
/// VKC_ASSIGN(auto listener, networking::Listener::create("10.8.0.1", port));
/// for (;;) {
///   VKC_ASSIGN(core::UniqueFd fd, listener->accept(100ms));
///   if (!fd.valid()) continue;  // none yet: look at quit here
///   VKC_ASSIGN(auto client, networking::StreamClient::adopt(std::move(fd)));
///   // read client as a connected one is read until it ends, then accept again
/// }
/// @endcode
///
/// @warning Not thread-safe: one thread uses it.
class VGRID_NETWORKING_API Listener {
 public:
  /// @brief Listen on @p host:@p port.
  /// @param host  A numeric IPv4 address of this host; 127.0.0.1 is one.
  /// @param port  The port; 0 takes a free one, which @ref port names.
  /// @return The listener; `Status::Code::InvalidArgument` for a host that
  ///         is empty, not numeric IPv4 or the wildcard 0.0.0.0;
  ///         `Status::Code::IoError` for an address that cannot be bound (in
  ///         use, or not this host's); `Status::Code::OutOfMemory` when it
  ///         cannot be allocated.
  static core::Result<std::unique_ptr<Listener>> create(const std::string& host,
                                                        std::uint16_t port);

  /// @brief Stop listening; a connection not yet accepted is refused.
  ~Listener();
  Listener(const Listener&) = delete;
  Listener& operator=(const Listener&) = delete;
  Listener(Listener&&) = delete;
  Listener& operator=(Listener&&) = delete;

  /// @brief Wait up to @p wait for the next connection.
  /// @param wait  The longest to wait; 0 or less takes only one already
  ///              waiting.
  /// @return The connection, non-blocking and close-on-exec, for
  ///         @ref StreamClient::adopt; an empty fd when none came, or when
  ///         one failed before it was taken; `Status::Code::IoError` (or
  ///         `Status::Code::OutOfMemory`) for an accept that fails otherwise,
  ///         as out of descriptors, which leaves the listener listening
  ///         (Linux keeps that connection queued; macOS drops it).
  core::Result<core::UniqueFd> accept(std::chrono::milliseconds wait);

  /// @return The port listened on: the one asked for, or the one taken for 0.
  std::uint16_t port() const noexcept;

 private:
  Listener(core::UniqueFd fd, std::uint16_t port, std::string name) noexcept;
  core::UniqueFd fd_;
  std::uint16_t port_ = 0;
  std::string name_;  // "Listener: host:port", what each error starts with
};

}  // namespace volumetric_kit::grid::networking
