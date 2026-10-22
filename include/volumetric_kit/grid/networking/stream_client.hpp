// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file networking/stream_client.hpp
/// @brief Read a session a @ref StreamServer serves, as it arrives.

#include <chrono>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "volumetric_kit/core/base/result.hpp"
#include "volumetric_kit/recon/networking/export.hpp"
#include "volumetric_kit/recon/networking/stream_server.hpp"  // kDefaultPort
#include "volumetric_kit/recon/stream/session.hpp"
#include "volumetric_kit/recon/stream/stream_reader.hpp"

namespace volumetric_kit::recon::networking {

/// @brief How a @ref StreamClient reads.
struct ClientOptions {
  /// No byte for this long, counted from @ref StreamClient::connect (the
  /// name lookup and the TCP connect included), ends the connection with
  /// `Status::Code::IoError`: a sender that stalled, or a path that died (the
  /// VPN dropped). @ref StreamClient::next judges it after taking what has
  /// come (the lookup's answer, the connect's end, bytes), so a caller back
  /// late keeps a sender that kept sending. Positive; `milliseconds::max()`
  /// is never.
  std::chrono::milliseconds idle_limit{10'000};
};

/// @brief Reads one connection to a @ref StreamServer: its session, then its
///        records in order, each checked against both CRCs
///        (`stream::ChunkParser`).
///
/// A chunk that fails a check ends the connection; nothing is resynced. A
/// seq may jump where the server skipped to a newer key, and a geometry
/// consumer then decodes again from the next key (`codec::Decoder`'s
/// `NotFound`). Its receive buffer is 2 MiB, set before the connect: the most
/// a stream carries a round trip, and what a reader that falls behind can
/// hide from the server's drop rule. Linux holds it to `net.core.rmem_max`.
///
/// **Messages.** Every error @ref next returns reads
/// `"StreamClient: <host>:<port>: <reason>"`, the host and port as given to
/// @ref connect, and the reason never names them again: a caller showing the
/// address drops the prefix. The reasons: "cannot look up the name: …" (the
/// resolver's words, or "no IPv4 address for TCP") and "cannot connect: …"
/// (the system's); past the idle limit, the stage it ended: "the lookup gave
/// no answer within N ms", "cannot connect within N ms", "connected, but no
/// session within N ms" or "no byte for N ms"; "the sender closed the
/// connection", with " before its session (it may already serve its most
/// viewers)" when none had come; a failed check's `stream::ChunkParser`
/// message; a socket call's ("cannot receive: …"); or "out of memory" for an
/// allocation that failed, bare when even the prefix cannot be allocated.
/// @ref connect's own refusals read `"StreamClient::connect: <reason>"`.
///
/// @code
/// VKC_ASSIGN(auto client, networking::StreamClient::connect(
///                             "192.168.21.217", networking::kDefaultPort));
/// stream::Record record;
/// std::vector<std::uint8_t> payload;
/// for (;;) {
///   VKC_ASSIGN(const bool got, client->next(10ms, &record, &payload));
///   if (got) decode(client->session()->tracks[record.info.track], payload);
///   // else: nothing yet, look at quit and park here
/// }
/// @endcode
///
/// @warning Not thread-safe: one thread uses it.
class VR_NETWORKING_API StreamClient {
 public:
  /// @brief A client of @p host's sender; never blocks. A name is looked up
  ///        at once, on a detached helper thread, which the client abandons
  ///        if it is destroyed first; @ref next makes the TCP connect. Since
  ///        @ref ClientOptions::idle_limit counts from here, call @ref next
  ///        soon after.
  /// @param host     A name, or a numeric IPv4 address.
  /// @param port     The sender's port.
  /// @param options  How it reads.
  /// @return The client; `Status::Code::InvalidArgument` for an empty host,
  ///         port 0 or an `idle_limit` of 0 or less;
  ///         `Status::Code::IoError` when the lookup's thread
  ///         cannot start; `Status::Code::OutOfMemory` when the client
  ///         cannot be allocated.
  static core::Result<std::unique_ptr<StreamClient>> connect(
      const std::string& host, std::uint16_t port,
      const ClientOptions& options = {});

  /// @brief Closes the connection; never waits for a lookup.
  ~StreamClient();
  StreamClient(const StreamClient&) = delete;
  StreamClient& operator=(const StreamClient&) = delete;
  StreamClient(StreamClient&&) = delete;
  StreamClient& operator=(StreamClient&&) = delete;

  /// @return The session once its chunk has arrived, which is before the
  ///         first record; null before.
  const stream::Session* session() const noexcept;

  /// @brief Wait up to @p wait for the next record, finishing the lookup and
  ///        the connect first. Each address is tried in turn; any connect
  ///        error moves to the next.
  /// @param wait          The longest to wait: however fast bytes come, a
  ///                      call ends within it and one receive (up to
  ///                      2 MiB) more. 0 takes only that receive of what
  ///                      has come; `milliseconds::max()` has no end.
  /// @param[out] record   Its header. Its `offset` is the chunk's offset in
  ///                      the bytes received, as in the file they would make.
  /// @param[out] payload  Its payload, resized to fit.
  /// @return `true` with a record; `false` when none was whole by then;
  ///         `Status::Code::NotFound` for a name that does not resolve;
  ///         `Status::Code::IoError` for a lookup that fails otherwise, a
  ///         connection refused at every address, reset or closed, or idle
  ///         past @ref ClientOptions::idle_limit;
  ///         `Status::Code::InvalidArgument` for bytes that are not a `.vrst`
  ///         stream, fail a check, or hold a session body over
  ///         `stream::kMaxSessionBytes`; `Status::Code::Unsupported` for
  ///         another version or session flags; `Status::Code::OutOfMemory`
  ///         for another chunk's body over
  ///         `stream::ChunkParser::kMaxBodyBytes` (64 MiB), before reading
  ///         it, or an allocation failure. A record setting an unknown flag
  ///         is skipped. After an error the connection is closed, and every
  ///         call returns that error again.
  core::Result<bool> next(std::chrono::milliseconds wait,
                          stream::Record* record,
                          std::vector<std::uint8_t>* payload);

 private:
  struct Impl;
  explicit StreamClient(std::unique_ptr<Impl> impl) noexcept;
  std::unique_ptr<Impl> impl_;
};

}  // namespace volumetric_kit::recon::networking
