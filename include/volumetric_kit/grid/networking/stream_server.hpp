// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file networking/stream_server.hpp
/// @brief Serve a session to viewers over TCP as it is recorded.

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

#include "volumetric_kit/core/base/result.hpp"
#include "volumetric_kit/recon/networking/export.hpp"
#include "volumetric_kit/recon/stream/session.hpp"

namespace volumetric_kit::recon::networking {

/// The port a sender serves on, and a viewer connects to when none is named.
inline constexpr std::uint16_t kDefaultPort = 47801;

/// @brief Where and how a @ref StreamServer serves.
struct ServerOptions {
  /// A numeric IPv4 address to listen on; empty listens on every interface.
  std::string host;
  /// The port; 0 takes a free one, which @ref StreamServer::port names.
  std::uint16_t port = kDefaultPort;
  /// Viewers served at once, first come: a connection past them is closed at
  /// accept, the first such refusal and then one in 10 s at most logged at
  /// Warning. With no authentication, any host reaching the port can take a
  /// slot, and keeps it until it leaves or stalls (@ref stall_limit).
  std::size_t max_viewers = 4;
  /// The most one GOP (a key record and every record after it) may hold. A
  /// record that would pass it is dropped with the whole GOP, and so is every
  /// record until the next key: the viewers freeze until it.
  std::uint64_t max_gop_bytes = std::uint64_t{64} << 20;
  /// A viewer that takes no byte for this long, with bytes waiting for it, is
  /// closed: a peer that vanished. A byte its end acknowledges is taken, so
  /// one that reads, however slowly, keeps its slot, as does one with nothing
  /// waiting (before the first key, or while nothing is published). Positive;
  /// `milliseconds::max()` is never.
  std::chrono::milliseconds stall_limit{10'000};
};

/// @brief Serves one session to viewers over TCP as it is recorded: each
///        gets the session header, the records from the newest key on, then
///        every record as it is published (a @ref StreamClient reads them).
///
/// **The wire is the file.** A viewer receives the `VRST` chunk, then `RECD`
/// chunks byte for byte as `stream::StreamWriter` writes them, with no footer
/// (`stream::ChunkEncoder`).
///
/// **Newest key wins.** The server keeps one GOP: the newest key record and
/// every record after it. A viewer joins at that key, so it decodes at once.
/// It takes its next record only while its socket holds under 128 KiB unsent
/// (`TCP_NOTSENT_LOWAT`), so little backlog hides in the kernel. When a newer
/// key arrives before it has sent the GOP, it finishes the chunk under way,
/// skips the rest and goes on from that key. A P-frame is never sent without
/// the records it is predicted from.
///
/// **Joining.** A viewer's first record, after its `VRST` chunk, is the GOP's
/// key, and the server keeps every record since. So once a viewer has the
/// session, it misses no record published after, except by newest key wins.
///
/// **Threads.** @ref publish never waits on a socket or a viewer: it copies
/// the record once, takes one lock and wakes the viewers. One thread accepts
/// and one thread per viewer sends. A viewer that joins, leaves or fails is
/// logged at Info, and never reaches @ref publish.
///
/// @code
/// stream::Session session;
/// session.writer = "serve_stream";
/// session.tracks.push_back({stream::TrackKind::Geometry});
/// VKC_ASSIGN(auto server, networking::StreamServer::start(session));
/// VKC_TRY(server->publish(info, frame.data(), frame.size()));
/// @endcode
///
/// @warning Call @ref publish from one thread at a time.
class VR_NETWORKING_API StreamServer {
 public:
  /// @brief Listen on @p options' address and start accepting viewers.
  /// @param session  What each viewer gets first: a session
  ///                 `stream::StreamWriter::create` takes, of one track.
  /// @param options  Where and how it serves.
  /// @return The server; `Status::Code::InvalidArgument` for a session the
  ///         writer would refuse, a host that is not numeric IPv4, a
  ///         `max_viewers` of 0 or a `stall_limit` of 0 or less;
  ///         `Status::Code::Unsupported` for more than
  ///         one track; `Status::Code::IoError` for a port that cannot be
  ///         bound (in use) or a thread that cannot start;
  ///         `Status::Code::OutOfMemory` when the session's chunk cannot be
  ///         allocated.
  static core::Result<std::unique_ptr<StreamServer>> start(
      const stream::Session& session, const ServerOptions& options = {});

  /// @brief Close every connection and join every thread: within one poll
  ///        slice (about 20 ms) and the write under way.
  ~StreamServer();
  StreamServer(const StreamServer&) = delete;
  StreamServer& operator=(const StreamServer&) = delete;
  StreamServer(StreamServer&&) = delete;
  StreamServer& operator=(StreamServer&&) = delete;

  /// @brief Hand one record to every viewer, now or as each catches up.
  /// @param info  By `stream::StreamWriter::append`'s rules, checked by the
  ///              same code: a track the session has, seq above the track's
  ///              last, a key's pts no lower than the last key's.
  /// @param data  The payload; may be null when @p size is 0.
  /// @param size  The payload's bytes.
  /// @return OK, with viewers or none; `Status::Code::InvalidArgument` for a
  ///         record breaking the rules, with nothing sent;
  ///         `Status::Code::OutOfMemory` when it cannot be copied, which
  ///         drops it and every record until the next key, as an overflow
  ///         does. Never a viewer's failure.
  core::Status publish(const stream::RecordInfo& info, const std::uint8_t* data,
                       std::size_t size);

  /// @return The port listened on: the one asked for, or the one taken for 0.
  std::uint16_t port() const noexcept;

 private:
  struct Impl;
  explicit StreamServer(std::unique_ptr<Impl> impl) noexcept;
  std::unique_ptr<Impl> impl_;
};

}  // namespace volumetric_kit::recon::networking
