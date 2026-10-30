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
#include "volumetric_kit/grid/networking/export.hpp"
#include "volumetric_kit/grid/stream/chunks.hpp"
#include "volumetric_kit/grid/stream/session.hpp"

namespace volumetric_kit::grid::networking {

/// The port a sender serves on, and a viewer connects to when none is named.
inline constexpr std::uint16_t kDefaultPort = 47801;

/// How long @ref StreamServer::finish lets the viewers take the records
/// queued for them before it closes the ones still behind: 1 s.
inline constexpr std::chrono::milliseconds kFinishLimit{1'000};

/// @brief Where and how a @ref StreamServer serves.
struct ServerOptions {
  /// A numeric IPv4 address to listen on; empty listens on every interface.
  std::string host;
  /// The port; 0 takes a free one, which @ref StreamServer::port names.
  std::uint16_t port = kDefaultPort;
  /// Viewers served at once, first come: a connection past them is closed at
  /// accept, the first such refusal and then one in 10 s at most logged at
  /// Warning, the rest at Debug. With no authentication, any host reaching
  /// the port can take a slot, and keeps it until it leaves or stalls
  /// (@ref stall_limit).
  std::size_t max_viewers = 4;
  /// With @ref replay_gop, the most one GOP (a key record and every record
  /// after it) may hold: a record that would pass it is dropped with the
  /// whole GOP, and so is every record until the next key, the viewers frozen
  /// until it. Without, how far behind the newest record a viewer may fall:
  /// one that a record would take past it waits for the next key, alone, and
  /// a record past it on its own goes to none, logged at Warning.
  std::uint64_t max_gop_bytes = std::uint64_t{64} << 20;
  /// A viewer that takes no byte for this long, with bytes waiting for it, is
  /// closed: a peer that vanished. A byte its end acknowledges is taken, so a
  /// slow reader keeps its slot while its end acknowledges a byte within the
  /// limit (one behind a large receive buffer may not), as does one with
  /// nothing waiting (before the first key, or while nothing is published).
  /// Positive; `milliseconds::max()` is never.
  std::chrono::milliseconds stall_limit{10'000};
  /// true: a joiner gets the GOP since the newest key, which the server
  /// keeps, so a sender that cannot make a key, as one replaying a file,
  /// still serves a joiner at once. false, for a sender that makes a key when
  /// @ref StreamServer::key_wanted rises: a joiner waits for the next key,
  /// and the log keeps only what connected viewers still need, at most
  /// @ref max_gop_bytes behind; a viewer that would fall further waits for
  /// the next key. Newest key wins either way.
  bool replay_gop = true;
};

/// @brief Serves one session to viewers over TCP as it is recorded: each
///        gets the session header, the records from the newest key on, or
///        from the next (@ref ServerOptions::replay_gop), then every record
///        as it is published (a @ref StreamClient reads them).
///
/// **The wire is the file.** A viewer receives the `VRST` chunk, then `RECD`
/// chunks byte for byte as `stream::StreamWriter` writes them, with no footer
/// (`stream::ChunkEncoder`).
///
/// A viewer takes its next record only while its socket holds under 128 KiB
/// unsent (`TCP_NOTSENT_LOWAT`), so little backlog hides in the kernel. A
/// P-frame is never sent without the records it is predicted from.
///
/// **Track 0 anchors the GOP.** A key on track 0 starts a GOP; every other
/// track's records are keys, each joining the GOP it is published in as a
/// P-frame does, so they share track 0's fate: a viewer that joins or skips
/// starts at a track-0 key and gets every record published after it, on any
/// track, and loses what was published before it.
///
/// **Newest key wins.** When a newer key arrives before a viewer has sent
/// the records before it, the viewer finishes the chunk under way, skips the
/// rest and goes on from that key, so one slower than the stream stays
/// within about a GOP of it.
///
/// **Joining at the GOP** (@ref ServerOptions::replay_gop, the default). The
/// server keeps one GOP: the newest key record and every record after it. A
/// viewer joins at that key, so it decodes at once. A viewer's first record,
/// after its `VRST` chunk, is the GOP's key, and the server keeps every
/// record since. So once a viewer has the session, it misses no record
/// published after, except by newest key wins.
///
/// **Keys on demand** (`replay_gop` false). A viewer joins waiting for the
/// next key, and from it gets every record, in order, except by newest key
/// wins, while it stays within @ref ServerOptions::max_gop_bytes of the
/// newest. One that a record would take further behind finishes the chunk
/// under way and waits for the next key again. The server keeps only the
/// records a viewer still has to send. Each join and each fall-back counts in
/// @ref key_wanted, so a live sender makes its next frame a key and the
/// viewer waits one frame.
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
/// @warning Call @ref publish and @ref finish from one thread at a time.
class VGRID_NETWORKING_API StreamServer {
 public:
  /// @brief Listen on @p options' address and start accepting viewers.
  /// @param session  What each viewer gets first: a session
  ///                 `stream::StreamWriter::create` takes, its track 0 the
  ///                 one whose keys start a GOP.
  /// @param options  Where and how it serves.
  /// @return The server; `Status::Code::InvalidArgument` for a session the
  ///         writer would refuse, a host that is not numeric IPv4, a
  ///         `max_viewers` of 0 or a `stall_limit` of 0 or less;
  ///         `Status::Code::IoError` for a port that cannot be
  ///         bound (in use) or a thread that cannot start;
  ///         `Status::Code::OutOfMemory` when the session's chunk cannot be
  ///         allocated.
  static core::Result<std::unique_ptr<StreamServer>> start(
      const stream::Session& session, const ServerOptions& options = {});

  /// @brief Close every connection at once, the records still queued for a
  ///        viewer lost (@ref finish sends them first), and join every
  ///        thread: within one poll slice (about 20 ms) and the write under
  ///        way.
  ~StreamServer();
  StreamServer(const StreamServer&) = delete;
  StreamServer& operator=(const StreamServer&) = delete;
  StreamServer(StreamServer&&) = delete;
  StreamServer& operator=(StreamServer&&) = delete;

  /// @brief Hand one record to every viewer, now or as each catches up.
  /// @param info  By `stream::StreamWriter::append`'s rules, checked by the
  ///              same code: a track the session has, seq above the track's
  ///              last, a key's pts no lower than the track's last key's;
  ///              and a key unless on track 0.
  /// @param data  The payload; may be null when @p size is 0.
  /// @param size  The payload's bytes.
  /// @return OK, with viewers or none; `Status::Code::InvalidArgument` for a
  ///         record breaking the rules, a payload over
  ///         `stream::ChunkParser::kMaxBodyBytes` (64 MiB), which no viewer
  ///         reads, or a call after @ref finish, with nothing sent;
  ///         `Status::Code::OutOfMemory` when it cannot be copied, which
  ///         drops it and every record until the next key, as an overflow
  ///         does. Never a viewer's failure.
  core::Status publish(const stream::RecordInfo& info, const std::uint8_t* data,
                       std::size_t size);

  /// @brief @ref publish a payload in up to two runs: its one copy made from
  ///        both, its CRC the one @p payload carries, not computed again.
  /// @return As the other @ref publish.
  core::Status publish(const stream::RecordInfo& info,
                       const stream::RecordPayload& payload);

  /// @brief End the stream: take no more records or viewers, and close each
  ///        viewer once it has sent every record queued for it, so it reads
  ///        the end right after the last record published.
  ///
  /// Returns once every viewer is closed and every thread joined: within
  /// @ref kFinishLimit, a poll slice and the write under way, however the
  /// viewers read. A viewer with nothing queued, as one waiting for a key,
  /// is closed at once, and so is one that leaves or fails; one that stalls
  /// past @ref ServerOptions::stall_limit is closed as ever; one still
  /// behind at @ref kFinishLimit is closed then, logged at Info. A
  /// connection meanwhile is closed at accept, before any byte, and one
  /// after is refused. Later calls do nothing.
  void finish() noexcept;

  /// @return The port listened on: the one asked for, or the one taken for 0.
  std::uint16_t port() const noexcept;

  /// @brief Viewers that joined or fell back to waiting for a key: without
  ///        @ref ServerOptions::replay_gop, each viewer counted as it joins,
  ///        and again each time it falls back, past `max_gop_bytes` behind or
  ///        when a record is lost (`publish`'s `OutOfMemory`), a waiting
  ///        viewer too when the lost record is a key. A key within
  ///        `max_gop_bytes` counts none, as every viewer goes on from it. A
  ///        sender that sees it rise makes its next frame a key. Safe from
  ///        any thread.
  /// @return The count so far, which never falls; always 0 with
  ///         `replay_gop`, whose joiners start at the GOP's key.
  std::uint64_t key_wanted() const noexcept;

  /// @brief The viewers being served now: each counted from its accepted
  ///        connection, before its first byte, until the server closes it:
  ///        as it leaves, fails or stalls past
  ///        @ref ServerOptions::stall_limit, or at @ref finish.
  ///        A viewer waiting for a key counts; a connection refused at
  ///        @ref ServerOptions::max_viewers never does. Safe from any thread.
  /// @return The count, at most `max_viewers`.
  std::size_t viewers() const noexcept;

 private:
  struct Impl;
  explicit StreamServer(std::unique_ptr<Impl> impl) noexcept;
  std::unique_ptr<Impl> impl_;
};

}  // namespace volumetric_kit::grid::networking
