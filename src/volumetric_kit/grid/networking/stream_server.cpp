// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include "volumetric_kit/grid/networking/stream_server.hpp"

#include <poll.h>

#include <atomic>
#include <condition_variable>
#include <list>
#include <mutex>
#include <optional>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#include "gop_log.hpp"
#include "socket.hpp"
#include "volumetric_kit/core/base/exception_status.hpp"
#include "volumetric_kit/grid/base/log.hpp"
#include "volumetric_kit/grid/stream/chunks.hpp"

namespace volumetric_kit::grid::networking {
namespace {

using Clock = std::chrono::steady_clock;

constexpr std::string_view kServer = "StreamServer";
constexpr std::uint8_t kZeros[8] = {};
// What a viewer thread reads at a time of the bytes a viewer sends, which it
// drops.
constexpr std::size_t kDrainBytes = 4096;
// The least time between two refusals logged at Warning; the rest go at Debug.
constexpr std::chrono::seconds kRefusalWarningEvery{10};

// grid's log_message, from a thread or a catch no exception may leave.
// Viewers come and go at Info, which the core's default sink drops and a
// sender prints, as serve_stream and fuse_orbbec do.
void log_at(core::LogLevel level, const std::string& message) noexcept {
  try {
    log_message(level, message);
  } catch (...) {
  }
}

}  // namespace

struct StreamServer::Impl {
  // One viewer: its socket and its thread.
  struct Viewer {
    core::UniqueFd fd;  // its thread's, which closes it
    std::string name;   // its address, for the log
    bool done = false;  // under the mutex: its thread has ended
    // Under the mutex: its place in the log, which it joins as its thread
    // starts and leaves as it ends.
    detail::GopLog::Cursor cursor;
    std::thread thread;
  };

  Impl(stream::ChunkEncoder encoder_, const ServerOptions& options_,
       detail::Chunk session_chunk_, core::UniqueFd listener_,
       std::uint16_t port_)
      : encoder(std::move(encoder_)),
        options(options_),
        session_chunk(std::move(session_chunk_)),
        listener(std::move(listener_)),
        port(port_),
        log(options_.max_gop_bytes, options_.replay_gop) {}

  // The accept thread: take each connection, admit it or close it, and join
  // the threads of viewers that have ended, until the stop.
  void accept_loop() noexcept;
  // Count `fd`'s viewer in and start its thread, or close it past
  // max_viewers.
  void admit(core::UniqueFd fd, std::string name);
  // Join the threads of the viewers that have ended.
  void reap();
  // A viewer's thread: join the log, send it everything, then close it, leave
  // the log and say why.
  void serve(Viewer& viewer) noexcept;
  // Send the session chunk, then the log's chunks as the socket takes them,
  // until the viewer leaves, fails or stalls, or the server stops: why.
  std::string send_all(Viewer& viewer);
  // Publish the log's count of the viewers that wanted a key. Under the
  // mutex.
  void count_key_wanted() noexcept { key_wanted = log.key_wanted(); }

  stream::ChunkEncoder encoder;  // the publishing thread's
  const ServerOptions options;
  const detail::Chunk session_chunk;
  const core::UniqueFd listener;
  const std::uint16_t port;

  std::mutex mutex;
  std::condition_variable wake;   // a record published, or the stop
  detail::GopLog log;             // under the mutex
  std::atomic<bool> stop{false};  // set under the mutex
  // The log's key_wanted(), set under the mutex and read from any thread.
  std::atomic<std::uint64_t> key_wanted{0};
  // The viewers being served: changed under the mutex, read from any thread.
  std::atomic<std::size_t> live{0};

  std::list<Viewer> viewers;  // the accept thread's, then the destructor's
  // The accept thread's: when a refusal was last logged at Warning.
  std::optional<Clock::time_point> refusal_warned;
  std::thread acceptor;
};

void StreamServer::Impl::accept_loop() noexcept {
  while (!stop) {
    try {
      reap();
      const core::Result<short> ready =
          detail::wait_for(kServer, listener.get(), POLLIN, detail::kPollSlice);
      if (ready.ok() && *ready == 0) continue;
      detail::Address peer;
      core::Result<core::UniqueFd> accepted =
          ready.ok() ? detail::accept_next(kServer, listener.get(), &peer)
                     : core::Result<core::UniqueFd>(ready.status());
      if (!accepted.ok()) {
        // A failure that may pass (out of descriptors): say so, wait a
        // slice and try again.
        log_at(core::LogLevel::Warning, accepted.status().message());
        std::this_thread::sleep_for(detail::kPollSlice);
        continue;
      }
      if (accepted->valid()) {
        admit(std::move(accepted).value(), detail::to_string(peer));
      }
    } catch (...) {
      log_at(core::LogLevel::Warning,
             core::exception_status(kServer, core::Status::Code::IoError)
                 .message());
      std::this_thread::sleep_for(detail::kPollSlice);
    }
  }
}

void StreamServer::Impl::admit(core::UniqueFd fd, std::string name) {
  std::size_t serving = 0;
  {
    // Only this thread adds viewers, so one admitted after this check still
    // fits.
    const std::lock_guard<std::mutex> lock(mutex);
    serving = live;
  }
  if (serving >= options.max_viewers) {
    fd.reset();  // closed at accept: the viewer sees its end before any byte
    // At Warning, which the default sink prints, so the owner sees why a
    // viewer gets nothing: the first refusal, then one in 10 s at most. The
    // rest at Debug, so a sender that prints Info is not flooded either.
    const Clock::time_point now = Clock::now();
    const bool warn =
        !refusal_warned || now - *refusal_warned >= kRefusalWarningEvery;
    if (warn) refusal_warned = now;
    log_at(warn ? core::LogLevel::Warning : core::LogLevel::Debug,
           std::string(kServer) + ": refused viewer " + name + ": " +
               std::to_string(serving) + " viewers already");
    return;
  }
  // Nothing from here to its thread's start throws but that start, which is
  // undone: a viewer counted with no thread would keep its slot and its
  // connection for the server's life.
  Viewer& viewer = viewers.emplace_back();
  viewer.fd = std::move(fd);
  viewer.name = std::move(name);
  {
    const std::lock_guard<std::mutex> lock(mutex);
    ++live;
    serving = live;
  }
  try {
    log_at(core::LogLevel::Info, std::string(kServer) + ": viewer " +
                                     viewer.name + " joined, " +
                                     std::to_string(serving) + " of " +
                                     std::to_string(options.max_viewers));
  } catch (...) {
  }
  try {
    viewer.thread = std::thread([this, &viewer] { serve(viewer); });
  } catch (...) {
    {
      const std::lock_guard<std::mutex> lock(mutex);
      --live;
    }
    viewers.pop_back();
    throw;
  }
}

void StreamServer::Impl::reap() {
  for (auto it = viewers.begin(); it != viewers.end();) {
    bool done = false;
    {
      const std::lock_guard<std::mutex> lock(mutex);
      done = it->done;
    }
    if (done) {
      it->thread.join();
      it = viewers.erase(it);
    } else {
      ++it;
    }
  }
}

void StreamServer::Impl::serve(Viewer& viewer) noexcept {
  std::string why;
  try {
    {
      // Counted before its first byte, so a live sender's key is on its way
      // while the session chunk goes.
      const std::lock_guard<std::mutex> lock(mutex);
      log.join(&viewer.cursor);
      count_key_wanted();
    }
    why = send_all(viewer);
  } catch (...) {
    try {
      why = "failed: " +
            core::exception_status("sending", core::Status::Code::IoError)
                .message();
    } catch (...) {
    }
  }
  viewer.fd.reset();
  {
    const std::lock_guard<std::mutex> lock(mutex);
    log.leave(&viewer.cursor);
    viewer.done = true;
    --live;
  }
  try {
    log_at(core::LogLevel::Info,
           std::string(kServer) + ": viewer " + viewer.name + " " + why);
  } catch (...) {
  }
}

std::string StreamServer::Impl::send_all(Viewer& viewer) {
  const std::string api = std::string(kServer) + ": viewer " + viewer.name;
  const int fd = viewer.fd.get();
  detail::Chunk chunk = session_chunk;  // the chunk under way; null between
  std::size_t sent = 0;                 // its bytes sent
  // Its place in the log, which it enters after the session chunk at the
  // GOP's key, or at the next key, waited for as it joined: the log keeps
  // every record since that it still needs.
  detail::GopLog::Cursor& cursor = viewer.cursor;
  // When the viewer last took a byte, or its socket could take one.
  Clock::time_point taking = Clock::now();
  std::uint64_t written = 0;  // the bytes sent on the socket
  std::uint64_t taken = 0;    // of them, the ones the viewer acknowledged
  std::uint8_t drain[kDrainBytes];
  for (;;) {
    if (stop) return "closed: the server stopped";
    const core::Result<short> ready =
        detail::wait_for(api, fd, POLLIN | POLLOUT, detail::kPollSlice);
    if (!ready.ok()) return "failed: " + ready.status().message();
    if ((*ready & (POLLIN | POLLHUP | POLLERR)) != 0) {
      // What a viewer sends is dropped; its end ends it.
      const core::Result<detail::Received> got =
          detail::receive_some(api, fd, drain, sizeof drain);
      if (!got.ok()) return "failed: " + got.status().message();
      if (got->closed) return "left";
    }
    if ((*ready & POLLOUT) == 0) {
      // Under the low-water mark the socket polls writable, so this one
      // holds at least that much; the viewer takes what it acknowledges,
      // which a slow one does above the mark.
      const core::Result<std::size_t> held = detail::unacknowledged(api, fd);
      if (!held.ok()) return "failed: " + held.status().message();
      if (*held < written - taken) {
        taken = written - *held;
        taking = Clock::now();
      } else if (Clock::now() >= detail::after(taking, options.stall_limit)) {
        return "closed: took no byte for " +
               std::to_string(options.stall_limit.count()) + " ms";
      }
      continue;
    }
    taking = Clock::now();
    if (!chunk) {
      // A chunk boundary: the next chunk, from a newer key if one has come
      // (replaying), or from the next key while it waits (keys on demand).
      std::unique_lock<std::mutex> lock(mutex);
      chunk = log.next(&cursor);
      if (!chunk) {
        wake.wait_for(lock, detail::kPollSlice);
        continue;
      }
      sent = 0;
    }
    const core::Result<std::size_t> n =
        detail::send_some(api, fd, chunk->data() + sent, chunk->size() - sent);
    if (!n.ok()) return "failed: " + n.status().message();
    sent += *n;
    written += *n;
    if (sent == chunk->size()) chunk = nullptr;
  }
}

StreamServer::StreamServer(std::unique_ptr<Impl> impl) noexcept
    : impl_(std::move(impl)) {}

StreamServer::~StreamServer() {
  Impl& s = *impl_;
  {
    const std::lock_guard<std::mutex> lock(s.mutex);
    s.stop = true;
  }
  s.wake.notify_all();
  if (s.acceptor.joinable()) s.acceptor.join();
  for (Impl::Viewer& viewer : s.viewers) {
    if (viewer.thread.joinable()) viewer.thread.join();
  }
}

core::Result<std::unique_ptr<StreamServer>> StreamServer::start(
    const stream::Session& session, const ServerOptions& options) {
  constexpr std::string_view kApi = "StreamServer::start";
  try {
    core::Result<stream::ChunkEncoder> encoder =
        stream::ChunkEncoder::create(session);
    if (!encoder.ok()) return std::move(encoder).status().with_context(kApi);
    if (session.tracks.size() > 1) {
      return core::Status::unsupported(std::string(kApi) + ": a session of " +
                                       std::to_string(session.tracks.size()) +
                                       " tracks; one is served");
    }
    if (options.max_viewers == 0) {
      return core::Status::invalid_argument(std::string(kApi) +
                                            ": a max_viewers of 0");
    }
    if (options.stall_limit <= std::chrono::milliseconds{0}) {
      return core::Status::invalid_argument(
          std::string(kApi) + ": a stall_limit of " +
          std::to_string(options.stall_limit.count()) + " ms");
    }
    detail::Address address;
    if (!detail::parse_ipv4(options.host.empty() ? "0.0.0.0" : options.host,
                            options.port, &address)) {
      return core::Status::invalid_argument(std::string(kApi) + ": host \"" +
                                            options.host +
                                            "\" is not a numeric IPv4 address");
    }
    VKC_ASSIGN(core::UniqueFd listener, detail::listen_on(kApi, address));
    VKC_ASSIGN(const std::uint16_t port,
               detail::bound_port(kApi, listener.get()));
    auto session_chunk = std::make_shared<const std::vector<std::uint8_t>>(
        encoder->session_chunk());
    std::unique_ptr<StreamServer> server(
        new StreamServer(std::make_unique<Impl>(
            std::move(encoder).value(), options, std::move(session_chunk),
            std::move(listener), port)));
    Impl* const s = server->impl_.get();
    s->acceptor = std::thread([s] { s->accept_loop(); });
    return server;
  } catch (...) {
    return core::exception_status(kApi, core::Status::Code::IoError);
  }
}

core::Status StreamServer::publish(const stream::RecordInfo& info,
                                   const std::uint8_t* data, std::size_t size) {
  constexpr std::string_view kApi = "StreamServer::publish";
  try {
    if (data == nullptr && size != 0) {
      return core::Status::invalid_argument(std::string(kApi) +
                                            ": a null payload of " +
                                            std::to_string(size) + " bytes");
    }
    return publish(info, stream::RecordPayload::of(data, size));
  } catch (...) {
    return core::exception_status(kApi, core::Status::Code::OutOfMemory);
  }
}

core::Status StreamServer::publish(const stream::RecordInfo& info,
                                   const stream::RecordPayload& payload) {
  constexpr std::string_view kApi = "StreamServer::publish";
  Impl& s = *impl_;
  try {
    const std::size_t size = payload.size();
    if (size > stream::ChunkParser::kMaxBodyBytes) {
      return core::Status::invalid_argument(
          std::string(kApi) + ": a payload of " + std::to_string(size) +
          " bytes, above the " +
          std::to_string(stream::ChunkParser::kMaxBodyBytes) +
          " a viewer reads");
    }
    core::Result<stream::ChunkEncoder::RecordFraming> framed =
        s.encoder.frame(info, payload);
    if (!framed.ok()) return std::move(framed).status().with_context(kApi);
    const stream::ChunkEncoder::RecordFraming& framing = *framed;
    const std::size_t bytes = framing.header.size() + size + framing.padding;
    bool kept = false;
    bool overflow = false;
    std::uint64_t fell_back = 0;  // viewers that now wait for a key
    try {
      // The one copy, made before the lock, of both runs.
      auto chunk = std::make_shared<std::vector<std::uint8_t>>();
      chunk->reserve(bytes);
      chunk->insert(chunk->end(), framing.header.begin(), framing.header.end());
      if (payload.first_size() != 0) {
        chunk->insert(chunk->end(), payload.first(),
                      payload.first() + payload.first_size());
      }
      if (payload.second_size() != 0) {
        chunk->insert(chunk->end(), payload.second(),
                      payload.second() + payload.second_size());
      }
      chunk->insert(chunk->end(), kZeros, kZeros + framing.padding);
      const std::lock_guard<std::mutex> lock(s.mutex);
      // Whether a viewer would take it: replaying, a key or a record of an
      // open GOP; on demand, a key while a viewer has joined, or a record
      // while one is in.
      const bool joins =
          info.key ? s.options.replay_gop || s.log.joined() : s.log.open();
      const std::uint64_t wanted = s.log.key_wanted();
      kept = s.log.append(std::move(chunk), info.key);
      // Replaying, the GOP went past max_gop_bytes; on demand, the record
      // alone is past it, so no viewer can take it.
      overflow =
          joins &&
          (s.options.replay_gop ? !kept : bytes > s.options.max_gop_bytes);
      fell_back = s.log.key_wanted() - wanted;
      s.count_key_wanted();
    } catch (...) {
      // Lost, so the records after it until the next key are too.
      try {
        const std::lock_guard<std::mutex> lock(s.mutex);
        s.log.break_gop(info.key);
        s.count_key_wanted();
      } catch (...) {
      }
      return core::exception_status(kApi, core::Status::Code::OutOfMemory);
    }
    if (kept) s.wake.notify_all();
    if (overflow) {
      log_at(core::LogLevel::Warning,
             std::string(kApi) + ": a " +
                 (s.options.replay_gop
                      ? std::string("GOP")
                      : "record of " + std::to_string(bytes) + " bytes,") +
                 " past max_gop_bytes (" +
                 std::to_string(s.options.max_gop_bytes) +
                 "): the viewers wait for the next key");
    } else if (fell_back != 0) {
      // At Info, as viewers come and go: the sender's next key brings it
      // back.
      log_at(core::LogLevel::Info,
             std::string(kApi) + ": " + std::to_string(fell_back) +
                 (fell_back == 1 ? " viewer" : " viewers") +
                 " past max_gop_bytes (" +
                 std::to_string(s.options.max_gop_bytes) +
                 ") behind: waiting for the next key");
    }
    return {};
  } catch (...) {
    return core::exception_status(kApi, core::Status::Code::OutOfMemory);
  }
}

std::uint16_t StreamServer::port() const noexcept { return impl_->port; }

std::uint64_t StreamServer::key_wanted() const noexcept {
  return impl_->key_wanted.load();
}

std::size_t StreamServer::viewers() const noexcept {
  return impl_->live.load();
}

}  // namespace volumetric_kit::grid::networking
