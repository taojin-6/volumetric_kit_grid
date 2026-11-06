// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include "volumetric_kit/grid/networking/stream_client.hpp"

#include <poll.h>

#include <algorithm>
#include <condition_variable>
#include <mutex>
#include <optional>
#include <string_view>
#include <thread>
#include <utility>

#include "note_wire.hpp"
#include "volumetric_kit/core/base/check.hpp"
#include "volumetric_kit/core/base/exception_status.hpp"
#include "volumetric_kit/core/net/address.hpp"
#include "volumetric_kit/core/net/socket.hpp"
#include "volumetric_kit/grid/stream/chunks.hpp"

namespace volumetric_kit::grid::networking {
namespace {

namespace net = core::net;
using Clock = std::chrono::steady_clock;

constexpr std::string_view kClient = "StreamClient";
// What one receive reads at most: as much as the receive buffer, so one
// next(0) takes a key (about 230 KB) already in the socket.
constexpr std::size_t kReceiveBytes = net::kReceiveBuffer;

std::chrono::milliseconds until(Clock::time_point when) {
  return std::max(
      std::chrono::ceil<std::chrono::milliseconds>(when - Clock::now()),
      std::chrono::milliseconds{0});
}

}  // namespace

struct StreamClient::Impl {
  // A name's lookup, which a detached thread fills and the client takes: it
  // outlives whichever of the two lets it go last.
  struct Lookup {
    std::mutex mutex;
    std::condition_variable done_cv;
    bool done = false;  // under the mutex, as the two below
    core::Status failure;
    std::vector<net::Endpoint> addresses;
  };

  Impl(std::string name_, const ClientOptions& options_)
      : options(options_),
        name(std::move(name_)),
        api(std::string(kClient) + ": " + name),
        last_byte(Clock::now()),
        buffer(kReceiveBytes) {}

  // Make one move toward the next record, waiting no later than `deadline`:
  // the lookup's end, the next address's connect, or a receive. True when
  // it moved: the lookup taken, a connect started or ended, or bytes read.
  core::Result<bool> step(Clock::time_point deadline);
  // The idle limit's failure, naming the stage it ended: the lookup, the
  // connect, the session or the stream.
  core::Status idle_failure() const;
  // Close the connection and save `status`, which every later call returns.
  core::Status fail(core::Status status);
  // Frame `note` as the next chunk up, in place of one not begun, with the
  // session's chunk before the first. Throws what an allocation throws, with
  // the note it would have replaced kept.
  core::Status frame_note(const ViewerNote& note);
  // Send what the socket takes now of the chunks up, once connected. A failed
  // send leaves them, and the connection's end to the read that reports it.
  void send_up() noexcept;

  const ClientOptions options;
  const std::string name;          // "host:port"
  const std::string api;           // what every message starts with
  Clock::time_point last_byte;     // connect or adopt, then each byte's arrival
  std::shared_ptr<Lookup> lookup;  // a name's, until it is taken
  std::vector<net::Endpoint> addresses;  // to try, in order
  std::size_t tried = 0;                 // those tried, fd's included
  core::Status refused;  // why the last address tried did not connect
  core::UniqueFd fd;
  bool connected = false;
  stream::ChunkParser parser;
  std::vector<std::uint8_t> buffer;  // what a receive reads into
  core::Status failure;
  // What goes up, framed for the first note on: the chunk under way and its
  // bytes sent, then the newest note's chunk, not begun.
  std::optional<stream::ChunkEncoder> up;
  std::uint64_t up_seq = 0;
  std::vector<std::uint8_t> up_now;
  std::size_t up_sent = 0;
  std::vector<std::uint8_t> up_next;
};

core::Result<bool> StreamClient::Impl::step(Clock::time_point deadline) {
  if (lookup) {
    std::unique_lock<std::mutex> lock(lookup->mutex);
    if (!lookup->done_cv.wait_until(lock, deadline,
                                    [&] { return lookup->done; })) {
      return false;
    }
    if (!lookup->failure.ok()) return lookup->failure;
    addresses = std::move(lookup->addresses);
    lock.unlock();
    lookup.reset();
    return true;
  }
  if (!fd.valid()) {
    if (tried == addresses.size()) {
      return refused.ok() ? core::Status::io_error(api + ": no address")
                          : refused;
    }
    core::Result<core::UniqueFd> started =
        net::start_connect(api, addresses[tried++], &connected);
    if (!started.ok()) {
      refused = std::move(started).status();
      return true;
    }
    fd = std::move(started).value();
    return true;
  }
  if (!connected) {
    VKC_ASSIGN(const short ready,
               net::wait_for(api, fd.get(), POLLOUT, until(deadline)));
    if (ready == 0) return false;
    core::Status done = net::finish_connect(api, fd.get());
    if (!done.ok()) {
      refused = std::move(done);
      fd.reset();
      return true;
    }
    connected = true;
    return true;
  }
  VKC_ASSIGN(const short ready,
             net::wait_for(api, fd.get(), POLLIN, until(deadline)));
  if (ready == 0) return false;
  VKC_ASSIGN(const net::Received got,
             net::receive_some(api, fd.get(), buffer.data(), buffer.size()));
  if (got.closed) {
    // A sender at its most viewers closes a connection before any byte.
    return core::Status::io_error(
        api + ": the sender closed the connection" +
        (parser.session() == nullptr
             ? " before its session (it may already serve its most viewers)"
             : ""));
  }
  if (got.bytes == 0) return false;
  last_byte = Clock::now();
  VKC_TRY(parser.feed(buffer.data(), got.bytes).with_context(api));
  return true;
}

core::Status StreamClient::Impl::idle_failure() const {
  const std::string limit = std::to_string(options.idle_limit.count()) + " ms";
  if (lookup) {
    return core::Status::io_error(api + ": the lookup gave no answer within " +
                                  limit);
  }
  if (!connected) {
    return core::Status::io_error(api + ": cannot connect within " + limit);
  }
  if (parser.session() == nullptr) {
    return core::Status::io_error(api + ": connected, but no session within " +
                                  limit);
  }
  return core::Status::io_error(api + ": no byte for " + limit);
}

core::Status StreamClient::Impl::fail(core::Status status) {
  fd.reset();
  failure = std::move(status);
  return failure;
}

core::Status StreamClient::Impl::frame_note(const ViewerNote& note) {
  if (!up) {
    stream::Session session;
    session.writer = "viewer";
    stream::Track track;
    track.kind = stream::TrackKind::Note;
    session.tracks.push_back(std::move(track));
    VKC_ASSIGN(stream::ChunkEncoder encoder,
               stream::ChunkEncoder::create(session));
    std::vector<std::uint8_t> session_chunk = encoder.session_chunk();
    up = std::move(encoder);
    up_now = std::move(session_chunk);
  }
  const std::vector<std::uint8_t> payload = detail::encode_note(note);
  stream::RecordInfo info;
  info.key = true;
  info.seq = up_seq++;  // a seq that a failure below skips is no harm
  VKC_ASSIGN(const stream::ChunkEncoder::RecordFraming framing,
             up->frame(info, payload.data(), payload.size()));
  std::vector<std::uint8_t> chunk(framing.header.begin(), framing.header.end());
  chunk.insert(chunk.end(), payload.begin(), payload.end());
  chunk.resize(chunk.size() + framing.padding);
  up_next = std::move(chunk);
  return {};
}

void StreamClient::Impl::send_up() noexcept {
  while (connected && fd.valid()) {
    if (up_sent == up_now.size()) {
      if (up_next.empty()) return;
      up_now.swap(up_next);
      up_next.clear();
      up_sent = 0;
    }
    const core::Result<std::size_t> n = net::send_some(
        api, fd.get(), up_now.data() + up_sent, up_now.size() - up_sent);
    if (!n.ok() || *n == 0) return;
    up_sent += *n;
  }
}

StreamClient::StreamClient(std::unique_ptr<Impl> impl) noexcept
    : impl_(std::move(impl)) {}

StreamClient::~StreamClient() = default;

core::Result<std::unique_ptr<StreamClient>> StreamClient::connect(
    const std::string& host, std::uint16_t port, const ClientOptions& options) {
  constexpr std::string_view kApi = "StreamClient::connect";
  try {
    if (host.empty()) {
      return core::Status::invalid_argument(std::string(kApi) +
                                            ": an empty host");
    }
    if (port == 0) {
      return core::Status::invalid_argument(std::string(kApi) + ": port 0");
    }
    if (options.idle_limit <= std::chrono::milliseconds{0}) {
      return core::Status::invalid_argument(
          std::string(kApi) + ": an idle_limit of " +
          std::to_string(options.idle_limit.count()) + " ms");
    }
    auto impl =
        std::make_unique<Impl>(host + ":" + std::to_string(port), options);
    if (const std::optional<net::Endpoint> address =
            net::parse_ipv4(host, port)) {
      impl->addresses.push_back(*address);
    } else {
      auto lookup = std::make_shared<Impl::Lookup>();
      std::thread([lookup, host, port, api = impl->api] {
        core::Status failure;
        std::vector<net::Endpoint> addresses;
        try {
          core::Result<std::vector<net::Endpoint>> found =
              net::resolve(api, host, port);
          if (found.ok()) {
            addresses = std::move(found).value();
          } else {
            failure = std::move(found).status();
          }
        } catch (...) {
          failure = core::exception_status(api, core::Status::Code::IoError);
        }
        const std::lock_guard<std::mutex> lock(lookup->mutex);
        lookup->failure = std::move(failure);
        lookup->addresses = std::move(addresses);
        lookup->done = true;
        lookup->done_cv.notify_all();
      }).detach();
      impl->lookup = std::move(lookup);
    }
    return std::unique_ptr<StreamClient>(new StreamClient(std::move(impl)));
  } catch (...) {
    return core::exception_status(kApi, core::Status::Code::IoError);
  }
}

core::Result<std::unique_ptr<StreamClient>> StreamClient::adopt(
    core::UniqueFd connected, const ClientOptions& options) {
  constexpr std::string_view kApi = "StreamClient::adopt";
  try {
    if (!connected.valid()) {
      return core::Status::invalid_argument(std::string(kApi) +
                                            ": an empty socket");
    }
    if (options.idle_limit <= std::chrono::milliseconds{0}) {
      return core::Status::invalid_argument(
          std::string(kApi) + ": an idle_limit of " +
          std::to_string(options.idle_limit.count()) + " ms");
    }
    VKC_ASSIGN(const net::Endpoint peer, net::peer_of(kApi, connected.get()));
    VKC_TRY(net::prepare(kApi, connected.get()));
    // Connected, so next goes straight to receiving: no address to try.
    auto impl = std::make_unique<Impl>(net::to_string(peer), options);
    impl->fd = std::move(connected);
    impl->connected = true;
    return std::unique_ptr<StreamClient>(new StreamClient(std::move(impl)));
  } catch (...) {
    return core::exception_status(kApi, core::Status::Code::IoError);
  }
}

const stream::Session* StreamClient::session() const noexcept {
  return impl_->parser.session();
}

core::Result<bool> StreamClient::next(std::chrono::milliseconds wait,
                                      stream::Record* record,
                                      std::vector<std::uint8_t>* payload) {
  return next(wait, record, payload, nullptr);
}

core::Result<bool> StreamClient::next(std::chrono::milliseconds wait,
                                      stream::Record* record,
                                      std::vector<std::uint8_t>* payload,
                                      bool* progressed) {
  if (progressed != nullptr) *progressed = false;
  VKC_CHECK(record != nullptr, "StreamClient::next takes a record");
  Impl& s = *impl_;
  try {
    if (!s.failure.ok()) return s.failure;
    const Clock::time_point deadline = net::deadline_after(Clock::now(), wait);
    // At least one step, so a wait of 0 still reads what has come. The idle
    // limit is judged only after a step that moved nothing, so a caller back
    // late first takes what came meanwhile: the lookup's answer, the
    // connect's end or bytes. The deadline ends the loop, though bytes keep
    // coming, and the parser keeps a part chunk for the next call.
    for (bool stepped = false, moved = false;; stepped = true) {
      s.send_up();
      core::Result<bool> got = s.parser.next(record, payload);
      if (!got.ok()) return s.fail(std::move(got).status().with_context(s.api));
      if (*got) {
        if (progressed != nullptr) *progressed = true;
        return true;
      }
      const Clock::time_point idle =
          net::deadline_after(s.last_byte, s.options.idle_limit);
      if (stepped && !moved && Clock::now() >= idle) {
        return s.fail(s.idle_failure());
      }
      if (stepped && Clock::now() >= deadline) return false;
      core::Result<bool> progress = s.step(std::min(deadline, idle));
      if (!progress.ok()) return s.fail(std::move(progress).status());
      moved = *progress;
      if (progressed != nullptr && moved) *progressed = true;
    }
  } catch (...) {
    // Saved and returned, each by its own call: a copy may find no memory.
    s.fd.reset();
    if (s.failure.ok()) {
      s.failure = core::exception_status(s.api, core::Status::Code::IoError);
    }
    return core::exception_status(s.api, core::Status::Code::IoError);
  }
}

core::Status StreamClient::read(const stream::Record& record, std::uint8_t* dst,
                                std::size_t capacity) {
  Impl& s = *impl_;
  if (!s.failure.ok()) return s.failure;
  // A refusal leaves the record held and the connection open: the caller's
  // to correct.
  return s.parser.read(record, dst, capacity);
}

core::Status StreamClient::send_note(const ViewerNote& note) {
  constexpr std::string_view kApi = "StreamClient::send_note";
  Impl& s = *impl_;
  try {
    if (note.line.size() > kMaxNoteLineBytes) {
      return core::Status::invalid_argument(
          std::string(kApi) + ": a line of " +
          std::to_string(note.line.size()) + " bytes, above the " +
          std::to_string(kMaxNoteLineBytes) + " a note holds");
    }
    if (!s.failure.ok()) return {};
    VKC_TRY(s.frame_note(note).with_context(kApi));
    s.send_up();
    return {};
  } catch (...) {
    return core::exception_status(kApi, core::Status::Code::OutOfMemory);
  }
}

}  // namespace volumetric_kit::grid::networking
