// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include "volumetric_kit/recon/networking/stream_client.hpp"

#include <poll.h>

#include <algorithm>
#include <condition_variable>
#include <mutex>
#include <string_view>
#include <thread>
#include <utility>

#include "socket.hpp"
#include "volumetric_kit/core/base/check.hpp"
#include "volumetric_kit/core/base/exception_status.hpp"
#include "volumetric_kit/recon/stream/chunks.hpp"

namespace volumetric_kit::recon::networking {
namespace {

using Clock = std::chrono::steady_clock;

constexpr std::string_view kClient = "StreamClient";
// What one receive reads at most: as much as the receive buffer, so one
// next(0) takes a key (about 230 KB) already in the socket.
constexpr std::size_t kReceiveBytes = detail::kReceiveBuffer;

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
    std::vector<detail::Address> addresses;
  };

  Impl(const std::string& host_, std::uint16_t port_,
       const ClientOptions& options_)
      : options(options_),
        name(host_ + ":" + std::to_string(port_)),
        api(std::string(kClient) + ": " + name),
        last_byte(Clock::now()),
        buffer(kReceiveBytes) {}

  // Make one move toward the next record, waiting no later than `deadline`:
  // the lookup's end, the next address's connect, or a receive.
  core::Status step(Clock::time_point deadline);
  // The idle limit's failure, naming the stage it ended: the lookup, the
  // connect, the session or the stream.
  core::Status idle_failure() const;
  // Close the connection and save `status`, which every later call returns.
  core::Status fail(core::Status status);

  const ClientOptions options;
  const std::string name;                  // "host:port"
  const std::string api;                   // what every message starts with
  Clock::time_point last_byte;             // connect, then each byte's arrival
  std::shared_ptr<Lookup> lookup;          // a name's, until it is taken
  std::vector<detail::Address> addresses;  // to try, in order
  std::size_t tried = 0;                   // those tried, fd's included
  core::Status refused;  // why the last address tried did not connect
  core::UniqueFd fd;
  bool connected = false;
  stream::ChunkParser parser;
  std::vector<std::uint8_t> buffer;  // what a receive reads into
  core::Status failure;
};

core::Status StreamClient::Impl::step(Clock::time_point deadline) {
  if (lookup) {
    std::unique_lock<std::mutex> lock(lookup->mutex);
    if (!lookup->done_cv.wait_until(lock, deadline,
                                    [&] { return lookup->done; })) {
      return {};
    }
    if (!lookup->failure.ok()) return lookup->failure;
    addresses = std::move(lookup->addresses);
    lock.unlock();
    lookup.reset();
    return {};
  }
  if (!fd.valid()) {
    if (tried == addresses.size()) {
      return refused.ok() ? core::Status::io_error(api + ": no address")
                          : refused;
    }
    core::Result<core::UniqueFd> started =
        detail::start_connect(api, addresses[tried++], &connected);
    if (!started.ok()) {
      refused = std::move(started).status();
      return {};
    }
    fd = std::move(started).value();
    return {};
  }
  if (!connected) {
    VKC_ASSIGN(const short ready,
               detail::wait_for(api, fd.get(), POLLOUT, until(deadline)));
    if (ready == 0) return {};
    core::Status done = detail::finish_connect(api, fd.get());
    if (!done.ok()) {
      refused = std::move(done);
      fd.reset();
      return {};
    }
    connected = true;
    return {};
  }
  VKC_ASSIGN(const short ready,
             detail::wait_for(api, fd.get(), POLLIN, until(deadline)));
  if (ready == 0) return {};
  VKC_ASSIGN(const detail::Received got,
             detail::receive_some(api, fd.get(), buffer.data(), buffer.size()));
  if (got.closed) {
    // A sender at its most viewers closes a connection before any byte.
    return core::Status::io_error(
        api + ": the sender closed the connection" +
        (parser.session() == nullptr
             ? " before its session (it may already serve its most viewers)"
             : ""));
  }
  if (got.bytes == 0) return {};
  last_byte = Clock::now();
  return parser.feed(buffer.data(), got.bytes).with_context(api);
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
    auto impl = std::make_unique<Impl>(host, port, options);
    detail::Address address;
    if (detail::parse_ipv4(host, port, &address)) {
      impl->addresses.push_back(address);
    } else {
      auto lookup = std::make_shared<Impl::Lookup>();
      std::thread([lookup, host, port, api = impl->api] {
        core::Status failure;
        std::vector<detail::Address> addresses;
        try {
          core::Result<std::vector<detail::Address>> found =
              detail::resolve(api, host, port);
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

const stream::Session* StreamClient::session() const noexcept {
  return impl_->parser.session();
}

core::Result<bool> StreamClient::next(std::chrono::milliseconds wait,
                                      stream::Record* record,
                                      std::vector<std::uint8_t>* payload) {
  VKC_CHECK(record != nullptr && payload != nullptr,
            "StreamClient::next takes a record and a payload");
  Impl& s = *impl_;
  try {
    if (!s.failure.ok()) return s.failure;
    const Clock::time_point deadline = detail::after(Clock::now(), wait);
    // At least one step, so a wait of 0 still reads what has come, and a
    // caller back late reads what came meanwhile before the idle limit is
    // judged; then the deadline ends the loop, though bytes keep coming, and
    // the parser keeps a part chunk for the next call.
    for (bool stepped = false;; stepped = true) {
      core::Result<bool> got = s.parser.next(record, payload);
      if (!got.ok()) return s.fail(std::move(got).status().with_context(s.api));
      if (*got) return true;
      const Clock::time_point idle =
          detail::after(s.last_byte, s.options.idle_limit);
      if (stepped && Clock::now() >= idle) return s.fail(s.idle_failure());
      if (stepped && Clock::now() >= deadline) return false;
      core::Status moved = s.step(std::min(deadline, idle));
      if (!moved.ok()) return s.fail(std::move(moved));
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

}  // namespace volumetric_kit::recon::networking
