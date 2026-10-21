// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include "socket.hpp"

#include <arpa/inet.h>
#include <fcntl.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <sys/socket.h>
#include <sys/types.h>  // IWYU pragma: keep (ssize_t on Linux)
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <memory>
#include <system_error>

namespace volumetric_kit::recon::networking::detail {
namespace {

#ifdef __APPLE__
constexpr int kSendFlags = 0;  // SO_NOSIGPIPE is set on every socket
#else
constexpr int kSendFlags = MSG_NOSIGNAL;
#endif

// Connections the kernel queues before the accept thread takes them.
constexpr int kBacklog = 8;
// One send or recv moves at most this much: macOS refuses a count above
// INT_MAX.
constexpr std::size_t kMaxIo = std::size_t{1} << 30;

// "<api>: <what>: <the system's reason>": OutOfMemory for the kernel's
// memory, IoError for the rest.
core::Status failure(std::string_view api, std::string_view what, int error) {
  std::string message(api);
  message += ": ";
  message += what;
  message += ": ";
  message += std::generic_category().message(error);
  if (error == ENOMEM || error == ENOBUFS) {
    return core::Status::out_of_memory(std::move(message));
  }
  return core::Status::io_error(std::move(message));
}

core::Status set_option(std::string_view api, int fd, int level, int name,
                        int value, std::string_view what) {
  if (::setsockopt(fd, level, name, &value, sizeof value) != 0) {
    return failure(api, what, errno);
  }
  return {};
}

// Close-on-exec, non-blocking, and on Apple no SIGPIPE.
core::Status prepare(std::string_view api, int fd) {
  const int fd_flags = ::fcntl(fd, F_GETFD);
  if (fd_flags < 0 || ::fcntl(fd, F_SETFD, fd_flags | FD_CLOEXEC) != 0) {
    return failure(api, "cannot set FD_CLOEXEC", errno);
  }
  const int flags = ::fcntl(fd, F_GETFL);
  if (flags < 0 || ::fcntl(fd, F_SETFL, flags | O_NONBLOCK) != 0) {
    return failure(api, "cannot set O_NONBLOCK", errno);
  }
#ifdef __APPLE__
  VKC_TRY(set_option(api, fd, SOL_SOCKET, SO_NOSIGPIPE, 1,
                     "cannot set SO_NOSIGPIPE"));
#endif
  return {};
}

core::Result<core::UniqueFd> new_socket(std::string_view api) {
  core::UniqueFd fd(::socket(AF_INET, SOCK_STREAM, 0));
  if (!fd.valid()) return failure(api, "cannot make a socket", errno);
  VKC_TRY(prepare(api, fd.get()));
  return fd;
}

const sockaddr* as_sockaddr(const Address& address) {
  return reinterpret_cast<const sockaddr*>(&address.in);
}

}  // namespace

bool parse_ipv4(const std::string& host, std::uint16_t port, Address* out) {
  Address address;
  address.in.sin_family = AF_INET;
  address.in.sin_port = htons(port);
  if (::inet_pton(AF_INET, host.c_str(), &address.in.sin_addr) != 1) {
    return false;
  }
  *out = address;
  return true;
}

core::Result<std::vector<Address>> resolve(std::string_view api,
                                           const std::string& host,
                                           std::uint16_t port) {
  addrinfo hints{};
  hints.ai_family = AF_INET;
  hints.ai_socktype = SOCK_STREAM;
  hints.ai_protocol = IPPROTO_TCP;
  hints.ai_flags = AI_NUMERICSERV;
  addrinfo* found = nullptr;
  const int error =
      ::getaddrinfo(host.c_str(), std::to_string(port).c_str(), &hints, &found);
  const int system_error = errno;  // before the message's allocations
  const std::unique_ptr<addrinfo, void (*)(addrinfo*)> owned(found,
                                                             ::freeaddrinfo);
  if (error != 0) {
    const std::string why =
        std::string(api) + ": cannot look up " + host + ": " +
        (error == EAI_SYSTEM ? std::generic_category().message(system_error)
                             : std::string(::gai_strerror(error)));
    if (error == EAI_NONAME) return core::Status::not_found(why);
    if (error == EAI_MEMORY) return core::Status::out_of_memory(why);
    return core::Status::io_error(why);
  }
  std::vector<Address> addresses;
  for (const addrinfo* a = found; a != nullptr; a = a->ai_next) {
    if (a->ai_family != AF_INET || a->ai_addrlen != sizeof(sockaddr_in)) {
      continue;
    }
    Address address;
    std::memcpy(&address.in, a->ai_addr, sizeof address.in);
    addresses.push_back(address);
  }
  if (addresses.empty()) {
    return core::Status::not_found(std::string(api) + ": " + host +
                                   " has no address for TCP");
  }
  return addresses;
}

std::string to_string(const Address& address) {
  char text[INET_ADDRSTRLEN] = {};
  ::inet_ntop(AF_INET, &address.in.sin_addr, text, sizeof text);
  return std::string(text) + ":" + std::to_string(ntohs(address.in.sin_port));
}

core::Result<core::UniqueFd> listen_on(std::string_view api,
                                       const Address& address) {
  VKC_ASSIGN(core::UniqueFd fd, new_socket(api));
  VKC_TRY(set_option(api, fd.get(), SOL_SOCKET, SO_REUSEADDR, 1,
                     "cannot set SO_REUSEADDR"));
  if (::bind(fd.get(), as_sockaddr(address), sizeof address.in) != 0 ||
      ::listen(fd.get(), kBacklog) != 0) {
    const int error = errno;  // before the message's allocations
    return failure(api, "cannot listen on " + to_string(address), error);
  }
  return fd;
}

core::Result<std::uint16_t> bound_port(std::string_view api, int fd) {
  Address address;
  socklen_t size = sizeof address.in;
  if (::getsockname(fd, reinterpret_cast<sockaddr*>(&address.in), &size) != 0) {
    return failure(api, "cannot name the socket", errno);
  }
  return ntohs(address.in.sin_port);
}

core::Result<core::UniqueFd> accept_next(std::string_view api, int listener,
                                         Address* peer) {
  for (;;) {
    *peer = {};
    socklen_t size = sizeof peer->in;
    core::UniqueFd fd(
        ::accept(listener, reinterpret_cast<sockaddr*>(&peer->in), &size));
    if (!fd.valid()) {
      switch (errno) {
        case EINTR:
          continue;
        // None waiting, or one that failed in the queue: Linux passes a
        // pending network error up through accept.
        case EAGAIN:
#if EWOULDBLOCK != EAGAIN
        case EWOULDBLOCK:
#endif
        case ECONNABORTED:
        case EPROTO:
        case ENETDOWN:
        case ENETUNREACH:
        case EHOSTUNREACH:
        case ENOPROTOOPT:
        case EOPNOTSUPP:
          return core::UniqueFd();
        default:
          return failure(api, "cannot accept", errno);
      }
    }
    VKC_TRY(prepare(api, fd.get()));
    VKC_TRY(set_option(api, fd.get(), IPPROTO_TCP, TCP_NODELAY, 1,
                       "cannot set TCP_NODELAY"));
    VKC_TRY(set_option(api, fd.get(), IPPROTO_TCP, TCP_NOTSENT_LOWAT,
                       kSendLowWater, "cannot set TCP_NOTSENT_LOWAT"));
    return fd;
  }
}

core::Result<core::UniqueFd> start_connect(std::string_view api,
                                           const Address& address,
                                           bool* connected) {
  VKC_ASSIGN(core::UniqueFd fd, new_socket(api));
  VKC_TRY(set_option(api, fd.get(), SOL_SOCKET, SO_RCVBUF, kReceiveBuffer,
                     "cannot set SO_RCVBUF"));
  *connected =
      ::connect(fd.get(), as_sockaddr(address), sizeof address.in) == 0;
  const int error = errno;  // before the message's allocations
  // Interrupted, a non-blocking connect goes on as one in progress does.
  if (!*connected && error != EINPROGRESS && error != EINTR) {
    return failure(api, "cannot connect to " + to_string(address), error);
  }
  return fd;
}

core::Status finish_connect(std::string_view api, int fd,
                            const Address& address) {
  int error = 0;
  socklen_t size = sizeof error;
  if (::getsockopt(fd, SOL_SOCKET, SO_ERROR, &error, &size) != 0) {
    error = errno;
  }
  if (error != 0) {
    return failure(api, "cannot connect to " + to_string(address), error);
  }
  return {};
}

core::Result<short> wait_for(std::string_view api, int fd, short events,
                             std::chrono::milliseconds timeout) {
  using Clock = std::chrono::steady_clock;
  const Clock::time_point deadline = Clock::now() + timeout;
  for (;;) {
    pollfd p{fd, events, 0};
    const auto left =
        std::chrono::ceil<std::chrono::milliseconds>(deadline - Clock::now());
    const int n = ::poll(
        &p, 1, static_cast<int>(std::max<std::int64_t>(left.count(), 0)));
    if (n > 0) return p.revents;
    if (n == 0) return short{0};
    if (errno != EINTR) return failure(api, "cannot poll", errno);
  }
}

core::Result<std::size_t> send_some(std::string_view api, int fd,
                                    const std::uint8_t* data,
                                    std::size_t size) {
  for (;;) {
    const ssize_t n = ::send(fd, data, std::min(size, kMaxIo), kSendFlags);
    if (n >= 0) return static_cast<std::size_t>(n);
    if (errno == EINTR) continue;
    if (errno == EAGAIN || errno == EWOULDBLOCK) return std::size_t{0};
    return failure(api, "cannot send", errno);
  }
}

core::Result<Received> receive_some(std::string_view api, int fd,
                                    std::uint8_t* out, std::size_t size) {
  for (;;) {
    const ssize_t n = ::recv(fd, out, std::min(size, kMaxIo), 0);
    if (n > 0) return Received{static_cast<std::size_t>(n), false};
    if (n == 0) return Received{0, true};
    if (errno == EINTR) continue;
    if (errno == EAGAIN || errno == EWOULDBLOCK) return Received{};
    return failure(api, "cannot receive", errno);
  }
}

}  // namespace volumetric_kit::recon::networking::detail
