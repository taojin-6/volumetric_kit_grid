// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// The networking tier's socket calls (src/volumetric_kit/recon/networking/
// socket.hpp) on loopback, 127.0.0.1 port 0 only: what each socket is set to,
// read back (non-blocking and close-on-exec on every one, the listener's
// SO_REUSEADDR, the accepted sender's TCP_NODELAY and 128 KiB
// TCP_NOTSENT_LOWAT, the client's 2 MiB receive buffer, SO_NOSIGPIPE on
// Apple); a send to a closed peer failing rather than raising SIGPIPE; the
// bytes a peer has not acknowledged, none, then all it has not taken in, then
// none once it has read them; a wait interrupted by a signal going on to its
// end; the end and an empty receive told apart; a deadline saturating at the
// clock's end; numeric IPv4 parsed, anything else not; and localhost resolved
// to IPv4 alone.

#include <fcntl.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <pthread.h>
#include <signal.h>
#include <sys/ioctl.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

#include "socket.hpp"
#include "test_check.hpp"

namespace vkc = volumetric_kit::core;
namespace detail = volumetric_kit::recon::networking::detail;

namespace {

using Clock = std::chrono::steady_clock;
using namespace std::chrono_literals;

int option(int fd, int level, int name) {
  int value = -1;
  socklen_t size = sizeof value;
  if (::getsockopt(fd, level, name, &value, &size) != 0) return -1;
  return value;
}

bool nonblocking_cloexec(int fd) {
  return (::fcntl(fd, F_GETFL) & O_NONBLOCK) != 0 &&
         (::fcntl(fd, F_GETFD) & FD_CLOEXEC) != 0;
}

bool no_sigpipe(int fd) {
#ifdef __APPLE__
  return option(fd, SOL_SOCKET, SO_NOSIGPIPE) != 0;
#else
  (void)fd;
  return true;  // MSG_NOSIGNAL on each send instead
#endif
}

detail::Address loopback(std::uint16_t port) {
  detail::Address a;
  (void)detail::parse_ipv4("127.0.0.1", port, &a);
  return a;
}

// A listener, a client connected to it, and the connection accepted.
struct Connected {
  vkc::UniqueFd listener;
  vkc::UniqueFd client;
  vkc::UniqueFd accepted;
  detail::Address peer;
  std::uint16_t port = 0;
};

int connect_pair(Connected* c) {
  vkc::Result<vkc::UniqueFd> listener = detail::listen_on("test", loopback(0));
  CHECK(listener.ok());
  c->listener = std::move(listener).value();
  vkc::Result<std::uint16_t> port =
      detail::bound_port("test", c->listener.get());
  CHECK(port.ok() && *port != 0);
  c->port = *port;
  bool connected = false;
  vkc::Result<vkc::UniqueFd> client =
      detail::start_connect("test", loopback(c->port), &connected);
  CHECK(client.ok());
  c->client = std::move(client).value();
  if (!connected) {
    const vkc::Result<short> ready =
        detail::wait_for("test", c->client.get(), POLLOUT, 2000ms);
    CHECK(ready.ok() && (*ready & POLLOUT) != 0);
    CHECK(detail::finish_connect("test", c->client.get()).ok());
  }
  CHECK(detail::wait_for("test", c->listener.get(), POLLIN, 2000ms).ok());
  vkc::Result<vkc::UniqueFd> accepted =
      detail::accept_next("test", c->listener.get(), &c->peer);
  CHECK(accepted.ok() && accepted->valid());
  c->accepted = std::move(accepted).value();
  return 0;
}

int settings_case() {
  Connected c;
  CHECK(connect_pair(&c) == 0);
  CHECK(nonblocking_cloexec(c.listener.get()));
  CHECK(option(c.listener.get(), SOL_SOCKET, SO_REUSEADDR) != 0);

  CHECK(nonblocking_cloexec(c.client.get()));
  CHECK(no_sigpipe(c.client.get()));
  // Linux holds what is asked for to net.core.rmem_max, then doubles it for
  // its bookkeeping.
  int asked = detail::kReceiveBuffer;
#ifdef __linux__
  std::ifstream rmem_max("/proc/sys/net/core/rmem_max");
  int most = 0;
  CHECK(rmem_max >> most);
  asked = 2 * std::min(asked, most);
#endif
  CHECK(option(c.client.get(), SOL_SOCKET, SO_RCVBUF) >= asked);

  CHECK(nonblocking_cloexec(c.accepted.get()));
  CHECK(no_sigpipe(c.accepted.get()));
  CHECK(option(c.accepted.get(), IPPROTO_TCP, TCP_NODELAY) != 0);
  CHECK(option(c.accepted.get(), IPPROTO_TCP, TCP_NOTSENT_LOWAT) ==
        detail::kSendLowWater);
  CHECK(detail::kSendLowWater == 128 << 10);
  CHECK(detail::kReceiveBuffer == 2 << 20);

  // The peer is the client, as named in the log.
  sockaddr_in local{};
  socklen_t size = sizeof local;
  CHECK(::getsockname(c.client.get(), reinterpret_cast<sockaddr*>(&local),
                      &size) == 0);
  CHECK(detail::to_string(c.peer) ==
        "127.0.0.1:" + std::to_string(ntohs(local.sin_port)));

  // None waiting: an empty fd, no error.
  detail::Address peer;
  const vkc::Result<vkc::UniqueFd> none =
      detail::accept_next("test", c.listener.get(), &peer);
  CHECK(none.ok() && !none->valid());
  return 0;
}

// Bytes, none yet, and the end, told apart; a send to a closed peer fails,
// with SIGPIPE left at its default, which would end the process.
int send_receive_case() {
  CHECK(::signal(SIGPIPE, SIG_DFL) != SIG_ERR);
  Connected c;
  CHECK(connect_pair(&c) == 0);
  std::uint8_t buffer[64];
  vkc::Result<detail::Received> got =
      detail::receive_some("test", c.client.get(), buffer, sizeof buffer);
  CHECK(got.ok() && got->bytes == 0 && !got->closed);
  const std::uint8_t hello[5] = {'h', 'e', 'l', 'l', 'o'};
  const vkc::Result<std::size_t> sent =
      detail::send_some("test", c.accepted.get(), hello, sizeof hello);
  CHECK(sent.ok() && *sent == sizeof hello);
  CHECK(detail::wait_for("test", c.client.get(), POLLIN, 2000ms).ok());
  got = detail::receive_some("test", c.client.get(), buffer, sizeof buffer);
  CHECK(got.ok() && got->bytes == sizeof hello && !got->closed);

  c.client.reset();
  CHECK(detail::wait_for("test", c.accepted.get(), POLLIN, 2000ms).ok());
  got = detail::receive_some("test", c.accepted.get(), buffer, sizeof buffer);
  CHECK(got.ok() && got->closed);
  bool failed = false;
  for (int i = 0; i < 100 && !failed; ++i) {
    const vkc::Result<std::size_t> n =
        detail::send_some("test", c.accepted.get(), hello, sizeof hello);
    failed = !n.ok();
    if (failed) CHECK(n.status().domain() == vkc::Status::Code::IoError);
    std::this_thread::sleep_for(1ms);
  }
  CHECK(failed);
  return 0;
}

// The bytes `fd` has received and not read (FIONREAD).
std::size_t unread(int fd) {
  int bytes = 0;
  if (::ioctl(fd, FIONREAD, &bytes) != 0) return 0;
  return static_cast<std::size_t>(bytes);
}

// What the peer has not acknowledged: none on a new connection; with a peer
// that does not read, what was sent less what the peer holds, and not none;
// none once the peer has read it all.
int unacknowledged_case() {
  Connected c;
  CHECK(connect_pair(&c) == 0);
  vkc::Result<std::size_t> held =
      detail::unacknowledged("test", c.accepted.get());
  CHECK(held.ok() && *held == 0);
  // Sent until the socket takes no more, even after a pause: past what the
  // peer can hold.
  const std::vector<std::uint8_t> bytes(std::size_t{1} << 20, 7);
  std::size_t sent = 0;
  const Clock::time_point end = Clock::now() + 2s;
  for (bool took = true; took && Clock::now() < end;) {
    took = false;
    for (;;) {
      const vkc::Result<std::size_t> n = detail::send_some(
          "test", c.accepted.get(), bytes.data(), bytes.size());
      CHECK(n.ok());
      if (*n == 0) break;
      sent += *n;
      took = true;
    }
    std::this_thread::sleep_for(20ms);
  }
  // The peer may acknowledge late: wait until the two sides agree.
  const Clock::time_point settled = Clock::now() + 2s;
  for (;;) {
    held = detail::unacknowledged("test", c.accepted.get());
    CHECK(held.ok());
    if (*held + unread(c.client.get()) == sent || Clock::now() >= settled) {
      break;
    }
    std::this_thread::sleep_for(10ms);
  }
  CHECK(*held > 0 && *held + unread(c.client.get()) == sent);

  std::vector<std::uint8_t> buffer(std::size_t{1} << 16);
  std::size_t taken = 0;
  const Clock::time_point drained = Clock::now() + 5s;
  while (taken < sent && Clock::now() < drained) {
    CHECK(detail::wait_for("test", c.client.get(), POLLIN, 100ms).ok());
    const vkc::Result<detail::Received> got = detail::receive_some(
        "test", c.client.get(), buffer.data(), buffer.size());
    CHECK(got.ok() && !got->closed);
    taken += got->bytes;
  }
  CHECK(taken == sent);
  const Clock::time_point acked = Clock::now() + 2s;
  do {
    held = detail::unacknowledged("test", c.accepted.get());
    CHECK(held.ok());
  } while (*held != 0 && Clock::now() < acked);
  CHECK(*held == 0);
  return 0;
}

void nothing(int) {}

// A wait a signal interrupts (a handler without SA_RESTART, as poll is never
// restarted) goes on until its time is up.
int interrupted_wait_case() {
  struct sigaction action{};
  action.sa_handler = nothing;
  sigemptyset(&action.sa_mask);
  struct sigaction before{};
  CHECK(::sigaction(SIGUSR1, &action, &before) == 0);
  Connected c;
  CHECK(connect_pair(&c) == 0);
  const pthread_t waiter = ::pthread_self();
  std::thread signaller([waiter] {
    for (int i = 0; i < 5; ++i) {
      std::this_thread::sleep_for(30ms);
      ::pthread_kill(waiter, SIGUSR1);
    }
  });
  const Clock::time_point start = Clock::now();
  const vkc::Result<short> ready =
      detail::wait_for("test", c.client.get(), POLLIN, 300ms);
  const auto took = Clock::now() - start;
  signaller.join();
  CHECK(::sigaction(SIGUSR1, &before, nullptr) == 0);
  CHECK(ready.ok() && *ready == 0);
  CHECK(took >= 290ms);
  return 0;
}

// A deadline saturates: milliseconds::max() is the clock's end, as is any
// limit past it; one of 0 or less is now.
int after_case() {
  const Clock::time_point now = Clock::now();
  CHECK(detail::after(now, 10ms) == now + 10ms);
  CHECK(detail::after(now, 0ms) == now);
  CHECK(detail::after(now, -5ms) == now);
  CHECK(detail::after(now, std::chrono::milliseconds::max()) ==
        Clock::time_point::max());
  CHECK(detail::after(Clock::time_point::max() - 1ms, 2ms) ==
        Clock::time_point::max());
  return 0;
}

int addresses_case() {
  detail::Address a;
  CHECK(detail::parse_ipv4("192.168.21.217", 47801, &a));
  CHECK(detail::to_string(a) == "192.168.21.217:47801");
  for (const char* host : {"", "localhost", "1.2.3", "::1", "256.1.1.1"}) {
    CHECK(!detail::parse_ipv4(host, 1, &a));
  }
  // IPv4 alone, though localhost names ::1 too.
  const vkc::Result<std::vector<detail::Address>> found =
      detail::resolve("test", "localhost", 9);
  CHECK(found.ok() && !found->empty());
  for (const detail::Address& address : *found) {
    CHECK(address.in.sin_family == AF_INET);
    CHECK(detail::to_string(address) == "127.0.0.1:9");
  }
  // A port in use is refused, though SO_REUSEADDR is set.
  Connected c;
  CHECK(connect_pair(&c) == 0);
  const vkc::Result<vkc::UniqueFd> taken =
      detail::listen_on("test", loopback(c.port));
  CHECK(!taken.ok() && taken.status().domain() == vkc::Status::Code::IoError);
  return 0;
}

}  // namespace

int main() {
  if (settings_case() != 0) return 1;
  if (send_receive_case() != 0) return 1;
  if (unacknowledged_case() != 0) return 1;
  if (interrupted_wait_case() != 0) return 1;
  if (after_case() != 0) return 1;
  if (addresses_case() != 0) return 1;
  std::puts("networking socket: OK");
  return 0;
}
