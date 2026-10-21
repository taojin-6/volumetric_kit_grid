// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// The networking tier's socket calls (src/volumetric_kit/recon/networking/
// socket.hpp) on loopback, 127.0.0.1 port 0 only: what each socket is set to,
// read back (non-blocking and close-on-exec on every one, the listener's
// SO_REUSEADDR, the accepted sender's TCP_NODELAY and 128 KiB
// TCP_NOTSENT_LOWAT, the client's 256 KiB receive buffer, SO_NOSIGPIPE on
// Apple); a send to a closed peer failing rather than raising SIGPIPE; a
// wait interrupted by a signal going on to its end; the end and an empty
// receive told apart; numeric IPv4 parsed, anything else not; and localhost
// resolved to IPv4 alone.

#include <fcntl.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <pthread.h>
#include <signal.h>
#include <sys/socket.h>
#include <unistd.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
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
    CHECK(detail::finish_connect("test", c->client.get(), loopback(c->port))
              .ok());
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
  // Linux doubles what is asked for, for its bookkeeping.
  CHECK(option(c.client.get(), SOL_SOCKET, SO_RCVBUF) >=
        detail::kReceiveBuffer);

  CHECK(nonblocking_cloexec(c.accepted.get()));
  CHECK(no_sigpipe(c.accepted.get()));
  CHECK(option(c.accepted.get(), IPPROTO_TCP, TCP_NODELAY) != 0);
  CHECK(option(c.accepted.get(), IPPROTO_TCP, TCP_NOTSENT_LOWAT) ==
        detail::kSendLowWater);
  CHECK(detail::kSendLowWater == 128 << 10);
  CHECK(detail::kReceiveBuffer == 256 << 10);

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
  if (interrupted_wait_case() != 0) return 1;
  if (addresses_case() != 0) return 1;
  std::puts("networking socket: OK");
  return 0;
}
