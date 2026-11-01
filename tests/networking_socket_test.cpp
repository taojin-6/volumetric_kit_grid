// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// The networking tier's socket calls (src/volumetric_kit/grid/networking/
// socket.hpp) on loopback, 127.0.0.1 port 0, but for one connection on Linux
// from the host's own address: what each socket is set to, read back
// (non-blocking and close-on-exec on every one, the listener's SO_REUSEADDR,
// the accepted sender's TCP_NODELAY and 128 KiB TCP_NOTSENT_LOWAT, the
// client's 2 MiB receive buffer, SO_NOSIGPIPE on Apple); a send to a closed
// peer failing rather than raising SIGPIPE; the bytes a peer has not
// acknowledged, none, then all it has not taken in, then none once it has read
// them; a wait interrupted by a signal going on to its end; the end and an
// empty receive told apart; a deadline saturating at the clock's end; numeric
// IPv4 parsed, anything else not; localhost resolved to IPv4 alone; and the
// congestion control: the allowed list read word by word and 127.0.0.0/8 told
// from other addresses, on every platform, and on Linux a socket accepted from
// the host's own address that is not loopback using BBR exactly when /proc
// lists it, one from loopback and one asked with a list that lacks it keeping
// the default, said once a process.

#include <fcntl.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <pthread.h>
#include <signal.h>
#include <sys/ioctl.h>
#include <sys/socket.h>
#include <unistd.h>
#ifdef __linux__
#include <ifaddrs.h>
#include <net/if.h>
#endif

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include "socket.hpp"
#include "volumetric_kit/core/base/log.hpp"
#include "volumetric_kit/core/testing/test_check.hpp"

namespace vkc = volumetric_kit::core;
namespace detail = volumetric_kit::grid::networking::detail;

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

// On `host`'s address, port 0, loopback by default.
int connect_pair(Connected* c, detail::Address host = loopback(0)) {
  vkc::Result<vkc::UniqueFd> listener = detail::listen_on("test", host);
  CHECK(listener.ok());
  c->listener = std::move(listener).value();
  vkc::Result<std::uint16_t> port =
      detail::bound_port("test", c->listener.get());
  CHECK(port.ok() && *port != 0);
  c->port = *port;
  host.in.sin_port = htons(c->port);
  bool connected = false;
  vkc::Result<vkc::UniqueFd> client =
      detail::start_connect("test", host, &connected);
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
  // Sent until the socket takes no more and, after a pause, holds bytes the
  // peer has not taken in: past what the peer can hold. A socket can stop
  // taking with all it sent taken in, the last of it not yet acknowledged,
  // as when the peer's buffer is just full; it takes more once acknowledged.
  const std::vector<std::uint8_t> bytes(std::size_t{1} << 20, 7);
  std::size_t sent = 0;
  const Clock::time_point end = Clock::now() + 4s;
  for (bool full = false; !full && Clock::now() < end;) {
    bool took = false;
    for (;;) {
      const vkc::Result<std::size_t> n = detail::send_some(
          "test", c.accepted.get(), bytes.data(), bytes.size());
      CHECK(n.ok());
      if (*n == 0) break;
      sent += *n;
      took = true;
    }
    std::this_thread::sleep_for(20ms);
    if (!took) {
      held = detail::unacknowledged("test", c.accepted.get());
      CHECK(held.ok());
      full = *held > 0 && *held + unread(c.client.get()) == sent;
    }
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

#ifdef __linux__
// The first line of `path`; empty if it cannot be read.
std::string first_line(const char* path) {
  std::ifstream file(path);
  std::string line;
  std::getline(file, line);
  return line;
}

// The congestion control `fd` uses (TCP_CONGESTION); empty on a failure.
std::string congestion(int fd) {
  char name[32] = {};
  socklen_t size = sizeof name;
  if (::getsockopt(fd, IPPROTO_TCP, TCP_CONGESTION, name, &size) != 0) {
    return "";
  }
  return std::string(name, ::strnlen(name, size));
}

// Whether /proc's allowed list names `name`, read apart from detail::lists.
bool proc_allows(const std::string& name) {
  std::istringstream words(
      first_line("/proc/sys/net/ipv4/tcp_allowed_congestion_control"));
  for (std::string word; words >> word;) {
    if (word == name) return true;
  }
  return false;
}

// The host's first IPv4 address on an interface that is up and not loopback,
// port 0; false if it has none.
bool own_address(detail::Address* out) {
  ifaddrs* all = nullptr;
  if (::getifaddrs(&all) != 0) return false;
  bool found = false;
  for (const ifaddrs* i = all; i != nullptr && !found; i = i->ifa_next) {
    if (i->ifa_addr == nullptr || i->ifa_addr->sa_family != AF_INET ||
        (i->ifa_flags & IFF_UP) == 0 || (i->ifa_flags & IFF_LOOPBACK) != 0) {
      continue;
    }
    std::memcpy(&out->in, i->ifa_addr, sizeof out->in);
    out->in.sin_port = 0;
    found = true;
  }
  ::freeifaddrs(all);
  return found;
}

std::atomic<int> infos{0};  // grid's Info messages, once counted
#endif

// The allowed list read word by word, and loopback told apart; on Linux, the
// accepted socket's congestion control against /proc.
int congestion_case() {
  CHECK(detail::kCongestion == "bbr");
  for (const char* allowed :
       {"bbr", "reno cubic bbr", "bbr reno", "reno\tbbr\n", "  bbr  "}) {
    CHECK(detail::lists(allowed, "bbr"));
  }
  for (const char* allowed : {"", " \n", "reno cubic", "reno cubic bbr2",
                              "bbr2 xbbr", "reno bb cubic", "renobbr"}) {
    CHECK(!detail::lists(allowed, "bbr"));
  }
  CHECK(detail::lists("reno cubic", "cubic"));
  CHECK(!detail::lists("reno cubic", ""));
  detail::Address a;
  for (const char* host :
       {"127.0.0.1", "127.0.0.0", "127.1.2.3", "127.255.255.255"}) {
    CHECK(detail::parse_ipv4(host, 1, &a) && detail::is_loopback(a));
  }
  for (const char* host : {"0.0.0.0", "10.0.0.1", "126.255.255.255",
                           "128.0.0.1", "192.168.1.20", "255.255.255.255"}) {
    CHECK(detail::parse_ipv4(host, 1, &a) && !detail::is_loopback(a));
  }

#ifdef __linux__
  vkc::set_log_handler(
      [](vkc::LogLevel level, std::string_view source, std::string_view) {
        if (level == vkc::LogLevel::Info && source == "vgrid") ++infos;
      });
  const std::string fallback =
      first_line("/proc/sys/net/ipv4/tcp_congestion_control");
  const bool allowed = proc_allows("bbr");
  CHECK(!fallback.empty());
  std::printf("networking socket: /proc allows bbr: %s, default %s\n",
              allowed ? "yes" : "no", fallback.c_str());

  // Loopback keeps the default, unsaid.
  Connected local;
  CHECK(connect_pair(&local) == 0);
  CHECK(congestion(local.accepted.get()) == fallback);
  CHECK(infos == 0);

  // A peer at the host's own address that is not loopback: BBR exactly when
  // /proc lists it, else the default, said.
  detail::Address own;
  if (own_address(&own)) {
    Connected remote;
    CHECK(connect_pair(&remote, own) == 0);
    CHECK(!detail::is_loopback(remote.peer));
    CHECK(congestion(remote.accepted.get()) == (allowed ? "bbr" : fallback));
    CHECK(infos == (allowed ? 0 : 1));
  } else {
    std::puts(
        "networking socket: SKIPPED the accepted congestion check: the host "
        "has no IPv4 address but loopback");
  }

  // A list that lacks it keeps the default whatever this process may choose,
  // as does a refusal; said once a process, here or above.
  CHECK(!detail::prefer_congestion("test", local.accepted.get(),
                                   "reno cubic bbr2"));
  CHECK(congestion(local.accepted.get()) == fallback);
  CHECK(!detail::prefer_congestion("test", -1, "reno bbr"));
  CHECK(infos == 1);
  if (allowed) {
    CHECK(detail::prefer_congestion(
        "test", local.accepted.get(),
        first_line("/proc/sys/net/ipv4/tcp_allowed_congestion_control")));
    CHECK(congestion(local.accepted.get()) == "bbr");
  }
  vkc::set_log_handler({});
#endif
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
  if (congestion_case() != 0) return 1;
  std::puts("networking socket: OK");
  return 0;
}
