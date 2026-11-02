// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// A sender that dials out (networking/dial.hpp, StreamServer::add_viewer,
// StreamClient::adopt) on loopback, 127.0.0.1 port 0 only: a viewer that
// dials a Listener, is added and is adopted reads the session and every
// record a TCP viewer reads, and a raw one the same bytes, in both
// replay_gop modes, an add counted in viewers() and on demand in
// key_wanted() by the time it returns; a server with listen false binds no
// socket, names port 0 and reads no host, where one with listen binds one;
// add_viewer refuses an empty fd, a socket that is not IPv4 or not a socket,
// one not connected, one past max_viewers and one once finished, each
// closed so its peer reads its end before any byte; an added socket, a
// blocking one the test connected, is set up as an accepted one is
// (non-blocking, TCP_NODELAY, the 128 KiB low-water mark); Listener refuses
// an empty, wildcard or named host, an address in use and one not this
// host's, takes a free port for 0, gives an empty fd with nothing waiting,
// within its wait, and accepted sockets the 2 MiB receive buffer, above the
// default, refuses connections once destroyed, and an accept out of
// descriptors is an IoError naming it, after which it goes on; dial refuses
// each bad entry with a message an address field can show, a closed port with
// the system's words and a connect that never finishes at its limit, and
// connects to a Listener with a non-blocking, close-on-exec socket; adopt
// refuses an empty fd, a Unix socket and an idle_limit of 0, and its errors
// name the dialler's address, a peer that sends nothing ending as "connected,
// but no session"; and a thread adding viewers while another publishes and
// finishes, for ThreadSanitizer.

#include <fcntl.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <sys/resource.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iterator>
#include <memory>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "volumetric_kit/core/testing/test_check.hpp"
#include "volumetric_kit/grid/networking/dial.hpp"
#include "volumetric_kit/grid/networking/stream_client.hpp"
#include "volumetric_kit/grid/networking/stream_server.hpp"
#include "volumetric_kit/grid/stream/chunks.hpp"
#include "volumetric_kit/grid/testing/networking_loopback.hpp"
#include "volumetric_kit/grid/testing/stream_files.hpp"

namespace vkc = volumetric_kit::core;
namespace networking = volumetric_kit::grid::networking;
namespace stream = volumetric_kit::grid::stream;
namespace vrst = vgrid_test::vrst;
namespace net = vgrid_test::net;

namespace {

using Code = vkc::Status::Code;
using net::Bytes;
using net::Clock;
using net::Got;
using namespace std::chrono_literals;

int option(int fd, int level, int name) {
  int value = -1;
  socklen_t size = sizeof value;
  if (::getsockopt(fd, level, name, &value, &size) != 0) return -1;
  return value;
}

bool nonblocking(int fd) { return (::fcntl(fd, F_GETFL) & O_NONBLOCK) != 0; }
bool cloexec(int fd) { return (::fcntl(fd, F_GETFD) & FD_CLOEXEC) != 0; }

std::uint16_t peer_port(int fd) {
  sockaddr_in a{};
  socklen_t size = sizeof a;
  if (::getpeername(fd, reinterpret_cast<sockaddr*>(&a), &size) != 0) return 0;
  return ntohs(a.sin_port);
}

bool starts_with(const std::string& s, const std::string& prefix) {
  return s.compare(0, prefix.size(), prefix) == 0;
}

// Whether `s` is `code` with `message`, or with a message starting with it
// when `prefix`; what it is instead on stderr.
bool is(const vkc::Status& s, Code code, const std::string& message,
        bool prefix = false) {
  if (s.domain() == code &&
      (prefix ? starts_with(s.message(), message) : s.message() == message)) {
    return true;
  }
  std::fprintf(stderr, "got %d \"%s\", wanted %d \"%s\"%s\n",
               static_cast<int>(s.domain()), s.message().c_str(),
               static_cast<int>(code), message.c_str(), prefix ? "..." : "");
  return false;
}

// Whether `fd`'s peer closed it before any byte, within 2 s.
bool ends_at_once(int fd) {
  Bytes bytes;
  bool eof = false;
  net::read_upto(fd, 1, 2s, &bytes, &eof);
  return eof && bytes.empty();
}

std::unique_ptr<networking::StreamServer> start(
    const stream::Session& session, const networking::ServerOptions& options) {
  vkc::Result<std::unique_ptr<networking::StreamServer>> server =
      networking::StreamServer::start(session, options);
  if (!server.ok()) {
    std::fprintf(stderr, "start: %s\n", server.status().message().c_str());
    return nullptr;
  }
  return std::move(server).value();
}

// A server that dials out: no listener.
std::unique_ptr<networking::StreamServer> dialling(
    const stream::Session& session,
    networking::ServerOptions options = networking::ServerOptions{}) {
  options.listen = false;
  return start(session, options);
}

std::unique_ptr<networking::Listener> listen_loopback() {
  vkc::Result<std::unique_ptr<networking::Listener>> listener =
      networking::Listener::create("127.0.0.1", 0);
  if (!listener.ok()) {
    std::fprintf(stderr, "Listener: %s\n", listener.status().message().c_str());
    return nullptr;
  }
  return std::move(listener).value();
}

vkc::UniqueFd dial_to(const networking::Listener& listener) {
  vkc::Result<vkc::UniqueFd> fd =
      networking::dial("127.0.0.1", listener.port(), 2s);
  if (!fd.ok()) {
    std::fprintf(stderr, "dial: %s\n", fd.status().message().c_str());
    return {};
  }
  return std::move(fd).value();
}

vkc::UniqueFd accept_from(networking::Listener& listener) {
  vkc::Result<vkc::UniqueFd> fd = listener.accept(2s);
  if (!fd.ok()) {
    std::fprintf(stderr, "accept: %s\n", fd.status().message().c_str());
    return {};
  }
  return std::move(fd).value();
}

std::unique_ptr<networking::StreamClient> adopt(
    vkc::UniqueFd fd, const networking::ClientOptions& options = {}) {
  vkc::Result<std::unique_ptr<networking::StreamClient>> client =
      networking::StreamClient::adopt(std::move(fd), options);
  if (!client.ok()) {
    std::fprintf(stderr, "adopt: %s\n", client.status().message().c_str());
    return nullptr;
  }
  return std::move(client).value();
}

struct Published {
  stream::RecordInfo info;
  Bytes payload;
};

// Two GOPs: keys at 0 and 4, an empty payload and one past 64 KiB.
std::vector<Published> two_gops() {
  const std::size_t sizes[] = {1000, 0, 7, 65537, 3000, 5};
  std::vector<Published> out;
  for (std::uint64_t seq = 0; seq < std::size(sizes); ++seq) {
    const bool key = seq == 0 || seq == 4;
    out.push_back(
        {net::record_info(seq, key), net::record_payload(seq, sizes[seq])});
  }
  return out;
}

// --- Dialled viewers read what TCP viewers read. ----------------------------

// Viewer A connects; B dials a Listener, is added and is adopted; C connects
// raw; D dials and is added, read raw. A and B read the same session and
// records, C and D the same bytes: the writer's chunks. Each add is counted
// in viewers(), and on demand in key_wanted(), as it returns.
int adopted_reads_as_connected_case(bool replay) {
  const stream::Session session = net::geometry_session();
  networking::ServerOptions options;
  options.replay_gop = replay;
  auto server = net::serve(session, options);
  CHECK(server != nullptr);
  auto listener = listen_loopback();
  CHECK(listener != nullptr);

  auto a = net::connect(server->port());
  CHECK(a != nullptr);
  CHECK(net::wait_session(*a));
  const vkc::UniqueFd c = net::connect_loopback(server->port());
  CHECK(c.valid());
  CHECK(net::serves_within(*server, 2));

  // Added: counted, and on demand wanting a key, once add_viewer returns.
  std::uint64_t wanted = server->key_wanted();
  CHECK(server->add_viewer(dial_to(*listener)).ok());
  CHECK(server->viewers() == 3);
  CHECK(server->key_wanted() == wanted + (replay ? 0 : 1));
  auto b = adopt(accept_from(*listener));
  CHECK(b != nullptr);
  CHECK(net::wait_session(*b));
  CHECK(vrst::same(*b->session(), *a->session()));
  CHECK(vrst::same(*b->session(), session));

  wanted = server->key_wanted();
  CHECK(server->add_viewer(dial_to(*listener)).ok());
  CHECK(server->viewers() == 4);
  CHECK(server->key_wanted() == wanted + (replay ? 0 : 1));
  const vkc::UniqueFd d = accept_from(*listener);
  CHECK(d.valid());

  // The wire every viewer gets: the session chunk, then each record's.
  vkc::Result<stream::ChunkEncoder> encoder =
      stream::ChunkEncoder::create(session);
  CHECK(encoder.ok());
  Bytes wire = encoder->session_chunk();
  Bytes wire_c;
  Bytes wire_d;
  net::read_upto(c.get(), wire.size(), 5s, &wire_c);
  net::read_upto(d.get(), wire.size(), 5s, &wire_d);
  CHECK(wire_c == wire && wire_d == wire);

  for (const Published& r : two_gops()) {
    CHECK(server->publish(r.info, r.payload.data(), r.payload.size()).ok());
    vkc::Result<stream::ChunkEncoder::RecordFraming> framed =
        encoder->frame(r.info, r.payload.data(), r.payload.size());
    CHECK(framed.ok());
    wire.insert(wire.end(), framed->header.begin(), framed->header.end());
    wire.insert(wire.end(), r.payload.begin(), r.payload.end());
    wire.insert(wire.end(), framed->padding, std::uint8_t{0});
    for (networking::StreamClient* viewer : {a.get(), b.get()}) {
      Got got;
      CHECK(net::next_within(*viewer, &got));
      CHECK(vrst::same(got.record.info, r.info));
      CHECK(got.payload == r.payload);
    }
    net::read_upto(c.get(), wire.size(), 5s, &wire_c);
    net::read_upto(d.get(), wire.size(), 5s, &wire_d);
    CHECK(wire_c == wire && wire_d == wire);
  }
  return 0;
}

// --- listen false. -----------------------------------------------------------

// The IPv4 sockets of this process bound to a port with no peer: each one
// listening, or bound to listen.
std::vector<int> bound_sockets() {
  std::vector<int> out;
  for (int fd = 0; fd < 4096; ++fd) {
    sockaddr_in local{};
    socklen_t size = sizeof local;
    if (::getsockname(fd, reinterpret_cast<sockaddr*>(&local), &size) != 0 ||
        local.sin_family != AF_INET || local.sin_port == 0) {
      continue;
    }
    sockaddr_in peer{};
    size = sizeof peer;
    if (::getpeername(fd, reinterpret_cast<sockaddr*>(&peer), &size) == 0) {
      continue;
    }
    out.push_back(fd);
  }
  return out;
}

// With listen false the server binds no socket, names port 0 and reads no
// host; with listen it binds one, at its port.
int listen_false_case() {
  const stream::Session session = net::geometry_session();
  const std::vector<int> before = bound_sockets();
  networking::ServerOptions options;
  options.host = "not an address";
  options.port = 0;
  auto server = dialling(session, options);
  CHECK(server != nullptr);
  CHECK(server->port() == 0);
  CHECK(bound_sockets() == before);
  CHECK(server->viewers() == 0);

  options.listen = true;
  options.host = "127.0.0.1";
  auto listening = start(session, options);
  CHECK(listening != nullptr);
  const std::vector<int> after = bound_sockets();
  CHECK(after.size() == before.size() + 1);
  std::vector<int> added;
  std::set_difference(after.begin(), after.end(), before.begin(), before.end(),
                      std::back_inserter(added));
  CHECK(added.size() == 1 && net::local_port(added[0]) == listening->port());
  return 0;
}

// --- add_viewer. -------------------------------------------------------------

// Each refusal: the domain, the message, the socket closed so its peer reads
// its end before any byte, and no viewer counted.
int add_viewer_refusals_case() {
  const stream::Session session = net::geometry_session();
  networking::ServerOptions options;
  options.max_viewers = 1;
  auto server = dialling(session, options);
  CHECK(server != nullptr);
  auto listener = listen_loopback();
  CHECK(listener != nullptr);

  vkc::Status s = server->add_viewer(vkc::UniqueFd());
  CHECK(is(s, Code::InvalidArgument,
           "StreamServer::add_viewer: an empty socket"));

  // A Unix socket: refused, and closed.
  int ends[2] = {-1, -1};
  CHECK(::socketpair(AF_UNIX, SOCK_STREAM, 0, ends) == 0);
  const vkc::UniqueFd unix_peer(ends[1]);
  s = server->add_viewer(vkc::UniqueFd(ends[0]));
  CHECK(is(s, Code::InvalidArgument,
           "StreamServer::add_viewer: not an IPv4 socket"));
  CHECK(ends_at_once(unix_peer.get()));
  // Not a socket.
  int pipe_fds[2] = {-1, -1};
  CHECK(::pipe(pipe_fds) == 0);
  const vkc::UniqueFd pipe_read(pipe_fds[0]);
  s = server->add_viewer(vkc::UniqueFd(pipe_fds[1]));
  CHECK(is(s, Code::InvalidArgument, "StreamServer::add_viewer: not a socket"));
  // Not connected.
  s = server->add_viewer(vkc::UniqueFd(::socket(AF_INET, SOCK_STREAM, 0)));
  CHECK(is(s, Code::IoError,
           "StreamServer::add_viewer: cannot name the peer: ", true));
  CHECK(server->viewers() == 0);

  // One fits; the next is past max_viewers.
  CHECK(server->add_viewer(dial_to(*listener)).ok());
  auto first = adopt(accept_from(*listener));
  CHECK(first != nullptr);
  CHECK(net::wait_session(*first));
  // Named by its peer, the reader's address.
  s = server->add_viewer(dial_to(*listener));
  CHECK(is(s, Code::IoError,
           "StreamServer::add_viewer: 127.0.0.1:" +
               std::to_string(listener->port()) + ": 1 viewers already"));
  vkc::UniqueFd refused = accept_from(*listener);
  CHECK(refused.valid() && ends_at_once(refused.get()));
  CHECK(server->viewers() == 1);

  // Once finished.
  server->finish();
  CHECK(server->viewers() == 0);
  s = server->add_viewer(dial_to(*listener));
  CHECK(is(s, Code::InvalidArgument,
           "StreamServer::add_viewer: the stream finished"));
  refused = accept_from(*listener);
  CHECK(refused.valid() && ends_at_once(refused.get()));
  CHECK(server->viewers() == 0);
  return 0;
}

// A blocking socket the test connected, added, is set up to send as an
// accepted one is: non-blocking, TCP_NODELAY, the 128 KiB low-water mark;
// and it is served.
int sender_setup_case() {
  const stream::Session session = net::geometry_session();
  auto server = dialling(session);
  CHECK(server != nullptr);
  auto listener = listen_loopback();
  CHECK(listener != nullptr);
  vkc::UniqueFd fd = net::connect_loopback(listener->port());
  CHECK(fd.valid());
  // A second descriptor of the same socket, to read its options back.
  const vkc::UniqueFd kept(::dup(fd.get()));
  CHECK(kept.valid());
  CHECK(!nonblocking(kept.get()));
  CHECK(option(kept.get(), IPPROTO_TCP, TCP_NODELAY) == 0);
  CHECK(option(kept.get(), IPPROTO_TCP, TCP_NOTSENT_LOWAT) != 128 << 10);
  CHECK(server->add_viewer(std::move(fd)).ok());
  CHECK(nonblocking(kept.get()));
  CHECK(option(kept.get(), IPPROTO_TCP, TCP_NODELAY) != 0);
  CHECK(option(kept.get(), IPPROTO_TCP, TCP_NOTSENT_LOWAT) == 128 << 10);
#ifdef __APPLE__
  CHECK(option(kept.get(), SOL_SOCKET, SO_NOSIGPIPE) != 0);
#endif
  auto client = adopt(accept_from(*listener));
  CHECK(client != nullptr);
  CHECK(net::wait_session(*client));
  return 0;
}

// --- Listener. ---------------------------------------------------------------

// The receive buffer of a connection accepted from a raw listener on
// loopback with none set, as the kernel reports it.
int default_accepted_receive_buffer() {
  const net::Listener l = net::bind_loopback();
  const vkc::UniqueFd c = net::connect_loopback(l.port);
  const vkc::UniqueFd accepted = net::accept_within(l.fd.get(), 2s);
  if (!c.valid() || !accepted.valid()) return -1;
  return option(accepted.get(), SOL_SOCKET, SO_RCVBUF);
}

// What a socket given 2 MiB reads back at least: Linux holds what is asked
// for to net.core.rmem_max, then doubles it for its bookkeeping; macOS
// rounds an accepted socket's up, by an amount that varies.
int receive_buffer_floor() {
  int asked = 2 << 20;
#ifdef __linux__
  std::ifstream rmem_max("/proc/sys/net/core/rmem_max");
  int most = 0;
  if (!(rmem_max >> most)) return -1;
  asked = 2 * std::min(asked, most);
#endif
  return asked;
}

int listener_case() {
  using networking::Listener;
  for (const char* host : {"", "0.0.0.0", "localhost"}) {
    const vkc::Result<std::unique_ptr<Listener>> refused =
        Listener::create(host, 0);
    CHECK(!refused.ok() && refused.status().domain() == Code::InvalidArgument);
    CHECK(starts_with(refused.status().message(), "Listener::create: "));
  }
  CHECK(Listener::create("", 0).status().message() ==
        "Listener::create: the host is empty");
  CHECK(Listener::create("0.0.0.0", 0).status().message() ==
        "Listener::create: 0.0.0.0 is the wildcard, not one host's address");
  CHECK(Listener::create("localhost", 0).status().message() ==
        "Listener::create: \"localhost\" is not a numeric IPv4 address");

  auto listener = listen_loopback();
  CHECK(listener != nullptr && listener->port() != 0);
  const std::string name = "127.0.0.1:" + std::to_string(listener->port());
  // In use; and not this host's (TEST-NET-1).
  vkc::Result<std::unique_ptr<Listener>> refused =
      Listener::create("127.0.0.1", listener->port());
  CHECK(is(refused.status(), Code::IoError,
           "Listener::create: cannot listen on " + name + ": ", true));
  refused = Listener::create("192.0.2.1", 0);
  CHECK(is(refused.status(), Code::IoError,
           "Listener::create: cannot listen on 192.0.2.1:0: ", true));

  // None waiting: an empty fd, at once or within the wait.
  vkc::Result<vkc::UniqueFd> none = listener->accept(0ms);
  CHECK(none.ok() && !none->valid());
  const Clock::time_point began = Clock::now();
  none = listener->accept(50ms);
  CHECK(none.ok() && !none->valid() && Clock::now() - began >= 45ms);

  // An accepted socket: non-blocking, close-on-exec, and the 2 MiB receive
  // buffer, above the default.
  const int asked = receive_buffer_floor();
  CHECK(asked > 0 && default_accepted_receive_buffer() < asked);
  const vkc::UniqueFd peer = net::connect_loopback(listener->port());
  CHECK(peer.valid());
  const vkc::UniqueFd accepted = accept_from(*listener);
  CHECK(accepted.valid());
  CHECK(nonblocking(accepted.get()) && cloexec(accepted.get()));
  CHECK(option(accepted.get(), SOL_SOCKET, SO_RCVBUF) >= asked);
  CHECK(peer_port(accepted.get()) == net::local_port(peer.get()));

  // Destroyed, it refuses a connection.
  const std::uint16_t port = listener->port();
  listener.reset();
  CHECK(!net::connect_loopback(port).valid());
  return 0;
}

// Out of descriptors, an accept is an IoError naming the listener and the
// system's words, and the listener goes on: the next accept takes the
// connection, which Linux kept queued, or a new one, as macOS drops it.
int listener_out_of_descriptors_case() {
  auto listener = listen_loopback();
  CHECK(listener != nullptr);
  const vkc::UniqueFd peer = net::connect_loopback(listener->port());
  CHECK(peer.valid());
  rlimit saved{};
  CHECK(::getrlimit(RLIMIT_NOFILE, &saved) == 0);
  const int lowest = ::dup(peer.get());  // the lowest free descriptor
  CHECK(lowest >= 0 && ::close(lowest) == 0);
  rlimit low = saved;
  low.rlim_cur = static_cast<rlim_t>(lowest);
  CHECK(::setrlimit(RLIMIT_NOFILE, &low) == 0);
  vkc::Result<vkc::UniqueFd> failed = listener->accept(2s);
  const bool restored = ::setrlimit(RLIMIT_NOFILE, &saved) == 0;
  CHECK(restored);
  CHECK(is(failed.status(), Code::IoError,
           "Listener: 127.0.0.1:" + std::to_string(listener->port()) +
               ": cannot accept: ",
           true));
  const vkc::UniqueFd next = net::connect_loopback(listener->port());
  CHECK(next.valid());
  const vkc::UniqueFd accepted = accept_from(*listener);
  CHECK(accepted.valid());
#ifdef __linux__
  CHECK(peer_port(accepted.get()) == net::local_port(peer.get()));
#else
  CHECK(peer_port(accepted.get()) == net::local_port(next.get()));
#endif
  return 0;
}

// --- dial. -------------------------------------------------------------------

int dial_case() {
  const auto refusal = [](const std::string& host, std::uint16_t port,
                          std::chrono::milliseconds limit) {
    const vkc::Result<vkc::UniqueFd> fd = networking::dial(host, port, limit);
    return fd.ok() || fd.status().domain() != Code::InvalidArgument
               ? std::string("(not refused)")
               : fd.status().message();
  };
  CHECK(refusal("", 7101, 2s) == "dial: the host is empty");
  CHECK(refusal("0.0.0.0", 7101, 2s) ==
        "dial: 0.0.0.0 is the wildcard, not one host's address");
  CHECK(refusal("localhost", 7101, 2s) ==
        "dial: \"localhost\" is not a numeric IPv4 address");
  CHECK(refusal("10.8.0.1 ", 7101, 2s) ==
        "dial: \"10.8.0.1 \" is not a numeric IPv4 address");
  CHECK(refusal("10.8.0.1", 0, 2s) == "dial: port 0 is not a port to dial");
  CHECK(refusal("10.8.0.1", 7101, 0ms) ==
        "dial: a limit of 0 ms; it must be positive");
  CHECK(refusal("10.8.0.1", 7101, -5ms) ==
        "dial: a limit of -5 ms; it must be positive");

  // A closed port: the system's words.
  const std::uint16_t closed = net::closed_port();
  CHECK(closed != 0);
  const vkc::Result<vkc::UniqueFd> refused =
      networking::dial("127.0.0.1", closed, 2s);
  CHECK(is(refused.status(), Code::IoError,
           "dial: 127.0.0.1:" + std::to_string(closed) + ": cannot connect: ",
           true));

  // A Listener: connected, non-blocking and close-on-exec.
  auto listener = listen_loopback();
  CHECK(listener != nullptr);
  const vkc::UniqueFd fd = dial_to(*listener);
  CHECK(fd.valid() && nonblocking(fd.get()) && cloexec(fd.get()));
  CHECK(peer_port(fd.get()) == listener->port());
  const vkc::UniqueFd accepted = accept_from(*listener);
  CHECK(accepted.valid() &&
        peer_port(accepted.get()) == net::local_port(fd.get()));
  return 0;
}

// A connect that never finishes ends at the limit. macOS drops a SYN to a
// socket bound and not listening, which Linux refuses; Linux drops one to a
// listener whose queue is full, which macOS resets. So the port is bound,
// on Linux listening with a queue of one, and connected to until a connect
// does not finish.
int dial_limit_case() {
  vkc::UniqueFd listener(::socket(AF_INET, SOCK_STREAM, 0));
  const sockaddr_in a = net::loopback(0);
  CHECK(listener.valid() &&
        ::bind(listener.get(), reinterpret_cast<const sockaddr*>(&a),
               sizeof a) == 0);
#ifndef __APPLE__
  CHECK(::listen(listener.get(), 0) == 0);
#endif
  const std::uint16_t port = net::local_port(listener.get());
  std::vector<vkc::UniqueFd> queued;
  bool full = false;
  for (int i = 0; i < 64 && !full; ++i) {
    vkc::UniqueFd fd(::socket(AF_INET, SOCK_STREAM, 0));
    CHECK(fd.valid() && ::fcntl(fd.get(), F_SETFL, O_NONBLOCK) == 0);
    const sockaddr_in to = net::loopback(port);
    (void)::connect(fd.get(), reinterpret_cast<const sockaddr*>(&to),
                    sizeof to);
    pollfd p{fd.get(), POLLOUT, 0};
    full = ::poll(&p, 1, 100) == 0;
    queued.push_back(std::move(fd));
  }
  CHECK(full);
  const Clock::time_point began = Clock::now();
  const vkc::Result<vkc::UniqueFd> fd =
      networking::dial("127.0.0.1", port, 300ms);
  const auto took = Clock::now() - began;
  CHECK(is(fd.status(), Code::IoError,
           "dial: 127.0.0.1:" + std::to_string(port) +
               ": cannot connect within 300 ms"));
  CHECK(took >= 300ms && took < 2s);
  return 0;
}

// --- adopt. ------------------------------------------------------------------

int adopt_case() {
  using networking::StreamClient;
  vkc::Result<std::unique_ptr<StreamClient>> r =
      StreamClient::adopt(vkc::UniqueFd());
  CHECK(is(r.status(), Code::InvalidArgument,
           "StreamClient::adopt: an empty socket"));
  int ends[2] = {-1, -1};
  CHECK(::socketpair(AF_UNIX, SOCK_STREAM, 0, ends) == 0);
  const vkc::UniqueFd unix_peer(ends[1]);
  r = StreamClient::adopt(vkc::UniqueFd(ends[0]));
  CHECK(is(r.status(), Code::InvalidArgument,
           "StreamClient::adopt: not an IPv4 socket"));
  CHECK(ends_at_once(unix_peer.get()));

  auto listener = listen_loopback();
  CHECK(listener != nullptr);
  {
    const vkc::UniqueFd dialled = dial_to(*listener);
    CHECK(dialled.valid());
    networking::ClientOptions zero;
    zero.idle_limit = 0ms;
    r = StreamClient::adopt(accept_from(*listener), zero);
    CHECK(is(r.status(), Code::InvalidArgument,
             "StreamClient::adopt: an idle_limit of 0 ms"));
    CHECK(ends_at_once(dialled.get()));
  }

  // A dialler that sends nothing: the errors name its address, and the stage.
  const vkc::UniqueFd dialled = dial_to(*listener);
  CHECK(dialled.valid());
  const std::uint16_t dialler = net::local_port(dialled.get());
  networking::ClientOptions options;
  options.idle_limit = 200ms;
  auto client = adopt(accept_from(*listener), options);
  CHECK(client != nullptr && client->session() == nullptr);
  Got got;
  vkc::Result<bool> next = false;
  const Clock::time_point began = Clock::now();
  while (next.ok() && Clock::now() - began < 5s) {
    next = client->next(10ms, &got.record, &got.payload);
  }
  CHECK(!next.ok() && next.status().domain() == Code::IoError);
  CHECK(net::client_reason(next.status().message(), dialler) ==
        "connected, but no session within 200 ms");
  CHECK(ends_at_once(dialled.get()));

  // One that closes before its session.
  vkc::UniqueFd closing = dial_to(*listener);
  CHECK(closing.valid());
  const std::uint16_t closer = net::local_port(closing.get());
  client = adopt(accept_from(*listener));
  CHECK(client != nullptr);
  closing.reset();
  next = false;
  while (next.ok() && Clock::now() - began < 10s) {
    next = client->next(10ms, &got.record, &got.payload);
  }
  CHECK(!next.ok() &&
        net::client_reason(next.status().message(), closer) ==
            "the sender closed the connection before its session (it may "
            "already serve its most viewers)");
  return 0;
}

// --- Threads. ----------------------------------------------------------------

// One thread dials and adds viewers while the main thread publishes and then
// finishes, and a reader keeps some connections and closes others, so
// viewers leave while others join: each add is served, refused past
// max_viewers or, once finished, refused as such.
int threads_case() {
  const stream::Session session = net::geometry_session();
  networking::ServerOptions options;
  options.replay_gop = false;
  options.max_viewers = 2;
  auto server = dialling(session, options);
  CHECK(server != nullptr);
  auto listener = listen_loopback();
  CHECK(listener != nullptr);
  std::atomic<bool> quit{false};
  std::atomic<int> added{0};
  std::atomic<int> full{0};
  std::atomic<int> finished{0};
  std::atomic<int> other{0};
  std::thread reader([&] {
    std::vector<vkc::UniqueFd> kept;
    std::uint8_t buffer[65536];
    for (int n = 0; !quit; ++n) {
      vkc::Result<vkc::UniqueFd> fd = listener->accept(5ms);
      if (fd.ok() && fd->valid() && n % 2 == 0) {
        kept.push_back(std::move(fd).value());
      }
      for (const vkc::UniqueFd& k : kept) {
        while (::recv(k.get(), buffer, sizeof buffer, MSG_DONTWAIT) > 0) {
        }
      }
      if (kept.size() > 2) kept.erase(kept.begin());  // it leaves
    }
  });
  std::thread adder([&] {
    for (;;) {
      vkc::Result<vkc::UniqueFd> fd =
          networking::dial("127.0.0.1", listener->port(), 5s);
      if (!fd.ok()) {
        ++other;
        return;
      }
      const vkc::Status s = server->add_viewer(std::move(fd).value());
      (void)server->viewers();
      (void)server->key_wanted();
      (void)server->draining();
      if (s.ok()) {
        ++added;
      } else if (s.domain() == Code::IoError) {
        ++full;
      } else if (s.domain() == Code::InvalidArgument) {
        ++finished;
        return;
      } else {
        ++other;
        return;
      }
      std::this_thread::sleep_for(1ms);
    }
  });
  std::uint64_t wanted = 0;
  for (std::uint64_t seq = 0; seq < 300; ++seq) {
    const bool key = server->key_wanted() != wanted || seq % 30 == 0;
    wanted = server->key_wanted();
    const Bytes payload = net::record_payload(seq, 1000);
    CHECK(server
              ->publish(net::record_info(seq, key), payload.data(),
                        payload.size())
              .ok());
    std::this_thread::sleep_for(1ms);
  }
  server->finish();
  adder.join();
  quit = true;
  reader.join();
  CHECK(added > 0 && finished == 1 && other == 0);
  CHECK(server->viewers() == 0);
  return 0;
}

}  // namespace

int main() {
  for (const bool replay : {true, false}) {
    if (adopted_reads_as_connected_case(replay) != 0) return 1;
  }
  if (listen_false_case() != 0) return 1;
  if (add_viewer_refusals_case() != 0) return 1;
  if (sender_setup_case() != 0) return 1;
  if (listener_case() != 0) return 1;
  if (listener_out_of_descriptors_case() != 0) return 1;
  if (dial_case() != 0) return 1;
  if (dial_limit_case() != 0) return 1;
  if (adopt_case() != 0) return 1;
  if (threads_case() != 0) return 1;
  std::puts("networking dial: OK");
  return 0;
}
