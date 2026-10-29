// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// StreamClient against raw senders on loopback (127.0.0.1, port 0) that send
// any bytes: a refused port is IoError; HTTP's bytes are InvalidArgument; a
// header claiming 1 GiB is refused before any of its body comes (a RECD's or
// another tag's OutOfMemory, a VRST's InvalidArgument); a corrupt body is
// InvalidArgument; a record held by a next with no payload is read into
// caller memory; the end after whole records is IoError once they are all
// read, and an end before the session says so; next(20 ms) with nothing
// returns false in about 20 ms, and within 500 ms while chunks it skips flood
// in; next(0 ms) alone connects and reads, a 230 KB record already in the
// socket in one call; progress distinguishes a partial record sent in pieces
// from idle input, stays true across a later idle step, includes
// buffered records and resets for each call, saved errors included;
// idle_limit counts from connect, against a listener that
// sends nothing and a port whose connects never finish, and then from the
// last byte, judged after taking what came while the caller was away (bytes,
// a connect's end or its refusal);
// milliseconds::max() as idle_limit or wait is never; localhost reaches a
// 127.0.0.1 server, and clients destroyed mid-lookup leave it to end alone;
// and an empty host, port 0 or an idle_limit of 0 or less is refused. Each
// error reads "StreamClient: 127.0.0.1:<port>: <reason>", the
// address in it once, with the stage the idle limit ended in as its reason;
// it closes the connection, as a sender sees, and is returned again by every
// later call.

#include <fcntl.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <initializer_list>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "networking_loopback.hpp"
#include "stream_files.hpp"
#include "volumetric_kit/core/testing/test_allocation_failure.hpp"
#include "volumetric_kit/core/testing/test_check.hpp"

namespace vkc = volumetric_kit::core;
namespace networking = volumetric_kit::recon::networking;
namespace stream = volumetric_kit::recon::stream;
namespace vrst = vr_test::vrst;
namespace net = vr_test::net;

namespace {

using Code = vkc::Status::Code;
using net::Bytes;
using net::Clock;
using net::Got;
using namespace std::chrono_literals;

// A raw sender: a listener, and the connection a client made to it.
struct Sender {
  net::Listener listener = net::bind_loopback();
  vkc::UniqueFd peer;

  // Connect `client` and take its connection.
  bool take(networking::StreamClient& client) {
    stream::Record record;
    Bytes payload;
    if (!client.next(0ms, &record, &payload).ok()) return false;
    peer = net::accept_within(listener.fd.get(), 2s);
    return peer.valid();
  }
};

// What a client's next calls return until one is not `false`, within 5 s.
vkc::Result<bool> settle(networking::StreamClient& client, Got* got) {
  const Clock::time_point end = Clock::now() + 5s;
  for (;;) {
    vkc::Result<bool> r = client.next(10ms, &got->record, &got->payload);
    if (!r.ok() || *r || Clock::now() >= end) return r;
  }
}

// The error a client gets from `bytes`, the connection left open, its message
// the client's prefix and the parser's reason; the same error from the call
// after; and the client's end of the connection closed, which the sender
// reads.
Code error_from(const Bytes& bytes) {
  Sender sender;
  auto client = net::connect(sender.listener.port);
  if (client == nullptr || !sender.take(*client) ||
      !net::send_bytes(sender.peer.get(), bytes)) {
    return Code::Ok;
  }
  Got got;
  const vkc::Result<bool> first = settle(*client, &got);
  if (first.ok() ||
      net::client_reason(first.status().message(), sender.listener.port)
              .rfind("ChunkParser: ", 0) != 0) {
    return Code::Ok;
  }
  bool progressed = true;
  const vkc::Result<bool> again =
      client->next(0ms, &got.record, &got.payload, &progressed);
  if (again.ok() || again.status().domain() != first.status().domain() ||
      again.status().message() != first.status().message() || progressed) {
    return Code::Ok;
  }
  Bytes rest;
  bool closed = false;
  net::read_upto(sender.peer.get(), 1, 2s, &rest, &closed);
  if (!closed) return Code::Ok;
  return first.status().domain();
}

Bytes session_bytes() {
  return vrst::chunk("VRST", vrst::session_fields(0, 1, 0),
                     vrst::session_body(1));
}

Bytes concat(std::initializer_list<Bytes> parts) {
  Bytes out;
  for (const Bytes& p : parts) out.insert(out.end(), p.begin(), p.end());
  return out;
}

int refused_case() {
  const std::uint16_t port = net::closed_port();
  CHECK(port != 0);
  auto client = net::connect(port);
  CHECK(client != nullptr);
  Got got;
  const vkc::Result<bool> r = settle(*client, &got);
  CHECK(!r.ok() && r.status().domain() == Code::IoError);
  CHECK(net::client_reason(r.status().message(), port)
            .rfind("cannot connect: ", 0) == 0);
  const vkc::Result<bool> again = client->next(0ms, &got.record, &got.payload);
  CHECK(!again.ok() && again.status().message() == r.status().message());
  return 0;
}

int not_a_stream_case() {
  const std::string http =
      "HTTP/1.1 200 OK\r\nContent-Length: 0\r\nConnection: close\r\n\r\n";
  CHECK(error_from(Bytes(http.begin(), http.end())) == Code::InvalidArgument);
  return 0;
}

// Headers claiming 1 GiB of body, each refused once its header is in.
int huge_header_case() {
  constexpr std::uint64_t kGiB = std::uint64_t{1} << 30;
  CHECK(error_from(vrst::header("VRST", vrst::session_fields(0, 1, 0), kGiB,
                                0)) == Code::InvalidArgument);
  CHECK(
      error_from(concat(
          {session_bytes(),
           vrst::header("RECD", vrst::record_fields(0, 1, 0, 0, stream::kNoSet),
                        kGiB, 0)})) == Code::OutOfMemory);
  CHECK(error_from(concat(
            {session_bytes(), vrst::header("ABCD", Bytes(16, 0), kGiB, 0)})) ==
        Code::OutOfMemory);
  return 0;
}

// A corrupt body is InvalidArgument; so is one after a whole record, sent at
// once, which is read first, the failure found once the record is taken.
int corrupt_body_case() {
  Bytes record = vrst::record(0, 1, 0, 0, stream::kNoSet, vrst::pattern(64, 1));
  record[56 + 10] ^= 0x20;
  CHECK(error_from(concat({session_bytes(), record})) == Code::InvalidArgument);

  Sender sender;
  auto client = net::connect(sender.listener.port);
  CHECK(client != nullptr && sender.take(*client));
  const Bytes whole =
      vrst::record(0, 1, 0, 0, stream::kNoSet, vrst::pattern(64, 2));
  CHECK(net::send_bytes(sender.peer.get(),
                        concat({session_bytes(), whole, record})));
  Got got;
  vkc::Result<bool> r = settle(*client, &got);
  CHECK(r.ok() && *r && got.record.info.seq == 0);
  r = settle(*client, &got);
  CHECK(!r.ok() && r.status().domain() == Code::InvalidArgument);
  CHECK(net::client_reason(r.status().message(), sender.listener.port)
            .rfind("ChunkParser: ", 0) == 0);
  return 0;
}

// next with no payload holds the record for read, which copies it into
// caller memory -- a slot with room after it, which stays as it was. A
// capacity short by one is refused with the connection open and the record
// held, so the read after it takes it; a corrupt body fails the null-payload
// next as it fails the other, and every read after returns that error.
int read_into_case() {
  Sender sender;
  auto client = net::connect(sender.listener.port);
  CHECK(client != nullptr && sender.take(*client));
  const Bytes a = vrst::pattern(1000, 3);
  const Bytes b = vrst::pattern(37, 4);
  Bytes bad = vrst::record(0, 0, 2, 2, stream::kNoSet, vrst::pattern(64, 5));
  bad[56 + 10] ^= 0x20;
  CHECK(net::send_bytes(
      sender.peer.get(),
      concat({session_bytes(), vrst::record(0, 1, 0, 0, stream::kNoSet, a),
              vrst::record(0, 0, 1, 1, stream::kNoSet, b), bad})));
  stream::Record record;
  const auto held = [&]() -> vkc::Result<bool> {
    const Clock::time_point end = Clock::now() + 5s;
    for (;;) {
      vkc::Result<bool> r = client->next(10ms, &record, nullptr);
      if (!r.ok() || *r || Clock::now() >= end) return r;
    }
  };
  vkc::Result<bool> r = held();
  CHECK(r.ok() && *r && record.info.seq == 0 && record.size == a.size());
  Bytes slot(a.size() + 8, 0xEE);
  CHECK(client->read(record, slot.data(), a.size() - 1).domain() ==
        Code::InvalidArgument);
  CHECK(client->read(record, slot.data(), slot.size()).ok());
  CHECK(Bytes(slot.begin(), slot.begin() + std::ptrdiff_t(a.size())) == a);
  CHECK(slot[a.size()] == 0xEE);
  r = held();
  CHECK(r.ok() && *r && record.info.seq == 1 && record.size == b.size());
  CHECK(client->read(record, slot.data(), slot.size()).ok());
  CHECK(Bytes(slot.begin(), slot.begin() + std::ptrdiff_t(b.size())) == b);
  r = held();
  CHECK(!r.ok() && r.status().domain() == Code::InvalidArgument);
  CHECK(net::client_reason(r.status().message(), sender.listener.port)
            .rfind("ChunkParser: ", 0) == 0);
  const vkc::Status after = client->read(record, slot.data(), slot.size());
  CHECK(after.domain() == Code::InvalidArgument &&
        after.message() == r.status().message());
  return 0;
}

// The end after whole records: they are read first, then IoError; and after
// half of one, the same.
int end_case() {
  for (const bool cut : {false, true}) {
    Sender sender;
    auto client = net::connect(sender.listener.port);
    CHECK(client != nullptr && sender.take(*client));
    const Bytes payload = vrst::pattern(1000, 7);
    Bytes bytes = concat({session_bytes(),
                          vrst::record(0, 1, 0, 0, stream::kNoSet, payload),
                          vrst::record(0, 0, 1, 1, stream::kNoSet, payload)});
    if (cut) {
      const Bytes third = vrst::record(0, 0, 2, 2, stream::kNoSet, payload);
      bytes.insert(bytes.end(), third.begin(), third.begin() + 500);
    }
    CHECK(net::send_bytes(sender.peer.get(), bytes));
    sender.peer.reset();
    Got got;
    for (std::uint64_t seq = 0; seq < 2; ++seq) {
      const vkc::Result<bool> r = settle(*client, &got);
      CHECK(r.ok() && *r && got.record.info.seq == seq);
      CHECK(got.payload == payload);
    }
    const vkc::Result<bool> r = settle(*client, &got);
    CHECK(!r.ok() && r.status().domain() == Code::IoError);
    CHECK(net::client_reason(r.status().message(), sender.listener.port) ==
          "the sender closed the connection");
  }
  return 0;
}

// A sender that closes before the session, as one serving its most viewers
// does, is told from one that ended.
int closed_before_session_case() {
  Sender sender;
  auto client = net::connect(sender.listener.port);
  CHECK(client != nullptr && sender.take(*client));
  sender.peer.reset();
  Got got;
  const vkc::Result<bool> r = settle(*client, &got);
  CHECK(!r.ok() && r.status().domain() == Code::IoError);
  CHECK(net::client_reason(r.status().message(), sender.listener.port) ==
        "the sender closed the connection before its session (it may "
        "already serve its most viewers)");
  return 0;
}

// With nothing coming, next(20 ms) returns false after about 20 ms.
int wait_case() {
  Sender sender;
  auto client = net::connect(sender.listener.port);
  CHECK(client != nullptr && sender.take(*client));
  CHECK(net::send_bytes(sender.peer.get(), session_bytes()));
  CHECK(net::wait_session(*client));
  Got got;
  for (int i = 0; i < 5; ++i) {
    const Clock::time_point start = Clock::now();
    const vkc::Result<bool> r = client->next(20ms, &got.record, &got.payload);
    const auto took = Clock::now() - start;
    CHECK(r.ok() && !*r);
    CHECK(took >= 19ms && took < 500ms);
  }
  const Clock::time_point start = Clock::now();
  const vkc::Result<bool> r = client->next(0ms, &got.record, &got.payload);
  CHECK(r.ok() && !*r && Clock::now() - start < 100ms);
  return 0;
}

// Chunks the client skips, sent faster than it parses them, never hold next
// past its wait: each next(20 ms) returns false within 500 ms while they come
// for 2 s.
int flood_case() {
  Sender sender;
  auto client = net::connect(sender.listener.port);
  CHECK(client != nullptr && sender.take(*client));
  CHECK(net::send_bytes(sender.peer.get(), session_bytes()));
  CHECK(net::wait_session(*client));
  // Records setting flag bit 1, which this version does not know, with a
  // 64-byte body, sent 1024 at a time.
  const Bytes one =
      vrst::record(0, 2, 0, 0, stream::kNoSet, vrst::pattern(64, 3));
  Bytes skipped;
  for (int i = 0; i < 1024; ++i) {
    skipped.insert(skipped.end(), one.begin(), one.end());
  }
  std::atomic<bool> stop{false};
  std::atomic<bool> ended{false};
  const Clock::time_point end = Clock::now() + 2s;
  std::thread flood([&] {
    while (!stop && Clock::now() < end &&
           net::send_bytes(sender.peer.get(), skipped)) {
    }
    ended = true;
  });
  // Checked once the flood is joined.
  bool all_skipped = true;
  bool prompt = true;
  Got got;
  for (int i = 0; i < 5; ++i) {
    const Clock::time_point start = Clock::now();
    const vkc::Result<bool> r = client->next(20ms, &got.record, &got.payload);
    all_skipped = all_skipped && r.ok() && !*r;
    prompt = prompt && Clock::now() - start < 500ms;
  }
  stop = true;
  // Read on until the flood ends, so a send under way finishes.
  while (!ended && client->next(10ms, &got.record, &got.payload).ok()) {
  }
  flood.join();
  CHECK(all_skipped);
#ifndef VKC_TEST_SANITIZED
  // Under ASan or TSan, the calls are checked without the time limit.
  CHECK(prompt);
#endif
  return 0;
}

// next(0 ms) alone moves: a client calling only it connects and reads a
// record; and one call returns a key-sized record (230 KB) already whole in
// the socket.
int zero_wait_case() {
  Sender sender;
  auto client = net::connect(sender.listener.port);
  CHECK(client != nullptr && sender.take(*client));
  CHECK(net::send_bytes(
      sender.peer.get(),
      concat({session_bytes(), vrst::record(0, 1, 0, 0, stream::kNoSet,
                                            vrst::pattern(1000, 7))})));
  Got got;
  vkc::Result<bool> r = false;
  const Clock::time_point end = Clock::now() + 5s;
  while (r.ok() && !*r && Clock::now() < end) {
    r = client->next(0ms, &got.record, &got.payload);
  }
  CHECK(r.ok() && *r && got.record.info.seq == 0);
  CHECK(net::send_bytes(
      sender.peer.get(),
      vrst::record(0, 0, 1, 1, stream::kNoSet, vrst::pattern(230'000, 8))));
  std::this_thread::sleep_for(200ms);  // on loopback, in the socket by now
  r = client->next(0ms, &got.record, &got.payload);
  CHECK(r.ok() && *r && got.record.info.seq == 1);
  return 0;
}

// A partial record is progress even though next returns false. Each piece is
// drained before another goes, and a later idle step in the same call must
// not erase its progress. A separate idle call resets it. Records sent
// together report progress too, including one already buffered by the call
// before it; no receive is required to return a record.
int progress_case() {
  Sender sender;
  auto client = net::connect(sender.listener.port);
  CHECK(client != nullptr);
  Got got;
  bool progressed = false;
  vkc::Result<bool> r =
      client->next(0ms, &got.record, &got.payload, &progressed);
  CHECK(r.ok() && !*r && progressed);  // starting the connect is progress
  sender.peer = net::accept_within(sender.listener.fd.get(), 2s);
  CHECK(sender.peer.valid());
  CHECK(net::send_bytes(sender.peer.get(), session_bytes()));
  CHECK(net::wait_session(*client));

  const Bytes payload = vrst::pattern(512 * 1024, 7);
  const Bytes record = vrst::record(0, 1, 0, 0, stream::kNoSet, payload);
  constexpr std::size_t kPiece = 128 * 1024;
  for (std::size_t begin = 0; begin < record.size(); begin += kPiece) {
    const std::size_t end = std::min(begin + kPiece, record.size());
    const Bytes piece(record.data() + begin, record.data() + end);
    CHECK(net::send_bytes(sender.peer.get(), piece));
    progressed = false;
    r = client->next(20ms, &got.record, &got.payload, &progressed);
    CHECK(r.ok() && progressed);
    CHECK(*r == (end == record.size()));
    if (!*r) {
      r = client->next(0ms, &got.record, &got.payload, &progressed);
      CHECK(r.ok() && !*r && !progressed);
    }
  }
  CHECK(got.record.info.seq == 0 && got.payload == payload);

  const Bytes small = vrst::pattern(64, 9);
  CHECK(net::send_bytes(
      sender.peer.get(),
      concat({vrst::record(0, 0, 1, 1, stream::kNoSet, small),
              vrst::record(0, 0, 2, 2, stream::kNoSet, small)})));
  for (std::uint64_t seq = 1; seq <= 2; ++seq) {
    progressed = false;
    r = client->next(20ms, &got.record, &got.payload, &progressed);
    CHECK(r.ok() && *r && progressed && got.record.info.seq == seq);
    CHECK(got.payload == small);
  }
  progressed = true;
  r = client->next(0ms, &got.record, &got.payload, &progressed);
  CHECK(r.ok() && !*r && !progressed);
  return 0;
}

// The progress output is reset even when copying a saved error allocates and
// fails. The overload still contains the exception and returns OutOfMemory.
int progress_oom_case() {
  Sender sender;
  auto client = net::connect(sender.listener.port);
  CHECK(client != nullptr && sender.take(*client));
  CHECK(net::send_bytes(sender.peer.get(), Bytes(64, 'x')));
  Got got;
  CHECK(!settle(*client, &got).ok());
  auto& failure = vkc::test::allocation_failure;
  failure = {};
  failure.measure = true;
  const vkc::Result<bool> measured =
      client->next(0ms, &got.record, &got.payload);
  failure.measure = false;
  CHECK(!measured.ok() && failure.bytes != 0);
  failure.site = failure.measured_site;
  failure.armed = true;
  bool progressed = true;
  const vkc::Result<bool> r =
      client->next(0ms, &got.record, &got.payload, &progressed);
  const bool injected = failure.injected;
  failure = {};
  CHECK(injected && !r.ok() && r.status().domain() == Code::OutOfMemory);
  CHECK(!progressed);
  return 0;
}

// idle_limit counts from connect: a listener that sends nothing ends the
// connection after it, as connected with no session; so does one that sends
// the session, then nothing, as no byte.
int idle_case() {
  networking::ClientOptions options;
  options.idle_limit = 300ms;
  for (const bool session : {false, true}) {
    Sender sender;
    const Clock::time_point start = Clock::now();
    auto client = net::connect(sender.listener.port, options);
    CHECK(client != nullptr && sender.take(*client));
    if (session) {
      CHECK(net::send_bytes(sender.peer.get(), session_bytes()));
    }
    Got got;
    vkc::Result<bool> r = false;
    while (r.ok() && Clock::now() - start < 5s) {
      r = client->next(50ms, &got.record, &got.payload);
      CHECK(!r.ok() || !*r);
    }
    const auto took = Clock::now() - start;
    CHECK(!r.ok() && r.status().domain() == Code::IoError);
    CHECK(took >= 300ms && took < 2s);
    CHECK(client->session() == nullptr || session);
    CHECK(net::client_reason(r.status().message(), sender.listener.port) ==
          (session ? "no byte for 300 ms"
                   : "connected, but no session within 300 ms"));
  }
  return 0;
}

// A connect that never finishes ends at idle_limit as a connect, not as a
// sender that stalled. macOS drops a SYN to a socket bound and not listening,
// which Linux refuses; Linux drops one to a listener whose queue is full,
// which macOS resets. So the port is bound, on Linux listening with a queue
// of one, and connected to until a connect does not finish.
int connect_within_case() {
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
  networking::ClientOptions options;
  options.idle_limit = 300ms;
  const Clock::time_point start = Clock::now();
  auto client = net::connect(port, options);
  CHECK(client != nullptr);
  Got got;
  const vkc::Result<bool> r = settle(*client, &got);
  const auto took = Clock::now() - start;
  CHECK(!r.ok() && r.status().domain() == Code::IoError);
  CHECK(took >= 300ms && took < 2s);
  CHECK(net::client_reason(r.status().message(), port) ==
        "cannot connect within 300 ms");
  return 0;
}

// idle_limit counts from the last byte: a record every 50 ms for 1 s keeps a
// client whose limit is 300 ms; once they stop, it ends 300 ms to 2 s after
// the last.
int idle_from_last_byte_case() {
  networking::ClientOptions options;
  options.idle_limit = 300ms;
  Sender sender;
  auto client = net::connect(sender.listener.port, options);
  CHECK(client != nullptr && sender.take(*client));
  CHECK(net::send_bytes(sender.peer.get(), session_bytes()));
  const Bytes payload = vrst::pattern(100, 5);
  Got got;
  const Clock::time_point start = Clock::now();
  Clock::time_point last = start;
  for (std::uint64_t seq = 0; Clock::now() - start < 1s; ++seq) {
    CHECK(net::send_bytes(
        sender.peer.get(),
        vrst::record(0, seq == 0 ? 1 : 0, seq, seq, stream::kNoSet, payload)));
    last = Clock::now();
    const vkc::Result<bool> r = settle(*client, &got);
    CHECK(r.ok() && *r && got.record.info.seq == seq);
    std::this_thread::sleep_for(50ms);
  }
  const vkc::Result<bool> r = settle(*client, &got);
  const auto quiet = Clock::now() - last;
  CHECK(!r.ok() && r.status().domain() == Code::IoError);
  CHECK(quiet >= 300ms && quiet < 2s);
  CHECK(net::client_reason(r.status().message(), sender.listener.port) ==
        "no byte for 300 ms");
  return 0;
}

// A caller back late takes what came meanwhile before it judges idle_limit,
// a client whose limit is 300 ms not called for 500 ms: a record every
// 100 ms, the first of which next(0) returns; a connect that ended, the
// session and a record sent at once, which next(0) returns, or the call
// after the one that ends the connect; and a refused connect, which the
// error names.
int late_caller_case() {
  networking::ClientOptions options;
  options.idle_limit = 300ms;
  const Bytes payload = vrst::pattern(100, 6);
  {
    Sender sender;
    auto client = net::connect(sender.listener.port, options);
    CHECK(client != nullptr && sender.take(*client));
    CHECK(net::send_bytes(sender.peer.get(), session_bytes()));
    CHECK(net::wait_session(*client));
    for (std::uint64_t seq = 0; seq < 5; ++seq) {
      std::this_thread::sleep_for(100ms);
      CHECK(net::send_bytes(sender.peer.get(),
                            vrst::record(0, seq == 0 ? 1 : 0, seq, seq,
                                         stream::kNoSet, payload)));
    }
    Got got;
    const vkc::Result<bool> r = client->next(0ms, &got.record, &got.payload);
    CHECK(r.ok() && *r && got.record.info.seq == 0);
  }
  {
    Sender sender;  // whose take makes one next(0), starting the connect
    auto client = net::connect(sender.listener.port, options);
    CHECK(client != nullptr && sender.take(*client));
    CHECK(net::send_bytes(
        sender.peer.get(),
        concat({session_bytes(),
                vrst::record(0, 1, 0, 0, stream::kNoSet, payload)})));
    std::this_thread::sleep_for(500ms);
    Got got;
    vkc::Result<bool> r = client->next(0ms, &got.record, &got.payload);
    if (r.ok() && !*r) r = client->next(0ms, &got.record, &got.payload);
    CHECK(r.ok() && *r && got.record.info.seq == 0);
  }
  {
    const std::uint16_t port = net::closed_port();
    CHECK(port != 0);
    auto client = net::connect(port, options);
    CHECK(client != nullptr);
    Got got;
    vkc::Result<bool> r = client->next(0ms, &got.record, &got.payload);
    CHECK(r.ok() && !*r);
    std::this_thread::sleep_for(500ms);
    for (int i = 0; i < 3 && r.ok(); ++i) {
      r = client->next(0ms, &got.record, &got.payload);
    }
    CHECK(!r.ok() && net::client_reason(r.status().message(), port)
                             .rfind("cannot connect: ", 0) == 0);
  }
  return 0;
}

// milliseconds::max() means never: as idle_limit it never ends a connection
// to a sender that sends nothing, and as next's wait it waits until a record
// comes, 100 ms later.
int never_case() {
  constexpr std::chrono::milliseconds kNever = std::chrono::milliseconds::max();
  networking::ClientOptions options;
  options.idle_limit = kNever;
  Sender sender;
  auto client = net::connect(sender.listener.port, options);
  CHECK(client != nullptr && sender.take(*client));
  Got got;
  for (int i = 0; i < 3; ++i) {
    const vkc::Result<bool> r = client->next(50ms, &got.record, &got.payload);
    CHECK(r.ok() && !*r);
  }
  CHECK(net::send_bytes(sender.peer.get(), session_bytes()));
  std::thread later([&] {
    std::this_thread::sleep_for(100ms);
    (void)net::send_bytes(
        sender.peer.get(),
        vrst::record(0, 1, 0, 0, stream::kNoSet, vrst::pattern(64, 9)));
  });
  const Clock::time_point start = Clock::now();
  const vkc::Result<bool> r = client->next(kNever, &got.record, &got.payload);
  const auto took = Clock::now() - start;
  later.join();
  CHECK(r.ok() && *r && got.record.info.seq == 0);
  CHECK(took >= 50ms);
  return 0;
}

// localhost reaches a server on 127.0.0.1, through the lookup's thread.
int localhost_case() {
  auto server = net::serve(net::geometry_session());
  CHECK(server != nullptr);
  auto client = net::connect(server->port(), {}, "localhost");
  CHECK(client != nullptr);
  CHECK(net::wait_session(*client));
  return 0;
}

// Clients of localhost destroyed while their lookups run, half before any
// next and half after next(0), leave the lookups to end alone: a lookup
// writing to a client gone would fail, under ASan and TSan or on its mutex.
// Every core is kept busy meanwhile, so a lookup's thread waits to run.
int abandoned_lookup_case() {
  const std::uint16_t port = net::closed_port();
  CHECK(port != 0);
  std::atomic<bool> stop{false};
  std::vector<std::thread> busy;
  for (unsigned i = 0; i < std::max(2u, std::thread::hardware_concurrency());
       ++i) {
    busy.emplace_back([&stop] {
      while (!stop) {
      }
    });
  }
  bool made = true;  // checked once the busy threads are joined
  for (int i = 0; i < 60 && made; ++i) {
    auto client = net::connect(port, {}, "localhost");
    made = client != nullptr;
    if (made && i % 2 == 1) {
      Got got;
      made = client->next(0ms, &got.record, &got.payload).ok();
    }
  }
  stop = true;
  for (std::thread& t : busy) t.join();
  CHECK(made);
  std::this_thread::sleep_for(200ms);  // the lookups end meanwhile
  return 0;
}

int refusals_case() {
  CHECK(networking::StreamClient::connect("", 47801).status().domain() ==
        Code::InvalidArgument);
  CHECK(networking::StreamClient::connect("127.0.0.1", 0).status().domain() ==
        Code::InvalidArgument);
  for (const std::chrono::milliseconds limit : {0ms, -1ms}) {
    networking::ClientOptions options;
    options.idle_limit = limit;
    CHECK(networking::StreamClient::connect("127.0.0.1", 47801, options)
              .status()
              .domain() == Code::InvalidArgument);
  }
  return 0;
}

}  // namespace

int main() {
  if (refused_case() != 0) return 1;
  if (not_a_stream_case() != 0) return 1;
  if (huge_header_case() != 0) return 1;
  if (corrupt_body_case() != 0) return 1;
  if (read_into_case() != 0) return 1;
  if (end_case() != 0) return 1;
  if (closed_before_session_case() != 0) return 1;
  if (wait_case() != 0) return 1;
  if (flood_case() != 0) return 1;
  if (zero_wait_case() != 0) return 1;
  if (progress_case() != 0) return 1;
  if (progress_oom_case() != 0) return 1;
  if (idle_case() != 0) return 1;
  if (connect_within_case() != 0) return 1;
  if (idle_from_last_byte_case() != 0) return 1;
  if (late_caller_case() != 0) return 1;
  if (never_case() != 0) return 1;
  if (localhost_case() != 0) return 1;
  if (abandoned_lookup_case() != 0) return 1;
  if (refusals_case() != 0) return 1;
  std::puts("networking client: OK");
  return 0;
}
