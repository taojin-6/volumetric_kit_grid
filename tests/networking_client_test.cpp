// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// StreamClient against raw senders on loopback (127.0.0.1, port 0) that send
// any bytes: a refused port is IoError; HTTP's bytes are InvalidArgument; a
// header claiming 1 GiB is refused before any of its body comes (a RECD's or
// another tag's OutOfMemory, a VRST's InvalidArgument); a corrupt body is
// InvalidArgument; the end after whole records is IoError once they are all
// read; next(20 ms) with nothing returns false in about 20 ms, and within
// 500 ms while chunks it skips flood in; next(0 ms) alone connects and reads,
// a 230 KB record already in the socket in one call; idle_limit counts from
// connect, against a listener that sends nothing; localhost reaches a
// 127.0.0.1 server; and an empty host or port 0 is refused. Each error is
// returned again by every later call.

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
#include "test_check.hpp"

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

// The error a client gets from `bytes`, the connection left open; the same
// error from the call after.
Code error_from(const Bytes& bytes) {
  Sender sender;
  auto client = net::connect(sender.listener.port);
  if (client == nullptr || !sender.take(*client) ||
      !net::send_bytes(sender.peer.get(), bytes)) {
    return Code::Ok;
  }
  Got got;
  const vkc::Result<bool> first = settle(*client, &got);
  if (first.ok()) return Code::Ok;
  const vkc::Result<bool> again = client->next(0ms, &got.record, &got.payload);
  if (again.ok() || again.status().domain() != first.status().domain() ||
      again.status().message() != first.status().message()) {
    return Code::Ok;
  }
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
  CHECK(r.status().message().find("127.0.0.1:" + std::to_string(port)) !=
        std::string::npos);
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

int corrupt_body_case() {
  Bytes record = vrst::record(0, 1, 0, 0, stream::kNoSet, vrst::pattern(64, 1));
  record[56 + 10] ^= 0x20;
  CHECK(error_from(concat({session_bytes(), record})) == Code::InvalidArgument);
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
  }
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
  bool prompt = true;
  Got got;
  for (int i = 0; i < 5; ++i) {
    const Clock::time_point start = Clock::now();
    const vkc::Result<bool> r = client->next(20ms, &got.record, &got.payload);
    prompt = prompt && r.ok() && !*r && Clock::now() - start < 500ms;
  }
  stop = true;
  // Read on until the flood ends, so a send under way finishes.
  while (!ended && client->next(10ms, &got.record, &got.payload).ok()) {
  }
  flood.join();
  CHECK(prompt);
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

// idle_limit counts from connect: a listener that sends nothing ends the
// connection after it; so does one that sends the session, then nothing.
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
  }
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

int refusals_case() {
  CHECK(networking::StreamClient::connect("", 47801).status().domain() ==
        Code::InvalidArgument);
  CHECK(networking::StreamClient::connect("127.0.0.1", 0).status().domain() ==
        Code::InvalidArgument);
  return 0;
}

}  // namespace

int main() {
  if (refused_case() != 0) return 1;
  if (not_a_stream_case() != 0) return 1;
  if (huge_header_case() != 0) return 1;
  if (corrupt_body_case() != 0) return 1;
  if (end_case() != 0) return 1;
  if (wait_case() != 0) return 1;
  if (flood_case() != 0) return 1;
  if (zero_wait_case() != 0) return 1;
  if (idle_case() != 0) return 1;
  if (localhost_case() != 0) return 1;
  if (refusals_case() != 0) return 1;
  std::puts("networking client: OK");
  return 0;
}
