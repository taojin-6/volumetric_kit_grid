// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// StreamServer and StreamClient on loopback (127.0.0.1, port 0): a raw
// viewer's bytes are StreamWriter's file for the same session and records,
// less its footer; a client reads the session, then every record's info,
// payload and file offset; a viewer joining gets the GOP from its key, and
// before any key nothing; a viewer that does not read while 3 GOPs of 1 MiB
// records are published gets whole chunks, a key after every jump and the
// newest GOP whole, with every publish back at once; a GOP past
// max_gop_bytes sends nothing until the next key; a viewer that takes no byte
// is closed after stall_limit; connection max_viewers + 1 is closed before
// any byte; a viewer that leaves never fails publish; the writer's refusals
// send nothing; the destructor returns within 1 s with no viewers, idle ones
// and a stalled one; and 100 SIGUSR1 to a no-op handler leave the stream
// whole. Every test that publishes first waits for the viewer's session.

#include <pthread.h>
#include <signal.h>
#include <sys/socket.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#include "networking_loopback.hpp"
#include "scratch_dir.hpp"
#include "stream_files.hpp"
#include "test_check.hpp"
#include "volumetric_kit/core/base/log.hpp"
#include "volumetric_kit/recon/stream/stream_reader.hpp"
#include "volumetric_kit/recon/stream/stream_writer.hpp"

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

struct Published {
  stream::RecordInfo info;
  Bytes payload;
};

// Keys and P-frames of every padding, an empty payload and payloads past
// 64 KiB; one GOP after the first key, then a second.
std::vector<Published> some_records() {
  const std::size_t sizes[] = {1000, 0, 1, 7, 8, 63, 65537, 4096, 300000, 5};
  std::vector<Published> out;
  for (std::uint64_t seq = 0; seq < std::size(sizes); ++seq) {
    const bool key = seq == 0 || seq == 6;
    out.push_back(
        {net::record_info(seq, key), net::record_payload(seq, sizes[seq])});
  }
  return out;
}

// What a log handler saw, for the tests that read the server's log.
struct Captured {
  std::mutex mutex;
  std::vector<std::string> lines;

  bool has(std::string_view a, std::string_view b = {}) {
    const std::lock_guard<std::mutex> lock(mutex);
    for (const std::string& line : lines) {
      if (line.find(a) != std::string::npos &&
          line.find(b) != std::string::npos) {
        return true;
      }
    }
    return false;
  }
  bool wait_for(std::string_view a, std::string_view b,
                std::chrono::milliseconds timeout) {
    const Clock::time_point end = Clock::now() + timeout;
    while (!has(a, b)) {
      if (Clock::now() >= end) return false;
      std::this_thread::sleep_for(5ms);
    }
    return true;
  }
};

// Installs a handler that keeps every line, until it goes.
struct LogCapture {
  explicit LogCapture(Captured* into) {
    vkc::set_log_handler(
        [into](vkc::LogLevel, std::string_view, std::string_view message) {
          const std::lock_guard<std::mutex> lock(into->mutex);
          into->lines.emplace_back(message);
        });
  }
  ~LogCapture() { vkc::set_log_handler({}); }
  LogCapture(const LogCapture&) = delete;
  LogCapture& operator=(const LogCapture&) = delete;
};

std::string peer_name(int fd) {
  return "127.0.0.1:" + std::to_string(net::local_port(fd));
}

// --- The wire.
// ----------------------------------------------------------------

// A raw viewer's bytes are the writer's file less its footer, and the whole of
// an unfinished writer's file. Each record is published once the viewer has
// the one before, so none is skipped.
int wire_is_the_file_case(const std::string& dir) {
  const stream::Session session = net::geometry_session();
  const std::vector<Published> records = some_records();
  auto server = net::serve(session);
  CHECK(server != nullptr);
  const vkc::UniqueFd viewer = net::connect_loopback(server->port());
  CHECK(viewer.valid());
  Bytes wire;
  const std::size_t header = net::session_chunk(session).size();
  CHECK(header > 0);
  net::read_upto(viewer.get(), header, 5s, &wire);
  CHECK(wire.size() == header);

  const std::string finished = dir + "/wire_finished.vrst";
  const std::string unfinished = dir + "/wire_unfinished.vrst";
  auto writer = stream::StreamWriter::create(finished, session);
  auto killed = stream::StreamWriter::create(unfinished, session);
  CHECK(writer.ok() && killed.ok());
  for (const Published& r : records) {
    CHECK((*writer)->append(r.info, r.payload.data(), r.payload.size()).ok());
    CHECK((*killed)->append(r.info, r.payload.data(), r.payload.size()).ok());
    CHECK(server->publish(r.info, r.payload.data(), r.payload.size()).ok());
    const std::size_t want = (*writer)->size();
    net::read_upto(viewer.get(), want, 5s, &wire);
    CHECK(wire.size() == want);
  }
  CHECK((*writer)->finish().ok());
  killed->reset();
  const Bytes file = vr_test::read_bytes(finished);
  CHECK(vrst::index_offset(file) == wire.size());
  CHECK(Bytes(file.begin(),
              file.begin() + static_cast<std::ptrdiff_t>(wire.size())) == wire);
  CHECK(vr_test::read_bytes(unfinished) == wire);
  return 0;
}

// A client reads the session, then each record's info and payload, its
// offset the file's.
int round_trip_case(const std::string& dir) {
  stream::Session session = net::geometry_session();
  session.calibration = "{\"device_calibration\": {}}";
  session.tracks[0].params = "params";
  const std::vector<Published> records = some_records();
  auto server = net::serve(session);
  CHECK(server != nullptr);
  auto client = net::connect(server->port());
  CHECK(client != nullptr);
  CHECK(client->session() == nullptr);
  CHECK(net::wait_session(*client));
  CHECK(vrst::same(*client->session(), session));

  const std::string path = dir + "/round_trip.vrst";
  {
    auto writer = stream::StreamWriter::create(path, session);
    CHECK(writer.ok());
    for (const Published& r : records) {
      CHECK((*writer)->append(r.info, r.payload.data(), r.payload.size()).ok());
    }
    CHECK((*writer)->finish().ok());
  }
  const auto reader = vrst::open(path);
  CHECK(reader != nullptr);
  const vrst::ReadBack file = vrst::read_all(*reader);
  CHECK(file.records.size() == records.size());
  for (std::size_t i = 0; i < records.size(); ++i) {
    const Published& r = records[i];
    CHECK(server->publish(r.info, r.payload.data(), r.payload.size()).ok());
    Got got;
    CHECK(net::next_within(*client, &got));
    CHECK(vrst::same(got.record.info, r.info));
    CHECK(got.payload == r.payload);
    CHECK(got.record.size == r.payload.size());
    CHECK(got.record.offset == file.records[i].offset);
  }
  return 0;
}

// --- Joining.
// -----------------------------------------------------------------

// A viewer joining after K P P gets K P P, then live; after a second key it
// starts there; before any key it gets nothing until one.
int join_case() {
  const stream::Session session = net::geometry_session();
  auto server = net::serve(session);
  CHECK(server != nullptr);
  std::vector<Bytes> payloads;
  for (std::uint64_t seq = 0; seq < 8; ++seq) {
    payloads.push_back(net::record_payload(seq, 100 + seq));
  }
  const auto publish = [&](std::uint64_t seq, bool key) {
    return server
        ->publish(net::record_info(seq, key), payloads[seq].data(),
                  payloads[seq].size())
        .ok();
  };
  // Published before the viewer connects, so read at once: the session
  // comes with the first.
  CHECK(publish(0, true));
  CHECK(publish(1, false));
  CHECK(publish(2, false));
  auto client = net::connect(server->port());
  CHECK(client != nullptr);
  for (std::uint64_t seq = 0; seq < 3; ++seq) {
    Got got;
    CHECK(net::next_within(*client, &got));
    CHECK(client->session() != nullptr);
    CHECK(got.record.info.seq == seq && got.record.info.key == (seq == 0));
    CHECK(got.payload == payloads[seq]);
  }
  CHECK(publish(3, false));
  Got got;
  CHECK(net::next_within(*client, &got));
  CHECK(got.record.info.seq == 3 && got.payload == payloads[3]);

  CHECK(publish(4, true));
  CHECK(publish(5, false));
  auto late = net::connect(server->port());
  CHECK(late != nullptr);
  CHECK(net::next_within(*late, &got));
  CHECK(got.record.info.seq == 4 && got.record.info.key);
  CHECK(net::next_within(*late, &got));
  CHECK(got.record.info.seq == 5);
  for (std::uint64_t seq = 4; seq < 6; ++seq) {
    CHECK(net::next_within(*client, &got));
    CHECK(got.record.info.seq == seq);
  }

  // Before any key: nothing, then the key.
  auto fresh = net::serve(session);
  CHECK(fresh != nullptr);
  auto early = net::connect(fresh->port());
  CHECK(early != nullptr);
  CHECK(net::wait_session(*early));
  CHECK(fresh
            ->publish(net::record_info(0, false), payloads[0].data(),
                      payloads[0].size())
            .ok());
  CHECK(fresh
            ->publish(net::record_info(1, false), payloads[1].data(),
                      payloads[1].size())
            .ok());
  CHECK(!net::next_within(*early, &got, 200ms));
  CHECK(fresh
            ->publish(net::record_info(2, true), payloads[2].data(),
                      payloads[2].size())
            .ok());
  CHECK(net::next_within(*early, &got));
  CHECK(got.record.info.seq == 2 && got.record.info.key);
  return 0;
}

// --- Newest key wins.
// ---------------------------------------------------------

// A viewer that stops reading while 3 GOPs of 1 MiB records are published,
// then drains: whole chunks (each passes its CRCs), seq rising, a key after
// every jump, the newest GOP whole and some of the stream skipped; every
// publish returned at once.
int newest_key_wins_case() {
  constexpr std::uint64_t kGop = 4;
  constexpr std::uint64_t kRecords = 3 * kGop;
  const stream::Session session = net::geometry_session();
  auto server = net::serve(session);
  CHECK(server != nullptr);
  auto client = net::connect(server->port());
  CHECK(client != nullptr);
  CHECK(net::wait_session(*client));
  for (std::uint64_t seq = 0; seq < kRecords; ++seq) {
    const Bytes payload = net::record_payload(seq, 1 << 20);
    const Clock::time_point start = Clock::now();
    CHECK(server
              ->publish(net::record_info(seq, seq % kGop == 0), payload.data(),
                        payload.size())
              .ok());
    CHECK(Clock::now() - start < 1s);
  }
  std::vector<stream::RecordInfo> got;
  for (Got g; got.empty() || got.back().seq + 1 < kRecords;) {
    CHECK(net::next_within(*client, &g));
    CHECK(g.payload == net::record_payload(g.record.info.seq, 1 << 20));
    got.push_back(g.record.info);
  }
  CHECK(got.front().seq == 0 && got.front().key);
  for (std::size_t i = 1; i < got.size(); ++i) {
    CHECK(got[i].seq > got[i - 1].seq);
    if (got[i].seq != got[i - 1].seq + 1) CHECK(got[i].key);
  }
  CHECK(got.size() >= kGop);
  for (std::uint64_t k = 0; k < kGop; ++k) {
    CHECK(got[got.size() - kGop + k].seq == kRecords - kGop + k);
  }
  // Under the low-water mark the kernel holds well under 1 MiB of a viewer
  // that does not read, so the first key was still under way when the rest
  // came: it is finished, then the newest GOP follows.
  CHECK(got.size() == 1 + kGop);
  std::printf("newest key wins: got %zu of %llu records\n", got.size(),
              static_cast<unsigned long long>(kRecords));
  return 0;
}

// --- Limits.
// ------------------------------------------------------------------

// A record that would take the GOP past max_gop_bytes drops the GOP: nothing
// more until the next key, for a live viewer and one joining meanwhile; a key
// past it alone likewise.
int gop_limit_case() {
  const stream::Session session = net::geometry_session();
  networking::ServerOptions options;
  options.max_gop_bytes = 3 * (56 + 1000);
  auto server = net::serve(session, options);
  CHECK(server != nullptr);
  auto live = net::connect(server->port());
  CHECK(live != nullptr);
  CHECK(net::wait_session(*live));
  const auto publish = [&](std::uint64_t seq, bool key, std::size_t size) {
    const Bytes payload = net::record_payload(seq, size);
    return server->publish(net::record_info(seq, key), payload.data(),
                           payload.size());
  };
  Got got;
  CHECK(publish(0, true, 1000).ok());
  CHECK(net::next_within(*live, &got) && got.record.info.seq == 0);
  CHECK(publish(1, false, 1000).ok());
  CHECK(net::next_within(*live, &got) && got.record.info.seq == 1);
  CHECK(publish(2, false, 1001).ok());  // one byte past: the GOP goes
  CHECK(publish(3, false, 10).ok());
  auto joined = net::connect(server->port());
  CHECK(joined != nullptr);
  CHECK(net::wait_session(*joined));
  CHECK(!net::next_within(*live, &got, 200ms));
  CHECK(!net::next_within(*joined, &got, 50ms));
  CHECK(publish(4, true, 1000).ok());
  CHECK(publish(5, false, 1000).ok());
  for (auto* client : {live.get(), joined.get()}) {
    CHECK(net::next_within(*client, &got) && got.record.info.seq == 4);
    CHECK(net::next_within(*client, &got) && got.record.info.seq == 5);
  }
  // A key past the limit alone.
  CHECK(publish(6, true, 3 * 1056).ok());
  CHECK(publish(7, false, 10).ok());
  CHECK(!net::next_within(*live, &got, 200ms));
  CHECK(publish(8, true, 10).ok());
  CHECK(net::next_within(*live, &got) && got.record.info.seq == 8);
  CHECK(net::next_within(*joined, &got) && got.record.info.seq == 8);
  return 0;
}

// A viewer that takes no byte while its socket is full is closed after
// stall_limit (200 ms), logged at Info, and its bytes then end; a reading
// viewer meanwhile gets every record.
int stall_limit_case() {
  Captured log;
  const LogCapture capture(&log);
  const stream::Session session = net::geometry_session();
  networking::ServerOptions options;
  options.stall_limit = 200ms;
  auto server = net::serve(session, options);
  CHECK(server != nullptr);
  auto reading = net::connect(server->port());
  CHECK(reading != nullptr);
  CHECK(net::wait_session(*reading));
  const vkc::UniqueFd stalled = net::connect_loopback(server->port(), 4096);
  CHECK(stalled.valid());
  Bytes header;
  net::read_upto(stalled.get(), net::session_chunk(session).size(), 5s,
                 &header);
  CHECK(header.size() == net::session_chunk(session).size());
  const std::string name = peer_name(stalled.get());
  CHECK(log.wait_for(name, "joined", 2s));

  const Clock::time_point start = Clock::now();
  std::uint64_t seq = 0;
  Got got;
  while (!log.has(name, "took no byte") && Clock::now() - start < 5s) {
    // A key every 30, so the GOP stays under its limit however long this
    // runs; the reading viewer, kept up, skips nothing.
    const Bytes payload = net::record_payload(seq, 256 << 10);
    CHECK(server
              ->publish(net::record_info(seq, seq % 30 == 0), payload.data(),
                        payload.size())
              .ok());
    CHECK(net::next_within(*reading, &got) && got.record.info.seq == seq);
    ++seq;
    std::this_thread::sleep_for(10ms);
  }
  const auto closed_after = Clock::now() - start;
  CHECK(log.has(name, "took no byte for 200 ms"));
  CHECK(closed_after >= 200ms);
  // What it was sent ends: the server closed it.
  Bytes rest;
  bool eof = false;
  net::read_upto(stalled.get(), std::size_t{1} << 30, 5s, &rest, &eof);
  CHECK(eof);
  CHECK(rest.size() < seq * (256 << 10));
  return 0;
}

// Connection max_viewers + 1 is closed before any byte; once a viewer leaves,
// a new one is served.
int max_viewers_case() {
  Captured log;
  const LogCapture capture(&log);
  const stream::Session session = net::geometry_session();
  networking::ServerOptions options;
  options.max_viewers = 2;
  auto server = net::serve(session, options);
  CHECK(server != nullptr);
  auto first = net::connect(server->port());
  auto second = net::connect(server->port());
  CHECK(first != nullptr && second != nullptr);
  CHECK(net::wait_session(*first));
  CHECK(net::wait_session(*second));
  const vkc::UniqueFd third = net::connect_loopback(server->port());
  CHECK(third.valid());
  Bytes bytes;
  bool eof = false;
  net::read_upto(third.get(), 1, 2s, &bytes, &eof);
  CHECK(eof && bytes.empty());
  CHECK(log.wait_for("refused viewer " + peer_name(third.get()), "2 viewers",
                     2s));

  first.reset();
  const std::size_t header = net::session_chunk(session).size();
  bool served = false;
  for (int attempt = 0; attempt < 100 && !served; ++attempt) {
    const vkc::UniqueFd next = net::connect_loopback(server->port());
    CHECK(next.valid());
    Bytes got;
    net::read_upto(next.get(), header, 1s, &got);
    served = got.size() == header;
    if (!served) std::this_thread::sleep_for(10ms);
  }
  CHECK(served);
  return 0;
}

// Viewers that leave mid-stream, by a close or a reset, never fail publish,
// and the one that stays gets every record.
int leaving_viewer_case() {
  Captured log;
  const LogCapture capture(&log);
  const stream::Session session = net::geometry_session();
  auto server = net::serve(session);
  CHECK(server != nullptr);
  auto stays = net::connect(server->port());
  auto leaves = net::connect(server->port());
  CHECK(stays != nullptr && leaves != nullptr);
  CHECK(net::wait_session(*stays));
  CHECK(net::wait_session(*leaves));
  vkc::UniqueFd resets = net::connect_loopback(server->port(), 4096);
  CHECK(resets.valid());
  const std::string resets_name = peer_name(resets.get());
  CHECK(log.wait_for(resets_name, "joined", 2s));
  constexpr std::uint64_t kRecords = 60;
  for (std::uint64_t seq = 0; seq < kRecords; ++seq) {
    const Bytes payload = net::record_payload(seq, 32 << 10);
    CHECK(server
              ->publish(net::record_info(seq, seq == 0), payload.data(),
                        payload.size())
              .ok());
    if (seq == 10) leaves.reset();
    if (seq == 20) {
      const linger hard{1, 0};  // close with a reset
      CHECK(::setsockopt(resets.get(), SOL_SOCKET, SO_LINGER, &hard,
                         sizeof hard) == 0);
      resets.reset();
    }
    std::this_thread::sleep_for(2ms);
  }
  for (std::uint64_t seq = 0; seq < kRecords; ++seq) {
    Got got;
    CHECK(net::next_within(*stays, &got) && got.record.info.seq == seq);
  }
  // The reset viewer's thread ended, as a failure or a leaving.
  const Clock::time_point end = Clock::now() + 2s;
  while (!log.has(resets_name, " failed: ") && !log.has(resets_name, " left") &&
         Clock::now() < end) {
    std::this_thread::sleep_for(5ms);
  }
  CHECK(log.has(resets_name, " failed: ") || log.has(resets_name, " left"));
  return 0;
}

// --- Rules.
// -------------------------------------------------------------------

// The writer's refusals, by the same code, with nothing sent; and start's.
int rules_case() {
  const stream::Session session = net::geometry_session();
  auto server = net::serve(session);
  CHECK(server != nullptr);
  auto client = net::connect(server->port());
  CHECK(client != nullptr);
  CHECK(net::wait_session(*client));
  const Bytes payload = net::record_payload(1, 64);
  const auto publish = [&](const stream::RecordInfo& info,
                           const std::uint8_t* data, std::size_t size) {
    return server->publish(info, data, size).domain();
  };
  CHECK(publish(vrst::info(0, true, 5000, 10), payload.data(), 64) == Code::Ok);
  CHECK(publish(vrst::info(1, false, 6000, 11), payload.data(), 64) ==
        Code::InvalidArgument);  // a track the session lacks
  CHECK(publish(vrst::info(0, false, 6000, 10), payload.data(), 64) ==
        Code::InvalidArgument);  // seq not above the last
  CHECK(publish(vrst::info(0, true, 4999, 12), payload.data(), 64) ==
        Code::InvalidArgument);  // a key's pts before the last key's
  CHECK(publish(vrst::info(0, false, 6000, 13), nullptr, 64) ==
        Code::InvalidArgument);
  const vkc::Status refused =
      server->publish(vrst::info(0, false, 6000, 10), payload.data(), 64);
  CHECK(refused.message().find("StreamServer::publish") == 0);
  CHECK(publish(vrst::info(0, false, 1, 14), payload.data(), 64) ==
        Code::Ok);  // a P-frame's pts may go back
  Got got;
  CHECK(net::next_within(*client, &got) && got.record.info.seq == 10);
  CHECK(net::next_within(*client, &got) && got.record.info.seq == 14);
  CHECK(!net::next_within(*client, &got, 100ms));

  const auto start = [](const stream::Session& s,
                        networking::ServerOptions options) {
    return networking::StreamServer::start(s, options).status().domain();
  };
  networking::ServerOptions local;
  local.host = "127.0.0.1";
  local.port = 0;
  CHECK(start(stream::Session{}, local) == Code::InvalidArgument);
  stream::Session no_kind = session;
  no_kind.tracks[0].kind = stream::TrackKind{};
  CHECK(start(no_kind, local) == Code::InvalidArgument);
  stream::Session two = session;
  two.tracks.push_back(session.tracks[0]);
  CHECK(start(two, local) == Code::Unsupported);
  for (const char* host : {"localhost", "::1", "1.2.3", "256.0.0.1", " "}) {
    networking::ServerOptions named = local;
    named.host = host;
    CHECK(start(session, named) == Code::InvalidArgument);
  }
  networking::ServerOptions none = local;
  none.max_viewers = 0;
  CHECK(start(session, none) == Code::InvalidArgument);
  networking::ServerOptions taken = local;
  taken.port = server->port();
  CHECK(start(session, taken) == Code::IoError);
  return 0;
}

// --- Teardown.
// ----------------------------------------------------------------

template <typename Gone>
bool within_1s(Gone&& gone) {
  const Clock::time_point start = Clock::now();
  gone();
  return Clock::now() - start < 1s;
}

// The destructor returns within 1 s with no viewers, with idle ones (which
// then see the end), and with one stalled mid-chunk.
int teardown_case() {
  const stream::Session session = net::geometry_session();
  auto alone = net::serve(session);
  CHECK(alone != nullptr);
  CHECK(within_1s([&] { alone.reset(); }));

  auto idle = net::serve(session);
  CHECK(idle != nullptr);
  auto a = net::connect(idle->port());
  auto b = net::connect(idle->port());
  CHECK(a != nullptr && b != nullptr);
  CHECK(net::wait_session(*a));
  CHECK(net::wait_session(*b));
  CHECK(within_1s([&] { idle.reset(); }));
  for (auto* client : {a.get(), b.get()}) CHECK(net::wait_end(*client, 1s));

  Captured log;
  const LogCapture capture(&log);
  auto busy = net::serve(session);
  CHECK(busy != nullptr);
  const vkc::UniqueFd stalled = net::connect_loopback(busy->port(), 4096);
  CHECK(stalled.valid());
  CHECK(log.wait_for(peer_name(stalled.get()), "joined", 2s));
  for (std::uint64_t seq = 0; seq < 4; ++seq) {
    const Bytes payload = net::record_payload(seq, 1 << 20);
    CHECK(busy->publish(net::record_info(seq, seq == 0), payload.data(),
                        payload.size())
              .ok());
  }
  std::this_thread::sleep_for(100ms);  // its socket full, mid-chunk
  CHECK(within_1s([&] { busy.reset(); }));
  return 0;
}

// --- Signals.
// -----------------------------------------------------------------

std::atomic<int> signals_handled{0};
void count_signal(int) { signals_handled.fetch_add(1); }

// 100 SIGUSR1 to a handler that does nothing, installed without SA_RESTART,
// while a viewer streams: to the process, which the publishing thread blocks
// so they land on the server's and the client's threads, and to the client's
// thread. Every record arrives intact, in order.
int signals_case() {
  struct sigaction action{};
  action.sa_handler = count_signal;
  sigemptyset(&action.sa_mask);
  action.sa_flags = 0;
  struct sigaction before{};
  CHECK(::sigaction(SIGUSR1, &action, &before) == 0);

  constexpr std::uint64_t kRecords = 400;
  const stream::Session session = net::geometry_session();
  auto server = net::serve(session);
  CHECK(server != nullptr);
  auto client = net::connect(server->port());
  CHECK(client != nullptr);
  CHECK(net::wait_session(*client));
  std::atomic<std::uint64_t> received{0};
  std::atomic<bool> intact{true};
  std::thread reader([&] {
    for (std::uint64_t seq = 0; seq < kRecords; ++seq) {
      Got got;
      if (!net::next_within(*client, &got, 10s) || got.record.info.seq != seq ||
          got.payload != net::record_payload(seq, 2048 + seq)) {
        intact = false;
        return;
      }
      received = seq + 1;
    }
  });
  sigset_t usr1;
  sigemptyset(&usr1);
  sigaddset(&usr1, SIGUSR1);
  CHECK(::pthread_sigmask(SIG_BLOCK, &usr1, nullptr) == 0);
  // Checked once the reader is joined.
  int sent = 0;
  bool published = true;
  bool signalled = true;
  for (std::uint64_t seq = 0; seq < kRecords && published; ++seq) {
    const Bytes payload = net::record_payload(seq, 2048 + seq);
    // One GOP, so no record is skipped however far the viewer falls behind.
    published = server
                    ->publish(net::record_info(seq, seq == 0), payload.data(),
                              payload.size())
                    .ok();
    if (seq % 4 == 0 && sent < 100 && intact) {
      signalled = signalled &&
                  (sent % 2 == 0
                       ? ::kill(::getpid(), SIGUSR1)
                       : ::pthread_kill(reader.native_handle(), SIGUSR1)) == 0;
      ++sent;
    }
    std::this_thread::sleep_for(1ms);
  }
  reader.join();
  CHECK(::pthread_sigmask(SIG_UNBLOCK, &usr1, nullptr) == 0);
  CHECK(::sigaction(SIGUSR1, &before, nullptr) == 0);
  CHECK(published && signalled);
  CHECK(sent == 100);
  CHECK(intact && received == kRecords);
  std::printf("signals: %d handled of %d sent\n", signals_handled.load(), sent);
  CHECK(signals_handled >= 50);
  return 0;
}

}  // namespace

int main() {
  vr_test::Scratch scratch("recon_networking_server_");
  const std::string dir = scratch.path.string();
  if (wire_is_the_file_case(dir) != 0) return 1;
  if (round_trip_case(dir) != 0) return 1;
  if (join_case() != 0) return 1;
  if (newest_key_wins_case() != 0) return 1;
  if (gop_limit_case() != 0) return 1;
  if (stall_limit_case() != 0) return 1;
  if (max_viewers_case() != 0) return 1;
  if (leaving_viewer_case() != 0) return 1;
  if (rules_case() != 0) return 1;
  if (teardown_case() != 0) return 1;
  if (signals_case() != 0) return 1;
  std::puts("networking server: OK");
  return 0;
}
