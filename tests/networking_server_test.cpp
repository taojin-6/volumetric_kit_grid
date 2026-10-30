// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// StreamServer and StreamClient on loopback (127.0.0.1, port 0): a raw
// viewer's bytes are StreamWriter's file for the same session and records,
// less its footer, a payload in two runs too, under the CRC its
// RecordPayload computed; a client reads the session, then every record's info,
// payload and file offset; a viewer that sends bytes still gets every record;
// a viewer joining gets the GOP from its key, and
// before any key nothing, and none is counted wanting a key; a viewer that
// stops reading in the first key while
// 3 GOPs of 1 MiB records are published gets that key whole, then the newest
// GOP whole, every publish back while it reads nothing, keys on demand too;
// without replay_gop, a joiner is counted and gets nothing until the next
// key, then every record, and one that stops reading falls back, counted,
// past max_gop_bytes, then gets the next key, while one reading gets every
// record, and a record past max_gop_bytes on its own is logged at Warning
// (a copy that fails: networking_server_oom_test.cpp); a Geometry track
// and an Audio track served, the wire the two-track file, and an audio
// record that is not a key refused with nothing changed; a viewer joining
// mid-GOP gets track 0's key, then every record after it, both tracks', and
// a burst of audio keys reaches a stuck viewer whole, while a newer track-0
// key skips the audio of the GOPs before it with their frames, replaying and
// on demand alike; a GOP past
// max_gop_bytes sends nothing until the next key; a viewer that takes no byte
// is closed after stall_limit, never when it is milliseconds::max(), and on
// Linux one that takes too few for its socket to poll writable within it is
// not; connection max_viewers + 1 is closed before any byte, the first such
// refusal logged at Warning and the next at Debug, and a client refused there
// is told it may have been; peers reset before they are accepted are dropped
// with no Warning while a viewer joins; a viewer that leaves never fails
// publish; viewers() counts a viewer from before its session, one waiting for
// a key too, until it leaves, fails or stalls, and never a refused
// connection; a viewer's join and leave are logged at Info, the leave once it
// no longer counts; the writer's refusals send nothing, and so does a payload
// over the parser's cap; start
// refuses (a stall_limit of 0 or less among them); the destructor returns
// within 1 s with no viewers, idle ones and a stalled one; the threads of
// viewers that ended are joined as they go, the address space (on macOS, its
// stacks) back near its start after 300; and 100 SIGUSR1 to a no-op handler
// leave the stream whole. Every test that publishes first waits for the
// viewer's session.

#include <pthread.h>
#include <signal.h>
#include <sys/socket.h>
#include <unistd.h>
#ifdef __APPLE__
#include <mach/mach.h>
#include <mach/mach_vm.h>
#endif
#ifdef __GLIBC__
#include <malloc.h>
#endif

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#include "volumetric_kit/core/base/log.hpp"
#include "volumetric_kit/core/testing/scratch_dir.hpp"
#include "volumetric_kit/core/testing/test_check.hpp"
#include "volumetric_kit/grid/stream/audio.hpp"
#include "volumetric_kit/grid/stream/stream_reader.hpp"
#include "volumetric_kit/grid/stream/stream_writer.hpp"
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
using net::serves_within;
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
  struct Line {
    vkc::LogLevel level;
    std::string text;
  };
  std::mutex mutex;
  std::vector<Line> lines;

  // The lines holding `a` and `b`, at `level` when one is given.
  std::size_t count(std::string_view a, std::string_view b = {},
                    std::optional<vkc::LogLevel> level = {}) {
    const std::lock_guard<std::mutex> lock(mutex);
    std::size_t n = 0;
    for (const Line& line : lines) {
      n += line.text.find(a) != std::string::npos &&
           line.text.find(b) != std::string::npos &&
           (!level || line.level == *level);
    }
    return n;
  }
  bool has(std::string_view a, std::string_view b = {}) {
    return count(a, b) > 0;
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
    vkc::set_log_handler([into](vkc::LogLevel level, std::string_view,
                                std::string_view message) {
      const std::lock_guard<std::mutex> lock(into->mutex);
      into->lines.push_back({level, std::string(message)});
    });
  }
  ~LogCapture() { vkc::set_log_handler({}); }
  LogCapture(const LogCapture&) = delete;
  LogCapture& operator=(const LogCapture&) = delete;
};

std::string peer_name(int fd) {
  return "127.0.0.1:" + std::to_string(net::local_port(fd));
}

// The records a raw viewer gets, each chunk passing its CRCs: from `wire`,
// what it has read, then from `fd` until the record of seq `last` on track
// `track`, within 10 s.
int drain(int fd, const Bytes& wire, std::uint64_t last, std::vector<Got>* got,
          std::uint16_t track = 0) {
  stream::ChunkParser parser;
  CHECK(parser.feed(wire.data(), wire.size()).ok());
  const Clock::time_point end = Clock::now() + 10s;
  const auto done = [&] {
    return !got->empty() && got->back().record.info.track == track &&
           got->back().record.info.seq >= last;
  };
  while (!done() && Clock::now() < end) {
    Got g;
    const vkc::Result<bool> r = parser.next(&g.record, &g.payload);
    CHECK(r.ok());
    if (*r) {
      got->push_back(std::move(g));
      continue;
    }
    Bytes more;
    net::read_upto(fd, std::size_t{1} << 20, 100ms, &more);
    CHECK(parser.feed(more.data(), more.size()).ok());
  }
  return 0;
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
  const Bytes file = vkc::test::read_bytes(finished);
  CHECK(vrst::index_offset(file) == wire.size());
  CHECK(Bytes(file.begin(),
              file.begin() + static_cast<std::ptrdiff_t>(wire.size())) == wire);
  CHECK(vkc::test::read_bytes(unfinished) == wire);
  return 0;
}

// A payload in two runs is published as a file holds its joined bytes: a raw
// viewer's bytes are the unfinished file's, at every split. Its CRC is
// RecordPayload::of's alone: bytes changed after `of` reach the viewer under
// the CRC of what they held, which its parser refuses, so the server
// computed none of its own.
int two_runs_case(const std::string& dir) {
  const stream::Session session = net::geometry_session();
  const std::vector<Published> records = some_records();
  auto server = net::serve(session);
  CHECK(server != nullptr);
  const vkc::UniqueFd viewer = net::connect_loopback(server->port());
  CHECK(viewer.valid());
  Bytes wire;
  const std::size_t header = net::session_chunk(session).size();
  net::read_upto(viewer.get(), header, 5s, &wire);
  CHECK(wire.size() == header);
  const std::string path = dir + "/two_runs.vrst";
  auto writer = stream::StreamWriter::create(path, session);
  CHECK(writer.ok());
  std::size_t i = 0;
  for (const Published& r : records) {
    const std::size_t n = r.payload.size();
    const std::size_t at = (i++ * 5) % (n + 1);
    CHECK((*writer)->append(r.info, r.payload.data(), n).ok());
    CHECK(server
              ->publish(r.info, stream::RecordPayload::of(r.payload.data(), at,
                                                          r.payload.data() + at,
                                                          n - at))
              .ok());
    const std::size_t want = (*writer)->size();
    net::read_upto(viewer.get(), want, 5s, &wire);
    CHECK(wire.size() == want);
  }
  writer->reset();
  CHECK(vkc::test::read_bytes(path) == wire);

  const Bytes head = vrst::pattern(32, 1);
  Bytes tail = vrst::pattern(64, 2);
  const stream::RecordPayload p = stream::RecordPayload::of(
      head.data(), head.size(), tail.data(), tail.size());
  tail[3] ^= 0xFF;  // after `of`
  CHECK(server->publish(net::record_info(records.size(), false), p).ok());
  stream::ChunkParser parser;
  CHECK(parser.feed(wire.data(), wire.size()).ok());
  Got g;
  for (std::size_t r = 0; r < records.size(); ++r) {
    CHECK(parser.next(&g.record, &g.payload).value());
  }
  const Clock::time_point end = Clock::now() + 10s;
  vkc::Status refused;
  while (refused.ok() && Clock::now() < end) {
    Bytes more;
    net::read_upto(viewer.get(), std::size_t{1} << 16, 100ms, &more);
    refused = parser.feed(more.data(), more.size());
    if (refused.ok()) refused = parser.next(&g.record, &g.payload).status();
  }
  CHECK(!refused.ok() && refused.message().find("CRC") != std::string::npos);
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

// What a viewer sends is read and dropped: one that sends 64 KiB mid-stream
// gets every record.
int viewer_sends_case() {
  constexpr std::uint64_t kRecords = 20;
  const stream::Session session = net::geometry_session();
  auto server = net::serve(session);
  CHECK(server != nullptr);
  const vkc::UniqueFd viewer = net::connect_loopback(server->port());
  CHECK(viewer.valid());
  const std::size_t header = net::session_chunk(session).size();
  Bytes wire;
  net::read_upto(viewer.get(), header, 5s, &wire);
  CHECK(wire.size() == header);
  for (std::uint64_t seq = 0; seq < kRecords; ++seq) {
    if (seq == kRecords / 2) {
      CHECK(net::send_bytes(viewer.get(), Bytes(64 << 10, 0x55)));
    }
    const Bytes payload = net::record_payload(seq, 1000);
    CHECK(server
              ->publish(net::record_info(seq, seq == 0), payload.data(),
                        payload.size())
              .ok());
  }
  std::vector<Got> got;
  CHECK(drain(viewer.get(), wire, kRecords - 1, &got) == 0);
  CHECK(got.size() == kRecords);
  for (std::uint64_t seq = 0; seq < kRecords; ++seq) {
    CHECK(got[seq].record.info.seq == seq);
    CHECK(got[seq].payload == net::record_payload(seq, 1000));
  }
  return 0;
}

// --- Joining.
// -----------------------------------------------------------------

// A viewer joining after K P P gets K P P, then live; after a second key it
// starts there; before any key it gets nothing until one, counted as a viewer
// while it waits.
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

  // Before any key: nothing, then the key; counted while it waits.
  auto fresh = net::serve(session);
  CHECK(fresh != nullptr);
  auto early = net::connect(fresh->port());
  CHECK(early != nullptr);
  CHECK(net::wait_session(*early));
  CHECK(fresh->viewers() == 1);
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
  // Every joiner started at a key, so none was counted wanting one.
  CHECK(server->key_wanted() == 0 && fresh->key_wanted() == 0);
  return 0;
}

// --- Newest key wins.
// ---------------------------------------------------------

// A viewer that stops reading while 3 GOPs of 1 MiB records are published,
// then drains, gets the first key whole, then the newest GOP whole, each
// chunk passing its CRCs; every publish returned while it read nothing. The
// raw viewer reads the first key's header before the rest is published, so
// the server is sending it then, and under the low-water mark the kernel holds
// well under 1 MiB of a viewer that does not read: the key is under way,
// however late its thread ran, until the viewer drains. Replaying the GOP and
// with keys on demand alike, the viewer counted only as it joined there.
int newest_key_wins_case(bool replay) {
  constexpr std::uint64_t kGop = 4;
  constexpr std::uint64_t kRecords = 3 * kGop;
  constexpr std::size_t kSize = 1 << 20;
  const stream::Session session = net::geometry_session();
  networking::ServerOptions options;
  options.replay_gop = replay;
  // Never closed: a publish that waited on the viewer, which takes nothing
  // until the last publish is back, would never return (ctest's TIMEOUT).
  // That is the check, with no clock. A stall_limit would end such a wait
  // after a time a starved host may also take to publish, as would a bound
  // on a publish's time: its CRC and copy of 1 MiB take 50 ms under
  // ThreadSanitizer, and seconds on a starved host.
  options.stall_limit = std::chrono::milliseconds::max();
  auto server = net::serve(session, options);
  CHECK(server != nullptr);
  const vkc::UniqueFd viewer = net::connect_loopback(server->port(), 4096);
  CHECK(viewer.valid());
  const std::size_t header = net::session_chunk(session).size();
  Bytes wire;
  net::read_upto(viewer.get(), header, 5s, &wire);
  CHECK(wire.size() == header);
  const auto publish = [&](std::uint64_t seq) {
    const Bytes payload = net::record_payload(seq, kSize);
    return server
        ->publish(net::record_info(seq, seq % kGop == 0), payload.data(),
                  payload.size())
        .ok();
  };
  CHECK(publish(0));
  net::read_upto(viewer.get(), header + 56, 5s, &wire);  // the key's header
  CHECK(wire.size() == header + 56);
  for (std::uint64_t seq = 1; seq < kRecords; ++seq) CHECK(publish(seq));
  std::vector<Got> got;
  CHECK(drain(viewer.get(), wire, kRecords - 1, &got) == 0);
  CHECK(got.size() == 1 + kGop);
  for (std::size_t i = 0; i < got.size(); ++i) {
    const std::uint64_t seq = i == 0 ? 0 : kRecords - kGop + i - 1;
    CHECK(got[i].record.info.seq == seq);
    CHECK(got[i].record.info.key == (seq % kGop == 0));
    CHECK(got[i].payload == net::record_payload(seq, kSize));
  }
  CHECK(server->key_wanted() == (replay ? 0 : 1));
  return 0;
}

// --- Keys on demand.
// ---------------------------------------------------------

// Without replay_gop: a viewer that joins after K P is counted at once, wanting
// a key and as a viewer, and gets its session, then nothing until the next
// key, then every record; a second joiner waits for the next key while the
// first gets every record; one leaving is not counted wanting a key, and is
// no longer a viewer.
int on_demand_join_case() {
  networking::ServerOptions options;
  options.replay_gop = false;
  auto server = net::serve(net::geometry_session(), options);
  CHECK(server != nullptr);
  const auto publish = [&](std::uint64_t seq, bool key) {
    const Bytes payload = net::record_payload(seq, 100 + seq);
    return server
        ->publish(net::record_info(seq, key), payload.data(), payload.size())
        .ok();
  };
  CHECK(publish(0, true));
  CHECK(publish(1, false));
  CHECK(server->key_wanted() == 0);
  // A client connects as it reads, so its session comes first; it was
  // counted before the session was sent, wanting a key and as a viewer while
  // it waits for one.
  auto first = net::connect(server->port());
  CHECK(first != nullptr && net::wait_session(*first));
  CHECK(server->key_wanted() == 1 && server->viewers() == 1);
  Got got;
  CHECK(publish(2, false));
  CHECK(!net::next_within(*first, &got, 200ms));
  CHECK(publish(3, true));
  CHECK(publish(4, false));
  CHECK(net::next_within(*first, &got));
  CHECK(got.record.info.seq == 3 && got.record.info.key);
  CHECK(got.payload == net::record_payload(3, 103));
  CHECK(net::next_within(*first, &got) && got.record.info.seq == 4);
  auto second = net::connect(server->port());
  CHECK(second != nullptr && net::wait_session(*second));
  CHECK(server->key_wanted() == 2 && server->viewers() == 2);
  CHECK(publish(5, false));
  CHECK(net::next_within(*first, &got) && got.record.info.seq == 5);
  CHECK(!net::next_within(*second, &got, 200ms));
  CHECK(publish(6, true));
  for (auto* client : {first.get(), second.get()}) {
    CHECK(net::next_within(*client, &got) && got.record.info.seq == 6);
  }
  second.reset();
  CHECK(publish(7, false));
  CHECK(net::next_within(*first, &got) && got.record.info.seq == 7);
  CHECK(server->key_wanted() == 2);
  CHECK(serves_within(*server, 1));
  return 0;
}

// Without replay_gop, at a max_gop_bytes of 512 KiB: a raw viewer that reads
// its session, then nothing, while one GOP of 64 KiB records is published,
// falls back, counted, once a record would leave it more than that behind
// what its socket took, each publish back while it reads nothing; a client
// reading each record as it comes gets every one and is not counted. Then a
// key and two P-frames: the raw viewer, draining, gets each chunk whole, a
// run from the first key, then the new key and the P-frames after it, so no
// P-frame without the records it is predicted from; the client gets them all.
int on_demand_fall_back_case() {
  constexpr std::size_t kSize = 64 << 10;
  const stream::Session session = net::geometry_session();
  networking::ServerOptions options;
  options.replay_gop = false;
  options.max_gop_bytes = 512 << 10;
  options.stall_limit = std::chrono::milliseconds::max();
  auto server = net::serve(session, options);
  CHECK(server != nullptr);
  auto fast = net::connect(server->port());
  CHECK(fast != nullptr && net::wait_session(*fast));
  const vkc::UniqueFd slow = net::connect_loopback(server->port(), 4096);
  CHECK(slow.valid());
  const std::size_t header = net::session_chunk(session).size();
  Bytes wire;
  net::read_upto(slow.get(), header, 5s, &wire);
  CHECK(wire.size() == header);
  CHECK(server->key_wanted() == 2);  // counted before their sessions
  // Never closed, a raw viewer that reads nothing until the end: a publish
  // that waited on it would never return, as in newest_key_wins_case.
  const auto publish = [&](std::uint64_t seq, bool key) {
    const Bytes payload = net::record_payload(seq, kSize);
    return server
        ->publish(net::record_info(seq, key), payload.data(), payload.size())
        .ok();
  };
  std::uint64_t seq = 0;
  Got got;
  while (server->key_wanted() == 2 && seq < 1024) {  // 64 MiB at most
    CHECK(publish(seq, seq == 0));
    CHECK(net::next_within(*fast, &got) && got.record.info.seq == seq);
    ++seq;
  }
  CHECK(server->key_wanted() == 3);
  const std::uint64_t key = seq;
  for (; seq < key + 3; ++seq) {
    CHECK(publish(seq, seq == key));
    CHECK(net::next_within(*fast, &got) && got.record.info.seq == seq);
  }
  std::vector<Got> drained;
  CHECK(drain(slow.get(), wire, key + 2, &drained) == 0);
  CHECK(drained.size() > 3);
  const std::size_t run = drained.size() - 3;
  CHECK(run < key);  // it missed records
  for (std::size_t i = 0; i < drained.size(); ++i) {
    const std::uint64_t want = i < run ? i : key + (i - run);
    CHECK(drained[i].record.info.seq == want);
    CHECK(drained[i].record.info.key == (want == 0 || want == key));
    CHECK(drained[i].payload == net::record_payload(want, kSize));
  }
  CHECK(server->key_wanted() == 3);
  std::printf("on demand: fell back after %llu records, %zu sent before\n",
              static_cast<unsigned long long>(key), run);
  return 0;
}

// Without replay_gop, at a max_gop_bytes of 3 records of 1000 bytes: a key
// past it on its own goes to none, logged at Warning, and the joiner waiting
// for it is not counted again; a P-frame past it sends the viewer in back,
// counted, at Warning too; the records after each, which no viewer takes,
// warn no more; a key within it lets the viewer in.
int on_demand_limit_case() {
  Captured log;
  const LogCapture capture(&log);
  networking::ServerOptions options;
  options.replay_gop = false;
  options.max_gop_bytes = 3 * (56 + 1000);
  auto server = net::serve(net::geometry_session(), options);
  CHECK(server != nullptr);
  auto client = net::connect(server->port());
  CHECK(client != nullptr && net::wait_session(*client));
  CHECK(server->key_wanted() == 1);
  const auto publish = [&](std::uint64_t seq, bool key, std::size_t size) {
    const Bytes payload = net::record_payload(seq, size);
    return server
        ->publish(net::record_info(seq, key), payload.data(), payload.size())
        .ok();
  };
  const auto warned = [&log] {
    return log.count("StreamServer::publish: a record of ",
                     " bytes, past max_gop_bytes (3168): the viewers wait "
                     "for the next key",
                     vkc::LogLevel::Warning);
  };
  Got got;
  CHECK(publish(0, true, 3 * 1056));
  CHECK(publish(1, false, 10));
  CHECK(warned() == 1);
  CHECK(!net::next_within(*client, &got, 200ms));
  CHECK(server->key_wanted() == 1);
  CHECK(publish(2, true, 1000));
  CHECK(net::next_within(*client, &got) && got.record.info.seq == 2);
  CHECK(publish(3, false, 3 * 1056));
  CHECK(publish(4, false, 10));
  CHECK(warned() == 2);
  CHECK(server->key_wanted() == 2);
  CHECK(!net::next_within(*client, &got, 200ms));
  CHECK(publish(5, true, 10));
  CHECK(net::next_within(*client, &got) && got.record.info.seq == 5);
  CHECK(warned() == 2 && server->key_wanted() == 2);
  return 0;
}

// --- Two tracks.
// -------------------------------------------------------------

constexpr std::uint64_t kFramePts = 40'000'000;  // 25 frames a second
constexpr std::uint64_t kChunkPts = 20'000'000;  // 20 ms chunks
constexpr std::size_t kChunkBytes = 1920;        // 960 mono s16 frames

// A Geometry track 0 and an Audio track 1, as a sender with a mic has.
stream::Session two_track_session() {
  stream::Session s = net::geometry_session();
  stream::Track audio;
  audio.kind = stream::TrackKind::Audio;
  audio.rate = {48000, 960};
  audio.params = stream::encode_audio_params({});
  s.tracks.push_back(audio);
  return s;
}

// Frame `seq` of track 0 at 40 ms a frame, a key when `key`, of `size` bytes.
Published frame(std::uint64_t seq, bool key, std::size_t size = 1000) {
  return {vrst::info(0, key, seq * kFramePts, seq),
          net::record_payload(seq, size)};
}

// Chunk `seq` of track 1 at 20 ms a chunk, a key as every one is.
Published chunk(std::uint64_t seq) {
  return {vrst::info(1, true, seq * kChunkPts, seq),
          net::record_payload(1000 + seq, kChunkBytes)};
}

// Frames `first` to `last`, a key every `gop`, each followed by the chunks
// CodecStream releases with it: those at or below its pts and above the
// frame's before (chunk 0 with frame 0; 2k - 1 and 2k with frame k).
std::vector<Published> muxed(std::uint64_t first, std::uint64_t last,
                             std::uint64_t gop) {
  std::vector<Published> out;
  for (std::uint64_t k = first; k <= last; ++k) {
    out.push_back(frame(k, k % gop == 0));
    if (k > 0) out.push_back(chunk(2 * k - 1));
    out.push_back(chunk(2 * k));
  }
  return out;
}

bool is(const Got& got, const Published& r) {
  return vrst::same(got.record.info, r.info) && got.payload == r.payload;
}

// Two tracks served: a raw viewer's bytes are the writer's file of the same
// two-track session and records, less its footer. A record of track 1 that
// is not a key is refused, naming it, with nothing sent or changed: the same
// seq as a key is taken after it, and the file never had it.
int two_tracks_wire_case(const std::string& dir) {
  const stream::Session session = two_track_session();
  auto server = net::serve(session);
  CHECK(server != nullptr);
  const vkc::UniqueFd viewer = net::connect_loopback(server->port());
  CHECK(viewer.valid());
  Bytes wire;
  const std::size_t header = net::session_chunk(session).size();
  net::read_upto(viewer.get(), header, 5s, &wire);
  CHECK(wire.size() == header);
  const std::string path = dir + "/two_tracks.vrst";
  auto writer = stream::StreamWriter::create(path, session);
  CHECK(writer.ok());
  for (const Published& r : muxed(0, 6, 3)) {
    if (r.info.track == 1 && r.info.seq == 5) {
      stream::RecordInfo inter = r.info;
      inter.key = false;
      const vkc::Status refused =
          server->publish(inter, r.payload.data(), r.payload.size());
      CHECK(refused.domain() == Code::InvalidArgument &&
            refused.message().find("track 1 seq 5 is not a key") !=
                std::string::npos);
    }
    CHECK((*writer)->append(r.info, r.payload.data(), r.payload.size()).ok());
    CHECK(server->publish(r.info, r.payload.data(), r.payload.size()).ok());
    const std::size_t want = (*writer)->size();
    net::read_upto(viewer.get(), want, 5s, &wire);
    CHECK(wire.size() == want);
  }
  CHECK((*writer)->finish().ok());
  const Bytes file = vkc::test::read_bytes(path);
  CHECK(vrst::index_offset(file) == wire.size());
  CHECK(Bytes(file.begin(),
              file.begin() + static_cast<std::ptrdiff_t>(wire.size())) == wire);
  return 0;
}

// A viewer joining mid-GOP gets track 0's key first, then every record
// published after it, the chunks at its pts and on included, and none from
// before: replaying, the GOP's key, which the newer chunks, keys too, never
// replace; keys on demand, nothing until the next track-0 key, the chunks
// meanwhile, keys too, letting it in no earlier.
int two_tracks_join_case(bool replay) {
  networking::ServerOptions options;
  options.replay_gop = replay;
  auto server = net::serve(two_track_session(), options);
  CHECK(server != nullptr);
  const std::vector<Published> records = muxed(0, 7, 3);  // keys 0, 3, 6
  // Where `records` holds track `track`'s record `seq`.
  const auto at = [&](std::uint16_t track, std::uint64_t seq) {
    for (std::size_t i = 0; i < records.size(); ++i) {
      if (records[i].info.track == track && records[i].info.seq == seq) {
        return i;
      }
    }
    return records.size();
  };
  const auto publish = [&](std::size_t from, std::size_t to) {
    for (std::size_t i = from; i < to; ++i) {
      const Published& r = records[i];
      if (!server->publish(r.info, r.payload.data(), r.payload.size()).ok()) {
        return false;
      }
    }
    return true;
  };
  std::unique_ptr<networking::StreamClient> client;
  const auto read = [&](std::size_t from, std::size_t to) {
    for (std::size_t i = from; i < to; ++i) {
      Got got;
      if (!net::next_within(*client, &got) || !is(got, records[i])) {
        std::fprintf(stderr, "joined: record %zu not as published\n", i);
        return false;
      }
    }
    return true;
  };
  // Up to frame 4 and its chunks, mid-GOP: frame 3 the newest key.
  const std::size_t joined_at = at(0, 5);
  CHECK(publish(0, joined_at));
  client = net::connect(server->port());
  CHECK(client != nullptr);
  // Replaying, it reads the GOP, whose key comes with its session, before a
  // newer key is published, which it would start at otherwise; on demand it
  // has joined once it has its session.
  if (replay) {
    CHECK(read(at(0, 3), joined_at));
  } else {
    CHECK(net::wait_session(*client));
  }
  // The rest, through frame 7's chunks, each read as it is published, so a
  // newer key skips none: on demand from frame 6, the next key.
  const std::size_t from = replay ? joined_at : at(0, 6);
  CHECK(publish(joined_at, from));
  for (std::size_t i = from; i < records.size(); ++i) {
    CHECK(publish(i, i + 1) && read(i, i + 1));
  }
  Got extra;
  CHECK(!net::next_within(*client, &extra, 100ms));
  return 0;
}

// A raw viewer with a 4 KiB receive buffer reads its session and the header
// of a 1 MiB track-0 key, then nothing while `records` are published after
// it, each publish back at once; then it drains until the last. What it
// gets.
int slow_viewer_gets(bool replay, const std::vector<Published>& records,
                     std::vector<Got>* got) {
  const stream::Session session = two_track_session();
  networking::ServerOptions options;
  options.replay_gop = replay;
  // Never closed, as in newest_key_wins_case.
  options.stall_limit = std::chrono::milliseconds::max();
  auto server = net::serve(session, options);
  CHECK(server != nullptr);
  const vkc::UniqueFd viewer = net::connect_loopback(server->port(), 4096);
  CHECK(viewer.valid());
  const std::size_t header = net::session_chunk(session).size();
  Bytes wire;
  net::read_upto(viewer.get(), header, 5s, &wire);
  CHECK(wire.size() == header);
  const Published key = frame(0, true, 1 << 20);
  CHECK(server->publish(key.info, key.payload.data(), key.payload.size()).ok());
  net::read_upto(viewer.get(), header + 56, 5s, &wire);  // the key's header
  CHECK(wire.size() == header + 56);
  for (const Published& r : records) {
    CHECK(server->publish(r.info, r.payload.data(), r.payload.size()).ok());
  }
  const stream::RecordInfo& last = records.back().info;
  CHECK(drain(viewer.get(), wire, last.seq, got, last.track) == 0);
  CHECK(!got->empty() && is(got->front(), key));
  return 0;
}

// A burst of 200 chunks, every one a key, while a viewer is stuck in a 1 MiB
// track-0 key, then a P-frame: the viewer gets the key, every chunk and the
// P-frame, as only track 0's keys start a GOP. Replaying and on demand.
int audio_burst_case(bool replay) {
  std::vector<Published> burst;
  for (std::uint64_t j = 0; j < 200; ++j) burst.push_back(chunk(j));
  burst.push_back(frame(1, false));
  std::vector<Got> got;
  CHECK(slow_viewer_gets(replay, burst, &got) == 0);
  CHECK(got.size() == 1 + burst.size());
  for (std::size_t i = 0; i < burst.size(); ++i)
    CHECK(is(got[i + 1], burst[i]));
  return 0;
}

// Newest key wins skips the chunks with the frames: a viewer stuck in a
// 1 MiB track-0 key while the rest of its GOP and two more, 1 MiB frames
// with their chunks, are published gets that key, then the newest GOP whole,
// its chunks with it, and no chunk of the GOPs it skipped. Replaying and on
// demand.
int newest_key_skips_audio_case(bool replay) {
  std::vector<Published> rest;
  for (std::uint64_t k = 1; k < 9; ++k) {  // keys at 3 and 6
    rest.push_back(frame(k, k % 3 == 0, 1 << 20));
    rest.push_back(chunk(2 * k - 1));
    rest.push_back(chunk(2 * k));
  }
  std::vector<Got> got;
  CHECK(slow_viewer_gets(replay, rest, &got) == 0);
  // The key, then frame 6 and everything after it.
  const std::size_t newest = 3 * (6 - 1);
  CHECK(rest[newest].info.track == 0 && rest[newest].info.seq == 6);
  CHECK(got.size() == 1 + rest.size() - newest);
  for (std::size_t i = newest; i < rest.size(); ++i) {
    CHECK(is(got[1 + i - newest], rest[i]));
  }
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
// stall_limit (200 ms), logged at Info, and no longer counted, and its bytes
// then end; a reading viewer meanwhile gets every record, still counted.
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
  CHECK(server->viewers() == 2);

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
  // Logged once it is no longer counted.
  CHECK(server->viewers() == 1);
  // What it was sent ends: the server closed it.
  Bytes rest;
  bool eof = false;
  net::read_upto(stalled.get(), std::size_t{1} << 30, 5s, &rest, &eof);
  CHECK(eof);
  CHECK(rest.size() < seq * (256 << 10));
  return 0;
}

// A stall_limit of milliseconds::max() is never: a viewer that takes no byte
// while 4 records of 1 MiB wait for it stays for 500 ms, 25 poll slices.
int stall_never_case() {
  Captured log;
  const LogCapture capture(&log);
  const stream::Session session = net::geometry_session();
  networking::ServerOptions options;
  options.stall_limit = std::chrono::milliseconds::max();
  auto server = net::serve(session, options);
  CHECK(server != nullptr);
  const vkc::UniqueFd stalled = net::connect_loopback(server->port(), 4096);
  CHECK(stalled.valid());
  Bytes header;
  net::read_upto(stalled.get(), net::session_chunk(session).size(), 5s,
                 &header);
  CHECK(header.size() == net::session_chunk(session).size());
  const std::string name = peer_name(stalled.get());
  CHECK(log.wait_for(name, "joined", 2s));
  for (std::uint64_t seq = 0; seq < 4; ++seq) {
    const Bytes payload = net::record_payload(seq, 1 << 20);
    CHECK(server
              ->publish(net::record_info(seq, seq == 0), payload.data(),
                        payload.size())
              .ok());
  }
  std::this_thread::sleep_for(500ms);
  CHECK(!log.has(name, " closed: ") && !log.has(name, " failed: "));
  return 0;
}

// A viewer that keeps taking bytes, though too slowly for its socket to poll
// writable within stall_limit (300 ms), is not closed: 1 KiB every 10 ms
// through a 4 KiB receive buffer, while 256 KiB records come for 2 s. Under
// the low-water mark again takes 64 KiB, 640 ms at this rate. Linux only:
// macOS's loopback ignores the small buffer and, behind a reader this slow,
// acknowledges in bursts seconds apart.
int slow_viewer_case() {
#ifdef __linux__
  Captured log;
  const LogCapture capture(&log);
  const stream::Session session = net::geometry_session();
  networking::ServerOptions options;
  options.stall_limit = 300ms;
  auto server = net::serve(session, options);
  CHECK(server != nullptr);
  const vkc::UniqueFd slow = net::connect_loopback(server->port(), 4096);
  CHECK(slow.valid());
  Bytes taken;
  net::read_upto(slow.get(), net::session_chunk(session).size(), 5s, &taken);
  CHECK(taken.size() == net::session_chunk(session).size());
  const std::string name = peer_name(slow.get());
  CHECK(log.wait_for(name, "joined", 2s));

  const Clock::time_point start = Clock::now();
  for (std::uint64_t seq = 0; Clock::now() - start < 2s; ++seq) {
    const Bytes payload = net::record_payload(seq, 256 << 10);
    CHECK(server
              ->publish(net::record_info(seq, seq % 30 == 0), payload.data(),
                        payload.size())
              .ok());
    const std::size_t before = taken.size();
    bool eof = false;
    net::read_upto(slow.get(), before + 1024, 1s, &taken, &eof);
    CHECK(!eof && taken.size() == before + 1024);
    std::this_thread::sleep_for(10ms);
  }
  CHECK(!log.has(name, "took no byte"));
#endif
  return 0;
}

// Connection max_viewers + 1 is closed before any byte, and never counted;
// once a viewer leaves, a new one is served.
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
  CHECK(server->viewers() == 2);
  // A client refused there is told it may have been.
  auto refused = net::connect(server->port());
  CHECK(refused != nullptr);
  Got got;
  vkc::Result<bool> r = false;
  for (int i = 0; i < 500 && r.ok() && !*r; ++i) {
    r = refused->next(10ms, &got.record, &got.payload);
  }
  CHECK(!r.ok() &&
        net::client_reason(r.status().message(), server->port()) ==
            "the sender closed the connection before its session (it may "
            "already serve its most viewers)");
  // The first refusal at Warning, which the default sink prints; the next,
  // within 10 s of it, at Debug, which a sender printing Info drops too.
  const Clock::time_point counted = Clock::now() + 2s;
  while (log.count("refused viewer ") < 2 && Clock::now() < counted) {
    std::this_thread::sleep_for(5ms);
  }
  CHECK(log.count("refused viewer ", peer_name(third.get()),
                  vkc::LogLevel::Warning) == 1);
  CHECK(log.count("refused viewer ", "", vkc::LogLevel::Warning) == 1);
  CHECK(log.count("refused viewer ", "", vkc::LogLevel::Debug) == 1);
  CHECK(server->viewers() == 2);

  first.reset();
  CHECK(serves_within(*server, 1));
  const std::size_t header = net::session_chunk(session).size();
  bool served = false;
  for (int attempt = 0; attempt < 100 && !served; ++attempt) {
    const vkc::UniqueFd next = net::connect_loopback(server->port());
    CHECK(next.valid());
    Bytes got;
    net::read_upto(next.get(), header, 1s, &got);
    served = got.size() == header;
    CHECK(!served || server->viewers() == 2);
    if (!served) std::this_thread::sleep_for(10ms);
  }
  CHECK(served);
  return 0;
}

// Peers that connect and reset before the accept thread takes them (macOS
// then fails the accepted socket's first setsockopt) are dropped quietly: no
// Warning, and a viewer joining while they come gets its session. Served
// ones end at once, and 64 slots keep any from being refused. They come in
// bursts the listener's queue (8) holds, 20 ms apart, so that an accept
// thread slowed by a sanitizer on a busy machine mostly empties it first:
// Linux drops a SYN to a full queue, and a connect then waits a second to
// send it again. So 100 resets get 10 s.
int reset_before_accept_case() {
  Captured log;
  const LogCapture capture(&log);
  networking::ServerOptions options;
  options.max_viewers = 64;
  auto server = net::serve(net::geometry_session(), options);
  CHECK(server != nullptr);
  std::atomic<bool> stop{false};
  std::atomic<int> resets{0};
  std::thread resetter([&] {
    while (!stop) {
      for (int i = 0; i < 8; ++i) {
        vkc::UniqueFd fd = net::connect_loopback(server->port());
        if (!fd.valid()) continue;
        const linger hard{1, 0};
        (void)::setsockopt(fd.get(), SOL_SOCKET, SO_LINGER, &hard, sizeof hard);
        fd.reset();
        ++resets;
      }
      std::this_thread::sleep_for(20ms);
    }
  });
  const Clock::time_point flooding = Clock::now() + 10s;
  while (resets < 100 && Clock::now() < flooding) {
    std::this_thread::sleep_for(5ms);
  }
  // As a player does, a viewer the kernel turned away tries again.
  bool joined = false;
  const Clock::time_point end = Clock::now() + 5s;
  while (!joined && Clock::now() < end) {
    auto client = net::connect(server->port());
    joined = client != nullptr && net::wait_session(*client, 1s);
  }
  stop = true;
  resetter.join();
  CHECK(resets >= 100);
  CHECK(joined);
  {
    const std::lock_guard<std::mutex> lock(log.mutex);
    for (const Captured::Line& line : log.lines) {
      if (line.level == vkc::LogLevel::Warning) {
        std::fprintf(stderr, "warned: %s\n", line.text.c_str());
      }
    }
  }
  CHECK(log.count("", "", vkc::LogLevel::Warning) == 0);
  return 0;
}

// Viewers that leave mid-stream, by a close or a reset, never fail publish,
// and are no longer counted; the one that stays gets every record.
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
  CHECK(serves_within(*server, 1));
  return 0;
}

// The count rises at each join, before the viewer's session, and falls as
// each leaves, the others still counted.
int viewer_count_case() {
  auto server = net::serve(net::geometry_session());
  CHECK(server != nullptr);
  CHECK(server->viewers() == 0);
  std::vector<std::unique_ptr<networking::StreamClient>> clients;
  for (std::size_t n = 1; n <= 3; ++n) {
    clients.push_back(net::connect(server->port()));
    CHECK(clients.back() != nullptr && net::wait_session(*clients.back()));
    CHECK(server->viewers() == n);
  }
  while (!clients.empty()) {
    clients.erase(clients.begin());
    CHECK(serves_within(*server, clients.size()));
  }
  return 0;
}

// A viewer that reads its session and closes is logged joining, then leaving,
// each at Info, which the senders print: the only lines that name it; it is
// no longer counted by the time it is logged leaving.
int join_leave_log_case() {
  Captured log;
  const LogCapture capture(&log);
  const stream::Session session = net::geometry_session();
  auto server = net::serve(session);
  CHECK(server != nullptr);
  vkc::UniqueFd viewer = net::connect_loopback(server->port());
  CHECK(viewer.valid());
  const std::string name = peer_name(viewer.get());
  Bytes header;
  net::read_upto(viewer.get(), net::session_chunk(session).size(), 5s, &header);
  CHECK(header.size() == net::session_chunk(session).size());
  viewer.reset();  // nothing unread, so a clean end
  CHECK(log.wait_for(name, " left", 2s));
  CHECK(server->viewers() == 0);  // logged once it is no longer counted
  const std::string line = "StreamServer: viewer " + name;
  CHECK(log.count(line + " joined, 1 of 4", "", vkc::LogLevel::Info) == 1);
  CHECK(log.count(line + " left", "", vkc::LogLevel::Info) == 1);
  CHECK(log.count(name) == 2);
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
  // A track the session lacks, named so, not as a track's record that is not
  // a key.
  const vkc::Status unknown =
      server->publish(vrst::info(1, false, 6000, 11), payload.data(), 64);
  CHECK(unknown.domain() == Code::InvalidArgument);
  CHECK(unknown.message().find("track 1, but the session has 1") !=
        std::string::npos);
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
  for (const char* host : {"localhost", "::1", "1.2.3", "256.0.0.1", " "}) {
    networking::ServerOptions named = local;
    named.host = host;
    CHECK(start(session, named) == Code::InvalidArgument);
  }
  networking::ServerOptions none = local;
  none.max_viewers = 0;
  CHECK(start(session, none) == Code::InvalidArgument);
  for (const std::chrono::milliseconds limit : {0ms, -1ms}) {
    networking::ServerOptions instant = local;
    instant.stall_limit = limit;
    CHECK(start(session, instant) == Code::InvalidArgument);
  }
  networking::ServerOptions taken = local;
  taken.port = server->port();
  CHECK(start(session, taken) == Code::IoError);

  // A payload over the parser's cap, which no client reads, is refused
  // though the GOP limit is raised past it, with nothing changed: the same
  // seq is published after.
  networking::ServerOptions wide;
  wide.max_gop_bytes = std::uint64_t{256} << 20;
  auto big = net::serve(session, wide);
  CHECK(big != nullptr);
  constexpr auto kOver =
      static_cast<std::size_t>(stream::ChunkParser::kMaxBodyBytes) + 1;
  const auto huge = std::make_unique<std::uint8_t[]>(kOver);
  CHECK(big->publish(vrst::info(0, true, 0, 0), huge.get(), kOver).domain() ==
        Code::InvalidArgument);
  CHECK(big->publish(vrst::info(0, true, 0, 0), payload.data(), 64).ok());
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
  const std::uint16_t port = idle->port();
  auto a = net::connect(port);
  auto b = net::connect(port);
  CHECK(a != nullptr && b != nullptr);
  CHECK(net::wait_session(*a));
  CHECK(net::wait_session(*b));
  CHECK(within_1s([&] { idle.reset(); }));
  for (auto* client : {a.get(), b.get()}) {
    CHECK(net::wait_end(*client, port, 1s));
  }

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

#ifndef VKC_TEST_SANITIZED
// The address space a thread never joined would grow, in bytes, or 0 when it
// cannot be read: on macOS the regions tagged as stacks, as malloc maps a
// 4 MiB region whenever a thread finds none of its own; elsewhere all of it.
std::uint64_t address_space() {
#ifdef __APPLE__
  std::uint64_t stacks = 0;
  for (mach_vm_address_t address = 0;;) {
    mach_vm_size_t size = 0;
    natural_t depth = 0;
    vm_region_submap_info_data_64_t info{};
    mach_msg_type_number_t count = VM_REGION_SUBMAP_INFO_COUNT_64;
    if (::mach_vm_region_recurse(
            mach_task_self(), &address, &size, &depth,
            reinterpret_cast<vm_region_recurse_info_t>(&info),
            &count) != KERN_SUCCESS) {
      return stacks;  // past the last region
    }
    if (info.user_tag == VM_MEMORY_STACK) stacks += size;
    address += size;
  }
#else
  std::ifstream statm("/proc/self/statm");
  std::uint64_t pages = 0;
  statm >> pages;
  return pages * static_cast<std::uint64_t>(::sysconf(_SC_PAGESIZE));
#endif
}
#endif

// The threads of viewers that ended are joined as they go, not when the
// server does: after 300 viewers join and leave one by one, the address
// space (on macOS, its stacks) is back within 4 MiB of where it was after
// the first 20. A thread never joined keeps its stack (8 MiB on Linux) or,
// on macOS, 32 KiB of it, and neither shows in a thread count. A viewer's
// thread sees its end within a poll slice, so 64 slots, and one refused
// tries again. Each reading waits, however long a loaded machine takes, for
// every viewer the log says joined to have ended, then for kHeld more to join
// in turn and stay: the accept loop joins every ended viewer's thread before
// each accept, so before it took the second. glibc keeps joined threads'
// stacks for its next threads, up to 40 MiB (4 of 8 MiB), how many hanging
// on whether ended viewers were still to join as the first came; the other
// five take every one kept, so a reading holds kHeld viewers' stacks alone.
// glibc gives a thread that finds no arena free one of its own, 64 MiB of
// address space kept, so main allows one. Not under ASan or TSan, which keep
// freed memory a while.
int reaping_case() {
#ifndef VKC_TEST_SANITIZED
  constexpr int kHeld = 6;
  Captured log;
  LogCapture capture(&log);
  const stream::Session session = net::geometry_session();
  networking::ServerOptions options;
  options.max_viewers = 64;
  auto server = net::serve(session, options);
  CHECK(server != nullptr);
  const std::size_t header = net::session_chunk(session).size();
  const auto join_and_leave = [&](int viewers) {
    for (int joined = 0, tries = 0; joined < viewers; ++tries) {
      if (tries == 20 * viewers) return false;
      const vkc::UniqueFd viewer = net::connect_loopback(server->port());
      Bytes got;
      net::read_upto(viewer.get(), header, 5s, &got);
      joined += got.size() == header;
      std::this_thread::sleep_for(got.size() == header ? 1ms : 5ms);
    }
    return true;
  };
  // The address space once every ended viewer's thread is joined, with kHeld
  // viewers held. A viewer's lines: one as it joins, one as it ends.
  const auto all_ended = [&] {
    return log.count(": viewer ") == 2 * log.count(": viewer ", " joined, ");
  };
  const auto reaped = [&](std::uint64_t* bytes) {
    const Clock::time_point end = Clock::now() + 60s;
    while (!all_ended()) {
      if (Clock::now() >= end) return false;
      std::this_thread::sleep_for(5ms);
    }
    vkc::UniqueFd held[kHeld];
    for (vkc::UniqueFd& viewer : held) {
      viewer = net::connect_loopback(server->port());
      Bytes got;
      net::read_upto(viewer.get(), header, 10s, &got);
      if (got.size() != header) return false;
    }
    *bytes = address_space();
    return true;
  };
  CHECK(join_and_leave(20));
  std::uint64_t before = 0;
  CHECK(reaped(&before));
  CHECK(before > 0);
  CHECK(join_and_leave(300));
  std::uint64_t after = 0;
  CHECK(reaped(&after));
  std::printf(
      "reaping: %+lld KiB after 300 viewers\n",
      (static_cast<long long>(after) - static_cast<long long>(before)) / 1024);
  CHECK(after < before + (4 << 20));
#endif
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
#if defined(__GLIBC__) && !defined(VKC_TEST_SANITIZED)
  // One malloc arena, for reaping_case's readings, set before any thread
  // starts: once a thread finds no arena free with more than 8 made, as the
  // cases before it can on a loaded machine, glibc fixes the limit at 8 per
  // core and ignores mallopt from then on.
  CHECK(::mallopt(M_ARENA_MAX, 1) == 1);
#endif
  vkc::test::Scratch scratch("grid_networking_server_");
  const std::string dir = scratch.path.string();
  if (wire_is_the_file_case(dir) != 0) return 1;
  if (two_runs_case(dir) != 0) return 1;
  if (round_trip_case(dir) != 0) return 1;
  if (viewer_sends_case() != 0) return 1;
  if (join_case() != 0) return 1;
  if (newest_key_wins_case(true) != 0) return 1;
  if (newest_key_wins_case(false) != 0) return 1;
  if (on_demand_join_case() != 0) return 1;
  if (on_demand_fall_back_case() != 0) return 1;
  if (on_demand_limit_case() != 0) return 1;
  if (two_tracks_wire_case(dir) != 0) return 1;
  for (const bool replay : {true, false}) {
    if (two_tracks_join_case(replay) != 0) return 1;
    if (audio_burst_case(replay) != 0) return 1;
    if (newest_key_skips_audio_case(replay) != 0) return 1;
  }
  if (gop_limit_case() != 0) return 1;
  if (stall_limit_case() != 0) return 1;
  if (stall_never_case() != 0) return 1;
  if (slow_viewer_case() != 0) return 1;
  if (max_viewers_case() != 0) return 1;
  if (reset_before_accept_case() != 0) return 1;
  if (leaving_viewer_case() != 0) return 1;
  if (viewer_count_case() != 0) return 1;
  if (join_leave_log_case() != 0) return 1;
  if (rules_case() != 0) return 1;
  if (teardown_case() != 0) return 1;
  if (reaping_case() != 0) return 1;
  if (signals_case() != 0) return 1;
  std::puts("networking server: OK");
  return 0;
}
