// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// StreamServer on loopback (127.0.0.1, port 0) when a record's copy fails, an
// allocation failure injected there: without replay_gop, publish returns
// OutOfMemory and has counted the viewer it sends back in key_wanted before
// it returns; a key lost so counts the viewer waiting for it again; the next
// key lets it in, with no record between. A viewer's notes when an allocation
// fails: reading them (the reader alone, as a viewer's thread cannot be armed
// from here) ends them with nothing thrown; newest_note returns none and then
// the note; send_note returns OutOfMemory, and the stream and the next note go
// on. Apart from the server's other tests, which stay free of the replaced
// operator new, so they link under ThreadSanitizer on Linux too.

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <optional>
#include <string>
#include <thread>
#include <utility>

#include "note_wire.hpp"
#include "volumetric_kit/core/testing/test_allocation_failure.hpp"
#include "volumetric_kit/core/testing/test_check.hpp"
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
using net::Got;
using namespace std::chrono_literals;

int lost_copy_case() {
  constexpr std::size_t kSize = 64 << 10;
  networking::ServerOptions options;
  options.replay_gop = false;
  auto server = net::serve(net::geometry_session(), options);
  CHECK(server != nullptr);
  auto client = net::connect(server->port());
  CHECK(client != nullptr && net::wait_session(*client));
  const auto publish = [&](std::uint64_t seq, bool key) {
    const Bytes payload = net::record_payload(seq, kSize);
    return server
        ->publish(net::record_info(seq, key), payload.data(), payload.size())
        .domain();
  };
  Got got;
  CHECK(publish(0, true) == Code::Ok);
  CHECK(net::next_within(*client, &got) && got.record.info.seq == 0);
  // A publish's largest allocation is its chunk's copy, the same at every
  // record of this size.
  auto& failure = vkc::test::allocation_failure;
  failure = {};
  failure.measure = true;
  const Code measured = publish(1, false);
  failure.measure = false;
  const std::size_t copy = failure.largest;
  const void* const site = failure.largest_site;
  failure = {};
  CHECK(measured == Code::Ok && copy > kSize);
  CHECK(net::next_within(*client, &got) && got.record.info.seq == 1);
  const auto lost = [&](std::uint64_t seq, bool key) {
    failure.bytes = copy;
    failure.site = site;
    failure.armed = true;
    const Code code = publish(seq, key);
    const bool injected = failure.injected;
    failure = {};
    return injected && code == Code::OutOfMemory;
  };
  CHECK(server->key_wanted() == 1);
  CHECK(lost(2, false));
  CHECK(server->key_wanted() == 2);
  CHECK(lost(3, true));
  CHECK(server->key_wanted() == 3);
  CHECK(publish(4, false) == Code::Ok);
  CHECK(!net::next_within(*client, &got, 200ms));
  CHECK(publish(5, true) == Code::Ok);
  CHECK(net::next_within(*client, &got) && got.record.info.seq == 5);
  CHECK(server->key_wanted() == 3);
  return 0;
}

// A line too long for a string's own storage, so that copying it allocates.
const std::string kLongLine(150, 'x');

networking::ViewerNote long_note() {
  networking::ViewerNote note;
  note.tone = networking::ViewerNote::Tone::Good;
  note.line = kLongLine;
  return note;
}

// A viewer's first bytes: its session and one note of the long line.
Bytes viewer_bytes() {
  stream::Session session;
  session.writer = "viewer";
  stream::Track track;
  track.kind = stream::TrackKind::Note;
  session.tracks.push_back(track);
  Bytes bytes = net::session_chunk(session);
  const Bytes record = vrst::record(
      0, 1, 0, 0, stream::kNoSet, networking::detail::encode_note(long_note()));
  bytes.insert(bytes.end(), record.begin(), record.end());
  return bytes;
}

// Whichever of the reader's largest and last allocations fails, nothing is
// thrown (feed is noexcept: it would end the process) and the reader ends,
// where one with no failure reads the note.
int reader_oom_case() {
  const Bytes bytes = viewer_bytes();
  auto& failure = vkc::test::allocation_failure;
  failure = {};
  failure.measure = true;
  networking::detail::NoteReader measured;
  const std::optional<networking::ViewerNote> note =
      measured.feed(bytes.data(), bytes.size());
  failure.measure = false;
  CHECK(note && note->line == kLongLine && !measured.ended());
  const std::pair<std::size_t, const void*> sites[] = {
      {failure.largest, failure.largest_site},
      {failure.bytes, failure.measured_site},
  };
  for (const auto& [size, site] : sites) {
    failure = {};
    failure.bytes = size;
    failure.site = site;
    failure.armed = true;
    networking::detail::NoteReader reader;
    const std::optional<networking::ViewerNote> got =
        reader.feed(bytes.data(), bytes.size());
    const bool injected = failure.injected;
    failure = {};
    CHECK(injected);
    CHECK(!got && reader.ended());
    CHECK(!reader.feed(bytes.data(), bytes.size()));
  }
  return 0;
}

// newest_note's copy fails: none, nothing thrown, and the note is still there.
int newest_note_oom_case() {
  networking::ServerOptions options;
  options.read_notes = true;
  auto server = net::serve(net::geometry_session(), options);
  CHECK(server != nullptr);
  auto client = net::connect(server->port());
  CHECK(client != nullptr && net::wait_session(*client));
  CHECK(client->send_note(long_note()).ok());
  const net::Clock::time_point end = net::Clock::now() + 5s;
  while (!server->newest_note() && net::Clock::now() < end) {
    std::this_thread::sleep_for(1ms);
  }
  auto& failure = vkc::test::allocation_failure;
  failure = {};
  failure.measure = true;
  const std::optional<networking::ViewerNote> measured = server->newest_note();
  failure.measure = false;
  CHECK(measured && measured->line == kLongLine);
  failure.site = failure.measured_site;
  failure.armed = true;
  const std::optional<networking::ViewerNote> none = server->newest_note();
  const bool injected = failure.injected;
  failure = {};
  CHECK(injected && !none);
  const std::optional<networking::ViewerNote> back = server->newest_note();
  CHECK(back && back->line == kLongLine);
  return 0;
}

// send_note's framing fails: OutOfMemory, with the connection, the stream and
// the next note as they were; so too for the first note, whose session must
// still go before the one after it.
int send_note_oom_case() {
  networking::ServerOptions options;
  options.read_notes = true;
  auto server = net::serve(net::geometry_session(), options);
  CHECK(server != nullptr);
  auto client = net::connect(server->port());
  CHECK(client != nullptr && net::wait_session(*client));
  auto& failure = vkc::test::allocation_failure;
  // Measured on a client of its own, whose first note allocates as this one's.
  auto other = net::connect(server->port());
  CHECK(other != nullptr && net::wait_session(*other));
  failure = {};
  failure.measure = true;
  const vkc::Status first = other->send_note(long_note());
  failure.measure = false;
  CHECK(first.ok());
  const std::size_t largest = failure.largest;
  const void* const site = failure.largest_site;
  const auto fails = [&] {
    failure = {};
    failure.bytes = largest;
    failure.site = site;
    failure.armed = true;
    const Code code = client->send_note(long_note()).domain();
    const bool injected = failure.injected;
    failure = {};
    return injected && code == Code::OutOfMemory;
  };
  CHECK(fails());
  const Bytes payload = net::record_payload(0, 1000);
  CHECK(
      server->publish(net::record_info(0, true), payload.data(), payload.size())
          .ok());
  Got got;
  CHECK(net::next_within(*client, &got) && got.payload == payload);
  CHECK(client->send_note(long_note()).ok());
  other.reset();
  CHECK(net::serves_within(*server, 1));
  const net::Clock::time_point end = net::Clock::now() + 5s;
  while (!server->newest_note() && net::Clock::now() < end) {
    CHECK(client->next(1ms, &got.record, &got.payload).ok());
  }
  const std::optional<networking::ViewerNote> note = server->newest_note();
  CHECK(note && note->line == kLongLine);
  return 0;
}

}  // namespace

int main() {
  if (lost_copy_case() != 0) return 1;
  if (reader_oom_case() != 0) return 1;
  if (newest_note_oom_case() != 0) return 1;
  if (send_note_oom_case() != 0) return 1;
  std::puts("networking server oom: OK");
  return 0;
}
