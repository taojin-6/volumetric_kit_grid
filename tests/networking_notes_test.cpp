// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// A viewer's notes (StreamClient::send_note, StreamServer::newest_note) on
// loopback, 127.0.0.1 port 0 only: a note arrives, tone and line whole, its
// received time its arrival; a burst, and two in one read, leave the newest; a
// note fed a byte at a time reads once whole; the wire is a Note track's
// session and records, 104 B and 80 B, with nothing up before the first note;
// later fields, a reserved byte and an unknown tone read, a note that cannot be
// read or of another kind is dropped and the next reads; a line over the limit
// is refused on sending; bytes that cannot be read as chunks end the notes,
// logged once, and never the stream; without read_notes, the default, a note
// changes nothing; with notes flowing a raw viewer's bytes are byte for byte
// what the writer's encoder makes, and a client reads every record; a viewer
// that never sends is as today; send_note on a socket that is full returns at
// once and the peer reads whole notes only, the last the newest; a note before
// the connect goes once connected, one after the end is OK, with next reporting
// the end; a note goes as its viewer leaves, so a reconnect starts with none; a
// dialled viewer's notes read as a connected one's; a note after finish draws
// a reset, which costs the reader at most the tail, never a torn record, while
// with no note it reads every record and a clean end; and a publisher, a client
// and a reader of newest_note at once, for ThreadSanitizer. An allocation that
// fails: networking_server_oom_test.cpp.

#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#include "volumetric_kit/core/base/log.hpp"
#include "volumetric_kit/core/net/dial.hpp"
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
using Note = networking::ViewerNote;
using Tone = networking::ViewerNote::Tone;
using net::Bytes;
using net::Clock;
using net::Got;
using net::serves_within;
using namespace std::chrono_literals;

// How long a check that a note did not arrive waits: the viewer thread reads
// within a poll slice (20 ms) of the bytes coming.
constexpr auto kNothingWithin = 50ms;
constexpr std::size_t kRecordBytes = 1000;
constexpr std::size_t kRecordChunkBytes = 56 + kRecordBytes;

std::unique_ptr<networking::StreamServer> note_server() {
  networking::ServerOptions options;
  options.read_notes = true;
  return net::serve(net::geometry_session(), options);
}

Note note_of(const std::string& line, Tone tone = Tone::Neutral) {
  Note note;
  note.tone = tone;
  note.line = line;
  return note;
}

// Whether `server`'s newest note is `line` within 5 s.
bool line_within(const networking::StreamServer& server,
                 const std::string& line) {
  const Clock::time_point end = Clock::now() + 5s;
  for (;;) {
    const std::optional<Note> note = server.newest_note();
    if (note && note->line == line) return true;
    if (Clock::now() >= end) return false;
    std::this_thread::sleep_for(1ms);
  }
}

// `a`, then `b`.
Bytes joined(Bytes a, const Bytes& b) {
  a.insert(a.end(), b.begin(), b.end());
  return a;
}

// A viewer's session up, with a track of each of `kinds`.
Bytes up_session(const std::vector<stream::TrackKind>& kinds = {
                     stream::TrackKind::Note}) {
  stream::Session session;
  session.writer = "viewer";
  for (const stream::TrackKind kind : kinds) {
    stream::Track track;
    track.kind = kind;
    session.tracks.push_back(std::move(track));
  }
  return net::session_chunk(session);
}

// A payload as stream/session.hpp spells it, by hand; `size` the L it claims.
Bytes payload_of(std::uint8_t tone, std::uint8_t reserved, std::uint16_t size,
                 const std::string& line, const Bytes& later = {}) {
  Bytes payload{tone, reserved, static_cast<std::uint8_t>(size & 0xff),
                static_cast<std::uint8_t>(size >> 8)};
  payload.insert(payload.end(), line.begin(), line.end());
  payload.insert(payload.end(), later.begin(), later.end());
  return payload;
}

Bytes note_payload(const std::string& line, std::uint8_t tone = 0) {
  return payload_of(tone, 0, static_cast<std::uint16_t>(line.size()), line);
}

// A record up, by hand: a key, pts 0.
Bytes record_up(std::uint64_t seq, const Bytes& payload,
                std::uint16_t track = 0) {
  return vrst::record(track, 1, 0, seq, stream::kNoSet, payload);
}

// A raw viewer of `server`, its session read.
vkc::UniqueFd raw_viewer(const networking::StreamServer& server) {
  vkc::UniqueFd viewer = net::connect_loopback(server.port());
  const std::size_t header = net::session_chunk(net::geometry_session()).size();
  Bytes wire;
  if (viewer.valid()) net::read_upto(viewer.get(), header, 5s, &wire);
  if (wire.size() != header) return {};
  return viewer;
}

// Publish records `from` on (a key at 0), and check the raw viewer `fd` reads
// each whole, the stream going on.
int stream_goes_on(networking::StreamServer& server, int fd, std::uint64_t from,
                   std::uint64_t count) {
  for (std::uint64_t seq = from; seq < from + count; ++seq) {
    const Bytes payload = net::record_payload(seq, kRecordBytes);
    CHECK(server
              .publish(net::record_info(seq, seq == 0), payload.data(),
                       payload.size())
              .ok());
    Bytes wire;
    net::read_upto(fd, kRecordChunkBytes, 5s, &wire);
    CHECK(wire.size() == kRecordChunkBytes);
    CHECK(Bytes(wire.begin() + 56, wire.end()) == payload);
  }
  return 0;
}

// --- A note arrives. --------------------------------------------------------

// Each tone and the limits of a line survive, and `received` is its arrival.
int arrives_case() {
  auto server = note_server();
  CHECK(server != nullptr);
  CHECK(!server->newest_note());
  auto client = net::connect(server->port());
  CHECK(client != nullptr && net::wait_session(*client));
  CHECK(!server->newest_note());
  const std::pair<Tone, std::string> notes[] = {
      {Tone::Neutral, "a plain line"},
      {Tone::Good, "a good line"},
      {Tone::Caution, "a line to mind"},
      {Tone::Good, ""},
      {Tone::Caution, std::string(networking::kMaxNoteLineBytes, 'x')},
      {Tone::Neutral, "\xc3\xa9t\xc3\xa9"},
  };
  for (const auto& [tone, line] : notes) {
    const Clock::time_point before = Clock::now();
    CHECK(client->send_note(note_of(line, tone)).ok());
    CHECK(line_within(*server, line));
    const Clock::time_point after = Clock::now();
    const std::optional<Note> note = server->newest_note();
    CHECK(note && note->tone == tone);
    CHECK(note->received >= before && note->received <= after);
  }
  CHECK(server->viewers() == 1);
  return 0;
}

// Sent with no read between them, the last is what stays.
int burst_case() {
  auto server = note_server();
  CHECK(server != nullptr);
  auto client = net::connect(server->port());
  CHECK(client != nullptr && net::wait_session(*client));
  constexpr int kBurst = 50;
  for (int i = 0; i < kBurst; ++i) {
    CHECK(client->send_note(note_of("note " + std::to_string(i))).ok());
  }
  CHECK(line_within(*server, "note " + std::to_string(kBurst - 1)));
  return 0;
}

// Two notes in one read leave the second, which is all a raw viewer's one send
// of both can show.
int two_in_one_read_case() {
  auto server = note_server();
  CHECK(server != nullptr);
  const vkc::UniqueFd viewer = raw_viewer(*server);
  CHECK(viewer.valid());
  CHECK(net::send_bytes(
      viewer.get(),
      joined(joined(up_session(), record_up(0, note_payload("first"))),
             record_up(1, note_payload("second", 1)))));
  CHECK(line_within(*server, "second"));
  CHECK(server->newest_note()->tone == Tone::Good);
  return 0;
}

// A note fed a byte at a time reads once its last byte comes, and not before:
// a line of 4 bytes, so its chunk has no padding.
int byte_at_a_time_case() {
  auto server = note_server();
  CHECK(server != nullptr);
  const vkc::UniqueFd viewer = raw_viewer(*server);
  CHECK(viewer.valid());
  const int on = 1;
  CHECK(::setsockopt(viewer.get(), IPPROTO_TCP, TCP_NODELAY, &on, sizeof on) ==
        0);
  const Bytes wire = joined(up_session(), record_up(0, note_payload("word")));
  for (std::size_t i = 0; i + 1 < wire.size(); ++i) {
    CHECK(net::send_bytes(viewer.get(), Bytes{wire[i]}));
  }
  std::this_thread::sleep_for(kNothingWithin);
  CHECK(!server->newest_note());
  CHECK(net::send_bytes(viewer.get(), Bytes{wire.back()}));
  CHECK(line_within(*server, "word"));
  return 0;
}

// --- The wire. --------------------------------------------------------------

// Nothing goes up before the first note, then a Note track's session and a key
// record of pts 0, whose payload is tone, reserved, L and the line.
int wire_form_case() {
  const net::LoopbackListener listener = net::bind_loopback();
  auto client = net::connect(listener.port);
  CHECK(client != nullptr);
  Got got;
  CHECK(client->next(0ms, &got.record, &got.payload).ok());
  const vkc::UniqueFd peer = net::accept_within(listener.fd.get(), 2s);
  CHECK(peer.valid());
  CHECK(client->next(0ms, &got.record, &got.payload).ok());
  Bytes wire;
  net::read_upto(peer.get(), 1, kNothingWithin, &wire);
  CHECK(wire.empty());

  const std::string line = "sixteen bytes ok";
  CHECK(line.size() == 16);
  CHECK(client->send_note(note_of(line, Tone::Good)).ok());
  net::read_upto(peer.get(), 104 + 80, 5s, &wire);
  CHECK(wire.size() == 104 + 80);
  stream::ChunkParser parser;
  CHECK(parser.feed(wire.data(), wire.size()).ok());
  stream::Record record;
  Bytes payload;
  const vkc::Result<bool> first = parser.next(&record, &payload);
  CHECK(first.ok() && *first);
  const stream::Session* const session = parser.session();
  CHECK(session != nullptr && session->writer == "viewer");
  CHECK(session->tracks.size() == 1);
  CHECK(session->tracks[0].kind == stream::TrackKind::Note);
  CHECK(record.info.track == 0 && record.info.key && record.info.pts_ns == 0);
  CHECK(payload == note_payload(line, 1));

  // The next has no session before it, and a seq above the last.
  const std::uint64_t seq = record.info.seq;
  const std::string longer = "twenty-one bytes long";
  CHECK(longer.size() == 21);
  CHECK(client->send_note(note_of(longer)).ok());
  wire.clear();
  net::read_upto(peer.get(), 88, 5s, &wire);
  CHECK(wire.size() == 88);
  CHECK(parser.feed(wire.data(), wire.size()).ok());
  const vkc::Result<bool> second = parser.next(&record, &payload);
  CHECK(second.ok() && *second);
  CHECK(record.info.seq > seq);
  CHECK(payload == note_payload(longer));
  return 0;
}

// What a reader takes of a payload, each followed by a note that shows the
// reader went on: later fields ignored, a reserved byte ignored, an unknown
// tone neutral; a payload too short for its head or its line, or a line over
// the limit, dropped; a record of another kind skipped, the note before it
// kept.
int payload_rules_case() {
  auto server = note_server();
  CHECK(server != nullptr);
  const vkc::UniqueFd viewer = raw_viewer(*server);
  CHECK(viewer.valid());
  const auto other = static_cast<stream::TrackKind>(99);
  CHECK(net::send_bytes(viewer.get(),
                        up_session({stream::TrackKind::Note, other})));
  std::uint64_t seq = 0;
  const auto send = [&](const Bytes& payload, std::uint16_t track = 0) {
    return net::send_bytes(viewer.get(), record_up(seq++, payload, track));
  };
  CHECK(send(payload_of(1, 0, 5, "later", Bytes{1, 2, 3})));
  CHECK(line_within(*server, "later"));
  CHECK(server->newest_note()->tone == Tone::Good);
  CHECK(send(payload_of(2, 7, 8, "reserved")));
  CHECK(line_within(*server, "reserved"));
  CHECK(server->newest_note()->tone == Tone::Caution);
  CHECK(send(payload_of(9, 0, 7, "unknown")));
  CHECK(line_within(*server, "unknown"));
  CHECK(server->newest_note()->tone == Tone::Neutral);

  const std::uint16_t over = networking::kMaxNoteLineBytes + 1;
  const Bytes dropped[] = {
      Bytes{},
      Bytes{1, 0, 0},
      payload_of(1, 0, 9, "short"),
      payload_of(1, 0, over, std::string(over, 'x')),
  };
  for (std::size_t i = 0; i < std::size(dropped); ++i) {
    const std::string next = "after " + std::to_string(i);
    CHECK(send(dropped[i]));
    CHECK(send(note_payload(next)));
    CHECK(line_within(*server, next));
  }
  CHECK(send(note_payload("kept")));
  CHECK(line_within(*server, "kept"));
  CHECK(send(note_payload("another kind"), 1));
  std::this_thread::sleep_for(kNothingWithin);
  CHECK(server->newest_note()->line == "kept");
  CHECK(stream_goes_on(*server, viewer.get(), 0, 3) == 0);
  return 0;
}

// A line over the limit is refused before it costs anything; the limit's own
// goes, and the stream never noticed.
int over_long_case() {
  auto server = note_server();
  CHECK(server != nullptr);
  auto client = net::connect(server->port());
  CHECK(client != nullptr && net::wait_session(*client));
  const std::string over(networking::kMaxNoteLineBytes + 1, 'x');
  CHECK(client->send_note(note_of(over)).domain() == Code::InvalidArgument);
  std::this_thread::sleep_for(kNothingWithin);
  CHECK(!server->newest_note());
  const std::string longest(networking::kMaxNoteLineBytes, 'y');
  CHECK(client->send_note(note_of(longest)).ok());
  CHECK(line_within(*server, longest));
  const Bytes payload = net::record_payload(0, kRecordBytes);
  CHECK(
      server->publish(net::record_info(0, true), payload.data(), payload.size())
          .ok());
  Got got;
  CHECK(net::next_within(*client, &got) && got.payload == payload);
  return 0;
}

// --- Bad bytes end the notes, never the stream. ------------------------------

// Counts the Warning lines that say a viewer's notes ended, until it goes.
struct EndsLogged {
  std::atomic<int> ends{0};
  EndsLogged() {
    vkc::set_log_handler([this](vkc::LogLevel level, std::string_view,
                                std::string_view message) {
      if (level == vkc::LogLevel::Warning &&
          message.find("its notes end") != std::string_view::npos) {
        ++ends;
      }
    });
  }
  ~EndsLogged() { vkc::set_log_handler({}); }
  EndsLogged(const EndsLogged&) = delete;
  EndsLogged& operator=(const EndsLogged&) = delete;

  bool within_5s() const {
    const Clock::time_point end = Clock::now() + 5s;
    while (ends == 0) {
      if (Clock::now() >= end) return false;
      std::this_thread::sleep_for(1ms);
    }
    return true;
  }
};

// After `first` (a good note in it reading `kept`, if `kept` is not empty),
// `bad` ends the notes: logged once; the note there was stays until the viewer
// leaves; a good note after is dropped; every record arrives.
int ends_the_notes(const Bytes& first, const std::string& kept,
                   const Bytes& bad) {
  EndsLogged logged;
  auto server = note_server();
  CHECK(server != nullptr);
  const vkc::UniqueFd viewer = raw_viewer(*server);
  CHECK(viewer.valid());
  CHECK(stream_goes_on(*server, viewer.get(), 0, 2) == 0);
  CHECK(net::send_bytes(viewer.get(), first));
  if (!kept.empty()) CHECK(line_within(*server, kept));
  CHECK(net::send_bytes(viewer.get(), bad));
  CHECK(logged.within_5s());
  CHECK(net::send_bytes(viewer.get(), record_up(100, note_payload("after"))));
  std::this_thread::sleep_for(kNothingWithin);
  const std::optional<Note> note = server->newest_note();
  CHECK(kept.empty() ? !note : note && note->line == kept);
  CHECK(server->viewers() == 1);
  CHECK(stream_goes_on(*server, viewer.get(), 2, 4) == 0);
  CHECK(logged.ends == 1);
  return 0;
}

int bad_bytes_case() {
  const Bytes good = joined(up_session(), record_up(0, note_payload("kept")));
  const Bytes record = record_up(1, note_payload("bad"));
  // A record's header, or its body, that fails its CRC.
  CHECK(ends_the_notes(good, "kept", vrst::flipped(record, 30, 0)) == 0);
  CHECK(ends_the_notes(good, "kept", vrst::flipped(record, 57, 0)) == 0);
  // A record of a track the session lacks.
  CHECK(ends_the_notes(good, "kept", record_up(1, note_payload("bad"), 5)) ==
        0);
  // A body past the parser's limit, refused by its header alone.
  const Bytes fields = vrst::record_fields(0, 1, 0, 1, stream::kNoSet);
  CHECK(ends_the_notes(good, "kept",
                       vrst::header("RECD", fields,
                                    (std::uint64_t{64} << 20) + 1, 0)) == 0);
  // 8 KiB toward a record that never gets whole.
  CHECK(ends_the_notes(good, "kept",
                       joined(vrst::header("RECD", fields, 1 << 20, 0),
                              Bytes(8 << 10, 0x55))) == 0);
  // 8 KiB of a chunk this version skips, with no record among them.
  CHECK(ends_the_notes(
            good, "kept",
            vrst::chunk("ABCD", Bytes(4, 0), Bytes(8 << 10, 0x55))) == 0);
  // Not a stream of chunks at all, from its first bytes: a peer's plain bytes,
  // and a session of a version not known.
  CHECK(ends_the_notes({}, "", Bytes(64 << 10, 0x55)) == 0);
  CHECK(ends_the_notes({}, "",
                       vrst::chunk("VRST", vrst::session_fields(1, 1, 0),
                                   vrst::session_body(1))) == 0);
  return 0;
}

// --- Off, and old ends. -----------------------------------------------------

// The default reads no note, from a client or a raw viewer, whose stream goes
// on: an old sender's reading and dropping, which send_note cannot tell.
int off_by_default_case() {
  CHECK(!networking::ServerOptions{}.read_notes);
  auto server = net::serve(net::geometry_session());
  CHECK(server != nullptr);
  auto client = net::connect(server->port());
  CHECK(client != nullptr && net::wait_session(*client));
  const vkc::UniqueFd viewer = raw_viewer(*server);
  CHECK(viewer.valid());
  CHECK(client->send_note(note_of("from a client")).ok());
  CHECK(net::send_bytes(
      viewer.get(),
      joined(up_session(), record_up(0, note_payload("from a raw viewer")))));
  std::this_thread::sleep_for(kNothingWithin);
  CHECK(!server->newest_note());
  CHECK(server->viewers() == 2);
  CHECK(stream_goes_on(*server, viewer.get(), 0, 3) == 0);
  for (std::uint64_t seq = 0; seq < 3; ++seq) {
    Got got;
    CHECK(net::next_within(*client, &got));
    CHECK(got.record.info.seq == seq);
    CHECK(got.payload == net::record_payload(seq, kRecordBytes));
  }
  return 0;
}

// A raw viewer's bytes with notes flowing are the encoder's byte for byte,
// reading notes or not: the wire down never changes, and a raw viewer that
// sends nothing reads the same, an old viewer on a new sender.
int wire_down_unchanged_case() {
  enum class Viewer { Sends, Silent };
  const auto check = [](bool read_notes, Viewer viewing) {
    constexpr std::uint64_t kRecords = 30;
    const stream::Session session = net::geometry_session();
    networking::ServerOptions options;
    options.read_notes = read_notes;
    auto server = net::serve(session, options);
    CHECK(server != nullptr);
    const vkc::UniqueFd viewer = net::connect_loopback(server->port());
    CHECK(viewer.valid());
    vkc::Result<stream::ChunkEncoder> encoder =
        stream::ChunkEncoder::create(session);
    CHECK(encoder.ok());
    Bytes want = encoder->session_chunk();
    Bytes wire;
    net::read_upto(viewer.get(), want.size(), 5s, &wire);
    CHECK(wire == want);
    const bool sends = viewing == Viewer::Sends;
    if (sends) CHECK(net::send_bytes(viewer.get(), up_session()));
    for (std::uint64_t seq = 0; seq < kRecords; ++seq) {
      if (sends) {
        CHECK(net::send_bytes(
            viewer.get(),
            record_up(seq, note_payload("note " + std::to_string(seq)))));
      }
      const Bytes payload = net::record_payload(seq, 1 + 77 * seq);
      const stream::RecordInfo info = net::record_info(seq, seq % 10 == 0);
      CHECK(server->publish(info, payload.data(), payload.size()).ok());
      const vkc::Result<stream::ChunkEncoder::RecordFraming> framing =
          encoder->frame(info, payload.data(), payload.size());
      CHECK(framing.ok());
      want.insert(want.end(), framing->header.begin(), framing->header.end());
      want.insert(want.end(), payload.begin(), payload.end());
      want.resize(want.size() + framing->padding);
      net::read_upto(viewer.get(), want.size(), 5s, &wire);
      CHECK(wire.size() == want.size());
    }
    CHECK(wire == want);
    if (read_notes && sends) {
      CHECK(line_within(*server, "note " + std::to_string(kRecords - 1)));
    } else {
      CHECK(!server->newest_note());
    }
    return 0;
  };
  CHECK(check(false, Viewer::Sends) == 0);
  CHECK(check(true, Viewer::Sends) == 0);
  CHECK(check(true, Viewer::Silent) == 0);
  return 0;
}

// A client reads every record, in order and whole, with a note sent after each
// and bursts of them between.
int client_reads_every_record_case() {
  constexpr std::uint64_t kRecords = 100;
  auto server = note_server();
  CHECK(server != nullptr);
  auto client = net::connect(server->port());
  CHECK(client != nullptr && net::wait_session(*client));
  for (std::uint64_t seq = 0; seq < kRecords; ++seq) {
    const Bytes payload = net::record_payload(seq, 4000);
    CHECK(server
              ->publish(net::record_info(seq, seq % 30 == 0), payload.data(),
                        payload.size())
              .ok());
    CHECK(client->send_note(note_of("note " + std::to_string(seq))).ok());
    Got got;
    CHECK(net::next_within(*client, &got));
    CHECK(got.record.info.seq == seq && got.payload == payload);
  }
  CHECK(line_within(*server, "note " + std::to_string(kRecords - 1)));
  return 0;
}

// --- send_note never waits. --------------------------------------------------

// A peer that never reads, behind small buffers: send_note, far past what the
// socket holds, returns each time, on a thread of its own so that one that
// waits fails here rather than hangs. Some were replaced, as newest wins, and
// next sends the rest: the peer reads whole notes only, the last the newest.
int full_socket_case() {
  constexpr int kNotes = 20000;
  const int small = 4096;
  const net::LoopbackListener listener = net::bind_loopback();
  vkc::UniqueFd fd = net::connect_loopback(listener.port);
  CHECK(fd.valid());
  const vkc::UniqueFd peer = net::accept_within(listener.fd.get(), 2s);
  CHECK(peer.valid());
  CHECK(::setsockopt(fd.get(), SOL_SOCKET, SO_SNDBUF, &small, sizeof small) ==
        0);
  CHECK(::setsockopt(peer.get(), SOL_SOCKET, SO_RCVBUF, &small, sizeof small) ==
        0);
  vkc::Result<std::unique_ptr<networking::StreamClient>> adopted =
      networking::StreamClient::adopt(std::move(fd));
  CHECK(adopted.ok());
  networking::StreamClient& client = **adopted;

  std::atomic<bool> done{false};
  std::atomic<bool> refused{false};
  std::thread sender([&] {
    for (int i = 0; i < kNotes; ++i) {
      if (!client.send_note(note_of("note " + std::to_string(i))).ok()) {
        refused = true;
      }
    }
    done = true;
  });
  const Clock::time_point end = Clock::now() + 10s;
  while (!done && Clock::now() < end) std::this_thread::sleep_for(1ms);
  if (!done) {
    std::fprintf(stderr, "FAIL: send_note waits on a full socket\n");
    std::_Exit(1);
  }
  sender.join();
  CHECK(!refused);

  const std::string newest = "note " + std::to_string(kNotes - 1);
  stream::ChunkParser parser;
  stream::Record record;
  Bytes payload;
  std::string last;
  int read = 0;
  const Clock::time_point read_end = Clock::now() + 5s;
  while (last != newest && Clock::now() < read_end) {
    Got ignored;
    CHECK(client.next(0ms, &ignored.record, &ignored.payload).ok());
    Bytes more;
    net::read_upto(peer.get(), 64 << 10, 2ms, &more);
    CHECK(parser.feed(more.data(), more.size()).ok());
    for (;;) {
      const vkc::Result<bool> got = parser.next(&record, &payload);
      CHECK(got.ok());
      if (!*got) break;
      ++read;
      CHECK(payload.size() > 4);
      last.assign(payload.begin() + 4, payload.end());
    }
  }
  CHECK(last == newest);
  CHECK(read < kNotes);
  return 0;
}

// A note before the connect goes once it is made; one on a connection that
// failed is OK, next reporting the end, its refusal of an over-long line alone
// still a refusal.
int connect_and_end_case() {
  auto server = note_server();
  CHECK(server != nullptr);
  auto client = net::connect(server->port());
  CHECK(client != nullptr);
  CHECK(client->send_note(note_of("before the connect")).ok());
  CHECK(net::wait_session(*client));
  CHECK(line_within(*server, "before the connect"));

  auto refused = net::connect(net::closed_port());
  CHECK(refused != nullptr);
  CHECK(refused->send_note(note_of("never goes")).ok());
  Got got;
  vkc::Result<bool> end = true;
  const Clock::time_point limit = Clock::now() + 5s;
  while (end.ok() && Clock::now() < limit) {
    end = refused->next(10ms, &got.record, &got.payload);
  }
  CHECK(!end.ok() && end.status().domain() == Code::IoError);
  CHECK(refused->send_note(note_of("after the end")).ok());
  const std::string over(networking::kMaxNoteLineBytes + 1, 'x');
  CHECK(refused->send_note(note_of(over)).domain() == Code::InvalidArgument);
  const vkc::Result<bool> again = refused->next(0ms, &got.record, &got.payload);
  CHECK(!again.ok() && again.status().message() == end.status().message());
  return 0;
}

// --- Who keeps it. ----------------------------------------------------------

// A note goes as its viewer leaves, by the time viewers() stops counting it; a
// reconnect starts with none. One kept across viewers: the newest from any; one
// from a viewer that stays is not touched by another's leaving, and one that
// was replaced does not come back.
int leaves_with_its_viewer_case() {
  auto server = note_server();
  CHECK(server != nullptr);
  auto a = net::connect(server->port());
  CHECK(a != nullptr && net::wait_session(*a));
  CHECK(a->send_note(note_of("from a")).ok());
  CHECK(line_within(*server, "from a"));
  a.reset();
  CHECK(serves_within(*server, 0));
  CHECK(!server->newest_note());

  a = net::connect(server->port());
  CHECK(a != nullptr && net::wait_session(*a));
  CHECK(!server->newest_note());
  CHECK(a->send_note(note_of("from a again")).ok());
  CHECK(line_within(*server, "from a again"));

  auto b = net::connect(server->port());
  CHECK(b != nullptr && net::wait_session(*b));
  CHECK(server->newest_note()->line == "from a again");
  CHECK(b->send_note(note_of("from b")).ok());
  CHECK(line_within(*server, "from b"));
  a.reset();
  CHECK(serves_within(*server, 1));
  CHECK(server->newest_note()->line == "from b");
  b.reset();
  CHECK(serves_within(*server, 0));
  CHECK(!server->newest_note());
  return 0;
}

// A sender that dials out reads its reader's notes, as the one case that needs
// them does: listen false, add_viewer, a Listener's socket adopted.
int dial_out_case() {
  networking::ServerOptions options;
  options.listen = false;
  options.read_notes = true;
  vkc::Result<std::unique_ptr<networking::StreamServer>> started =
      networking::StreamServer::start(net::geometry_session(), options);
  CHECK(started.ok());
  networking::StreamServer& server = **started;
  vkc::Result<std::unique_ptr<vkc::net::Listener>> listener =
      vkc::net::Listener::create("127.0.0.1", 0);
  CHECK(listener.ok());
  vkc::Result<vkc::UniqueFd> dialled =
      vkc::net::dial("127.0.0.1", (*listener)->port(), 2s);
  CHECK(dialled.ok());
  CHECK(server.add_viewer(std::move(dialled).value()).ok());
  vkc::Result<vkc::UniqueFd> accepted = (*listener)->accept(2s);
  CHECK(accepted.ok());
  vkc::Result<std::unique_ptr<networking::StreamClient>> client =
      networking::StreamClient::adopt(std::move(accepted).value());
  CHECK(client.ok());
  CHECK(
      (*client)->send_note(note_of("up the dialled socket", Tone::Good)).ok());
  CHECK(line_within(server, "up the dialled socket"));
  CHECK(server.newest_note()->tone == Tone::Good);
  return 0;
}

// --- The end of a take. ------------------------------------------------------

// A sender finishes with bytes still in its kernel for a reader that has not
// read them, few enough for its socket to poll writable (under half the 128 KiB
// mark on Linux), so finish does not wait. With no note the reader reads every
// record and the sender's end. A note that lands after the close draws a reset,
// which may cost the tail: what the reader reads is still whole records, in
// order, from the first, then one end, the same on every later call.
int at_the_end_case() {
  constexpr std::size_t kSize = 4 << 10;
  constexpr std::uint64_t kRecords = 12;
  const auto read_a_take = [](bool note, std::uint64_t* read,
                              std::string* end) {
    auto server = note_server();
    CHECK(server != nullptr);
    vkc::UniqueFd fd = net::connect_loopback(server->port(), 8 << 10);
    CHECK(fd.valid());
    vkc::Result<std::unique_ptr<networking::StreamClient>> adopted =
        networking::StreamClient::adopt(std::move(fd));
    CHECK(adopted.ok());
    networking::StreamClient& client = **adopted;
    CHECK(net::wait_session(client));
    for (std::uint64_t seq = 0; seq < kRecords; ++seq) {
      const Bytes payload = net::record_payload(seq, kSize);
      CHECK(server
                ->publish(net::record_info(seq, seq == 0), payload.data(),
                          payload.size())
                .ok());
    }
    const Clock::time_point before = Clock::now();
    server->finish();
    CHECK(Clock::now() - before < networking::kFinishLimit);
    CHECK(server->viewers() == 0 && !server->newest_note());
    if (note) CHECK(client.send_note(note_of("too late")).ok());
    *read = 0;
    Got got;
    vkc::Result<bool> r = true;
    const Clock::time_point limit = Clock::now() + 5s;
    while (Clock::now() < limit) {
      r = client.next(10ms, &got.record, &got.payload);
      if (!r.ok()) break;
      if (*r) {
        CHECK(got.record.info.seq == *read);
        CHECK(got.payload == net::record_payload(*read, kSize));
        ++*read;
      }
    }
    CHECK(!r.ok() && r.status().domain() == Code::IoError);
    *end = r.status().message();
    const vkc::Result<bool> again = client.next(0ms, &got.record, &got.payload);
    CHECK(!again.ok() && again.status().message() == *end);
    return 0;
  };
  std::uint64_t read = 0;
  std::string end;
  CHECK(read_a_take(false, &read, &end) == 0);
  CHECK(read == kRecords);
  CHECK(end.find("the sender closed the connection") != std::string::npos);
  CHECK(read_a_take(true, &read, &end) == 0);
  CHECK(read <= kRecords);
  return 0;
}

// --- Threads. ---------------------------------------------------------------

// A publisher, a client that reads and says, and a thread reading newest_note
// and viewers() all at once, joined by a client that comes and goes, for
// ThreadSanitizer: every record read in order, and a note is always one sent.
int threads_case() {
  constexpr std::uint64_t kRecords = 400;
  auto server = note_server();
  CHECK(server != nullptr);
  auto client = net::connect(server->port());
  CHECK(client != nullptr && net::wait_session(*client));
  std::atomic<bool> stop{false};
  std::atomic<bool> wrong{false};
  std::thread reader([&] {
    while (!stop) {
      const std::optional<Note> note = server->newest_note();
      if (note && note->line.rfind("note", 0) != 0) wrong = true;
      if (server->viewers() > 2) wrong = true;
    }
  });
  std::thread visitor([&] {
    const std::uint16_t port = server->port();
    while (!stop) {
      auto other = net::connect(port);
      if (other == nullptr || !net::wait_session(*other, 1s)) continue;
      (void)other->send_note(note_of("note from a visitor"));
      // Gone before the next one comes, for the reader's count of 2.
      other.reset();
      (void)serves_within(*server, 1);
    }
  });
  std::thread publisher([&] {
    for (std::uint64_t seq = 0; seq < kRecords; ++seq) {
      const Bytes payload = net::record_payload(seq, 1000);
      (void)server->publish(net::record_info(seq, seq % 40 == 0),
                            payload.data(), payload.size());
    }
  });
  std::uint64_t last = 0;
  bool order = true;
  bool got_last = false;
  const Clock::time_point end = Clock::now() + 10s;
  while (!got_last && Clock::now() < end) {
    Got got;
    if (!net::next_within(*client, &got, 100ms)) continue;
    order = order && (got.record.info.seq == 0 || got.record.info.seq > last);
    last = got.record.info.seq;
    got_last = last == kRecords - 1;
    (void)client->send_note(note_of("note " + std::to_string(last)));
  }
  publisher.join();
  stop = true;
  visitor.join();
  reader.join();
  CHECK(!wrong);
  CHECK(order && got_last);
  return 0;
}

}  // namespace

int main() {
  if (arrives_case() != 0) return 1;
  if (burst_case() != 0) return 1;
  if (two_in_one_read_case() != 0) return 1;
  if (byte_at_a_time_case() != 0) return 1;
  if (wire_form_case() != 0) return 1;
  if (payload_rules_case() != 0) return 1;
  if (over_long_case() != 0) return 1;
  if (bad_bytes_case() != 0) return 1;
  if (off_by_default_case() != 0) return 1;
  if (wire_down_unchanged_case() != 0) return 1;
  if (client_reads_every_record_case() != 0) return 1;
  if (full_socket_case() != 0) return 1;
  if (connect_and_end_case() != 0) return 1;
  if (leaves_with_its_viewer_case() != 0) return 1;
  if (dial_out_case() != 0) return 1;
  if (at_the_end_case() != 0) return 1;
  if (threads_case() != 0) return 1;
  std::puts("networking notes: OK");
  return 0;
}
