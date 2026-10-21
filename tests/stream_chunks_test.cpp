// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// Host-only tests of the .vrst chunks in memory (stream/chunks.hpp): the
// encoder's bytes against the writer's file on 1 and 2 tracks; its refusals
// against the writer's, messages included; the parser fed a writer's file,
// finished or not, whole, a byte at a time and in pseudo-random pieces,
// against the reader; each refusal at the byte that completes what it
// refuses, bodies of 1 GiB before any of them arrives; the chunks it skips;
// and a failure returned by every later call.

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "stream_files.hpp"
#include "test_check.hpp"
#include "volumetric_kit/recon/stream/chunks.hpp"
#include "volumetric_kit/recon/stream/stream_reader.hpp"
#include "volumetric_kit/recon/stream/stream_writer.hpp"

namespace vkc = volumetric_kit::core;
namespace stream = volumetric_kit::recon::stream;
namespace vrst = vr_test::vrst;

namespace {

using Code = vkc::Status::Code;
using vrst::Bytes;
using vrst::info;
using vrst::pattern;
using vrst::same;

stream::Session session_of(std::size_t tracks) {
  stream::Session s;
  for (std::uint8_t i = 0; i < 16; ++i) s.id[i] = static_cast<std::uint8_t>(i);
  s.created_ns = 1760000000000000001ull;
  s.writer = "chunks";
  s.calibration = "{\"device_calibration\": {}}";
  for (std::size_t t = 0; t < tracks; ++t) {
    stream::Track track;
    track.kind = stream::TrackKind::Geometry;
    track.rate = {30, 1};
    track.sensor = "sensor " + std::to_string(t);
    s.tracks.push_back(track);
  }
  return s;
}

struct Appended {
  stream::RecordInfo info;
  Bytes payload;
};

// Records on `tracks` tracks, keys and not, an empty payload of each, every
// padding from 0 to 7, and payloads above 64 KiB. The last pads by 5.
std::vector<Appended> records_on(std::size_t tracks) {
  const std::size_t sizes[] = {0,  0,  1,   2,    3,     4,      5,  6,
                               7,  8,  9,   63,   64,    65,     56, 4096,
                               57, 15, 300, 4097, 65537, 300000, 3};
  std::vector<Appended> out;
  std::vector<std::uint64_t> seqs(tracks, 0);
  std::uint64_t i = 0;
  for (const std::size_t size : sizes) {
    const auto track = static_cast<std::uint16_t>(i % tracks);
    seqs[track] += track == 1 ? 2 : 1;  // track 1 lost records
    const bool key = i % 3 == 0;
    const std::uint64_t set = i % 4 == 0 ? stream::kNoSet : i;
    out.push_back(
        {info(track, key, 1000 * i, seqs[track], set), pattern(size, i)});
    ++i;
  }
  return out;
}

std::unique_ptr<stream::StreamWriter> create(const std::string& path,
                                             const stream::Session& session) {
  vkc::Result<std::unique_ptr<stream::StreamWriter>> w =
      stream::StreamWriter::create(path, session);
  if (!w.ok()) {
    std::fprintf(stderr, "create %s: %s\n", path.c_str(),
                 w.status().message().c_str());
    return nullptr;
  }
  return std::move(w).value();
}

// Write `records` to a new file at `path`, finished or not.
bool write_file(const std::string& path, const stream::Session& session,
                const std::vector<Appended>& records, bool finish) {
  auto w = create(path, session);
  if (w == nullptr) return false;
  for (const Appended& a : records) {
    if (!w->append(a.info, a.payload.data(), a.payload.size()).ok()) {
      return false;
    }
  }
  return !finish || w->finish().ok();
}

// The record's chunk, as `framing` frames it.
void append_chunk(Bytes& out,
                  const stream::ChunkEncoder::RecordFraming& framing,
                  const Bytes& payload) {
  out.insert(out.end(), framing.header.begin(), framing.header.end());
  out.insert(out.end(), payload.begin(), payload.end());
  out.resize(out.size() + framing.padding, 0);
}

// --- The encoder. ------------------------------------------------------------

// The encoder's chunks are the writer's file before its footer, and the
// whole of an unfinished writer's file.
int encoder_bytes_case(const std::string& dir) {
  for (const std::size_t tracks : {1, 2}) {
    const stream::Session session = session_of(tracks);
    const std::vector<Appended> records = records_on(tracks);
    const std::string base = dir + "/encoder_" + std::to_string(tracks);
    CHECK(write_file(base + "_finished.vrst", session, records, true));
    CHECK(write_file(base + "_unfinished.vrst", session, records, false));

    vkc::Result<stream::ChunkEncoder> encoder =
        stream::ChunkEncoder::create(session);
    CHECK(encoder.ok());
    Bytes bytes = encoder->session_chunk();
    CHECK(bytes.size() % 8 == 0);
    for (const Appended& a : records) {
      const vkc::Result<stream::ChunkEncoder::RecordFraming> framing =
          encoder->frame(a.info, a.payload.data(), a.payload.size());
      CHECK(framing.ok());
      CHECK(framing->padding == vrst::padding(a.payload.size()));
      append_chunk(bytes, *framing, a.payload);
    }
    const Bytes finished = vr_test::read_bytes(base + "_finished.vrst");
    const std::uint64_t footer = vrst::index_offset(finished);
    CHECK(footer == bytes.size());
    CHECK(Bytes(finished.begin(),
                finished.begin() + static_cast<std::ptrdiff_t>(footer)) ==
          bytes);
    CHECK(vr_test::read_bytes(base + "_unfinished.vrst") == bytes);
  }
  return 0;
}

// Each session and record the writer refuses, the encoder refuses, and the
// writer's message is the encoder's, after the call and the path. A refused
// record leaves both as they were.
int encoder_refusals_case(const std::string& dir) {
  const std::string fresh = dir + "/refused.vrst";
  const auto refused_alike = [&](const stream::Session& s) {
    const vkc::Status encoder = stream::ChunkEncoder::create(s).status();
    const vkc::Status writer = stream::StreamWriter::create(fresh, s).status();
    return encoder.domain() == Code::InvalidArgument &&
           writer.domain() == Code::InvalidArgument &&
           writer.message() ==
               "StreamWriter::create: " + fresh + ": " + encoder.message() &&
           vr_test::read_bytes(fresh).empty();
  };
  const stream::Session good = session_of(1);
  stream::Session s = good;
  s.tracks.clear();
  CHECK(refused_alike(s));
  s = good;
  s.tracks[0].kind = static_cast<stream::TrackKind>(0);
  CHECK(refused_alike(s));
  s = good;
  s.tracks[0].rate = {30, 0};
  CHECK(refused_alike(s));
  s = good;
  s.calibration.assign(stream::kMaxSessionBytes, 'c');
  CHECK(refused_alike(s));
  s = good;
  s.tracks.assign(65536, good.tracks[0]);
  CHECK(refused_alike(s));
  s.tracks.resize(65535);  // the most
  CHECK(stream::ChunkEncoder::create(s).ok());

  const std::string path = dir + "/rules.vrst";
  const stream::Session two = session_of(2);
  vkc::Result<stream::ChunkEncoder> encoder = stream::ChunkEncoder::create(two);
  CHECK(encoder.ok());
  Bytes expected = encoder->session_chunk();
  {
    auto w = create(path, two);
    CHECK(w != nullptr);
    const Bytes p = pattern(9, 1);
    const auto framed = [&](const stream::RecordInfo& i, const Bytes& payload,
                            const std::uint8_t* data) {
      const vkc::Result<stream::ChunkEncoder::RecordFraming> f =
          encoder->frame(i, data, payload.size());
      if (!f.ok() || !w->append(i, data, payload.size()).ok()) return false;
      append_chunk(expected, *f, payload);
      return true;
    };
    const auto refused = [&](const stream::RecordInfo& i,
                             const std::uint8_t* data, std::size_t n) {
      const std::uint64_t size = w->size();
      const vkc::Status e = encoder->frame(i, data, n).status();
      const vkc::Status a = w->append(i, data, n);
      return e.domain() == Code::InvalidArgument &&
             a.domain() == Code::InvalidArgument &&
             a.message() ==
                 "StreamWriter::append: " + path + ": " + e.message() &&
             w->size() == size;
    };
    CHECK(framed(info(0, true, 100, 5), p, p.data()));
    CHECK(refused(info(2, true, 100, 6), p.data(), p.size()));   // no track 2
    CHECK(refused(info(0, false, 100, 5), p.data(), p.size()));  // seq equal
    CHECK(refused(info(0, false, 100, 4), p.data(), p.size()));  // seq lower
    CHECK(refused(info(0, true, 99, 6), p.data(), p.size()));    // key pts back
    CHECK(refused(info(0, false, 100, 6), nullptr, 3));          // no payload
    // What the rules allow, seq 6 included, which no refusal counted:
    // non-key pts going back, a key at the last key's pts, another track's
    // own seq, and an empty payload with no pointer.
    CHECK(framed(info(0, false, 1, 6), p, p.data()));
    CHECK(framed(info(0, true, 100, 7), p, p.data()));
    CHECK(framed(info(1, true, 0, 0), p, p.data()));
    CHECK(framed(info(1, false, 0, 1), Bytes{}, nullptr));
  }
  CHECK(vr_test::read_bytes(path) == expected);
  return 0;
}

// --- The parser. -------------------------------------------------------------

// What a parser gave, fed some bytes a piece at a time and drained between.
struct Parsed {
  std::vector<stream::Record> records;
  std::vector<Bytes> payloads;
  vkc::Status failure;  // the first, from feed or next
  bool failed_in_feed = false;
  std::size_t fed = 0;         // the bytes fed when it failed
  std::size_t session_at = 0;  // the bytes fed when session() was first set
};

// Feed `bytes` to `parser` in pieces `piece` sizes (0 is 1), taking every
// record after each piece, until the bytes end or a call fails.
template <class Piece>
Parsed parse(stream::ChunkParser& parser, const Bytes& bytes, Piece piece) {
  Parsed out;
  std::size_t at = 0;
  while (at < bytes.size()) {
    const std::size_t n = std::min<std::size_t>(
        std::max<std::size_t>(piece(), 1), bytes.size() - at);
    const vkc::Status fed = parser.feed(bytes.data() + at, n);
    at += n;
    if (out.session_at == 0 && parser.session() != nullptr) {
      out.session_at = at;
    }
    if (!fed.ok()) {
      out.failure = fed;
      out.failed_in_feed = true;
      out.fed = at;
      return out;
    }
    for (;;) {
      stream::Record record;
      Bytes payload;
      const vkc::Result<bool> got = parser.next(&record, &payload);
      if (!got.ok()) {
        out.failure = got.status();
        out.fed = at;
        return out;
      }
      if (!got.value()) break;
      out.records.push_back(record);
      out.payloads.push_back(std::move(payload));
    }
  }
  return out;
}

Parsed parse_whole(const Bytes& bytes) {
  stream::ChunkParser parser;
  return parse(parser, bytes, [&] { return bytes.size(); });
}

Parsed parse_bytewise(const Bytes& bytes) {
  stream::ChunkParser parser;
  return parse(parser, bytes, [] { return std::size_t{1}; });
}

// Pieces of 1 to `most` bytes, from a seeded LCG.
Parsed parse_split(const Bytes& bytes, std::uint64_t seed, std::size_t most) {
  stream::ChunkParser parser;
  std::uint64_t x = seed;
  return parse(parser, bytes, [&] {
    x = x * 6364136223846793005ull + 1442695040888963407ull;
    return static_cast<std::size_t>(1 + (x >> 33) % most);
  });
}

// The parse found what the reader of the same bytes reads: the same records,
// offsets and payloads, with no failure.
bool same_as_reader(const Parsed& parsed, const vrst::ReadBack& back) {
  if (!parsed.failure.ok()) {
    std::fprintf(stderr, "parse: %s\n", parsed.failure.message().c_str());
    return false;
  }
  if (parsed.records.size() != back.records.size() ||
      back.payloads.size() != back.records.size()) {
    return false;
  }
  for (std::size_t i = 0; i < back.records.size(); ++i) {
    const stream::Record& a = parsed.records[i];
    const stream::Record& b = back.records[i];
    if (!same(a.info, b.info) || a.offset != b.offset || a.size != b.size ||
        parsed.payloads[i] != back.payloads[i]) {
      return false;
    }
  }
  return true;
}

// Every way of feeding `bytes` gives the session and the records the reader
// of `path`, holding the same bytes, gives.
int parses_as_read(const std::string& path, const Bytes& bytes) {
  auto reader = vrst::open(path);
  CHECK(reader != nullptr);
  const vrst::ReadBack back = vrst::read_all(*reader);
  CHECK(back.ended && back.read_errors.empty());
  std::vector<Parsed> parses;
  parses.push_back(parse_whole(bytes));
  parses.push_back(parse_bytewise(bytes));
  for (const std::size_t most : {16, 600, 70000, 1 << 20}) {
    for (std::uint64_t seed = 1; seed <= 3; ++seed) {
      parses.push_back(parse_split(bytes, seed * most, most));
    }
  }
  // The session is in once its header and body are, before its padding.
  const std::uint64_t session_bytes =
      vrst::get(bytes, 4, 4) + vrst::get(bytes, 8, 8);
  CHECK(parses[1].session_at == session_bytes);
  for (const Parsed& parsed : parses) {
    CHECK(same_as_reader(parsed, back));
    CHECK(parsed.session_at != 0 && parsed.session_at >= session_bytes);
  }
  stream::ChunkParser parser;
  CHECK(parser.session() == nullptr);
  CHECK(parser.feed(bytes.data(), bytes.size()).ok());
  CHECK(parser.session() != nullptr &&
        same(*parser.session(), reader->session()));
  return 0;
}

int parser_round_trip_case(const std::string& dir) {
  for (const std::size_t tracks : {1, 2}) {
    const stream::Session session = session_of(tracks);
    const std::vector<Appended> records = records_on(tracks);
    const std::string base = dir + "/parse_" + std::to_string(tracks);
    // A finished file: its footer, INDX and TAIL, skipped.
    const std::string finished = base + "_finished.vrst";
    CHECK(write_file(finished, session, records, true));
    CHECK(parses_as_read(finished, vr_test::read_bytes(finished)) == 0);
    // A killed writer's, cut inside the last record's padding: it is whole.
    const std::string cut = base + "_cut.vrst";
    CHECK(write_file(cut, session, records, false));
    Bytes bytes = vr_test::read_bytes(cut);
    bytes.resize(bytes.size() - 2);
    CHECK(vrst::write_bytes(cut, bytes));
    CHECK(parses_as_read(cut, bytes) == 0);
    CHECK(parse_whole(bytes).records.size() == records.size());
  }
  return 0;
}

// Another tag, a record's wider header, a record setting a flag this version
// does not know, and a second VRST, after the session, are skipped as the
// reader skips them; the reader returns the flagged record as Unsupported.
int parser_skips_case(const std::string& dir) {
  const std::string path = dir + "/skips.vrst";
  CHECK(write_file(path, session_of(1), {{info(0, true, 1, 0), pattern(20, 1)}},
                   false));
  Bytes bytes = vr_test::read_bytes(path);
  for (const Bytes& c :
       {vrst::chunk("ZZZZ", Bytes(16, 0x5A), pattern(13, 2)),
        vrst::record(0, 1, 2, 1, 9, pattern(77, 3), 8),  // H 64
        vrst::record(0, 2, 3, 2, 9, pattern(5, 4)),      // flag 2
        vrst::chunk("VRST", vrst::session_fields(0, 1, 0),
                    vrst::session_body(1)),
        vrst::record(0, 0, 4, 3, 9, pattern(9, 5))}) {
    bytes.insert(bytes.end(), c.begin(), c.end());
  }
  CHECK(vrst::write_bytes(path, bytes));
  auto reader = vrst::open(path);
  CHECK(reader != nullptr);
  const vrst::ReadBack back = vrst::read_all(*reader);
  CHECK(back.records.size() == 3 && back.next_errors.size() == 1 &&
        back.next_errors[0] == Code::Unsupported);
  CHECK(parses_as_read(path, bytes) == 0);
  return 0;
}

// Each refusal comes from the feed of the byte that completes what it
// refuses, `at` bytes in.
struct Refusal {
  const char* name;
  Bytes bytes;
  std::size_t at;
  Code code;
};

int parser_refusals_case(const std::string&) {
  const Bytes body = vrst::session_body(1);
  const Bytes good = vrst::chunk("VRST", vrst::session_fields(0, 1, 0), body);
  const std::size_t g = good.size();
  const auto after = [&](const Bytes& chunk) {
    Bytes b = good;
    b.insert(b.end(), chunk.begin(), chunk.end());
    return b;
  };
  const std::uint64_t gib = std::uint64_t{1} << 30;
  const std::uint64_t most = stream::ChunkParser::kMaxBodyBytes;
  const Bytes recd = vrst::record_fields(0, 0, 0, 0, 0);
  const Bytes record = vrst::record(0, 1, 0, 0, 0, pattern(20, 1));
  const Bytes zzzz = vrst::chunk("ZZZZ", Bytes(16, 0x5A), pattern(13, 2));
  Bytes extra = body;
  extra.push_back(0);
  const Bytes http = {'H', 'T', 'T', 'P', '/', '1',  '.',  '1',  ' ', '2',
                      '0', '0', ' ', 'O', 'K', '\r', '\n', '\r', '\n'};
  const std::vector<Refusal> refusals = {
      {"not VRST", http, 8, Code::InvalidArgument},
      {"session version",
       vrst::chunk("VRST", vrst::session_fields(1, 1, 0), body), 56,
       Code::Unsupported},
      {"session flags",
       vrst::chunk("VRST", vrst::session_fields(0, 1, 4), body), 56,
       Code::Unsupported},
      {"session H below 56", vrst::chunk("VRST", Bytes(24, 0), body), 8,
       Code::InvalidArgument},
      {"session H above 4096", vrst::chunk("VRST", Bytes(4080, 0), body), 8,
       Code::InvalidArgument},
      {"session header CRC", vrst::flipped(good, 30, 0), 56,
       Code::InvalidArgument},
      {"session body of 1 GiB",
       vrst::header("VRST", vrst::session_fields(0, 1, 0), gib, 0), 56,
       Code::InvalidArgument},
      {"session body CRC", vrst::flipped(good, 60, 3), 56 + body.size(),
       Code::InvalidArgument},
      {"session body past its strings",
       vrst::chunk("VRST", vrst::session_fields(0, 1, 0), extra),
       56 + extra.size(), Code::InvalidArgument},
      {"H not a multiple of 8", after(vrst::header("ZZZZ", Bytes(4, 0), 0, 0)),
       g + 8, Code::InvalidArgument},
      {"RECD H below 56", after(vrst::header("RECD", Bytes(24, 0), 0, 0)),
       g + 8, Code::InvalidArgument},
      {"INDX H below 40", after(vrst::header("INDX", Bytes(8, 0), 0, 0)), g + 8,
       Code::InvalidArgument},
      {"RECD header CRC", after(vrst::flipped(record, 30, 0)), g + 56,
       Code::InvalidArgument},
      {"RECD of no track", after(vrst::record(1, 1, 0, 0, 0, pattern(4, 1))),
       g + 56, Code::InvalidArgument},
      {"RECD body of 1 GiB", after(vrst::header("RECD", recd, gib, 0)), g + 56,
       Code::OutOfMemory},
      {"RECD body one past the most",
       after(vrst::header("RECD", recd, most + 1, 0)), g + 56,
       Code::OutOfMemory},
      {"another tag's body of 1 GiB",
       after(vrst::header("ZZZZ", Bytes(16, 0x5A), gib, 0)), g + 40,
       Code::OutOfMemory},
      {"RECD body CRC", after(vrst::flipped(record, 56 + 3, 2)), g + 56 + 20,
       Code::InvalidArgument},
      {"another tag's body CRC", after(vrst::flipped(zzzz, 40 + 1, 0)),
       g + 40 + 13, Code::InvalidArgument},
  };
  for (const Refusal& r : refusals) {
    const Parsed parsed = parse_bytewise(r.bytes);
    if (parsed.failure.domain() != r.code || !parsed.failed_in_feed ||
        parsed.fed != r.at) {
      std::fprintf(stderr, "refusal %s: %s after %zu bytes (from %s): %s\n",
                   r.name,
                   std::string(vkc::to_string(parsed.failure.domain())).c_str(),
                   parsed.fed, parsed.failed_in_feed ? "feed" : "next",
                   parsed.failure.message().c_str());
      return 1;
    }
  }
  // A body of the most is taken: the parser waits for it.
  stream::ChunkParser parser;
  const Bytes at_most = after(vrst::header("RECD", recd, most, 0));
  CHECK(parser.feed(at_most.data(), at_most.size()).ok());
  stream::Record none;
  Bytes payload;
  const vkc::Result<bool> waits = parser.next(&none, &payload);
  CHECK(waits.ok() && !waits.value());
  return 0;
}

// A failure is final: the record before it is taken, then every call
// returns the failure, more bytes fed included, and the session stays.
int parser_latch_case(const std::string&) {
  const Bytes good =
      vrst::chunk("VRST", vrst::session_fields(0, 1, 0), vrst::session_body(1));
  Bytes bytes = good;
  for (const Bytes& c :
       {vrst::record(0, 1, 0, 0, 0, pattern(20, 1)),
        vrst::flipped(vrst::record(0, 0, 1, 1, 1, pattern(20, 2)), 56 + 3, 2),
        vrst::record(0, 0, 2, 2, 2, pattern(20, 3))}) {
    bytes.insert(bytes.end(), c.begin(), c.end());
  }
  stream::ChunkParser parser;
  // The whole feed stops at the first record, so the damage after it is
  // found by the next() after that record.
  CHECK(parser.feed(bytes.data(), bytes.size()).ok());
  stream::Record record;
  Bytes payload;
  const vkc::Result<bool> first = parser.next(&record, &payload);
  CHECK(first.ok() && first.value() && payload == pattern(20, 1));
  const vkc::Result<bool> damaged = parser.next(&record, &payload);
  CHECK(damaged.status().domain() == Code::InvalidArgument);
  const std::string message = damaged.status().message();
  CHECK(message.find("fails its CRC") != std::string::npos);
  for (int i = 0; i < 3; ++i) {
    const vkc::Status fed = parser.feed(good.data(), good.size());
    CHECK(fed.domain() == Code::InvalidArgument && fed.message() == message);
    const vkc::Result<bool> next = parser.next(&record, &payload);
    CHECK(next.status().domain() == Code::InvalidArgument &&
          next.status().message() == message);
  }
  CHECK(parser.session() != nullptr);
  return 0;
}

}  // namespace

int main() {
  vr_test::Scratch scratch("recon_stream_chunks_");
  const std::string dir = scratch.path.string();
  if (encoder_bytes_case(dir) != 0) return 1;
  if (encoder_refusals_case(dir) != 0) return 1;
  if (parser_round_trip_case(dir) != 0) return 1;
  if (parser_skips_case(dir) != 0) return 1;
  if (parser_refusals_case(dir) != 0) return 1;
  if (parser_latch_case(dir) != 0) return 1;
  std::puts("stream chunks: OK");
  return 0;
}
