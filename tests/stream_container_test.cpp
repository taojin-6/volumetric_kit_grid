// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// Host-only tests of the .vrst container through its public API: golden
// bytes, whose CRCs Python's zlib.crc32 computed; the CRC of every length to
// 64 at every alignment against a bitwise one; round trips of the session,
// every record field and the payloads; seeking by key; unknown kinds, chunks
// and header fields; the payload limit and a body size near 2^64; every
// writer and reader refusal, a file cut short after its open included; reads
// into caller memory; two threads reading one reader; and a track's nominal
// period, 30 a second's for an unknown rate.

#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "stream_files.hpp"
#include "test_check.hpp"
#include "volumetric_kit/recon/stream/stream_reader.hpp"
#include "volumetric_kit/recon/stream/stream_writer.hpp"

namespace vkc = volumetric_kit::core;
namespace stream = volumetric_kit::recon::stream;
namespace vrst = vr_test::vrst;

namespace {

using Code = vkc::Status::Code;
using vrst::Bytes;
using vrst::info;
using vrst::open;
using vrst::pattern;
using vrst::same;

Bytes unhex(const char* hex) {
  Bytes b;
  for (const char* p = hex; p[0] != '\0' && p[1] != '\0'; p += 2) {
    b.push_back(
        static_cast<std::uint8_t>(std::stoi(std::string(p, 2), nullptr, 16)));
  }
  return b;
}

stream::Track geometry() {
  stream::Track t;
  t.kind = stream::TrackKind::Geometry;
  return t;
}

stream::Session one_track(const std::string& writer = "test") {
  stream::Session s;
  s.writer = writer;
  s.tracks.push_back(geometry());
  return s;
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

Code open_code(const std::string& path) {
  return stream::StreamReader::open(path).status().domain();
}

bool append(stream::StreamWriter& w, const stream::RecordInfo& i,
            const Bytes& payload) {
  return w.append(i, payload.data(), payload.size()).ok();
}

std::uint64_t file_size(const std::string& path) {
  struct stat info{};
  return ::stat(path.c_str(), &info) == 0
             ? static_cast<std::uint64_t>(info.st_size)
             : ~std::uint64_t{0};
}

// --- The CRC. ----------------------------------------------------------------

// Every length to 64, and a few longer, from each of the 8 alignments of a
// buffer: each record's body_crc is the bitwise CRC of what was appended.
int crc_case(const std::string& dir) {
  CHECK(vrst::crc32(reinterpret_cast<const std::uint8_t*>("123456789"), 9) ==
        0xCBF43926u);
  const std::string path = dir + "/crc.vrst";
  auto w = create(path, one_track());
  CHECK(w != nullptr);
  const Bytes buffer = pattern(5000, 1);
  std::vector<std::uint32_t> expected;
  std::uint64_t seq = 0;
  for (std::size_t align = 0; align < 8; ++align) {
    for (std::size_t n : {0,  1,  2,  3,  4,  5,  6,  7,   8,    9,    15,  16,
                          17, 31, 32, 33, 63, 64, 65, 255, 4096, 4097, 4991}) {
      CHECK(w->append(info(0, false, 0, seq++), buffer.data() + align, n).ok());
      expected.push_back(vrst::crc32(buffer.data() + align, n));
    }
    for (std::size_t n = 10; n < 64; ++n) {
      CHECK(w->append(info(0, false, 0, seq++), buffer.data() + align, n).ok());
      expected.push_back(vrst::crc32(buffer.data() + align, n));
    }
  }
  CHECK(w->finish().ok());
  const Bytes file = vr_test::read_bytes(path);
  std::size_t i = 0;
  for (const vrst::ChunkAt& c : vrst::chunks(file)) {
    if (c.tag != "RECD") continue;
    CHECK(i < expected.size());
    CHECK(vrst::get(file, c.offset + 16, 4) == expected[i]);
    ++i;
  }
  CHECK(i == expected.size());
  return 0;
}

// --- Golden bytes. -----------------------------------------------------------

// Two tracks, three records, the footer: 552 bytes. Chunks at 0 (VRST), 144,
// 208 and 264 (RECD), 328 (INDX) and 512 (TAIL).
constexpr const char* kGoldenSmall =
    "565253543800000057000000000000000d36988c000000000001020304050607"
    "08090a0b0c0d0e0f15cd0bdcacc66c1802000000d14b580d010001001e000000"
    "010000000000000000000000000000000201000030750000e903000080020000"
    "400200000000000006000000676f6c64656e020000007b7d0000000000000000"
    "04000000434c324103000000703d310052454344380000000300000000000000"
    "c241243500000100000000000000000000000000000000000000000000000000"
    "000000006347f7e6616263000000000052454344380000000000000000000000"
    "000000000100000000c29be0acc66c180700000000000000ffffffffffffffff"
    "000000000844d19b524543443800000008000000000000009f68aa8800000100"
    "55a0fc01000000000100000000000000010000000000000000000000655ff4d9"
    "0001020304050607494e445828000000900000000000000015e3a3b802000000"
    "020000000000000000000000052264a802000000000000000200000000000000"
    "0b00000000000000000000000000000055a0fc01000000000100000000000000"
    "0000000000000000000000000000000000c29be0acc66c1800c29be0acc66c18"
    "9000000000000000000000000000000000000000000000000000000000000000"
    "080100000000000055a0fc010000000001000000000000000000000000000000"
    "5441494c28000000000000000000000000000000000000004801000000000000"
    "000000001582d9ae";

// One record of 65,543 bytes, byte i = (i^2 + 13 i + 7) mod 256: its CRC
// (0xc9e7ec26) runs 8,192 slices and a tail. The session chunk (104 bytes),
// the record's header, and the footer after the record's padding.
constexpr const char* kGoldenBigSession =
    "56525354380000002b000000000000001704f680000000000000000000000000"
    "000000000000000000000000000000000100000064a6c2fa0100010000000000"
    "0100000000000000000000000000000003000000626967000000000000000000"
    "0000000000000000";
constexpr const char* kGoldenBigRecord =
    "5245434438000000070001000000000026ece7c9000001000500000000000000"
    "0000000000000000ffffffffffffffff00000000e2fd3b21";
constexpr const char* kGoldenBigFooter =
    "494e445828000000480000000000000094bd8f18010000000100000000000000"
    "0000000009367246010000000000000001000000000000000700010000000000"
    "0500000000000000050000000000000068000000000000000500000000000000"
    "ffffffffffffffff00000000000000005441494c280000000000000000000000"
    "0000000000000000a80001000000000000000000033500c8";

int golden_case(const std::string& dir) {
  {
    stream::Session s;
    for (std::uint8_t i = 0; i < 16; ++i) s.id[i] = i;
    s.created_ns = 1760000000123456789ull;
    s.writer = "golden";
    s.calibration = "{}";
    stream::Track t0 = geometry();
    t0.rate = {30, 1};
    stream::Track t1;
    t1.kind = static_cast<stream::TrackKind>(0x0102);
    t1.clock = stream::kHostClock;
    t1.rate = {30000, 1001};
    t1.width = 640;
    t1.height = 576;
    t1.sensor = "CL2A";
    t1.params = "p=1";
    s.tracks = {t0, t1};
    const std::string path = dir + "/golden.vrst";
    auto w = create(path, s);
    CHECK(w != nullptr);
    CHECK(append(*w, info(0, true, 0, 0, 0), {'a', 'b', 'c'}));
    CHECK(append(*w, info(1, false, 1760000000200000000ull, 7), {}));
    CHECK(append(*w, info(0, true, 33333333, 1, 1), {0, 1, 2, 3, 4, 5, 6, 7}));
    CHECK(w->finish().ok());
    CHECK(w->size() == 552);
    const Bytes file = vr_test::read_bytes(path);
    const Bytes expected = unhex(kGoldenSmall);
    for (std::size_t i = 0; i < file.size() && i < expected.size(); ++i) {
      if (file[i] != expected[i]) {
        std::fprintf(stderr, "golden: byte %zu differs\n", i);
        break;
      }
    }
    CHECK(file == expected);
  }
  {
    const std::string path = dir + "/golden_big.vrst";
    auto w = create(path, one_track("big"));
    CHECK(w != nullptr);
    Bytes payload(65543);
    for (std::size_t i = 0; i < payload.size(); ++i) {
      payload[i] = static_cast<std::uint8_t>((i * i + 13 * i + 7) & 0xFF);
    }
    CHECK(append(*w, info(0, true, 5, 0), payload));
    CHECK(w->finish().ok());
    const Bytes file = vr_test::read_bytes(path);
    CHECK(file.size() == 65856);
    const Bytes session = unhex(kGoldenBigSession);
    const Bytes record = unhex(kGoldenBigRecord);
    const Bytes footer = unhex(kGoldenBigFooter);
    CHECK(Bytes(file.begin(), file.begin() + 104) == session);
    CHECK(Bytes(file.begin() + 104, file.begin() + 160) == record);
    CHECK(Bytes(file.begin() + 160, file.begin() + 160 + 65543) == payload);
    CHECK(Bytes(file.end() - 152, file.end()) == footer);
    auto r = open(path);
    CHECK(r != nullptr);
    const vrst::ReadBack back = vrst::read_all(*r);
    CHECK(back.payloads.size() == 1 && back.payloads[0] == payload);
  }
  return 0;
}

// --- Round trips. ------------------------------------------------------------

struct Appended {
  stream::RecordInfo info;
  Bytes payload;
};

int round_trip_case(const std::string& dir) {
  stream::Session s;
  for (std::uint8_t i = 0; i < 16; ++i)
    s.id[i] = static_cast<std::uint8_t>(i * 17);
  s.created_ns = 1760000000000000001ull;
  s.writer = "round trip";
  s.calibration = std::string("{\"device_calibration\": {}}\0binary\xff", 34);
  stream::Track host = geometry();
  host.clock = stream::kHostClock;
  host.rate = {30, 1};
  stream::Track colour;
  colour.kind = static_cast<stream::TrackKind>(2);
  colour.clock = 1;
  colour.rate = {25, 1};
  colour.width = 3840;
  colour.height = 2160;
  colour.sensor = "CL2A141000N";
  colour.params = "colour description";
  stream::Track depth;
  depth.kind = static_cast<stream::TrackKind>(3);
  depth.clock = 2;
  depth.rate = {0, 1};
  depth.width = 640;
  depth.height = 576;
  depth.sensor = "CL2A141000M";
  s.tracks = {host, colour, depth};

  const std::string path = dir + "/round_trip.vrst";
  auto w = create(path, s);
  CHECK(w != nullptr);
  CHECK(w->size() == file_size(path));
  const std::size_t sizes[] = {0, 1, 7, 8, 9, 63, 4096, 65537, 300000};
  std::vector<Appended> appended;
  std::vector<stream::TrackStats> expected(3);
  std::uint64_t seqs[3] = {0, 100, 5};
  for (int i = 0; i < 27; ++i) {
    const auto track = static_cast<std::uint16_t>(i % 3);
    const bool key = i % 4 == 0 || i < 3;
    // Non-key pts go back now and then; key pts never do.
    const std::uint64_t pts =
        key ? 1000000000ull * static_cast<std::uint64_t>(i)
            : 1000000000ull * static_cast<std::uint64_t>(i) - 1500000000ull;
    seqs[track] += track == 2 ? 3 : 1;  // track 2 lost records
    const std::uint64_t set =
        track == 1 ? stream::kNoSet : static_cast<std::uint64_t>(i / 3);
    Appended a{info(track, key, pts, seqs[track], set),
               pattern(sizes[i % 9], static_cast<std::uint64_t>(i))};
    CHECK(append(*w, a.info, a.payload));
    CHECK(w->size() == file_size(path));
    stream::TrackStats& e = expected[track];
    if (e.records == 0) e.min_pts_ns = e.max_pts_ns = pts;
    e.min_pts_ns = std::min(e.min_pts_ns, pts);
    e.max_pts_ns = std::max(e.max_pts_ns, pts);
    ++e.records;
    e.keys += key ? 1 : 0;
    e.payload_bytes += a.payload.size();
    appended.push_back(std::move(a));
  }
  CHECK(w->finish().ok());
  CHECK(w->size() == file_size(path));

  auto r = open(path);
  CHECK(r != nullptr);
  CHECK(same(r->session(), s));
  const stream::Recovery& rec = r->recovery();
  CHECK(rec.finished && !rec.footer_damaged);
  CHECK(rec.end == file_size(path) && rec.torn_bytes == 0);
  CHECK(rec.damaged_bytes == 0 && rec.damaged_spans == 0);
  CHECK(r->stats().size() == 3);
  for (std::size_t t = 0; t < 3; ++t) {
    CHECK(same(r->stats()[t], expected[t]));
    CHECK(r->key_count(static_cast<std::uint16_t>(t)) == expected[t].keys);
  }
  const vrst::ReadBack back = vrst::read_all(*r);
  CHECK(back.ended && back.next_errors.empty() && back.read_errors.empty());
  CHECK(back.records.size() == appended.size());
  for (std::size_t i = 0; i < appended.size(); ++i) {
    CHECK(same(back.records[i].info, appended[i].info));
    CHECK(back.records[i].size == appended[i].payload.size());
    CHECK(back.records[i].offset % 8 == 0);
    CHECK(back.payloads[i] == appended[i].payload);
  }
  // The end stays the end.
  stream::Position end{rec.end};
  stream::Record none;
  const vkc::Result<bool> after = r->next(&end, &none);
  CHECK(after.ok() && !after.value());

  // One record, as a single coded frame ships.
  const std::string one = dir + "/one.vrst";
  auto w1 = create(one, one_track("codec_mesh"));
  CHECK(w1 != nullptr);
  const Bytes frame = pattern(66945, 99);
  CHECK(append(*w1, info(0, true, 0, 0, 0), frame));
  CHECK(w1->finish().ok());
  // The size cost: 112 (VRST) + 56 + 7 (RECD) + 112 (INDX) + 40 (TAIL).
  CHECK(file_size(one) == frame.size() + 327);
  auto r1 = open(one);
  CHECK(r1 != nullptr && r1->recovery().finished);
  const vrst::ReadBack back1 = vrst::read_all(*r1);
  CHECK(back1.payloads.size() == 1 && back1.payloads[0] == frame);
  const vkc::Result<stream::Position> first = r1->key(0, 0);
  CHECK(first.ok() && first.value().offset == r1->begin().offset);

  // A writer destroyed without finish leaves every record, unfinished.
  const std::string unfinished = dir + "/unfinished.vrst";
  {
    auto w2 = create(unfinished, s);
    CHECK(w2 != nullptr);
    for (std::size_t i = 0; i < 5; ++i) {
      CHECK(append(*w2, appended[i].info, appended[i].payload));
    }
  }
  auto r2 = open(unfinished);
  CHECK(r2 != nullptr);
  CHECK(!r2->recovery().finished && !r2->recovery().footer_damaged);
  CHECK(r2->recovery().torn_bytes == 0 && r2->recovery().damaged_spans == 0);
  const vrst::ReadBack back2 = vrst::read_all(*r2);
  CHECK(back2.payloads.size() == 5);
  for (std::size_t i = 0; i < 5; ++i) {
    CHECK(back2.payloads[i] == appended[i].payload);
  }
  return 0;
}

// --- Seeking. ----------------------------------------------------------------

int seek_case(const std::string& dir) {
  stream::Session s = one_track();
  s.tracks.push_back(geometry());
  const std::string path = dir + "/seek.vrst";
  auto w = create(path, s);
  CHECK(w != nullptr);
  // Track 0: keys at 100, 200, 200, 300 with non-keys between; track 1 keys
  // at other times, interleaved.
  const std::uint64_t key_pts[] = {100, 200, 200, 300};
  std::uint64_t seq0 = 0;
  std::uint64_t seq1 = 0;
  for (std::uint64_t k = 0; k < 4; ++k) {
    CHECK(append(*w, info(0, true, key_pts[k], seq0++), pattern(10, k)));
    CHECK(append(*w, info(1, true, 1000 + k, seq1++), pattern(3, k)));
    CHECK(append(*w, info(0, false, key_pts[k] + 50, seq0++), pattern(5, k)));
  }
  CHECK(w->finish().ok());
  auto r = open(path);
  CHECK(r != nullptr);
  CHECK(r->key_count(0) == 4 && r->key_count(1) == 4 && r->key_count(2) == 0);
  for (std::size_t n = 0; n < 4; ++n) {
    const vkc::Result<stream::Position> at = r->key(0, n);
    CHECK(at.ok());
    stream::Position pos = at.value();
    stream::Record record;
    const vkc::Result<bool> got = r->next(&pos, &record);
    CHECK(got.ok() && got.value());
    CHECK(record.info.track == 0 && record.info.key &&
          record.info.pts_ns == key_pts[n] && record.info.seq == 2 * n);
  }
  CHECK(r->key(0, 4).status().domain() == Code::NotFound);
  CHECK(r->key(2, 0).status().domain() == Code::InvalidArgument);

  const auto key_at_seq = [&](std::uint64_t pts) -> std::int64_t {
    const vkc::Result<stream::Position> at = r->key_at(0, pts);
    if (!at.ok()) return -1;
    stream::Position pos = at.value();
    stream::Record record;
    const vkc::Result<bool> got = r->next(&pos, &record);
    if (!got.ok() || !got.value() || !record.info.key) return -2;
    return static_cast<std::int64_t>(record.info.seq);
  };
  CHECK(key_at_seq(100) == 0);  // exact
  CHECK(key_at_seq(150) == 0);  // between
  CHECK(key_at_seq(200) == 4);  // the last of two at one pts
  CHECK(key_at_seq(299) == 4);
  CHECK(key_at_seq(300) == 6);
  CHECK(key_at_seq(~std::uint64_t{0}) == 6);  // after the last
  CHECK(r->key_at(0, 99).status().domain() == Code::NotFound);  // before
  CHECK(r->key_at(2, 100).status().domain() == Code::InvalidArgument);

  // Key pts that decrease, which this writer refuses to write.
  Bytes file =
      vrst::chunk("VRST", vrst::session_fields(0, 1, 0), vrst::session_body(1));
  for (const Bytes& c : {vrst::record(0, 1, 200, 0, 0, pattern(4, 1)),
                         vrst::record(0, 1, 100, 1, 1, pattern(4, 2))}) {
    file.insert(file.end(), c.begin(), c.end());
  }
  const std::string unordered = dir + "/unordered.vrst";
  CHECK(vrst::write_bytes(unordered, file));
  auto ru = open(unordered);
  CHECK(ru != nullptr);
  CHECK(ru->key_count(0) == 2 && ru->key(0, 1).ok());
  CHECK(ru->key_at(0, 300).status().domain() == Code::InvalidArgument);
  return 0;
}

// --- Unknown kinds, chunks and fields. ---------------------------------------

int unknown_case(const std::string& dir) {
  stream::Session s = one_track();
  stream::Track odd;
  odd.kind = static_cast<stream::TrackKind>(0x7777);
  odd.params = "whatever 0x7777 means";
  s.tracks.push_back(odd);
  const std::string path = dir + "/unknown.vrst";
  {
    auto w = create(path, s);
    CHECK(w != nullptr);
    CHECK(append(*w, info(1, true, 1, 0), pattern(20, 1)));
  }  // no footer: chunks follow
  Bytes file = vr_test::read_bytes(path);
  const Bytes zzzz = vrst::chunk("ZZZZ", Bytes(16, 0x5A), pattern(13, 2));
  const Bytes wide = vrst::record(0, 1, 2, 0, 9, pattern(77, 3), 8);  // H 64
  file.insert(file.end(), zzzz.begin(), zzzz.end());
  file.insert(file.end(), wide.begin(), wide.end());
  CHECK(vrst::write_bytes(path, file));

  auto r = open(path);
  CHECK(r != nullptr);
  CHECK(static_cast<std::uint16_t>(r->session().tracks[1].kind) == 0x7777);
  CHECK(same(r->session(), s));
  CHECK(r->recovery().damaged_spans == 0 && r->recovery().torn_bytes == 0);
  CHECK(r->stats()[0].records == 1 && r->stats()[1].records == 1);
  const vrst::ReadBack back = vrst::read_all(*r);
  CHECK(back.next_errors.empty() && back.read_errors.empty());
  CHECK(back.records.size() == 2);
  CHECK(same(back.records[0].info, info(1, true, 1, 0)));
  CHECK(back.payloads[0] == pattern(20, 1));
  CHECK(same(back.records[1].info, info(0, true, 2, 0, 9)));
  CHECK(back.payloads[1] == pattern(77, 3));
  return 0;
}

// --- Limits. -----------------------------------------------------------------

int limits_case(const std::string& dir) {
  const std::string path = dir + "/limits.vrst";
  {
    auto w = create(path, one_track());
    CHECK(w != nullptr);
    CHECK(append(*w, info(0, true, 0, 0), pattern(50, 1)));
    CHECK(append(*w, info(0, false, 1, 1), pattern(200, 2)));
    CHECK(append(*w, info(0, false, 2, 2), pattern(80, 3)));
  }
  stream::ReaderOptions small;
  small.max_payload_bytes = 100;
  auto r = open(path, small);
  CHECK(r != nullptr);
  const vrst::ReadBack back = vrst::read_all(*r);
  CHECK(back.records.size() == 3 && back.payloads.size() == 2);
  CHECK(back.read_errors.size() == 1 &&
        back.read_errors[0] == Code::OutOfMemory);
  CHECK(back.payloads[1] == pattern(80, 3));  // the one after it reads

  // A header claiming 2^64 - 8 body bytes: the scan stops there rather than
  // wrapping to a chunk before it.
  Bytes file = vr_test::read_bytes(path);
  const std::uint64_t at = file.size();
  const Bytes huge = vrst::header("RECD", vrst::record_fields(0, 0, 3, 3, 3),
                                  ~std::uint64_t{0} - 7, 0);
  file.insert(file.end(), huge.begin(), huge.end());
  const Bytes after = vrst::record(0, 0, 4, 4, 4, pattern(8, 4));
  file.insert(file.end(), after.begin(), after.end());
  CHECK(vrst::write_bytes(path, file));
  auto rh = open(path);
  CHECK(rh != nullptr);
  CHECK(rh->recovery().end == at &&
        rh->recovery().torn_bytes == file.size() - at);
  CHECK(rh->stats()[0].records == 3);
  const vrst::ReadBack back_h = vrst::read_all(*rh);
  CHECK(back_h.ended && back_h.records.size() == 3 &&
        back_h.next_errors.empty());
  stream::Record forged;
  forged.info = info(0, false, 3, 3, 3);
  forged.offset = at;
  forged.size = ~std::uint64_t{0} - 7;
  Bytes payload;
  CHECK(rh->read(forged, &payload).domain() == Code::OutOfMemory);
  forged.size = 8;  // the header says otherwise
  CHECK(rh->read(forged, &payload).domain() == Code::InvalidArgument);

  // A session header claiming a body above 64 MiB, in a (sparse) file that
  // holds that much, is refused by the limit, unread.
  const std::string big = dir + "/big_session.vrst";
  const Bytes session = vrst::header("VRST", vrst::session_fields(0, 1, 0),
                                     stream::kMaxSessionBytes + 8, 0);
  CHECK(vrst::write_bytes(big, session));
  CHECK(vrst::cut(big, session.size() + stream::kMaxSessionBytes + 8));
  const vkc::Status over = stream::StreamReader::open(big).status();
  CHECK(over.domain() == Code::InvalidArgument);
  CHECK(over.message().find("above " +
                            std::to_string(stream::kMaxSessionBytes)) !=
        std::string::npos);
  return 0;
}

// --- Writer refusals. --------------------------------------------------------

int writer_refusals_case(const std::string& dir) {
  const auto create_code = [](const std::string& path,
                              const stream::Session& s) {
    return stream::StreamWriter::create(path, s).status().domain();
  };
  const stream::Session good = one_track();
  CHECK(create_code("", good) == Code::InvalidArgument);
  CHECK(create_code(dir + std::string("/nul\0.vrst", 10), good) ==
        Code::InvalidArgument);

  // Something already there: a file, a directory, a dangling link.
  const std::string existing = dir + "/existing.vrst";
  CHECK(vrst::write_bytes(existing, {'k', 'e', 'e', 'p'}));
  CHECK(create_code(existing, good) == Code::InvalidArgument);
  CHECK(vr_test::read_bytes(existing) == Bytes({'k', 'e', 'e', 'p'}));
  const std::string directory = dir + "/directory";
  CHECK(::mkdir(directory.c_str(), 0755) == 0);
  CHECK(create_code(directory, good) == Code::InvalidArgument);
  const std::string link = dir + "/link.vrst";
  CHECK(::symlink((dir + "/nowhere").c_str(), link.c_str()) == 0);
  CHECK(create_code(link, good) == Code::InvalidArgument);
  CHECK(file_size(dir + "/nowhere") == ~std::uint64_t{0});

  CHECK(create_code(dir + "/missing/x.vrst", good) == Code::NotFound);

  // Sessions refused before anything is created.
  const std::string fresh = dir + "/refused.vrst";
  stream::Session s = good;
  s.tracks.clear();
  CHECK(create_code(fresh, s) == Code::InvalidArgument);
  s = good;
  s.tracks[0].kind = static_cast<stream::TrackKind>(0);
  CHECK(create_code(fresh, s) == Code::InvalidArgument);
  s = good;
  s.tracks[0].rate = {30, 0};
  CHECK(create_code(fresh, s) == Code::InvalidArgument);
  s = good;
  s.calibration.assign(stream::kMaxSessionBytes, 'c');
  CHECK(create_code(fresh, s) == Code::InvalidArgument);
  s = good;
  s.tracks.assign(65536, geometry());
  CHECK(create_code(fresh, s) == Code::InvalidArgument);
  CHECK(file_size(fresh) == ~std::uint64_t{0});
  s.tracks.resize(65535);  // the most
  CHECK(create_code(fresh, s) == Code::Ok);

  // Records refused, nothing written.
  const std::string path = dir + "/appends.vrst";
  stream::Session two = good;
  two.tracks.push_back(geometry());
  auto w = create(path, two);
  CHECK(w != nullptr);
  const Bytes p = pattern(9, 1);
  CHECK(append(*w, info(0, true, 100, 5), p));
  const std::uint64_t size = w->size();
  const auto refused = [&](const stream::RecordInfo& i, const std::uint8_t* d,
                           std::size_t n) {
    return w->append(i, d, n).domain() == Code::InvalidArgument &&
           w->size() == size && file_size(path) == size;
  };
  CHECK(refused(info(2, true, 100, 6), p.data(), p.size()));   // no track 2
  CHECK(refused(info(0, false, 100, 5), p.data(), p.size()));  // seq equal
  CHECK(refused(info(0, false, 100, 4), p.data(), p.size()));  // seq lower
  CHECK(refused(info(0, true, 99, 6), p.data(), p.size()));    // key pts back
  CHECK(refused(info(0, false, 100, 6), nullptr, 3));          // no payload
  // What the rules allow: non-key pts going back, a key at the last key's
  // pts, another track's own seq, and an empty payload with no pointer.
  CHECK(append(*w, info(0, false, 1, 6), p));
  CHECK(append(*w, info(0, true, 100, 7), p));
  CHECK(append(*w, info(1, true, 0, 0), p));
  CHECK(w->append(info(1, false, 0, 1), nullptr, 0).ok());
  CHECK(w->sync().ok());
  CHECK(w->finish().ok());
  const std::uint64_t finished = w->size();
  CHECK(file_size(path) == finished);
  CHECK(w->append(info(0, false, 1, 8), p.data(), p.size()).domain() ==
        Code::InvalidArgument);
  CHECK(w->sync().domain() == Code::InvalidArgument);
  CHECK(w->finish().domain() == Code::InvalidArgument);
  CHECK(file_size(path) == finished && w->size() == finished);
  return 0;
}

// --- Reader refusals. --------------------------------------------------------

int reader_refusals_case(const std::string& dir) {
  CHECK(open_code("") == Code::InvalidArgument);
  CHECK(open_code(dir + std::string("/a\0b", 4)) == Code::InvalidArgument);
  CHECK(open_code(dir + "/absent.vrst") == Code::NotFound);
  CHECK(open_code(dir + "/absent/x.vrst") == Code::NotFound);
  CHECK(open_code(dir) == Code::InvalidArgument);  // a directory

  const std::string path = dir + "/refused.vrst";
  const auto code = [&](const Bytes& bytes) {
    return vrst::write_bytes(path, bytes) ? open_code(path) : Code::IoError;
  };
  const Bytes body = vrst::session_body(1);
  const Bytes good = vrst::chunk("VRST", vrst::session_fields(0, 1, 0), body);
  CHECK(code(good) == Code::Ok);
  CHECK(code(Bytes(good.begin(), good.begin() + 55)) == Code::InvalidArgument);
  // Cut inside the session header's padding: no records, nothing torn.
  const auto pad = static_cast<std::size_t>(vrst::padding(body.size()));
  CHECK(pad != 0);
  for (std::size_t size = good.size() - pad; size < good.size(); ++size) {
    Bytes cut = good;
    cut.resize(size);
    CHECK(code(cut) == Code::Ok);
    auto rc = open(path);
    CHECK(rc != nullptr && rc->begin().offset == good.size());
    CHECK(rc->recovery().end == size && rc->recovery().torn_bytes == 0);
    const vrst::ReadBack none = vrst::read_all(*rc);
    CHECK(none.ended && none.records.empty() && none.next_errors.empty());
  }
  Bytes magic = good;
  magic[0] = 'X';
  CHECK(code(magic) == Code::InvalidArgument);
  CHECK(code(vrst::flipped(good, 30, 0)) == Code::InvalidArgument);  // CRC
  CHECK(code(vrst::chunk("VRST", vrst::session_fields(1, 1, 0), body)) ==
        Code::Unsupported);
  CHECK(code(vrst::chunk("VRST", vrst::session_fields(0, 1, 4), body)) ==
        Code::Unsupported);
  CHECK(code(vrst::chunk("VRST", Bytes(24, 0), body)) ==
        Code::InvalidArgument);  // H 48, below the session header's 56
  CHECK(code(vrst::flipped(good, 60, 3)) == Code::InvalidArgument);  // body
  Bytes extra = body;
  extra.push_back(0);
  CHECK(code(vrst::chunk("VRST", vrst::session_fields(0, 1, 0), extra)) ==
        Code::InvalidArgument);
  CHECK(code(vrst::chunk("VRST", vrst::session_fields(0, 2, 0), body)) ==
        Code::InvalidArgument);  // two tracks named, one there
  CHECK(code(vrst::chunk("VRST", vrst::session_fields(0, 0, 0), Bytes{})) ==
        Code::InvalidArgument);
  Bytes kind0 = body;
  kind0[0] = 0;
  CHECK(code(vrst::chunk("VRST", vrst::session_fields(0, 1, 0), kind0)) ==
        Code::InvalidArgument);

  // A record of a track the session lacks, and one setting an unknown flag:
  // each refused alone, the records around it read.
  Bytes file = good;
  for (const Bytes& c : {vrst::record(0, 1, 0, 0, 0, pattern(5, 1)),
                         vrst::record(1, 0, 1, 1, 1, pattern(5, 2)),
                         vrst::record(0, 2, 2, 2, 2, pattern(5, 3)),
                         vrst::record(0, 0, 3, 3, 3, pattern(5, 4))}) {
    file.insert(file.end(), c.begin(), c.end());
  }
  CHECK(vrst::write_bytes(path, file));
  auto r = open(path);
  CHECK(r != nullptr);
  CHECK(r->stats()[0].records == 2);
  CHECK(r->recovery().damaged_spans == 1);  // the record of track 1
  const vrst::ReadBack back = vrst::read_all(*r);
  CHECK(back.next_errors.size() == 2);
  CHECK(back.next_errors[0] == Code::InvalidArgument);
  CHECK(back.next_errors[1] == Code::Unsupported);
  CHECK(back.payloads.size() == 2 && back.payloads[0] == pattern(5, 1) &&
        back.payloads[1] == pattern(5, 4));

  stream::Position odd{r->begin().offset + 4};
  stream::Record record;
  CHECK(r->next(&odd, &record).status().domain() == Code::InvalidArgument);
  CHECK(odd.offset == r->begin().offset + 4);
  stream::Position early{0};
  CHECK(r->next(&early, &record).status().domain() == Code::InvalidArgument);
  // A record whose fields are not the file's, each changed alone.
  stream::Position pos = r->begin();
  CHECK(r->next(&pos, &record).ok());
  Bytes payload;
  CHECK(r->read(record, &payload).ok());
  using Change = void (*)(stream::Record&);
  const Change changes[] = {
      [](stream::Record& c) { c.info.track = 1; },
      [](stream::Record& c) { c.info.key = !c.info.key; },
      [](stream::Record& c) { c.info.pts_ns += 1; },
      [](stream::Record& c) { c.info.seq += 1; },
      [](stream::Record& c) { c.info.set += 1; },
      [](stream::Record& c) { c.size += 1; },
      [](stream::Record& c) { c.offset += 8; },
  };
  for (const Change change : changes) {
    stream::Record changed = record;
    change(changed);
    CHECK(r->read(changed, &payload).domain() == Code::InvalidArgument);
  }

  // A file cut short after the open: what is gone is IoError.
  const std::string shrunk = dir + "/shrunk.vrst";
  {
    auto w = create(shrunk, one_track());
    CHECK(w != nullptr);
    CHECK(append(*w, info(0, true, 0, 0), pattern(64, 1)));
    CHECK(append(*w, info(0, true, 1, 1), pattern(64, 2)));
  }
  auto rs = open(shrunk);
  CHECK(rs != nullptr);
  CHECK(vrst::cut(shrunk, rs->begin().offset + 80));  // in the payload
  pos = rs->begin();
  CHECK(rs->next(&pos, &record).ok());
  CHECK(rs->read(record, &payload).domain() == Code::IoError);
  CHECK(rs->next(&pos, &record).status().domain() == Code::IoError);
  return 0;
}

// --- Reads into caller memory.
// ------------------------------------------------

// The pointer read gives what the vector read gives, into memory the caller
// sized -- a slot with room after the payload, which stays as it was -- and a
// payload of no bytes into none. A capacity short by one, or no memory for a
// payload of any bytes, is refused before anything is read; a changed header,
// a payload failing its CRC and a file cut short after the open are refused
// as the vector read refuses them.
int caller_memory_case(const std::string& dir) {
  const std::string path = dir + "/caller.vrst";
  const std::vector<Bytes> payloads = {pattern(300, 1), Bytes{},
                                       pattern(17, 2)};
  {
    auto w = create(path, one_track());
    CHECK(w != nullptr);
    for (std::uint64_t i = 0; i < payloads.size(); ++i) {
      CHECK(append(*w, info(0, true, i, i), payloads[i]));
    }
    CHECK(w->finish().ok());
  }
  auto r = open(path);
  CHECK(r != nullptr);
  std::vector<stream::Record> records;
  stream::Position pos = r->begin();
  for (;;) {
    stream::Record record;
    const vkc::Result<bool> more = r->next(&pos, &record);
    CHECK(more.ok());
    if (!more.value()) break;
    records.push_back(record);
  }
  CHECK(records.size() == payloads.size());
  const auto untouched = [](const Bytes& b, std::size_t from) {
    return std::all_of(b.begin() + std::ptrdiff_t(from), b.end(),
                       [](std::uint8_t x) { return x == 0xEE; });
  };
  for (std::size_t i = 0; i < records.size(); ++i) {
    const auto size = static_cast<std::size_t>(records[i].size);
    Bytes slot(size + 8, 0xEE);
    CHECK(r->read(records[i], slot.data(), slot.size()).ok());
    CHECK(Bytes(slot.begin(), slot.begin() + std::ptrdiff_t(size)) ==
          payloads[i]);
    CHECK(untouched(slot, size));
    Bytes vector;
    CHECK(r->read(records[i], &vector).ok() && vector == payloads[i]);
  }
  CHECK(r->read(records[1], nullptr, 0).ok());
  Bytes slot(299, 0xEE);
  CHECK(r->read(records[0], slot.data(), slot.size()).domain() ==
        Code::InvalidArgument);
  CHECK(r->read(records[0], nullptr, 300).domain() == Code::InvalidArgument);
  CHECK(untouched(slot, 0));
  slot.assign(308, 0xEE);
  stream::Record changed = records[0];
  changed.info.seq += 1;
  CHECK(r->read(changed, slot.data(), slot.size()).domain() ==
        Code::InvalidArgument);

  // The first payload's fifth byte flipped.
  const std::string flip = dir + "/caller_flip.vrst";
  CHECK(vrst::write_bytes(flip, vrst::flipped(vr_test::read_bytes(path),
                                              records[0].offset + 60, 2)));
  auto rf = open(flip);
  CHECK(rf != nullptr);
  const vkc::Status crc = rf->read(records[0], slot.data(), slot.size());
  CHECK(crc.domain() == Code::InvalidArgument &&
        crc.message().find("CRC") != std::string::npos);
  CHECK(rf->read(records[2], slot.data(), slot.size()).ok());

  CHECK(vrst::cut(path, records[0].offset + 80));
  CHECK(r->read(records[0], slot.data(), slot.size()).domain() ==
        Code::IoError);
  return 0;
}

// --- Two threads, one reader. ------------------------------------------------

int concurrency_case(const std::string& dir) {
  const std::string path = dir + "/threads.vrst";
  std::vector<Bytes> payloads;
  {
    auto w = create(path, one_track());
    CHECK(w != nullptr);
    for (std::uint64_t i = 0; i < 64; ++i) {
      payloads.push_back(pattern(1 + (i * 7919) % 40000, i));
      CHECK(append(*w, info(0, i % 8 == 0, i, i), payloads.back()));
    }
    CHECK(w->finish().ok());
  }
  auto r = open(path);
  CHECK(r != nullptr);
  vrst::ReadBack backs[2];
  std::thread a([&] { backs[0] = vrst::read_all(*r); });
  std::thread b([&] { backs[1] = vrst::read_all(*r); });
  a.join();
  b.join();
  for (const vrst::ReadBack& back : backs) {
    CHECK(back.ended && back.payloads == payloads);
  }
  return 0;
}

// --- A track's nominal period. -----------------------------------------------

int nominal_period_case(const std::string&) {
  CHECK(stream::nominal_period_ns({25, 1}) == 40'000'000);
  CHECK(stream::nominal_period_ns({30000, 1001}) == 33'366'666);
  CHECK(stream::nominal_period_ns({1, 4'000'000'000u}) ==
        4'000'000'000'000'000'000ull);
  // Unknown, and a den no reader returns: 30 a second.
  CHECK(stream::nominal_period_ns({}) == 33'333'333);
  CHECK(stream::nominal_period_ns({30, 0}) == 33'333'333);
  return 0;
}

}  // namespace

int main() {
  vr_test::Scratch scratch("recon_stream_container_");
  const std::string dir = scratch.path.string();
  if (crc_case(dir) != 0) return 1;
  if (golden_case(dir) != 0) return 1;
  if (round_trip_case(dir) != 0) return 1;
  if (seek_case(dir) != 0) return 1;
  if (unknown_case(dir) != 0) return 1;
  if (limits_case(dir) != 0) return 1;
  if (writer_refusals_case(dir) != 0) return 1;
  if (reader_refusals_case(dir) != 0) return 1;
  if (caller_memory_case(dir) != 0) return 1;
  if (concurrency_case(dir) != 0) return 1;
  if (nominal_period_case(dir) != 0) return 1;
  std::puts("stream container: OK");
  return 0;
}
