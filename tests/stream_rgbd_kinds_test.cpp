// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// Host-only tests of the RGB-D kinds (TrackKind::RgbdDepth 3 and RgbdColor 4)
// and a payload in three runs (RecordPayload). A session whose track 0 holds
// pictures and track 1 depth, their params opaque bytes: its chunk against
// bytes made by hand from the layout; its records, each frame's colour record
// then its depth record in one set, the depth in three runs as a writer hands
// them, written and read back from the footer and by a scan, fed to a parser,
// and served to a client on loopback, all alike, the kinds included. A
// payload in three runs, at every split, is framed, written and served as the
// same bytes in one run are, under the CRC RecordPayload::of computed; a third
// run with no second, or a null third run, is the shorter payload.

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <initializer_list>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "volumetric_kit/core/testing/scratch_dir.hpp"
#include "volumetric_kit/core/testing/test_check.hpp"
#include "volumetric_kit/grid/stream/chunks.hpp"
#include "volumetric_kit/grid/stream/stream_reader.hpp"
#include "volumetric_kit/grid/stream/stream_writer.hpp"
#include "volumetric_kit/grid/testing/networking_loopback.hpp"
#include "volumetric_kit/grid/testing/stream_files.hpp"

namespace vkc = volumetric_kit::core;
namespace stream = volumetric_kit::grid::stream;
namespace vrst = vgrid_test::vrst;
namespace net = vgrid_test::net;

namespace {

using Code = vkc::Status::Code;
using vrst::Bytes;
using vrst::info;
using vrst::pattern;
using vrst::put;
using vrst::same;
using namespace std::chrono_literals;

// A depth record's prefix, rows and plane, and a colour record's prefix, at
// the sensor package's sizes: to grid, bytes like any others.
constexpr std::uint32_t kDepthWidth = 256;
constexpr std::uint32_t kDepthHeight = 192;
constexpr std::uint32_t kColorWidth = 1920;
constexpr std::uint32_t kColorHeight = 1440;
constexpr std::size_t kDepthPrefixBytes = 392;
constexpr std::size_t kRowsBytes = std::size_t{2} * kDepthWidth * kDepthHeight;
constexpr std::size_t kPlaneBytes = std::size_t{kDepthWidth} * kDepthHeight;
constexpr std::size_t kColorPrefixBytes = 128;

constexpr std::uint64_t kStart = 1762100000000000000ull;  // ns, host clock
constexpr std::uint64_t kPeriod = 100'000'000;            // 10 a second
constexpr std::uint64_t kFrames = 12;
constexpr std::uint64_t kKeyEvery = 6;  // colour keys at frames 0 and 6

std::string as_string(const Bytes& b) {
  return std::string(b.begin(), b.end());
}

Bytes joined(std::initializer_list<const Bytes*> runs) {
  Bytes out;
  for (const Bytes* r : runs) out.insert(out.end(), r->begin(), r->end());
  return out;
}

stream::Session rgbd_session() {
  stream::Session s;
  for (std::size_t i = 0; i < s.id.size(); ++i) {
    s.id[i] = static_cast<std::uint8_t>(0xA0 + i);
  }
  s.created_ns = 1762100000123456789ull;
  s.writer = "stream_rgbd_kinds_test";
  s.calibration = "{\"cameras\": []}";
  stream::Track color;
  color.kind = stream::TrackKind::RgbdColor;
  color.clock = stream::kHostClock;
  color.rate = {10, 1};
  color.width = kColorWidth;
  color.height = kColorHeight;
  color.sensor = "rgbd-0";
  color.params = as_string(pattern(4, 40));
  stream::Track depth = color;
  depth.kind = stream::TrackKind::RgbdDepth;
  depth.width = kDepthWidth;
  depth.height = kDepthHeight;
  // Opaque bytes, a zero among them, as long as a sensor's description.
  Bytes params = pattern(420, 41);
  params[5] = 0;
  depth.params = as_string(params);
  s.tracks = {color, depth};
  return s;
}

// `session`'s VRST chunk, built from session.hpp's layout, its kinds written
// as the numbers 4 and 3.
Bytes session_chunk_by_hand(const stream::Session& s) {
  Bytes fields;
  put(fields, 0, 4);  // version
  fields.insert(fields.end(), s.id.begin(), s.id.end());
  put(fields, s.created_ns, 8);
  put(fields, s.tracks.size(), 2);
  put(fields, 0, 2);  // flags
  const std::uint16_t kinds[] = {4, 3};
  Bytes body;
  for (std::size_t i = 0; i < s.tracks.size(); ++i) {
    const stream::Track& t = s.tracks[i];
    put(body, kinds[i], 2);
    put(body, t.clock, 2);
    put(body, t.rate.num, 4);
    put(body, t.rate.den, 4);
    put(body, t.width, 4);
    put(body, t.height, 4);
    put(body, 0, 4);  // reserved
  }
  const auto string = [&](const std::string& v) {
    put(body, v.size(), 4);
    body.insert(body.end(), v.begin(), v.end());
  };
  string(s.writer);
  string(s.calibration);
  for (const stream::Track& t : s.tracks) {
    string(t.sensor);
    string(t.params);
  }
  return vrst::chunk("VRST", fields, body);
}

// One frame's records as a writer holds them, each run its own buffer.
struct Frame {
  stream::RecordInfo color_info;
  Bytes color_prefix;
  Bytes picture;  // a coded picture, a key's larger
  stream::RecordInfo depth_info;
  Bytes depth_prefix;
  Bytes rows;
  Bytes plane;
};

std::vector<Frame> rgbd_frames() {
  std::vector<Frame> frames(kFrames);
  for (std::uint64_t n = 0; n < kFrames; ++n) {
    Frame& f = frames[n];
    const bool key = n % kKeyEvery == 0;
    const std::uint64_t pts = kStart + n * kPeriod;
    f.color_info = info(0, key, pts, n, n);
    f.color_prefix = pattern(kColorPrefixBytes, 100 + n);
    f.picture = pattern(key ? 30001 + n : 1500 + 13 * n, 200 + n);
    f.depth_info = info(1, true, pts, n, n);
    f.depth_prefix = pattern(kDepthPrefixBytes, 300 + n);
    f.rows = pattern(kRowsBytes, 400 + n);
    f.plane = pattern(kPlaneBytes, 500 + n);
  }
  return frames;
}

// A record as appended: its runs, borrowed from a Frame, and their bytes
// joined.
struct Appended {
  stream::RecordInfo info;
  stream::RecordPayload runs;
  Bytes payload;
};

// Each frame's colour record, then its depth record.
std::vector<Appended> rgbd_records(const std::vector<Frame>& frames) {
  std::vector<Appended> out;
  for (const Frame& f : frames) {
    out.push_back(
        {f.color_info,
         stream::RecordPayload::of(f.color_prefix.data(), f.color_prefix.size(),
                                   f.picture.data(), f.picture.size()),
         joined({&f.color_prefix, &f.picture})});
    out.push_back(
        {f.depth_info,
         stream::RecordPayload::of(f.depth_prefix.data(), f.depth_prefix.size(),
                                   f.rows.data(), f.rows.size(), f.plane.data(),
                                   f.plane.size()),
         joined({&f.depth_prefix, &f.rows, &f.plane})});
  }
  return out;
}

// Write `records` through their runs: finished, or as a killed writer
// leaves the file.
bool write(const std::string& path, const stream::Session& session,
           const std::vector<Appended>& records, bool finish) {
  vkc::Result<std::unique_ptr<stream::StreamWriter>> w =
      stream::StreamWriter::create(path, session);
  if (!w.ok()) return false;
  for (const Appended& a : records) {
    if (!(*w)->append(a.info, a.runs).ok()) return false;
  }
  return !finish || (*w)->finish().ok();
}

// --- The kinds. --------------------------------------------------------------

int session_case() {
  CHECK(static_cast<std::uint16_t>(stream::TrackKind::RgbdDepth) == 3);
  CHECK(static_cast<std::uint16_t>(stream::TrackKind::RgbdColor) == 4);
  const stream::Session s = rgbd_session();
  vkc::Result<stream::ChunkEncoder> encoder = stream::ChunkEncoder::create(s);
  CHECK(encoder.ok());
  CHECK(encoder->session_chunk() == session_chunk_by_hand(s));
  return 0;
}

// The records as `back` read them, in append order.
int same_records(const vrst::ReadBack& back,
                 const std::vector<Appended>& records) {
  CHECK(back.ended && back.next_errors.empty() && back.read_errors.empty());
  CHECK(back.records.size() == records.size());
  CHECK(back.payloads.size() == records.size());
  for (std::size_t i = 0; i < records.size(); ++i) {
    CHECK(same(back.records[i].info, records[i].info));
    CHECK(back.payloads[i] == records[i].payload);
  }
  return 0;
}

// A file of the two kinds reads back from its footer and by a scan, as a
// parser and a client on loopback read it, the kinds included.
int file_case(const std::string& dir) {
  const stream::Session s = rgbd_session();
  const std::vector<Frame> frames = rgbd_frames();
  const std::vector<Appended> records = rgbd_records(frames);
  CHECK(records[1].payload.size() == 147'848);  // 392 + 2 w h + w h
  CHECK(records[1].runs.size() == 147'848);
  CHECK(records[1].runs.crc() == vrst::crc32(records[1].payload));

  const std::string path = dir + "/rgbd.vrst";
  const std::string killed = dir + "/rgbd_killed.vrst";
  CHECK(write(path, s, records, true));
  CHECK(write(killed, s, records, false));

  std::vector<stream::TrackStats> expected(2);
  for (const Appended& a : records) {
    stream::TrackStats& e = expected[a.info.track];
    if (e.records == 0) e.min_pts_ns = a.info.pts_ns;
    e.max_pts_ns = a.info.pts_ns;
    ++e.records;
    e.keys += a.info.key ? 1 : 0;
    e.payload_bytes += a.payload.size();
  }
  CHECK(expected[0].keys == 2 && expected[1].keys == kFrames);
  CHECK(expected[1].payload_bytes == kFrames * 147'848);

  // From the footer and by a scan.
  auto footer = vrst::open(path);
  auto scan = vrst::open(killed);
  CHECK(footer != nullptr && footer->recovery().finished);
  CHECK(scan != nullptr && !scan->recovery().finished);
  CHECK(scan->recovery().torn_bytes == 0 &&
        scan->recovery().damaged_spans == 0);
  for (const stream::StreamReader* r : {footer.get(), scan.get()}) {
    CHECK(same(r->session(), s));
    CHECK(r->session().tracks[0].kind == stream::TrackKind::RgbdColor);
    CHECK(r->session().tracks[1].kind == stream::TrackKind::RgbdDepth);
    CHECK(r->stats().size() == 2);
    CHECK(same(r->stats()[0], expected[0]));
    CHECK(same(r->stats()[1], expected[1]));
    CHECK(r->key_count(0) == 2 && r->key_count(1) == kFrames);
    CHECK(same_records(vrst::read_all(*r), records) == 0);
  }
  const vrst::ReadBack file = vrst::read_all(*footer);

  // A parser fed the finished file in pieces: the same session and records,
  // at the same offsets, the footer skipped.
  const Bytes bytes = vkc::test::read_bytes(path);
  stream::ChunkParser parser;
  std::vector<net::Got> parsed;
  for (std::size_t at = 0; at < bytes.size(); at += 4093) {
    const std::size_t n = std::min<std::size_t>(4093, bytes.size() - at);
    CHECK(parser.feed(bytes.data() + at, n).ok());
    for (;;) {
      net::Got g;
      const vkc::Result<bool> got = parser.next(&g.record, &g.payload);
      CHECK(got.ok());
      if (!*got) break;
      parsed.push_back(std::move(g));
    }
  }
  CHECK(parser.session() != nullptr && same(*parser.session(), s));
  CHECK(parsed.size() == records.size());
  for (std::size_t i = 0; i < records.size(); ++i) {
    CHECK(same(parsed[i].record.info, records[i].info));
    CHECK(parsed[i].record.offset == file.records[i].offset);
    CHECK(parsed[i].payload == records[i].payload);
  }

  // A client on loopback: the same session, then every record the server
  // was handed in runs, at the file's offsets. Colour is track 0, whose keys
  // anchor the GOP; every depth record is a key.
  auto server = net::serve(s);
  CHECK(server != nullptr);
  auto client = net::connect(server->port());
  CHECK(client != nullptr && net::wait_session(*client));
  CHECK(same(*client->session(), s));
  CHECK(client->session()->tracks[1].kind == stream::TrackKind::RgbdDepth);
  for (std::size_t i = 0; i < records.size(); ++i) {
    CHECK(server->publish(records[i].info, records[i].runs).ok());
    net::Got g;
    CHECK(net::next_within(*client, &g));
    CHECK(same(g.record.info, records[i].info));
    CHECK(g.record.offset == file.records[i].offset);
    CHECK(g.payload == records[i].payload);
  }
  return 0;
}

// --- A payload in three runs. ------------------------------------------------

// A payload as a writer hands it, and its bytes joined.
struct Split {
  Bytes payload;
  stream::RecordPayload runs;
};

stream::Session depth_only_session() {
  stream::Session s = rgbd_session();
  s.tracks.erase(s.tracks.begin());
  return s;
}

// Each split's payload, a record of its own on a one-track session, every
// record a key: framed with the header its joined bytes get, written as the
// file of them joined writes, and served to a raw viewer as that file holds
// them, less its footer. Its CRC is RecordPayload::of's, the joined bytes'.
int alike_case(const std::string& dir, const std::string& name,
               const std::vector<Split>& splits) {
  const stream::Session s = depth_only_session();
  vkc::Result<stream::ChunkEncoder> joined_encoder =
      stream::ChunkEncoder::create(s);
  vkc::Result<stream::ChunkEncoder> runs_encoder =
      stream::ChunkEncoder::create(s);
  CHECK(joined_encoder.ok() && runs_encoder.ok());
  const std::string joined_path = dir + "/" + name + "_joined.vrst";
  const std::string runs_path = dir + "/" + name + "_runs.vrst";
  auto joined_writer = stream::StreamWriter::create(joined_path, s);
  auto runs_writer = stream::StreamWriter::create(runs_path, s);
  CHECK(joined_writer.ok() && runs_writer.ok());
  auto server = net::serve(s);
  CHECK(server != nullptr);
  const vkc::UniqueFd viewer = net::connect_loopback(server->port());
  CHECK(viewer.valid());
  Bytes wire;
  const std::size_t header = net::session_chunk(s).size();
  net::read_upto(viewer.get(), header, 5s, &wire);
  CHECK(wire.size() == header);

  std::uint64_t seq = 0;
  for (const Split& split : splits) {
    const stream::RecordInfo record = info(0, true, 1000 * seq, seq);
    ++seq;
    CHECK(split.runs.size() == split.payload.size());
    CHECK(split.runs.crc() == vrst::crc32(split.payload));
    const vkc::Result<stream::ChunkEncoder::RecordFraming> a =
        joined_encoder->frame(record, split.payload.data(),
                              split.payload.size());
    const vkc::Result<stream::ChunkEncoder::RecordFraming> b =
        runs_encoder->frame(record, split.runs);
    CHECK(a.ok() && b.ok());
    CHECK(a->header == b->header && a->padding == b->padding);
    CHECK((*joined_writer)
              ->append(record, split.payload.data(), split.payload.size())
              .ok());
    CHECK((*runs_writer)->append(record, split.runs).ok());
    CHECK(server->publish(record, split.runs).ok());
    const std::size_t want = (*runs_writer)->size();
    net::read_upto(viewer.get(), want, 5s, &wire);
    CHECK(wire.size() == want);
  }
  CHECK((*joined_writer)->finish().ok() && (*runs_writer)->finish().ok());
  const Bytes file = vkc::test::read_bytes(runs_path);
  CHECK(vkc::test::read_bytes(joined_path) == file);
  CHECK(vrst::index_offset(file) == wire.size());
  CHECK(Bytes(file.begin(),
              file.begin() + static_cast<std::ptrdiff_t>(wire.size())) == wire);
  return 0;
}

// Three runs at every pair of splits, empty runs included.
int three_runs_case(const std::string& dir) {
  const Bytes body = pattern(301, 7);
  const std::size_t cuts[] = {0, 1, 7, 150, 300, 301};
  std::vector<Split> splits;
  for (const std::size_t a : cuts) {
    for (const std::size_t b : cuts) {
      if (b < a) continue;
      const stream::RecordPayload p =
          stream::RecordPayload::of(body.data(), a, body.data() + a, b - a,
                                    body.data() + b, body.size() - b);
      CHECK(p.first() == body.data() && p.first_size() == a);
      CHECK(p.second() == body.data() + a && p.second_size() == b - a);
      CHECK(p.third() == body.data() + b && p.third_size() == body.size() - b);
      splits.push_back({body, p});
    }
  }
  CHECK(splits.size() == 21);
  CHECK(alike_case(dir, "three_runs", splits) == 0);

  // Bytes changed after `of`, in the third run, are written under the CRC of
  // what they held, which the reader refuses: the writer computed none.
  const std::string changed = dir + "/changed.vrst";
  {
    const Bytes head = pattern(40, 8);
    const Bytes rows = pattern(64, 9);
    Bytes plane = pattern(32, 10);
    const stream::RecordPayload p =
        stream::RecordPayload::of(head.data(), head.size(), rows.data(),
                                  rows.size(), plane.data(), plane.size());
    plane[3] ^= 0xFF;  // after `of`
    auto w = stream::StreamWriter::create(changed, depth_only_session());
    CHECK(w.ok());
    CHECK((*w)->append(info(0, true, 0, 0), p).ok());
    CHECK((*w)->finish().ok());
  }
  auto r = vrst::open(changed);
  CHECK(r != nullptr);
  stream::Position at = r->begin();
  stream::Record record;
  CHECK(r->next(&at, &record).value());
  Bytes payload;
  const vkc::Status crc = r->read(record, &payload);
  CHECK(crc.domain() == Code::InvalidArgument &&
        crc.message().find("CRC") != std::string::npos);
  return 0;
}

// A third run with no second, and a null third run, are the shorter
// payloads: their runs, their CRC, and the bytes framed, written and served.
int shorter_case(const std::string& dir) {
  const Bytes first = pattern(100, 11);
  const Bytes third = pattern(77, 12);
  const Bytes both = joined({&first, &third});

  const stream::RecordPayload no_second = stream::RecordPayload::of(
      first.data(), first.size(), nullptr, 0, third.data(), third.size());
  const stream::RecordPayload two = stream::RecordPayload::of(
      first.data(), first.size(), third.data(), third.size());
  CHECK(no_second.second() == nullptr && no_second.second_size() == 0);
  CHECK(no_second.third() == third.data() && no_second.size() == both.size());
  CHECK(no_second.crc() == two.crc() && two.crc() == vrst::crc32(both));

  const stream::RecordPayload no_third = stream::RecordPayload::of(
      first.data(), first.size(), third.data(), third.size(), nullptr, 0);
  CHECK(no_third.third() == nullptr && no_third.third_size() == 0);
  CHECK(two.third() == nullptr && two.third_size() == 0);
  CHECK(no_third.size() == two.size() && no_third.crc() == two.crc());

  const stream::RecordPayload third_alone = stream::RecordPayload::of(
      nullptr, 0, nullptr, 0, third.data(), third.size());
  CHECK(third_alone.first() == nullptr && third_alone.size() == third.size());
  CHECK(third_alone.crc() == vrst::crc32(third));

  const stream::RecordPayload none =
      stream::RecordPayload::of(nullptr, 0, nullptr, 0, nullptr, 0);
  CHECK(none.size() == 0 && none.crc() == 0 && none.third() == nullptr);

  CHECK(alike_case(dir, "shorter",
                   {{both, no_second},
                    {both, no_third},
                    {third, third_alone},
                    {Bytes{}, none}}) == 0);
  return 0;
}

}  // namespace

int main() {
  vkc::test::Scratch scratch("grid_stream_rgbd_kinds_");
  const std::string dir = scratch.path.string();
  if (session_case() != 0) return 1;
  if (file_case(dir) != 0) return 1;
  if (three_runs_case(dir) != 0) return 1;
  if (shorter_case(dir) != 0) return 1;
  std::puts("stream rgbd kinds: OK");
  return 0;
}
