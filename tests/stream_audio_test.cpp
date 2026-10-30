// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// Host-only tests of a .vrst Audio track (stream/audio.hpp): its params'
// golden bytes, read back from bytes made by hand too; round trips, trailing
// bytes ignored; each refusal, the length's and the codec's first; and a file
// of a Geometry track and an Audio track of a synthetic tone, written and read
// back: the session, every payload, each track's stats, seeking on the Audio
// track, the Geometry track's seeking as in a file without audio, and the
// footer and a scan agreeing.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "volumetric_kit/core/testing/test_check.hpp"
#include "volumetric_kit/grid/stream/audio.hpp"
#include "volumetric_kit/grid/stream/stream_reader.hpp"
#include "volumetric_kit/grid/stream/stream_writer.hpp"
#include "volumetric_kit/grid/testing/stream_files.hpp"

namespace vkc = volumetric_kit::core;
namespace stream = volumetric_kit::grid::stream;
namespace vrst = vgrid_test::vrst;

namespace {

using Code = vkc::Status::Code;
using vrst::Bytes;
using vrst::info;
using vrst::open;
using vrst::pattern;
using vrst::same;

std::string unhex(const char* hex) {
  std::string b;
  for (const char* p = hex; p[0] != '\0' && p[1] != '\0'; p += 2) {
    b.push_back(static_cast<char>(std::stoi(std::string(p, 2), nullptr, 16)));
  }
  return b;
}

stream::AudioParams params(std::uint16_t channels, std::uint32_t rate,
                           std::uint16_t codec = 1) {
  stream::AudioParams p;
  p.codec = static_cast<stream::AudioCodec>(codec);
  p.channels = channels;
  p.sample_rate = rate;
  return p;
}

bool same_params(const stream::AudioParams& a, const stream::AudioParams& b) {
  return a.codec == b.codec && a.channels == b.channels &&
         a.sample_rate == b.sample_rate;
}

Code decode_code(const std::string& bytes) {
  return stream::decode_audio_params(bytes).status().domain();
}

// --- The params. -------------------------------------------------------------

int golden_case() {
  struct Golden {
    stream::AudioParams params;
    const char* hex;
  };
  const Golden goldens[] = {
      {stream::AudioParams{}, "0100010080bb0000"},  // PcmS16, mono, 48 kHz
      {params(2, 44100), "0100020044ac0000"},
      {params(64, 96000), "0100400000770100"},
      {params(7, 0x89ABCDEFu), "01000700efcdab89"},
  };
  CHECK(stream::kAudioParamsBytes == 8 && stream::kMaxAudioChannels == 64);
  for (const Golden& g : goldens) {
    CHECK(stream::encode_audio_params(g.params) == unhex(g.hex));
    // Bytes made by hand, not by the encoder, read back.
    const vkc::Result<stream::AudioParams> back =
        stream::decode_audio_params(unhex(g.hex));
    CHECK(back.ok() && same_params(back.value(), g.params));
  }
  return 0;
}

int round_trip_case() {
  const std::uint16_t channel_counts[] = {1, 2, 6, 8, 63, 64};
  const std::uint32_t rates[] = {1, 8000, 44100, 48000, 192000, 0xFFFFFFFF};
  for (const std::uint16_t channels : channel_counts) {
    for (const std::uint32_t rate : rates) {
      const stream::AudioParams p = params(channels, rate);
      const std::string bytes = stream::encode_audio_params(p);
      CHECK(bytes.size() == stream::kAudioParamsBytes);
      const vkc::Result<stream::AudioParams> back =
          stream::decode_audio_params(bytes);
      CHECK(back.ok() && same_params(back.value(), p));
      // Bytes past the 8, a later codec's room, are ignored.
      const vkc::Result<stream::AudioParams> longer =
          stream::decode_audio_params(bytes + std::string("\x38\x01\0\0", 4));
      CHECK(longer.ok() && same_params(longer.value(), p));
    }
  }
  return 0;
}

int refusals_case() {
  const std::string good = stream::encode_audio_params({});
  for (std::size_t n = 0; n < good.size(); ++n) {
    CHECK(decode_code(good.substr(0, n)) == Code::InvalidArgument);
  }
  CHECK(decode_code(stream::encode_audio_params(params(0, 48000))) ==
        Code::InvalidArgument);
  CHECK(decode_code(stream::encode_audio_params(params(65, 48000))) ==
        Code::InvalidArgument);
  CHECK(decode_code(stream::encode_audio_params(params(0xFFFF, 48000))) ==
        Code::InvalidArgument);
  CHECK(decode_code(stream::encode_audio_params(params(1, 0))) ==
        Code::InvalidArgument);
  const vkc::Status channels =
      stream::decode_audio_params(stream::encode_audio_params(params(65, 1)))
          .status();
  CHECK(channels.message().find("65 channels") != std::string::npos);
  // An unknown codec, whatever else the bytes hold: a reader skips the
  // track.
  const std::uint16_t unknown[] = {0, 2, 0x7F00, 0xFFFF};
  for (const std::uint16_t codec : unknown) {
    CHECK(decode_code(stream::encode_audio_params(params(1, 48000, codec))) ==
          Code::Unsupported);
    CHECK(decode_code(stream::encode_audio_params(params(0, 0, codec))) ==
          Code::Unsupported);
  }
  // Too short is refused before the codec is read.
  CHECK(decode_code(std::string("\x02\0\x01\0\x80\xbb\0", 7)) ==
        Code::InvalidArgument);
  return 0;
}

// --- A file with an Audio track. ---------------------------------------------

constexpr std::uint64_t kStart = 1761800000000000000ull;  // ns, host clock
constexpr std::uint64_t kFramePeriod = 40'000'000;        // 25 a second
constexpr std::uint64_t kChunkPeriod = 20'000'000;        // 960 at 48 kHz
constexpr std::uint32_t kChunkFrames = 960;

// A 440 Hz tone at 48 kHz, mono, chunk `n` of it: s16 little-endian.
Bytes tone_chunk(std::uint64_t n) {
  constexpr double kTwoPi = 6.283185307179586;
  Bytes b;
  for (std::uint32_t i = 0; i < kChunkFrames; ++i) {
    const double t = static_cast<double>(n * kChunkFrames + i) / 48000.0;
    const auto s = static_cast<std::int16_t>(
        std::lround(8000.0 * std::sin(kTwoPi * 440.0 * t)));
    const auto u = static_cast<std::uint16_t>(s);
    b.push_back(static_cast<std::uint8_t>(u & 0xFF));
    b.push_back(static_cast<std::uint8_t>(u >> 8));
  }
  return b;
}

struct Appended {
  stream::RecordInfo info;
  Bytes payload;
};

// 10 geometry frames, keys at 0 and 5, and 20 audio chunks over the same
// 400 ms, in pts order, as a muxer appends them.
std::vector<Appended> two_track_records() {
  std::vector<Appended> out;
  std::uint64_t frame = 0;
  std::uint64_t chunk = 0;
  while (frame < 10 || chunk < 20) {
    const std::uint64_t frame_pts = kStart + frame * kFramePeriod;
    const std::uint64_t chunk_pts = kStart + chunk * kChunkPeriod;
    if (frame < 10 && (chunk == 20 || frame_pts <= chunk_pts)) {
      out.push_back(
          {info(0, frame % 5 == 0, frame_pts, frame, frame),
           pattern(static_cast<std::size_t>(100 + 37 * frame), frame)});
      ++frame;
    } else {
      out.push_back({info(1, true, chunk_pts, chunk), tone_chunk(chunk)});
      ++chunk;
    }
  }
  return out;
}

stream::Session two_track_session() {
  stream::Track geometry;
  geometry.kind = stream::TrackKind::Geometry;
  geometry.clock = stream::kHostClock;
  geometry.rate = {25, 1};
  stream::Track audio;
  audio.kind = stream::TrackKind::Audio;
  audio.clock = stream::kHostClock;
  audio.rate = {48000, kChunkFrames};
  audio.sensor = "usb-mic-0";
  audio.params = stream::encode_audio_params({});
  stream::Session s;
  s.writer = "stream_audio_test";
  s.tracks = {geometry, audio};
  return s;
}

// Write `records` of `session`'s tracks to `path`: finished, or left as a
// killed writer leaves it, for a reader to scan.
bool write(const std::string& path, const stream::Session& session,
           const std::vector<Appended>& records, bool finish) {
  vkc::Result<std::unique_ptr<stream::StreamWriter>> w =
      stream::StreamWriter::create(path, session);
  if (!w.ok()) return false;
  for (const Appended& a : records) {
    if (a.info.track >= session.tracks.size()) continue;
    if (!w.value()->append(a.info, a.payload.data(), a.payload.size()).ok()) {
      return false;
    }
  }
  return !finish || w.value()->finish().ok();
}

// The seq of the record at `track`'s last key at or before `pts`, or -1.
std::int64_t key_at_seq(const stream::StreamReader& r, std::uint16_t track,
                        std::uint64_t pts) {
  const vkc::Result<stream::Position> at = r.key_at(track, pts);
  if (!at.ok()) return -1;
  stream::Position pos = at.value();
  stream::Record record;
  const vkc::Result<bool> got = r.next(&pos, &record);
  if (!got.ok() || !got.value() || record.info.track != track ||
      !record.info.key) {
    return -2;
  }
  return static_cast<std::int64_t>(record.info.seq);
}

int file_case(const std::string& dir) {
  const stream::Session s = two_track_session();
  const std::vector<Appended> records = two_track_records();
  const std::string path = dir + "/audio.vrst";
  CHECK(write(path, s, records, true));

  std::vector<stream::TrackStats> expected(2);
  for (const Appended& a : records) {
    stream::TrackStats& e = expected[a.info.track];
    if (e.records == 0) e.min_pts_ns = a.info.pts_ns;
    e.max_pts_ns = a.info.pts_ns;
    ++e.records;
    e.keys += a.info.key ? 1 : 0;
    e.payload_bytes += a.payload.size();
  }
  CHECK(expected[1].records == 20 && expected[1].keys == 20);
  CHECK(expected[1].payload_bytes == 20 * 1920);

  auto r = open(path);
  CHECK(r != nullptr && r->recovery().finished);
  CHECK(same(r->session(), s));
  CHECK(r->session().tracks[1].kind == stream::TrackKind::Audio);
  const vkc::Result<stream::AudioParams> p =
      stream::decode_audio_params(r->session().tracks[1].params);
  CHECK(p.ok() && same_params(p.value(), stream::AudioParams{}));
  CHECK(r->stats().size() == 2);
  CHECK(same(r->stats()[0], expected[0]) && same(r->stats()[1], expected[1]));
  CHECK(r->key_count(0) == 2 && r->key_count(1) == 20);

  const vrst::ReadBack back = vrst::read_all(*r);
  CHECK(back.ended && back.next_errors.empty() && back.read_errors.empty());
  CHECK(back.records.size() == records.size());
  for (std::size_t i = 0; i < records.size(); ++i) {
    CHECK(same(back.records[i].info, records[i].info));
    CHECK(back.payloads[i] == records[i].payload);
  }
  // Chunk 1's first sample, 440 Hz at 960 / 48000 s: 8000 sin(17.6 pi),
  // -7608 (0xE248), little-endian.
  CHECK(records[2].info.track == 1 && records[2].info.seq == 1);
  CHECK(back.payloads[2][0] == 0x48 && back.payloads[2][1] == 0xE2);

  // Seeking on the Audio track: every record a key.
  CHECK(key_at_seq(*r, 1, kStart) == 0);
  CHECK(key_at_seq(*r, 1, kStart + 2 * kChunkPeriod + 1) == 2);
  CHECK(key_at_seq(*r, 1, kStart + 19 * kChunkPeriod) == 19);
  CHECK(key_at_seq(*r, 1, ~std::uint64_t{0}) == 19);
  CHECK(r->key_at(1, kStart - 1).status().domain() == Code::NotFound);

  // The Geometry track seeks as in a file of its records alone.
  const std::string alone = dir + "/geometry_alone.vrst";
  stream::Session geometry_only = s;
  geometry_only.tracks.resize(1);
  CHECK(write(alone, geometry_only, records, true));
  auto ra = open(alone);
  CHECK(ra != nullptr && ra->key_count(0) == 2);
  CHECK(same(ra->stats()[0], r->stats()[0]));
  for (std::uint64_t pts = kStart - kChunkPeriod;
       pts <= kStart + 12 * kFramePeriod; pts += kChunkPeriod / 4) {
    CHECK(key_at_seq(*r, 0, pts) == key_at_seq(*ra, 0, pts));
  }
  CHECK(key_at_seq(*r, 0, kStart + 4 * kFramePeriod) == 0);
  CHECK(key_at_seq(*r, 0, kStart + 5 * kFramePeriod) == 5);

  // The footer and a scan agree: the same records, unfinished.
  const std::string scanned = dir + "/audio_scanned.vrst";
  CHECK(write(scanned, s, records, false));
  auto rs = open(scanned);
  CHECK(rs != nullptr && !rs->recovery().finished);
  CHECK(rs->recovery().torn_bytes == 0 && rs->recovery().damaged_spans == 0);
  for (std::uint16_t t = 0; t < 2; ++t) {
    CHECK(same(rs->stats()[t], r->stats()[t]));
    CHECK(rs->key_count(t) == r->key_count(t));
    for (std::size_t n = 0; n < r->key_count(t); ++n) {
      const vkc::Result<stream::Position> a = r->key(t, n);
      const vkc::Result<stream::Position> b = rs->key(t, n);
      CHECK(a.ok() && b.ok() && a.value().offset == b.value().offset);
    }
  }
  return 0;
}

}  // namespace

int main() {
  vkc::test::Scratch scratch("grid_stream_audio_");
  const std::string dir = scratch.path.string();
  if (golden_case() != 0) return 1;
  if (round_trip_case() != 0) return 1;
  if (refusals_case() != 0) return 1;
  if (file_case(dir) != 0) return 1;
  std::puts("stream audio: OK");
  return 0;
}
