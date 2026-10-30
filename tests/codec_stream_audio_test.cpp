// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// A codec stream's Audio tracks (codec_stream/codec_stream.hpp, append), on a
// grid holding a sphere, each record a chunk of a synthetic tone: sessions
// refused on create, naming the track, and one served refused with the
// server's Unsupported, no file made; every refusal of append, nothing
// changed; records held until a frame at or past their pts, equal pts
// included, then right after it, from a frame in halves too, in pts order
// across two tracks, the lower track first at equal pts, and one at or below
// the last frame's out at once; dropped below the first frame, before it and
// after; the 1 s bound on what is held, before the first frame and while the
// geometry stalls; finish writing what is held, or dropping it before any
// frame; and each file read back: frames and records in that order, every
// payload intact, each track's stats. The device part skips with no device.

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "volumetric_kit/core/testing/gpu_test.hpp"
#include "volumetric_kit/core/testing/scratch_dir.hpp"
#include "volumetric_kit/core/vulkan/command_batch.hpp"
#include "volumetric_kit/grid/base/device_requirements.hpp"
#include "volumetric_kit/grid/codec_stream/codec_stream.hpp"
#include "volumetric_kit/grid/stream/audio.hpp"
#include "volumetric_kit/grid/stream/stream_reader.hpp"
#include "volumetric_kit/grid/testing/codec_fixture.hpp"
#include "volumetric_kit/grid/testing/stream_files.hpp"

namespace vgrid = volumetric_kit::grid;
namespace vkc = volumetric_kit::core;
namespace codec = volumetric_kit::grid::codec;
namespace codec_stream = volumetric_kit::grid::codec_stream;
namespace stream = volumetric_kit::grid::stream;
namespace vrst = vgrid_test::vrst;
namespace fs = std::filesystem;
using namespace codec_fixture;

namespace {

using Code = vkc::Status::Code;

constexpr std::uint64_t kMs = 1'000'000;
constexpr std::uint64_t kT0 = 1761800000000000000ull;  // a host clock's
constexpr std::uint32_t kChunkFrames = 960;            // 20 ms at 48 kHz

// Chunk `seq` of track `track`'s tone, mono s16 little-endian at 48 kHz:
// 440 Hz on track 1, 660 Hz on track 2.
std::vector<std::uint8_t> tone(std::uint16_t track, std::uint64_t seq) {
  constexpr double kTwoPi = 6.283185307179586;
  const double hz = 220.0 * (track + 1);
  std::vector<std::uint8_t> b;
  for (std::uint32_t i = 0; i < kChunkFrames; ++i) {
    const double t = static_cast<double>(seq * kChunkFrames + i) / 48000.0;
    const auto u = static_cast<std::uint16_t>(static_cast<std::int16_t>(
        std::lround(8000.0 * std::sin(kTwoPi * hz * t))));
    b.push_back(static_cast<std::uint8_t>(u & 0xFF));
    b.push_back(static_cast<std::uint8_t>(u >> 8));
  }
  return b;
}

// A Geometry track and `audio` Audio tracks, all on the host clock.
stream::Session session(std::size_t audio) {
  stream::Session s;
  s.writer = "grid_codec_stream_audio";
  stream::Track geometry;
  geometry.kind = stream::TrackKind::Geometry;
  geometry.clock = stream::kHostClock;
  geometry.rate = {25, 1};
  s.tracks.push_back(geometry);
  for (std::size_t i = 0; i < audio; ++i) {
    stream::Track track;
    track.kind = stream::TrackKind::Audio;
    track.clock = stream::kHostClock;
    track.rate = {48000, kChunkFrames};
    track.sensor = "mic " + std::to_string(i);
    track.params = stream::encode_audio_params({});
    s.tracks.push_back(track);
  }
  return s;
}

// A stream of `audio` Audio tracks writing `path`, without the decode.
vkc::Result<codec_stream::CodecStream> make(vkc::test::GpuContext& gpu,
                                            std::size_t audio,
                                            const fs::path& path) {
  codec_stream::CodecOutputs outputs;
  outputs.stream = path.string();
  outputs.decode = false;
  return codec_stream::CodecStream::create(gpu.device, gpu.allocator, {},
                                           session(audio), outputs);
}

// Append track `track`'s chunk `seq` at `pts_ns`, its tone.
vkc::Status append(codec_stream::CodecStream& coder, std::uint16_t track,
                   std::uint64_t seq, std::uint64_t pts_ns) {
  const std::vector<std::uint8_t> chunk = tone(track, seq);
  return coder.append(vrst::info(track, true, pts_ns, seq), chunk.data(),
                      chunk.size());
}

bool counts_are(const codec_stream::CodecStream& coder, std::uint16_t track,
                std::uint64_t held, std::uint64_t dropped,
                std::uint64_t written) {
  const codec_stream::AppendedRecords c = coder.appended(track);
  if (c.held == held && c.dropped == dropped && c.written == written) {
    return true;
  }
  std::fprintf(stderr, "track %u: held %llu dropped %llu written %llu\n",
               unsigned(track), static_cast<unsigned long long>(c.held),
               static_cast<unsigned long long>(c.dropped),
               static_cast<unsigned long long>(c.written));
  return false;
}

// A record as the file holds it: frames by set, records by seq.
struct Entry {
  std::uint16_t track = 0;
  std::uint64_t id = 0;  // a frame's set, a record's seq
  std::uint64_t pts_ns = 0;
};

// The records at `path` so far (a stream not finished reads as a killed
// writer's), each Audio payload its tone, every record a key on an Audio
// track; empty, the reason on stderr, otherwise.
std::vector<Entry> read_back(const std::string& path) {
  const std::unique_ptr<stream::StreamReader> reader = vrst::open(path);
  if (!reader) return {};
  const vrst::ReadBack back = vrst::read_all(*reader);
  if (!back.ended || !back.next_errors.empty() || !back.read_errors.empty()) {
    std::fprintf(stderr, "read_back %s: not read whole\n", path.c_str());
    return {};
  }
  std::vector<Entry> entries;
  for (std::size_t i = 0; i < back.records.size(); ++i) {
    const stream::RecordInfo& info = back.records[i].info;
    const bool frame = info.track == 0;
    if (!frame &&
        (!info.key || back.payloads[i] != tone(info.track, info.seq) ||
         info.set != stream::kNoSet)) {
      std::fprintf(stderr, "read_back: track %u seq %llu not as appended\n",
                   unsigned(info.track),
                   static_cast<unsigned long long>(info.seq));
      return {};
    }
    entries.push_back({info.track, frame ? info.set : info.seq, info.pts_ns});
  }
  return entries;
}

bool same(const std::vector<Entry>& got, const std::vector<Entry>& expected) {
  bool equal = got.size() == expected.size();
  for (std::size_t i = 0; equal && i < got.size(); ++i) {
    equal = got[i].track == expected[i].track && got[i].id == expected[i].id &&
            got[i].pts_ns == expected[i].pts_ns;
  }
  if (!equal) {
    std::fprintf(stderr, "file order (track:id):");
    for (const Entry& e : got) {
      std::fprintf(stderr, " %u:%llu", unsigned(e.track),
                   static_cast<unsigned long long>(e.id));
    }
    std::fprintf(stderr, "\n");
  }
  return equal;
}

// Sessions the stream cannot mux are refused before any output is made,
// naming the track: a second Geometry track, another kind, params no reader
// plays (none, no channel, a codec it does not know), another clock than
// the frames'. Served, a session with an Audio track is the server's
// Unsupported, as it serves one track, with no file made.
int refused_sessions(vkc::test::GpuContext& gpu,
                     const vkc::test::Scratch& scratch) {
  codec_stream::CodecOutputs outputs;
  outputs.stream = (scratch.path / "refused.vrst").string();
  const auto refused = [&](const stream::Session& s, Code code,
                           const std::string& named) {
    const vkc::Result<codec_stream::CodecStream> made =
        codec_stream::CodecStream::create(gpu.device, gpu.allocator, {}, s,
                                          outputs);
    const bool ok = !made.ok() && made.status().domain() == code &&
                    made.status().message().find(named) != std::string::npos;
    if (!ok) {
      std::fprintf(stderr, "refused: %s\n",
                   made.ok() ? "made" : made.status().message().c_str());
    }
    return ok && !fs::exists(outputs.stream);
  };
  stream::Session two_geometry = session(1);
  two_geometry.tracks[1] = two_geometry.tracks[0];
  stream::Session other_kind = session(1);
  other_kind.tracks[1].kind = static_cast<stream::TrackKind>(7);
  stream::Session no_params = session(2);
  no_params.tracks[2].params.clear();
  stream::AudioParams silent;
  silent.channels = 0;
  stream::Session no_channel = session(1);
  no_channel.tracks[1].params = stream::encode_audio_params(silent);
  stream::AudioParams unknown;
  unknown.codec = static_cast<stream::AudioCodec>(9);
  stream::Session unknown_codec = session(1);
  unknown_codec.tracks[1].params = stream::encode_audio_params(unknown);
  stream::Session other_clock = session(1);
  other_clock.tracks[1].clock = 3;
  CHECK(refused(two_geometry, Code::InvalidArgument, "track 1 is of kind 1"));
  CHECK(refused(other_kind, Code::InvalidArgument, "track 1 is of kind 7"));
  CHECK(refused(no_params, Code::InvalidArgument, "track 2: "));
  CHECK(refused(no_channel, Code::InvalidArgument, "track 1: "));
  CHECK(refused(unknown_codec, Code::InvalidArgument, "track 1: "));
  CHECK(refused(other_clock, Code::InvalidArgument,
                "track 1 is on clock 3, track 0 on clock 0"));

  outputs.serve.emplace();
  outputs.serve->host = "127.0.0.1";
  outputs.serve->port = 0;
  CHECK(refused(session(1), Code::Unsupported, "StreamServer::start"));
  return 0;
}

// Each refusal of append changes nothing, the counts and the file as they
// were: track 0, a track the session lacks, a record that is not a key, a
// seq not above the track's last, a pts below it, a null payload with bytes,
// and any record once finished or moved from. A seq above the last at the
// same pts, and an empty payload, are taken. No count on track 0, a track
// the session lacks, or a moved-from stream.
int refused_records(vkc::test::GpuContext& gpu, vol::VoxelBlockGrid& source,
                    const vkc::test::Scratch& scratch) {
  const fs::path path = scratch.path / "refused_records.vrst";
  vkc::Result<codec_stream::CodecStream> made = make(gpu, 1, path);
  CHECK(made.ok());
  codec_stream::CodecStream& coder = made.value();
  CHECK(coder.code(source, 0, kT0).ok());
  CHECK(append(coder, 1, 5, kT0 + 10 * kMs).ok());
  CHECK(counts_are(coder, 1, 1, 0, 0));
  const std::uintmax_t size = fs::file_size(path);
  const std::vector<std::uint8_t> chunk = tone(1, 6);
  const auto refused = [&](const stream::RecordInfo& info,
                           const std::uint8_t* data, const std::string& named) {
    const vkc::Status s = coder.append(info, data, chunk.size());
    const bool ok = s.domain() == Code::InvalidArgument &&
                    s.message().find(named) != std::string::npos;
    if (!ok) std::fprintf(stderr, "append: %s\n", s.message().c_str());
    return ok && counts_are(coder, 1, 1, 0, 0) && fs::file_size(path) == size;
  };
  const std::uint64_t pts = kT0 + 10 * kMs;
  CHECK(refused(vrst::info(0, true, pts, 6), chunk.data(), "track 0 is"));
  CHECK(refused(vrst::info(2, true, pts, 6), chunk.data(), "track 2 is not"));
  CHECK(refused(vrst::info(1, false, pts, 6), chunk.data(), "not a key"));
  CHECK(refused(vrst::info(1, true, pts, 5), chunk.data(), "seq 5 does not"));
  CHECK(refused(vrst::info(1, true, pts, 4), chunk.data(), "seq 4 does not"));
  CHECK(refused(vrst::info(1, true, pts - 1, 6), chunk.data(), "goes back"));
  CHECK(refused(vrst::info(1, true, pts, 6), nullptr, "null payload"));
  CHECK(append(coder, 1, 6, pts).ok());
  CHECK(coder.append(vrst::info(1, true, pts, 7), nullptr, 0).ok());
  CHECK(counts_are(coder, 1, 3, 0, 0));
  CHECK(coder.finish().ok() && counts_are(coder, 1, 0, 0, 3));
  CHECK(append(coder, 1, 8, pts).domain() == Code::InvalidArgument);
  CHECK(counts_are(coder, 1, 0, 0, 3));
  const codec_stream::AppendedRecords none = coder.appended(0);
  const codec_stream::AppendedRecords absent = coder.appended(2);
  CHECK(none.held + none.dropped + none.written == 0);
  CHECK(absent.held + absent.dropped + absent.written == 0);

  vkc::Result<codec_stream::CodecStream> first =
      make(gpu, 1, scratch.path / "moved.vrst");
  CHECK(first.ok() && first->code(source, 0, kT0).ok());
  CHECK(append(*first, 1, 0, kT0 + kMs).ok());
  codec_stream::CodecStream moved = std::move(first).value();
  CHECK(append(*first, 1, 1, kT0 + 2 * kMs).domain() == Code::InvalidArgument);
  const codec_stream::AppendedRecords gone = first->appended(1);
  CHECK(gone.held + gone.dropped + gone.written == 0);
  CHECK(counts_are(moved, 1, 1, 0, 0));
  CHECK(append(moved, 1, 1, kT0 + 2 * kMs).ok() && moved.finish().ok());
  CHECK(counts_are(moved, 1, 0, 0, 2));
  return 0;
}

// Two Audio tracks muxed with three frames, the second coded in halves. Held
// before the first frame, which drops the one below its pts and releases the
// one at it; one on track 2 below the first frame dropped at once; frame 1
// releasing the two at or below its pts, and one at its pts appended after
// it going out at once; five across both tracks held for frame 2, which
// releases four in pts order, track 1 first at equal pts; finish writing the
// last. The file: each record right after its frame, the records' pts never
// going back, each payload its tone, each track's stats.
int muxed(vkc::test::GpuContext& gpu, vol::VoxelBlockGrid& source,
          const vkc::test::Scratch& scratch) {
  const fs::path path = scratch.path / "muxed.vrst";
  vkc::Result<codec_stream::CodecStream> made = make(gpu, 2, path);
  CHECK(made.ok());
  codec_stream::CodecStream& coder = made.value();
  CHECK(append(coder, 1, 0, kT0 - 20 * kMs).ok());
  CHECK(append(coder, 1, 1, kT0).ok());
  CHECK(append(coder, 1, 2, kT0 + 20 * kMs).ok());
  CHECK(append(coder, 1, 3, kT0 + 40 * kMs).ok());
  CHECK(counts_are(coder, 1, 4, 0, 0) && read_back(path.string()).empty());

  CHECK(coder.code(source, 0, kT0).ok());
  CHECK(counts_are(coder, 1, 2, 1, 1));
  CHECK(same(read_back(path.string()), {{0, 0, kT0}, {1, 1, kT0}}));
  CHECK(append(coder, 2, 0, kT0 - 5 * kMs).ok());
  CHECK(counts_are(coder, 2, 0, 1, 0));

  {  // Frame 1 in halves.
    vkc::CommandBatch front(gpu.device, gpu.allocator);
    const vkc::Result<codec::PendingEncode> pending =
        coder.record_front(front, source);
    CHECK(pending.ok());
    vkc::Result<vkc::PendingBatch> done = front.submit_async({}, {});
    CHECK(done.ok() && done->wait().ok());
    coder.front_ran(done->device_rows());
    CHECK(coder.finish_frame(*pending, 1, kT0 + 40 * kMs).ok());
  }
  CHECK(counts_are(coder, 1, 0, 1, 3));
  CHECK(append(coder, 2, 1, kT0 + 40 * kMs).ok());
  CHECK(counts_are(coder, 2, 0, 1, 1));

  CHECK(append(coder, 1, 4, kT0 + 60 * kMs).ok());
  CHECK(append(coder, 2, 2, kT0 + 50 * kMs).ok());
  CHECK(append(coder, 2, 3, kT0 + 80 * kMs).ok());
  CHECK(append(coder, 1, 5, kT0 + 80 * kMs).ok());
  CHECK(append(coder, 2, 4, kT0 + 90 * kMs).ok());
  CHECK(counts_are(coder, 1, 2, 1, 3) && counts_are(coder, 2, 3, 1, 1));
  CHECK(coder.code(source, 2, kT0 + 80 * kMs).ok());
  CHECK(counts_are(coder, 1, 0, 1, 5) && counts_are(coder, 2, 1, 1, 3));
  CHECK(coder.finish().ok());
  CHECK(counts_are(coder, 1, 0, 1, 5) && counts_are(coder, 2, 0, 1, 4));

  const std::vector<Entry> got = read_back(path.string());
  CHECK(same(got, {{0, 0, kT0},
                   {1, 1, kT0},
                   {0, 1, kT0 + 40 * kMs},
                   {1, 2, kT0 + 20 * kMs},
                   {1, 3, kT0 + 40 * kMs},
                   {2, 1, kT0 + 40 * kMs},
                   {0, 2, kT0 + 80 * kMs},
                   {2, 2, kT0 + 50 * kMs},
                   {1, 4, kT0 + 60 * kMs},
                   {1, 5, kT0 + 80 * kMs},
                   {2, 3, kT0 + 80 * kMs},
                   {2, 4, kT0 + 90 * kMs}}));
  // Across tracks, a record's pts never below the one before it, and never
  // above the frame before it but for the last, which finish wrote.
  std::uint64_t last = 0;
  std::uint64_t frame = 0;
  for (std::size_t i = 0; i < got.size(); ++i) {
    if (got[i].track == 0) {
      frame = got[i].pts_ns;
      continue;
    }
    CHECK(got[i].pts_ns >= last &&
          (got[i].pts_ns <= frame || i + 1 == got.size()));
    last = got[i].pts_ns;
  }
  const std::unique_ptr<stream::StreamReader> reader =
      vrst::open(path.string());
  CHECK(reader != nullptr && reader->recovery().finished);
  const std::uint64_t chunk = 2 * kChunkFrames;
  const stream::TrackStats& one = reader->stats()[1];
  const stream::TrackStats& two = reader->stats()[2];
  CHECK(reader->stats()[0].records == 3);
  CHECK(one.records == 5 && one.keys == 5 && one.payload_bytes == 5 * chunk);
  CHECK(one.min_pts_ns == kT0 && one.max_pts_ns == kT0 + 80 * kMs);
  CHECK(two.records == 4 && two.keys == 4 && two.payload_bytes == 4 * chunk);
  CHECK(two.min_pts_ns == kT0 + 40 * kMs && two.max_pts_ns == kT0 + 90 * kMs);
  return 0;
}

// What is held stays within 1 s: before the first frame, a record more than
// 1 s below the newest is dropped, one exactly 1 s below kept; while the
// geometry stalls, records up to exactly 1 s past the last frame are held,
// and the first further past sends everything held, in pts order, then
// itself, and each after it goes out at once until a frame catches up.
int bounded(vkc::test::GpuContext& gpu, vol::VoxelBlockGrid& source,
            const vkc::test::Scratch& scratch) {
  const fs::path path = scratch.path / "bounded.vrst";
  vkc::Result<codec_stream::CodecStream> made = make(gpu, 1, path);
  CHECK(made.ok());
  codec_stream::CodecStream& coder = made.value();
  // Chunk k at kT0 - 1180 ms + 20 k ms: 59 at kT0, 9 exactly 1 s below it.
  const auto at = [](std::uint64_t k) {
    return kT0 - 1180 * kMs + 20 * kMs * k;
  };
  for (std::uint64_t k = 0; k < 60; ++k) CHECK(append(coder, 1, k, at(k)).ok());
  CHECK(counts_are(coder, 1, 51, 9, 0));
  // The first frame at chunk 44's pts: 44 goes out; 45-59 are held, 300 ms
  // past it at most.
  const std::uint64_t first = at(44);
  CHECK(coder.code(source, 0, first).ok());
  CHECK(counts_are(coder, 1, 15, 44, 1));
  // The geometry stalls: 60-94 held, 94 exactly 1 s past the frame.
  for (std::uint64_t k = 60; k < 95; ++k)
    CHECK(append(coder, 1, k, at(k)).ok());
  CHECK(at(94) == first + codec_stream::kHoldLimitNs);
  CHECK(counts_are(coder, 1, 50, 44, 1) &&
        read_back(path.string()).size() == 2);
  CHECK(append(coder, 1, 95, at(95)).ok());
  CHECK(counts_are(coder, 1, 0, 44, 52));
  CHECK(append(coder, 1, 96, at(96)).ok());
  CHECK(counts_are(coder, 1, 0, 44, 53));
  // A frame catches up; the next record is held again, and finish writes it.
  CHECK(coder.code(source, 1, at(96) + 10 * kMs).ok());
  CHECK(append(coder, 1, 97, at(97)).ok());
  CHECK(counts_are(coder, 1, 1, 44, 53));
  CHECK(coder.finish().ok() && counts_are(coder, 1, 0, 44, 54));

  std::vector<Entry> expected = {{0, 0, first}};
  for (std::uint64_t k = 44; k < 97; ++k) expected.push_back({1, k, at(k)});
  expected.push_back({0, 1, at(96) + 10 * kMs});
  expected.push_back({1, 97, at(97)});
  CHECK(same(read_back(path.string()), expected));
  return 0;
}

// Before any frame, finish drops what is held and writes a file of no
// record; a stream dropped before its first frame removes its file, records
// held or not.
int no_frame(vkc::test::GpuContext& gpu, const vkc::test::Scratch& scratch) {
  const fs::path path = scratch.path / "no_frame.vrst";
  {
    vkc::Result<codec_stream::CodecStream> made = make(gpu, 1, path);
    CHECK(made.ok() && append(*made, 1, 0, kT0).ok());
  }
  CHECK(!fs::exists(path));
  vkc::Result<codec_stream::CodecStream> made = make(gpu, 1, path);
  CHECK(made.ok());
  for (std::uint64_t k = 0; k < 3; ++k) {
    CHECK(append(*made, 1, k, kT0 + 20 * kMs * k).ok());
  }
  CHECK(made->finish().ok() && counts_are(*made, 1, 0, 3, 0));
  const std::unique_ptr<stream::StreamReader> reader =
      vrst::open(path.string());
  CHECK(reader != nullptr && reader->recovery().finished);
  CHECK(reader->stats()[0].records == 0 && reader->stats()[1].records == 0);
  return 0;
}

int gpu_main(vkc::test::GpuContext& gpu) {
  const Sphere sphere{vgrid::Vec3f(0.01f, -0.02f, 0.03f), 0.09f};
  vkc::Result<vol::VoxelBlockGrid> source = sphere_grid(gpu, sphere);
  CHECK(source.ok());
  vkc::test::Scratch scratch("grid_codec_stream_audio_");
  CHECK(refused_sessions(gpu, scratch) == 0);
  CHECK(refused_records(gpu, *source, scratch) == 0);
  CHECK(muxed(gpu, *source, scratch) == 0);
  CHECK(bounded(gpu, *source, scratch) == 0);
  CHECK(no_frame(gpu, scratch) == 0);
  std::puts("grid_codec_stream_audio: OK");
  return 0;
}

}  // namespace

int main() {
  return vkc::test::run_on_gpu(vgrid::device_requirements(), gpu_main);
}
