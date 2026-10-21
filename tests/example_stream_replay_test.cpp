// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// serve_stream's replay (examples/common/stream_replay.hpp), on a written
// .vrst: the records from the track's first key; pass 0 unchanged; across
// passes seq rising and key pts never falling, so the writer's rules take
// three passes in a row; dues following the pts gaps, a pass one mean step
// after the last; a lone record paced by the track's rate, and the loop's
// own allocation failure refused; a record with a damaged header or a flag
// this version does not know left out, and an allocation failure reading
// past the damage refused; a damaged first key left out with the P-frames
// after it; and a track with no key, no undamaged key, or none at all,
// refused. Host-only.

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include "scratch_dir.hpp"
#include "stream_files.hpp"
#include "stream_replay.hpp"
#include "test_allocation_failure.hpp"
#include "test_check.hpp"
#include "volumetric_kit/recon/stream/chunks.hpp"
#include "volumetric_kit/recon/stream/stream_writer.hpp"

namespace vkc = volumetric_kit::core;
namespace stream = volumetric_kit::recon::stream;
namespace vrst = vr_test::vrst;

namespace {

using Code = vkc::Status::Code;
using Clock = vr_example::ReplayLoop::Clock;

constexpr std::uint64_t kStep = 33'333'333;

stream::Session session() {
  stream::Session s;
  s.writer = "replay test";
  stream::Track track;
  track.kind = stream::TrackKind::Geometry;
  track.rate = {30, 1};
  s.tracks.push_back(track);
  return s;
}

bool write(const std::string& path,
           const std::vector<stream::RecordInfo>& infos) {
  auto writer = stream::StreamWriter::create(path, session());
  if (!writer.ok()) return false;
  for (const stream::RecordInfo& info : infos) {
    const vrst::Bytes payload = vrst::pattern(100, info.seq);
    if (!(*writer)->append(info, payload.data(), payload.size()).ok()) {
      return false;
    }
  }
  return (*writer)->finish().ok();
}

std::unique_ptr<stream::StreamReader> reader_of(const std::string& path) {
  return vrst::open(path);
}

// A P-frame before the first key, then two GOPs; a lost record (seq 6 to 8),
// and a P-frame whose pts goes back.
int loop_case(const std::string& dir) {
  const std::string path = dir + "/loop.vrst";
  const std::vector<stream::RecordInfo> infos = {
      vrst::info(0, false, 0 * kStep, 0), vrst::info(0, true, 1 * kStep, 1),
      vrst::info(0, false, 2 * kStep, 2), vrst::info(0, false, 4 * kStep, 3),
      vrst::info(0, false, 3 * kStep, 4), vrst::info(0, true, 5 * kStep, 5),
      vrst::info(0, false, 6 * kStep, 6), vrst::info(0, false, 7 * kStep, 8)};
  CHECK(write(path, infos));
  const auto reader = reader_of(path);
  CHECK(reader != nullptr);
  vkc::Result<vr_example::ReplayLoop> made =
      vr_example::ReplayLoop::create(*reader);
  CHECK(made.ok());
  const vr_example::ReplayLoop& loop = *made;
  CHECK(loop.records().size() == infos.size() - 1);
  CHECK(loop.records().front().info.key &&
        loop.records().front().info.seq == 1);
  for (std::size_t i = 0; i < loop.records().size(); ++i) {
    CHECK(vrst::same(loop.records()[i].info, infos[i + 1]));
    CHECK(vrst::same(loop.restamp(loop.records()[i].info, 0), infos[i + 1]));
  }

  // Three passes in a row pass the writer's rules, as the server checks them.
  vkc::Result<stream::ChunkEncoder> encoder =
      stream::ChunkEncoder::create(session());
  CHECK(encoder.ok());
  std::uint64_t last_seq = 0;
  std::uint64_t last_key_pts = 0;
  for (std::uint64_t pass = 0; pass < 3; ++pass) {
    for (const stream::Record& r : loop.records()) {
      const stream::RecordInfo info = loop.restamp(r.info, pass);
      CHECK(encoder->frame(info, nullptr, 0).ok());
      CHECK(pass == 0 || info.seq > last_seq);
      if (info.key) {
        CHECK(info.pts_ns >= last_key_pts);
        last_key_pts = info.pts_ns;
      }
      last_seq = info.seq;
    }
  }
  // A pass's seq span: 1 to 8, so pass 1 starts at 9.
  CHECK(loop.restamp(loop.records().front().info, 1).seq == 9);

  // Dues follow the pts gaps from the least pts, the key's, the one that goes
  // back included; a pass is the span (1 to 7 kStep) and one mean step later.
  const Clock::time_point start = Clock::now();
  const auto ns = [&](Clock::time_point t) {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(t - start)
        .count();
  };
  CHECK(ns(loop.due(start, 0, 1 * kStep)) == 0);
  CHECK(ns(loop.due(start, 0, 4 * kStep)) == static_cast<long long>(3 * kStep));
  CHECK(ns(loop.due(start, 0, 3 * kStep)) == static_cast<long long>(2 * kStep));
  const std::uint64_t mean = (7 * kStep - 1 * kStep) / 6;
  const std::uint64_t pass_ns = 6 * kStep + mean;
  CHECK(ns(loop.due(start, 1, 1 * kStep)) == static_cast<long long>(pass_ns));
  CHECK(ns(loop.due(start, 1, 1 * kStep)) - ns(loop.due(start, 0, 7 * kStep)) ==
        static_cast<long long>(mean));
  CHECK(loop.restamp(loop.records().front().info, 2).pts_ns ==
        1 * kStep + 2 * pass_ns);
  return 0;
}

// One record: a pass is the track's frame period. The loop's own last
// allocation, its records' growth, failed is OutOfMemory.
int lone_record_case(const std::string& dir) {
  const std::string path = dir + "/lone.vrst";
  CHECK(write(path, {vrst::info(0, true, 500, 4)}));
  const auto reader = reader_of(path);
  CHECK(reader != nullptr);
  auto& failure = vr_test::allocation_failure;
  failure = {};
  failure.measure = true;
  vkc::Result<vr_example::ReplayLoop> loop =
      vr_example::ReplayLoop::create(*reader);
  failure.measure = false;
  CHECK(loop.ok());
  const stream::RecordInfo next = loop->restamp(loop->records()[0].info, 1);
  CHECK(next.seq == 5 && next.pts_ns == 500 + 1'000'000'000 / 30);

  CHECK(failure.bytes == sizeof(stream::Record));
  failure.site = failure.measured_site;
  failure.armed = true;
  const Code refused =
      vr_example::ReplayLoop::create(*reader).status().domain();
  const bool injected = failure.injected;
  failure = {};
  CHECK(injected && refused == Code::OutOfMemory);
  return 0;
}

// Seq 2's header fails its CRC (InvalidArgument) and seq 4 sets flag bit 1
// (Unsupported): both are left out. An allocation failure reading past the
// damage, here its message's, is the loop's failure, not more damage.
int damage_case(const std::string& dir) {
  const std::string path = dir + "/damaged.vrst";
  std::vector<stream::RecordInfo> infos;
  for (std::uint64_t seq = 0; seq < 5; ++seq) {
    infos.push_back(vrst::info(0, seq % 3 == 0, seq * kStep, seq));
  }
  CHECK(write(path, infos));
  vrst::Bytes file = vr_test::read_bytes(path);
  std::vector<vrst::ChunkAt> records;
  for (const vrst::ChunkAt& c : vrst::chunks(file)) {
    if (c.tag == "RECD") records.push_back(c);
  }
  CHECK(records.size() == infos.size());
  const std::uint64_t damaged = records[2].offset;
  file = vrst::flipped(file, damaged + 26, 0);
  // The flags, then the header's CRC over the bytes before it.
  const vrst::ChunkAt& flagged = records[4];
  file[flagged.offset + 22] |= 2;
  const std::uint64_t crc_at = flagged.offset + flagged.header_bytes - 4;
  const std::uint32_t crc =
      vrst::crc32(file.data() + flagged.offset, flagged.header_bytes - 4);
  for (int i = 0; i < 4; ++i) {
    file[crc_at + i] = static_cast<std::uint8_t>(crc >> (8 * i));
  }
  CHECK(vrst::write_bytes(path, file));
  const auto reader = reader_of(path);
  CHECK(reader != nullptr);
  vkc::Result<vr_example::ReplayLoop> loop =
      vr_example::ReplayLoop::create(*reader);
  CHECK(loop.ok() && loop->records().size() == 3);
  CHECK(loop->records()[1].info.seq == 1 && loop->records()[2].info.seq == 3);

  // As stream_oom_test does: measure the read at the damage, then fail its
  // last allocation at the same site.
  auto& failure = vr_test::allocation_failure;
  failure = {};
  failure.measure = true;
  stream::Position pos{damaged};
  stream::Record record;
  const Code measured = reader->next(&pos, &record).status().domain();
  failure.measure = false;
  CHECK(measured == Code::InvalidArgument);
  CHECK(failure.bytes != 0 && failure.measured_site != nullptr);
  failure.site = failure.measured_site;
  failure.armed = true;
  const Code refused =
      vr_example::ReplayLoop::create(*reader).status().domain();
  const bool injected = failure.injected;
  failure = {};
  CHECK(injected && refused == Code::OutOfMemory);
  return 0;
}

// A finished file's key whose header fails its CRC, which its footer still
// lists: the loop starts at the next key read whole, and without one is
// refused.
int damaged_key_case(const std::string& dir) {
  std::vector<stream::RecordInfo> infos;
  for (std::uint64_t seq = 0; seq < 4; ++seq) {
    infos.push_back(vrst::info(0, seq % 2 == 0, seq * kStep, seq));
  }
  const std::string first = dir + "/first_key.vrst";
  CHECK(write(first, infos));
  vrst::Bytes file = vr_test::read_bytes(first);
  const std::vector<vrst::ChunkAt> chunks = vrst::chunks(file);  // VRST first
  CHECK(chunks[1].tag == "RECD" && chunks[3].tag == "RECD");
  file = vrst::flipped(file, chunks[1].offset + 26, 0);
  CHECK(vrst::write_bytes(first, file));
  const auto later = reader_of(first);
  CHECK(later != nullptr && later->recovery().finished);
  vkc::Result<vr_example::ReplayLoop> loop =
      vr_example::ReplayLoop::create(*later);
  CHECK(loop.ok() && loop->records().size() == 2 &&
        loop->records()[0].info.seq == 2);

  const std::string both = dir + "/both_keys.vrst";
  CHECK(vrst::write_bytes(both, vrst::flipped(file, chunks[3].offset + 26, 0)));
  const auto none = reader_of(both);
  CHECK(none != nullptr && none->recovery().finished);
  CHECK(vr_example::ReplayLoop::create(*none).status().domain() ==
        Code::NotFound);
  return 0;
}

int refusals_case(const std::string& dir) {
  const std::string path = dir + "/no_key.vrst";
  CHECK(write(path, {vrst::info(0, false, 0, 0), vrst::info(0, false, 1, 1)}));
  const auto reader = reader_of(path);
  CHECK(reader != nullptr);
  CHECK(vr_example::ReplayLoop::create(*reader).status().domain() ==
        Code::NotFound);
  CHECK(vr_example::ReplayLoop::create(*reader, 1).status().domain() ==
        Code::InvalidArgument);
  return 0;
}

}  // namespace

int main() {
  vr_test::Scratch scratch("recon_example_stream_replay_");
  const std::string dir = scratch.path.string();
  if (loop_case(dir) != 0) return 1;
  if (lone_record_case(dir) != 0) return 1;
  if (damage_case(dir) != 0) return 1;
  if (damaged_key_case(dir) != 0) return 1;
  if (refusals_case(dir) != 0) return 1;
  std::puts("stream replay: OK");
  return 0;
}
