// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// A codec stream (codec_stream/codec_stream.hpp), on a grid holding a
// sphere: outputs it cannot make refused on create, before the encoder, whose
// refusal takes the stream's file back, a session that is not one Geometry
// track refused, and none written without outputs; each frame as it
// went out, its type, bytes and blocks the frame's and its encode and decode
// rows its own; the .vrst's track as the session gave it, and every encoded
// frame a record of it, a frame the decoder refuses included, counted by
// coded(), and the stream going on after it, byte for byte what an encoder
// returns, a key exactly when an I-frame, and the same file on a rerun; after
// a frame encoded but not decoded, an I-frame where the encoder would code a
// P-frame, whole and in halves; a frame out of order, or any call once
// finished or moved from, refused before anything is encoded; a stream never
// finished still read whole, and one dropped before its first frame removed;
// frames in halves around a caller's batch the file code() writes, byte for
// byte, the batch's device rows split between the frame and the caller, the
// front half again on a grid changed after it, a pending frame since
// superseded refused, and a viewer's request for a key read by the front
// half; and, served on 127.0.0.1 port 0 (never every interface), every
// frame the file records reaching a client as it is coded,
// with the file's seq, key, pts, set, bytes and offset, and the end once
// finished, a frame the file refuses never served, serving alone writing no
// file, a port in use refused before any file is made, and neither a client
// that leaves nor a record the server refuses failing a frame. Without the
// decode, the same file and records, frames with no decode rows, and no
// player grid. A geometry-only session's file and wire, byte for byte as
// before audio. The device part skips with no device.

#include <signal.h>
#include <sys/resource.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "volumetric_kit/core/base/file_io.hpp"
#include "volumetric_kit/core/base/stage_metrics.hpp"
#include "volumetric_kit/core/testing/gpu_test.hpp"
#include "volumetric_kit/core/testing/scratch_dir.hpp"
#include "volumetric_kit/core/vulkan/command_batch.hpp"
#include "volumetric_kit/core/vulkan/gpu_timer.hpp"
#include "volumetric_kit/grid/base/device_requirements.hpp"
#include "volumetric_kit/grid/codec/decoder.hpp"
#include "volumetric_kit/grid/codec/encoder.hpp"
#include "volumetric_kit/grid/codec_stream/codec_stream.hpp"
#include "volumetric_kit/grid/networking/stream_client.hpp"
#include "volumetric_kit/grid/networking/stream_server.hpp"
#include "volumetric_kit/grid/stream/chunks.hpp"
#include "volumetric_kit/grid/stream/stream_reader.hpp"
#include "volumetric_kit/grid/stream/stream_writer.hpp"
#include "volumetric_kit/grid/testing/codec_fixture.hpp"
#include "volumetric_kit/grid/testing/networking_loopback.hpp"
#include "volumetric_kit/grid/testing/stage_rows.hpp"

namespace vgrid = volumetric_kit::grid;
namespace vkc = volumetric_kit::core;
namespace codec = volumetric_kit::grid::codec;
namespace codec_stream = volumetric_kit::grid::codec_stream;
namespace networking = volumetric_kit::grid::networking;
namespace stream = volumetric_kit::grid::stream;
namespace net = vgrid_test::net;
namespace fs = std::filesystem;
using namespace codec_fixture;

namespace {

// A frame's time, 30 a second from frame 0 at 0, to the nearest ns.
std::uint64_t pts(std::size_t frame) {
  return static_cast<std::uint64_t>(std::llround(double(frame) * 1e9 / 30.0));
}

// The session the tests write: one Geometry track at `rate`.
stream::Session session(stream::Rate rate = {}) {
  stream::Session s;
  s.writer = "grid_codec_stream";
  stream::Track track;
  track.kind = stream::TrackKind::Geometry;
  track.rate = rate;
  s.tracks.push_back(track);
  return s;
}

// A config the encoder refuses. create makes the outputs before the codec's
// kernels, so an output refused with it is refused for its own cause.
codec::EncoderConfig unusable_config() {
  codec::EncoderConfig config;
  config.intra_period = 0;
  return config;
}

// Whether `frame` holds its encode's rows and, when decoded, its decode's.
bool rows_of(const codec_stream::SentFrame& frame, bool decoded) {
  return vgrid_test::find_row(frame.encode, "codec encode") != nullptr &&
         (vgrid_test::find_row(frame.decode, "codec decode") != nullptr) ==
             decoded;
}

// Outputs that cannot be made are refused by create with the writer's code,
// naming the path, before the encoder; outputs that can be made get the
// encoder's refusal, and their file is taken back. A session that is not one
// Geometry track is refused before any output is made.
int refused(vkc::test::GpuContext& gpu, const vkc::test::Scratch& scratch) {
  const auto refuses = [&gpu](const codec_stream::CodecOutputs& outputs,
                              const stream::Session& s, vkc::Status::Code code,
                              const std::string& named) {
    vkc::Result<codec_stream::CodecStream> made =
        codec_stream::CodecStream::create(gpu.device, gpu.allocator,
                                          unusable_config(), s, outputs);
    return !made.ok() && made.status().domain() == code &&
           made.status().message().find(named) != std::string::npos;
  };
  // A stream must be new: what is at its path is left as it was.
  codec_stream::CodecOutputs existing;
  existing.stream = (scratch.path / "earlier.vrst").string();
  CHECK(vkc::write_file(existing.stream, "earlier", vkc::WriteMode::CreateNew)
            .ok());
  CHECK(refuses(existing, session(), vkc::Status::Code::InvalidArgument,
                existing.stream));
  const vkc::Result<std::string> kept = vkc::read_file(existing.stream);
  CHECK(kept.ok() && *kept == "earlier");
  codec_stream::CodecOutputs orphan;
  orphan.stream = (scratch.path / "missing" / "frames.vrst").string();
  CHECK(refuses(orphan, session(), vkc::Status::Code::NotFound, orphan.stream));
  codec_stream::CodecOutputs made;
  made.stream = (scratch.path / "uncoded.vrst").string();
  CHECK(refuses(made, session(), vkc::Status::Code::InvalidArgument,
                "intra_period"));
  CHECK(!fs::exists(made.stream));
  // No track, a track of another kind, and two tracks, before the file.
  stream::Session none = session();
  none.tracks.clear();
  stream::Session other = session();
  other.tracks[0].kind = static_cast<stream::TrackKind>(7);
  stream::Session two = session();
  two.tracks.push_back(two.tracks[0]);
  for (const stream::Session* s : {&none, &other, &two}) {
    CHECK(refuses(made, *s, vkc::Status::Code::InvalidArgument, "Geometry"));
    CHECK(!fs::exists(made.stream));
  }
  return 0;
}

// With no outputs, the stream codes but writes no file, so nothing lands in
// the working directory; a frame out of order, and any call once finished, is
// refused all the same, with no writer to refuse it, a front half included,
// and a frame whose front half ran before finish, nothing coded.
// A stream moved from refuses every call,
// its halves' too, and reports nothing, the one it moved into going on.
int writes_nothing(vkc::test::GpuContext& gpu, vol::VoxelBlockGrid& source,
                   const vkc::test::Scratch& scratch) {
  const fs::path quiet = scratch.path / "quiet";
  CHECK(fs::create_directory(quiet));
  vkc::Result<codec_stream::CodecStream> made =
      codec_stream::CodecStream::create(gpu.device, gpu.allocator, {},
                                        session());
  CHECK(made.ok());
  const fs::path before = fs::current_path();
  fs::current_path(quiet);
  const vkc::Status coded[] = {
      made->code(source, 0, pts(0)).status(),
      made->code(source, 2, pts(2)).status(),
      // Out of order: a frame not above the last, a pts below the last.
      made->code(source, 2, pts(3)).status(),
      made->code(source, 3, pts(1)).status(), made->finish(),
      // Finished: a frame in order, and a second finish.
      made->code(source, 4, pts(4)).status(), made->finish()};
  fs::current_path(before);
  CHECK(coded[0].ok() && coded[1].ok() && coded[4].ok());
  for (const int i : {2, 3, 5, 6}) {
    CHECK(coded[i].domain() == vkc::Status::Code::InvalidArgument);
  }
  CHECK(made->coded() == 2 && made->last_pts_ns() == pts(2));
  CHECK(fs::is_empty(quiet));
  {
    vkc::CommandBatch front(gpu.device, gpu.allocator);
    CHECK(made->record_front(front, source).status().domain() ==
          vkc::Status::Code::InvalidArgument);
    CHECK(made->coded() == 2);
  }
  {  // A front half run before finish, its frame finished after it.
    vkc::Result<codec_stream::CodecStream> halted =
        codec_stream::CodecStream::create(gpu.device, gpu.allocator, {},
                                          session());
    CHECK(halted.ok());
    vkc::CommandBatch front(gpu.device, gpu.allocator);
    const vkc::Result<codec::PendingEncode> pending =
        halted->record_front(front, source);
    CHECK(pending.ok());
    vkc::Result<vkc::PendingBatch> done = front.submit_async({}, {});
    CHECK(done.ok() && done->wait().ok());
    halted->front_ran(done->device_rows());
    CHECK(halted->finish().ok());
    const vkc::Status late = halted->finish_frame(*pending, 0, pts(0)).status();
    CHECK(late.domain() == vkc::Status::Code::InvalidArgument &&
          late.message() == "CodecStream: finished");
    CHECK(halted->coded() == 0 && halted->player() == nullptr);
  }

  vkc::Result<codec_stream::CodecStream> first =
      codec_stream::CodecStream::create(gpu.device, gpu.allocator, {},
                                        session());
  CHECK(first.ok() && first->code(source, 0, pts(0)).ok());
  codec_stream::CodecStream moved = std::move(first).value();
  CHECK(first->code(source, 1, pts(1)).status().domain() ==
        vkc::Status::Code::InvalidArgument);
  CHECK(first->finish().domain() == vkc::Status::Code::InvalidArgument);
  CHECK(first->coded() == 0 && first->last_pts_ns() == 0);
  CHECK(first->player() == nullptr && first->server() == nullptr);
  CHECK(!first->valid() && moved.valid());
  {
    vkc::CommandBatch front(gpu.device, gpu.allocator);
    CHECK(first->record_front(front, source).status().domain() ==
          vkc::Status::Code::InvalidArgument);
    vkc::StageMetrics rest;
    first->front_ran({vkc::StageRow{"codec encode", 0.0, 1.0, true}}, &rest);
    CHECK(rest.rows().empty());
    const codec::PendingEncode none;
    CHECK(first->redo(none, source).status().domain() ==
          vkc::Status::Code::InvalidArgument);
    CHECK(first->finish_frame(none, 1, pts(1)).status().domain() ==
          vkc::Status::Code::InvalidArgument);
  }
  CHECK(moved.coded() == 1 && moved.player() != nullptr);
  CHECK(moved.code(source, 1, pts(1)).ok() && moved.coded() == 2);
  return 0;
}

// Frames 3 and 7 of `source`, frames out of order, `other`'s frame 9, which
// the decoder refuses, `source`'s frame 10, then finish, into `outputs`.
// `sent` receives the frames that went out whole.
int code_frames(vkc::test::GpuContext& gpu, vol::VoxelBlockGrid& source,
                vol::VoxelBlockGrid& other,
                const codec_stream::CodecOutputs& outputs,
                std::vector<codec_stream::SentFrame>* sent) {
  vkc::Result<codec_stream::CodecStream> made =
      codec_stream::CodecStream::create(gpu.device, gpu.allocator, {},
                                        session({30000, 1001}), outputs);
  CHECK(made.ok());
  codec_stream::CodecStream coder = std::move(made).value();
  // The stream's file is there before any frame is coded.
  CHECK(fs::is_regular_file(outputs.stream));
  CHECK(coder.coded() == 0 && coder.player() == nullptr);
  CHECK(coder.server() == nullptr && coder.last_pts_ns() == 0);

  sent->clear();
  for (const std::size_t frame : {3, 7}) {
    vkc::Result<codec_stream::SentFrame> out =
        coder.code(source, frame, pts(frame));
    CHECK(out.ok() && out->type == codec::FrameType::kIntra);
    CHECK(out->bytes > 0 && out->blocks > 0 && !out->tail);
    CHECK(rows_of(*out, true));
    sent->push_back(std::move(out).value());
  }
  CHECK(coder.coded() == 2 && coder.player() != nullptr);
  CHECK(coder.last_pts_ns() == pts(7));

  // A frame index or a pts out of order is refused before it is encoded,
  // so it has no record.
  const vkc::Status again = coder.code(source, 7, pts(8)).status();
  CHECK(again.domain() == vkc::Status::Code::InvalidArgument);
  CHECK(again.message().find("frame 7") != std::string::npos);
  CHECK(coder.code(source, 8, pts(6)).status().domain() ==
        vkc::Status::Code::InvalidArgument);
  CHECK(coder.coded() == 2);

  // A frame the decoder refuses -- another grid's geometry than the
  // player's -- has a record, counted, and the stream goes on after it.
  CHECK(coder.code(other, 9, pts(9)).status().domain() ==
        vkc::Status::Code::InvalidArgument);
  CHECK(coder.coded() == 3 && coder.last_pts_ns() == pts(9));
  vkc::Result<codec_stream::SentFrame> last = coder.code(source, 10, pts(10));
  CHECK(last.ok() && coder.coded() == 4);
  sent->push_back(std::move(last).value());

  // Finished, nothing more is coded.
  CHECK(coder.finish().ok());
  CHECK(coder.code(source, 11, pts(11)).status().domain() ==
        vkc::Status::Code::InvalidArgument);
  CHECK(coder.finish().domain() == vkc::Status::Code::InvalidArgument);
  CHECK(coder.coded() == 4);
  return 0;
}

// code_frames' .vrst: the session's one Geometry track, and a key record for
// each frame the encoder returned, in order, its payload what a separate
// encoder returns for the same grid.
int check_records(vkc::test::GpuContext& gpu, vol::VoxelBlockGrid& source,
                  const std::string& path,
                  const std::vector<codec_stream::SentFrame>& sent,
                  float other_voxel) {
  vkc::Result<std::unique_ptr<stream::StreamReader>> opened =
      stream::StreamReader::open(path);
  CHECK(opened.ok());
  const stream::StreamReader& reader = **opened;
  const stream::Session& s = reader.session();
  CHECK(s.writer == "grid_codec_stream");
  CHECK(s.id == (std::array<std::uint8_t, 16>{}));
  CHECK(s.created_ns == 0 && s.calibration.empty());
  CHECK(s.tracks.size() == 1);
  const stream::Track& track = s.tracks[0];
  CHECK(track.kind == stream::TrackKind::Geometry && track.clock == 1);
  CHECK(track.rate.num == 30000 && track.rate.den == 1001);
  CHECK(track.width == 0 && track.height == 0);
  CHECK(track.sensor.empty() && track.params.empty());
  CHECK(reader.recovery().finished && reader.recovery().torn_bytes == 0);
  // Every frame an I-frame, at the default period of 1: every record a key.
  CHECK(reader.stats()[0].records == 4 && reader.stats()[0].keys == 4);

  vkc::Result<codec::Encoder> enc =
      codec::Encoder::create(gpu.device, gpu.allocator, {});
  CHECK(enc.ok());
  const vkc::Result<std::vector<std::uint8_t>> expected = enc->encode(source);
  CHECK(expected.ok());
  const std::size_t sets[] = {3, 7, 9, 10};
  stream::Position pos = reader.begin();
  stream::Record record;
  std::vector<std::uint8_t> payload;
  std::size_t whole = 0;  // the next frame that went out whole
  for (std::size_t i = 0; i < std::size(sets); ++i) {
    const vkc::Result<bool> more = reader.next(&pos, &record);
    CHECK(more.ok() && *more);
    CHECK(record.info.track == 0 && record.info.key);
    CHECK(record.info.seq == i && record.info.set == sets[i]);
    CHECK(record.info.pts_ns == pts(sets[i]));
    CHECK(reader.read(record, &payload).ok());
    const vkc::Result<codec::FrameInfo> info =
        codec::read_frame_info(payload.data(), payload.size());
    CHECK(info.ok());
    if (sets[i] != 9) {
      CHECK(payload == *expected);
      CHECK(payload.size() == sent[whole].bytes &&
            info->block_count == sent[whole].blocks);
      ++whole;
    } else {  // the frame the decoder refused
      CHECK(info->voxel_size == other_voxel);
    }
  }
  CHECK(whole == sent.size());
  const vkc::Result<bool> end = reader.next(&pos, &record);
  CHECK(end.ok() && !*end);
  return 0;
}

// A stream destroyed without finish leaves a file that reads as a killed
// writer's: not finished, every record whole. Destroyed before its first
// frame, it leaves none, so a rerun onto its path is not refused.
int unfinished(vkc::test::GpuContext& gpu, vol::VoxelBlockGrid& source,
               const vkc::test::Scratch& scratch) {
  codec_stream::CodecOutputs outputs;
  outputs.stream = (scratch.path / "unfinished.vrst").string();
  {  // Dropped before its first frame: the file is taken back.
    vkc::Result<codec_stream::CodecStream> made =
        codec_stream::CodecStream::create(gpu.device, gpu.allocator, {},
                                          session(), outputs);
    CHECK(made.ok() && fs::exists(outputs.stream));
  }
  CHECK(!fs::exists(outputs.stream));
  {  // Created again at the same path, and dropped with two frames.
    vkc::Result<codec_stream::CodecStream> made =
        codec_stream::CodecStream::create(gpu.device, gpu.allocator, {},
                                          session(), outputs);
    CHECK(made.ok());
    CHECK(made->code(source, 0, pts(0)).ok());
    CHECK(made->code(source, 1, pts(1)).ok());
  }
  vkc::Result<std::unique_ptr<stream::StreamReader>> opened =
      stream::StreamReader::open(outputs.stream);
  CHECK(opened.ok());
  const stream::StreamReader& reader = **opened;
  CHECK(!reader.recovery().finished && reader.recovery().torn_bytes == 0);
  CHECK(reader.stats()[0].records == 2);
  stream::Position pos = reader.begin();
  stream::Record record;
  std::vector<std::uint8_t> payload;
  for (std::uint64_t seq = 0; seq < 2; ++seq) {
    const vkc::Result<bool> more = reader.next(&pos, &record);
    CHECK(more.ok() && *more && record.info.seq == seq);
    CHECK(reader.read(record, &payload).ok() && !payload.empty());
  }
  return 0;
}

// At a period of 4, `source` coded at frames 0-2 and 5-9 and `other`, which
// the player refuses, at 3 and 4: frame 3 an I-frame on its geometry, and 4
// one too, where the encoder would code a P-frame on 3, which the player
// lacks; 5 an I-frame, the player's geometry again, and the period from it.
// Each frame that went out whole has its type, and each record is a key
// exactly when it holds an I-frame.
int periodic(vkc::test::GpuContext& gpu, vol::VoxelBlockGrid& source,
             vol::VoxelBlockGrid& other, const vkc::test::Scratch& scratch) {
  constexpr codec::FrameType I = codec::FrameType::kIntra;
  constexpr codec::FrameType P = codec::FrameType::kInter;
  codec_stream::CodecOutputs outputs;
  outputs.stream = (scratch.path / "periodic.vrst").string();
  codec::EncoderConfig config;
  config.intra_period = 4;
  vkc::Result<codec_stream::CodecStream> made =
      codec_stream::CodecStream::create(gpu.device, gpu.allocator, config,
                                        session(), outputs);
  CHECK(made.ok());
  codec_stream::CodecStream& coder = made.value();
  const codec::FrameType types[] = {I, P, P, I, I, I, P, P, P, I};
  for (std::size_t frame = 0; frame < std::size(types); ++frame) {
    const bool refused = frame == 3 || frame == 4;
    const vkc::Result<codec_stream::SentFrame> s =
        coder.code(refused ? other : source, frame, pts(frame));
    CHECK(s.ok() != refused);
    CHECK(refused ==
          (s.status().domain() == vkc::Status::Code::InvalidArgument));
    CHECK(refused || (s->type == types[frame] && rows_of(*s, true)));
  }
  CHECK(coder.finish().ok() && coder.coded() == std::size(types));

  vkc::Result<std::unique_ptr<stream::StreamReader>> opened =
      stream::StreamReader::open(outputs.stream);
  CHECK(opened.ok());
  const stream::StreamReader& reader = **opened;
  CHECK(reader.stats()[0].records == 10 && reader.stats()[0].keys == 5);
  stream::Position pos = reader.begin();
  stream::Record record;
  std::vector<std::uint8_t> payload;
  for (std::size_t frame = 0; frame < std::size(types); ++frame) {
    const vkc::Result<bool> more = reader.next(&pos, &record);
    CHECK(more.ok() && *more && record.info.set == frame);
    CHECK(reader.read(record, &payload).ok());
    const vkc::Result<codec::FrameInfo> info =
        codec::read_frame_info(payload.data(), payload.size());
    CHECK(info.ok() && info->type == types[frame]);
    CHECK(record.info.key == (types[frame] == I));
  }
  return 0;
}

// Outputs serving on 127.0.0.1, a free port, besides the stream at
// `stream_path`, if any.
codec_stream::CodecOutputs served_on_loopback(
    const fs::path& stream_path = {}) {
  codec_stream::CodecOutputs outputs;
  outputs.stream = stream_path.string();
  outputs.serve.emplace();
  outputs.serve->host = "127.0.0.1";
  outputs.serve->port = 0;
  return outputs;
}

// A client of `coder`'s server that has its session, so it misses no frame
// coded after; null, the reason on stderr, if it cannot.
std::unique_ptr<networking::StreamClient> joined(
    const codec_stream::CodecStream& coder) {
  if (coder.server() == nullptr) return nullptr;
  std::unique_ptr<networking::StreamClient> client =
      net::connect(coder.server()->port());
  if (!client || !net::wait_session(*client)) return nullptr;
  return client;
}

// At a period of 3, `source` coded at frames 0-6 but 3, `other`'s frame,
// which the decoder refuses, with a client joined from the start and reading
// each frame once it is coded: it gets every record of the file, the refused
// frame's included, with the file's seq, key, pts, set, payload and offset,
// under the file's session, then the end once the stream is finished, which
// leaves the stream without a server.
int serves(vkc::test::GpuContext& gpu, vol::VoxelBlockGrid& source,
           vol::VoxelBlockGrid& other, const vkc::test::Scratch& scratch) {
  const codec_stream::CodecOutputs outputs =
      served_on_loopback(scratch.path / "served.vrst");
  codec::EncoderConfig config;
  config.intra_period = 3;
  vkc::Result<codec_stream::CodecStream> made =
      codec_stream::CodecStream::create(gpu.device, gpu.allocator, config,
                                        session(), outputs);
  CHECK(made.ok());
  codec_stream::CodecStream& coder = made.value();
  CHECK(coder.server() != nullptr && coder.server()->port() != 0);
  const std::uint16_t port = coder.server()->port();
  const std::unique_ptr<networking::StreamClient> client = joined(coder);
  CHECK(client != nullptr);
  std::vector<net::Got> got;
  for (std::size_t frame = 0; frame < 7; ++frame) {
    const bool refused = frame == 3;
    CHECK(coder.code(refused ? other : source, frame, pts(frame)).ok() !=
          refused);
    net::Got record;
    CHECK(net::next_within(*client, &record));
    got.push_back(std::move(record));
  }
  CHECK(coder.finish().ok());
  CHECK(coder.server() == nullptr);
  CHECK(net::wait_end(*client, port));

  vkc::Result<std::unique_ptr<stream::StreamReader>> opened =
      stream::StreamReader::open(outputs.stream);
  CHECK(opened.ok());
  const stream::StreamReader& reader = **opened;
  CHECK(vgrid_test::vrst::same(*client->session(), reader.session()));
  CHECK(reader.stats()[0].records == got.size());
  stream::Position pos = reader.begin();
  stream::Record record;
  std::vector<std::uint8_t> payload;
  for (std::size_t i = 0; i < got.size(); ++i) {
    const vkc::Result<bool> more = reader.next(&pos, &record);
    CHECK(more.ok() && *more);
    CHECK(reader.read(record, &payload).ok());
    const stream::Record& served = got[i].record;
    CHECK(served.info.seq == i && served.info.set == i);
    CHECK(served.info.pts_ns == pts(i));
    CHECK(vgrid_test::vrst::same(served.info, record.info));
    CHECK(served.offset == record.offset && got[i].payload == payload);
  }
  return 0;
}

// Frames in halves around a caller's timed batch, each front half recorded
// inside the caller's own scope ("set"), then finished: the file code()
// writes, byte for byte, I- and P-frames, and two frames on `other` that the
// decoder refuses, the second an I-frame as code() makes it, where the
// encoder would code a P-frame on the first; each frame published once,
// before its decode, its rows its encode with "crc", "write", "publish" and
// its wait, and each frame as code() returns it. Of the batch's device rows,
// the front half's are the frame's alone and the caller's scope the
// caller's. A frame out of order is refused, its pending frame dropped, and
// the stream goes on. A front half on `other`, then again on `source` once
// its batch ran, makes code()'s frame of `source`, the wait timed, and the
// pending frame it replaced is refused.
int halves(vkc::test::GpuContext& gpu, vol::VoxelBlockGrid& source,
           vol::VoxelBlockGrid& other, const vkc::test::Scratch& scratch) {
  codec::EncoderConfig config;
  config.intra_period = 3;
  codec_stream::CodecOutputs by_code;
  by_code.stream = (scratch.path / "halves_code.vrst").string();
  const codec_stream::CodecOutputs by_halves =
      served_on_loopback(scratch.path / "halves.vrst");
  vkc::Result<codec_stream::CodecStream> a = codec_stream::CodecStream::create(
      gpu.device, gpu.allocator, config, session(), by_code);
  vkc::Result<codec_stream::CodecStream> b = codec_stream::CodecStream::create(
      gpu.device, gpu.allocator, config, session(), by_halves);
  vkc::Result<vkc::GpuTimer> timer = vkc::GpuTimer::create(gpu.device, 64);
  CHECK(a.ok() && b.ok() && timer.ok());
  // `grid`'s front half into a timed batch, inside the caller's "set" scope,
  // run; its rows, the caller's included, into `rows`.
  const auto front_half =
      [&](vol::VoxelBlockGrid& grid,
          vkc::StageMetrics* rows) -> vkc::Result<codec::PendingEncode> {
    vkc::CommandBatch front(gpu.device, gpu.allocator);
    front.time(*timer);
    vkc::Result<codec::PendingEncode> pending = [&] {
      const vkc::StageScope set(rows, "set");
      return b->record_front(front, grid);
    }();
    if (!pending.ok()) return pending;
    vkc::Result<vkc::PendingBatch> done = front.submit_async({}, {});
    if (!done.ok()) return vkc::Result<codec::PendingEncode>(done.status());
    const vkc::Status waited = done->wait();
    if (!waited.ok()) return vkc::Result<codec::PendingEncode>(waited);
    b->front_ran(done->device_rows(), rows);
    return pending;
  };
  const auto timed = [](const vkc::StageMetrics& rows, const char* name) {
    const vkc::StageRow* row = vgrid_test::find_row(rows, name);
    return row != nullptr && row->device != vkc::DeviceTime::None;
  };
  for (std::size_t frame = 0; frame < 5; ++frame) {
    const bool refused = frame == 2 || frame == 3;
    vol::VoxelBlockGrid& grid = refused ? other : source;
    const vkc::Result<codec_stream::SentFrame> x =
        a->code(grid, frame, pts(frame));
    CHECK(x.ok() != refused);
    vkc::StageMetrics rows;
    const vkc::Result<codec::PendingEncode> pending = front_half(grid, &rows);
    CHECK(pending.ok());
    // The caller's scope holds its device half; the frame's are not here.
    CHECK(timed(rows, "set") &&
          vgrid_test::find_row(rows, "  ..forward") == nullptr);
    int published = 0;
    const vkc::Result<codec_stream::SentFrame> y = b->finish_frame(
        *pending, frame, pts(frame), &rows, [&published] { ++published; });
    CHECK(y.ok() != refused && published == 1);
    for (const char* row : {"codec encode", "  ..forward", "crc", "write",
                            "publish", "wait: frame"}) {
      CHECK(vgrid_test::find_row(rows, row) != nullptr);
    }
    if (refused) continue;
    CHECK(y->type == x->type && y->bytes == x->bytes && y->blocks == x->blocks);
    CHECK(rows_of(*y, true) && timed(y->encode, "  ..forward"));
    CHECK(vgrid_test::find_row(y->encode, "set") == nullptr);
  }
  // Out of order: refused, its pending frame dropped, so the next records
  // afresh.
  vkc::StageMetrics rows;
  int published = 0;
  const auto count = [&published] { ++published; };
  vkc::Result<codec::PendingEncode> late = front_half(source, &rows);
  CHECK(late.ok());
  CHECK(b->finish_frame(*late, 4, pts(4), &rows, count).status().domain() ==
            vkc::Status::Code::InvalidArgument &&
        published == 0);
  CHECK(a->code(source, 5, pts(5)).ok());
  vkc::Result<codec::PendingEncode> next = front_half(source, &rows);
  CHECK(next.ok());
  CHECK(b->finish_frame(*next, 5, pts(5), &rows, count).ok() && published == 1);
  // On `other`, then again on `source`: code()'s frame of `source`.
  CHECK(a->code(source, 6, pts(6)).ok());
  vkc::StageMetrics again_rows;
  vkc::Result<codec::PendingEncode> first = front_half(other, &again_rows);
  CHECK(first.ok());
  vkc::Result<codec::PendingEncode> again =
      b->redo(*first, source, &again_rows);
  CHECK(again.ok() &&
        vgrid_test::find_row(again_rows, "wait: redo") != nullptr);
  CHECK(b->finish_frame(*first, 6, pts(6)).status().domain() ==
        vkc::Status::Code::InvalidArgument);
  const vkc::Result<codec_stream::SentFrame> redone =
      b->finish_frame(*again, 6, pts(6), &again_rows, count);
  CHECK(redone.ok() && published == 2 && rows_of(*redone, true));
  CHECK(a->finish().ok() && b->finish().ok());
  CHECK(a->coded() == 7 && b->coded() == 7);
  CHECK(vkc::test::read_bytes(by_code.stream) ==
        vkc::test::read_bytes(by_halves.stream));
  return 0;
}

// Served without replay_gop at a period of 3, a viewer joining after frame
// 0 asks for a key, which frame 1's front half reads: frame 1, coded in
// halves, is an I-frame where the encoder would code a P-frame, and the
// viewer's first record, a key; frame 2 a P-frame on it.
int keyed_halves(vkc::test::GpuContext& gpu, vol::VoxelBlockGrid& source) {
  constexpr codec::FrameType I = codec::FrameType::kIntra;
  constexpr codec::FrameType P = codec::FrameType::kInter;
  codec_stream::CodecOutputs outputs = served_on_loopback();
  outputs.serve->replay_gop = false;
  codec::EncoderConfig config;
  config.intra_period = 3;
  vkc::Result<codec_stream::CodecStream> made =
      codec_stream::CodecStream::create(gpu.device, gpu.allocator, config,
                                        session(), outputs);
  CHECK(made.ok());
  codec_stream::CodecStream& coder = made.value();
  CHECK(coder.code(source, 0, pts(0)).ok());
  const std::unique_ptr<networking::StreamClient> client = joined(coder);
  CHECK(client != nullptr && coder.server()->key_wanted() == 1);
  for (std::size_t frame = 1; frame < 3; ++frame) {
    vkc::CommandBatch front(gpu.device, gpu.allocator);
    const vkc::Result<codec::PendingEncode> pending =
        coder.record_front(front, source);
    CHECK(pending.ok());
    vkc::Result<vkc::PendingBatch> done = front.submit_async({}, {});
    CHECK(done.ok() && done->wait().ok());
    coder.front_ran(done->device_rows());
    const vkc::Result<codec_stream::SentFrame> sent =
        coder.finish_frame(*pending, frame, pts(frame));
    CHECK(sent.ok() && sent->type == (frame == 1 ? I : P));
    net::Got got;
    CHECK(net::next_within(*client, &got));
    CHECK(got.record.info.seq == frame && got.record.info.key == (frame == 1));
  }
  CHECK(coder.finish().ok());
  return 0;
}

// Serving alone, the stream codes its frames and serves each, a key at the
// default period of 1, writing no file.
int serves_alone(vkc::test::GpuContext& gpu, vol::VoxelBlockGrid& source,
                 const vkc::test::Scratch& scratch) {
  const fs::path quiet = scratch.path / "serving";
  CHECK(fs::create_directory(quiet));
  vkc::Result<codec_stream::CodecStream> made =
      codec_stream::CodecStream::create(gpu.device, gpu.allocator, {},
                                        session(), served_on_loopback());
  CHECK(made.ok());
  codec_stream::CodecStream& coder = made.value();
  const std::unique_ptr<networking::StreamClient> client = joined(coder);
  CHECK(client != nullptr);
  const std::uint16_t port = coder.server()->port();
  const fs::path before = fs::current_path();
  fs::current_path(quiet);
  std::optional<codec_stream::SentFrame> coded[3];
  net::Got got[3];
  bool arrived[3] = {};
  for (std::size_t frame = 0; frame < 3; ++frame) {
    vkc::Result<codec_stream::SentFrame> sent =
        coder.code(source, frame, pts(frame));
    if (sent.ok()) coded[frame] = std::move(sent).value();
    arrived[frame] = net::next_within(*client, &got[frame]);
  }
  const vkc::Status finished = coder.finish();
  fs::current_path(before);
  CHECK(finished.ok() && net::wait_end(*client, port));
  for (std::size_t frame = 0; frame < 3; ++frame) {
    CHECK(coded[frame].has_value() && arrived[frame]);
    const stream::RecordInfo& info = got[frame].record.info;
    CHECK(info.seq == frame && info.set == frame && info.key);
    CHECK(info.pts_ns == pts(frame));
    CHECK(got[frame].payload.size() == coded[frame]->bytes);
  }
  CHECK(fs::is_empty(quiet));
  return 0;
}

// A port in use fails create with the server's IoError, naming it, before
// the stream or the encoder is made.
int port_in_use(vkc::test::GpuContext& gpu, const vkc::test::Scratch& scratch) {
  const net::Listener taken = net::bind_loopback();
  CHECK(taken.port != 0);
  codec_stream::CodecOutputs outputs =
      served_on_loopback(scratch.path / "unserved.vrst");
  outputs.serve->port = taken.port;
  vkc::Result<codec_stream::CodecStream> made =
      codec_stream::CodecStream::create(gpu.device, gpu.allocator,
                                        unusable_config(), session(), outputs);
  CHECK(!made.ok() && made.status().domain() == vkc::Status::Code::IoError);
  CHECK(made.status().message().find(":" + std::to_string(taken.port)) !=
        std::string::npos);
  CHECK(!fs::exists(outputs.stream));
  return 0;
}

// A client that leaves mid-run, after two frames, never fails a frame: the
// rest are coded and recorded as before.
int viewer_leaves(vkc::test::GpuContext& gpu, vol::VoxelBlockGrid& source,
                  const vkc::test::Scratch& scratch) {
  const codec_stream::CodecOutputs outputs =
      served_on_loopback(scratch.path / "left.vrst");
  codec::EncoderConfig config;
  config.intra_period = 3;
  vkc::Result<codec_stream::CodecStream> made =
      codec_stream::CodecStream::create(gpu.device, gpu.allocator, config,
                                        session(), outputs);
  CHECK(made.ok());
  codec_stream::CodecStream& coder = made.value();
  std::unique_ptr<networking::StreamClient> client = joined(coder);
  CHECK(client != nullptr);
  for (std::size_t frame = 0; frame < 2; ++frame) {
    CHECK(coder.code(source, frame, pts(frame)).ok());
    net::Got got;
    CHECK(net::next_within(*client, &got) && got.record.info.seq == frame);
  }
  client.reset();
  for (std::size_t frame = 2; frame < 8; ++frame) {
    CHECK(coder.code(source, frame, pts(frame)).ok());
  }
  CHECK(coder.finish().ok() && coder.coded() == 8);
  vkc::Result<std::unique_ptr<stream::StreamReader>> opened =
      stream::StreamReader::open(outputs.stream);
  CHECK(opened.ok() && (*opened)->stats()[0].records == 8);
  return 0;
}

// A frame the file refuses is never served: with the stream's file at its
// size limit (SIGXFSZ ignored), frame 1's record cannot be appended, and the
// client, which got frame 0, sees the end once the stream is finished, with
// nothing between.
int unrecorded_unserved(vkc::test::GpuContext& gpu, vol::VoxelBlockGrid& source,
                        const vkc::test::Scratch& scratch) {
  const codec_stream::CodecOutputs outputs =
      served_on_loopback(scratch.path / "full.vrst");
  vkc::Result<codec_stream::CodecStream> made =
      codec_stream::CodecStream::create(gpu.device, gpu.allocator, {},
                                        session(), outputs);
  CHECK(made.ok());
  codec_stream::CodecStream& coder = made.value();
  const std::unique_ptr<networking::StreamClient> client = joined(coder);
  CHECK(client != nullptr);
  const std::uint16_t port = coder.server()->port();
  CHECK(coder.code(source, 0, pts(0)).ok());
  net::Got first;
  CHECK(net::next_within(*client, &first) && first.record.info.seq == 0);
  // Only this process's limit and disposition change, for one frame, and
  // nothing else is written meanwhile.
  const std::uintmax_t size = fs::file_size(outputs.stream);
  std::fflush(nullptr);
  const auto handler = ::signal(SIGXFSZ, SIG_IGN);
  CHECK(handler != SIG_ERR);
  rlimit previous{};
  CHECK(::getrlimit(RLIMIT_FSIZE, &previous) == 0);
  const rlimit limited{static_cast<rlim_t>(size), previous.rlim_max};
  CHECK(::setrlimit(RLIMIT_FSIZE, &limited) == 0);
  const vkc::Status full = coder.code(source, 1, pts(1)).status();
  CHECK(::setrlimit(RLIMIT_FSIZE, &previous) == 0);
  CHECK(::signal(SIGXFSZ, handler) != SIG_ERR);
  CHECK(full.domain() == vkc::Status::Code::IoError);
  // The writer's failure sticks, so its footer fails too; the viewers are
  // closed all the same.
  CHECK(!coder.finish().ok() && coder.server() == nullptr);
  CHECK(net::wait_end(*client, port));
  return 0;
}

// A record the server refuses fails no frame: with a key at seq 1000 already
// published to the stream's server, each record after is out of order there,
// yet every frame is coded and recorded, each refusal a Warning naming it.
int refused_publish(vkc::test::GpuContext& gpu, vol::VoxelBlockGrid& source,
                    const vkc::test::Scratch& scratch) {
  const codec_stream::CodecOutputs outputs =
      served_on_loopback(scratch.path / "refused.vrst");
  vkc::Result<codec_stream::CodecStream> made =
      codec_stream::CodecStream::create(gpu.device, gpu.allocator, {},
                                        session(), outputs);
  CHECK(made.ok());
  codec_stream::CodecStream& coder = made.value();
  // server() hands the stream's own server out as const.
  auto* const server = const_cast<networking::StreamServer*>(coder.server());
  CHECK(server != nullptr);
  const net::Bytes ahead = net::record_payload(1000, 4);
  CHECK(
      server->publish(net::record_info(1000, true), ahead.data(), ahead.size())
          .ok());
  const std::size_t before = gpu.log.warnings().size();
  for (std::size_t frame = 0; frame < 3; ++frame) {
    CHECK(coder.code(source, frame, pts(frame)).ok());
  }
  const std::vector<std::string> logged = gpu.log.warnings();
  std::vector<std::string> warnings;  // the stream's since `before`
  for (std::size_t i = before; i < logged.size(); ++i) {
    if (logged[i].find("not served") != std::string::npos) {
      warnings.push_back(logged[i]);
    }
  }
  CHECK(warnings.size() == 3);
  for (std::size_t frame = 0; frame < 3; ++frame) {
    CHECK(warnings[frame].find("frame " + std::to_string(frame) +
                               " not served") != std::string::npos);
  }
  CHECK(coder.finish().ok());
  vkc::Result<std::unique_ptr<stream::StreamReader>> opened =
      stream::StreamReader::open(outputs.stream);
  CHECK(opened.ok() && (*opened)->stats()[0].records == 3);
  return 0;
}

// At a period of 3, `source` coded at frames 0-4, served and recorded, with
// the decode and without. Without, the frames are coded, served and recorded
// as with it, the .vrst byte for byte the same; each frame has its type and
// encode rows and no decode rows, and no player grid is made.
int decode_off(vkc::test::GpuContext& gpu, vol::VoxelBlockGrid& source,
               const vkc::test::Scratch& scratch) {
  constexpr codec::FrameType I = codec::FrameType::kIntra;
  constexpr codec::FrameType P = codec::FrameType::kInter;
  const codec::FrameType types[] = {I, P, P, I, P};
  codec::EncoderConfig config;
  config.intra_period = 3;
  std::vector<std::uint8_t> recorded[2];  // without the decode, then with
  for (const bool decode : {false, true}) {
    const std::string name = decode ? "decoded" : "undecoded";
    codec_stream::CodecOutputs outputs =
        served_on_loopback(scratch.path / (name + ".vrst"));
    outputs.decode = decode;
    vkc::Result<codec_stream::CodecStream> made =
        codec_stream::CodecStream::create(gpu.device, gpu.allocator, config,
                                          session(), outputs);
    CHECK(made.ok());
    codec_stream::CodecStream& coder = made.value();
    const std::unique_ptr<networking::StreamClient> client = joined(coder);
    CHECK(client != nullptr);
    for (std::size_t frame = 0; frame < std::size(types); ++frame) {
      const vkc::Result<codec_stream::SentFrame> sent =
          coder.code(source, frame, pts(frame));
      CHECK(sent.ok() && sent->type == types[frame] && sent->bytes > 0);
      CHECK(rows_of(*sent, decode) && sent->decode.rows().empty() != decode);
      net::Got got;
      CHECK(net::next_within(*client, &got) && got.record.info.seq == frame);
    }
    CHECK(coder.finish().ok());
    CHECK((coder.player() != nullptr) == decode);
    recorded[decode ? 1 : 0] = vkc::test::read_bytes(outputs.stream);
  }
  CHECK(!recorded[0].empty() && recorded[0] == recorded[1]);
  return 0;
}

// A session with no audio writes and serves the bytes grid 0.4.0's stream did
// (01bdc64, before CodecStream took audio): its file is what a StreamWriter
// writes of the same session and frames, byte for byte, and a raw viewer's
// bytes what a ChunkEncoder frames of them, the file less its footer -- the
// bytes grid_stream_container pins. A live sender's session (codec_session's,
// on a rig) and serving (without replay_gop), at a period of 3, so P-frames
// go out too; the frames from an encoder of their own, as check_records does.
int geometry_only_bytes(vkc::test::GpuContext& gpu, vol::VoxelBlockGrid& source,
                        const vkc::test::Scratch& scratch) {
  using namespace std::chrono_literals;
  stream::Session s;
  s.writer = "fuse_orbbec";
  s.calibration = "{}";
  stream::Track track;
  track.kind = stream::TrackKind::Geometry;
  track.rate = {25, 1};
  s.tracks.push_back(track);
  codec::EncoderConfig config;
  config.intra_period = 3;
  constexpr std::size_t kFrames = 5;
  constexpr std::uint64_t kT0 = 1761800000000000000ull;  // a host clock's
  const auto frame_pts = [](std::size_t frame) {
    return kT0 + frame * std::uint64_t{40'000'000};
  };

  // What the stream wrote before: each frame a record, a key exactly when an
  // I-frame, its seq and set its index.
  const std::string expected_path =
      (scratch.path / "geometry_only_expected.vrst").string();
  vkc::Result<codec::Encoder> enc =
      codec::Encoder::create(gpu.device, gpu.allocator, config);
  vkc::Result<std::unique_ptr<stream::StreamWriter>> writer =
      stream::StreamWriter::create(expected_path, s);
  vkc::Result<stream::ChunkEncoder> framer = stream::ChunkEncoder::create(s);
  CHECK(enc.ok() && writer.ok() && framer.ok());
  net::Bytes wire = framer->session_chunk();
  std::vector<std::size_t> sent_by;  // the wire's bytes up to each frame
  std::size_t inter = 0;
  for (std::size_t frame = 0; frame < kFrames; ++frame) {
    const vkc::Result<std::vector<std::uint8_t>> bytes = enc->encode(source);
    CHECK(bytes.ok());
    const vkc::Result<codec::FrameInfo> info =
        codec::read_frame_info(bytes->data(), bytes->size());
    CHECK(info.ok());
    inter += info->type == codec::FrameType::kInter ? 1 : 0;
    stream::RecordInfo record;
    record.key = info->type == codec::FrameType::kIntra;
    record.pts_ns = frame_pts(frame);
    record.seq = frame;
    record.set = frame;
    CHECK((*writer)->append(record, bytes->data(), bytes->size()).ok());
    const vkc::Result<stream::ChunkEncoder::RecordFraming> framing =
        framer->frame(record, bytes->data(), bytes->size());
    CHECK(framing.ok());
    wire.insert(wire.end(), framing->header.begin(), framing->header.end());
    wire.insert(wire.end(), bytes->begin(), bytes->end());
    wire.resize(wire.size() + framing->padding, 0);
    sent_by.push_back(wire.size());
  }
  CHECK((*writer)->finish().ok() && inter == 3);

  codec_stream::CodecOutputs outputs =
      served_on_loopback(scratch.path / "geometry_only.vrst");
  outputs.serve->replay_gop = false;
  vkc::Result<codec_stream::CodecStream> made =
      codec_stream::CodecStream::create(gpu.device, gpu.allocator, config, s,
                                        outputs);
  CHECK(made.ok());
  codec_stream::CodecStream& coder = made.value();
  // A raw viewer, joined once it has the session's chunk, reading each frame
  // as it goes out.
  const vkc::UniqueFd viewer = net::connect_loopback(coder.server()->port());
  CHECK(viewer.valid());
  net::Bytes served;
  net::read_upto(viewer.get(), net::session_chunk(s).size(), 5s, &served);
  for (std::size_t frame = 0; frame < kFrames; ++frame) {
    CHECK(coder.code(source, frame, frame_pts(frame)).ok());
    net::read_upto(viewer.get(), sent_by[frame], 5s, &served);
  }
  CHECK(coder.finish().ok());
  bool ended = false;
  net::read_upto(viewer.get(), wire.size() + 1, 5s, &served, &ended);
  CHECK(ended && served == wire);
  const std::vector<std::uint8_t> file = vkc::test::read_bytes(outputs.stream);
  CHECK(file == vkc::test::read_bytes(expected_path));
  CHECK(file.size() > wire.size() &&
        std::equal(wire.begin(), wire.end(), file.begin()));
  return 0;
}

int gpu_main(vkc::test::GpuContext& gpu) {
  const Sphere sphere{vgrid::Vec3f(0.01f, -0.02f, 0.03f), 0.09f};
  vkc::Result<vol::VoxelBlockGrid> source = sphere_grid(gpu, sphere);
  CHECK(source.ok());
  GridShape coarser;
  coarser.voxel_size = 2.0f * kVoxel;
  vkc::Result<vol::VoxelBlockGrid> other = sphere_grid(gpu, sphere, coarser);
  CHECK(other.ok());

  vkc::test::Scratch scratch("grid_codec_stream_");
  CHECK(refused(gpu, scratch) == 0);
  CHECK(writes_nothing(gpu, *source, scratch) == 0);

  const auto outputs_in = [&scratch](const char* name) {
    const fs::path dir = scratch.path / name;
    fs::create_directory(dir);
    codec_stream::CodecOutputs outputs;
    outputs.stream = (dir / "frames.vrst").string();
    return outputs;
  };
  const codec_stream::CodecOutputs first = outputs_in("first");
  std::vector<codec_stream::SentFrame> sent;
  CHECK(code_frames(gpu, *source, *other, first, &sent) == 0);
  CHECK(check_records(gpu, *source, first.stream, sent, coarser.voxel_size) ==
        0);
  // A rerun writes the same file, byte for byte.
  const codec_stream::CodecOutputs second = outputs_in("second");
  std::vector<codec_stream::SentFrame> rerun;
  CHECK(code_frames(gpu, *source, *other, second, &rerun) == 0);
  const std::vector<std::uint8_t> bytes = vkc::test::read_bytes(first.stream);
  CHECK(!bytes.empty() && bytes == vkc::test::read_bytes(second.stream));

  CHECK(periodic(gpu, *source, *other, scratch) == 0);
  CHECK(unfinished(gpu, *source, scratch) == 0);
  CHECK(serves(gpu, *source, *other, scratch) == 0);
  CHECK(halves(gpu, *source, *other, scratch) == 0);
  CHECK(keyed_halves(gpu, *source) == 0);
  CHECK(serves_alone(gpu, *source, scratch) == 0);
  CHECK(port_in_use(gpu, scratch) == 0);
  CHECK(viewer_leaves(gpu, *source, scratch) == 0);
  CHECK(unrecorded_unserved(gpu, *source, scratch) == 0);
  CHECK(refused_publish(gpu, *source, scratch) == 0);
  CHECK(decode_off(gpu, *source, scratch) == 0);
  CHECK(geometry_only_bytes(gpu, *source, scratch) == 0);
  std::puts("grid_codec_stream: OK");
  return 0;
}

}  // namespace

int main() {
  return vkc::test::run_on_gpu(vgrid::device_requirements(), gpu_main);
}
