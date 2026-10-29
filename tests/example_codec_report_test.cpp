// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// The codec examples' report of a stream (examples/common/codec_report.hpp).
// On the host: a row's type and host and device milliseconds from its calls'
// stage rows, its CSV line, a frame type's sizes, a GOP's bytes and bitrate, a
// frame's pts, the comparison options the codec examples share, a path that
// is the stream's or the CSV's file through a link refused, one beside them
// taken and one that cannot be checked refused, and a sender's set stamped and
// its frame's row kept from what the set step returned. On a grid holding a
// sphere, coded through codec_stream::CodecStream: a CSV refused after the
// stream is made, which takes the stream's file back, or because it is the
// stream's file under another name; a row per frame, under the source's index,
// whose type, bytes and blocks are the frame's, and each type's stage rows its
// own frames'; the CSV replaced on create and written as each row comes; a CSV
// that stops taking rows refused, the row kept and every row after refused;
// and without the decode, rows with no decode columns and no decode stage
// rows. The device part skips with no device.

#include <signal.h>
#include <sys/resource.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "codec_config.hpp"
#include "codec_fixture.hpp"
#include "codec_report.hpp"
#include "gpu_test.hpp"
#include "scratch_dir.hpp"
#include "set_log.hpp"
#include "volumetric_kit/core/base/file_io.hpp"
#include "volumetric_kit/recon/codec_stream/codec_stream.hpp"
#include "volumetric_kit/recon/sender/set_sender.hpp"
#include "volumetric_kit/recon/stream/session.hpp"
#include "volumetric_kit/recon/stream/stream_reader.hpp"

namespace codec = volumetric_kit::recon::codec;
namespace codec_stream = volumetric_kit::recon::codec_stream;
namespace eval = volumetric_kit::recon::eval;
namespace sender = volumetric_kit::recon::sender;
namespace stream = volumetric_kit::recon::stream;
namespace fs = std::filesystem;
using namespace codec_fixture;

namespace {

constexpr double kFps = 30.0;

std::uint64_t pts(std::size_t frame) {
  return vr_example::frame_pts_ns(frame, kFps);
}

int row_times() {
  // Host and device alike: the stage, its breakdowns inside it.
  // A call's host column is its busy time: its waits on the device, which a
  // call that waits inside its row holds, left out.
  vkc::StageMetrics encode;
  encode.add_cpu("codec encode", 3.5);
  encode.add_wait("codec encode", 0.5);
  encode.add_gpu("codec encode", 1.0);
  encode.add_cpu("  ..active set", 0.5);
  encode.add_gpu("  ..active set", 0.25);
  vkc::StageMetrics decode;  // no timer ran
  decode.add_cpu("codec decode", 2.0);
  decode.add_cpu("  ..rans decode", 1.5);
  const vr_example::CodecRow row = vr_example::make_codec_row(
      7, codec::FrameType::kInter, 4096, 12, encode, &decode);
  CHECK(row.frame == 7 && row.type == codec::FrameType::kInter);
  CHECK(row.bytes == 4096 && row.blocks == 12);
  CHECK(row.encode_ms == 3.0 && row.encode_device_ms == 1.0);
  CHECK(row.decode_ms == 2.0 && !row.decode_device_ms);
  // A frame not decoded has no decode columns.
  const vr_example::CodecRow encoded = vr_example::make_codec_row(
      7, codec::FrameType::kInter, 4096, 12, encode, nullptr);
  CHECK(encoded.encode_ms == 3.0 && encoded.encode_device_ms == 1.0);
  CHECK(!encoded.decode_ms && !encoded.decode_device_ms);
  // The sweep's device columns count the same way.
  CHECK(vr_example::device_ms(encode) == 1.0 && !vr_example::device_ms(decode));
  // A stage whose device work ran untimed leaves the column unset, as does
  // one the queue cannot time; an untimed breakdown under a timed stage
  // does not.
  vkc::StageMetrics partial = encode;
  partial.add_device("codec encode", vkc::DeviceTime::NotTimed);
  CHECK(!vr_example::device_ms(partial));
  vkc::StageMetrics unavailable;
  unavailable.add_cpu("codec decode", 2.0);
  unavailable.add_device("codec decode", vkc::DeviceTime::Unavailable);
  CHECK(!vr_example::device_ms(unavailable));
  vkc::StageMetrics beside = encode;  // a timed stage beside an untimed one
  beside.add_cpu("resize", 1.0);
  beside.add_device("resize", vkc::DeviceTime::NotTimed);
  CHECK(!vr_example::device_ms(beside));
  vkc::StageMetrics deep = encode;
  deep.add_device("  ..observed", vkc::DeviceTime::NotTimed);
  CHECK(vr_example::device_ms(deep) == 1.0);
  return 0;
}

int csv_lines() {
  // The header, then a line per row, its type I or P and an unset column
  // empty: a device column, or both decode columns of a frame not decoded.
  constexpr codec::FrameType I = codec::FrameType::kIntra;
  constexpr codec::FrameType P = codec::FrameType::kInter;
  const vr_example::CodecRow rows[] = {
      {3, I, 1000, 10, 1.5, 0.5, 2.0, std::nullopt},
      {7, P, 3000, 30, 2.5, std::nullopt, 1.0, std::nullopt},
      {11, P, 2000, 20, 0.25, 0.75, 3.0, std::nullopt},
      {13, P, 500, 5, 0.5, 0.25, std::nullopt, std::nullopt}};
  std::string text = vr_example::kCodecRowsCsvHeader;
  for (const vr_example::CodecRow& row : rows) {
    text += vr_example::codec_row_csv(row);
  }
  CHECK(text ==
        "frame,type,bytes,blocks,encode_host_ms,encode_device_ms,"
        "decode_host_ms,decode_device_ms\n"
        "3,I,1000,10,1.500,0.500,2.000,\n"
        "7,P,3000,30,2.500,,1.000,\n"
        "11,P,2000,20,0.250,0.750,3.000,\n"
        "13,P,500,5,0.500,0.250,,\n");
  return 0;
}

int type_sizes() {
  // Each type's frames apart: two I-frames, and between them P-frames of 200
  // down to 10 bytes, whose nearest-rank p50 and p95 are the 10th and 19th
  // smallest of 20.
  std::vector<vr_example::CodecRow> rows(2);
  rows[0].bytes = 1000;
  rows[1].bytes = 3000;
  for (std::size_t i = 1; i <= 20; ++i) {
    vr_example::CodecRow row;
    row.type = codec::FrameType::kInter;
    row.bytes = 10 * i;
    rows.insert(rows.begin() + 1, row);
  }
  const auto intra = vr_example::frame_sizes(rows, codec::FrameType::kIntra);
  CHECK(intra.count == 2 && intra.mean == 2000.0 && intra.p50 == 1000);
  CHECK(intra.p95 == 3000 && intra.max == 3000);
  const auto inter = vr_example::frame_sizes(rows, codec::FrameType::kInter);
  CHECK(inter.count == 20 && inter.mean == 105.0 && inter.p50 == 100);
  CHECK(inter.p95 == 190 && inter.max == 200);
  rows.resize(1);
  const auto none = vr_example::frame_sizes(rows, codec::FrameType::kInter);
  CHECK(none.count == 0 && none.mean == 0.0 && none.max == 0);
  return 0;
}

int gop_rates() {
  // An I-frame of 1000 bytes and P-frames of 100: a GOP of 4 holds one
  // I-frame and three P-frames, 325 bytes a frame; a GOP of 1 the I-frame.
  vr_example::FrameSizes intra;
  intra.mean = 1000.0;
  vr_example::FrameSizes inter;
  inter.mean = 100.0;
  CHECK(vr_example::gop_bytes(intra, inter, 4) == 325.0);
  CHECK(vr_example::gop_bytes(intra, inter, 1) == 1000.0);
  CHECK(vr_example::mbit_per_s(325.0, 30.0) == 0.078);
  return 0;
}

int frame_times() {
  // From frame 0 at 0, to the nearest ns.
  CHECK(vr_example::frame_pts_ns(0, 30.0) == 0);
  CHECK(vr_example::frame_pts_ns(1, 30.0) == 33'333'333);
  CHECK(vr_example::frame_pts_ns(2, 30.0) == 66'666'667);
  CHECK(vr_example::frame_pts_ns(3, 30.0) == 100'000'000);
  CHECK(vr_example::frame_pts_ns(400, 30.0) == 13'333'333'333);
  return 0;
}

int compare_options() {
  // The band's width of reach, 2 cm at least; 1 vertex in 4; F at half a
  // voxel.
  const eval::CompareOptions wide =
      vr_example::codec_compare_options(0.01f, 0.04f);
  CHECK(wide.reach == 0.04f && wide.stride == 4 &&
        wide.fscore_threshold == 0.005f);
  const eval::CompareOptions narrow =
      vr_example::codec_compare_options(0.002f, 0.008f);
  CHECK(narrow.reach == 0.02f && narrow.fscore_threshold == 0.001f);
  return 0;
}

// check_apart: a path that does not exist, or is another file, or beside an
// output that does not exist, is taken; a link to the stream or the CSV is
// refused, naming both; and a path that cannot be looked up is refused.
int apart() {
  vr_test::Scratch scratch("recon_example_codec_report_apart_");
  const std::string stream_path = (scratch.path / "frames.vrst").string();
  const std::string csv_path = (scratch.path / "rows.csv").string();
  const std::vector<std::string> outputs = {stream_path, csv_path};
  const fs::path other = scratch.path / "other.ply";
  for (const fs::path& file :
       {fs::path(stream_path), fs::path(csv_path), other}) {
    std::ofstream(file) << "x";
  }
  CHECK(vr_example::check_apart((scratch.path / "rig.ply").string(), outputs)
            .ok());
  CHECK(vr_example::check_apart(other.string(), outputs).ok());
  CHECK(vr_example::check_apart(other.string(), {}).ok());
  CHECK(vr_example::check_apart(other.string(), {"", ""}).ok());
  for (const std::string& target : outputs) {
    const fs::path link =
        scratch.path / (fs::path(target).filename().string() + ".ply");
    fs::create_symlink(target, link);
    const vkc::Status refused = vr_example::check_apart(link.string(), outputs);
    CHECK(refused.domain() == vkc::Status::Code::InvalidArgument);
    CHECK(refused.message().find(link.string()) != std::string::npos &&
          refused.message().find(target) != std::string::npos);
  }
  CHECK(vr_example::check_apart(other.string(),
                                {(scratch.path / "missing.vrst").string()})
            .ok());
  // A name past every file system's limit cannot be looked up.
  const fs::path unknowable = scratch.path / std::string(300, 'x');
  CHECK(vr_example::check_apart(unknowable.string(), outputs).domain() ==
        vkc::Status::Code::IoError);
  return 0;
}

// A sender's set into the run's figures: a set fused again whose frame, a
// P-frame, took a tail, stamped so, its frame's row kept under the set's
// index; a set not coded stamped with no type, and no row.
int sent_sets() {
  vkc::Result<vr_example::CodecReport> made =
      vr_example::CodecReport::create("", "", false, 30);
  CHECK(made.ok());
  vr_example::CodecReport& report = made.value();
  vr_example::SetLog log(nullptr);
  sender::SetSender::Sent coded;
  coded.fused.redone = true;
  coded.coded.emplace();
  coded.coded->type = codec::FrameType::kInter;
  coded.coded->bytes = 1234;
  coded.coded->blocks = 56;
  coded.coded->tail = true;
  vr_example::SetLog::Set& first = log.begin();
  CHECK(
      vr_example::report_sent(coded, 4, "example_codec_report", first, &report)
          .ok());
  CHECK(first.redone && first.tail && first.type == codec::FrameType::kInter);
  CHECK(report.rows().size() == 1 && report.rows()[0].frame == 4);
  CHECK(report.rows()[0].bytes == 1234 && report.rows()[0].blocks == 56);
  CHECK(!report.rows()[0].decode_ms);
  const sender::SetSender::Sent fused;
  vr_example::SetLog::Set& second = log.begin();
  CHECK(
      vr_example::report_sent(fused, 5, "example_codec_report", second, &report)
          .ok());
  CHECK(!second.redone && !second.tail && !second.type);
  CHECK(report.rows().size() == 1);
  return 0;
}

std::vector<std::string> lines_of(const std::string& path) {
  std::ifstream file(path);
  std::vector<std::string> lines;
  for (std::string line; std::getline(file, line);) lines.push_back(line);
  return lines;
}

stream::Session session() {
  return vr_example::codec_session("example_codec_report", {30, 1});
}

// A CSV refused once the stream is made fails the report's create, naming
// the CSV, and the stream, dropped unfinished before its first frame, takes
// its file back.
int csv_refused(vr_test::GpuContext& gpu, const vr_test::Scratch& scratch) {
  codec_stream::CodecOutputs outputs;
  outputs.stream = (scratch.path / "late.vrst").string();
  const std::string csv = (scratch.path / "missing" / "rows.csv").string();
  {
    vkc::Result<codec_stream::CodecStream> made =
        codec_stream::CodecStream::create(gpu.device, gpu.allocator, {},
                                          session(), outputs);
    CHECK(made.ok() && fs::exists(outputs.stream));
    const vkc::Result<vr_example::CodecReport> report =
        vr_example::CodecReport::create(csv, outputs.stream, true, 1);
    CHECK(!report.ok() &&
          report.status().domain() == vkc::Status::Code::IoError &&
          report.status().message().find(csv) != std::string::npos);
  }
  CHECK(!fs::exists(outputs.stream) && !fs::exists(csv));
  return 0;
}

// The CSV must not truncate the stream through another name: refused,
// naming both, the stream's new file taken back and any symlink kept. A
// corrected configuration can retry at the same stream path, with a CSV that
// does not exist yet; both files are readable after finish.
int aliased_outputs(vr_test::GpuContext& gpu, const vr_test::Scratch& scratch) {
  const fs::path dir = scratch.path / "aliases";
  CHECK(fs::create_directory(dir));
  const fs::path stream_path = dir / "frames.vrst";
  const fs::path csv_link = dir / "rows.csv";
  fs::create_symlink(stream_path.filename(), csv_link);
  const fs::path dir_link = scratch.path / "linked_outputs";
  fs::create_directory_symlink(dir, dir_link);
  std::vector<fs::path> csv_paths = {csv_link, dir_link / "frames.vrst",
                                     stream_path, dir / "." / "frames.vrst"};
  // The temporary directory's filesystem determines whether case aliases
  // exist, independently of the operating system.
  if (fs::exists(scratch.path / "ALIASES")) {
    csv_paths.push_back(dir / "FRAMES.vrst");
  } else {
    std::puts("aliased_outputs: case-sensitive filesystem; case alias skipped");
  }
  codec_stream::CodecOutputs outputs;
  outputs.stream = stream_path.string();
  outputs.decode = false;
  for (const fs::path& csv_path : csv_paths) {
    CHECK(!fs::exists(stream_path));
    {
      vkc::Result<codec_stream::CodecStream> made =
          codec_stream::CodecStream::create(gpu.device, gpu.allocator, {},
                                            session(), outputs);
      CHECK(made.ok());
      const vkc::Result<vr_example::CodecReport> report =
          vr_example::CodecReport::create(csv_path.string(), outputs.stream,
                                          false, 1);
      CHECK(!report.ok());
      CHECK(report.status().domain() == vkc::Status::Code::InvalidArgument);
      CHECK(report.status().message().find(csv_path.string()) !=
            std::string::npos);
      CHECK(report.status().message().find(outputs.stream) !=
            std::string::npos);
    }
    CHECK(!fs::exists(stream_path));
    CHECK(fs::is_symlink(csv_link));
    CHECK(fs::read_symlink(csv_link) == stream_path.filename());
    CHECK(fs::is_symlink(dir_link));
  }
  const std::string separate = (dir / "separate.csv").string();
  vkc::Result<codec_stream::CodecStream> made =
      codec_stream::CodecStream::create(gpu.device, gpu.allocator, {},
                                        session(), outputs);
  CHECK(made.ok());
  const vkc::Result<vr_example::CodecReport> report =
      vr_example::CodecReport::create(separate, outputs.stream, false, 1);
  CHECK(report.ok() && made->finish().ok());
  const vkc::Result<std::string> csv = vkc::read_file(separate);
  CHECK(csv.ok() && *csv == vr_example::kCodecRowsCsvHeader);
  CHECK(stream::StreamReader::open(outputs.stream).ok());
  return 0;
}

// At a period of 4, `source` coded at frames 0-2 and 5-9 and `other`, which
// the player refuses, at 3 and 4, each frame that went out added to a report
// over a CSV an earlier run left: the CSV replaced, its header at once and
// each row as it is added; a row per frame added, its type, bytes and blocks
// the frame's; and each type's host busy spans, summed by stage, its rows'
// milliseconds.
int rows_and_stages(vr_test::GpuContext& gpu, vol::VoxelBlockGrid& source,
                    vol::VoxelBlockGrid& other,
                    const vr_test::Scratch& scratch) {
  constexpr codec::FrameType I = codec::FrameType::kIntra;
  constexpr codec::FrameType P = codec::FrameType::kInter;
  codec_stream::CodecOutputs outputs;
  outputs.stream = (scratch.path / "periodic.vrst").string();
  const std::string csv = (scratch.path / "periodic.csv").string();
  CHECK(vkc::write_file(csv, "stale\nstale\n", vkc::WriteMode::CreateNew).ok());
  codec::EncoderConfig config;
  config.intra_period = 4;
  vkc::Result<codec_stream::CodecStream> made =
      codec_stream::CodecStream::create(gpu.device, gpu.allocator, config,
                                        session(), outputs);
  CHECK(made.ok());
  vkc::Result<vr_example::CodecReport> listed =
      vr_example::CodecReport::create(csv, outputs.stream, true, 4);
  CHECK(listed.ok());
  vr_example::CodecReport& report = listed.value();
  const std::vector<std::string> started = lines_of(csv);
  CHECK(started.size() == 1 &&
        started[0] + "\n" == vr_example::kCodecRowsCsvHeader);
  const codec::FrameType types[] = {I, P, P, I, I, I, P, P, P, I};
  std::vector<codec_stream::SentFrame> sent;
  for (std::size_t frame = 0; frame < std::size(types); ++frame) {
    const bool refused = frame == 3 || frame == 4;
    vkc::Result<codec_stream::SentFrame> s =
        made->code(refused ? other : source, frame, pts(frame));
    CHECK(s.ok() != refused);
    if (refused) continue;
    CHECK(report.add(frame, *s).ok());
    CHECK(lines_of(csv).size() == report.rows().size() + 1);
    sent.push_back(std::move(s).value());
  }
  CHECK(made->finish().ok());
  const std::size_t decoded[] = {0, 1, 2, 5, 6, 7, 8, 9};
  const std::vector<vr_example::CodecRow>& rows = report.rows();
  const std::vector<std::string> lines = lines_of(csv);
  CHECK(rows.size() == std::size(decoded) && lines.size() == rows.size() + 1);
  for (std::size_t i = 0; i < rows.size(); ++i) {
    CHECK(rows[i].frame == decoded[i] && rows[i].type == types[decoded[i]]);
    CHECK(rows[i].bytes == sent[i].bytes && rows[i].blocks == sent[i].blocks);
    CHECK(rows[i].encode_ms > 0.0 && rows[i].decode_ms &&
          *rows[i].decode_ms > 0.0);
    CHECK(lines[i + 1] + "\n" == vr_example::codec_row_csv(rows[i]));
  }
  const auto near = [](double a, double b) {
    return std::fabs(a - b) <= 1e-9 * std::fabs(b);
  };
  const auto busy = [](const vkc::StageMetrics& m) {
    return m.total_cpu_ms() - m.total_wait_ms();
  };
  for (const codec::FrameType type : {I, P}) {
    double encode = 0.0;
    double decode = 0.0;
    for (const vr_example::CodecRow& row : rows) {
      if (row.type != type) continue;
      encode += row.encode_ms;
      decode += row.decode_ms.value_or(0.0);
    }
    CHECK(encode > 0.0 && decode > 0.0);
    CHECK(near(busy(report.encode_stages(type)), encode));
    CHECK(near(busy(report.decode_stages(type)), decode));
  }
  report.print(kFps, 1);
  return 0;
}

// A CSV that stops taking rows after frame 0's -- the file size limit
// reached, SIGXFSZ ignored -- refuses frame 1's row once it is added, IoError
// naming the CSV, the row kept; the failure sticks, so every row after is
// refused too.
int csv_fails(vr_test::GpuContext& gpu, vol::VoxelBlockGrid& source,
              const vr_test::Scratch& scratch) {
  const std::string csv = (scratch.path / "limited.csv").string();
  vkc::Result<codec_stream::CodecStream> made =
      codec_stream::CodecStream::create(gpu.device, gpu.allocator, {},
                                        session());
  vkc::Result<vr_example::CodecReport> listed =
      vr_example::CodecReport::create(csv, "", true, 1);
  CHECK(made.ok() && listed.ok());
  vr_example::CodecReport& report = listed.value();
  vkc::Result<codec_stream::SentFrame> first = made->code(source, 0, pts(0));
  CHECK(first.ok() && report.add(0, *first).ok());
  const std::uintmax_t size = fs::file_size(csv);
  std::vector<codec_stream::SentFrame> sent;
  for (std::size_t frame = 1; frame < 4; ++frame) {
    vkc::Result<codec_stream::SentFrame> s =
        made->code(source, frame, pts(frame));
    CHECK(s.ok());
    sent.push_back(std::move(s).value());
  }
  // Only this process's limit and disposition change, for one row, and
  // nothing else is written meanwhile.
  std::fflush(nullptr);
  const auto handler = ::signal(SIGXFSZ, SIG_IGN);
  CHECK(handler != SIG_ERR);
  rlimit previous{};
  CHECK(::getrlimit(RLIMIT_FSIZE, &previous) == 0);
  const rlimit limited{static_cast<rlim_t>(size), previous.rlim_max};
  CHECK(::setrlimit(RLIMIT_FSIZE, &limited) == 0);
  const vkc::Status refused = report.add(1, sent[0]);
  CHECK(::setrlimit(RLIMIT_FSIZE, &previous) == 0);
  CHECK(::signal(SIGXFSZ, handler) != SIG_ERR);
  CHECK(refused.domain() == vkc::Status::Code::IoError &&
        refused.message().find(csv) != std::string::npos);
  for (std::size_t frame = 2; frame < 4; ++frame) {
    CHECK(report.add(frame, sent[frame - 1]).domain() ==
          vkc::Status::Code::IoError);
  }
  CHECK(report.rows().size() == 4 && fs::file_size(csv) == size);
  return 0;
}

// Without the decode, each row has its frame's type and encode and no
// decode, as its CSV line has, and no type has decode stage rows.
int decode_off(vr_test::GpuContext& gpu, vol::VoxelBlockGrid& source,
               const vr_test::Scratch& scratch) {
  constexpr codec::FrameType I = codec::FrameType::kIntra;
  constexpr codec::FrameType P = codec::FrameType::kInter;
  const codec::FrameType types[] = {I, P, P, I, P};
  codec::EncoderConfig config;
  config.intra_period = 3;
  codec_stream::CodecOutputs outputs;
  outputs.decode = false;
  const std::string csv = (scratch.path / "undecoded.csv").string();
  vkc::Result<codec_stream::CodecStream> made =
      codec_stream::CodecStream::create(gpu.device, gpu.allocator, config,
                                        session(), outputs);
  vkc::Result<vr_example::CodecReport> listed =
      vr_example::CodecReport::create(csv, "", false, 3);
  CHECK(made.ok() && listed.ok());
  vr_example::CodecReport& report = listed.value();
  for (std::size_t frame = 0; frame < std::size(types); ++frame) {
    vkc::Result<codec_stream::SentFrame> s =
        made->code(source, frame, pts(frame));
    CHECK(s.ok() && report.add(frame, *s).ok());
  }
  CHECK(made->finish().ok());
  const std::vector<vr_example::CodecRow>& rows = report.rows();
  const std::vector<std::string> lines = lines_of(csv);
  CHECK(rows.size() == std::size(types) && lines.size() == rows.size() + 1);
  for (std::size_t i = 0; i < rows.size(); ++i) {
    CHECK(rows[i].frame == i && rows[i].type == types[i]);
    CHECK(rows[i].bytes > 0 && rows[i].encode_ms > 0.0);
    CHECK(!rows[i].decode_ms && !rows[i].decode_device_ms);
    CHECK(lines[i + 1] + "\n" == vr_example::codec_row_csv(rows[i]));
  }
  for (const codec::FrameType type : {I, P}) {
    CHECK(!report.encode_stages(type).rows().empty());
    CHECK(report.decode_stages(type).rows().empty());
  }
  report.print(kFps, 1);
  return 0;
}

int gpu_main(vr_test::GpuContext& gpu) {
  const Sphere sphere{vr::Vec3f(0.01f, -0.02f, 0.03f), 0.09f};
  vkc::Result<vol::VoxelBlockGrid> source = sphere_grid(gpu, sphere);
  CHECK(source.ok());
  GridShape coarser;
  coarser.voxel_size = 2.0f * kVoxel;
  vkc::Result<vol::VoxelBlockGrid> other = sphere_grid(gpu, sphere, coarser);
  CHECK(other.ok());

  vr_test::Scratch scratch("recon_example_codec_report_");
  CHECK(csv_refused(gpu, scratch) == 0);
  CHECK(aliased_outputs(gpu, scratch) == 0);
  CHECK(rows_and_stages(gpu, *source, *other, scratch) == 0);
  CHECK(csv_fails(gpu, *source, scratch) == 0);
  CHECK(decode_off(gpu, *source, scratch) == 0);
  std::puts("example_codec_report: OK");
  return 0;
}

}  // namespace

int main() {
  CHECK(row_times() == 0);
  CHECK(csv_lines() == 0);
  CHECK(type_sizes() == 0);
  CHECK(gop_rates() == 0);
  CHECK(frame_times() == 0);
  CHECK(compare_options() == 0);
  CHECK(apart() == 0);
  CHECK(sent_sets() == 0);
  return vr_test::run_on_gpu(gpu_main);
}
