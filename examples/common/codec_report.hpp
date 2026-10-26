// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file examples/common/codec_report.hpp
/// @brief What the codec examples report of a stream: a row per coded frame,
///        built from what `sender::CodecStream::code` and
///        `sender::SetSender::send` return, which the report sums by frame
///        type and the CSV lists, and the comparisons of a decoded surface.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <ios>
#include <optional>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#include "fuse_frame.hpp"
#include "set_log.hpp"
#include "stage_table.hpp"
#include "volumetric_kit/core/base/check.hpp"
#include "volumetric_kit/core/base/result.hpp"
#include "volumetric_kit/core/base/stage_metrics.hpp"
#include "volumetric_kit/recon/codec/codec_params.hpp"
#include "volumetric_kit/recon/eval/mesh_distance.hpp"
#include "volumetric_kit/recon/sender/codec_stream.hpp"
#include "volumetric_kit/recon/sender/set_sender.hpp"

namespace vr_example {

namespace vr = volumetric_kit::recon;
namespace vkc = volumetric_kit::core;

using Bytes = std::vector<std::uint8_t>;

/// Raw tsdf + weight per block: what a frame compresses.
inline constexpr double kRawBytesPerBlock = 512.0 * 2.0 * sizeof(float);

/// @return A row's host milliseconds, or 0 if @p m has no row of that name.
inline double row_ms(const vkc::StageMetrics& m, const char* name) {
  const vkc::StageRow* row = find_row(m, name);
  return row != nullptr ? row->cpu_ms : 0.0;
}

/// @return A row's device milliseconds, or 0 if @p m has no row of that
///         name or it was not timed on the device.
inline double row_device_ms(const vkc::StageMetrics& m, const char* name) {
  const vkc::StageRow* row = find_row(m, name);
  return row != nullptr && row->device == vkc::DeviceTime::Measured
             ? row->gpu_ms
             : 0.0;
}

/// @brief Print an accuracy / coverage / F-score comparison, in mm.
inline void print_comparison(const vr::eval::MeshComparison& c, float voxel) {
  auto line = [voxel](const char* name, const vr::eval::DistanceStats& s) {
    std::printf(
        "    %-9s mean %6.3f  rms %6.3f  p95 %6.3f  max %6.3f mm "
        "(%.3f voxels rms), %zu of %zu beyond reach\n",
        name, s.mean * 1e3, s.rms * 1e3, s.p95 * 1e3, s.max * 1e3,
        s.rms / double(voxel), s.beyond_reach, s.count);
  };
  line("accuracy", c.accuracy);
  line("coverage", c.coverage);
  if (c.fscore.threshold > 0.0f) {
    std::printf("    F-score   %.4f at %.1f mm (precision %.4f, recall %.4f)\n",
                c.fscore.f, double(c.fscore.threshold) * 1e3,
                c.fscore.precision, c.fscore.recall);
  }
}

/// @return How the codec examples compare a decoded surface with its source,
///         inside the reconstruction's own resolution so the figures move
///         with the codec: a band's width of reach (@p trunc, at least 2 cm),
///         about 1 vertex in 4 both ways, and an F-score at half a voxel
///         (@p voxel).
inline vr::eval::CompareOptions codec_compare_options(float voxel,
                                                      float trunc) {
  vr::eval::CompareOptions compare;
  compare.reach = std::max(trunc, 0.02f);
  compare.stride = 4;
  compare.fscore_threshold = 0.5f * voxel;
  return compare;
}

/// @return @p type's name in a stream's rows: "I" or "P".
inline const char* frame_type_name(vr::codec::FrameType type) {
  return type == vr::codec::FrameType::kIntra ? "I" : "P";
}

/// @brief One coded frame: its type and size, and what encoding and decoding
///        it cost. A device column is unset where its call timed no device
///        span, and both decode columns where the frame was not decoded.
struct CodecRow {
  std::size_t frame = 0;  ///< The source's index of the frame coded.
  /// An I-frame, or a P-frame on the frame coded before.
  vr::codec::FrameType type = vr::codec::FrameType::kIntra;
  std::size_t bytes = 0;     ///< The compressed frame's size.
  std::uint32_t blocks = 0;  ///< The blocks it carries.
  /// Encode, host busy: its waits on the device left out, wherever they ran.
  double encode_ms = 0.0;
  std::optional<double> encode_device_ms;  ///< Encode, device.
  std::optional<double> decode_ms;         ///< Decode, host busy.
  std::optional<double> decode_device_ms;  ///< Decode, device.
};

/// @return A codec call's device time, its stages' summed
///         (`StageMetrics::total_gpu_ms`: a breakdown's is inside its
///         stage's); unset when no stage was timed, or when one with a device
///         half was not, so a partial sum never passes for the call's.
inline std::optional<double> device_ms(const vkc::StageMetrics& m) {
  bool measured = false;
  for (const vkc::StageRow& row : m.rows()) {
    if (row.device == vkc::DeviceTime::None ||
        std::strncmp(row.name, vkc::StageMetrics::kBreakdownPrefix,
                     std::strlen(vkc::StageMetrics::kBreakdownPrefix)) == 0) {
      continue;
    }
    if (row.device != vkc::DeviceTime::Measured) return std::nullopt;
    measured = true;
  }
  if (!measured) return std::nullopt;
  return m.total_gpu_ms();
}

/// @brief A frame's row from the stage rows its encode and decode filled:
///        each call's host busy span (`StageMetrics::total_cpu_ms` less
///        `total_wait_ms`, its stages without the breakdowns inside them or
///        their waits on the device: one meaning whether a call waits inside
///        its rows, as `Encoder::encode` does, or a sender waits outside
///        them) and its @ref device_ms.
/// @param decode  The decode's rows, or null for a frame not decoded.
inline CodecRow make_codec_row(std::size_t frame, vr::codec::FrameType type,
                               std::size_t bytes, std::uint32_t blocks,
                               const vkc::StageMetrics& encode,
                               const vkc::StageMetrics* decode) {
  CodecRow row;
  row.frame = frame;
  row.type = type;
  row.bytes = bytes;
  row.blocks = blocks;
  row.encode_ms = encode.total_cpu_ms() - encode.total_wait_ms();
  row.encode_device_ms = device_ms(encode);
  if (decode != nullptr) {
    row.decode_ms = decode->total_cpu_ms() - decode->total_wait_ms();
    row.decode_device_ms = device_ms(*decode);
  }
  return row;
}

/// The first line of a stream's CSV, naming @ref codec_row_csv's columns.
inline constexpr const char kCodecRowsCsvHeader[] =
    "frame,type,bytes,blocks,encode_host_ms,encode_device_ms,decode_host_ms,"
    "decode_device_ms\n";

/// @return @p row as a line under @ref kCodecRowsCsvHeader: its type as
///         @ref frame_type_name names it, milliseconds to the microsecond,
///         and an unset column empty.
inline std::string codec_row_csv(const CodecRow& row) {
  const auto ms = [](const std::optional<double>& value) {
    char text[32] = "";
    if (value) std::snprintf(text, sizeof text, "%.3f", *value);
    return std::string(text);
  };
  char text[64];
  std::snprintf(text, sizeof text, "%zu,%s,%zu,%u,", row.frame,
                frame_type_name(row.type), row.bytes, row.blocks);
  return text + ms(row.encode_ms) + "," + ms(row.encode_device_ms) + "," +
         ms(row.decode_ms) + "," + ms(row.decode_device_ms) + "\n";
}

/// @brief The sizes of a stream's frames of one type, in bytes.
struct FrameSizes {
  std::size_t count = 0;  ///< Frames of the type.
  double mean = 0.0;
  std::size_t p50 = 0;  ///< The nearest-rank median.
  std::size_t p95 = 0;  ///< The nearest-rank 95th percentile.
  std::size_t max = 0;
};

/// @return The sizes of @p rows' frames of @p type; every field 0 when there
///         is none.
inline FrameSizes frame_sizes(const std::vector<CodecRow>& rows,
                              vr::codec::FrameType type) {
  std::vector<std::size_t> sizes;
  double total = 0.0;
  for (const CodecRow& row : rows) {
    if (row.type != type) continue;
    sizes.push_back(row.bytes);
    total += double(row.bytes);
  }
  FrameSizes s;
  s.count = sizes.size();
  if (sizes.empty()) return s;
  std::sort(sizes.begin(), sizes.end());
  s.mean = total / double(s.count);
  s.p50 = nearest_rank(sizes, 50);
  s.p95 = nearest_rank(sizes, 95);
  s.max = sizes.back();
  return s;
}

/// @return A GOP's bytes a frame at intra period @p period: an I-frame and
///         @p period - 1 P-frames, each at its type's mean (@p intra's and
///         @p inter's), the rate a long stream settles at whatever I-frames
///         its own run forced. @pre @p period is at least 1.
inline double gop_bytes(const FrameSizes& intra, const FrameSizes& inter,
                        std::uint32_t period) {
  VKC_CHECK(period >= 1, "gop_bytes needs a period of at least 1");
  return (intra.mean + double(period - 1) * inter.mean) / double(period);
}

/// @return @p bytes_per_frame at @p frames_per_s, in Mbit/s.
inline double mbit_per_s(double bytes_per_frame, double frames_per_s) {
  return bytes_per_frame * 8.0 * frames_per_s / 1e6;
}

/// The rate a source with no time of its own is coded at, frames a second:
/// Replica's frames and a mesh sequence's numbers.
inline constexpr std::uint32_t kNominalFps = 30;

/// @return Source frame @p frame's time at @p fps, frame 0 at 0, in ns
///         rounded to the nearest. @pre @p fps is positive.
inline std::uint64_t frame_pts_ns(std::size_t frame, double fps) {
  VKC_CHECK(fps > 0.0, "frame_pts_ns needs a positive frame rate");
  return std::uint64_t(std::llround(double(frame) * 1e9 / fps));
}

/// @brief Refuse to write @p path when it is @p file under another name,
///        through a link or a case alias: the core compares paths as text.
/// @return OK when nothing is at @p path or at @p file, or they are two
///         files; InvalidArgument naming both when they are one; IoError
///         when that cannot be told, as for a name too long to look up.
inline vkc::Status check_not_same_file(const std::string& path,
                                       const std::string& file) {
  // Both checked for existence first: libc++'s `equivalent` reports an
  // error when one is missing, libstdc++'s returns false.
  std::error_code error;
  const bool same = std::filesystem::exists(path, error) &&
                    std::filesystem::exists(file, error) &&
                    std::filesystem::equivalent(path, file, error);
  if (error) {
    return vkc::Status::io_error("cannot tell whether " + path + " is " + file);
  }
  if (same) {
    return vkc::Status::invalid_argument(path + " is the same file as " + file);
  }
  return {};
}

/// @brief Refuse to write @p path over any of @p outputs (the stream's file,
///        the CSV), once they exist (@ref check_not_same_file); an empty one
///        is skipped.
/// @return OK, or the first refusal.
inline vkc::Status check_apart(const std::string& path,
                               const std::vector<std::string>& outputs) {
  for (const std::string& output : outputs) {
    if (!output.empty()) VKC_TRY(check_not_same_file(path, output));
  }
  return {};
}

/// @brief A stream's report: a row per frame coded and decoded, each type's
///        stage rows summed, the CSV written as each row comes, and the
///        summary the codec examples print.
class CodecReport {
 public:
  /// @brief Open the CSV, its header flushed, once the stream's file is
  ///        made: making it resolves dangling symlinks and case aliases, so
  ///        the CSV is checked not to be that file before it is truncated.
  /// @param csv           A file of @ref kCodecRowsCsvHeader, then the
  ///                      @ref codec_row_csv of each row; replaced if it
  ///                      exists; empty writes none. The core writes whole
  ///                      files only, so this one, extended a row at a
  ///                      time, is the report's own.
  /// @param stream        The stream's file, or empty.
  /// @param decode        Whether the stream decodes its frames: the rows'
  ///                      decode columns and the decode's stage rows.
  /// @param intra_period  The encoder's, for a GOP's bitrate.
  /// @return The report; @ref check_not_same_file's error when the CSV is the
  ///         stream's file; or IoError naming a CSV that cannot be written.
  static vkc::Result<CodecReport> create(const std::string& csv,
                                         const std::string& stream, bool decode,
                                         std::uint32_t intra_period) {
    CodecReport report(csv, decode, intra_period);
    if (csv.empty()) return report;
    const vkc::Status apart = check_not_same_file(csv, stream);
    if (!apart.ok()) return apart.with_context("CodecReport: CSV");
    report.csv_.open(csv, std::ios::binary | std::ios::trunc);
    report.csv_ << kCodecRowsCsvHeader << std::flush;
    if (!report.csv_) return report.csv_error();
    return report;
  }

  /// @brief Keep frame @p frame's row, @p sent as it went out, and its stage
  ///        rows in its type's, then write the row to the CSV, flushed.
  /// @return OK; or IoError naming the CSV, the row kept, and for every row
  ///         after.
  vkc::Status add(std::uint64_t frame, const vr::sender::SentFrame& sent) {
    const auto t = static_cast<std::size_t>(sent.type);
    encode_rows_[t].merge(sent.encode);
    if (decode_) decode_rows_[t].merge(sent.decode);
    grew_ = grew_ || find_row(sent.decode, "  ..grow") != nullptr;
    rows_.push_back(make_codec_row(static_cast<std::size_t>(frame), sent.type,
                                   sent.bytes, sent.blocks, sent.encode,
                                   decode_ ? &sent.decode : nullptr));
    if (csv_.is_open()) {
      csv_ << codec_row_csv(rows_.back()) << std::flush;
      if (!csv_) return csv_error();
    }
    return {};
  }

  /// @return A row per frame added, in coding order.
  const std::vector<CodecRow>& rows() const noexcept { return rows_; }

  /// @return The encode stage rows of every frame of @p type, summed: what
  ///         @ref print prints, a frame of that type each.
  const vkc::StageMetrics& encode_stages(vr::codec::FrameType type) const {
    return encode_rows_[static_cast<std::size_t>(type)];
  }

  /// @return As @ref encode_stages, the decode's: none when the stream does
  ///         not decode.
  const vkc::StageMetrics& decode_stages(vr::codec::FrameType type) const {
    return decode_rows_[static_cast<std::size_t>(type)];
  }

  /// @brief Print what the stream cost: its size and bitrate, a GOP's
  ///        bitrate at the intra period, and for each frame type its sizes
  ///        (@ref frame_sizes) and where its encode time went, and its
  ///        decode's when the stream decodes.
  ///        @pre rows() is not empty.
  /// @param fps    The source's frame rate.
  /// @param every  Source frames per coded frame; 0 when only one frame was
  ///               coded, which has no bitrate.
  void print(double fps, int every) const {
    const auto frames = double(rows_.size());
    double bytes = 0.0;
    double blocks = 0.0;
    for (const CodecRow& row : rows_) {
      bytes += double(row.bytes);
      blocks += double(row.blocks);
    }
    const double per_frame = bytes / frames;
    const double per_block = blocks > 0 ? bytes / blocks : 0.0;
    const double coded_fps = every > 0 ? fps / double(every) : 0.0;
    std::printf("stream: %zu frames coded%s\n", rows_.size(),
                grew_ ? " (the player grid grew)" : "");
    std::printf(
        "  %.0f bytes per frame, %.0f blocks; %.1f bytes per block, %.0fx "
        "under raw",
        per_frame, blocks / frames, per_block,
        per_block > 0 ? kRawBytesPerBlock / per_block : 0.0);
    if (every > 0) {
      std::printf("; %.2f Mbit/s at %.1f coded frames/s (1 in %d at %.0f fps)",
                  mbit_per_s(per_frame, coded_fps), coded_fps, every, fps);
    }
    std::printf("\n");
    const vr::codec::FrameType types[] = {vr::codec::FrameType::kIntra,
                                          vr::codec::FrameType::kInter};
    const FrameSizes sizes[] = {frame_sizes(rows_, types[0]),
                                frame_sizes(rows_, types[1])};
    if (every > 0 && intra_period_ > 1 && sizes[0].count > 0 &&
        sizes[1].count > 0) {
      std::printf(
          "  GOP of %u: %.2f Mbit/s, an I-frame and %u P-frames at their "
          "means\n",
          intra_period_,
          mbit_per_s(gop_bytes(sizes[0], sizes[1], intra_period_), coded_fps),
          intra_period_ - 1);
    }
    for (std::size_t t = 0; t < 2; ++t) {
      const FrameSizes& s = sizes[t];
      if (s.count == 0) continue;
      const std::string name =
          std::string(frame_type_name(types[t])) + "-frame";
      std::printf("  %zu %ss: bytes mean %.0f, p50 %zu, p95 %zu, max %zu\n",
                  s.count, name.c_str(), s.mean, s.p50, s.p95, s.max);
      const vkc::StageMetrics& encode = encode_stages(types[t]);
      const vkc::StageMetrics& decode = decode_stages(types[t]);
      print_stage_rows("encode, per " + name, encode, s.count, 4);
      if (decode_) print_stage_rows("decode, per " + name, decode, s.count, 4);
      // The coders' device time: their host halves are only their
      // recording, as each runs in its caller's batch.
      const double enc =
          row_device_ms(encode, "  ..rans encode") / double(s.count);
      const double dec =
          row_device_ms(decode, "  ..rans decode") / double(s.count);
      std::printf("    rANS: %.2f ms encode", enc);
      if (decode_) std::printf(", %.2f ms decode", dec);
      std::printf(" on the device per %s", name.c_str());
      if (every > 0) {
        // Against the source's interval: what coding every frame live must
        // fit. A standalone mesh or final-grid snapshot has no measured
        // frame rate.
        const double budget = 1e3 / fps;
        std::printf(" (%.0f%%", 100.0 * enc / budget);
        if (decode_) std::printf(" / %.0f%%", 100.0 * dec / budget);
        std::printf(" of a %.1f ms source frame interval)", budget);
      }
      std::printf("\n");
    }
  }

 private:
  CodecReport(std::string csv, bool decode, std::uint32_t intra_period)
      : csv_path_(std::move(csv)),
        decode_(decode),
        intra_period_(intra_period) {}

  vkc::Status csv_error() const {
    return vkc::Status::io_error("CodecReport: cannot write the CSV " +
                                 csv_path_);
  }

  std::string csv_path_;
  std::ofstream csv_;  // open when csv_path_ names a file
  bool decode_;
  std::uint32_t intra_period_;
  // Every frame's, summed by stage, by frame type: I-frames', then P-frames'.
  vkc::StageMetrics encode_rows_[2];
  vkc::StageMetrics decode_rows_[2];
  std::vector<CodecRow> rows_;
  bool grew_ = false;  // a decode grew the player grid
};

/// @brief What a sender's set did, into the run's figures: a free pass's
///        refusal on stderr as "<program>: free blocks: <why>", what the
///        fusion did to the grid (@ref print_fuse_report), the set's row
///        stamped -- fused again, its frame's type and tail -- and its
///        frame's row added to @p report, when given.
/// @return OK, or @ref CodecReport::add's error.
inline vkc::Status report_sent(const vr::sender::SetSender::Sent& sent,
                               std::uint64_t index, const char* program,
                               SetLog::Set& set, CodecReport* report) {
  if (!sent.freed.ok()) {
    std::fprintf(stderr, "%s: free blocks: %s\n", program,
                 sent.freed.message().c_str());
  }
  print_fuse_report(sent.fused);
  set.redone = sent.fused.redone;
  if (!sent.coded) return {};
  set.type = sent.coded->type;
  set.tail = sent.coded->tail;
  return report != nullptr ? report->add(index, *sent.coded) : vkc::Status{};
}

}  // namespace vr_example
