// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file examples/common/stream_replay.hpp
/// @brief A recording's tracks replayed together in a loop as if live, as
///        serve_stream sends them.

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <string>
#include <thread>
#include <vector>

#include "volumetric_kit/core/base/exception_status.hpp"
#include "volumetric_kit/core/base/result.hpp"
#include "volumetric_kit/grid/stream/stream_reader.hpp"

namespace vgrid_example {

namespace vgrid = volumetric_kit::grid;
namespace vkc = volumetric_kit::core;

/// @brief A recording's records, every track's, replayed pass after pass:
///        each due at its pts from the pass's start, and restamped so that
///        each track's seq keeps rising and its key pts never go back, as a
///        live sender's do. Every track moves by the same pts a pass, so the
///        tracks stay together across passes.
class ReplayLoop {
 public:
  using Clock = std::chrono::steady_clock;

  /// @brief The loop over @p reader's records: from track 0's first key
  ///        read whole, every track's, read whole, in file order; a span the
  ///        reader finds damaged is left out.
  /// @return The loop; `NotFound` for a track 0 with no key read whole; the
  ///         reader's `IoError`; `OutOfMemory`, the reader's or the loop's
  ///         own.
  static vkc::Result<ReplayLoop> create(
      const vgrid::stream::StreamReader& reader) {
    try {
      VKC_ASSIGN(vgrid::stream::Position pos, reader.key(0, 0));
      ReplayLoop loop;
      for (;;) {
        vgrid::stream::Record record;
        const vkc::Result<bool> more = reader.next(&pos, &record);
        if (!more.ok()) {
          // Only damage leaves pos past it; anything else is the loop's
          // failure, as an allocation's, which may leave pos where it was.
          const vkc::Status::Code code = more.status().domain();
          if (code != vkc::Status::Code::InvalidArgument &&
              code != vkc::Status::Code::Unsupported) {
            return more.status();
          }
          continue;  // damage: pos is past it
        }
        if (!*more) break;
        // Start at a track-0 key read whole: the P-frames after a damaged
        // first key lack theirs.
        if (!loop.records_.empty() ||
            (record.info.track == 0 && record.info.key)) {
          loop.records_.push_back(record);
        }
      }
      if (loop.records_.empty()) {
        return vkc::Status::not_found(
            "ReplayLoop: every key of track 0 is damaged");
      }
      const std::size_t tracks = reader.session().tracks.size();
      std::vector<SeqSpan> seqs(tracks);
      std::uint64_t least = loop.records_.front().info.pts_ns;
      std::uint64_t most = least;
      std::uint64_t least0 = least;  // track 0's
      std::uint64_t most0 = least;
      std::uint64_t frames = 0;  // track 0's records
      for (const vgrid::stream::Record& r : loop.records_) {
        least = std::min(least, r.info.pts_ns);
        most = std::max(most, r.info.pts_ns);
        if (r.info.track == 0) {
          least0 = std::min(least0, r.info.pts_ns);
          most0 = std::max(most0, r.info.pts_ns);
          ++frames;
        }
        SeqSpan& span = seqs[r.info.track];
        if (!span.seen) span.first = r.info.seq;
        span.seen = true;
        span.last = r.info.seq;
      }
      // One mean step of track 0 past the span; with one record of it, or
      // all at one pts, its nominal period.
      std::uint64_t step = 0;
      if (frames > 1) step = (most0 - least0) / (frames - 1);
      if (step == 0) {
        step =
            vgrid::stream::nominal_period_ns(reader.session().tracks[0].rate);
      }
      loop.first_pts_ = least;
      loop.pass_ns_ = most - least + step;
      loop.pass_seqs_.assign(tracks, 0);
      for (std::size_t t = 0; t < tracks; ++t) {
        if (seqs[t].seen) loop.pass_seqs_[t] = seqs[t].last - seqs[t].first + 1;
      }
      return loop;
    } catch (...) {
      return vkc::exception_status("ReplayLoop",
                                   vkc::Status::Code::OutOfMemory);
    }
  }

  /// @return The records each pass replays, in file order.
  const std::vector<vgrid::stream::Record>& records() const noexcept {
    return records_;
  }

  /// @return When the record at @p pts_ns is due in pass @p pass of a loop
  ///         started at @p start: its pts after the records' least, a pass's
  ///         span after the pass before.
  Clock::time_point due(Clock::time_point start, std::uint64_t pass,
                        std::uint64_t pts_ns) const {
    return start +
           std::chrono::duration_cast<Clock::duration>(
               std::chrono::nanoseconds(pass * pass_ns_ + pts_ns - first_pts_));
  }

  /// @return @p info as pass @p pass sends it: seq past its track's in the
  ///         passes before, and pts a pass's span (the records' pts span
  ///         and one mean step of track 0) later each pass, the same on
  ///         every track. Pass 0 is unchanged.
  vgrid::stream::RecordInfo restamp(vgrid::stream::RecordInfo info,
                                    std::uint64_t pass) const {
    if (info.track < pass_seqs_.size()) {
      info.seq += pass * pass_seqs_[info.track];
    }
    info.pts_ns += pass * pass_ns_;
    return info;
  }

  /// @brief Sends pass @p pass of a loop started at @p start: each record,
  ///        once @ref due, read from @p reader and @ref restamp ed, as
  ///        `send(info, payload)`. A payload that fails its CRC
  ///        (`InvalidArgument`) is not sent, nor are its track's records
  ///        after it that are not keys, which lack their reference, up to
  ///        its track's next key in the pass; pass 0 prints each such
  ///        record to stderr, as every pass meets the same.
  /// @param reader  The reader the loop was made from.
  /// @param send    Called as `vkc::Status send(const
  /// vgrid::stream::RecordInfo&,
  ///                const std::vector<std::uint8_t>&)`.
  /// @return OK; `NotFound` when no track-0 key's payload reads whole, so
  ///         the pass sends no frame; the reader's other errors; or
  ///         @p send 's.
  template <typename Send>
  vkc::Status send_pass(const vgrid::stream::StreamReader& reader,
                        Clock::time_point start, std::uint64_t pass,
                        Send&& send) const {
    std::vector<std::uint8_t> payload;
    // Per track: past a damaged payload, until the track's next key.
    std::vector<char> resync(pass_seqs_.size(), 0);
    bool sent = false;  // a track-0 record
    for (const vgrid::stream::Record& record : records_) {
      char& skipping = resync[record.info.track];
      if (skipping && !record.info.key) continue;
      skipping = 0;
      std::this_thread::sleep_until(due(start, pass, record.info.pts_ns));
      const vkc::Status read = reader.read(record, &payload);
      if (read.domain() == vkc::Status::Code::InvalidArgument) {
        if (pass == 0) {
          std::fprintf(stderr,
                       "serve_stream: %s; every pass skips track %u seq %llu "
                       "up to its track's next key\n",
                       read.message().c_str(), unsigned{record.info.track},
                       static_cast<unsigned long long>(record.info.seq));
        }
        skipping = 1;
        continue;
      }
      VKC_TRY(read);
      VKC_TRY(send(restamp(record.info, pass), payload));
      sent = sent || record.info.track == 0;
    }
    if (!sent) {
      return vkc::Status::not_found(
          "ReplayLoop: no track-0 key's payload reads whole");
    }
    return {};
  }

 private:
  // A track's first and last seq among the records, if it has one.
  struct SeqSpan {
    std::uint64_t first = 0;
    std::uint64_t last = 0;
    bool seen = false;
  };

  ReplayLoop() = default;

  std::vector<vgrid::stream::Record> records_;
  std::uint64_t first_pts_ = 0;  // the records' least pts
  std::uint64_t pass_ns_ = 0;    // a pass's pts span and one mean step
  // Per track: a pass's seq span and one, 0 for a track with no record.
  std::vector<std::uint64_t> pass_seqs_;
};

}  // namespace vgrid_example
