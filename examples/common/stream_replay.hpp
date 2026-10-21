// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file examples/common/stream_replay.hpp
/// @brief A recorded track replayed in a loop as if live, as serve_stream
///        sends it.

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <string>
#include <vector>

#include "volumetric_kit/core/base/result.hpp"
#include "volumetric_kit/recon/stream/stream_reader.hpp"

namespace vr_example {

namespace vr = volumetric_kit::recon;
namespace vkc = volumetric_kit::core;

/// @brief One track's records from its first key, replayed pass after pass:
///        each due at its pts from the pass's start, and restamped so that
///        seq keeps rising and key pts never go back, as a live sender's do.
class ReplayLoop {
 public:
  using Clock = std::chrono::steady_clock;

  /// @brief The loop over @p reader's track @p track: its records from its
  ///        first key on, in file order; a span the reader finds damaged is
  ///        left out.
  /// @return The loop; `InvalidArgument` for a track the session lacks;
  ///         `NotFound` for a track with no key; the reader's `IoError`.
  static vkc::Result<ReplayLoop> create(const vr::stream::StreamReader& reader,
                                        std::uint16_t track = 0) {
    VKC_ASSIGN(vr::stream::Position pos, reader.key(track, 0));
    ReplayLoop loop;
    for (;;) {
      vr::stream::Record record;
      const vkc::Result<bool> more = reader.next(&pos, &record);
      if (!more.ok()) {
        if (more.status().domain() == vkc::Status::Code::IoError) {
          return more.status();
        }
        continue;  // damage: pos is past it
      }
      if (!*more) break;
      if (record.info.track == track) loop.records_.push_back(record);
    }
    if (loop.records_.empty()) {
      return vkc::Status::not_found("ReplayLoop: track " +
                                    std::to_string(track) +
                                    "'s first key is damaged, and no record "
                                    "follows it");
    }
    std::uint64_t least = loop.records_.front().info.pts_ns;
    std::uint64_t most = least;
    for (const vr::stream::Record& r : loop.records_) {
      least = std::min(least, r.info.pts_ns);
      most = std::max(most, r.info.pts_ns);
    }
    // One mean step past the span; with one record, or all at one pts, the
    // track's nominal rate, else 30 a second.
    std::uint64_t step = 0;
    if (loop.records_.size() > 1) {
      step = (most - least) / (loop.records_.size() - 1);
    }
    if (step == 0) {
      const vr::stream::Rate rate = reader.session().tracks[track].rate;
      step = rate.num > 0 ? std::uint64_t{1'000'000'000} * rate.den / rate.num
                          : 33'333'333;
    }
    loop.first_pts_ = least;
    loop.pass_ns_ = most - least + step;
    loop.pass_seqs_ =
        loop.records_.back().info.seq - loop.records_.front().info.seq + 1;
    return loop;
  }

  /// @return The records each pass replays, in file order.
  const std::vector<vr::stream::Record>& records() const noexcept {
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

  /// @return @p info as pass @p pass sends it: seq past the passes' before,
  ///         and pts a pass's span (the records' pts span and one mean step)
  ///         later each pass. Pass 0 is unchanged.
  vr::stream::RecordInfo restamp(vr::stream::RecordInfo info,
                                 std::uint64_t pass) const {
    info.seq += pass * pass_seqs_;
    info.pts_ns += pass * pass_ns_;
    return info;
  }

 private:
  ReplayLoop() = default;

  std::vector<vr::stream::Record> records_;
  std::uint64_t first_pts_ = 0;  // the records' least pts
  std::uint64_t pass_ns_ = 0;    // a pass's pts span and one mean step
  std::uint64_t pass_seqs_ = 0;  // a pass's seq span and one
};

}  // namespace vr_example
