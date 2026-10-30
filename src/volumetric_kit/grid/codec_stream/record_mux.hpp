// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

// The pts mux of CodecStream::append: what holds the records of the tracks
// after the frames', and lets each go once a frame at or past its pts has
// gone out. No file, socket or device: what goes out goes to a sink, which
// writes and serves it. Internal: StreamCoder owns one.

#include <cstddef>
#include <cstdint>
#include <deque>
#include <functional>
#include <vector>

#include "volumetric_kit/core/base/result.hpp"
#include "volumetric_kit/grid/codec_stream/codec_stream.hpp"
#include "volumetric_kit/grid/stream/session.hpp"

namespace volumetric_kit::grid::codec_stream::detail {

// Where a record goes once it goes out: written and served, in this order.
using RecordSink =
    std::function<core::Status(const stream::RecordInfo& info,
                               const std::uint8_t* data, std::size_t size)>;

class RecordMux {
 public:
  // For a session of `tracks` tracks, track 0 the frames'.
  explicit RecordMux(std::size_t tracks);

  // As CodecStream::append, on a stream not finished: refused with nothing
  // changed, or held, dropped or sent to `sink` with what it releases.
  core::Status append(const stream::RecordInfo& info, const std::uint8_t* data,
                      std::size_t size, const RecordSink& sink);

  // A frame at `pts_ns` went out: on the first, drop what is below it; then
  // send what it releases, and everything held if the newest held is more
  // than kHoldLimitNs past it.
  core::Status frame(std::uint64_t pts_ns, const RecordSink& sink);

  // As CodecStream::finish: send everything held, in pts order, or drop it
  // before any frame.
  core::Status flush(const RecordSink& sink);

  // As CodecStream::appended.
  AppendedRecords counts(std::uint16_t track) const noexcept;

 private:
  struct Held {
    stream::RecordInfo info;
    std::vector<std::uint8_t> payload;
  };
  // A track after the frames': its held records in its order, its last
  // record taken, and its counts but `held`, which is `records.size()`.
  struct Lane {
    std::deque<Held> records;
    bool any = false;
    std::uint64_t last_seq = 0;
    std::uint64_t last_pts_ns = 0;
    AppendedRecords counts;
  };

  // Send every held record at or below `until`, the least pts first across
  // tracks, the lower track first at equal pts. Each leaves the lane before
  // it is sent, so a sink's failure, returned, loses that record alone.
  core::Status release(std::uint64_t until, const RecordSink& sink);
  // The greatest pts held, or 0 with none held.
  std::uint64_t newest_held() const noexcept;
  // Bound what is held: before the first frame, drop what is more than
  // kHoldLimitNs below the newest; after it, send everything once the
  // newest is more than kHoldLimitNs past the last frame.
  core::Status bound(const RecordSink& sink);

  std::vector<Lane> lanes_;         // by track - 1
  bool started_ = false;            // a frame went out
  std::uint64_t first_pts_ns_ = 0;  // the first frame's, once started_
  std::uint64_t last_pts_ns_ = 0;   // the last frame's, once started_
};

}  // namespace volumetric_kit::grid::codec_stream::detail
