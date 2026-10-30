// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include "record_mux.hpp"

#include <limits>
#include <new>
#include <string>
#include <utility>
#include <vector>

#include "volumetric_kit/core/base/exception_status.hpp"

namespace volumetric_kit::grid::codec_stream::detail {

RecordMux::RecordMux(std::size_t tracks)
    : lanes_(tracks > 0 ? tracks - 1 : 0) {}

core::Status RecordMux::append(const stream::RecordInfo& info,
                               const std::uint8_t* data, std::size_t size,
                               const RecordSink& sink) {
  try {
    if (info.track == 0) {
      return core::Status::invalid_argument(
          "CodecStream: track 0 is the frames'; code() and finish_frame() "
          "write it");
    }
    if (info.track > lanes_.size()) {
      return core::Status::invalid_argument(
          "CodecStream: track " + std::to_string(info.track) +
          " is not in the session's " + std::to_string(lanes_.size() + 1) +
          " tracks");
    }
    const std::string track =
        "CodecStream: track " + std::to_string(info.track);
    if (!info.key) {
      return core::Status::invalid_argument(
          track + ": record seq " + std::to_string(info.seq) +
          " is not a key, as every Audio record is");
    }
    if (data == nullptr && size != 0) {
      return core::Status::invalid_argument(track + ": a null payload of " +
                                            std::to_string(size) + " bytes");
    }
    Lane& lane = lanes_[info.track - 1];
    if (lane.any && info.seq <= lane.last_seq) {
      return core::Status::invalid_argument(
          track + ": seq " + std::to_string(info.seq) +
          " does not follow seq " + std::to_string(lane.last_seq));
    }
    if (lane.any && info.pts_ns < lane.last_pts_ns) {
      return core::Status::invalid_argument(
          track + ": pts " + std::to_string(info.pts_ns) +
          " ns goes back from " + std::to_string(lane.last_pts_ns) + " ns");
    }
    const bool late = started_ && info.pts_ns < first_pts_ns_;
    if (!late) {
      lane.records.push_back(
          {info, std::vector<std::uint8_t>(data, data + size)});
    }
    lane.any = true;
    lane.last_seq = info.seq;
    lane.last_pts_ns = info.pts_ns;
    if (late) {
      ++lane.counts.dropped;
      return {};
    }
    // At or below the last frame, it goes out now: nothing held precedes it.
    if (started_) VKC_TRY(release(last_pts_ns_, sink));
    return bound(sink);
  } catch (const std::bad_alloc&) {
    return core::exception_status("CodecStream::append",
                                  core::Status::Code::OutOfMemory);
  }
}

core::Status RecordMux::frame(std::uint64_t pts_ns, const RecordSink& sink) {
  if (!started_) {
    started_ = true;
    first_pts_ns_ = pts_ns;
    for (Lane& lane : lanes_) {
      while (!lane.records.empty() &&
             lane.records.front().info.pts_ns < pts_ns) {
        lane.records.pop_front();
        ++lane.counts.dropped;
      }
    }
  }
  last_pts_ns_ = pts_ns;
  VKC_TRY(release(pts_ns, sink));
  return bound(sink);
}

core::Status RecordMux::flush(const RecordSink& sink) {
  if (started_) return release(std::numeric_limits<std::uint64_t>::max(), sink);
  for (Lane& lane : lanes_) {
    lane.counts.dropped += lane.records.size();
    lane.records.clear();
  }
  return {};
}

AppendedRecords RecordMux::counts(std::uint16_t track) const noexcept {
  if (track == 0 || track > lanes_.size()) return {};
  const Lane& lane = lanes_[track - 1];
  AppendedRecords counts = lane.counts;
  counts.held = lane.records.size();
  return counts;
}

core::Status RecordMux::release(std::uint64_t until, const RecordSink& sink) {
  for (;;) {
    Lane* next = nullptr;
    for (Lane& lane : lanes_) {
      if (lane.records.empty() || lane.records.front().info.pts_ns > until) {
        continue;
      }
      if (next == nullptr || lane.records.front().info.pts_ns <
                                 next->records.front().info.pts_ns) {
        next = &lane;
      }
    }
    if (next == nullptr) return {};
    const Held held = std::move(next->records.front());
    next->records.pop_front();
    VKC_TRY(sink(held.info, held.payload.data(), held.payload.size()));
    ++next->counts.written;
  }
}

std::uint64_t RecordMux::newest_held() const noexcept {
  std::uint64_t newest = 0;
  for (const Lane& lane : lanes_) {
    if (!lane.records.empty() && lane.records.back().info.pts_ns > newest) {
      newest = lane.records.back().info.pts_ns;
    }
  }
  return newest;
}

core::Status RecordMux::bound(const RecordSink& sink) {
  const std::uint64_t newest = newest_held();
  if (started_) {
    // Everything held is past the last frame.
    if (newest > last_pts_ns_ && newest - last_pts_ns_ > kHoldLimitNs) {
      return release(newest, sink);
    }
    return {};
  }
  for (Lane& lane : lanes_) {
    while (!lane.records.empty() &&
           newest - lane.records.front().info.pts_ns > kHoldLimitNs) {
      lane.records.pop_front();
      ++lane.counts.dropped;
    }
  }
  return {};
}

}  // namespace volumetric_kit::grid::codec_stream::detail
