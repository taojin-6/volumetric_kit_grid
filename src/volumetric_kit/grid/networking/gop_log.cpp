// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include "gop_log.hpp"

#include <algorithm>
#include <utility>

namespace volumetric_kit::grid::networking::detail {

GopLog::GopLog(std::uint64_t max_gop_bytes, bool replay) noexcept
    : max_gop_bytes_(max_gop_bytes), replay_(replay) {}

void GopLog::join(Cursor* cursor) {
  if (replay_) return;
  *cursor = {};
  viewers_.push_back(cursor);
  ++key_wanted_;
}

void GopLog::leave(const Cursor* cursor) noexcept {
  viewers_.erase(std::remove(viewers_.begin(), viewers_.end(), cursor),
                 viewers_.end());
}

bool GopLog::append(Chunk chunk, bool key) {
  return replay_ ? append_replay(std::move(chunk), key)
                 : append_on_demand(std::move(chunk), key);
}

bool GopLog::append_replay(Chunk chunk, bool key) {
  if (key) {
    ++gop_;
    chunks_.clear();
    bytes_ = 0;
    open_ = true;
  }
  if (!open_) return false;
  const std::uint64_t size = chunk->size();
  if (size > max_gop_bytes_ - bytes_) {
    break_gop(key);
    return false;
  }
  try {
    chunks_.push_back(std::move(chunk));
  } catch (...) {
    break_gop(key);
    throw;
  }
  bytes_ += size;
  return true;
}

bool GopLog::append_on_demand(Chunk chunk, bool key) {
  const std::uint64_t size = chunk->size();
  if (key && size <= max_gop_bytes_) {
    // Newest key wins: every viewer goes on from this key, the waiting ones
    // let in, so what came before goes and none falls behind by it.
    first_ = head();
    held_.clear();
    if (viewers_.empty()) return false;
    try {
      held_.push_back({std::move(chunk), kept_});
    } catch (...) {
      break_gop(true);
      throw;
    }
    kept_ += size;
    for (Cursor* viewer : viewers_) *viewer = {0, first_, false};
    return true;
  }
  // Every viewer in has at most max_gop_bytes behind it, so none of these
  // differences wraps; one chunk past the limit alone, a key too, takes
  // every viewer.
  for (Cursor* viewer : viewers_) {
    if (!viewer->waiting && size > max_gop_bytes_ - behind(*viewer)) {
      fall_back(viewer);
    }
  }
  trim();
  if (!open()) return false;
  try {
    held_.push_back({std::move(chunk), kept_});
  } catch (...) {
    break_gop(key);
    throw;
  }
  kept_ += size;
  return true;
}

void GopLog::break_gop(bool key) noexcept {
  if (replay_) {
    chunks_.clear();
    bytes_ = 0;
    open_ = false;
    return;
  }
  for (Cursor* viewer : viewers_) {
    if (!viewer->waiting || key) fall_back(viewer);
  }
  trim();
}

Chunk GopLog::next(Cursor* cursor) const noexcept {
  if (!replay_) {
    if (cursor->waiting || cursor->next >= head()) return nullptr;
    return held_[cursor->next++ - first_].chunk;
  }
  if (cursor->gop != gop_) *cursor = {gop_, 0, false};
  if (cursor->next >= chunks_.size()) return nullptr;
  return chunks_[cursor->next++];
}

bool GopLog::open() const noexcept {
  if (replay_) return open_;
  return std::any_of(viewers_.begin(), viewers_.end(),
                     [](const Cursor* viewer) { return !viewer->waiting; });
}

bool GopLog::joined() const noexcept { return !viewers_.empty(); }

std::uint64_t GopLog::key_wanted() const noexcept { return key_wanted_; }

std::uint64_t GopLog::bytes() const noexcept {
  if (replay_) return bytes_;
  return held_.empty() ? 0 : kept_ - held_.front().start;
}

std::uint64_t GopLog::head() const noexcept { return first_ + held_.size(); }

std::uint64_t GopLog::behind(const Cursor& cursor) const noexcept {
  if (cursor.next >= head()) return 0;
  return kept_ - held_[cursor.next - first_].start;
}

void GopLog::trim() noexcept {
  std::uint64_t needed = head();
  for (const Cursor* viewer : viewers_) {
    if (!viewer->waiting) needed = std::min(needed, viewer->next);
  }
  while (first_ < needed) {
    held_.pop_front();
    ++first_;
  }
}

void GopLog::fall_back(Cursor* cursor) noexcept {
  cursor->waiting = true;
  ++key_wanted_;
}

}  // namespace volumetric_kit::grid::networking::detail
