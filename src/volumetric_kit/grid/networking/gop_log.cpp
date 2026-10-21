// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include "gop_log.hpp"

#include <utility>

namespace volumetric_kit::recon::networking::detail {

GopLog::GopLog(std::uint64_t max_gop_bytes) noexcept
    : max_gop_bytes_(max_gop_bytes) {}

bool GopLog::append(Chunk chunk, bool key) {
  if (key) {
    ++gop_;
    chunks_.clear();
    bytes_ = 0;
    open_ = true;
  }
  if (!open_) return false;
  const std::uint64_t size = chunk->size();
  if (size > max_gop_bytes_ - bytes_) {
    break_gop();
    return false;
  }
  try {
    chunks_.push_back(std::move(chunk));
  } catch (...) {
    break_gop();
    throw;
  }
  bytes_ += size;
  return true;
}

void GopLog::break_gop() noexcept {
  chunks_.clear();
  bytes_ = 0;
  open_ = false;
}

GopLog::Cursor GopLog::join() const noexcept { return {gop_, 0}; }

Chunk GopLog::next(Cursor* cursor) const noexcept {
  if (cursor->gop != gop_) *cursor = {gop_, 0};
  if (cursor->next >= chunks_.size()) return nullptr;
  return chunks_[cursor->next++];
}

bool GopLog::open() const noexcept { return open_; }

}  // namespace volumetric_kit::recon::networking::detail
