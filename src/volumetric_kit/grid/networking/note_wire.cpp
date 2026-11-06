// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include "note_wire.hpp"

#include <utility>

namespace volumetric_kit::grid::networking::detail {
namespace {

constexpr std::size_t kNoteHeadBytes = 4;  // tone, reserved, the line's size

}  // namespace

std::vector<std::uint8_t> encode_note(const ViewerNote& note) {
  const std::size_t size = note.line.size();
  std::vector<std::uint8_t> payload;
  payload.reserve(kNoteHeadBytes + size);
  payload.push_back(static_cast<std::uint8_t>(note.tone));
  payload.push_back(0);
  payload.push_back(static_cast<std::uint8_t>(size & 0xff));
  payload.push_back(static_cast<std::uint8_t>(size >> 8));
  payload.insert(payload.end(), note.line.begin(), note.line.end());
  return payload;
}

std::optional<ViewerNote> decode_note(
    const std::vector<std::uint8_t>& payload) {
  if (payload.size() < kNoteHeadBytes) return std::nullopt;
  const std::size_t size =
      std::size_t{payload[2]} | (std::size_t{payload[3]} << 8);
  if (size > kMaxNoteLineBytes || payload.size() - kNoteHeadBytes < size) {
    return std::nullopt;
  }
  ViewerNote note;
  if (payload[0] <= static_cast<std::uint8_t>(ViewerNote::Tone::Caution)) {
    note.tone = static_cast<ViewerNote::Tone>(payload[0]);
  }
  note.line.assign(
      reinterpret_cast<const char*>(payload.data()) + kNoteHeadBytes, size);
  return note;
}

std::optional<ViewerNote> NoteReader::feed(const std::uint8_t* data,
                                           std::size_t size) noexcept {
  std::optional<ViewerNote> newest;
  if (ended_) return newest;
  try {
    const core::Status fed = parser_.feed(data, size);
    if (!fed.ok()) {
      ended_ = true;
      why_ = fed.message();
      return newest;
    }
    since_record_ += size;
    stream::Record record;
    for (;;) {
      const core::Result<bool> got = parser_.next(&record, &payload_);
      if (!got.ok()) {
        ended_ = true;
        why_ = got.status().message();
        return newest;
      }
      if (!*got) break;
      since_record_ = 0;
      if (parser_.session()->tracks[record.info.track].kind !=
          stream::TrackKind::Note) {
        continue;
      }
      if (std::optional<ViewerNote> note = decode_note(payload_)) {
        newest = std::move(note);
      }
    }
    if (since_record_ >= kNoteReaderLimit) {
      ended_ = true;
      why_ = std::to_string(since_record_) + " bytes with no whole record";
    }
  } catch (...) {
    ended_ = true;
  }
  return newest;
}

}  // namespace volumetric_kit::grid::networking::detail
