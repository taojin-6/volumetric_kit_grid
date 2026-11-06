// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

// A note's payload (stream/session.hpp, TrackKind::Note), and what a viewer
// thread reads notes with. The chunks are the stream tier's: a viewer frames
// them with ChunkEncoder, and this reads them with ChunkParser. No sockets,
// threads or clock.

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "volumetric_kit/grid/networking/stream_server.hpp"
#include "volumetric_kit/grid/stream/chunks.hpp"

namespace volumetric_kit::grid::networking::detail {

// The bytes a viewer may send with no whole record before its notes end, which
// bound what the parser holds. A note's whole chunk is at most 264.
inline constexpr std::size_t kNoteReaderLimit = std::size_t{8} << 10;

// The payload of `note`, whose line the caller has kept within
// kMaxNoteLineBytes. Throws what an allocation throws.
std::vector<std::uint8_t> encode_note(const ViewerNote& note);

// The note in `payload`, its later fields ignored and an unknown tone neutral;
// none for a payload shorter than it says or a line above kMaxNoteLineBytes.
// Throws what an allocation throws.
std::optional<ViewerNote> decode_note(const std::vector<std::uint8_t>& payload);

// Reads the notes in the bytes one viewer sends, as they arrive, for a viewer
// thread that must not fail because of them. A record that is not a note, or
// a note that cannot be read, is dropped. What leaves the chunks unreadable,
// nothing resyncing them, ends it: a chunk failing its checks, kNoteReaderLimit
// bytes with no whole record (a viewer that does not send chunks fails at its
// first 8), or an allocation that fails.
class NoteReader {
 public:
  // Take the next `size` bytes.
  // Returns the newest note they complete, if any; ended() may be true with it.
  std::optional<ViewerNote> feed(const std::uint8_t* data,
                                 std::size_t size) noexcept;
  // Whether it has ended, and takes no more bytes.
  bool ended() const noexcept { return ended_; }
  // Why, when an allocation left room to say; empty otherwise.
  const std::string& why() const noexcept { return why_; }

 private:
  stream::ChunkParser parser_;
  std::vector<std::uint8_t> payload_;
  std::size_t since_record_ = 0;  // the bytes fed since the last whole record
  bool ended_ = false;
  std::string why_;
};

}  // namespace volumetric_kit::grid::networking::detail
