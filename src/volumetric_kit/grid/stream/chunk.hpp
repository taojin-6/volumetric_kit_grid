// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

// The .vrst chunks in code (stream/session.hpp holds the layout): the tags
// and sizes, CRC-32, little-endian words, each chunk's encoding and decoding,
// the checks a chunk passes, on bytes in memory, and reading a chunk header
// from the file, with the search past damage for the next one.

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include "volumetric_kit/core/base/result.hpp"
#include "volumetric_kit/recon/stream/session.hpp"

namespace volumetric_kit::recon::stream::detail {

// --- Little-endian words. ----------------------------------------------------

inline std::uint16_t load_le16(const std::uint8_t* p) {
  return static_cast<std::uint16_t>(p[0] | (p[1] << 8));
}
inline std::uint32_t load_le32(const std::uint8_t* p) {
  return static_cast<std::uint32_t>(p[0]) |
         (static_cast<std::uint32_t>(p[1]) << 8) |
         (static_cast<std::uint32_t>(p[2]) << 16) |
         (static_cast<std::uint32_t>(p[3]) << 24);
}
inline std::uint64_t load_le64(const std::uint8_t* p) {
  return static_cast<std::uint64_t>(load_le32(p)) |
         (static_cast<std::uint64_t>(load_le32(p + 4)) << 32);
}
inline void store_le16(std::uint8_t* p, std::uint16_t v) {
  p[0] = static_cast<std::uint8_t>(v);
  p[1] = static_cast<std::uint8_t>(v >> 8);
}
inline void store_le32(std::uint8_t* p, std::uint32_t v) {
  for (int i = 0; i < 4; ++i) p[i] = static_cast<std::uint8_t>(v >> (8 * i));
}
inline void store_le64(std::uint8_t* p, std::uint64_t v) {
  for (int i = 0; i < 8; ++i) p[i] = static_cast<std::uint8_t>(v >> (8 * i));
}

// --- CRC-32. -----------------------------------------------------------------

// zlib's crc32: `crc` is the value of the bytes before, so a CRC can be
// computed in pieces (0 to start).
std::uint32_t crc32(const void* data, std::size_t size,
                    std::uint32_t crc = 0) noexcept;

// --- Tags and sizes. ---------------------------------------------------------

// A tag as the little-endian word its four bytes make.
constexpr std::uint32_t make_tag(const char (&name)[5]) {
  return static_cast<std::uint32_t>(static_cast<std::uint8_t>(name[0])) |
         static_cast<std::uint32_t>(static_cast<std::uint8_t>(name[1])) << 8 |
         static_cast<std::uint32_t>(static_cast<std::uint8_t>(name[2])) << 16 |
         static_cast<std::uint32_t>(static_cast<std::uint8_t>(name[3])) << 24;
}
inline constexpr std::uint32_t kSessionTag = make_tag("VRST");
inline constexpr std::uint32_t kRecordTag = make_tag("RECD");
inline constexpr std::uint32_t kIndexTag = make_tag("INDX");
inline constexpr std::uint32_t kTailTag = make_tag("TAIL");

inline constexpr std::uint32_t kSessionVersion = 0;

// H's range, and each tag's H as this version writes it (a reader takes a
// larger one).
inline constexpr std::uint32_t kMinHeaderBytes = 24;
inline constexpr std::uint32_t kMaxHeaderBytes = 4096;
inline constexpr std::uint32_t kSessionHeaderBytes = 56;
inline constexpr std::uint32_t kRecordHeaderBytes = 56;
inline constexpr std::uint32_t kIndexHeaderBytes = 40;
inline constexpr std::uint32_t kTailBytes = 40;

inline constexpr std::uint64_t kTrackEntryBytes = 24;
inline constexpr std::uint64_t kSummaryBytes = 40;
inline constexpr std::uint64_t kKeyEntryBytes = 32;

// The record flags this version knows.
inline constexpr std::uint16_t kKeyFlag = 1;

// The zero bytes after `n` that bring it to a multiple of 8.
constexpr std::uint64_t padding(std::uint64_t n) { return (8 - n % 8) % 8; }

// --- Encoding. ---------------------------------------------------------------

// The whole `VRST` chunk for `session`, padding included; InvalidArgument,
// naming `api`, for a session it cannot hold (no tracks or more than 65535,
// a kind of 0, a rate_den of 0, a body over kMaxSessionBytes).
core::Status encode_session(std::string_view api, const Session& session,
                            std::vector<std::uint8_t>* chunk);

// A `RECD` header for a payload of `size` bytes whose CRC is `body_crc`.
void encode_record_header(const RecordInfo& info, std::uint64_t size,
                          std::uint32_t body_crc,
                          std::array<std::uint8_t, kRecordHeaderBytes>& header);

// One key entry of the `INDX` body, appended to `entries`.
void append_key_entry(std::vector<std::uint8_t>& entries, std::uint64_t offset,
                      const RecordInfo& info);

// The footer: the `INDX` chunk over `stats` and `key_entries` (as
// append_key_entry made them) at `index_offset`, then the `TAIL` naming it.
std::vector<std::uint8_t> encode_footer(
    const std::vector<TrackStats>& stats,
    const std::vector<std::uint8_t>& key_entries, std::uint64_t index_offset);

// Count a record of `size` payload bytes in its track's stats.
void count_record(TrackStats& stats, const RecordInfo& info,
                  std::uint64_t size);

// --- Decoding. ---------------------------------------------------------------

// One key entry of the `INDX` body, as append_key_entry wrote it.
struct KeyEntry {
  std::uint64_t offset = 0;  // its `RECD` chunk's
  std::uint64_t pts_ns = 0;
  std::uint64_t set = kNoSet;
  std::uint16_t track = 0;
};

// The track summary, of kSummaryBytes, at `p`.
TrackStats load_summary(const std::uint8_t* p);

// The key entry, of kKeyEntryBytes, at `p`.
KeyEntry load_key_entry(const std::uint8_t* p);

// The file a reader reads: its descriptor, its path for messages, its size
// at open and its session's track count.
struct FileView {
  int fd = -1;
  const std::string* path = nullptr;
  std::uint64_t size = 0;
  std::uint32_t tracks = 0;
};

// A chunk header that passed its checks, and what its tag's fields say.
struct Chunk {
  std::uint32_t tag = 0;
  std::uint32_t header_bytes = 0;  // H
  std::uint64_t body_bytes = 0;    // B
  std::uint32_t body_crc = 0;
  RecordInfo record;               // RECD
  std::uint16_t flags = 0;         // RECD: every flag bit, kKeyFlag included
  std::uint32_t tracks = 0;        // INDX: T
  std::uint64_t keys = 0;          // INDX: K
  std::uint64_t index_offset = 0;  // TAIL
};

// --- Checks, on bytes in memory. ---------------------------------------------
//
// The file reader and ChunkParser (stream/chunks.hpp) both check a chunk with
// these, each check on the bytes it needs: header_fits on its first 8, then
// header_crc_ok and decode_chunk on its H. The session header, the first
// chunk, has its own three, each naming `api` and `path` (the bytes' name: a
// file's path) in its refusal.

// The least H of a chunk tagged `tag` after the session header: its tag's
// own, or 24 for a tag this version does not read (a `VRST` there too).
std::uint32_t least_header_bytes(std::uint32_t tag);

// Whether H = `h` is a header size, at least `least`, that fits in the
// `available` bytes left.
bool header_fits(std::uint32_t h, std::uint32_t least, std::uint64_t available);

// Whether the `h` bytes of `header` pass their CRC, the last 4.
bool header_crc_ok(const std::uint8_t* header, std::uint32_t h);

// What the `h` header bytes at `header`, which passed the checks above, say,
// into `out`: false, with `out` filled all the same, for a `RECD` of a track
// not below `tracks`.
bool decode_chunk(const std::uint8_t* header, std::uint32_t h,
                  std::uint32_t tracks, Chunk* out);

// Whether `c` is a `RECD` setting only the flags this version knows; one
// setting another is skipped (stream/session.hpp).
inline bool known_record(const Chunk& c) {
  return c.tag == kRecordTag && (c.flags & ~kKeyFlag) == 0;
}

// The session header's H, from its first 8 bytes at `header`:
// InvalidArgument unless they start with `VRST` and H is a header size of at
// least 56 within `available`.
core::Result<std::uint32_t> session_header_bytes(std::string_view api,
                                                 std::string_view path,
                                                 const std::uint8_t* header,
                                                 std::uint64_t available);

// The session header's B, from its `h` bytes at `header`: InvalidArgument
// when they fail their CRC or B is above kMaxSessionBytes; Unsupported for a
// version or flag this version does not know.
core::Result<std::uint64_t> session_body_bytes(std::string_view api,
                                               std::string_view path,
                                               const std::uint8_t* header,
                                               std::uint32_t h);

// The session the header at `header` and its body, `size` bytes at `body`,
// hold, into `session`: InvalidArgument when the body fails its CRC or does
// not parse exactly.
core::Status parse_session(std::string_view api, std::string_view path,
                           const std::uint8_t* header, const std::uint8_t* body,
                           std::uint64_t size, Session* session);

// --- The file. ---------------------------------------------------------------

// Read the chunk header at `offset`: `true`, with `out` filled, when it lies
// inside the file and passes its range and CRC checks, and a known tag's
// fields are in range (a `RECD` of a track the session has); `false` when
// not; IoError when the read fails. Its body is not checked.
core::Result<bool> read_chunk(std::string_view api, const FileView& file,
                              std::uint64_t offset, Chunk* out);

// The first chunk after damage at `from`: the least `from + 8 k` (k >= 1)
// below `end` where read_chunk finds a `RECD`, `INDX` or `TAIL`, searched in
// blocks of 4 KiB doubling to 1 MiB; `end` when there is none.
core::Result<std::uint64_t> resync(std::string_view api, const FileView& file,
                                   std::uint64_t from, std::uint64_t end);

// Read the session header, the file's first chunk, into `session`: where the
// chunk after it starts. InvalidArgument for a file shorter than a session
// header or without its magic, and for a header out of range, failing a CRC
// or not parsing exactly; Unsupported for a version or flag this version
// does not know; IoError when a read fails. `file.tracks` is not read.
core::Result<std::uint64_t> read_session(std::string_view api,
                                         const FileView& file,
                                         Session* session);

}  // namespace volumetric_kit::recon::stream::detail
