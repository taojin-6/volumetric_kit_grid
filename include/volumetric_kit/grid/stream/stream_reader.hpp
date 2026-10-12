// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file stream/stream_reader.hpp
/// @brief Read a `.vrst` file: its session, its records in file order, and
///        its key records by track, from a finished file or a killed
///        writer's.

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "volumetric_kit/core/base/result.hpp"
#include "volumetric_kit/core/base/unique_fd.hpp"
#include "volumetric_kit/recon/stream/export.hpp"
#include "volumetric_kit/recon/stream/session.hpp"

namespace volumetric_kit::recon::stream {

/// @brief How a @ref StreamReader reads.
struct ReaderOptions {
  /// The largest payload @ref StreamReader::read allocates; a larger one is
  /// `Status::Code::OutOfMemory` before any allocation.
  std::size_t max_payload_bytes = std::vector<std::uint8_t>().max_size();
};

/// @brief What opening a file found besides its records.
///
/// None of it is an error: a killed writer leaves no footer and perhaps
/// part of a record, and damage in the middle costs only the records it
/// hit.
struct Recovery {
  /// The file ends with the footer @ref StreamWriter::finish writes: a
  /// `TAIL` naming the `INDX` just before it, both intact, which counts the
  /// tracks and key records the file holds.
  bool finished = false;
  /// The file ends with a `TAIL`, but the footer it closes is not intact.
  bool footer_damaged = false;
  /// Where the records end: what follows is @ref torn_bytes.
  std::uint64_t end = 0;
  /// The bytes after the last whole chunk: part of a chunk, or damage with
  /// no whole chunk after it.
  std::uint64_t torn_bytes = 0;
  /// The bytes of the damaged spans, each followed by a whole chunk.
  std::uint64_t damaged_bytes = 0;
  std::uint64_t damaged_spans = 0;  ///< Their count.
};

/// @brief Where a @ref StreamReader::next reads from: a chunk's offset in
///        the file.
struct Position {
  std::uint64_t offset = 0;  ///< A multiple of 8, past the session header.
};

/// @brief A record's header, as @ref StreamReader::next found it.
struct Record {
  RecordInfo info;           ///< What the header says.
  std::uint64_t offset = 0;  ///< Its `RECD` chunk's offset in the file.
  std::uint64_t size = 0;    ///< Its payload's bytes.
};

/// @brief Reads one `.vrst` file (stream/session.hpp).
///
/// **Open scans.** @ref open checks the session header, then reads one
/// small chunk header after another, never a payload: the records' counts
/// (@ref stats) and the key records of each track (@ref key, @ref key_at)
/// come from that scan. A chunk header that fails its CRC, or holds what
/// no chunk can, starts a search, 8 bytes at a time, for the next `RECD`,
/// `INDX` or `TAIL` with a valid header; the bytes skipped are a damaged
/// span (@ref Recovery), and with no chunk after them, the torn tail.
///
/// **Reading.** @ref next returns record headers in file order from a
/// @ref Position the caller holds, and @ref read a record's payload, checked
/// against its CRC. A failed @ref next moves the position past the damage,
/// so iteration continues; a geometry consumer then resynchronises at the
/// next key record.
///
/// The file is read as it was at @ref open: its size then is where reading
/// stops.
///
/// @code
/// VKC_ASSIGN(auto reader, stream::StreamReader::open(path));
/// stream::Position pos = reader->begin();
/// stream::Record record;
/// std::vector<std::uint8_t> payload;
/// for (;;) {
///   const core::Result<bool> more = reader->next(&pos, &record);
///   if (!more.ok()) continue;  // damage: pos is past it
///   if (!more.value()) break;
///   if (reader->read(record, &payload).ok()) decode(payload);
/// }
/// @endcode
///
/// Every method is `const` and safe to call from several threads at once,
/// each with its own @ref Position.
class VR_STREAM_API StreamReader {
 public:
  /// @brief Open and scan the file at @p path.
  /// @param path     A `.vrst` file.
  /// @param options  How it is read.
  /// @return The reader; `Status::Code::InvalidArgument` for an empty path or
  ///         one with a NUL, something other than a regular file, a file
  ///         under 56 bytes or without the `VRST` magic, or a session header
  ///         out of range, failing a CRC or unparsable;
  ///         `Status::Code::Unsupported` for a session header of another
  ///         version or setting a flag; `Status::Code::NotFound` when the
  ///         file or its directory does not exist; otherwise
  ///         `Status::Code::IoError` or `Status::Code::OutOfMemory`.
  static core::Result<std::unique_ptr<StreamReader>> open(
      const std::string& path, const ReaderOptions& options = {});

  ~StreamReader() = default;
  StreamReader(const StreamReader&) = delete;
  StreamReader& operator=(const StreamReader&) = delete;
  StreamReader(StreamReader&&) = delete;
  StreamReader& operator=(StreamReader&&) = delete;

  /// @return The session header.
  const Session& session() const noexcept;
  /// @return What the scan found besides the records.
  const Recovery& recovery() const noexcept;
  /// @return Each track's records, by track. A record setting a flag this
  ///         version does not know is not counted.
  const std::vector<TrackStats>& stats() const noexcept;

  /// @param track  A track's index.
  /// @return The key records of @p track; 0 for a track that does not
  ///         exist.
  std::size_t key_count(std::uint16_t track) const noexcept;

  /// @brief Where @p track 's key record @p n is.
  /// @param track  A track's index.
  /// @param n      Which key, from 0 in file order.
  /// @return Its position, from which @ref next returns it;
  ///         `Status::Code::InvalidArgument` for a track that does not exist;
  ///         `Status::Code::NotFound` for @p n of @ref key_count or more.
  core::Result<Position> key(std::uint16_t track, std::size_t n) const;

  /// @brief Where @p track 's last key record with a pts at or before
  ///        @p pts_ns is: where playback seeking to @p pts_ns starts.
  /// @param track   A track's index.
  /// @param pts_ns  The time sought, on the track's clock.
  /// @return Its position; `Status::Code::InvalidArgument` for a track that
  ///         does not exist, or whose key pts decrease (a file this
  ///         library's writer did not make); `Status::Code::NotFound` when
  ///         its first key is later than @p pts_ns, or it has none.
  core::Result<Position> key_at(std::uint16_t track,
                                std::uint64_t pts_ns) const;

  /// @return The position of the first chunk after the session header.
  Position begin() const noexcept;

  /// @brief Read the header of the next record at or after @p pos, skipping
  ///        other chunks.
  /// @param[in,out] pos     Where to read; moved past what was read, or
  ///                        past the damage on a failure.
  /// @param[out]    record  The record's header, on `true`.
  /// @return `true` with a record; `false` at the end; or, with @p pos past
  ///         the damage, `Status::Code::InvalidArgument` for a chunk header
  ///         failing its CRC or out of range, or a record of a track that
  ///         does not exist, and `Status::Code::Unsupported` for a record
  ///         setting a flag this version does not know.
  ///         `Status::Code::InvalidArgument` too, with @p pos unchanged, for
  ///         a position that is not a multiple of 8 or lies before
  ///         @ref begin; otherwise `Status::Code::IoError`.
  core::Result<bool> next(Position* pos, Record* record) const;

  /// @brief Read a record's payload, checking its header again and its
  ///        payload against its CRC.
  /// @param record   A record @ref next returned from this reader.
  /// @param payload  Receives the payload, resized to fit.
  /// @return OK; `Status::Code::InvalidArgument` naming the offset when the
  ///         payload fails its CRC or the header is no longer
  ///         @p record 's; `Status::Code::OutOfMemory` for a payload above
  ///         `ReaderOptions::max_payload_bytes` or an allocation failure;
  ///         `Status::Code::IoError` for a read failure or a file shorter
  ///         than at open.
  core::Status read(const Record& record,
                    std::vector<std::uint8_t>* payload) const;

 private:
  // A key record, as the scan found it.
  struct KeyPoint {
    std::uint64_t offset = 0;
    std::uint64_t pts_ns = 0;
  };

  StreamReader() = default;

  // Count the records and find the key records and the footer, from
  // `begin_` to the file's end.
  core::Status scan();

  core::UniqueFd fd_;
  std::string path_;
  ReaderOptions options_;
  std::uint64_t file_size_ = 0;  // at open
  std::uint64_t begin_ = 0;      // the first chunk after the session header
  Session session_;
  Recovery recovery_;
  std::vector<TrackStats> stats_;
  std::vector<std::vector<KeyPoint>> keys_;  // by track, in file order
  std::vector<bool> keys_unordered_;         // by track: key pts decrease
};

}  // namespace volumetric_kit::recon::stream
