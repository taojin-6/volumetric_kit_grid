// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file stream/session.hpp
/// @brief A recording session as a `.vrst` file holds it, and the file's
///        byte layout, version 0.
///
/// A `.vrst` file is a session header and the records appended after it, in
/// tracks, each payload stored and returned byte for byte as appended.
/// @ref StreamWriter writes one and @ref StreamReader reads it.
///
/// **Chunks.** All fields are little-endian. The file is a sequence of
/// chunks, each starting at a multiple of 8 from the start of the file, all
/// framed alike (for every tag and version):
///
/// @code
///   offset  size  field
///        0     4  tag, 4 ASCII bytes
///        4     4  H, the header size: a multiple of 8, 24 <= H <= 4096
///        8     8  B, the body size
///       16     4  body_crc: CRC-32 of the B body bytes (0 when B = 0)
///       20  H-24  the tag's own fields
///      H-4     4  header_crc: CRC-32 of bytes [0, H-4)
/// @endcode
///
/// The body starts at offset H and the next chunk at `offset + H +
/// align8(B)`; the padding bytes are zero and covered by neither CRC. The
/// CRC is zlib's and PNG's CRC-32 (reflected polynomial 0xEDB88320, initial
/// value and final XOR 0xFFFFFFFF; `crc32("123456789") = 0xCBF43926`), so
/// Python's `zlib.crc32` checks any chunk.
///
/// **`VRST`**, the session header: the first chunk, its tag the file's
/// magic (H = 56).
///
/// @code
///       20     4  version (0)
///       24    16  session_id (all zero: none)
///       40     8  created_ns: system_clock ns since the Unix epoch; 0 unknown
///       48     2  T, the track count (1..65535)
///       50     2  flags (0)
///       52     4  header_crc
/// @endcode
///
/// Its body, at most @ref kMaxSessionBytes and parsed exactly: T track
/// entries of 24 bytes, then strings, each a `u32` length and its bytes: the
/// writer, the calibration, and for each track its sensor, then its params.
/// A track's id is its index in the table.
///
/// @code
///   track entry
///        0     2  kind (never 0)
///        2     2  clock (0: the host's system_clock; n >= 1: clock domain n)
///        4     4  rate_num (0: unknown)
///        8     4  rate_den (>= 1)
///       12     4  width
///       16     4  height (both 0 when the track is not pictures)
///       20     4  reserved
/// @endcode
///
/// **`RECD`**, one record (H = 56); its body is the payload.
///
/// @code
///       20     2  track (< T)
///       22     2  flags: bit 0 key
///       24     8  pts_ns, on the track's clock
///       32     8  seq
///       40     8  set (all ones: kNoSet)
///       48     4  reserved
///       52     4  header_crc
/// @endcode
///
/// **`INDX`**, the footer index (H = 40), and **`TAIL`**, the last 40 bytes
/// of a finished file (H = 40, B = 0), which names where `INDX` starts.
///
/// @code
///   INDX                                   TAIL
///       20     4  T                            20     4  reserved
///       24     8  K, the key records           24     8  index_offset
///       32     4  reserved                     32     4  reserved
///       36     4  header_crc                   36     4  header_crc
/// @endcode
///
/// The `INDX` body, B = 40 T + 32 K: T @ref TrackStats of five `u64`s, in
/// field order, then K key entries in file order: the `u64` offset of the
/// `RECD` chunk, `u64` pts_ns, `u64` set, `u16` track, 6 reserved bytes.
///
/// **Rules.**
/// - A record's track's records are in decode order, its seq strictly
///   increasing (a jump: records lost before the writer saw them), and a key
///   record's pts never below the previous key's; other pts may go back.
///   Tracks interleave in append order.
/// - A reader skips an unknown tag by its size and ignores the trailing
///   fields of an H larger than it knows. Reserved fields are written as zero
///   and ignored when read.
/// - A flag bit a reader does not know is `Unsupported`: at file level for
///   `VRST`, for that record alone for `RECD`.
/// - The version changes only for a change an old reader must not ignore.

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace volumetric_kit::recon::stream {

/// @brief What a track's payloads are. A kind comes with its first writer;
///        a reader passes the values it does not know through as read.
enum class TrackKind : std::uint16_t {
  /// One codec frame per record, exactly what `codec::Encoder::encode`
  /// returned; a key record is an intra frame. The track has no width,
  /// height or sensor: the frame's own header says what it is. Its params
  /// are empty, or the bytes of `io::encode_transform`
  /// (io/mesh_normalization.hpp): the transform that placed a mesh source in
  /// metres.
  Geometry = 1,
};

/// @brief The clock of @ref Track::clock that is the host's
///        `std::chrono::system_clock`: ns since the Unix epoch, comparable
///        across every track on it. Any other value is a clock domain whose
///        origin is arbitrary, comparable only among tracks naming it.
inline constexpr std::uint16_t kHostClock = 0;

/// @brief The @ref RecordInfo::set of a record in no set.
inline constexpr std::uint64_t kNoSet = ~std::uint64_t{0};

/// @brief The most a session's `VRST` body -- its track table and strings
///        -- may hold.
inline constexpr std::uint64_t kMaxSessionBytes = std::uint64_t{64} << 20;

/// @brief A rate as a fraction, records a second for a track.
struct Rate {
  std::uint32_t num = 0;  ///< 0 when unknown.
  std::uint32_t den = 1;  ///< At least 1.
};

/// @brief One track of a session: a stream of records of one kind.
struct Track {
  TrackKind kind{};  ///< Never 0.
  /// @ref kHostClock, or the clock domain its pts are on.
  std::uint16_t clock = 1;
  Rate rate;                 ///< Its nominal rate.
  std::uint32_t width = 0;   ///< Picture width; 0 when not pictures.
  std::uint32_t height = 0;  ///< Picture height; 0 when not pictures.
  std::string sensor;        ///< The sensor's id, or empty.
  std::string params;        ///< What the kind defines (@ref TrackKind).
};

/// @brief What a session header holds.
///
/// @code
/// stream::Session session;
/// session.writer = "codec_replica";
/// session.tracks.push_back({stream::TrackKind::Geometry});
/// @endcode
struct Session {
  std::array<std::uint8_t, 16> id{};  ///< All zero for none.
  /// The host's `system_clock` at the start, ns since the Unix epoch; 0 when
  /// unknown. The writer records what it is given, so two runs given the
  /// same session write the same bytes.
  std::uint64_t created_ns = 0;
  std::string writer;  ///< The program that wrote the file (UTF-8).
  /// The camera tier's calibration document, byte for byte, or empty.
  std::string calibration;
  std::vector<Track> tracks;  ///< At least 1, at most 65535.
};

/// @brief What a record's header says about its payload.
struct RecordInfo {
  std::uint16_t track = 0;  ///< Its track's index in @ref Session::tracks.
  /// A record decoding can start at: for Geometry, an intra frame.
  bool key = false;
  std::uint64_t pts_ns = 0;  ///< Its time on its track's clock.
  /// Its number in its track, strictly increasing; a jump says records were
  /// lost before the writer saw them.
  std::uint64_t seq = 0;
  /// The capture set it belongs to, across tracks, or @ref kNoSet.
  std::uint64_t set = kNoSet;
};

/// @brief A track's records, as the footer index and a scan both count
///        them.
struct TrackStats {
  std::uint64_t records = 0;        ///< Its records.
  std::uint64_t keys = 0;           ///< Its key records.
  std::uint64_t payload_bytes = 0;  ///< Its payloads' bytes.
  std::uint64_t min_pts_ns = 0;     ///< Its least pts; 0 with no records.
  std::uint64_t max_pts_ns = 0;     ///< Its greatest pts; 0 with no records.
};

}  // namespace volumetric_kit::recon::stream
