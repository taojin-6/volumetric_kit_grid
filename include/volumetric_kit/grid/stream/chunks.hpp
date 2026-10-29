// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file stream/chunks.hpp
/// @brief The `.vrst` chunks in memory (stream/session.hpp): what
///        @ref StreamWriter writes, made without a file, and what
///        @ref StreamReader checks, read from bytes as they arrive.

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

#include "volumetric_kit/core/base/result.hpp"
#include "volumetric_kit/grid/stream/export.hpp"
#include "volumetric_kit/grid/stream/session.hpp"
#include "volumetric_kit/grid/stream/stream_reader.hpp"

namespace volumetric_kit::grid::stream {

/// @brief A record's payload as one or two runs of bytes, and their CRC,
///        computed once, by @ref of, for every chunk that frames it: a
///        file's and a server's.
///
/// Borrowed: the bytes must outlive every call given this, unchanged since
/// @ref of. A chunk framed from it carries the CRC @ref of computed, whatever
/// the bytes hold by then.
///
/// @code
/// const stream::RecordPayload payload =
///     stream::RecordPayload::of(head.data(), head.size(), body, body_size);
/// VKC_TRY(writer->append(info, payload));   // one writev, no joined copy
/// VKC_TRY(server->publish(info, payload));  // its one copy, no second CRC
/// @endcode
class VGRID_STREAM_API RecordPayload {
 public:
  /// @brief The payload @p first then @p second, and its CRC.
  /// @param first        Its first run; may be null when @p first_size is 0.
  /// @param first_size   Its bytes.
  /// @param second       Its second run, or null for none.
  /// @param second_size  Its bytes; 0 with a null @p second.
  /// @return The payload. A null run with bytes is a programmer error
  ///         (`VKC_CHECK`).
  static RecordPayload of(const std::uint8_t* first, std::size_t first_size,
                          const std::uint8_t* second = nullptr,
                          std::size_t second_size = 0) noexcept;
  /// @return The first run.
  const std::uint8_t* first() const noexcept { return first_; }
  /// @return Its bytes.
  std::size_t first_size() const noexcept { return first_size_; }
  /// @return The second run, or null.
  const std::uint8_t* second() const noexcept { return second_; }
  /// @return Its bytes.
  std::size_t second_size() const noexcept { return second_size_; }
  /// @return Both runs' bytes.
  std::size_t size() const noexcept { return first_size_ + second_size_; }
  /// @return The CRC of both, as a `RECD` header carries it.
  std::uint32_t crc() const noexcept { return crc_; }

 private:
  RecordPayload() = default;

  const std::uint8_t* first_ = nullptr;
  std::size_t first_size_ = 0;
  const std::uint8_t* second_ = nullptr;
  std::size_t second_size_ = 0;
  std::uint32_t crc_ = 0;
};

/// @brief What @ref StreamWriter writes, without a file: the session chunk,
///        then each record's framing, by the writer's rules.
///
/// The session chunk, then each record's header, payload and padding, are
/// byte for byte the file a @ref StreamWriter makes of the same session and
/// records, less its footer: the writer frames through one.
///
/// @code
/// VKC_ASSIGN(stream::ChunkEncoder encoder,
///            stream::ChunkEncoder::create(session));
/// send(encoder.session_chunk());
/// VKC_ASSIGN(const stream::ChunkEncoder::RecordFraming framing,
///            encoder.frame(info, payload.data(), payload.size()));
/// send(framing.header, payload, framing.padding);  // the padding zero bytes
/// @endcode
///
/// @warning Not thread-safe: call @ref frame from one thread.
class VGRID_STREAM_API ChunkEncoder {
 public:
  /// @brief Encode @p session's chunk.
  /// @param session  As @ref StreamWriter::create takes it: at least one
  ///                 track and at most 65535, each of a nonzero kind with a
  ///                 nonzero `rate.den`, and a body of at most
  ///                 @ref kMaxSessionBytes.
  /// @return The encoder, with no record framed;
  ///         `Status::Code::InvalidArgument` for a session refused above;
  ///         `Status::Code::OutOfMemory` when the chunk or a diagnostic
  ///         cannot be allocated.
  static core::Result<ChunkEncoder> create(const Session& session);

  /// @return The `VRST` chunk, padding included.
  const std::vector<std::uint8_t>& session_chunk() const noexcept;

  /// @brief A record's chunk is `header`, the payload, then `padding` zero
  ///        bytes, which end it at a multiple of 8.
  struct RecordFraming {
    std::array<std::uint8_t, 56> header{};  ///< The `RECD` header (H = 56).
    std::size_t padding = 0;                ///< 0 to 7.
  };

  /// @brief Frame one record, by @ref StreamWriter::append's rules.
  /// @param info  Its track (below the track count) and fields; @ref
  ///              RecordInfo::seq above the track's last framed record's,
  ///              and a key's pts no lower than the track's last framed
  ///              key's.
  /// @param data  The payload, whose CRC the header carries; may be null
  ///              when @p size is 0.
  /// @param size  The payload's bytes.
  /// @return The framing, with the record counted as the track's last;
  ///         with nothing counted, `Status::Code::InvalidArgument` for a
  ///         record breaking the rules above or a null @p data with a
  ///         nonzero @p size, and `Status::Code::OutOfMemory` when a
  ///         diagnostic cannot be allocated.
  core::Result<RecordFraming> frame(const RecordInfo& info,
                                    const std::uint8_t* data, std::size_t size);

  /// @brief @ref frame a payload in up to two runs, its CRC the one
  ///        @p payload carries, not computed again.
  /// @return As the other @ref frame.
  core::Result<RecordFraming> frame(const RecordInfo& info,
                                    const RecordPayload& payload);

 private:
  // What the record rules check of a track's last framed records.
  struct TrackState {
    bool any = false;      // a record framed
    bool any_key = false;  // a key framed
    std::uint64_t last_seq = 0;
    std::uint64_t last_key_pts_ns = 0;
  };

  ChunkEncoder(std::vector<std::uint8_t> session_chunk, std::size_t tracks);

  std::vector<std::uint8_t> session_chunk_;
  std::vector<TrackState> state_;  // by track
};

/// @brief Reads `.vrst` bytes as they arrive: the session, then each
///        record, with @ref StreamReader's checks. A failed check is final.
///
/// **A header before its body.** The bytes must start with the session
/// chunk. Each check runs once the bytes it needs are in, a chunk header's
/// before any of its body is needed: the `VRST` magic, the version and the
/// flags; H, a multiple of 8 from the least its tag takes to 4096; the
/// header's CRC; a `RECD`'s track; and B, at most @ref kMaxSessionBytes for
/// the session and @ref kMaxBodyBytes for any other chunk. Every chunk's
/// body CRC is checked once the body is whole.
///
/// **What is skipped.** A chunk of another tag (`INDX` and `TAIL`, a
/// finished file's footer, included) and a `RECD` setting a flag this
/// version does not know are skipped by their size, once checked, so a
/// whole file whose chunks are within the limits above parses, finished or
/// not. Nothing resyncs: after a failure, every call returns it again.
///
/// **Memory.** @ref feed copies its bytes. The parser holds the chunk it is
/// reading and what was fed after it: about one chunk, when it is fed a piece
/// at a time and @ref next is called until it returns `false` before the
/// next piece.
///
/// @code
/// stream::ChunkParser parser;
/// stream::Record record;
/// std::vector<std::uint8_t> payload;
/// for (;;) {
///   const std::size_t n = receive(buffer);  // some bytes, as they come
///   VKC_TRY(parser.feed(buffer.data(), n));
///   for (;;) {
///     VKC_ASSIGN(const bool got, parser.next(&record, &payload));
///     if (!got) break;
///     decode(parser.session()->tracks[record.info.track], payload);
///   }
/// }
/// @endcode
///
/// @warning Not thread-safe: one thread uses it.
class VGRID_STREAM_API ChunkParser {
 public:
  /// The largest body of a chunk other than the session's: a larger one is
  /// refused once its header is in, whatever of its body has come.
  static constexpr std::uint64_t kMaxBodyBytes = std::uint64_t{64} << 20;

  /// @brief Take the next @p size bytes, and check what they complete.
  /// @param data  The bytes, copied; may be null when @p size is 0.
  /// @param size  Their count.
  /// @return OK; or the failure, which every later call returns again:
  ///         `Status::Code::InvalidArgument` for bytes that do not start
  ///         with `VRST`, a chunk header out of range or of a `RECD` of a
  ///         track the session lacks, a CRC that fails, a session body above
  ///         @ref kMaxSessionBytes or not parsing exactly;
  ///         `Status::Code::Unsupported` for a session of another version or
  ///         setting a flag; `Status::Code::OutOfMemory` for another body
  ///         above @ref kMaxBodyBytes, or an allocation failure.
  core::Status feed(const std::uint8_t* data, std::size_t size);

  /// @return The session, once its chunk is in, which is before the first
  ///         record; null before.
  const Session* session() const noexcept;

  /// @brief Take the next record, skipping other chunks.
  ///
  /// With a payload, the @ref read below into it, resized to fit.
  /// @param[out] record   Its header. Its `offset` is its chunk's offset in
  ///                      the bytes fed, as in the file they would make.
  /// @param[out] payload  Its payload, resized to fit; unspecified after a
  ///                      failure. Or null: its body, checked against its
  ///                      CRC, stays here for @ref read until the next call
  ///                      of this, which drops it unread.
  /// @return `true` with a record; `false` when it needs more bytes; or the
  ///         failure, as @ref feed returns it.
  core::Result<bool> next(Record* record, std::vector<std::uint8_t>* payload);

  /// @brief Copy the body of the record the last @ref next returned with no
  ///        payload into caller memory, such as a decoder's input slot.
  /// @param record    That record.
  /// @param dst       Receives the body, `record.size` bytes.
  /// @param capacity  Bytes at @p dst.
  /// @return OK, the record taken; with nothing copied and the record still
  ///         held, `Status::Code::InvalidArgument` for a @p capacity under
  ///         `record.size`, a null @p dst for a body of any bytes, or a
  ///         record that is not the one held; or the failure, as @ref feed
  ///         returns it. A read that succeeds allocates nothing.
  core::Status read(const Record& record, std::uint8_t* dst,
                    std::size_t capacity);

 private:
  // Check what was fed, up to the next record: true when one is whole and
  // checked at `next_`.
  core::Result<bool> advance();
  // Save `failure`, which every later call returns, and return it.
  core::Status fail(core::Status failure);
  // fail() for the exception being handled, within a catch.
  core::Status fail_on_exception() noexcept;

  std::vector<std::uint8_t> buffer_;  // the bytes fed from base_ on
  std::uint64_t base_ = 0;            // buffer_[0]'s offset in the bytes fed
  std::uint64_t next_ = 0;  // the next chunk's offset; past the bytes fed
                            // while its padding is still to come
  bool ready_ = false;      // a checked record starts at next_
  // The bytes of the chunk at next_ that next() returned and read() has not
  // taken; 0 for none.
  std::uint64_t held_ = 0;
  std::optional<Session> session_;
  core::Status failure_;
};

}  // namespace volumetric_kit::grid::stream
