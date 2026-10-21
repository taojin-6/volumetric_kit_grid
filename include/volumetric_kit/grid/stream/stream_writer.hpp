// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file stream/stream_writer.hpp
/// @brief Record a session to a new `.vrst` file, a record at a time.

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "volumetric_kit/core/base/result.hpp"
#include "volumetric_kit/core/base/unique_fd.hpp"
#include "volumetric_kit/recon/stream/chunks.hpp"
#include "volumetric_kit/recon/stream/export.hpp"
#include "volumetric_kit/recon/stream/session.hpp"

namespace volumetric_kit::recon::stream {

/// @brief Writes one `.vrst` file (stream/session.hpp): the session header,
///        then each record as it is appended, then on @ref finish the
///        footer index. The chunks, and the rules a record follows, are
///        @ref ChunkEncoder's.
///
/// **Visible and recoverable while written.** The file appears at its path
/// when @ref create makes it, and every @ref append is one `writev` with no
/// buffering here, so a killed process loses nothing it appended, and a
/// reader recovers every whole record from a file that was never finished.
/// A power loss can lose what came after the last @ref sync.
///
/// **A failed write sticks.** After a write or flush fails, every call
/// returns the saved failure and writes nothing more; the file reads as a
/// killed writer's. After @ref finish, every call is refused.
/// Host allocation failures, including building or copying a diagnostic,
/// return `Status::Code::OutOfMemory`. Validation or preparation allocation
/// failures leave the writer unchanged; a failed write or flush remains
/// sticky even when its diagnostic cannot be allocated.
///
/// @code
/// stream::Session session;
/// session.writer = "codec_replica";
/// session.tracks.push_back({stream::TrackKind::Geometry});
/// VKC_ASSIGN(auto writer, stream::StreamWriter::create(path, session));
/// for (std::uint64_t i = 0; i < frames.size(); ++i) {
///   stream::RecordInfo info;
///   info.key = true;
///   info.pts_ns = i * 33'333'333;
///   info.seq = i;
///   VKC_TRY(writer->append(info, frames[i].data(), frames[i].size()));
/// }
/// VKC_TRY(writer->finish());
/// @endcode
///
/// @warning Not thread-safe: call it from one thread.
class VR_STREAM_API StreamWriter {
 public:
  /// @brief Create the file at @p path and write @p session's header to it,
  ///        flushed to storage with its directory entry.
  /// @param path     Where the new file goes; nothing may be there, and its
  ///                 directory must exist.
  /// @param session  The header. At least one track and at most 65535, each
  ///                 of a nonzero kind with a nonzero `rate.den`, and a body
  ///                 of at most @ref kMaxSessionBytes.
  /// @return The writer; `Status::Code::InvalidArgument` for an empty path or
  ///         one with a NUL, a session refused above (with nothing created),
  ///         or anything already at @p path, a symbolic link included (left
  ///         untouched); `Status::Code::NotFound` when the directory does not
  ///         exist; otherwise `Status::Code::IoError` or
  ///         `Status::Code::OutOfMemory`, with any file it made removed.
  static core::Result<std::unique_ptr<StreamWriter>> create(
      const std::string& path, const Session& session);

  /// @brief Closes the file. Without @ref finish it has no footer, and reads
  ///        as a killed writer's: every record it holds, `finished` false.
  ~StreamWriter();
  StreamWriter(const StreamWriter&) = delete;
  StreamWriter& operator=(const StreamWriter&) = delete;
  StreamWriter(StreamWriter&&) = delete;
  StreamWriter& operator=(StreamWriter&&) = delete;

  /// @brief Append one record: its header and @p size payload bytes, stored
  ///        as given.
  /// @param info  Its track (below the track count) and fields; @ref
  ///              RecordInfo::seq above the track's last record's, and a
  ///              key's pts no lower than the track's last key's.
  /// @param data  The payload; may be null when @p size is 0.
  /// @param size  The payload's bytes.
  /// @return OK once written; with nothing written,
  ///         `Status::Code::InvalidArgument` for a record breaking the rules
  ///         above, a null @p data with a nonzero @p size, or a finished
  ///         writer, and `Status::Code::OutOfMemory` when a key's room in the
  ///         footer index cannot be allocated; or the saved I/O failure.
  ///         Diagnostic allocation failures are also
  ///         `Status::Code::OutOfMemory`.
  core::Status append(const RecordInfo& info, const std::uint8_t* data,
                      std::size_t size);

  /// @brief Flush what is written to storage: `fsync`, or on Apple
  ///        `F_FULLFSYNC`, as `fsync` there stops at the drive's cache.
  /// @return OK; `Status::Code::InvalidArgument` once finished; the saved I/O
  ///         failure; or `Status::Code::OutOfMemory` for diagnostic allocation.
  core::Status sync();

  /// @brief Write the footer (`INDX` and `TAIL`), flush and close the file.
  /// @return OK; `Status::Code::InvalidArgument` once finished;
  ///         `Status::Code::OutOfMemory` when the index cannot be built (the
  ///         writer unchanged) or a diagnostic cannot be allocated; or the
  ///         saved I/O failure.
  core::Status finish();

  /// @return The file's bytes written so far.
  std::uint64_t size() const noexcept;

 private:
  StreamWriter(core::UniqueFd fd, std::string path, ChunkEncoder encoder,
               std::size_t tracks);

  // The sticky failure, or the refusal of a call after `finish`.
  core::Status check_open(std::string_view api) const;

  core::UniqueFd fd_;
  std::string path_;
  ChunkEncoder encoder_;  // the session's chunk and the record rules
  std::uint64_t size_ = 0;
  std::vector<TrackStats> stats_;  // by track
  // The footer index's key entries, encoded as they are appended.
  std::vector<std::uint8_t> index_keys_;
  core::Status failure_;  // the sticky write or flush failure
  bool finished_ = false;
};

}  // namespace volumetric_kit::recon::stream
