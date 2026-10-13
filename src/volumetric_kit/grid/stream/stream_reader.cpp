// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include "volumetric_kit/recon/stream/stream_reader.hpp"

#include <algorithm>
#include <iterator>
#include <limits>
#include <utility>

#include "chunk.hpp"
#include "file.hpp"
#include "volumetric_kit/core/base/check.hpp"
#include "volumetric_kit/core/base/exception_status.hpp"
#include "volumetric_kit/core/base/file_io.hpp"

namespace volumetric_kit::recon::stream {
namespace {

// What a footer's index body is read in.
constexpr std::size_t kFooterBlock = std::size_t{1} << 20;

// Whether `index`, at `index_at`, and `tail`, at `tail_at`, are the footer
// StreamWriter::finish writes: the tail ends the file and names the index,
// which ends where the tail starts, counts the session's tracks, is sized
// for them and its keys, and passes its body's CRC.
core::Result<bool> footer_intact(std::string_view api,
                                 const detail::FileView& file,
                                 std::uint64_t index_at,
                                 const detail::Chunk& index,
                                 std::uint64_t tail_at,
                                 const detail::Chunk& tail) {
  if (tail.body_bytes != 0 || tail.header_bytes != file.size - tail_at ||
      tail.index_offset != index_at || index.tracks != file.tracks) {
    return false;
  }
  const std::uint64_t summaries = detail::kSummaryBytes * file.tracks;
  if (index.keys > (std::numeric_limits<std::uint64_t>::max() - summaries) /
                       detail::kKeyEntryBytes ||
      index.body_bytes != summaries + detail::kKeyEntryBytes * index.keys ||
      index_at + index.header_bytes + index.body_bytes != tail_at) {
    return false;
  }
  std::vector<std::uint8_t> block(static_cast<std::size_t>(
      std::min<std::uint64_t>(kFooterBlock, index.body_bytes)));
  std::uint32_t crc = 0;
  for (std::uint64_t done = 0; done < index.body_bytes;) {
    const auto n = static_cast<std::size_t>(
        std::min<std::uint64_t>(block.size(), index.body_bytes - done));
    VKC_TRY(detail::read_at(api, *file.path, file.fd,
                            index_at + index.header_bytes + done, block.data(),
                            n));
    crc = detail::crc32(block.data(), n, crc);
    done += n;
  }
  return crc == index.body_crc;
}

}  // namespace

core::Result<std::unique_ptr<StreamReader>> StreamReader::open(
    const std::string& path, const ReaderOptions& options) {
  constexpr std::string_view kApi = "StreamReader::open";
  try {
    VKC_TRY(core::check_path(kApi, path));
    VKC_ASSIGN(core::UniqueFd fd, detail::open_file(kApi, path));
    VKC_ASSIGN(const std::uint64_t size,
               detail::regular_file_size(kApi, path, fd.get()));
    std::unique_ptr<StreamReader> reader(new StreamReader());
    const detail::FileView file{fd.get(), &path, size, 0};
    VKC_ASSIGN(reader->begin_,
               detail::read_session(kApi, file, &reader->session_));
    reader->fd_ = std::move(fd);
    reader->path_ = path;
    reader->options_ = options;
    reader->file_size_ = size;
    // TODO(stream): open a finished file from its footer, reading no record,
    // and scan only a file without an intact one.
    VKC_TRY(reader->scan());
    return reader;
  } catch (...) {
    return core::exception_status(kApi, core::Status::Code::IoError);
  }
}

core::Status StreamReader::scan() {
  constexpr std::string_view kApi = "StreamReader::open";
  const std::size_t tracks = session_.tracks.size();
  stats_.assign(tracks, TrackStats{});
  keys_.assign(tracks, {});
  keys_unordered_.assign(tracks, false);
  const detail::FileView file{fd_.get(), &path_, file_size_,
                              static_cast<std::uint32_t>(tracks)};
  // The chunk just before the one read, while it is an INDX: a TAIL after
  // it may close the footer.
  bool after_index = false;
  std::uint64_t index_at = 0;
  detail::Chunk index;
  // Where a footer's TAIL starts, and whether the scan looked for a chunk
  // there; elsewhere those bytes are inside a chunk (a payload, perhaps).
  const std::uint64_t tail_at = file_size_ - detail::kTailBytes;
  bool tail_reached = false;
  std::uint64_t pos = begin_;
  while (pos < file_size_) {
    tail_reached = tail_reached || pos == tail_at;
    detail::Chunk c;
    VKC_ASSIGN(const bool valid, detail::read_chunk(kApi, file, pos, &c));
    if (!valid) {
      VKC_ASSIGN(const std::uint64_t found,
                 detail::resync(kApi, file, pos, file_size_));
      if (found == file_size_) break;  // the rest is torn
      recovery_.damaged_bytes += found - pos;
      ++recovery_.damaged_spans;
      after_index = false;
      pos = found;
      continue;
    }
    if (c.body_bytes > file_size_ - pos - c.header_bytes) break;  // torn
    if (c.tag == detail::kRecordTag && (c.flags & ~detail::kKeyFlag) == 0) {
      detail::count_record(stats_[c.record.track], c.record, c.body_bytes);
      if (c.record.key) {
        std::vector<KeyPoint>& keys = keys_[c.record.track];
        if (!keys.empty() && c.record.pts_ns < keys.back().pts_ns) {
          keys_unordered_[c.record.track] = true;
        }
        keys.push_back({pos, c.record.pts_ns});
      }
    } else if (c.tag == detail::kTailTag && after_index) {
      VKC_ASSIGN(recovery_.finished,
                 footer_intact(kApi, file, index_at, index, pos, c));
    }
    after_index = c.tag == detail::kIndexTag;
    if (after_index) {
      index_at = pos;
      index = c;
    }
    pos += c.header_bytes + c.body_bytes + detail::padding(c.body_bytes);
  }
  recovery_.end = std::min(pos, file_size_);
  recovery_.torn_bytes = file_size_ - recovery_.end;
  // A chunk tagged TAIL where the footer's starts, intact or not: the file
  // was finished, and its footer is damaged.
  if (!recovery_.finished && tail_reached) {
    std::uint8_t tag[4];
    VKC_TRY(detail::read_at(kApi, path_, fd_.get(), tail_at, tag, sizeof(tag)));
    recovery_.footer_damaged = detail::load_le32(tag) == detail::kTailTag;
  }
  return {};
}

const Session& StreamReader::session() const noexcept { return session_; }

const Recovery& StreamReader::recovery() const noexcept { return recovery_; }

const std::vector<TrackStats>& StreamReader::stats() const noexcept {
  return stats_;
}

std::size_t StreamReader::key_count(std::uint16_t track) const noexcept {
  return track < keys_.size() ? keys_[track].size() : 0;
}

core::Result<Position> StreamReader::key(std::uint16_t track,
                                         std::size_t n) const {
  constexpr std::string_view kApi = "StreamReader::key";
  try {
    if (track >= keys_.size()) {
      return core::Status::invalid_argument(
          "StreamReader::key: " + path_ + ": track " + std::to_string(track) +
          ", but the session has " + std::to_string(keys_.size()));
    }
    if (n >= keys_[track].size()) {
      return core::Status::not_found(
          "StreamReader::key: " + path_ + ": track " + std::to_string(track) +
          " has " + std::to_string(keys_[track].size()) + " key records, no " +
          std::to_string(n));
    }
    return Position{keys_[track][n].offset};
  } catch (...) {
    return core::exception_status(kApi, core::Status::Code::IoError);
  }
}

core::Result<Position> StreamReader::key_at(std::uint16_t track,
                                            std::uint64_t pts_ns) const {
  constexpr std::string_view kApi = "StreamReader::key_at";
  try {
    const auto where = [&] {
      return "StreamReader::key_at: " + path_ + ": track " +
             std::to_string(track);
    };
    if (track >= keys_.size()) {
      return core::Status::invalid_argument(where() + ", but the session has " +
                                            std::to_string(keys_.size()));
    }
    if (keys_unordered_[track]) {
      return core::Status::invalid_argument(
          where() + "'s key pts decrease, so no key is found by its pts");
    }
    const std::vector<KeyPoint>& keys = keys_[track];
    const auto after =
        std::upper_bound(keys.begin(), keys.end(), pts_ns,
                         [](std::uint64_t pts, const KeyPoint& key) {
                           return pts < key.pts_ns;
                         });
    if (after == keys.begin()) {
      return core::Status::not_found(where() +
                                     " has no key record at or before pts " +
                                     std::to_string(pts_ns));
    }
    return Position{std::prev(after)->offset};
  } catch (...) {
    return core::exception_status(kApi, core::Status::Code::IoError);
  }
}

Position StreamReader::begin() const noexcept { return Position{begin_}; }

core::Result<bool> StreamReader::next(Position* pos, Record* record) const {
  constexpr std::string_view kApi = "StreamReader::next";
  VKC_CHECK(pos != nullptr && record != nullptr,
            "StreamReader::next takes a position and a record");
  try {
    if (pos->offset % 8 != 0 || pos->offset < begin_) {
      return core::Status::invalid_argument(std::string(kApi) + ": " + path_ +
                                            ": " + std::to_string(pos->offset) +
                                            " is not a chunk position");
    }
    const detail::FileView file{
        fd_.get(), &path_, file_size_,
        static_cast<std::uint32_t>(session_.tracks.size())};
    while (pos->offset < recovery_.end) {
      const std::uint64_t at = pos->offset;
      detail::Chunk c;
      VKC_ASSIGN(const bool valid, detail::read_chunk(kApi, file, at, &c));
      if (!valid) {
        VKC_ASSIGN(pos->offset, detail::resync(kApi, file, at, recovery_.end));
        return core::Status::invalid_argument(
            std::string(kApi) + ": " + path_ + ": a damaged chunk header at " +
            std::to_string(at) + "; reading resumes at " +
            std::to_string(pos->offset));
      }
      if (c.body_bytes > file_size_ - at - c.header_bytes) {
        pos->offset = recovery_.end;  // the torn tail, as the scan found it
        return false;
      }
      pos->offset =
          at + c.header_bytes + c.body_bytes + detail::padding(c.body_bytes);
      if (c.tag != detail::kRecordTag) continue;
      if ((c.flags & ~detail::kKeyFlag) != 0) {
        return core::Status::unsupported(
            std::string(kApi) + ": " + path_ + ": the record at " +
            std::to_string(at) + " sets flags " + std::to_string(c.flags) +
            ", which version 0 does not define");
      }
      record->info = c.record;
      record->offset = at;
      record->size = c.body_bytes;
      return true;
    }
    return false;
  } catch (...) {
    return core::exception_status(kApi, core::Status::Code::IoError);
  }
}

core::Status StreamReader::read(const Record& record,
                                std::vector<std::uint8_t>* payload) const {
  constexpr std::string_view kApi = "StreamReader::read";
  VKC_CHECK(payload != nullptr, "StreamReader::read takes a payload");
  try {
    const auto where = [&] {
      return std::string(kApi) + ": " + path_ + ": the record at " +
             std::to_string(record.offset);
    };
    if (record.size > options_.max_payload_bytes ||
        record.size > payload->max_size()) {
      return core::Status::out_of_memory(
          where() + " holds " + std::to_string(record.size) +
          " bytes, above the limit of " +
          std::to_string(options_.max_payload_bytes));
    }
    const detail::FileView file{
        fd_.get(), &path_, file_size_,
        static_cast<std::uint32_t>(session_.tracks.size())};
    detail::Chunk c;
    bool valid = false;
    if (record.offset % 8 == 0 && record.offset >= begin_) {
      VKC_ASSIGN(valid, detail::read_chunk(kApi, file, record.offset, &c));
    }
    const RecordInfo& info = record.info;
    if (!valid || c.tag != detail::kRecordTag || c.body_bytes != record.size ||
        c.body_bytes > file_size_ - record.offset - c.header_bytes ||
        c.record.track != info.track || c.record.key != info.key ||
        c.record.pts_ns != info.pts_ns || c.record.seq != info.seq ||
        c.record.set != info.set) {
      return core::Status::invalid_argument(where() +
                                            " does not match its header");
    }
    payload->resize(static_cast<std::size_t>(record.size));
    VKC_TRY(detail::read_at(kApi, path_, fd_.get(),
                            record.offset + c.header_bytes, payload->data(),
                            payload->size()));
    if (detail::crc32(payload->data(), payload->size()) != c.body_crc) {
      return core::Status::invalid_argument(where() +
                                            ": its payload fails its CRC");
    }
    return {};
  } catch (...) {
    return core::exception_status(kApi, core::Status::Code::IoError);
  }
}

}  // namespace volumetric_kit::recon::stream
