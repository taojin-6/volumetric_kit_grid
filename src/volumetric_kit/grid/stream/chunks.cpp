// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include "volumetric_kit/recon/stream/chunks.hpp"

#include <algorithm>
#include <cstring>
#include <string>
#include <string_view>
#include <utility>

#include "chunk.hpp"
#include "volumetric_kit/core/base/check.hpp"
#include "volumetric_kit/core/base/exception_status.hpp"

namespace volumetric_kit::recon::stream {
namespace {

static_assert(sizeof(ChunkEncoder::RecordFraming::header) ==
                  detail::kRecordHeaderBytes,
              "a record's framing holds its RECD header");

// The parser's messages name it and the bytes, as a reader's name the call
// and the file.
constexpr std::string_view kParser = "ChunkParser";
constexpr std::string_view kStream = "the stream";

// No bound on the bytes still to come, for header_fits.
constexpr std::uint64_t kUnbounded = ~std::uint64_t{0};

}  // namespace

RecordPayload RecordPayload::of(const std::uint8_t* first,
                                std::size_t first_size,
                                const std::uint8_t* second,
                                std::size_t second_size) noexcept {
  VKC_CHECK((first != nullptr || first_size == 0) &&
                (second != nullptr || second_size == 0),
            "RecordPayload::of takes bytes for every run with a size");
  RecordPayload payload;
  payload.first_ = first;
  payload.first_size_ = first_size;
  payload.second_ = second;
  payload.second_size_ = second_size;
  // The second run's CRC continues the first's, as zlib's does.
  payload.crc_ =
      detail::crc32(second, second_size, detail::crc32(first, first_size));
  return payload;
}

ChunkEncoder::ChunkEncoder(std::vector<std::uint8_t> session_chunk,
                           std::size_t tracks)
    : session_chunk_(std::move(session_chunk)), state_(tracks) {}

core::Result<ChunkEncoder> ChunkEncoder::create(const Session& session) {
  constexpr std::string_view kApi = "ChunkEncoder::create";
  try {
    std::vector<std::uint8_t> chunk;
    VKC_TRY(detail::encode_session(kApi, session, &chunk));
    return ChunkEncoder(std::move(chunk), session.tracks.size());
  } catch (...) {
    return core::exception_status(kApi, core::Status::Code::InvalidArgument);
  }
}

const std::vector<std::uint8_t>& ChunkEncoder::session_chunk() const noexcept {
  return session_chunk_;
}

core::Result<ChunkEncoder::RecordFraming> ChunkEncoder::frame(
    const RecordInfo& info, const std::uint8_t* data, std::size_t size) {
  if (data == nullptr && size != 0) return detail::null_payload(size);
  return frame(info, RecordPayload::of(data, size));
}

core::Result<ChunkEncoder::RecordFraming> ChunkEncoder::frame(
    const RecordInfo& info, const RecordPayload& payload) {
  constexpr std::string_view kApi = "ChunkEncoder::frame";
  try {
    const auto refuse = [&](const std::string& why) {
      return core::Status::invalid_argument(std::string(kApi) + ": " + why);
    };
    if (info.track >= state_.size()) {
      return refuse("track " + std::to_string(info.track) +
                    ", but the session has " + std::to_string(state_.size()));
    }
    const std::size_t size = payload.size();
    TrackState& state = state_[info.track];
    if (state.any && info.seq <= state.last_seq) {
      return refuse("track " + std::to_string(info.track) + "'s seq " +
                    std::to_string(info.seq) + " does not follow its last, " +
                    std::to_string(state.last_seq));
    }
    if (info.key && state.any_key && info.pts_ns < state.last_key_pts_ns) {
      return refuse("track " + std::to_string(info.track) + "'s key pts " +
                    std::to_string(info.pts_ns) +
                    " is before its last key's, " +
                    std::to_string(state.last_key_pts_ns));
    }
    RecordFraming framing;
    detail::encode_record_header(info, size, payload.crc(), framing.header);
    framing.padding = static_cast<std::size_t>(detail::padding(size));
    state.any = true;
    state.last_seq = info.seq;
    if (info.key) {
      state.any_key = true;
      state.last_key_pts_ns = info.pts_ns;
    }
    return framing;
  } catch (...) {
    return core::exception_status(kApi, core::Status::Code::InvalidArgument);
  }
}

core::Status ChunkParser::feed(const std::uint8_t* data, std::size_t size) {
  VKC_CHECK(data != nullptr || size == 0, "ChunkParser::feed takes bytes");
  try {
    if (!failure_.ok()) return failure_;
    // What was passed goes.
    const auto passed = static_cast<std::size_t>(
        std::min<std::uint64_t>(next_ - base_, buffer_.size()));
    buffer_.erase(buffer_.begin(),
                  buffer_.begin() + static_cast<std::ptrdiff_t>(passed));
    base_ += passed;
    buffer_.insert(buffer_.end(), data, data + size);
    VKC_ASSIGN(ready_, advance());
    return {};
  } catch (...) {
    return fail_on_exception();
  }
}

const Session* ChunkParser::session() const noexcept {
  return session_ ? &*session_ : nullptr;
}

core::Result<bool> ChunkParser::next(Record* record,
                                     std::vector<std::uint8_t>* payload) {
  VKC_CHECK(record != nullptr, "ChunkParser::next takes a record");
  try {
    if (!failure_.ok()) return failure_;
    // A record returned with no payload and never read is dropped.
    if (held_ != 0) {
      next_ += held_;
      held_ = 0;
      ready_ = false;
    }
    VKC_ASSIGN(ready_, advance());
    if (!ready_) return false;
    const std::uint8_t* const at =
        buffer_.data() + static_cast<std::size_t>(next_ - base_);
    detail::Chunk c;
    (void)detail::decode_chunk(
        at, detail::load_le32(at + 4),
        static_cast<std::uint32_t>(session_->tracks.size()), &c);
    record->info = c.record;
    record->offset = next_;
    record->size = c.body_bytes;
    held_ = c.header_bytes + c.body_bytes + detail::padding(c.body_bytes);
    if (payload == nullptr) return true;
    payload->resize(static_cast<std::size_t>(c.body_bytes));
    VKC_TRY(read(*record, payload->data(), payload->size()));
    return true;
  } catch (...) {
    return fail_on_exception();
  }
}

core::Status ChunkParser::read(const Record& record, std::uint8_t* dst,
                               std::size_t capacity) {
  try {
    if (!failure_.ok()) return failure_;
    if (held_ == 0 || record.offset != next_) {
      return core::Status::invalid_argument(
          "ChunkParser::read: the record is not the one next returned last");
    }
    const std::uint8_t* const at =
        buffer_.data() + static_cast<std::size_t>(next_ - base_);
    detail::Chunk c;
    (void)detail::decode_chunk(
        at, detail::load_le32(at + 4),
        static_cast<std::uint32_t>(session_->tracks.size()), &c);
    const RecordInfo& info = record.info;
    if (c.body_bytes != record.size || c.record.track != info.track ||
        c.record.key != info.key || c.record.pts_ns != info.pts_ns ||
        c.record.seq != info.seq || c.record.set != info.set) {
      return core::Status::invalid_argument(
          "ChunkParser::read: the record does not match its held header");
    }
    if (capacity < record.size || (dst == nullptr && record.size != 0)) {
      return core::Status::invalid_argument(
          "ChunkParser::read: the record holds " + std::to_string(record.size) +
          " bytes, more than the " + std::to_string(capacity) +
          " bytes given (or none)");
    }
    if (record.size != 0) {
      std::memcpy(dst, at + c.header_bytes,
                  static_cast<std::size_t>(c.body_bytes));
    }
    next_ += held_;
    held_ = 0;
    ready_ = false;
    return {};
  } catch (...) {
    // Only a refusal's message allocates; the record stays held.
    return core::exception_status(kParser, core::Status::Code::InvalidArgument);
  }
}

core::Result<bool> ChunkParser::advance() {
  if (ready_) return true;
  for (;;) {
    if (next_ - base_ > buffer_.size()) return false;  // padding to come
    const auto skip = static_cast<std::size_t>(next_ - base_);
    const std::uint8_t* const at = buffer_.data() + skip;
    const std::size_t available = buffer_.size() - skip;
    if (available < 8) return false;
    if (!session_) {
      core::Result<std::uint32_t> h =
          detail::session_header_bytes(kParser, kStream, at, kUnbounded);
      if (!h.ok()) return fail(std::move(h).status());
      if (available < *h) return false;
      core::Result<std::uint64_t> size =
          detail::session_body_bytes(kParser, kStream, at, *h);
      if (!size.ok()) return fail(std::move(size).status());
      if (available - *h < *size) return false;
      Session session;
      core::Status parsed =
          detail::parse_session(kParser, kStream, at, at + *h, *size, &session);
      if (!parsed.ok()) return fail(std::move(parsed));
      session_ = std::move(session);
      next_ += *h + *size + detail::padding(*size);
      continue;
    }
    const auto where = [&] {
      return std::string(kParser) + ": " + std::string(kStream) +
             ": the chunk at " + std::to_string(next_);
    };
    const std::uint32_t h = detail::load_le32(at + 4);
    if (!detail::header_fits(
            h, detail::least_header_bytes(detail::load_le32(at)), kUnbounded)) {
      return fail(core::Status::invalid_argument(where() + " has a header of " +
                                                 std::to_string(h) +
                                                 " bytes, out of range"));
    }
    if (available < h) return false;
    if (!detail::header_crc_ok(at, h)) {
      return fail(core::Status::invalid_argument(
          where() + " has a header that fails its CRC"));
    }
    detail::Chunk c;
    const auto tracks = static_cast<std::uint32_t>(session_->tracks.size());
    if (!detail::decode_chunk(at, h, tracks, &c)) {
      return fail(core::Status::invalid_argument(
          where() + " is a record of track " + std::to_string(c.record.track) +
          ", but the session has " + std::to_string(tracks)));
    }
    if (c.body_bytes > kMaxBodyBytes) {
      return fail(core::Status::out_of_memory(
          where() + " holds " + std::to_string(c.body_bytes) +
          " bytes, above the limit of " + std::to_string(kMaxBodyBytes)));
    }
    if (available - h < c.body_bytes) return false;
    if (detail::crc32(at + h, static_cast<std::size_t>(c.body_bytes)) !=
        c.body_crc) {
      return fail(
          core::Status::invalid_argument(where() + ": its body fails its CRC"));
    }
    if (detail::known_record(c)) return true;
    // Another tag, or a record setting a flag this version does not know.
    next_ += h + c.body_bytes + detail::padding(c.body_bytes);
  }
}

core::Status ChunkParser::fail(core::Status failure) {
  failure_ = std::move(failure);
  return failure_;
}

core::Status ChunkParser::fail_on_exception() noexcept {
  // Saved and returned, each by its own call: a copy may find no memory.
  if (failure_.ok()) {
    failure_ =
        core::exception_status(kParser, core::Status::Code::InvalidArgument);
  }
  return core::exception_status(kParser, core::Status::Code::InvalidArgument);
}

}  // namespace volumetric_kit::recon::stream
