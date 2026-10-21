// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include "volumetric_kit/recon/stream/chunks.hpp"

#include <algorithm>
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
  constexpr std::string_view kApi = "ChunkEncoder::frame";
  try {
    const auto refuse = [&](const std::string& why) {
      return core::Status::invalid_argument(std::string(kApi) + ": " + why);
    };
    if (info.track >= state_.size()) {
      return refuse("track " + std::to_string(info.track) +
                    ", but the session has " + std::to_string(state_.size()));
    }
    if (data == nullptr && size != 0) {
      return refuse("a null payload of " + std::to_string(size) + " bytes");
    }
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
    detail::encode_record_header(info, size, detail::crc32(data, size),
                                 framing.header);
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
  try {
    if (!failure_.ok()) return failure_;
    // What was passed goes, and so does any padding still to come at the
    // start of these bytes.
    const auto passed = static_cast<std::size_t>(
        std::min<std::uint64_t>(next_ - base_, buffer_.size()));
    buffer_.erase(buffer_.begin(),
                  buffer_.begin() + static_cast<std::ptrdiff_t>(passed));
    base_ += passed;
    const auto padding =
        static_cast<std::size_t>(std::min<std::uint64_t>(next_ - base_, size));
    base_ += padding;
    if (size > padding) {
      buffer_.insert(buffer_.end(), data + padding, data + size);
    }
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
  VKC_CHECK(record != nullptr && payload != nullptr,
            "ChunkParser::next takes a record and a payload");
  try {
    if (!failure_.ok()) return failure_;
    VKC_ASSIGN(ready_, advance());
    if (!ready_) return false;
    const std::uint8_t* const at =
        buffer_.data() + static_cast<std::size_t>(next_ - base_);
    detail::Chunk c;
    (void)detail::decode_chunk(
        at, detail::load_le32(at + 4),
        static_cast<std::uint32_t>(session_->tracks.size()), &c);
    const std::uint8_t* const body = at + c.header_bytes;
    payload->assign(body, body + static_cast<std::size_t>(c.body_bytes));
    record->info = c.record;
    record->offset = next_;
    record->size = c.body_bytes;
    next_ += c.header_bytes + c.body_bytes + detail::padding(c.body_bytes);
    ready_ = false;
    return true;
  } catch (...) {
    return fail_on_exception();
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
    if (c.tag == detail::kRecordTag && (c.flags & ~detail::kKeyFlag) == 0) {
      return true;
    }
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
