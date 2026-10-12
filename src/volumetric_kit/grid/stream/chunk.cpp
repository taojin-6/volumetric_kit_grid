// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include "chunk.hpp"

#include <algorithm>
#include <array>
#include <cstring>
#include <limits>

#include "file.hpp"

namespace volumetric_kit::recon::stream::detail {
namespace {

// --- CRC-32, sliced by 8. ----------------------------------------------------
//
// Table s holds the CRC of a byte followed by s zero bytes, so eight bytes
// fold into the CRC with eight lookups rather than eight dependent steps.

using CrcTables = std::array<std::array<std::uint32_t, 256>, 8>;

constexpr CrcTables make_crc_tables() {
  CrcTables t{};
  for (std::uint32_t i = 0; i < 256; ++i) {
    std::uint32_t c = i;
    for (int k = 0; k < 8; ++k) {
      c = (c & 1u) != 0 ? (c >> 1) ^ 0xEDB88320u : c >> 1;
    }
    t[0][i] = c;
  }
  for (std::size_t s = 1; s < 8; ++s) {
    for (std::size_t i = 0; i < 256; ++i) {
      t[s][i] = (t[s - 1][i] >> 8) ^ t[0][t[s - 1][i] & 0xFFu];
    }
  }
  return t;
}

constexpr CrcTables kCrcTables = make_crc_tables();

// --- Framing. ----------------------------------------------------------------

constexpr std::size_t kHeaderSizeAt = 4;
constexpr std::size_t kBodySizeAt = 8;
constexpr std::size_t kBodyCrcAt = 16;
// The session header's fields.
constexpr std::size_t kVersionAt = 20;
constexpr std::size_t kSessionIdAt = 24;
constexpr std::size_t kCreatedAt = 40;
constexpr std::size_t kTrackCountAt = 48;
constexpr std::size_t kSessionFlagsAt = 50;
// A record's.
constexpr std::size_t kTrackAt = 20;
constexpr std::size_t kRecordFlagsAt = 22;
constexpr std::size_t kPtsAt = 24;
constexpr std::size_t kSeqAt = 32;
constexpr std::size_t kSetAt = 40;
// The index's and the tail's.
constexpr std::size_t kIndexTracksAt = 20;
constexpr std::size_t kIndexKeysAt = 24;
constexpr std::size_t kIndexOffsetAt = 24;

// How much a scan reads first of a chunk header: all of a record's.
constexpr std::size_t kFirstRead = 64;
// What a resync reads at a time.
constexpr std::size_t kResyncBlock = std::size_t{1} << 20;

// Fill the framing of a header of `header_bytes` whose own fields are in
// place, then its CRC.
void seal_header(std::uint8_t* header, std::uint32_t tag,
                 std::uint32_t header_bytes, std::uint64_t body_bytes,
                 std::uint32_t body_crc) {
  store_le32(header, tag);
  store_le32(header + kHeaderSizeAt, header_bytes);
  store_le64(header + kBodySizeAt, body_bytes);
  store_le32(header + kBodyCrcAt, body_crc);
  store_le32(header + header_bytes - 4, crc32(header, header_bytes - 4));
}

core::Status bad_session(std::string_view api, const std::string& path,
                         const std::string& why) {
  return core::Status::invalid_argument(std::string(api) + ": " + path +
                                        ": the session header " + why);
}

// Bounds-checked reads through the session body; a read past its end sets
// `failed` and returns 0.
struct BodyReader {
  const std::uint8_t* data;
  std::uint64_t size;
  std::uint64_t pos = 0;
  bool failed = false;

  const std::uint8_t* take(std::uint64_t n) {
    if (failed || n > size - pos) {
      failed = true;
      return nullptr;
    }
    const std::uint8_t* at = data + pos;
    pos += n;
    return at;
  }
  std::uint16_t u16() {
    const std::uint8_t* p = take(2);
    return p != nullptr ? load_le16(p) : 0;
  }
  std::uint32_t u32() {
    const std::uint8_t* p = take(4);
    return p != nullptr ? load_le32(p) : 0;
  }
  std::string string() {
    const std::uint32_t n = u32();
    const std::uint8_t* p = take(n);
    return p != nullptr ? std::string(reinterpret_cast<const char*>(p), n)
                        : std::string();
  }
};

void put_string(std::uint8_t*& at, const std::string& s) {
  store_le32(at, static_cast<std::uint32_t>(s.size()));
  if (!s.empty()) std::memcpy(at + 4, s.data(), s.size());
  at += 4 + s.size();
}

}  // namespace

std::uint32_t crc32(const void* data, std::size_t size,
                    std::uint32_t crc) noexcept {
  const auto* p = static_cast<const std::uint8_t*>(data);
  const CrcTables& t = kCrcTables;
  std::uint32_t c = ~crc;
  for (; size >= 8; size -= 8, p += 8) {
    const std::uint32_t lo = load_le32(p) ^ c;
    const std::uint32_t hi = load_le32(p + 4);
    c = t[7][lo & 0xFFu] ^ t[6][(lo >> 8) & 0xFFu] ^ t[5][(lo >> 16) & 0xFFu] ^
        t[4][lo >> 24] ^ t[3][hi & 0xFFu] ^ t[2][(hi >> 8) & 0xFFu] ^
        t[1][(hi >> 16) & 0xFFu] ^ t[0][hi >> 24];
  }
  for (; size > 0; --size, ++p) c = t[0][(c ^ *p) & 0xFFu] ^ (c >> 8);
  return ~c;
}

core::Status encode_session(std::string_view api, const Session& session,
                            std::vector<std::uint8_t>* chunk) {
  const auto refuse = [&](const std::string& why) {
    return core::Status::invalid_argument(std::string(api) + ": " + why);
  };
  const std::size_t tracks = session.tracks.size();
  if (tracks == 0 || tracks > std::numeric_limits<std::uint16_t>::max()) {
    return refuse("a session holds 1 to 65535 tracks, not " +
                  std::to_string(tracks));
  }
  std::uint64_t body = kTrackEntryBytes * tracks + 8 + session.writer.size() +
                       session.calibration.size();
  for (std::size_t i = 0; i < tracks; ++i) {
    const Track& track = session.tracks[i];
    if (static_cast<std::uint16_t>(track.kind) == 0) {
      return refuse("track " + std::to_string(i) + " has kind 0");
    }
    if (track.rate.den == 0) {
      return refuse("track " + std::to_string(i) + " has a rate_den of 0");
    }
    body += 8 + std::uint64_t{track.sensor.size()} + track.params.size();
  }
  if (body > kMaxSessionBytes) {
    return refuse("the session header's body would hold " +
                  std::to_string(body) + " bytes, above " +
                  std::to_string(kMaxSessionBytes));
  }
  chunk->assign(
      static_cast<std::size_t>(kSessionHeaderBytes + body + padding(body)), 0);
  std::uint8_t* header = chunk->data();
  store_le32(header + kVersionAt, kSessionVersion);
  std::memcpy(header + kSessionIdAt, session.id.data(), session.id.size());
  store_le64(header + kCreatedAt, session.created_ns);
  store_le16(header + kTrackCountAt, static_cast<std::uint16_t>(tracks));
  store_le16(header + kSessionFlagsAt, 0);
  std::uint8_t* const start = header + kSessionHeaderBytes;
  std::uint8_t* at = start;
  for (const Track& track : session.tracks) {
    store_le16(at, static_cast<std::uint16_t>(track.kind));
    store_le16(at + 2, track.clock);
    store_le32(at + 4, track.rate.num);
    store_le32(at + 8, track.rate.den);
    store_le32(at + 12, track.width);
    store_le32(at + 16, track.height);
    at += kTrackEntryBytes;
  }
  put_string(at, session.writer);
  put_string(at, session.calibration);
  for (const Track& track : session.tracks) {
    put_string(at, track.sensor);
    put_string(at, track.params);
  }
  seal_header(header, kSessionTag, kSessionHeaderBytes, body,
              crc32(start, static_cast<std::size_t>(body)));
  return {};
}

void encode_record_header(const RecordInfo& info, std::uint64_t size,
                          std::uint32_t body_crc,
                          std::uint8_t (&header)[kRecordHeaderBytes]) {
  std::memset(header, 0, sizeof(header));
  store_le16(header + kTrackAt, info.track);
  store_le16(header + kRecordFlagsAt, info.key ? kKeyFlag : 0);
  store_le64(header + kPtsAt, info.pts_ns);
  store_le64(header + kSeqAt, info.seq);
  store_le64(header + kSetAt, info.set);
  seal_header(header, kRecordTag, kRecordHeaderBytes, size, body_crc);
}

void append_key_entry(std::vector<std::uint8_t>& entries, std::uint64_t offset,
                      const RecordInfo& info) {
  const std::size_t at = entries.size();
  entries.resize(at + kKeyEntryBytes, 0);
  std::uint8_t* p = entries.data() + at;
  store_le64(p, offset);
  store_le64(p + 8, info.pts_ns);
  store_le64(p + 16, info.set);
  store_le16(p + 24, info.track);
}

std::vector<std::uint8_t> encode_footer(
    const std::vector<TrackStats>& stats,
    const std::vector<std::uint8_t>& key_entries, std::uint64_t index_offset) {
  const std::uint64_t body = kSummaryBytes * stats.size() + key_entries.size();
  std::vector<std::uint8_t> out(
      static_cast<std::size_t>(kIndexHeaderBytes + body + kTailBytes), 0);
  std::uint8_t* index = out.data();
  store_le32(index + kIndexTracksAt, static_cast<std::uint32_t>(stats.size()));
  store_le64(index + kIndexKeysAt, key_entries.size() / kKeyEntryBytes);
  std::uint8_t* at = index + kIndexHeaderBytes;
  for (const TrackStats& s : stats) {
    store_le64(at, s.records);
    store_le64(at + 8, s.keys);
    store_le64(at + 16, s.payload_bytes);
    store_le64(at + 24, s.min_pts_ns);
    store_le64(at + 32, s.max_pts_ns);
    at += kSummaryBytes;
  }
  if (!key_entries.empty()) {
    std::memcpy(at, key_entries.data(), key_entries.size());
  }
  seal_header(index, kIndexTag, kIndexHeaderBytes, body,
              crc32(index + kIndexHeaderBytes, static_cast<std::size_t>(body)));
  std::uint8_t* tail = index + kIndexHeaderBytes + body;
  store_le64(tail + kIndexOffsetAt, index_offset);
  seal_header(tail, kTailTag, kTailBytes, 0, 0);
  return out;
}

void count_record(TrackStats& stats, const RecordInfo& info,
                  std::uint64_t size) {
  if (stats.records == 0) {
    stats.min_pts_ns = stats.max_pts_ns = info.pts_ns;
  } else {
    stats.min_pts_ns = std::min(stats.min_pts_ns, info.pts_ns);
    stats.max_pts_ns = std::max(stats.max_pts_ns, info.pts_ns);
  }
  ++stats.records;
  if (info.key) ++stats.keys;
  stats.payload_bytes += size;
}

core::Result<bool> read_chunk(std::string_view api, const FileView& file,
                              std::uint64_t offset, Chunk* out) {
  if (offset >= file.size || file.size - offset < kMinHeaderBytes) {
    return false;
  }
  const std::uint64_t available = file.size - offset;
  std::uint8_t header[kMaxHeaderBytes];
  const auto first =
      static_cast<std::size_t>(std::min<std::uint64_t>(kFirstRead, available));
  VKC_TRY(read_at(api, *file.path, file.fd, offset, header, first));
  const std::uint32_t h = load_le32(header + kHeaderSizeAt);
  if (h % 8 != 0 || h < kMinHeaderBytes || h > kMaxHeaderBytes ||
      h > available) {
    return false;
  }
  if (h > first) {
    VKC_TRY(read_at(api, *file.path, file.fd, offset + first, header + first,
                    h - first));
  }
  if (crc32(header, h - 4) != load_le32(header + h - 4)) return false;
  Chunk c;
  c.tag = load_le32(header);
  c.header_bytes = h;
  c.body_bytes = load_le64(header + kBodySizeAt);
  c.body_crc = load_le32(header + kBodyCrcAt);
  if (c.tag == kRecordTag) {
    if (h < kRecordHeaderBytes) return false;
    c.record.track = load_le16(header + kTrackAt);
    c.flags = load_le16(header + kRecordFlagsAt);
    c.record.key = (c.flags & kKeyFlag) != 0;
    c.record.pts_ns = load_le64(header + kPtsAt);
    c.record.seq = load_le64(header + kSeqAt);
    c.record.set = load_le64(header + kSetAt);
    if (c.record.track >= file.tracks) return false;
  } else if (c.tag == kIndexTag) {
    if (h < kIndexHeaderBytes) return false;
    c.tracks = load_le32(header + kIndexTracksAt);
    c.keys = load_le64(header + kIndexKeysAt);
  } else if (c.tag == kTailTag) {
    if (h < kTailBytes) return false;
    c.index_offset = load_le64(header + kIndexOffsetAt);
  }
  *out = c;
  return true;
}

core::Result<std::uint64_t> resync(std::string_view api, const FileView& file,
                                   std::uint64_t from, std::uint64_t end) {
  std::vector<std::uint8_t> block(kResyncBlock);
  for (std::uint64_t at = from + 8; at < end; at += kResyncBlock) {
    const auto n = static_cast<std::size_t>(
        std::min<std::uint64_t>(kResyncBlock, file.size - at));
    VKC_TRY(read_at(api, *file.path, file.fd, at, block.data(), n));
    for (std::size_t i = 0; i + 4 <= n && at + i < end; i += 8) {
      const std::uint32_t tag = load_le32(block.data() + i);
      if (tag != kRecordTag && tag != kIndexTag && tag != kTailTag) continue;
      Chunk c;
      VKC_ASSIGN(const bool valid, read_chunk(api, file, at + i, &c));
      if (valid) return at + i;
    }
  }
  return end;
}

core::Result<std::uint32_t> decode_session_header(std::string_view api,
                                                  const std::string& path,
                                                  const std::uint8_t* header,
                                                  Session* session) {
  const std::uint32_t version = load_le32(header + kVersionAt);
  if (version != kSessionVersion) {
    return core::Status::unsupported(std::string(api) + ": " + path +
                                     " is version " + std::to_string(version) +
                                     "; this library reads version 0");
  }
  const std::uint16_t flags = load_le16(header + kSessionFlagsAt);
  if (flags != 0) {
    return core::Status::unsupported(
        std::string(api) + ": " + path + " sets session flags " +
        std::to_string(flags) + ", which version 0 does not define");
  }
  std::memcpy(session->id.data(), header + kSessionIdAt, session->id.size());
  session->created_ns = load_le64(header + kCreatedAt);
  return std::uint32_t{load_le16(header + kTrackCountAt)};
}

core::Status decode_session_body(std::string_view api, const std::string& path,
                                 const std::uint8_t* body, std::uint64_t size,
                                 std::uint32_t tracks, Session* session) {
  if (tracks == 0) return bad_session(api, path, "has no tracks");
  BodyReader in{body, size};
  session->tracks.resize(tracks);
  for (std::uint32_t i = 0; i < tracks; ++i) {
    Track& track = session->tracks[i];
    track.kind = static_cast<TrackKind>(in.u16());
    track.clock = in.u16();
    track.rate.num = in.u32();
    track.rate.den = in.u32();
    track.width = in.u32();
    track.height = in.u32();
    (void)in.u32();  // reserved
    if (in.failed) break;
    if (static_cast<std::uint16_t>(track.kind) == 0) {
      return bad_session(api, path,
                         "gives track " + std::to_string(i) + " kind 0");
    }
    if (track.rate.den == 0) {
      return bad_session(
          api, path, "gives track " + std::to_string(i) + " a rate_den of 0");
    }
  }
  session->writer = in.string();
  session->calibration = in.string();
  for (Track& track : session->tracks) {
    track.sensor = in.string();
    track.params = in.string();
  }
  if (in.failed) return bad_session(api, path, "body ends inside a field");
  if (in.pos != size) {
    return bad_session(api, path,
                       "body has " + std::to_string(size - in.pos) +
                           " bytes after its last string");
  }
  return {};
}

}  // namespace volumetric_kit::recon::stream::detail
