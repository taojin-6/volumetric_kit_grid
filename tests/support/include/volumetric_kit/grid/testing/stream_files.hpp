// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

// The stream tests' own .vrst bytes, built from the layout stream/session.hpp
// states and not from the library: a bitwise CRC-32, chunks with any fields,
// a footer index read and written, and the edits a test makes to a file. Also
// what the tests share of the library: payloads, sessions, record fields and
// stats compared, an open that says why it failed, and a reader's every
// record, read in file order as a player would.

#include <sys/types.h>
#include <unistd.h>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <initializer_list>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "volumetric_kit/core/testing/scratch_dir.hpp"
#include "volumetric_kit/grid/stream/stream_reader.hpp"

namespace vgrid_test::vrst {

namespace stream = volumetric_kit::grid::stream;
using Bytes = std::vector<std::uint8_t>;

// zlib's CRC-32, a bit at a time.
inline std::uint32_t crc32(const std::uint8_t* data, std::size_t size,
                           std::uint32_t crc = 0) {
  crc = ~crc;
  for (std::size_t i = 0; i < size; ++i) {
    crc ^= data[i];
    for (int k = 0; k < 8; ++k) {
      crc = (crc & 1u) != 0 ? (crc >> 1) ^ 0xEDB88320u : crc >> 1;
    }
  }
  return ~crc;
}
inline std::uint32_t crc32(const Bytes& b) { return crc32(b.data(), b.size()); }

// `v`'s low `bytes` bytes (at most 8), little-endian.
inline void put(Bytes& out, std::uint64_t v, int bytes) {
  for (int i = 0; i < bytes; ++i) {
    out.push_back(static_cast<std::uint8_t>(v >> (8 * i)));
  }
}
inline std::uint64_t get(const Bytes& b, std::uint64_t at, int bytes) {
  std::uint64_t v = 0;
  for (int i = 0; i < bytes; ++i) {
    v |= std::uint64_t{b[static_cast<std::size_t>(at) + i]} << (8 * i);
  }
  return v;
}

inline std::uint64_t padding(std::uint64_t n) { return (8 - n % 8) % 8; }

// A chunk header: `tag`, then `fields` (bytes 20 to H-4, so H is their size
// plus 24), claiming a body of `body_bytes` whose CRC is `body_crc`.
inline Bytes header(const char* tag, const Bytes& fields,
                    std::uint64_t body_bytes, std::uint32_t body_crc) {
  Bytes h(tag, tag + 4);
  put(h, 24 + fields.size(), 4);
  put(h, body_bytes, 8);
  put(h, body_crc, 4);
  h.insert(h.end(), fields.begin(), fields.end());
  put(h, crc32(h), 4);
  return h;
}

// A whole chunk: its header, `body` and the padding.
inline Bytes chunk(const char* tag, const Bytes& fields, const Bytes& body) {
  Bytes c = header(tag, fields, body.size(), body.empty() ? 0 : crc32(body));
  c.insert(c.end(), body.begin(), body.end());
  c.resize(c.size() + static_cast<std::size_t>(padding(body.size())), 0);
  return c;
}

// A `RECD`'s fields, with `extra` trailing bytes of fields a later version
// might add (H = 56 + extra).
inline Bytes record_fields(std::uint16_t track, std::uint16_t flags,
                           std::uint64_t pts, std::uint64_t seq,
                           std::uint64_t set, std::size_t extra = 0) {
  Bytes f;
  put(f, track, 2);
  put(f, flags, 2);
  put(f, pts, 8);
  put(f, seq, 8);
  put(f, set, 8);
  put(f, 0, 4);
  for (std::size_t i = 0; i < extra; ++i) {
    f.push_back(static_cast<std::uint8_t>(0xA0 + i));
  }
  return f;
}

inline Bytes record(std::uint16_t track, std::uint16_t flags, std::uint64_t pts,
                    std::uint64_t seq, std::uint64_t set, const Bytes& payload,
                    std::size_t extra = 0) {
  return chunk("RECD", record_fields(track, flags, pts, seq, set, extra),
               payload);
}

// A `VRST`'s fields: `version`, a zero id and time, `tracks`, `flags`.
inline Bytes session_fields(std::uint32_t version, std::uint16_t tracks,
                            std::uint16_t flags) {
  Bytes f;
  put(f, version, 4);
  f.resize(f.size() + 16 + 8, 0);
  put(f, tracks, 2);
  put(f, flags, 2);
  return f;
}

// A `VRST` body of one Geometry track per `tracks`, writer "t", no strings
// else.
inline Bytes session_body(std::uint16_t tracks) {
  Bytes b;
  for (std::uint16_t i = 0; i < tracks; ++i) {
    put(b, 1, 2);   // kind
    put(b, 1, 2);   // clock
    put(b, 30, 4);  // rate
    put(b, 1, 4);
    b.resize(b.size() + 12, 0);  // width, height, reserved
  }
  put(b, 1, 4);
  b.push_back('t');
  put(b, 0, 4);
  b.resize(b.size() + 8 * std::size_t{tracks}, 0);  // empty sensors, params
  return b;
}

// A footer index as its bytes say: the header's T and K, then the body's
// track summaries and key entries.
struct KeyEntry {
  std::uint64_t offset = 0;
  std::uint64_t pts = 0;
  std::uint64_t set = 0;
  std::uint16_t track = 0;
};
struct Index {
  std::uint32_t tracks = 0;
  std::uint64_t keys = 0;
  std::vector<stream::TrackStats> summaries;
  std::vector<KeyEntry> entries;
};

// The `INDX` chunk of `index`.
inline Bytes index_chunk(const Index& index) {
  Bytes fields;
  put(fields, index.tracks, 4);
  put(fields, index.keys, 8);
  put(fields, 0, 4);
  Bytes body;
  for (const stream::TrackStats& s : index.summaries) {
    for (const std::uint64_t v :
         {s.records, s.keys, s.payload_bytes, s.min_pts_ns, s.max_pts_ns}) {
      put(body, v, 8);
    }
  }
  for (const KeyEntry& e : index.entries) {
    put(body, e.offset, 8);
    put(body, e.pts, 8);
    put(body, e.set, 8);
    put(body, e.track, 2);
    put(body, 0, 6);
  }
  return chunk("INDX", fields, body);
}

// A `TAIL`'s fields, naming `index_offset`, and the `TAIL`.
inline Bytes tail_fields(std::uint64_t index_offset) {
  Bytes fields;
  put(fields, 0, 4);
  put(fields, index_offset, 8);
  put(fields, 0, 4);
  return fields;
}
inline Bytes tail_chunk(std::uint64_t index_offset) {
  return chunk("TAIL", tail_fields(index_offset), {});
}

// Where a finished file's `INDX` starts, as its `TAIL` says.
inline std::uint64_t index_offset(const Bytes& file) {
  return get(file, file.size() - 40 + 24, 8);
}

// A finished file's footer index.
inline Index read_index(const Bytes& file) {
  const std::uint64_t at = index_offset(file);
  Index index;
  index.tracks = static_cast<std::uint32_t>(get(file, at + 20, 4));
  index.keys = get(file, at + 24, 8);
  std::uint64_t p = at + get(file, at + 4, 4);
  for (std::uint32_t t = 0; t < index.tracks; ++t, p += 40) {
    index.summaries.push_back({get(file, p, 8), get(file, p + 8, 8),
                               get(file, p + 16, 8), get(file, p + 24, 8),
                               get(file, p + 32, 8)});
  }
  for (std::uint64_t k = 0; k < index.keys; ++k, p += 32) {
    index.entries.push_back({get(file, p, 8), get(file, p + 8, 8),
                             get(file, p + 16, 8),
                             static_cast<std::uint16_t>(get(file, p + 24, 2))});
  }
  return index;
}

// One chunk of a file, as its framing says.
struct ChunkAt {
  std::string tag;
  std::uint64_t offset = 0;
  std::uint64_t header_bytes = 0;
  std::uint64_t body_bytes = 0;
  std::uint64_t end() const { return offset + header_bytes + body_bytes; }
  std::uint64_t next() const { return end() + padding(body_bytes); }
};

// The chunks of a well-formed file, walked by their framing alone.
inline std::vector<ChunkAt> chunks(const Bytes& file) {
  std::vector<ChunkAt> out;
  for (std::uint64_t at = 0; at + 24 <= file.size();) {
    ChunkAt c;
    c.tag.assign(file.begin() + static_cast<std::ptrdiff_t>(at),
                 file.begin() + static_cast<std::ptrdiff_t>(at + 4));
    c.offset = at;
    c.header_bytes = get(file, at + 4, 4);
    c.body_bytes = get(file, at + 8, 8);
    out.push_back(c);
    at = c.next();
  }
  return out;
}

inline bool write_bytes(const std::string& path, const Bytes& bytes) {
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  out.write(reinterpret_cast<const char*>(bytes.data()),
            static_cast<std::streamsize>(bytes.size()));
  return static_cast<bool>(out);
}

// Flip bit `bit` of byte `offset` of a copy of `bytes`.
inline Bytes flipped(Bytes bytes, std::uint64_t offset, int bit) {
  bytes[static_cast<std::size_t>(offset)] ^=
      static_cast<std::uint8_t>(1u << bit);
  return bytes;
}

inline bool cut(const std::string& path, std::uint64_t size) {
  return ::truncate(path.c_str(), static_cast<off_t>(size)) == 0;
}

// Deterministic bytes, distinct per seed.
inline Bytes pattern(std::size_t size, std::uint64_t seed) {
  Bytes b(size);
  std::uint64_t x = seed * 0x9E3779B97F4A7C15ull + 1;
  for (std::size_t i = 0; i < size; ++i) {
    x = x * 6364136223846793005ull + 1442695040888963407ull;
    b[i] = static_cast<std::uint8_t>(x >> 56);
  }
  return b;
}

inline bool same(const stream::RecordInfo& a, const stream::RecordInfo& b) {
  return a.track == b.track && a.key == b.key && a.pts_ns == b.pts_ns &&
         a.seq == b.seq && a.set == b.set;
}

inline bool same(const stream::TrackStats& a, const stream::TrackStats& b) {
  return a.records == b.records && a.keys == b.keys &&
         a.payload_bytes == b.payload_bytes && a.min_pts_ns == b.min_pts_ns &&
         a.max_pts_ns == b.max_pts_ns;
}

inline bool same(const stream::Track& a, const stream::Track& b) {
  return a.kind == b.kind && a.clock == b.clock && a.rate.num == b.rate.num &&
         a.rate.den == b.rate.den && a.width == b.width &&
         a.height == b.height && a.sensor == b.sensor && a.params == b.params;
}

inline bool same(const stream::Session& a, const stream::Session& b) {
  if (a.id != b.id || a.created_ns != b.created_ns || a.writer != b.writer ||
      a.calibration != b.calibration || a.tracks.size() != b.tracks.size()) {
    return false;
  }
  for (std::size_t i = 0; i < a.tracks.size(); ++i) {
    if (!same(a.tracks[i], b.tracks[i])) return false;
  }
  return true;
}

inline stream::RecordInfo info(std::uint16_t track, bool key, std::uint64_t pts,
                               std::uint64_t seq,
                               std::uint64_t set = stream::kNoSet) {
  stream::RecordInfo i;
  i.track = track;
  i.key = key;
  i.pts_ns = pts;
  i.seq = seq;
  i.set = set;
  return i;
}

// The reader of `path`, or null, with the reason on stderr.
inline std::unique_ptr<stream::StreamReader> open(
    const std::string& path, const stream::ReaderOptions& options = {}) {
  volumetric_kit::core::Result<std::unique_ptr<stream::StreamReader>> r =
      stream::StreamReader::open(path, options);
  if (!r.ok()) {
    std::fprintf(stderr, "open %s: %s\n", path.c_str(),
                 r.status().message().c_str());
    return nullptr;
  }
  return std::move(r).value();
}

// What reading every record of a reader in file order found.
struct ReadBack {
  std::vector<stream::Record> records;  // each next() returned
  std::vector<Bytes> payloads;          // each read() that passed
  std::vector<volumetric_kit::core::Status::Code> next_errors;
  std::vector<volumetric_kit::core::Status::Code> read_errors;
  bool ended = false;  // next() said the end
};

inline ReadBack read_all(const stream::StreamReader& reader,
                         stream::Position pos) {
  ReadBack back;
  for (int guard = 0; guard < 1 << 20; ++guard) {
    stream::Record record;
    const volumetric_kit::core::Result<bool> more = reader.next(&pos, &record);
    if (!more.ok()) {
      back.next_errors.push_back(more.status().domain());
      continue;
    }
    if (!more.value()) {
      back.ended = true;
      break;
    }
    back.records.push_back(record);
    Bytes payload;
    const volumetric_kit::core::Status read = reader.read(record, &payload);
    if (read.ok()) {
      back.payloads.push_back(std::move(payload));
    } else {
      back.read_errors.push_back(read.domain());
    }
  }
  return back;
}
inline ReadBack read_all(const stream::StreamReader& reader) {
  return read_all(reader, reader.begin());
}

}  // namespace vgrid_test::vrst
