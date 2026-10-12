// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

// The stream tests' own .vrst bytes, built from the layout stream/session.hpp
// states and not from the library: a bitwise CRC-32, chunks with any fields,
// and the edits a test makes to a file. Also what both tests share of the
// library: payloads, record fields, an open that says why it failed, and a
// reader's every record, read in file order as a player would.

#include <sys/types.h>
#include <unistd.h>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "scratch_dir.hpp"
#include "volumetric_kit/recon/stream/stream_reader.hpp"

namespace vr_test::vrst {

namespace stream = volumetric_kit::recon::stream;
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

}  // namespace vr_test::vrst
