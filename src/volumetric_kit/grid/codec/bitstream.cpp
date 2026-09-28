// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include "bitstream.hpp"

#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <string>
#include <utility>

#include "rans.hpp"

namespace volumetric_kit::recon::codec::detail {
namespace {

// --- The models: one frequency table each, in this order in TABLES. ---------
//
// The coordinate models are split by what the sort already guarantees about
// the delta: after a block in the same (y, z) row the x step is at least 1
// (kDxRun codes step - 1, so a run is a stream of zeros), within the same z
// slice the y step is non-negative (kDySame), and after a step in z the other
// two are free to go either way.
enum Model : std::uint32_t {
  kDz = 0,     // unsigned: z step
  kDySame,     // unsigned: y step, z unchanged
  kDyFree,     // signed:   y step after a z step
  kDxRun,      // unsigned: x step - 1, y and z unchanged
  kDxFree,     // signed:   x step after a y or z step
  kMaskClass,  // 0 all observed, 1 none observed, 2 partial
  kMaskByte,   // a partial mask's bytes
  kFirstCoef,  // then one model per coefficient index j < K
};

// A class is a bit length: 0 for zero, else 1 + floor(log2 |v|). Coordinate
// deltas reach 2^32 - 1 (INT32_MIN to INT32_MAX), so 33 classes; quantized
// coefficients are within +-32767 < 2^15, so 16.
constexpr std::uint32_t kCoordClasses = 33;
constexpr std::uint32_t kCoefClasses = 16;
constexpr std::uint32_t kMaskClasses = 3;
constexpr std::uint32_t kMaskByteSymbols = 256;
constexpr std::uint32_t kMaskFull = 0;
constexpr std::uint32_t kMaskEmpty = 1;
constexpr std::uint32_t kMaskPartial = 2;

std::uint32_t model_count(std::uint32_t k) { return kFirstCoef + k; }

std::uint32_t alphabet(std::uint32_t model) {
  switch (model) {
    case kDz:
    case kDySame:
    case kDyFree:
    case kDxRun:
    case kDxFree:
      return kCoordClasses;
    case kMaskClass:
      return kMaskClasses;
    case kMaskByte:
      return kMaskByteSymbols;
    default:
      return kCoefClasses;
  }
}

std::uint32_t bit_length(std::uint64_t v) {
  std::uint32_t n = 0;
  while (v != 0) {
    ++n;
    v >>= 1;
  }
  return n;
}

// --- The one description of a block's symbols.
// --------------------------------
//
// Templated on a sink so the histogram pass and the encode pass walk exactly
// the same symbols: a table built from one walk and a stream written by
// another could only disagree by construction, never by accident. The reader
// below mirrors it by hand, and the round-trip tests are what hold the two
// together.

template <typename Sink>
void emit_unsigned(Sink& sink, std::uint32_t model, std::uint64_t u) {
  const std::uint32_t c = bit_length(u);
  sink.symbol(model, c);
  if (c > 1) {
    sink.bits(static_cast<std::uint32_t>(u - (std::uint64_t(1) << (c - 1))),
              c - 1);
  }
}

template <typename Sink>
void emit_signed(Sink& sink, std::uint32_t model, std::int64_t v) {
  const std::uint64_t m = v < 0 ? std::uint64_t(-v) : std::uint64_t(v);
  const std::uint32_t c = bit_length(m);
  sink.symbol(model, c);
  if (c > 0) {
    sink.bits(v < 0 ? 1u : 0u, 1);
    if (c > 1) {
      sink.bits(static_cast<std::uint32_t>(m - (std::uint64_t(1) << (c - 1))),
                c - 1);
    }
  }
}

template <typename Sink>
void emit_block(Sink& sink, const Vec3i* prev, const Vec3i& cur,
                const std::uint32_t* mask, const std::int32_t* coeffs,
                std::uint32_t k) {
  if (prev == nullptr) {
    // A segment's first block, in full: the segment decodes on its own.
    sink.bits(static_cast<std::uint32_t>(cur.x), 32);
    sink.bits(static_cast<std::uint32_t>(cur.y), 32);
    sink.bits(static_cast<std::uint32_t>(cur.z), 32);
  } else {
    const std::int64_t dz = std::int64_t(cur.z) - prev->z;  // >= 0: sorted
    emit_unsigned(sink, kDz, std::uint64_t(dz));
    const std::int64_t dy = std::int64_t(cur.y) - prev->y;
    const std::int64_t dx = std::int64_t(cur.x) - prev->x;
    if (dz == 0) {
      emit_unsigned(sink, kDySame, std::uint64_t(dy));  // >= 0: sorted
      if (dy == 0) {
        emit_unsigned(sink, kDxRun, std::uint64_t(dx - 1));  // dx >= 1
      } else {
        emit_signed(sink, kDxFree, dx);
      }
    } else {
      emit_signed(sink, kDyFree, dy);
      emit_signed(sink, kDxFree, dx);
    }
  }

  bool full = true;
  bool none = true;
  for (std::uint32_t w = 0; w < kMaskWordsPerBlock; ++w) {
    full = full && mask[w] == ~0u;
    none = none && mask[w] == 0u;
  }
  sink.symbol(kMaskClass, full ? kMaskFull : none ? kMaskEmpty : kMaskPartial);
  if (!full && !none) {
    for (std::uint32_t w = 0; w < kMaskWordsPerBlock; ++w) {
      for (std::uint32_t b = 0; b < 4; ++b) {
        sink.symbol(kMaskByte, (mask[w] >> (8 * b)) & 0xFFu);
      }
    }
  }

  for (std::uint32_t j = 0; j < k; ++j) {
    emit_signed(sink, kFirstCoef + j, coeffs[j]);
  }
}

struct CountSink {
  std::vector<std::vector<std::uint64_t>>& counts;
  void symbol(std::uint32_t model, std::uint32_t s) { ++counts[model][s]; }
  void bits(std::uint32_t, std::uint32_t) {}
};

struct WriteSink {
  RansWriter& writer;
  const std::vector<FrequencyTable>& tables;
  void symbol(std::uint32_t model, std::uint32_t s) {
    writer.put(tables[model], s);
  }
  void bits(std::uint32_t value, std::uint32_t n) { writer.put_bits(value, n); }
};

// --- Byte-level serialization.
// ------------------------------------------------

class ByteWriter {
 public:
  explicit ByteWriter(std::vector<std::uint8_t>& out) : out_(out) {}
  void u8(std::uint32_t v) { out_.push_back(static_cast<std::uint8_t>(v)); }
  void u16(std::uint32_t v) {
    u8(v & 0xFFu);
    u8((v >> 8) & 0xFFu);
  }
  void u32(std::uint32_t v) {
    u16(v & 0xFFFFu);
    u16(v >> 16);
  }
  void f32(float f) {
    std::uint32_t bits = 0;
    std::memcpy(&bits, &f, sizeof(bits));
    u32(bits);
  }
  void varint(std::uint32_t v) {  // LEB128
    while (v >= 0x80u) {
      u8((v & 0x7Fu) | 0x80u);
      v >>= 7;
    }
    u8(v);
  }
  void bytes(const std::vector<std::uint8_t>& b) {
    out_.insert(out_.end(), b.begin(), b.end());
  }

 private:
  std::vector<std::uint8_t>& out_;
};

// Bounds-checked and sticky, like RansReader: a read past the end sets failed
// and returns 0, so a parser checks once per structure rather than per field.
class ByteReader {
 public:
  ByteReader(const std::uint8_t* data, std::size_t size)
      : data_(data), size_(size) {}
  std::uint32_t u8() {
    if (pos_ >= size_) {
      failed_ = true;
      return 0;
    }
    return data_[pos_++];
  }
  std::uint32_t u16() {
    const std::uint32_t lo = u8();
    return lo | (u8() << 8);
  }
  std::uint32_t u32() {
    const std::uint32_t lo = u16();
    return lo | (u16() << 16);
  }
  float f32() {
    const std::uint32_t bits = u32();
    float f = 0.0f;
    std::memcpy(&f, &bits, sizeof(f));
    return f;
  }
  // Canonical LEB128 only, the one encoding ByteWriter::varint makes: a
  // final byte of 0 after the first adds nothing, so {0x80, 0x00} is refused
  // rather than read as a second spelling of 0.
  std::uint32_t varint() {
    std::uint32_t v = 0;
    for (std::uint32_t shift = 0; shift < 35; shift += 7) {
      const std::uint32_t b = u8();
      if (shift == 28 && (b & 0xF0u) != 0) {
        failed_ = true;  // more than 32 bits, or a sixth byte
        return 0;
      }
      v |= (b & 0x7Fu) << shift;
      if ((b & 0x80u) == 0) {
        if (b == 0 && shift != 0) {
          failed_ = true;  // overlong
          return 0;
        }
        return v;
      }
    }
    failed_ = true;
    return 0;
  }
  bool failed() const noexcept { return failed_; }
  bool at_end() const noexcept { return pos_ == size_; }

 private:
  const std::uint8_t* data_;
  std::size_t size_;
  std::size_t pos_ = 0;
  bool failed_ = false;
};

Status bad(const std::string& why) {
  return Status::invalid_argument("codec frame: " + why);
}

Status bad_write(const std::string& why) {
  return Status::invalid_argument("write_intra_frame: " + why);
}

bool positive_finite(float f) { return std::isfinite(f) && f > 0.0f; }

// Tables: per model, a varint count of used symbols, then per used symbol the
// gap from the previous one (the first counts from -1) and its frequency.
void write_tables(ByteWriter& w, const std::vector<FrequencyTable>& tables) {
  for (const FrequencyTable& t : tables) {
    std::uint32_t used = 0;
    for (std::uint16_t f : t.freq) {
      used += f != 0 ? 1u : 0u;
    }
    w.varint(used);
    std::uint32_t next = 0;  // the first symbol the gap counts from
    for (std::uint32_t s = 0; s < t.freq.size(); ++s) {
      if (t.freq[s] != 0) {
        w.varint(s - next);
        w.varint(t.freq[s]);
        next = s + 1;
      }
    }
  }
}

Status read_tables(const std::uint8_t* data, std::size_t size, std::uint32_t k,
                   std::vector<FrequencyTable>& tables) {
  ByteReader r(data, size);
  tables.assign(model_count(k), FrequencyTable{});
  for (std::uint32_t m = 0; m < model_count(k); ++m) {
    const std::uint32_t n = alphabet(m);
    FrequencyTable& t = tables[m];
    t.freq.assign(n, 0);
    const std::uint32_t used = r.varint();
    if (r.failed() || used > n) {
      return bad("table " + std::to_string(m) + " is malformed");
    }
    std::uint64_t next = 0;
    std::uint32_t sum = 0;
    for (std::uint32_t i = 0; i < used; ++i) {
      const std::uint64_t s = next + r.varint();
      const std::uint32_t f = r.varint();
      if (r.failed() || s >= n || f == 0 || f > kRansScale) {
        return bad("table " + std::to_string(m) + " is malformed");
      }
      t.freq[s] = static_cast<std::uint16_t>(f);
      sum += f;
      next = s + 1;
    }
    if (used != 0 && sum != kRansScale) {
      return bad("table " + std::to_string(m) + " does not sum to 4096");
    }
    t.finalize();
    t.build_decode();
  }
  if (!r.at_end()) {
    return bad("the TABLES section has trailing bytes");
  }
  return {};
}

// The reader's mirror of emit_unsigned / emit_signed. Class c > 1 carries c-1
// raw bits below its leading one.
std::uint64_t get_unsigned(RansReader& r, const FrequencyTable& t) {
  const std::uint32_t c = r.get(t);
  if (c <= 1) {
    return c;
  }
  return (std::uint64_t(1) << (c - 1)) + r.get_bits(c - 1);
}

std::int64_t get_signed(RansReader& r, const FrequencyTable& t) {
  const std::uint32_t c = r.get(t);
  if (c == 0) {
    return 0;
  }
  const bool negative = r.get_bits(1) != 0;
  const std::uint64_t m =
      c == 1 ? 1 : (std::uint64_t(1) << (c - 1)) + r.get_bits(c - 1);
  return negative ? -std::int64_t(m) : std::int64_t(m);
}

// A coordinate the deltas produced, or false if it left int32.
bool to_coord(std::int64_t v, std::int32_t& out) {
  if (v < std::numeric_limits<std::int32_t>::min() ||
      v > std::numeric_limits<std::int32_t>::max()) {
    return false;
  }
  out = static_cast<std::int32_t>(v);
  return true;
}

// The mirror of emit_block. Returns false for a coordinate outside int32; the
// reader's own failure flag carries everything else, including a failure
// partway through the deltas, whose zeros would otherwise decode as a run step
// and could be misreported as leaving int32.
bool read_block(RansReader& r, const std::vector<FrequencyTable>& t,
                const Vec3i* prev, Vec3i& cur, std::uint32_t* mask,
                std::int32_t* coeffs, std::uint32_t k) {
  if (prev == nullptr) {
    cur.x = static_cast<std::int32_t>(r.get_bits(32));
    cur.y = static_cast<std::int32_t>(r.get_bits(32));
    cur.z = static_cast<std::int32_t>(r.get_bits(32));
  } else {
    const std::int64_t dz = std::int64_t(get_unsigned(r, t[kDz]));
    std::int64_t dy = 0;
    std::int64_t dx = 0;
    if (dz == 0) {
      dy = std::int64_t(get_unsigned(r, t[kDySame]));
      dx = dy == 0 ? std::int64_t(get_unsigned(r, t[kDxRun])) + 1
                   : get_signed(r, t[kDxFree]);
    } else {
      dy = get_signed(r, t[kDyFree]);
      dx = get_signed(r, t[kDxFree]);
    }
    if (r.failed()) {
      return true;  // the caller's finish() reports the corrupt segment
    }
    if (!to_coord(prev->x + dx, cur.x) || !to_coord(prev->y + dy, cur.y) ||
        !to_coord(prev->z + dz, cur.z)) {
      return false;
    }
  }

  // TODO(codec): refuse a partial mask that decodes all-full or all-empty.
  // The writer never makes one and it decodes correctly, so it is a second
  // spelling of one frame rather than a wrong one (the 2026-09-27 entry).
  const std::uint32_t cls = r.get(t[kMaskClass]);
  for (std::uint32_t w = 0; w < kMaskWordsPerBlock; ++w) {
    if (cls == kMaskPartial) {
      std::uint32_t word = 0;
      for (std::uint32_t b = 0; b < 4; ++b) {
        word |= r.get(t[kMaskByte]) << (8 * b);
      }
      mask[w] = word;
    } else {
      mask[w] = cls == kMaskFull ? ~0u : 0u;
    }
  }

  for (std::uint32_t j = 0; j < k; ++j) {
    // Class < 16, so |value| <= 32767: every coefficient is in range.
    coeffs[j] = static_cast<std::int32_t>(get_signed(r, t[kFirstCoef + j]));
  }
  return true;
}

// Where a known section's body sits in the frame.
struct SectionBody {
  const std::uint8_t* data = nullptr;
  std::uint32_t size = 0;
};

const SectionBody& section(const std::array<SectionBody, kSectionCount>& found,
                           SectionId id) {
  return found[static_cast<std::size_t>(id) - 1];
}

}  // namespace

Result<std::vector<std::uint8_t>> write_intra_frame(
    const IntraFrame& frame, const FrameWriteOptions& options) {
  const DctBlocks& b = frame.blocks;
  if (!positive_finite(frame.voxel_size)) {
    return bad_write("voxel_size must be finite and positive");
  }
  if (!positive_finite(b.trunc_dist)) {
    return bad_write("trunc_dist must be finite and positive");
  }
  VR_TRY(b.params.validate());
  if (options.segment_size == 0) {
    return bad_write("segment_size must be at least 1");
  }
  const std::size_t n = frame.coords.size();
  const std::uint32_t k = b.params.coefficient_count;
  if (n > std::numeric_limits<std::uint32_t>::max()) {
    return bad_write("more than 2^32 - 1 blocks");
  }
  // Sizes in 64 bits: n * K wraps a 32-bit size_t long before 2^32 blocks.
  const std::uint64_t coefficient_count = std::uint64_t(n) * k;
  const std::uint64_t mask_count = std::uint64_t(n) * kMaskWordsPerBlock;
  if (b.coefficients.size() != coefficient_count) {
    return bad_write("expected " + std::to_string(coefficient_count) +
                     " coefficients, got " +
                     std::to_string(b.coefficients.size()));
  }
  if (b.masks.size() != mask_count) {
    return bad_write("expected " + std::to_string(mask_count) +
                     " mask words, got " + std::to_string(b.masks.size()));
  }
  const std::uint32_t r_size = options.segment_size;
  const std::uint64_t segments =
      n == 0 ? 0 : (std::uint64_t(n) - 1) / r_size + 1;
  if (segments > std::numeric_limits<std::uint32_t>::max() / 4) {
    return bad_write(
        "more than 2^30 - 1 segments: SEGMENTS would outgrow its "
        "u32 length");
  }
  for (std::size_t i = 1; i < n; ++i) {
    if (!coord_less(frame.coords[i - 1], frame.coords[i])) {
      return bad_write(
          "coordinates are not strictly increasing in (z, y, x) "
          "at entry " +
          std::to_string(i));
    }
  }
  for (std::int32_t c : b.coefficients) {
    if (c < -kMaxQuantizedMagnitude || c > kMaxQuantizedMagnitude) {
      return bad_write("a coefficient is outside +-" +
                       std::to_string(kMaxQuantizedMagnitude));
    }
  }

  auto prev_of = [&](std::size_t i) -> const Vec3i* {
    return i % r_size == 0 ? nullptr : &frame.coords[i - 1];
  };

  // Pass 1: every model's histogram, over exactly the symbols pass 2 writes.
  std::vector<std::vector<std::uint64_t>> counts(model_count(k));
  for (std::uint32_t m = 0; m < model_count(k); ++m) {
    counts[m].assign(alphabet(m), 0);
  }
  CountSink count_sink{counts};
  for (std::size_t i = 0; i < n; ++i) {
    emit_block(count_sink, prev_of(i), frame.coords[i],
               &b.masks[i * kMaskWordsPerBlock], &b.coefficients[i * k], k);
  }
  std::vector<FrequencyTable> tables;
  tables.reserve(counts.size());
  for (const std::vector<std::uint64_t>& c : counts) {
    tables.push_back(normalize_counts(c));
  }

  // Pass 2: each segment as its own stream, appended straight onto the
  // payload.
  // TODO(codec): code the segments on several host threads, which they allow
  // by construction, if a finer voxel or K = 128 is wanted in real time (the
  // 2026-09-27 defaults decision); here and in the reader's loop.
  std::vector<std::uint8_t> payload;
  std::vector<std::uint8_t> segments_body;
  ByteWriter sw(segments_body);
  RansWriter writer;
  WriteSink write_sink{writer, tables};
  for (std::uint64_t s = 0; s < segments; ++s) {
    const std::size_t first = static_cast<std::size_t>(s * r_size);  // < n
    const std::size_t end = static_cast<std::size_t>(
        std::min<std::uint64_t>(n, std::uint64_t(first) + r_size));
    for (std::size_t i = first; i < end; ++i) {
      emit_block(write_sink, prev_of(i), frame.coords[i],
                 &b.masks[i * kMaskWordsPerBlock], &b.coefficients[i * k], k);
    }
    const std::size_t stream_at = payload.size();
    if (!writer.finish(payload)) {
      // Pass 1 counted every symbol pass 2 puts, so this is a bug here.
      return Status::io_error(
          "write_intra_frame: a table refused a symbol it was counted from");
    }
    sw.u32(static_cast<std::uint32_t>(payload.size() - stream_at));
  }
  // Every segment's length is at most the payload's, so this bounds both.
  if (std::uint64_t(payload.size()) >
      std::numeric_limits<std::uint32_t>::max()) {
    return bad_write("the payload outgrows its u32 length (4 GiB)");
  }

  std::vector<std::uint8_t> tables_body;  // under 32 KB
  ByteWriter tw(tables_body);
  write_tables(tw, tables);

  std::vector<std::uint8_t> out;
  ByteWriter w(out);
  for (std::uint8_t c : kFrameMagic) {
    w.u8(c);
  }
  w.u16(kFrameVersion);
  w.u8(static_cast<std::uint32_t>(FrameType::kIntra));
  w.u8(0);  // reserved
  w.f32(frame.voxel_size);
  w.f32(b.trunc_dist);
  w.u32(static_cast<std::uint32_t>(kBlockSize));
  w.u32(k);
  w.f32(b.params.dc_step);
  w.f32(b.params.ac_step);
  w.u32(static_cast<std::uint32_t>(n));
  w.u32(r_size);
  const std::array<std::pair<SectionId, const std::vector<std::uint8_t>*>,
                   kSectionCount>
      sections = {{{SectionId::kTables, &tables_body},
                   {SectionId::kSegments, &segments_body},
                   {SectionId::kPayload, &payload}}};
  w.u32(kSectionCount);
  for (const auto& [id, body] : sections) {
    w.u16(static_cast<std::uint32_t>(id));
    w.u16(kSectionRequired);
    w.u32(static_cast<std::uint32_t>(body->size()));  // each bounded above
  }
  for (const auto& [id, body] : sections) {
    w.bytes(*body);
  }
  return out;
}

Result<FrameHeader> read_frame_header(const std::uint8_t* data,
                                      std::size_t size) {
  if (data == nullptr || size < kFrameHeaderBytes) {
    return bad("shorter than a frame header");
  }
  if (std::memcmp(data, kFrameMagic, sizeof(kFrameMagic)) != 0) {
    return bad("not a codec frame (bad magic)");
  }
  ByteReader h(data + sizeof(kFrameMagic), kFrameHeaderBytes - 4);
  const std::uint32_t version = h.u16();
  const std::uint32_t type = h.u8();
  const std::uint32_t reserved = h.u8();
  FrameHeader header;
  header.voxel_size = h.f32();
  header.trunc_dist = h.f32();
  const std::uint32_t block_size = h.u32();
  header.params.coefficient_count = h.u32();
  header.params.dc_step = h.f32();
  header.params.ac_step = h.f32();
  header.block_count = h.u32();
  header.segment_size = h.u32();
  header.section_count = h.u32();

  if (version != kFrameVersion) {
    return Status::unsupported("codec frame: version " +
                               std::to_string(version) + " (this reads " +
                               std::to_string(kFrameVersion) + ")");
  }
  if (type != static_cast<std::uint32_t>(FrameType::kIntra)) {
    return Status::unsupported("codec frame: type " + std::to_string(type) +
                               " (this reads intra)");
  }
  if (reserved != 0) {
    return bad("the reserved header byte is not zero");
  }
  if (block_size != static_cast<std::uint32_t>(kBlockSize)) {
    return Status::unsupported(
        "codec frame: block size " + std::to_string(block_size) +
        " (the codec's is " + std::to_string(kBlockSize) + ")");
  }
  if (!positive_finite(header.voxel_size) ||
      !positive_finite(header.trunc_dist)) {
    return bad("voxel_size and trunc_dist must be finite and positive");
  }
  if (Status s = header.params.validate(); !s.ok()) {
    return bad("invalid codec params: " + s.message());
  }
  if (header.segment_size == 0) {
    return bad("segment size 0");
  }
  return header;
}

Result<IntraFrame> read_intra_frame(const std::uint8_t* data, std::size_t size,
                                    std::uint32_t max_blocks) {
  VR_ASSIGN(const FrameHeader header, read_frame_header(data, size));
  IntraFrame frame;
  frame.voxel_size = header.voxel_size;
  frame.blocks.trunc_dist = header.trunc_dist;
  frame.blocks.params = header.params;
  const std::uint32_t n = header.block_count;
  const std::uint32_t r_size = header.segment_size;
  const std::uint32_t section_count = header.section_count;

  // The section table, then each body's place in the frame. Lengths are summed
  // in 64 bits, and must account for every byte after the table exactly.
  if (section_count > (size - kFrameHeaderBytes) / kSectionEntryBytes) {
    return bad("the section table runs past the end");
  }
  ByteReader st(data + kFrameHeaderBytes, section_count * kSectionEntryBytes);
  const std::uint8_t* body =
      data + kFrameHeaderBytes + section_count * kSectionEntryBytes;
  const std::uint64_t body_bytes =
      size - kFrameHeaderBytes - section_count * kSectionEntryBytes;
  std::uint64_t offset = 0;
  std::array<SectionBody, kSectionCount> found{};  // by id - 1
  for (std::uint32_t i = 0; i < section_count; ++i) {
    const std::uint32_t id = st.u16();
    const std::uint32_t flags = st.u16();
    const std::uint32_t length = st.u32();
    if (offset + length > body_bytes) {
      return bad("a section runs past the end");
    }
    if (id >= 1 && id <= kSectionCount) {
      if ((flags & ~std::uint32_t(kSectionKnownFlags)) != 0) {
        return Status::unsupported(
            "codec frame: section " + std::to_string(id) + " sets flags " +
            std::to_string(flags) + ", which v1 does not define");
      }
      SectionBody& f = found[id - 1];
      if (f.data != nullptr) {
        return bad("section " + std::to_string(id) + " appears twice");
      }
      f = SectionBody{body + offset, length};
    } else if ((flags & kSectionRequired) != 0) {
      return Status::unsupported("codec frame: unknown required section " +
                                 std::to_string(id));
    }
    // TODO(codec): the optional CRC section of the 2026-09-27 entry, should a
    // consumer ever keep frames where nothing else checks them. A v1 reader
    // skips it here, so adding it needs no new version.
    offset += length;
  }
  if (offset != body_bytes) {
    return bad("bytes after the last section");
  }
  for (std::uint32_t i = 0; i < kSectionCount; ++i) {
    if (found[i].data == nullptr) {
      return bad("section " + std::to_string(i + 1) + " is missing");
    }
  }
  const SectionBody& tables_section = section(found, SectionId::kTables);
  const SectionBody& segments_section = section(found, SectionId::kSegments);
  const SectionBody& payload_section = section(found, SectionId::kPayload);

  const std::uint32_t k = frame.blocks.params.coefficient_count;
  std::vector<FrequencyTable> tables;
  VR_TRY(read_tables(tables_section.data, tables_section.size, k, tables));

  const std::uint64_t segments =
      n == 0 ? 0 : (std::uint64_t(n) - 1) / r_size + 1;
  if (segments_section.size != segments * 4) {
    return bad("the SEGMENTS section does not list one length per segment");
  }
  ByteReader lens(segments_section.data, segments_section.size);
  std::vector<std::uint32_t> lengths(segments);
  std::uint64_t total = 0;
  for (std::uint64_t s = 0; s < segments; ++s) {
    lengths[s] = lens.u32();
    if (lengths[s] < 4 || lengths[s] % 2 != 0) {
      return bad("segment " + std::to_string(s) + " has an impossible length");
    }
    total += lengths[s];
  }
  if (total != payload_section.size) {
    return bad("the segment lengths do not add up to the PAYLOAD section");
  }
  // Only now, with the block count consistent with the segment table: a count
  // that is not is a corrupt frame, and one that is but exceeds the caller's
  // room is a sound frame too big to hold -- which the caller answers by
  // growing, not by dropping the frame.
  if (n > max_blocks) {
    return Status::out_of_memory("codec frame: it holds " + std::to_string(n) +
                                 " blocks, more than the " +
                                 std::to_string(max_blocks) +
                                 " the caller can hold");
  }

  // max_blocks bounds these in bytes, but not below what a 32-bit size_t can
  // count: n * K reaches 2^41. Past this check every index below fits.
  if (n > frame.coords.max_size() ||
      std::uint64_t(n) * k > frame.blocks.coefficients.max_size() ||
      std::uint64_t(n) * kMaskWordsPerBlock > frame.blocks.masks.max_size()) {
    return Status::out_of_memory(
        "codec frame: " + std::to_string(n) + " blocks of " +
        std::to_string(k) +
        " coefficients exceed this platform's address space");
  }
  frame.coords.resize(n);
  frame.blocks.coefficients.resize(std::size_t(n) * k);
  frame.blocks.masks.resize(std::size_t(n) * kMaskWordsPerBlock);
  const std::uint8_t* stream = payload_section.data;
  for (std::uint64_t s = 0; s < segments; ++s) {
    RansReader r(stream, lengths[s]);
    const std::size_t first = static_cast<std::size_t>(s * r_size);  // < n
    const std::size_t end = static_cast<std::size_t>(
        std::min<std::uint64_t>(n, std::uint64_t(first) + r_size));
    for (std::size_t i = first; i < end && !r.failed(); ++i) {
      const Vec3i* prev = i == first ? nullptr : &frame.coords[i - 1];
      if (!read_block(r, tables, prev, frame.coords[i],
                      &frame.blocks.masks[i * kMaskWordsPerBlock],
                      &frame.blocks.coefficients[i * k], k)) {
        return bad("segment " + std::to_string(s) +
                   " steps a coordinate outside int32");
      }
    }
    if (!r.finish()) {
      return bad("segment " + std::to_string(s) + " is corrupt");
    }
    // Within a segment the deltas cannot step backwards; its first coordinate
    // is raw bits, which no end check sees, so the order across segments is
    // checked here. Without it a flipped bit decodes a duplicate, and the
    // inverse transform races two workgroups on one block.
    if (s > 0 && !coord_less(frame.coords[first - 1], frame.coords[first])) {
      return bad("segment " + std::to_string(s) +
                 " does not start after the one before ends");
    }
    stream += lengths[s];
  }
  return frame;
}

}  // namespace volumetric_kit::recon::codec::detail
