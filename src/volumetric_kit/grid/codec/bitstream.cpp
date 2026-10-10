// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include "bitstream.hpp"

#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <string>
#include <utility>

#include "dct_tables.hpp"
#include "rans.hpp"

namespace volumetric_kit::recon::codec::detail {
namespace {

// --- The models and the block grammar. -------------------------------------
//
// shaders/frame_models.glsl and shaders/frame_grammar.glsl, the kernels' own
// GLSL, compiled here as C++ once these four names are defined.

using uint = std::uint32_t;
using glm::ivec3;
using glm::uvec2;

// GLSL's findMSB: the index of the highest set bit, -1 for 0.
int findMSB(uint v) { return v == 0 ? -1 : 31 - __builtin_clz(v); }

#include "shaders/frame_models.glsl"

static_assert(kMaskWords == kMaskWordsPerBlock &&
                  kLinesPerPlane == std::uint32_t(kBlockSize) &&
                  kPlanes == std::uint32_t(kBlockSize),
              "the frame's block is the codec's");

// The grammar's writer half over a sink: the histogram pass and the encode
// pass walk exactly the same symbols, as the device's count and coding passes
// do, so a table built from one walk and a stream written by another could
// only disagree by construction, never by accident.
template <typename Sink>
struct BlockWriter {
  Sink& sink;
  const uint* g_mask = nullptr;  // the block's
  const std::int16_t* coefficients = nullptr;

  void sink_symbol(uint model, uint symbol) { sink.symbol(model, symbol); }
  void sink_bits(uint value, uint bits) { sink.bits(value, bits); }
  int coefficient(uint j) const { return coefficients[j]; }

#define VR_FRAME_WRITER
#include "shaders/frame_grammar.glsl"
#undef VR_FRAME_WRITER
};

struct CountSink {
  std::vector<std::vector<std::uint64_t>>& counts;
  void symbol(uint model, uint s) { ++counts[model][s]; }
  void bits(uint, uint) {}
};

struct WriteSink {
  RansWriter& writer;
  const std::vector<FrequencyTable>& tables;
  void symbol(uint model, uint s) { writer.put(tables[model], s); }
  void bits(uint value, uint n) { writer.put_bits(value, n); }
};

// The grammar's reader half over one segment's stream: the device's decode
// reads it with the same lines.
struct BlockReader {
  RansReader& coder;
  const std::vector<FrequencyTable>& tables;
  uint* g_mask = nullptr;  // the block's
  std::int16_t* coefficients = nullptr;
  ivec3 g_coord{0};  // the block before's, then this one's

  uint get(uint model) { return coder.get(tables[model]); }
  uint get_bits(uint bits) { return coder.get_bits(bits); }
  bool coder_failed() const { return coder.failed(); }
  void store_coefficient(uint j, int value) {
    coefficients[j] = static_cast<std::int16_t>(value);
  }

#define VR_FRAME_READER
#include "shaders/frame_grammar.glsl"
#undef VR_FRAME_READER
};

}  // namespace

std::uint32_t frame_model_count(std::uint32_t k) { return kFirstCoef + k; }

std::uint32_t frame_model_alphabet(std::uint32_t model) {
  return model_alphabet(model);
}

std::uint32_t frame_model_base(std::uint32_t model) {
  return model_base(model);
}

namespace {

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

core::Status bad(const std::string& why) {
  return core::Status::invalid_argument("codec frame: " + why);
}

core::Status bad_write(const std::string& why) {
  return core::Status::invalid_argument("write_intra_frame: " + why);
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

core::Status read_tables(const std::uint8_t* data, std::size_t size,
                         std::uint32_t k, std::vector<FrequencyTable>& tables) {
  ByteReader r(data, size);
  tables.assign(frame_model_count(k), FrequencyTable{});
  for (std::uint32_t m = 0; m < frame_model_count(k); ++m) {
    const std::uint32_t n = model_alphabet(m);
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
  }
  if (!r.at_end()) {
    return bad("the TABLES section has trailing bytes");
  }
  return {};
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

// The checks write_intra_frame makes before coding anything.
core::Status check_intra_frame(const IntraFrame& frame,
                               const FrameWriteOptions& options) {
  const DctBlocks& b = frame.blocks;
  if (!positive_finite(frame.voxel_size)) {
    return bad_write("voxel_size must be finite and positive");
  }
  if (!positive_finite(b.trunc_dist)) {
    return bad_write("trunc_dist must be finite and positive");
  }
  VKC_TRY(b.params.validate());
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
  if (frame_segment_count(n, options.segment_size) > kMaxFrameSegments) {
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
  for (std::int16_t c : b.coefficients) {
    if (c < -kMaxQuantizedMagnitude || c > kMaxQuantizedMagnitude) {
      return bad_write("a coefficient is outside +-" +
                       std::to_string(kMaxQuantizedMagnitude));
    }
  }
  return {};
}

}  // namespace

core::Result<std::vector<std::uint8_t>> write_intra_frame(
    const IntraFrame& frame, const FrameWriteOptions& options) {
  VKC_TRY(check_intra_frame(frame, options));
  const DctBlocks& b = frame.blocks;
  const std::size_t n = frame.coords.size();
  const std::uint32_t k = b.params.coefficient_count;
  const std::uint32_t r_size = options.segment_size;
  const std::uint64_t segments = frame_segment_count(n, r_size);

  // Block i through a writer.
  auto emit = [&](auto& writer, std::size_t i) {
    const bool first = i % r_size == 0;
    writer.g_mask = &b.masks[i * kMaskWordsPerBlock];
    writer.coefficients = &b.coefficients[i * k];
    writer.emit_block(first, frame.coords[first ? i : i - 1], frame.coords[i],
                      k);
  };

  // Pass 1: every model's histogram, over exactly the symbols pass 2 writes.
  std::vector<std::vector<std::uint64_t>> counts(frame_model_count(k));
  for (std::uint32_t m = 0; m < frame_model_count(k); ++m) {
    counts[m].assign(model_alphabet(m), 0);
  }
  CountSink count_sink{counts};
  BlockWriter<CountSink> counter{count_sink};
  for (std::size_t i = 0; i < n; ++i) {
    emit(counter, i);
  }
  std::vector<FrequencyTable> tables = frame_tables(counts);

  // Pass 2: each segment as its own stream, appended straight onto the
  // payload.
  // TODO(codec): code the segments on several host threads, which they allow
  // by construction, if a finer voxel or K = 128 is wanted in real time (the
  // 2026-09-27 defaults decision); here and in the reader's loop.
  std::vector<std::uint8_t> payload;
  std::vector<std::uint32_t> lengths;
  lengths.reserve(static_cast<std::size_t>(segments));
  RansWriter writer;
  WriteSink write_sink{writer, tables};
  BlockWriter<WriteSink> coder{write_sink};
  for (std::uint64_t s = 0; s < segments; ++s) {
    const std::size_t first = static_cast<std::size_t>(s * r_size);  // < n
    const std::size_t end = static_cast<std::size_t>(
        std::min<std::uint64_t>(n, std::uint64_t(first) + r_size));
    for (std::size_t i = first; i < end; ++i) {
      emit(coder, i);
    }
    const std::size_t stream_at = payload.size();
    if (!writer.finish(payload)) {
      // Pass 1 counted every symbol pass 2 puts, so this is a bug here.
      return core::Status::io_error(
          "write_intra_frame: a table refused a symbol it was counted from");
    }
    lengths.push_back(static_cast<std::uint32_t>(payload.size() - stream_at));
  }
  CodedFrame coded;
  coded.voxel_size = frame.voxel_size;
  coded.trunc_dist = b.trunc_dist;
  coded.params = b.params;
  coded.block_count = static_cast<std::uint32_t>(n);
  coded.segment_size = r_size;
  coded.tables = std::move(tables);
  coded.segment_lengths = std::move(lengths);
  coded.payload = std::move(payload);
  return assemble_intra_frame(coded);
}

std::vector<FrequencyTable> frame_tables(
    const std::vector<std::vector<std::uint64_t>>& counts) {
  std::vector<FrequencyTable> tables;
  tables.reserve(counts.size());
  for (const std::vector<std::uint64_t>& c : counts) {
    tables.push_back(normalize_counts(c));
  }
  return tables;
}

core::Result<std::vector<std::uint8_t>> assemble_intra_frame(
    const CodedFrame& frame) {
  // The header's fields as check_intra_frame checks them, for the device
  // writer, which codes without an IntraFrame.
  if (!positive_finite(frame.voxel_size)) {
    return bad_write("voxel_size must be finite and positive");
  }
  if (!positive_finite(frame.trunc_dist)) {
    return bad_write("trunc_dist must be finite and positive");
  }
  VKC_TRY(frame.params.validate());
  if (frame.segment_size == 0) {
    return bad_write("segment_size must be at least 1");
  }
  const std::uint32_t k = frame.params.coefficient_count;
  const std::uint64_t segments =
      frame_segment_count(frame.block_count, frame.segment_size);
  if (segments > kMaxFrameSegments) {
    return bad_write(
        "more than 2^30 - 1 segments: SEGMENTS would outgrow its "
        "u32 length");
  }
  if (frame.tables.size() != frame_model_count(k) ||
      frame.segment_lengths.size() != segments) {
    return bad_write("tables or segment lengths disagree with the header");
  }
  // Every segment's length is at most the payload's, so this bounds both.
  if (std::uint64_t(frame.payload.size()) >
      std::numeric_limits<std::uint32_t>::max()) {
    return bad_write("the payload outgrows its u32 length (4 GiB)");
  }
  std::uint64_t total = 0;
  for (std::uint32_t length : frame.segment_lengths) total += length;
  if (total != frame.payload.size()) {
    return bad_write("the segment lengths do not sum to the payload");
  }
  std::vector<std::uint8_t> segments_body;
  ByteWriter sw(segments_body);
  for (std::uint32_t length : frame.segment_lengths) {
    sw.u32(length);
  }

  std::vector<std::uint8_t> tables_body;  // under 32 KB
  ByteWriter tw(tables_body);
  write_tables(tw, frame.tables);

  std::vector<std::uint8_t> out;
  ByteWriter w(out);
  for (std::uint8_t c : kFrameMagic) {
    w.u8(c);
  }
  w.u16(kFrameVersion);
  w.u8(static_cast<std::uint32_t>(FrameType::kIntra));
  w.u8(0);  // reserved
  w.f32(frame.voxel_size);
  w.f32(frame.trunc_dist);
  w.u32(static_cast<std::uint32_t>(kBlockSize));
  w.u32(k);
  w.f32(frame.params.quantization_scale);
  w.u32(k);  // one weight per kept basis
  w.u32(frame.block_count);
  w.u32(frame.segment_size);
  const std::array<std::pair<SectionId, const std::vector<std::uint8_t>*>,
                   kSectionCount>
      sections = {{{SectionId::kTables, &tables_body},
                   {SectionId::kSegments, &segments_body},
                   {SectionId::kPayload, &frame.payload}}};
  w.u32(kSectionCount);
  static const auto zigzag = zigzag_order();
  for (std::uint32_t j = 0; j < k; ++j) {
    w.f32(frame.params.quantization_weights[zigzag[j]]);
  }
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

core::Result<FrameHeader> read_frame_header(const std::uint8_t* data,
                                            std::size_t size) {
  if (data == nullptr || size < 8) {
    return bad("shorter than a frame header");
  }
  if (std::memcmp(data, kFrameMagic, sizeof(kFrameMagic)) != 0) {
    return bad("not a codec frame (bad magic)");
  }
  ByteReader h(data + sizeof(kFrameMagic), size - sizeof(kFrameMagic));
  const std::uint32_t version = h.u16();
  const std::uint32_t type = h.u8();
  const std::uint32_t reserved = h.u8();
  if (version != kFrameVersion) {
    return core::Status::unsupported("codec frame: version " +
                                     std::to_string(version) + " (this reads " +
                                     std::to_string(kFrameVersion) + ")");
  }
  if (size < kFramePrefixBytes) {
    return bad("shorter than a frame header");
  }
  FrameHeader header;
  header.voxel_size = h.f32();
  header.trunc_dist = h.f32();
  const std::uint32_t block_size = h.u32();
  header.params.coefficient_count = h.u32();
  header.params.quantization_scale = h.f32();
  const std::uint32_t weight_count = h.u32();
  header.block_count = h.u32();
  header.segment_size = h.u32();
  header.section_count = h.u32();
  // K bounds the weights to read; validate() below checks the rest.
  const std::uint32_t k = header.params.coefficient_count;
  if (k < 1 || k > kVoxelsPerBlock) {
    return bad("coefficient count must be in [1, " +
               std::to_string(kVoxelsPerBlock) + "]");
  }
  if (weight_count != k) {
    return bad("quantization weight count must equal the coefficient count");
  }
  if (size < frame_header_bytes(k)) {
    return bad(
        "shorter than a frame header (including the quantization weights)");
  }
  static const auto zigzag = zigzag_order();
  for (std::uint32_t j = 0; j < k; ++j) {
    header.params.quantization_weights[zigzag[j]] = h.f32();
  }

  if (type != static_cast<std::uint32_t>(FrameType::kIntra)) {
    return core::Status::unsupported(
        "codec frame: type " + std::to_string(type) + " (this reads intra)");
  }
  if (reserved != 0) {
    return bad("the reserved header byte is not zero");
  }
  if (block_size != static_cast<std::uint32_t>(kBlockSize)) {
    return core::Status::unsupported(
        "codec frame: block size " + std::to_string(block_size) +
        " (the codec's is " + std::to_string(kBlockSize) + ")");
  }
  if (!positive_finite(header.voxel_size) ||
      !positive_finite(header.trunc_dist)) {
    return bad("voxel_size and trunc_dist must be finite and positive");
  }
  if (core::Status s = header.params.validate(); !s.ok()) {
    return bad("invalid codec params: " + s.message());
  }
  if (header.segment_size == 0) {
    return bad("segment size 0");
  }
  return header;
}

core::Result<ParsedFrame> parse_intra_frame(const std::uint8_t* data,
                                            std::size_t size,
                                            std::uint32_t max_blocks) {
  VKC_ASSIGN(const FrameHeader header, read_frame_header(data, size));
  ParsedFrame frame;
  frame.header = header;
  const std::uint32_t n = header.block_count;
  const std::uint32_t r_size = header.segment_size;
  const std::uint32_t section_count = header.section_count;
  const std::size_t header_bytes =
      frame_header_bytes(header.params.coefficient_count);

  // The section table, then each body's place in the frame. Lengths are summed
  // in 64 bits, and must account for every byte after the table exactly.
  if (section_count > (size - header_bytes) / kSectionEntryBytes) {
    return bad("the section table runs past the end");
  }
  ByteReader st(data + header_bytes, section_count * kSectionEntryBytes);
  const std::uint8_t* body =
      data + header_bytes + section_count * kSectionEntryBytes;
  const std::uint64_t body_bytes =
      size - header_bytes - section_count * kSectionEntryBytes;
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
        return core::Status::unsupported(
            "codec frame: section " + std::to_string(id) + " sets flags " +
            std::to_string(flags) + ", which v3 does not define");
      }
      SectionBody& f = found[id - 1];
      if (f.data != nullptr) {
        return bad("section " + std::to_string(id) + " appears twice");
      }
      f = SectionBody{body + offset, length};
    } else if ((flags & kSectionRequired) != 0) {
      return core::Status::unsupported(
          "codec frame: unknown required section " + std::to_string(id));
    }
    // TODO(codec): the optional CRC section of the 2026-09-27 entry, should a
    // consumer ever keep frames where nothing else checks them. A v3 reader
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

  const std::uint32_t k = header.params.coefficient_count;
  VKC_TRY(
      read_tables(tables_section.data, tables_section.size, k, frame.tables));

  const std::uint64_t segments = frame_segment_count(n, r_size);
  if (segments_section.size != segments * 4) {
    return bad("the SEGMENTS section does not list one length per segment");
  }
  ByteReader lens(segments_section.data, segments_section.size);
  std::vector<std::uint32_t>& lengths = frame.segment_lengths;
  lengths.resize(segments);
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
    return core::Status::out_of_memory(
        "codec frame: it holds " + std::to_string(n) +
        " blocks, more than the " + std::to_string(max_blocks) +
        " the caller can hold");
  }
  frame.payload = payload_section.data;
  frame.payload_size = payload_section.size;
  return frame;
}

core::Status check_segment(std::uint64_t s, SegmentFault fault,
                           const Vec3i* prev_last, const Vec3i& first) {
  if (fault == SegmentFault::kCoordOverflow) {
    return bad("segment " + std::to_string(s) +
               " steps a coordinate outside int32");
  }
  if (fault != SegmentFault::kNone) {
    return bad("segment " + std::to_string(s) + " is corrupt");
  }
  // Within a segment the deltas cannot step backwards; its first coordinate
  // is raw bits, which no end check sees, so the order across segments is
  // checked here. Without it a flipped bit decodes a duplicate, and the
  // inverse transform races two workgroups on one block.
  if (prev_last != nullptr && !coord_less(*prev_last, first)) {
    return bad("segment " + std::to_string(s) +
               " does not start after the one before ends");
  }
  return {};
}

core::Result<IntraFrame> read_intra_frame(const std::uint8_t* data,
                                          std::size_t size,
                                          std::uint32_t max_blocks) {
  VKC_ASSIGN(ParsedFrame parsed, parse_intra_frame(data, size, max_blocks));
  return decode_intra_frame(std::move(parsed));
}

core::Result<IntraFrame> decode_intra_frame(ParsedFrame parsed) {
  for (FrequencyTable& t : parsed.tables) t.build_decode();
  const FrameHeader& header = parsed.header;
  const std::uint32_t n = header.block_count;
  const std::uint32_t r_size = header.segment_size;
  const std::uint32_t k = header.params.coefficient_count;
  IntraFrame frame;
  frame.voxel_size = header.voxel_size;
  frame.blocks.trunc_dist = header.trunc_dist;
  frame.blocks.params = header.params;
  // max_blocks bounds these in bytes, but not below what a 32-bit size_t can
  // count: n * K reaches 2^41. Past this check every index below fits.
  if (n > frame.coords.max_size() ||
      std::uint64_t(n) * k > frame.blocks.coefficients.max_size() ||
      std::uint64_t(n) * kMaskWordsPerBlock > frame.blocks.masks.max_size()) {
    return core::Status::out_of_memory(
        "codec frame: " + std::to_string(n) + " blocks of " +
        std::to_string(k) +
        " coefficients exceed this platform's address space");
  }
  frame.coords.resize(n);
  frame.blocks.coefficients.resize(std::size_t(n) * k);
  frame.blocks.masks.resize(std::size_t(n) * kMaskWordsPerBlock);
  const std::uint8_t* stream = parsed.payload;
  for (std::size_t s = 0; s < parsed.segment_lengths.size(); ++s) {
    RansReader r(stream, parsed.segment_lengths[s]);
    BlockReader reader{r, parsed.tables};
    const std::size_t first = s * r_size;  // < n
    const std::size_t end = static_cast<std::size_t>(
        std::min<std::uint64_t>(n, std::uint64_t(first) + r_size));
    SegmentFault fault = SegmentFault::kNone;
    for (std::size_t i = first; i < end && !r.failed(); ++i) {
      reader.g_mask = &frame.blocks.masks[i * kMaskWordsPerBlock];
      reader.coefficients = &frame.blocks.coefficients[i * k];
      if (!reader.read_block(i == first, k)) {
        fault = SegmentFault::kCoordOverflow;
        break;
      }
      frame.coords[i] = reader.g_coord;
    }
    if (fault == SegmentFault::kNone && !r.finish()) {
      fault = SegmentFault::kCorrupt;
    }
    VKC_TRY(check_segment(s, fault, s > 0 ? &frame.coords[first - 1] : nullptr,
                          frame.coords[first]));
    stream += parsed.segment_lengths[s];
  }
  return frame;
}

}  // namespace volumetric_kit::recon::codec::detail
