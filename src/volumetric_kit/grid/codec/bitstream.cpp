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
static_assert(std::uint32_t(FrameType::kIntra) == kFrameIntra &&
                  std::uint32_t(FrameType::kInter) == kFrameInter,
              "the frame types are the models'");
static_assert(std::uint32_t(BlockMode::kKeep) == kModeKeep &&
                  std::uint32_t(BlockMode::kSkip) == kModeSkip &&
                  std::uint32_t(BlockMode::kZero) == kModeZero &&
                  std::uint32_t(BlockMode::kRemove) == kModeRemove &&
                  std::uint32_t(BlockMode::kMotion) == kModeMotion &&
                  kModeSymbols == kModeMotion + 1,
              "BlockMode is the mode model's alphabet");

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

std::uint32_t frame_model_count(FrameType type, std::uint32_t k) {
  return model_count(static_cast<uint>(type), k);
}

std::uint32_t frame_model_alphabet(std::uint32_t model, std::uint32_t k) {
  return model_alphabet(model, k);
}

std::uint32_t frame_model_base(std::uint32_t model, std::uint32_t k) {
  return model_base(model, k);
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
  return core::Status::invalid_argument("write_frame: " + why);
}

bool positive_finite(float f) { return std::isfinite(f) && f > 0.0f; }

bool same_bits(float a, float b) { return std::memcmp(&a, &b, sizeof(a)) == 0; }

// The first of a P-frame's coding parameters that is not its reference's, bit
// for bit, or null: the one check the reader and both writers make.
const char* coding_difference(const FrameHeader& reference, float voxel_size,
                              float trunc_dist, const CodecParams& params) {
  const CodecParams& held = reference.params;
  if (!same_bits(voxel_size, reference.voxel_size)) return "voxel_size";
  if (!same_bits(trunc_dist, reference.trunc_dist)) return "trunc_dist";
  if (params.coefficient_count != held.coefficient_count) {
    return "coefficient count";
  }
  if (!same_bits(params.quantization_scale, held.quantization_scale)) {
    return "quantization scale";
  }
  static const auto zigzag = zigzag_order();
  for (std::uint32_t j = 0; j < params.coefficient_count; ++j) {
    if (!same_bits(params.quantization_weights[zigzag[j]],
                   held.quantization_weights[zigzag[j]])) {
      return "kept quantization weights";
    }
  }
  return nullptr;
}

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
                         FrameType type, std::uint32_t k,
                         std::vector<FrequencyTable>& tables) {
  ByteReader r(data, size);
  const std::uint32_t models = frame_model_count(type, k);
  tables.assign(models, FrequencyTable{});
  for (std::uint32_t m = 0; m < models; ++m) {
    const std::uint32_t n = model_alphabet(m, k);
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
  // Right after the mode table, the last: a later motion design may append
  // models there only when motion has a frequency (frame_models.glsl).
  if (type == FrameType::kInter &&
      tables[mode_model(k)].freq[kModeMotion] != 0) {
    return core::Status::unsupported(
        "codec frame: a P-frame that uses the motion mode, which this reader "
        "does not decode");
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

// The checks write_frame makes before coding anything.
core::Status check_frame(const Frame& frame, const FrameWriteOptions& options) {
  VKC_TRY(check_frame_type(frame.type, FrameType::kInter, "write_frame"));
  const bool inter = frame.type == FrameType::kInter;
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
  if (inter && options.update_size == 0) {
    return bad_write("update_size must be at least 1");
  }
  // M reference entries, then A blocks: an intra frame's M is 0. Stream
  // positions are u32, on the device too.
  const std::uint64_t m = inter ? frame.reference.header.block_count : 0;
  const std::uint64_t a = frame.coords.size();
  if (m + a > std::numeric_limits<std::uint32_t>::max()) {
    return bad_write("more than 2^32 - 1 entries");
  }
  if (frame.modes.size() != m) {
    return bad_write("expected " + std::to_string(m) +
                     " modes, one per reference block, got " +
                     std::to_string(frame.modes.size()));
  }
  for (BlockMode mode : frame.modes) {
    if (static_cast<uint>(mode) >= kModeSymbols) {
      return bad_write("mode " + std::to_string(static_cast<uint>(mode)) +
                       " is not a mode");
    }
  }
  for (BlockMode mode : frame.modes) {
    if (mode == BlockMode::kMotion) {
      return core::Status::unsupported(
          "write_frame: the motion mode, which this writer does not code");
    }
  }
  const std::uint32_t k = b.params.coefficient_count;
  // Sizes in 64 bits: (M + A) * K wraps a 32-bit size_t long before 2^32
  // entries.
  const std::uint64_t coefficient_count = (m + a) * k;
  const std::uint64_t mask_count = (m + a) * kMaskWordsPerBlock;
  if (b.coefficients.size() != coefficient_count) {
    return bad_write("expected " + std::to_string(coefficient_count) +
                     " coefficients, got " +
                     std::to_string(b.coefficients.size()));
  }
  if (b.masks.size() != mask_count) {
    return bad_write("expected " + std::to_string(mask_count) +
                     " mask words, got " + std::to_string(b.masks.size()));
  }
  if (frame_segment_count(m, options.update_size) +
          frame_segment_count(a, options.segment_size) >
      kMaxFrameSegments) {
    return bad_write(
        "more than 2^30 - 1 segments: SEGMENTS would outgrow its "
        "u32 length");
  }
  for (std::size_t i = 1; i < a; ++i) {
    if (!coord_less(frame.coords[i - 1], frame.coords[i])) {
      return bad_write(
          "coordinates are not strictly increasing in (z, y, x) "
          "at entry " +
          std::to_string(i));
    }
  }
  // Only what is written: a skip or remove entry's payload is ignored.
  for (std::uint64_t p = 0; p < m + a; ++p) {
    if (p < m && frame.modes[p] != BlockMode::kKeep &&
        frame.modes[p] != BlockMode::kZero) {
      continue;
    }
    for (std::uint32_t j = 0; j < k; ++j) {
      const std::int16_t c = b.coefficients[std::size_t(p * k + j)];
      if (c < -kMaxQuantizedMagnitude || c > kMaxQuantizedMagnitude) {
        return bad_write("a coefficient or residual is outside +-" +
                         std::to_string(kMaxQuantizedMagnitude));
      }
    }
  }
  return {};
}

}  // namespace

core::Status check_frame_type(FrameType type, FrameType newest,
                              const char* who) {
  if (static_cast<uint>(type) <= static_cast<uint>(newest)) return {};
  return core::Status::unsupported(
      std::string(who) + ": frame type " +
      std::to_string(static_cast<uint>(type)) +
      (newest == FrameType::kIntra ? " (only intra is coded)"
                                   : " (only intra and P-frames are coded)"));
}

core::Result<std::vector<std::uint8_t>> write_frame(
    const Frame& frame, const FrameWriteOptions& options) {
  VKC_TRY(check_frame(frame, options));
  const bool inter = frame.type == FrameType::kInter;
  const DctBlocks& b = frame.blocks;
  const std::uint32_t k = b.params.coefficient_count;
  // Each below 2^32 together (check_frame).
  const auto m =
      static_cast<uint>(inter ? frame.reference.header.block_count : 0);
  const auto a = static_cast<uint>(frame.coords.size());
  const std::uint32_t u_size = inter ? options.update_size : 0;
  const std::uint32_t r_size = options.segment_size;
  const std::uint64_t segments =
      frame_segment_count(m, u_size) + frame_segment_count(a, r_size);

  // Segment s through a writer: an update segment's reference entries, or a
  // block segment's blocks.
  auto emit_segment = [&](auto& writer, std::uint64_t s) {
    const uvec2 range =
        segment_range(static_cast<uint>(s), m, u_size, a, r_size);
    for (uint p = range.x; p < range.y; ++p) {
      writer.g_mask = &b.masks[std::size_t(p) * kMaskWordsPerBlock];
      writer.coefficients = &b.coefficients[std::size_t(p) * k];
      if (p < m) {
        writer.emit_entry(static_cast<uint>(frame.modes[p]), k);
      } else {
        const bool first = p == range.x;
        const uint q = p - m;
        writer.emit_block(first, frame.coords[first ? q : q - 1],
                          frame.coords[q], k);
      }
    }
  };

  // Pass 1: every model's histogram, over exactly the symbols pass 2 writes.
  const std::uint32_t models = frame_model_count(frame.type, k);
  std::vector<std::vector<std::uint64_t>> counts(models);
  for (std::uint32_t model = 0; model < models; ++model) {
    counts[model].assign(model_alphabet(model, k), 0);
  }
  CountSink count_sink{counts};
  BlockWriter<CountSink> counter{count_sink};
  for (std::uint64_t s = 0; s < segments; ++s) {
    emit_segment(counter, s);
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
    emit_segment(coder, s);
    const std::size_t stream_at = payload.size();
    if (!writer.finish(payload)) {
      // Pass 1 counted every symbol pass 2 puts, so this is a bug here.
      return core::Status::io_error(
          "write_frame: a table refused a symbol it was counted from");
    }
    lengths.push_back(static_cast<std::uint32_t>(payload.size() - stream_at));
  }
  std::uint32_t survivors = 0;
  for (BlockMode mode : frame.modes) {
    survivors += mode != BlockMode::kRemove ? 1u : 0u;
  }
  CodedFrame coded;
  coded.voxel_size = frame.voxel_size;
  coded.trunc_dist = b.trunc_dist;
  coded.params = b.params;
  coded.block_count = survivors + a;
  coded.segment_size = r_size;
  coded.reference = frame.reference;
  coded.update_size = u_size;
  coded.new_count = inter ? a : 0;
  coded.tables = std::move(tables);
  coded.segment_lengths = std::move(lengths);
  coded.payload = std::move(payload);
  return assemble_frame(frame.type, coded);
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

core::Result<std::vector<std::uint8_t>> assemble_frame(
    FrameType type, const CodedFrame& frame) {
  VKC_TRY(check_frame_type(type, FrameType::kInter, "assemble_frame"));
  const bool inter = type == FrameType::kInter;
  // The header's fields as check_frame checks them, for the device
  // writer, which codes without a Frame.
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
  const std::uint64_t m = frame.reference.header.block_count;
  if (!inter && (frame.reference.hash != 0 || m != 0 ||
                 frame.update_size != 0 || frame.new_count != 0)) {
    return bad_write("an intra frame has no reference");
  }
  if (inter && frame.update_size == 0) {
    return bad_write("update_size must be at least 1");
  }
  // An intra frame's blocks are all new, against no reference.
  const std::uint64_t a = inter ? frame.new_count : frame.block_count;
  if (m + a > std::numeric_limits<std::uint32_t>::max()) {
    return bad_write("more than 2^32 - 1 entries");
  }
  if (frame.block_count < a || frame.block_count > m + a) {
    return bad_write(
        "the block count is not the survivors plus the new "
        "blocks");
  }
  if (inter) {
    if (const char* d =
            coding_difference(frame.reference.header, frame.voxel_size,
                              frame.trunc_dist, frame.params)) {
      return bad_write(std::string("the P-frame's ") + d +
                       " is not its reference's");
    }
  }
  const std::uint32_t k = frame.params.coefficient_count;
  const std::uint64_t segments = frame_segment_count(m, frame.update_size) +
                                 frame_segment_count(a, frame.segment_size);
  if (segments > kMaxFrameSegments) {
    return bad_write(
        "more than 2^30 - 1 segments: SEGMENTS would outgrow its "
        "u32 length");
  }
  if (frame.tables.size() != frame_model_count(type, k) ||
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

  // REFERENCE, a P-frame's, first.
  std::vector<std::pair<SectionId, const std::vector<std::uint8_t>*>> sections;
  std::vector<std::uint8_t> reference_body;
  if (inter) {
    ByteWriter rw(reference_body);
    rw.u32(static_cast<std::uint32_t>(frame.reference.hash));
    rw.u32(static_cast<std::uint32_t>(frame.reference.hash >> 32));
    rw.u32(static_cast<std::uint32_t>(m));
    rw.u32(frame.update_size);
    rw.u32(frame.new_count);
    sections.emplace_back(SectionId::kReference, &reference_body);
  }
  sections.emplace_back(SectionId::kTables, &tables_body);
  sections.emplace_back(SectionId::kSegments, &segments_body);
  sections.emplace_back(SectionId::kPayload, &frame.payload);

  std::vector<std::uint8_t> out;
  ByteWriter w(out);
  for (std::uint8_t c : kFrameMagic) {
    w.u8(c);
  }
  w.u16(kFrameVersion);
  w.u8(static_cast<std::uint32_t>(type));
  w.u8(0);  // reserved
  w.f32(frame.voxel_size);
  w.f32(frame.trunc_dist);
  w.u32(static_cast<std::uint32_t>(kBlockSize));
  w.u32(k);
  w.f32(frame.params.quantization_scale);
  w.u32(k);  // one weight per kept basis
  w.u32(frame.block_count);
  w.u32(frame.segment_size);
  w.u32(static_cast<std::uint32_t>(sections.size()));
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

  header.type = static_cast<FrameType>(type);
  VKC_TRY(check_frame_type(header.type, FrameType::kInter, "codec frame"));
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

core::Result<ParsedFrame> parse_frame(const std::uint8_t* data,
                                      std::size_t size,
                                      std::uint32_t max_blocks,
                                      const FrameReference* reference) {
  VKC_ASSIGN(const FrameHeader header, read_frame_header(data, size));
  ParsedFrame frame;
  frame.header = header;
  const bool inter = header.type == FrameType::kInter;
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
    offset += length;
  }
  if (offset != body_bytes) {
    return bad("bytes after the last section");
  }
  // Every section, but REFERENCE on a P-frame alone.
  for (std::uint32_t i = 0; i < kSectionCount; ++i) {
    const bool present = found[i].data != nullptr;
    if (static_cast<SectionId>(i + 1) == SectionId::kReference) {
      if (present != inter) {
        return bad(inter ? "a P-frame's REFERENCE section is missing"
                         : "an intra frame has a REFERENCE section");
      }
    } else if (!present) {
      return bad("section " + std::to_string(i + 1) + " is missing");
    }
  }
  const SectionBody& tables_section = section(found, SectionId::kTables);
  const SectionBody& segments_section = section(found, SectionId::kSegments);
  const SectionBody& payload_section = section(found, SectionId::kPayload);

  // A P-frame's M reference entries in update segments of U, then its A new
  // blocks; an intra frame's blocks are all new.
  std::uint64_t reference_hash = 0;
  std::uint32_t m = 0;
  std::uint32_t u_size = 0;
  std::uint32_t a = n;
  if (inter) {
    const SectionBody& r = section(found, SectionId::kReference);
    if (r.size != kReferenceBodyBytes) {
      return bad("the REFERENCE section is not " +
                 std::to_string(kReferenceBodyBytes) + " bytes");
    }
    ByteReader rr(r.data, r.size);
    const std::uint64_t low = rr.u32();
    reference_hash = low | std::uint64_t(rr.u32()) << 32;
    m = rr.u32();
    u_size = rr.u32();
    a = rr.u32();
    if (u_size == 0) {
      return bad("update size 0");
    }
    if (a > n || n > std::uint64_t(m) + a) {
      return bad("it holds " + std::to_string(n) + " blocks, not between its " +
                 std::to_string(a) + " new blocks and those plus the " +
                 std::to_string(m) + " of its reference");
    }
    // Stream positions are u32, on the device too.
    if (std::uint64_t(m) + a > std::numeric_limits<std::uint32_t>::max()) {
      return bad("more than 2^32 - 1 entries");
    }
  }

  const std::uint32_t k = header.params.coefficient_count;
  VKC_TRY(read_tables(tables_section.data, tables_section.size, header.type, k,
                      frame.tables));

  const std::uint64_t segments =
      frame_segment_count(m, u_size) + frame_segment_count(a, r_size);
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
  // A P-frame against the caller's reference only now, so a corrupt frame
  // stays InvalidArgument and a motion frame Unsupported whatever is held;
  // M, a count the caller holds, then bounds the reader's M + A entries.
  if (inter) {
    if (reference == nullptr) {
      return core::Status::not_found(
          "codec frame: a P-frame, and no reference frame is held; decode an "
          "I-frame first");
    }
    if (reference_hash != reference->hash ||
        m != reference->header.block_count) {
      return core::Status::not_found(
          "codec frame: a P-frame against a reference of " + std::to_string(m) +
          " blocks that is not the one held (" +
          std::to_string(reference->header.block_count) +
          " blocks); decode an I-frame first");
    }
    if (const char* d = coding_difference(reference->header, header.voxel_size,
                                          header.trunc_dist, header.params)) {
      return bad(std::string("the P-frame's ") + d + " is not its reference's");
    }
    frame.reference = *reference;
    frame.update_size = u_size;
  }
  frame.new_count = a;
  // A block count that is not consistent with the segment table is a corrupt
  // frame, and one that is but exceeds the caller's room is a sound frame too
  // big to hold -- which the caller answers by growing, not by dropping the
  // frame.
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
                           const Vec3i* prev_last, const Vec3i* first) {
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
  if (prev_last != nullptr && first != nullptr &&
      !coord_less(*prev_last, *first)) {
    return bad("segment " + std::to_string(s) +
               " does not start after the one before ends");
  }
  return {};
}

core::Status check_survivors(const std::vector<BlockMode>& modes,
                             std::uint32_t n, std::uint32_t a) {
  std::uint64_t survivors = 0;
  for (BlockMode mode : modes) {
    survivors += mode == BlockMode::kKeep || mode == BlockMode::kSkip ||
                         mode == BlockMode::kZero
                     ? 1
                     : 0;
  }
  if (survivors + a != n) {
    return bad(std::to_string(survivors) + " reference blocks survive and " +
               std::to_string(a) + " are new, but the frame holds " +
               std::to_string(n));
  }
  return {};
}

core::Result<Frame> read_frame(const std::uint8_t* data, std::size_t size,
                               std::uint32_t max_blocks,
                               const FrameReference* reference) {
  VKC_ASSIGN(ParsedFrame parsed,
             parse_frame(data, size, max_blocks, reference));
  return decode_frame(std::move(parsed));
}

core::Result<Frame> decode_frame(ParsedFrame parsed) {
  VKC_TRY(
      check_frame_type(parsed.header.type, FrameType::kInter, "decode_frame"));
  for (FrequencyTable& t : parsed.tables) t.build_decode();
  const FrameHeader& header = parsed.header;
  const bool inter = header.type == FrameType::kInter;
  const std::uint32_t n = header.block_count;
  const std::uint32_t r_size = header.segment_size;
  const std::uint32_t k = header.params.coefficient_count;
  // An intra frame is M = 0: its blocks are all new.
  const std::uint32_t m = inter ? parsed.reference.header.block_count : 0;
  const std::uint32_t u_size = inter ? parsed.update_size : 0;
  const std::uint32_t a = inter ? parsed.new_count : n;
  const std::uint64_t entries = std::uint64_t(m) + a;
  Frame frame;
  frame.type = header.type;
  frame.voxel_size = header.voxel_size;
  frame.reference = parsed.reference;
  frame.blocks.trunc_dist = header.trunc_dist;
  frame.blocks.params = header.params;
  // max_blocks and the held reference bound these in bytes, but not below
  // what a 32-bit size_t can count: entries * K reaches 2^41. Past this check
  // every index below fits.
  if (a > frame.coords.max_size() || m > frame.modes.max_size() ||
      entries * k > frame.blocks.coefficients.max_size() ||
      entries * kMaskWordsPerBlock > frame.blocks.masks.max_size()) {
    return core::Status::out_of_memory(
        "codec frame: " + std::to_string(entries) + " entries of " +
        std::to_string(k) +
        " coefficients exceed this platform's address space");
  }
  frame.modes.resize(m);
  frame.coords.resize(a);
  frame.blocks.coefficients.resize(std::size_t(entries) * k);
  frame.blocks.masks.resize(std::size_t(entries) * kMaskWordsPerBlock);
  const std::uint8_t* stream = parsed.payload;
  const Vec3i* prev_last = nullptr;  // the block segment before's last
  for (std::size_t s = 0; s < parsed.segment_lengths.size(); ++s) {
    RansReader r(stream, parsed.segment_lengths[s]);
    BlockReader reader{r, parsed.tables};
    const uvec2 range =
        segment_range(static_cast<uint>(s), m, u_size, a, r_size);
    SegmentFault fault = SegmentFault::kNone;
    for (uint p = range.x; p < range.y && !r.failed(); ++p) {
      reader.g_mask = &frame.blocks.masks[std::size_t(p) * kMaskWordsPerBlock];
      reader.coefficients = &frame.blocks.coefficients[std::size_t(p) * k];
      if (p < m) {
        frame.modes[p] = static_cast<BlockMode>(reader.read_entry(k));
      } else {
        if (!reader.read_block(p == range.x, k)) {
          fault = SegmentFault::kCoordOverflow;
          break;
        }
        frame.coords[p - m] = reader.g_coord;
      }
    }
    if (fault == SegmentFault::kNone && !r.finish()) {
      fault = SegmentFault::kCorrupt;
    }
    // An update segment has no coordinate, and a block segment follows the
    // block segment before it, if any.
    const bool blocks = range.x >= m;
    VKC_TRY(check_segment(s, fault, blocks ? prev_last : nullptr,
                          blocks ? &frame.coords[range.x - m] : nullptr));
    if (blocks) prev_last = &frame.coords[range.y - 1 - m];
    stream += parsed.segment_lengths[s];
  }
  if (inter) VKC_TRY(check_survivors(frame.modes, n, a));
  return frame;
}

core::Result<MergedFrame> merge_frame(const std::vector<Vec3i>& reference,
                                      const std::vector<BlockMode>& modes,
                                      const std::vector<Vec3i>& added) {
  if (modes.size() != reference.size()) {
    return bad("expected " + std::to_string(reference.size()) +
               " modes, one per reference block, got " +
               std::to_string(modes.size()));
  }
  if (std::uint64_t(reference.size()) + added.size() >
      std::numeric_limits<std::uint32_t>::max()) {
    return bad("more than 2^32 - 1 entries");
  }
  const auto m = static_cast<std::uint32_t>(reference.size());
  MergedFrame out;
  auto emit = [&out](const Vec3i& c, std::uint32_t from_reference,
                     std::uint32_t payload) {
    out.coords.push_back(c);
    out.sources.push_back(ApplySource{from_reference, payload});
  };
  VKC_TRY(merge_by_coord(
      reference.size(),
      [&](std::size_t e) -> const Vec3i& { return reference[e]; }, added.size(),
      [&](std::size_t q) -> const Vec3i& { return added[q]; },
      [&](std::size_t e, std::size_t q) -> core::Status {
        // One spelling per block: a reference block is recoded with zero,
        // never removed and added.
        if (e != kAbsent && q != kAbsent) {
          return bad("new block " + std::to_string(q) +
                     " is on reference block " + std::to_string(e));
        }
        if (q != kAbsent) {
          emit(added[q], kNoSource, m + static_cast<std::uint32_t>(q));
          return {};
        }
        const auto i = static_cast<std::uint32_t>(e);
        switch (modes[e]) {
          case BlockMode::kKeep:
            emit(reference[e], i, i);
            return {};
          case BlockMode::kSkip:
            emit(reference[e], i, kNoSource);
            return {};
          case BlockMode::kZero:
            emit(reference[e], kNoSource, i);
            return {};
          case BlockMode::kRemove:
            return {};
          case BlockMode::kMotion:
            return core::Status::unsupported(
                "codec frame: the motion mode, which this reader does not "
                "decode");
        }
        return bad("mode " + std::to_string(static_cast<uint>(modes[e])) +
                   " is not a mode");
      }));
  return out;
}

}  // namespace volumetric_kit::recon::codec::detail
