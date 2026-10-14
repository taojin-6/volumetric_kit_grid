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

// --- The models. -------------------------------------------------------------
//
// shaders/frame_models.glsl, the kernels' own GLSL, compiled here as C++ once
// these two names are defined: in a namespace of their own, so a model
// function this file does not call is not an unused one.
namespace glsl {

using uint = std::uint32_t;
using glm::uvec2;

#include "shaders/frame_models.glsl"

}  // namespace glsl

namespace {

using namespace glsl;

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

}  // namespace

std::uint32_t frame_model_count(FrameType type, std::uint32_t k) {
  return model_count(static_cast<uint>(type), k);
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

// Bounds-checked and sticky: a read past the end sets failed and returns 0,
// so a parser checks once per structure rather than per field.
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
  return core::Status::invalid_argument("frame writer: " + why);
}

bool positive_finite(float f) { return std::isfinite(f) && f > 0.0f; }

bool same_bits(float a, float b) { return std::memcmp(&a, &b, sizeof(a)) == 0; }

}  // namespace

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

namespace {

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

}  // namespace

core::Status check_frame_type(FrameType type, const char* who) {
  if (static_cast<uint>(type) <= static_cast<uint>(FrameType::kInter)) {
    return {};
  }
  return core::Status::unsupported(std::string(who) + ": frame type " +
                                   std::to_string(static_cast<uint>(type)) +
                                   " (only intra and P-frames are coded)");
}

core::Status check_segment_size(const char* who, const char* name,
                                std::uint32_t size) {
  if (size >= 1 && size <= kMaxDeviceDecodeSegmentSize) return {};
  return core::Status::invalid_argument(
      std::string(who) + ": " + name + " " + std::to_string(size) +
      " is not in [1, " + std::to_string(kMaxDeviceDecodeSegmentSize) + "]");
}

core::Status check_coding(float voxel_size, float trunc_dist,
                          const CodecParams& params) {
  if (!positive_finite(voxel_size)) {
    return bad_write("voxel_size must be finite and positive");
  }
  if (!positive_finite(trunc_dist)) {
    return bad_write("trunc_dist must be finite and positive");
  }
  return params.validate();
}

core::Status check_entries(std::uint32_t largest_mode, bool motion,
                           bool out_of_range) {
  if (largest_mode != 0) {
    return bad_write("mode " + std::to_string(largest_mode) + " is not a mode");
  }
  if (motion) {
    return core::Status::unsupported(
        "frame writer: the motion mode, which this writer does not code");
  }
  if (out_of_range) {
    return bad_write("a coefficient or residual is outside +-" +
                     std::to_string(kMaxQuantizedMagnitude));
  }
  return {};
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
  VKC_TRY(check_frame_type(type, "assemble_frame"));
  const bool inter = type == FrameType::kInter;
  // The header's fields, which the caller coded the segments for.
  VKC_TRY(check_coding(frame.voxel_size, frame.trunc_dist, frame.params));
  VKC_TRY(
      check_segment_size("frame writer", "segment_size", frame.segment_size));
  const std::uint64_t m = frame.reference.header.block_count;
  if (!inter) {
    const char* set = frame.reference.hash != 0 ? "reference hash"
                      : m != 0                  ? "reference block count"
                      : frame.update_size != 0  ? "update_size"
                      : frame.new_count != 0    ? "new_count"
                                                : nullptr;
    if (set != nullptr) {
      return bad_write(
          std::string("an intra frame has no reference, but its ") + set +
          " is not 0");
    }
  }
  if (inter) {
    VKC_TRY(
        check_segment_size("frame writer", "update_size", frame.update_size));
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
  VKC_TRY(check_frame_type(header.type, "codec frame"));
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
  VKC_TRY(
      check_segment_size("codec frame", "segment size", header.segment_size));
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
    VKC_TRY(check_segment_size("codec frame", "update size", u_size));
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
    frame.new_count = a;
  }
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

core::Status check_segment(std::uint64_t s, SegmentFault fault) {
  switch (fault) {
    case SegmentFault::kNone:
      return {};
    case SegmentFault::kCoordOverflow:
      return bad("segment " + std::to_string(s) +
                 " steps a coordinate outside int32");
    case SegmentFault::kOutOfOrder:
      return bad("segment " + std::to_string(s) +
                 " does not start after the one before ends");
    case SegmentFault::kCorrupt:
      break;
  }
  return bad("segment " + std::to_string(s) + " is corrupt");
}

core::Status check_survivors(std::uint64_t survivors, std::uint32_t n,
                             std::uint32_t a) {
  if (survivors + a != n) {
    return bad(std::to_string(survivors) + " reference blocks survive and " +
               std::to_string(a) + " are new, but the frame holds " +
               std::to_string(n));
  }
  return {};
}

core::Status check_new_block(std::uint32_t q, std::uint32_t e) {
  return bad("new block " + std::to_string(q) + " is on reference block " +
             std::to_string(e));
}

}  // namespace volumetric_kit::recon::codec::detail
