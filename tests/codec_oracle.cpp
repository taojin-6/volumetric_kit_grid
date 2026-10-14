// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include "codec_oracle.hpp"

#include <algorithm>
#include <limits>
#include <string>

#include "codec_oracle_rans.hpp"

namespace codec_oracle {

using vr::Vec3i;

// The kernels' own models and block grammar, compiled here as C++ once these
// four names are defined: in a namespace of their own, so a model function
// this file does not call is not an unused one.
namespace glsl {

using uint = std::uint32_t;
using glm::ivec3;
using glm::uvec2;

// GLSL's findMSB: the index of the highest set bit, -1 for 0.
int findMSB(uint v) { return v == 0 ? -1 : 31 - __builtin_clz(v); }

#include "shaders/frame_models.glsl"

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

// The coder steps a walk puts: RansWriter's ops.
struct StepSink {
  std::uint32_t steps = 0;
  void symbol(uint, uint) { ++steps; }
  void bits(uint, uint n) {
    steps += (n + kRansMaxRawBits - 1) / kRansMaxRawBits;
  }
};

struct WriteSink {
  RansWriter& writer;
  const std::vector<d::FrequencyTable>& tables;
  void symbol(uint model, uint s) { writer.put(tables[model], s); }
  void bits(uint value, uint n) { writer.put_bits(value, n); }
};

// The grammar's reader half over one segment's stream.
struct BlockReader {
  RansReader& coder;
  const std::vector<DecodeTable>& tables;
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

}  // namespace glsl

namespace {

using glsl::uint;

vkc::Status bad_write(const std::string& why) {
  return vkc::Status::invalid_argument("write_frame: " + why);
}

// The checks write_frame makes before coding anything, in the order the
// device writer makes its own: the library's where they are shared.
vkc::Status check_frame(const Frame& frame, const FrameWriteOptions& options) {
  VKC_TRY(d::check_frame_type(frame.type, "write_frame"));
  const bool inter = frame.type == d::FrameType::kInter;
  const DctBlocks& b = frame.blocks;
  VKC_TRY(d::check_coding(frame.voxel_size, b.trunc_dist, b.params));
  VKC_TRY(d::check_segment_size("write_frame", "segment_size",
                                options.segment_size));
  if (inter) {
    VKC_TRY(d::check_segment_size("write_frame", "update_size",
                                  options.update_size));
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
  const std::uint32_t k = b.params.coefficient_count;
  // Sizes in 64 bits: (M + A) * K wraps a 32-bit size_t long before 2^32
  // entries.
  const std::uint64_t coefficient_count = (m + a) * k;
  const std::uint64_t mask_count = (m + a) * codec::kMaskWordsPerBlock;
  if (b.coefficients.size() != coefficient_count) {
    return bad_write("expected " + std::to_string(coefficient_count) +
                     " coefficients, got " +
                     std::to_string(b.coefficients.size()));
  }
  if (b.masks.size() != mask_count) {
    return bad_write("expected " + std::to_string(mask_count) +
                     " mask words, got " + std::to_string(b.masks.size()));
  }
  if (d::frame_segment_count(m, options.update_size) +
          d::frame_segment_count(a, options.segment_size) >
      d::kMaxFrameSegments) {
    return bad_write(
        "more than 2^30 - 1 segments: SEGMENTS would outgrow its "
        "u32 length");
  }
  for (std::size_t i = 1; i < a; ++i) {
    if (!d::coord_less(frame.coords[i - 1], frame.coords[i])) {
      return bad_write(
          "coordinates are not strictly increasing in (z, y, x) "
          "at entry " +
          std::to_string(i));
    }
  }
  // What the device writer's count pass finds too. Only what is written: a
  // skip or remove entry's payload is ignored.
  std::uint32_t largest_mode = 0;
  bool motion = false;
  bool out_of_range = false;
  for (std::uint64_t p = 0; p < m + a; ++p) {
    if (p < m) {
      const auto mode = static_cast<uint>(frame.modes[std::size_t(p)]);
      if (mode >= glsl::kModeSymbols) {
        largest_mode = std::max(largest_mode, mode);
      }
      motion = motion || mode == glsl::kModeMotion;
      if (mode != glsl::kModeKeep && mode != glsl::kModeZero) continue;
    }
    for (std::uint32_t j = 0; j < k; ++j) {
      const std::int16_t c = b.coefficients[std::size_t(p * k + j)];
      out_of_range = out_of_range || c < -codec::kMaxQuantizedMagnitude ||
                     c > codec::kMaxQuantizedMagnitude;
    }
  }
  return d::check_entries(largest_mode, motion, out_of_range);
}

}  // namespace

std::uint32_t model_alphabet(std::uint32_t model, std::uint32_t k) {
  return glsl::model_alphabet(model, k);
}

std::vector<std::uint32_t> entry_steps(const Frame& frame,
                                       std::uint32_t segment_size) {
  const DctBlocks& b = frame.blocks;
  const std::uint32_t k = b.params.coefficient_count;
  const auto m = static_cast<uint>(frame.modes.size());
  const auto n = static_cast<uint>(m + frame.coords.size());
  std::vector<std::uint32_t> out(n);
  for (uint p = 0; p < n; ++p) {
    glsl::StepSink sink;
    glsl::BlockWriter<glsl::StepSink> walk{sink};
    walk.g_mask = &b.masks[std::size_t(p) * codec::kMaskWordsPerBlock];
    walk.coefficients = &b.coefficients[std::size_t(p) * k];
    if (p < m) {
      const auto mode = static_cast<uint>(frame.modes[p]);
      if (mode < glsl::kModeSymbols) walk.emit_entry(mode, k);
    } else {
      const uint q = p - m;
      const bool first = q % segment_size == 0;
      walk.emit_block(first, frame.coords[first ? q : q - 1], frame.coords[q],
                      k);
    }
    out[p] = sink.steps;
  }
  return out;
}

StepLayout step_layout(const std::vector<std::uint32_t>& steps, std::uint32_t m,
                       std::uint32_t u, std::uint32_t r) {
  const auto n = static_cast<uint>(steps.size());
  const std::uint64_t segments =
      d::frame_segment_count(m, u) + d::frame_segment_count(n - m, r);
  StepLayout out;
  out.starts.assign(std::size_t(n) + 1, 0);
  out.slot_offsets.assign(std::size_t(segments) + 1, 0);
  for (uint p = 0; p < n; ++p) out.starts[p + 1] = out.starts[p] + steps[p];
  for (std::uint64_t s = 0; s < segments; ++s) {
    const glsl::uvec2 range =
        glsl::segment_range(static_cast<uint>(s), m, u, n - m, r);
    const std::uint64_t words = out.starts[range.y] - out.starts[range.x];
    out.slot_offsets[s + 1] = out.slot_offsets[s] + ((words + 3) & ~1ull);
  }
  return out;
}

vkc::Result<std::vector<std::uint8_t>> write_frame(
    const Frame& frame, const FrameWriteOptions& options) {
  VKC_TRY(check_frame(frame, options));
  const bool inter = frame.type == d::FrameType::kInter;
  const DctBlocks& b = frame.blocks;
  const std::uint32_t k = b.params.coefficient_count;
  // Each below 2^32 together (check_frame).
  const auto m =
      static_cast<uint>(inter ? frame.reference.header.block_count : 0);
  const auto a = static_cast<uint>(frame.coords.size());
  const std::uint32_t u_size = inter ? options.update_size : 0;
  const std::uint32_t r_size = options.segment_size;
  const std::uint64_t segments =
      d::frame_segment_count(m, u_size) + d::frame_segment_count(a, r_size);

  // Segment s through a writer: an update segment's reference entries, or a
  // block segment's blocks.
  auto emit_segment = [&](auto& writer, std::uint64_t s) {
    const glsl::uvec2 range =
        glsl::segment_range(static_cast<uint>(s), m, u_size, a, r_size);
    for (uint p = range.x; p < range.y; ++p) {
      writer.g_mask = &b.masks[std::size_t(p) * codec::kMaskWordsPerBlock];
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
  const std::uint32_t models =
      glsl::model_count(static_cast<uint>(frame.type), k);
  std::vector<std::vector<std::uint64_t>> counts(models);
  for (std::uint32_t model = 0; model < models; ++model) {
    counts[model].assign(glsl::model_alphabet(model, k), 0);
  }
  glsl::CountSink count_sink{counts};
  glsl::BlockWriter<glsl::CountSink> counter{count_sink};
  for (std::uint64_t s = 0; s < segments; ++s) {
    emit_segment(counter, s);
  }
  std::vector<d::FrequencyTable> tables = d::frame_tables(counts);

  // Pass 2: each segment as its own stream, appended straight onto the
  // payload.
  std::vector<std::uint8_t> payload;
  std::vector<std::uint32_t> lengths;
  lengths.reserve(static_cast<std::size_t>(segments));
  RansWriter writer;
  glsl::WriteSink write_sink{writer, tables};
  glsl::BlockWriter<glsl::WriteSink> coder{write_sink};
  for (std::uint64_t s = 0; s < segments; ++s) {
    emit_segment(coder, s);
    const std::size_t stream_at = payload.size();
    if (!writer.finish(payload)) {
      // Pass 1 counted every symbol pass 2 puts, so this is a bug here.
      return vkc::Status::io_error(
          "write_frame: a table refused a symbol it was counted from");
    }
    lengths.push_back(static_cast<std::uint32_t>(payload.size() - stream_at));
  }
  std::uint32_t survivors = 0;
  for (d::BlockMode mode : frame.modes) {
    survivors += mode != d::BlockMode::kRemove ? 1u : 0u;
  }
  d::CodedFrame coded;
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
  return d::assemble_frame(frame.type, coded);
}

vkc::Result<Frame> read_frame(const std::uint8_t* data, std::size_t size,
                              std::uint32_t max_blocks,
                              const d::FrameReference* reference) {
  VKC_ASSIGN(const d::ParsedFrame parsed,
             d::parse_frame(data, size, max_blocks, reference));
  return decode_frame(parsed);
}

vkc::Result<Frame> decode_frame(const d::ParsedFrame& parsed) {
  VKC_TRY(d::check_frame_type(parsed.header.type, "decode_frame"));
  std::vector<DecodeTable> tables;
  tables.reserve(parsed.tables.size());
  for (const d::FrequencyTable& t : parsed.tables) {
    tables.push_back(decode_table(t));
  }
  const d::FrameHeader& header = parsed.header;
  const bool inter = header.type == d::FrameType::kInter;
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
      entries * codec::kMaskWordsPerBlock > frame.blocks.masks.max_size()) {
    return vkc::Status::out_of_memory(
        "codec frame: " + std::to_string(entries) + " entries of " +
        std::to_string(k) +
        " coefficients exceed this platform's address space");
  }
  frame.modes.resize(m);
  frame.coords.resize(a);
  frame.blocks.coefficients.resize(std::size_t(entries) * k);
  frame.blocks.masks.resize(std::size_t(entries) * codec::kMaskWordsPerBlock);
  const std::uint8_t* stream = parsed.payload;
  const Vec3i* prev_last = nullptr;  // the block segment before's last
  for (std::size_t s = 0; s < parsed.segment_lengths.size(); ++s) {
    RansReader r(stream, parsed.segment_lengths[s]);
    glsl::BlockReader reader{r, tables};
    const glsl::uvec2 range =
        glsl::segment_range(static_cast<uint>(s), m, u_size, a, r_size);
    d::SegmentFault fault = d::SegmentFault::kNone;
    for (uint p = range.x; p < range.y && !r.failed(); ++p) {
      reader.g_mask =
          &frame.blocks.masks[std::size_t(p) * codec::kMaskWordsPerBlock];
      reader.coefficients = &frame.blocks.coefficients[std::size_t(p) * k];
      if (p < m) {
        frame.modes[p] = static_cast<d::BlockMode>(reader.read_entry(k));
      } else {
        if (!reader.read_block(p == range.x, k)) {
          fault = d::SegmentFault::kCoordOverflow;
          break;
        }
        frame.coords[p - m] = reader.g_coord;
      }
    }
    if (fault == d::SegmentFault::kNone && !r.finish()) {
      fault = d::SegmentFault::kCorrupt;
    }
    // An update segment has no coordinate, and a block segment follows the
    // block segment before it, if any.
    const bool blocks = range.x >= m;
    if (fault == d::SegmentFault::kNone && blocks && prev_last != nullptr &&
        !d::coord_less(*prev_last, frame.coords[range.x - m])) {
      fault = d::SegmentFault::kOutOfOrder;
    }
    VKC_TRY(d::check_segment(s, fault));
    if (blocks) prev_last = &frame.coords[range.y - 1 - m];
    stream += parsed.segment_lengths[s];
  }
  if (inter) {
    std::uint64_t survivors = 0;
    for (const d::BlockMode mode : frame.modes) {
      survivors += mode == d::BlockMode::kKeep || mode == d::BlockMode::kSkip ||
                           mode == d::BlockMode::kZero
                       ? 1
                       : 0;
    }
    VKC_TRY(d::check_survivors(survivors, n, a));
  }
  return frame;
}

}  // namespace codec_oracle
