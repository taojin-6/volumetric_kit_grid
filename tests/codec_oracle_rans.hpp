// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

// The CPU rANS coder: a segment's stream written and read on the host, as
// rans.hpp lays it out, for the tests' oracle (codec_oracle.hpp) that pins
// the kernels' bytes. Integer arithmetic on fixed widths alone, as theirs.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "rans.hpp"

namespace codec_oracle {

namespace d = volumetric_kit::grid::codec::detail;

// L, the state's lower bound: the state the encoder starts in, so the one a
// correct decode ends in.
inline constexpr std::uint32_t kRansLower = 1u << 16;
// The widest raw chunk one coder step carries; put_bits splits a wider field.
inline constexpr std::uint32_t kRansMaxRawBits = d::kRansScaleBits;
// The widest raw field put_bits and get_bits take: one uint32_t.
inline constexpr std::uint32_t kRansMaxFieldBits = 32;
// The encoder's renormalization shift: a word leaves the state before a
// symbol of frequency f whenever (x >> 20) >= f.
inline constexpr std::uint32_t kRansRenormShift = 32 - d::kRansScaleBits;

// A table as a decoder reads it: its frequencies, and the symbol whose
// [cum, cum + freq) holds each slot of [0, M). An empty table has no slots,
// so a decoder that reaches for it fails.
struct DecodeTable {
  std::vector<std::uint16_t> freq;
  std::vector<std::uint16_t> cum;
  std::vector<std::uint8_t> slot_symbol;  // no codec alphabet passes 256
};

inline DecodeTable decode_table(const d::FrequencyTable& table) {
  DecodeTable t{table.freq, table.cum, {}};
  if (table.cum.empty() || table.cum.back() == 0) return t;
  t.slot_symbol.resize(d::kRansScale);
  for (std::size_t s = 0; s < table.freq.size(); ++s) {
    std::fill(t.slot_symbol.begin() + table.cum[s],
              t.slot_symbol.begin() + table.cum[s + 1],
              static_cast<std::uint8_t>(s));
  }
  return t;
}

// Collects symbols in decode order and encodes them, backwards, on finish().
//
// A symbol its table cannot encode -- past its alphabet, or of frequency 0, so
// the table was counted without it -- or a raw field wider than
// kRansMaxFieldBits is recorded as a coder step of frequency 0, which finish()
// refuses at the division it would otherwise make. In band rather than a
// flag: a failure flag stored in put() made a room0-sized frame's write
// 47.5 ms against this form's 42.8 (Apple M5 Max, Release).
class RansWriter {
 public:
  void put(const d::FrequencyTable& table, std::uint32_t symbol) {
    ops_.push_back(symbol < table.freq.size()
                       ? Op{table.cum[symbol], table.freq[symbol]}
                       : Op{0, 0});
  }

  // The low @p bits bits of @p value as uniform raw bits, low chunk first, in
  // chunks of at most kRansMaxRawBits.
  void put_bits(std::uint32_t value, std::uint32_t bits) {
    if (bits > kRansMaxFieldBits) {
      ops_.push_back(Op{0, 0});
      return;
    }
    while (bits > 0) {
      const std::uint32_t n = std::min(bits, kRansMaxRawBits);
      const std::uint32_t chunk = value & ((1u << n) - 1u);
      const std::uint32_t shift = d::kRansScaleBits - n;
      ops_.push_back(Op{static_cast<std::uint16_t>(chunk << shift),
                        static_cast<std::uint16_t>(1u << shift)});
      value >>= n;  // n <= 12, so never a full-width shift
      bits -= n;
    }
  }

  // Encodes everything put as one stream, appended to @p out so a frame's
  // segments go straight into its payload, and empties the writer. False,
  // appending nothing, if anything put could not be encoded.
  bool finish(std::vector<std::uint8_t>& out) {
    std::uint32_t x = kRansLower;
    words_.clear();  // in emission order, reversed below
    for (std::size_t i = ops_.size(); i-- > 0;) {
      const Op op = ops_[i];
      if (op.freq == 0) {
        ops_.clear();
        return false;
      }
      if ((x >> kRansRenormShift) >= op.freq) {
        words_.push_back(static_cast<std::uint16_t>(x & 0xFFFFu));
        x >>= 16;
      }
      x = ((x / op.freq) << d::kRansScaleBits) + (x % op.freq) + op.start;
    }
    ops_.clear();
    // push_back, which grows geometrically. Not reserve: that takes exactly
    // what it is asked for, so a reserve per appended segment copied the
    // frame once per segment. And not resize-then-index: GCC 13 at -O3
    // cannot see that resize grew the caller's vector and fails the build
    // with -Wstringop-overflow. All three forms that avoid both time the
    // same.
    auto put_word = [&out](std::uint16_t w) {
      out.push_back(static_cast<std::uint8_t>(w & 0xFFu));
      out.push_back(static_cast<std::uint8_t>(w >> 8));
    };
    put_word(static_cast<std::uint16_t>(x >> 16));
    put_word(static_cast<std::uint16_t>(x & 0xFFFFu));
    for (std::size_t i = words_.size(); i-- > 0;) {
      put_word(words_[i]);
    }
    return true;
  }

 private:
  struct Op {
    std::uint16_t start;
    std::uint16_t freq;
  };
  std::vector<Op> ops_;
  // finish()'s scratch, a member so a frame's segments reuse one allocation.
  std::vector<std::uint16_t> words_;
};

// Decodes one stream RansWriter::finish made, never reading outside
// [data, data + size), which need not be aligned.
//
// A failure -- a stream too short to hold a state, a state out of range, a
// read past the end, a symbol from an empty table, a raw field wider than any
// writer puts -- is sticky: failed() is set, every later call returns 0, and
// finish() reports it. So a caller decodes a whole segment and checks once,
// and garbage input costs a wasted loop, never a crash.
class RansReader {
 public:
  RansReader(const std::uint8_t* data, std::size_t size)
      : data_(data), size_(size) {
    if (size_ < 4 || size_ % 2 != 0) {
      failed_ = true;
      return;
    }
    x_ = (std::uint32_t(word_at(0)) << 16) | word_at(2);
    pos_ = 4;
    if (x_ < kRansLower) {
      failed_ = true;
    }
  }

  std::uint32_t get(const DecodeTable& table) {
    if (failed_ || table.slot_symbol.empty()) {
      failed_ = true;
      return 0;
    }
    const std::uint32_t slot = x_ & (d::kRansScale - 1u);
    const std::uint32_t s = table.slot_symbol[slot];
    x_ = table.freq[s] * (x_ >> d::kRansScaleBits) + slot - table.cum[s];
    renormalize();
    return failed_ ? 0 : s;
  }

  // The next @p bits raw bits, low chunk first.
  std::uint32_t get_bits(std::uint32_t bits) {
    if (bits > kRansMaxFieldBits) {
      failed_ = true;
      return 0;
    }
    std::uint32_t value = 0;
    std::uint32_t shift_out = 0;
    while (bits > 0 && !failed_) {
      const std::uint32_t n = std::min(bits, kRansMaxRawBits);
      const std::uint32_t shift = d::kRansScaleBits - n;
      const std::uint32_t slot = x_ & (d::kRansScale - 1u);
      const std::uint32_t chunk = slot >> shift;
      // freq = 2^shift, start = chunk << shift.
      x_ = (x_ >> d::kRansScaleBits << shift) + slot - (chunk << shift);
      renormalize();
      value |= chunk << shift_out;
      shift_out += n;
      bits -= n;
    }
    return failed_ ? 0 : value;
  }

  bool failed() const noexcept { return failed_; }

  // Whether the stream decoded cleanly and completely: no failure, every word
  // consumed, and the state back at kRansLower, where the encoder started.
  //
  // A consistency check, not an integrity check. It catches a truncated
  // stream always, and a stream read with the wrong tables or the wrong
  // symbol count, or with a flipped bit in a table symbol, nearly always. It
  // cannot catch a flipped raw bit: those bits leave the state as they are
  // decoded, and every value of them is valid. Nor can it catch a flip that
  // swaps a symbol for another of equal frequency, after which the decoder
  // resynchronizes. codec_rans_test measures both.
  bool finish() const noexcept {
    return !failed_ && pos_ == size_ && x_ == kRansLower;
  }

 private:
  std::uint16_t word_at(std::size_t at) const {
    return static_cast<std::uint16_t>(data_[at] | (data_[at + 1] << 8));
  }

  void renormalize() {
    if (x_ < kRansLower) {
      if (pos_ + 2 > size_) {
        failed_ = true;
        return;
      }
      x_ = (x_ << 16) | word_at(pos_);
      pos_ += 2;
    }
  }

  const std::uint8_t* data_;
  std::size_t size_;
  std::size_t pos_ = 0;
  std::uint32_t x_ = 0;
  bool failed_ = false;
};

}  // namespace codec_oracle
