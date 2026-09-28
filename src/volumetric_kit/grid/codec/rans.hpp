// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file rans.hpp
/// @brief The codec's entropy coder: static-table rANS with a 32-bit state,
///        16-bit output words and 12-bit probabilities.
///
/// Internal (under src/, never installed), and the **reference** the GPU
/// kernels of the 2026-09-26 decision's fifth PR must match byte for byte. So
/// everything here is integer arithmetic on fixed widths, nothing depends on
/// the host's float behaviour, and nothing needs more than 32-bit arithmetic.
/// The renormalization bound is the one place that takes care: it is tested
/// as `(x >> 20) >= f`, since the equivalent `x >= f << 20` is 2^32 at a
/// probability of one and wraps to 0 in 32 bits.
///
/// The scheme is Giesen's word-oriented rANS: the state `x` lives in
/// `[L, L << 16)` with `L = 2^16`, a symbol of frequency `f` at cumulative
/// start `c` (out of `M = 2^12`) encodes as
/// `x' = (x / f) * M + x % f + c`, and one 16-bit word leaves the state first
/// whenever `x >= f << 20` -- one is always enough, in both directions. A
/// symbol with `f == M` costs nothing, which is what makes an all-zero band
/// free rather than cheap.
///
/// rANS decodes in the reverse of the order it encodes, so @ref RansWriter
/// takes symbols in **decode order** and runs the coder backwards when it
/// finishes; a caller never reverses anything.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace volumetric_kit::recon::codec::detail {

/// Probability precision: every table's frequencies sum to `2^12`.
inline constexpr std::uint32_t kRansScaleBits = 12;
/// `M`, the total every non-empty table's frequencies sum to.
inline constexpr std::uint32_t kRansScale = 1u << kRansScaleBits;
/// `L`, the state's lower bound; also the state the encoder starts in, which
/// is what a correct decode must end in.
inline constexpr std::uint32_t kRansLower = 1u << 16;
/// The widest raw field one coder step carries (@ref RansWriter::put_bits
/// splits wider ones).
inline constexpr std::uint32_t kRansMaxRawBits = kRansScaleBits;
/// The widest raw field @ref RansWriter::put_bits and
/// @ref RansReader::get_bits take: one `uint32_t`.
inline constexpr std::uint32_t kRansMaxFieldBits = 32;
/// The encoder's renormalization shift: a word leaves the state before a
/// symbol of frequency `f` whenever `x >= f << 20`, which is `(x >> 20) >= f`.
/// 20 is 32 minus the probability bits, since the state tops out at 2^32.
inline constexpr std::uint32_t kRansRenormShift = 32 - kRansScaleBits;

/// @brief A normalized frequency table over an alphabet of
///        `freq.size()` symbols.
///
/// Either **empty** (every frequency zero: the model was never used, and a
/// decoder that reaches for it has been handed a corrupt stream) or summing to
/// exactly @ref kRansScale with every used symbol at least 1.
struct FrequencyTable {
  std::vector<std::uint16_t> freq;  ///< Per symbol.
  std::vector<std::uint16_t> cum;   ///< `cum[s]` = sum of `freq[0..s)`; size
                                    ///< alphabet + 1.
  /// Decode lookup: the symbol whose `[cum, cum + freq)` holds each slot of
  /// `[0, M)`. Built by @ref build_decode; empty until then, and always for
  /// an empty table. `uint8_t`, since no codec alphabet exceeds 256.
  std::vector<std::uint8_t> slot_symbol;

  /// @return `true` if no symbol has a frequency (the model was not used).
  bool empty() const noexcept { return cum.empty() || cum.back() == 0; }

  /// Recompute @ref cum from @ref freq (after building or parsing one).
  void finalize() {
    cum.assign(freq.size() + 1, 0);
    for (std::size_t s = 0; s < freq.size(); ++s) {
      cum[s + 1] = static_cast<std::uint16_t>(cum[s] + freq[s]);
    }
  }

  /// Fill @ref slot_symbol, which a decoder needs and an encoder does not.
  /// A no-op on an empty table.
  void build_decode() {
    slot_symbol.clear();
    if (empty()) {
      return;
    }
    slot_symbol.resize(kRansScale);
    for (std::size_t s = 0; s < freq.size(); ++s) {
      std::fill(slot_symbol.begin() + cum[s], slot_symbol.begin() + cum[s + 1],
                static_cast<std::uint8_t>(s));
    }
  }
};

/// @brief Normalize symbol counts to a @ref FrequencyTable.
///
/// Integer-only and deterministic, so the GPU path can hand its histograms to
/// this same function and get the same tables. Every counted symbol gets at
/// least 1 (it must stay encodable), the rest is proportional, and the
/// rounding residue goes to (or comes from) the most frequent symbols, lowest
/// index first on a tie.
/// @param counts  One count per symbol of the alphabet (at most 256 symbols,
///                at most @ref kRansScale of them counted).
/// @return The table; empty when every count is zero.
inline FrequencyTable normalize_counts(
    const std::vector<std::uint64_t>& counts) {
  FrequencyTable t;
  t.freq.assign(counts.size(), 0);
  std::uint64_t total = 0;
  for (std::uint64_t c : counts) {
    total += c;
  }
  if (total == 0) {
    t.finalize();
    return t;
  }
  std::uint32_t sum = 0;
  std::size_t top = 0;  // the most frequent symbol, lowest index on a tie
  for (std::size_t s = 0; s < counts.size(); ++s) {
    if (counts[s] == 0) {
      continue;
    }
    // counts[s] * M fits: a frame holds far fewer than 2^52 symbols per model.
    const std::uint64_t scaled = counts[s] * kRansScale / total;
    t.freq[s] = static_cast<std::uint16_t>(std::max<std::uint64_t>(1, scaled));
    sum += t.freq[s];
    if (counts[s] > counts[top]) {
      top = s;
    }
  }
  // Floor rounding leaves sum <= M, except where the at-least-1 bumps push it
  // over. Give a shortfall to the most frequent symbol, and take an excess
  // from whichever symbol has the most to spare, one unit at a time.
  if (sum < kRansScale) {
    t.freq[top] = static_cast<std::uint16_t>(t.freq[top] + (kRansScale - sum));
  }
  while (sum > kRansScale) {
    std::size_t widest = 0;
    for (std::size_t s = 1; s < t.freq.size(); ++s) {
      if (t.freq[s] > t.freq[widest]) {
        widest = s;
      }
    }
    --t.freq[widest];
    --sum;
  }
  t.finalize();
  return t;
}

/// @brief Collects symbols in decode order and encodes them, backwards, on
///        @ref finish.
///
/// A symbol its table cannot encode, or a raw field wider than
/// @ref kRansMaxFieldBits, is recorded as a coder step of frequency 0, which
/// @ref finish refuses at the division it would otherwise make. In band
/// rather than a flag: a failure flag stored in @ref put made a room0-sized
/// frame's write 47.5 ms against this form's 42.8 (Apple M5 Max, Release).
class RansWriter {
 public:
  /// Append one symbol of @p table. One the table cannot encode -- past its
  /// alphabet, or of frequency 0, so the table was built from counts that
  /// did not include it -- makes @ref finish fail.
  void put(const FrequencyTable& table, std::uint32_t symbol) {
    ops_.push_back(symbol < table.freq.size()
                       ? Op{table.cum[symbol], table.freq[symbol]}
                       : Op{0, 0});
  }

  /// Append the low @p bits bits of @p value as uniform raw bits, low chunk
  /// first, in chunks of at most @ref kRansMaxRawBits. More than
  /// @ref kRansMaxFieldBits makes @ref finish fail.
  void put_bits(std::uint32_t value, std::uint32_t bits) {
    if (bits > kRansMaxFieldBits) {
      ops_.push_back(Op{0, 0});
      return;
    }
    while (bits > 0) {
      const std::uint32_t n = std::min(bits, kRansMaxRawBits);
      const std::uint32_t chunk = value & ((1u << n) - 1u);
      const std::uint32_t shift = kRansScaleBits - n;
      ops_.push_back(Op{static_cast<std::uint16_t>(chunk << shift),
                        static_cast<std::uint16_t>(1u << shift)});
      value >>= n;  // n <= 12, so never a full-width shift
      bits -= n;
    }
  }

  /// @return How many coder steps have been appended.
  std::size_t size() const noexcept { return ops_.size(); }

  /// @brief Encode everything appended onto the end of @p out.
  ///
  /// The stream is the final state as two 16-bit words (high first), then the
  /// renormalization words in the order the decoder reads them; every word is
  /// little-endian. Appends, so a frame's segments are written straight into
  /// the frame. Leaves the writer empty either way.
  /// @return `true` having appended the stream -- always at least 4 bytes,
  ///         always even -- or `false`, appending nothing, if anything put
  ///         could not be encoded (see the class comment).
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
      x = ((x / op.freq) << kRansScaleBits) + (x % op.freq) + op.start;
    }
    ops_.clear();
    // resize, not reserve: reserve takes exactly what it is asked for, so
    // appending a frame's segments one reserve at a time would copy the frame
    // once per segment, where resize grows geometrically.
    std::size_t at = out.size();
    out.resize(at + 4 + 2 * words_.size());
    auto put_word = [&out, &at](std::uint16_t w) {
      out[at++] = static_cast<std::uint8_t>(w & 0xFFu);
      out[at++] = static_cast<std::uint8_t>(w >> 8);
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

/// @brief Decodes one stream made by @ref RansWriter::finish.
///
/// Never reads outside `[data, data + size)`. A failure -- a stream too short
/// to hold a state, a state out of range, a read past the end, a symbol from
/// an empty table -- is **sticky**: it sets @ref failed, every later call
/// returns 0, and @ref finish reports it. So a caller decodes a whole segment
/// and checks once, and garbage input costs a wasted loop, never a crash.
class RansReader {
 public:
  /// @param data  The stream (need not be aligned).
  /// @param size  Its length in bytes.
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

  /// @return The next symbol of @p table (built with
  ///         @ref FrequencyTable::build_decode), or 0 once failed.
  std::uint32_t get(const FrequencyTable& table) {
    if (failed_ || table.slot_symbol.empty()) {
      failed_ = true;
      return 0;
    }
    const std::uint32_t slot = x_ & (kRansScale - 1u);
    const std::uint32_t s = table.slot_symbol[slot];
    x_ = table.freq[s] * (x_ >> kRansScaleBits) + slot - table.cum[s];
    renormalize();
    return failed_ ? 0 : s;
  }

  /// @return The next @p bits raw bits, low chunk first, or 0 once failed.
  ///         More than @ref kRansMaxFieldBits fails the reader, since no
  ///         writer can have put them.
  std::uint32_t get_bits(std::uint32_t bits) {
    if (bits > kRansMaxFieldBits) {
      failed_ = true;
      return 0;
    }
    std::uint32_t value = 0;
    std::uint32_t shift_out = 0;
    while (bits > 0 && !failed_) {
      const std::uint32_t n = std::min(bits, kRansMaxRawBits);
      const std::uint32_t shift = kRansScaleBits - n;
      const std::uint32_t slot = x_ & (kRansScale - 1u);
      const std::uint32_t chunk = slot >> shift;
      // freq = 2^shift, start = chunk << shift.
      x_ = (x_ >> kRansScaleBits << shift) + slot - (chunk << shift);
      renormalize();
      value |= chunk << shift_out;
      shift_out += n;
      bits -= n;
    }
    return failed_ ? 0 : value;
  }

  /// @return `true` once anything has gone wrong (see the class comment).
  bool failed() const noexcept { return failed_; }

  /// @return `true` when the stream decoded cleanly and completely: no
  ///         failure, every word consumed, and the state back at
  ///         @ref kRansLower, where the encoder started.
  ///
  /// A **consistency** check, not an integrity check. It catches a truncated
  /// stream always, and a stream read with the wrong tables or the wrong
  /// symbol count, or with a flipped bit in a table symbol, nearly always.
  /// It cannot catch a flipped raw bit: those bits leave the state as they
  /// are decoded, and every value of them is valid. Nor can it catch a flip
  /// that swaps a symbol for another of equal frequency, after which the
  /// decoder resynchronizes. `codec_rans_test` measures both.
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

}  // namespace volumetric_kit::recon::codec::detail
