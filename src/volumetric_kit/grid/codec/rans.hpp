// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file rans.hpp
/// @brief The codec's entropy code: static-table rANS with a 32-bit state,
///        16-bit output words and 12-bit probabilities, and the frequency
///        tables a frame carries.
///
/// Internal (under src/, never installed). The kernels code it
/// (`shaders/rans_*.comp`); the tests' CPU coder
/// (`tests/codec_oracle_rans.hpp`) is the oracle that pins their bytes.
/// Everything is integer arithmetic on fixed widths, and nothing needs more
/// than 32-bit arithmetic.
///
/// The scheme is Giesen's word-oriented rANS: the state `x` lives in
/// `[L, L << 16)` with `L = 2^16`, a symbol of frequency `f` at cumulative
/// start `c` (out of `M = 2^12`) encodes as
/// `x' = (x / f) * M + x % f + c`, and one 16-bit word leaves the state first
/// whenever `x >= f << 20` -- one is always enough, in both directions. That
/// bound is tested as `(x >> 20) >= f`, since `f << 20` is 2^32 at a
/// probability of one and wraps to 0 in 32 bits. A symbol with `f == M` costs
/// nothing, which is what makes an all-zero band free rather than cheap. A raw
/// field of up to 32 bits is coded low chunk first, in chunks of at most 12
/// bits, each a symbol of frequency `2^(12 - n)` for its `n` bits.
///
/// A segment's stream is the final state as two 16-bit words (high first),
/// then the renormalization words in the order the decoder reads them, every
/// word little-endian: always even, and at least 4 bytes. The encoder starts
/// in state `L`, so a correct decode consumes every word and ends there.
/// rANS decodes in the reverse of the order it encodes, so a coder takes a
/// segment's symbols in decode order and encodes them last first.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace volumetric_kit::grid::codec::detail {

/// Probability precision: every table's frequencies sum to `2^12`.
inline constexpr std::uint32_t kRansScaleBits = 12;
/// `M`, the total every non-empty table's frequencies sum to.
inline constexpr std::uint32_t kRansScale = 1u << kRansScaleBits;

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

  /// Recompute @ref cum from @ref freq (after building or parsing one).
  void finalize() {
    cum.assign(freq.size() + 1, 0);
    for (std::size_t s = 0; s < freq.size(); ++s) {
      cum[s + 1] = static_cast<std::uint16_t>(cum[s] + freq[s]);
    }
  }
};

/// @brief Normalize symbol counts to a @ref FrequencyTable.
///
/// Integer-only and deterministic, so a frame's tables are the same wherever
/// its counts were made. Every counted symbol gets at least 1 (it must stay
/// encodable), the rest is proportional, and the rounding residue goes to (or
/// comes from) the most frequent symbols, lowest index first on a tie.
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

}  // namespace volumetric_kit::grid::codec::detail
