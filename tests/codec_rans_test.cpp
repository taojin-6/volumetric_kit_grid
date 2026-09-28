// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// Host-only tests for the codec's rANS reference coder (rans.hpp): table
// normalization, round trips over symbols and raw bits, the cost of a
// probability-one symbol, size against the ideal code length, what a
// truncated or bit-flipped stream does, and the puts and gets either side
// refuses. CPU-only, so these always run; the
// fuzz cases are what the sanitizer job turns into out-of-bounds detectors.
//
// Random inputs come from a fixed-seed LCG rather than <random>'s
// distributions, whose output the standard leaves to the implementation: a
// threshold that passes here must pass on every leg.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

#include "rans.hpp"

namespace d = volumetric_kit::recon::codec::detail;

#define CHECK(cond)                                                        \
  do {                                                                     \
    if (!(cond)) {                                                         \
      std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
      return 1;                                                            \
    }                                                                      \
  } while (0)

namespace {

struct Lcg {
  std::uint64_t state;
  std::uint32_t next() {
    state = state * 6364136223846793005ull + 1442695040888963407ull;
    return static_cast<std::uint32_t>(state >> 32);
  }
  std::uint32_t below(std::uint32_t n) { return next() % n; }
};

// The invariants every table must keep: empty, or summing to M with every
// counted symbol at least 1 and every uncounted one 0.
int check_table(const std::vector<std::uint64_t>& counts,
                const d::FrequencyTable& t) {
  CHECK(t.freq.size() == counts.size());
  CHECK(t.cum.size() == counts.size() + 1);
  std::uint64_t total = 0;
  std::uint32_t sum = 0;
  for (std::size_t s = 0; s < counts.size(); ++s) {
    total += counts[s];
    sum += t.freq[s];
    CHECK((counts[s] == 0) == (t.freq[s] == 0));
    CHECK(t.cum[s + 1] == t.cum[s] + t.freq[s]);
  }
  CHECK(t.empty() == (total == 0));
  CHECK(sum == (total == 0 ? 0u : d::kRansScale));
  return 0;
}

int normalize_case() {
  // Nothing counted: an empty table.
  const std::vector<std::uint64_t> none = {0, 0, 0};
  CHECK(check_table(none, d::normalize_counts(none)) == 0);
  CHECK(d::normalize_counts(none).empty());

  // One symbol: probability one.
  const std::vector<std::uint64_t> one = {0, 5, 0};
  const d::FrequencyTable t1 = d::normalize_counts(one);
  CHECK(check_table(one, t1) == 0);
  CHECK(t1.freq[1] == d::kRansScale);

  // Exactly proportional where the counts allow it.
  const std::vector<std::uint64_t> three_to_one = {3000, 1000};
  const d::FrequencyTable t2 = d::normalize_counts(three_to_one);
  CHECK(t2.freq[0] == 3072 && t2.freq[1] == 1024);

  // The shortfall path: floors leave the sum under M, the top symbol takes it.
  std::vector<std::uint64_t> short_sum(256, 1);
  short_sum[0] = 1000;
  const d::FrequencyTable t3 = d::normalize_counts(short_sum);
  CHECK(check_table(short_sum, t3) == 0);

  // The excess path: 255 symbols too rare to round to 1 are bumped to it,
  // pushing the sum past M, and the widest symbol pays the difference.
  std::vector<std::uint64_t> excess(256, 1);
  excess[0] = 1000000;
  const d::FrequencyTable t4 = d::normalize_counts(excess);
  CHECK(check_table(excess, t4) == 0);
  CHECK(t4.freq[0] == d::kRansScale - 255);

  // Deterministic, and the invariants hold over arbitrary tables.
  Lcg rng{7};
  for (int trial = 0; trial < 500; ++trial) {
    std::vector<std::uint64_t> counts(2 + rng.below(255));
    for (std::uint64_t& c : counts) {
      const std::uint32_t pick = rng.below(4);
      c = pick == 0 ? 0 : pick == 1 ? rng.below(3) : rng.below(1u << 20);
    }
    const d::FrequencyTable a = d::normalize_counts(counts);
    const d::FrequencyTable b = d::normalize_counts(counts);
    CHECK(check_table(counts, a) == 0);
    CHECK(a.freq == b.freq);
  }
  return 0;
}

// A mixed sequence of table symbols and raw fields, as a frame writes it.
struct Op {
  int table;            // -1 for raw bits
  std::uint32_t value;  // the symbol, or the raw value
  std::uint32_t bits;   // raw width, 1..32
};

std::vector<d::FrequencyTable> random_tables(Lcg& rng, int count) {
  std::vector<d::FrequencyTable> tables;
  for (int i = 0; i < count; ++i) {
    std::vector<std::uint64_t> counts(2 + rng.below(255));
    for (std::uint64_t& c : counts) {
      c = rng.below(3) == 0 ? 0 : 1 + rng.below(i % 2 == 0 ? 10 : 100000);
    }
    counts[rng.below(static_cast<std::uint32_t>(counts.size()))] += 1;
    d::FrequencyTable t = d::normalize_counts(counts);
    t.build_decode();
    tables.push_back(std::move(t));
  }
  return tables;
}

std::vector<Op> random_ops(Lcg& rng, const std::vector<d::FrequencyTable>& t,
                           std::size_t count) {
  std::vector<Op> ops;
  for (std::size_t i = 0; i < count; ++i) {
    if (rng.below(4) == 0) {
      const std::uint32_t bits = 1 + rng.below(32);
      const std::uint32_t mask = bits == 32 ? ~0u : (1u << bits) - 1u;
      ops.push_back(Op{-1, rng.next() & mask, bits});
    } else {
      const int ti = static_cast<int>(rng.below(std::uint32_t(t.size())));
      std::uint32_t s = 0;
      do {
        s = rng.below(std::uint32_t(t[std::size_t(ti)].freq.size()));
      } while (t[std::size_t(ti)].freq[s] == 0);
      ops.push_back(Op{ti, s, 0});
    }
  }
  return ops;
}

std::vector<std::uint8_t> encode(const std::vector<d::FrequencyTable>& t,
                                 const std::vector<Op>& ops) {
  d::RansWriter w;
  for (const Op& op : ops) {
    if (op.table < 0) {
      w.put_bits(op.value, op.bits);
    } else {
      w.put(t[std::size_t(op.table)], op.value);
    }
  }
  std::vector<std::uint8_t> out;
  const bool ok = w.finish(out);
  return ok ? out : std::vector<std::uint8_t>{};
}

// Decode ops.size() operations; true when every one matched and the stream
// finished cleanly.
bool decodes_to(const std::vector<d::FrequencyTable>& t,
                const std::vector<Op>& ops, const std::uint8_t* data,
                std::size_t size) {
  d::RansReader r(data, size);
  bool same = true;
  for (const Op& op : ops) {
    const std::uint32_t got =
        op.table < 0 ? r.get_bits(op.bits) : r.get(t[std::size_t(op.table)]);
    same = same && got == op.value;
  }
  return same && r.finish();
}

int round_trip_case() {
  Lcg rng{11};
  const std::vector<d::FrequencyTable> tables = random_tables(rng, 8);
  for (std::size_t count : {0u, 1u, 2u, 17u, 1000u, 100000u}) {
    const std::vector<Op> ops = random_ops(rng, tables, count);
    const std::vector<std::uint8_t> stream = encode(tables, ops);
    CHECK(stream.size() >= 4 && stream.size() % 2 == 0);
    CHECK(decodes_to(tables, ops, stream.data(), stream.size()));
    // The same input always makes the same bytes.
    CHECK(encode(tables, ops) == stream);
  }
  // An empty stream is exactly the initial state.
  const std::vector<std::uint8_t> empty = encode(tables, {});
  CHECK(empty.size() == 4);
  return 0;
}

// A probability-one symbol moves no state, so any number of them costs
// nothing past the 4-byte state -- what makes an all-zero band free.
int zero_cost_case() {
  d::FrequencyTable certain = d::normalize_counts({0, 0, 9});
  certain.build_decode();
  const std::vector<d::FrequencyTable> tables = {certain};
  const std::vector<Op> ops(100000, Op{0, 2, 0});
  const std::vector<std::uint8_t> stream = encode(tables, ops);
  CHECK(stream.size() == 4);
  CHECK(decodes_to(tables, ops, stream.data(), stream.size()));
  return 0;
}

// The stream is within a whisker of the ideal code length under its table:
// -sum log2(f_s / M) over the symbols, plus the raw bits.
int size_case() {
  Lcg rng{23};
  // A skewed 16-symbol source, like a band's magnitude classes.
  std::vector<std::uint64_t> counts(16);
  std::vector<std::uint32_t> symbols;
  for (int i = 0; i < 200000; ++i) {
    std::uint32_t s = 0;
    while (s < 15 && rng.below(3) != 0) {
      ++s;
    }
    symbols.push_back(s);
    ++counts[s];
  }
  d::FrequencyTable t = d::normalize_counts(counts);
  t.build_decode();
  const std::vector<d::FrequencyTable> tables = {t};
  std::vector<Op> ops;
  double ideal_bits = 0.0;
  for (std::uint32_t s : symbols) {
    ops.push_back(Op{0, s, 0});
    ideal_bits -= std::log2(double(t.freq[s]) / d::kRansScale);
    if (s % 5 == 0) {
      ops.push_back(Op{-1, s, 7});
      ideal_bits += 7;
    }
  }
  const std::vector<std::uint8_t> stream = encode(tables, ops);
  const double actual_bits = 8.0 * double(stream.size());
  std::printf("rANS size: %.0f bits against an ideal %.0f (%+.4f%%)\n",
              actual_bits, ideal_bits,
              100.0 * (actual_bits - ideal_bits) / ideal_bits);
  CHECK(actual_bits <= ideal_bits * 1.001 + 64);
  CHECK(decodes_to(tables, ops, stream.data(), stream.size()));
  return 0;
}

int corrupt_case() {
  Lcg rng{31};
  const std::vector<d::FrequencyTable> tables = random_tables(rng, 4);
  const std::vector<Op> ops = random_ops(rng, tables, 5000);
  const std::vector<std::uint8_t> stream = encode(tables, ops);

  // Too short for a state, or not whole words: failed from the start.
  for (std::size_t len : {0u, 1u, 2u, 3u, 5u}) {
    d::RansReader r(stream.data(), len);
    CHECK(r.failed());
    CHECK(!r.finish());
  }
  // Every truncation is caught: a correct decode consumes every word, so a
  // shorter stream must run out somewhere.
  for (std::size_t len = 4; len < stream.size(); len += 2) {
    CHECK(!decodes_to(tables, ops, stream.data(), len));
  }
  // What finish() (the state back at L, every word consumed) does and does
  // not catch, measured rather than assumed. It is a consistency check, not an
  // integrity check:
  //  - a flip in RAW bits is invisible by construction: a raw field's bits
  //    leave the state as they are decoded and every value of them is valid,
  //    so the value changes and nothing downstream does;
  //  - a flip that changes a TABLE symbol is caught almost always, but not
  //    always: when it moves the slot into another symbol of the same
  //    frequency at the same offset, the next state is identical and the
  //    decoder resynchronizes one substituted symbol later. Equal
  //    frequencies are common (every rare symbol is bumped to 1).
  // So integrity is the transport's (the 2026-09-27 decision). And nothing
  // ever reads out of bounds, which is what the sanitizer job checks.
  int caught = 0;
  int raw_only = 0;
  int silent_symbol_change = 0;
  const int flips = 4000;
  for (int i = 0; i < flips; ++i) {
    std::vector<std::uint8_t> bad = stream;
    bad[rng.below(std::uint32_t(bad.size()))] ^=
        static_cast<std::uint8_t>(1u << rng.below(8));
    d::RansReader r(bad.data(), bad.size());
    bool symbols_same = true;
    for (const Op& op : ops) {
      if (op.table < 0) {
        r.get_bits(op.bits);
      } else {
        // Decode first: a short-circuit here would stop consuming the
        // stream at the first mismatch and fail finish() for the wrong
        // reason.
        const std::uint32_t got = r.get(tables[std::size_t(op.table)]);
        symbols_same = symbols_same && got == op.value;
      }
    }
    if (!r.finish()) {
      ++caught;
    } else if (symbols_same) {
      ++raw_only;
    } else {
      ++silent_symbol_change;
    }
  }
  std::printf(
      "rANS bit flips: %d caught, %d in raw bits only, %d silent symbol "
      "changes, of %d\n",
      caught, raw_only, silent_symbol_change, flips);
  CHECK(caught + raw_only + silent_symbol_change == flips);
  // Of the flips that changed a symbol, finish() caught at least 95%.
  CHECK(caught * 100 >= 95 * (caught + silent_symbol_change));

  // Reading fewer symbols than were written is caught by the state alone. One
  // 8-bit raw field fits in the initial state without emitting a word, so the
  // stream is just its 4-byte state: a reader that decodes nothing has
  // consumed every byte, and only the state (2^24, not L) says it stopped
  // early.
  d::RansWriter one;
  one.put_bits(0xA5, 8);
  std::vector<std::uint8_t> short_stream;
  CHECK(one.finish(short_stream));
  CHECK(short_stream.size() == 4);
  {
    d::RansReader early(short_stream.data(), short_stream.size());
    CHECK(!early.failed());
    CHECK(!early.finish());
    d::RansReader full(short_stream.data(), short_stream.size());
    CHECK(full.get_bits(8) == 0xA5);
    CHECK(full.finish());
  }

  // Reaching for a model that was never used is a corrupt stream, not a
  // default symbol.
  const d::FrequencyTable unused = d::normalize_counts({0, 0});
  d::RansReader r(stream.data(), stream.size());
  CHECK(r.get(unused) == 0);
  CHECK(r.failed());
  CHECK(r.get_bits(8) == 0);  // sticky
  CHECK(!r.finish());
  return 0;
}

// What each side refuses rather than encoding or decoding wrong, and that
// finish() appends -- the frame writes every segment straight onto one
// payload.
int refusal_case() {
  d::FrequencyTable t = d::normalize_counts({3, 0, 5});
  t.build_decode();

  // A symbol of frequency 0, and one past the alphabet: finish() fails and
  // appends nothing, whatever surrounds the bad symbol, and the writer is
  // usable again after it.
  for (std::uint32_t symbol : {1u, 3u, 999u}) {
    d::RansWriter w;
    w.put(t, 0);
    w.put(t, symbol);
    w.put(t, 2);
    std::vector<std::uint8_t> out = {0xAB};
    CHECK(!w.finish(out));
    CHECK(out.size() == 1);
    CHECK(w.size() == 0);
    w.put(t, 2);
    CHECK(w.finish(out));
    CHECK(out.size() == 1 + 4);
  }
  // A raw field wider than 32 bits, on either side.
  {
    d::RansWriter w;
    w.put_bits(0, d::kRansMaxFieldBits + 1);
    std::vector<std::uint8_t> out;
    CHECK(!w.finish(out) && out.empty());

    // Three whole 12-bit chunks, which a 36-bit read would consume cleanly
    // if nothing refused it -- so only the width check can fail it.
    d::RansWriter ok;
    for (int i = 0; i < 3; ++i) {
      ok.put_bits(0xFFF, d::kRansMaxRawBits);
    }
    std::vector<std::uint8_t> stream;
    CHECK(ok.finish(stream));
    d::RansReader r(stream.data(), stream.size());
    CHECK(r.get_bits(3 * d::kRansMaxRawBits) == 0);
    CHECK(r.failed());
    d::RansReader full(stream.data(), stream.size());
    for (int i = 0; i < 3; ++i) {
      CHECK(full.get_bits(d::kRansMaxRawBits) == 0xFFF);
    }
    CHECK(full.finish());
  }
  // finish() appends after what is already there, the same bytes a fresh
  // vector receives.
  {
    Lcg rng{41};
    const std::vector<d::FrequencyTable> tables = random_tables(rng, 3);
    const std::vector<Op> ops = random_ops(rng, tables, 300);
    const std::vector<std::uint8_t> alone = encode(tables, ops);
    d::RansWriter w;
    for (const Op& op : ops) {
      if (op.table < 0) {
        w.put_bits(op.value, op.bits);
      } else {
        w.put(tables[std::size_t(op.table)], op.value);
      }
    }
    std::vector<std::uint8_t> out = {1, 2, 3};
    CHECK(w.finish(out));
    CHECK(out.size() == 3 + alone.size());
    CHECK(out[0] == 1 && out[1] == 2 && out[2] == 3);
    CHECK(std::vector<std::uint8_t>(out.begin() + 3, out.end()) == alone);
  }
  return 0;
}

}  // namespace

int main() {
  if (normalize_case() != 0) return 1;
  if (round_trip_case() != 0) return 1;
  if (zero_cost_case() != 0) return 1;
  if (size_case() != 0) return 1;
  if (corrupt_case() != 0) return 1;
  if (refusal_case() != 0) return 1;
  std::printf("codec rANS: OK\n");
  return 0;
}
