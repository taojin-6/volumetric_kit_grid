// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// The server's GOP log (src/volumetric_kit/recon/networking/gop_log.hpp), with
// no sockets: before any key nothing is handed out; a viewer joining mid-GOP
// starts at its key; a viewer skips only at a chunk boundary and only to a
// newer key, and the chunk it holds survives the log's replacement; a GOP past
// max_gop_bytes, or one whose record could not be copied, is dropped until the
// next key; and, over seeded random runs of records and viewers, no P-frame is
// ever handed out without every record since its key. Replaying counts no
// join. Keys on demand: a joiner gets nothing until the next key, then every
// record; one that would fall past max_gop_bytes behind gets no P-frame
// without its references, then the next key, while a viewer keeping up gets
// every record; a lost copy sends every viewer back to waiting; each join
// and each fall-back counts in key_wanted; and, over seeded random runs, the
// bytes held stay within max_gop_bytes, none with no viewer in, and each
// viewer's records are its key's run, a gap only to a key after a counted
// fall-back.

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <iterator>
#include <list>
#include <memory>
#include <random>
#include <vector>

#include "gop_log.hpp"
#include "test_check.hpp"

namespace detail = volumetric_kit::recon::networking::detail;

namespace {

using detail::Chunk;
using detail::GopLog;

// A chunk of `size` bytes (at least 9) holding its seq and whether it is a
// key, as a record's chunk holds its header.
Chunk chunk(std::uint64_t seq, bool key, std::size_t size = 16) {
  auto c = std::make_shared<std::vector<std::uint8_t>>(size < 9 ? 9 : size);
  for (int i = 0; i < 8; ++i) {
    (*c)[i] = static_cast<std::uint8_t>(seq >> (8 * i));
  }
  (*c)[8] = key ? 1 : 0;
  return c;
}

std::uint64_t seq_of(const Chunk& c) {
  std::uint64_t seq = 0;
  for (int i = 0; i < 8; ++i) seq |= std::uint64_t{(*c)[i]} << (8 * i);
  return seq;
}
bool key_of(const Chunk& c) { return (*c)[8] != 0; }

// The seqs `log` hands the viewer at `cursor` until it has none.
std::vector<std::uint64_t> drain(const GopLog& log, GopLog::Cursor* cursor) {
  std::vector<std::uint64_t> seqs;
  while (const Chunk c = log.next(cursor)) seqs.push_back(seq_of(c));
  return seqs;
}

using Seqs = std::vector<std::uint64_t>;

// Before any key nothing is kept or handed out; the first key starts it.
int before_any_key_case() {
  GopLog log(1 << 20);
  GopLog::Cursor early;
  CHECK(!log.open());
  CHECK(log.next(&early) == nullptr);
  CHECK(!log.append(chunk(0, false), false));
  CHECK(!log.append(chunk(1, false), false));
  CHECK(log.next(&early) == nullptr);
  CHECK(log.append(chunk(2, true), true));
  CHECK(log.open());
  CHECK(log.append(chunk(3, false), false));
  CHECK(drain(log, &early) == (Seqs{2, 3}));
  CHECK(log.append(chunk(4, false), false));
  CHECK(drain(log, &early) == (Seqs{4}));
  return 0;
}

// A viewer joining mid-GOP starts at its key, whatever its viewers took.
int join_mid_gop_case() {
  GopLog log(1 << 20);
  CHECK(log.append(chunk(0, true), true));
  CHECK(log.append(chunk(1, false), false));
  GopLog::Cursor first;
  CHECK(drain(log, &first) == (Seqs{0, 1}));
  CHECK(log.append(chunk(2, false), false));
  GopLog::Cursor late;
  CHECK(drain(log, &late) == (Seqs{0, 1, 2}));
  CHECK(drain(log, &first) == (Seqs{2}));
  // After a second key, a join starts there.
  CHECK(log.append(chunk(3, true), true));
  CHECK(log.append(chunk(4, false), false));
  GopLog::Cursor later;
  CHECK(drain(log, &later) == (Seqs{3, 4}));
  return 0;
}

// A viewer behind in its GOP gets every chunk of it in order while no newer
// key comes; once one does, its next chunk is that key. The chunk it holds
// stays whole when the log lets the old GOP go.
int newest_key_wins_case() {
  GopLog log(1 << 20);
  CHECK(log.append(chunk(0, true, 4096), true));
  GopLog::Cursor slow;
  GopLog::Cursor fast;
  for (std::uint64_t seq = 1; seq < 10; ++seq) {
    CHECK(log.append(chunk(seq, false, 4096), false));
  }
  CHECK(drain(log, &fast) == (Seqs{0, 1, 2, 3, 4, 5, 6, 7, 8, 9}));
  // The slow one takes three, then holds the fourth under way.
  CHECK(seq_of(log.next(&slow)) == 0);
  CHECK(seq_of(log.next(&slow)) == 1);
  CHECK(seq_of(log.next(&slow)) == 2);
  const Chunk under_way = log.next(&slow);
  CHECK(under_way != nullptr && seq_of(under_way) == 3);
  CHECK(log.append(chunk(10, true, 4096), true));
  CHECK(log.append(chunk(11, false, 4096), false));
  // What it holds is whole, though the log no longer has it.
  CHECK(under_way.use_count() == 1);
  CHECK(under_way->size() == 4096 && seq_of(under_way) == 3);
  // At the boundary, the rest of the old GOP is skipped for the new key.
  CHECK(drain(log, &slow) == (Seqs{10, 11}));
  CHECK(drain(log, &fast) == (Seqs{10, 11}));
  // Two newer keys before it asks again: the newest wins.
  CHECK(log.append(chunk(12, false, 4096), false));
  CHECK(log.append(chunk(13, true, 4096), true));
  CHECK(log.append(chunk(14, true, 4096), true));
  CHECK(log.append(chunk(15, false, 4096), false));
  CHECK(drain(log, &slow) == (Seqs{14, 15}));
  return 0;
}

// A chunk that would take the GOP past max_gop_bytes drops it, and every
// record until the next key; a key past it alone is dropped too.
int overflow_case() {
  GopLog log(3000);
  CHECK(log.append(chunk(0, true, 1000), true));
  CHECK(log.append(chunk(1, false, 1000), false));
  GopLog::Cursor live;
  CHECK(drain(log, &live) == (Seqs{0, 1}));
  // Exactly at the limit is kept.
  CHECK(log.append(chunk(2, false, 1000), false));
  CHECK(drain(log, &live) == (Seqs{2}));
  CHECK(!log.append(chunk(3, false, 1), false));
  CHECK(!log.open());
  GopLog::Cursor joined;
  CHECK(log.next(&joined) == nullptr);
  CHECK(log.next(&live) == nullptr);
  CHECK(!log.append(chunk(4, false, 10), false));
  CHECK(log.next(&live) == nullptr);
  CHECK(log.append(chunk(5, true, 500), true));
  CHECK(log.append(chunk(6, false, 500), false));
  CHECK(drain(log, &live) == (Seqs{5, 6}));
  CHECK(drain(log, &joined) == (Seqs{5, 6}));
  // A viewer behind in the broken GOP gets nothing more of it.
  GopLog::Cursor behind;
  CHECK(seq_of(log.next(&behind)) == 5);
  CHECK(!log.append(chunk(7, false, 2500), false));
  CHECK(log.next(&behind) == nullptr);
  // A key past the limit alone: nothing until the next.
  CHECK(!log.append(chunk(8, true, 3001), true));
  CHECK(!log.append(chunk(9, false, 10), false));
  CHECK(log.next(&live) == nullptr);
  CHECK(log.append(chunk(10, true, 3000), true));
  CHECK(drain(log, &live) == (Seqs{10}));
  CHECK(drain(log, &behind) == (Seqs{10}));
  return 0;
}

// A record that could not be copied breaks the GOP as an overflow does.
int failed_copy_case() {
  GopLog log(1 << 20);
  CHECK(log.append(chunk(0, true), true));
  GopLog::Cursor live;
  CHECK(drain(log, &live) == (Seqs{0}));
  log.break_gop();
  CHECK(!log.open());
  CHECK(!log.append(chunk(2, false), false));
  CHECK(log.next(&live) == nullptr);
  GopLog::Cursor joined;
  CHECK(log.next(&joined) == nullptr);
  CHECK(log.append(chunk(3, true), true));
  CHECK(drain(log, &live) == (Seqs{3}));
  CHECK(drain(log, &joined) == (Seqs{3}));
  return 0;
}

// Seeded runs of records (a key now and then, sizes past a small limit at
// times, a lost copy at times) and viewers that join, pull and fall behind at
// random: a viewer's first chunk is a key, and each P-frame it gets follows
// the chunk it got before, so it has every record since its key.
int never_without_key_case() {
  for (std::uint32_t seed = 1; seed <= 40; ++seed) {
    std::mt19937 rng(seed);
    GopLog log(std::uniform_int_distribution<std::uint64_t>(200, 4000)(rng));
    struct Viewer {
      GopLog::Cursor cursor;
      bool any = false;
      std::uint64_t last = 0;
      std::size_t got = 0;
    };
    std::vector<Viewer> viewers;
    std::uint64_t seq = 0;
    std::size_t keys_seen = 0;
    for (int step = 0; step < 2000; ++step) {
      const int what = std::uniform_int_distribution<int>(0, 99)(rng);
      if (what < 30) {
        const bool key = std::uniform_int_distribution<int>(0, 5)(rng) == 0;
        const std::size_t size =
            std::uniform_int_distribution<std::size_t>(9, 600)(rng);
        keys_seen += key ? 1 : 0;
        (void)log.append(chunk(seq++, key, size), key);
      } else if (what < 32) {
        log.break_gop();
        ++seq;  // the record lost
      } else if (what < 36 && viewers.size() < 6) {
        viewers.emplace_back();
      } else if (!viewers.empty()) {
        Viewer& v = viewers[std::uniform_int_distribution<std::size_t>(
            0, viewers.size() - 1)(rng)];
        const Chunk c = log.next(&v.cursor);
        if (c == nullptr) continue;
        const std::uint64_t s = seq_of(c);
        if (!key_of(c)) {
          if (!v.any || s != v.last + 1) {
            std::fprintf(stderr,
                         "FAIL seed %u: P-frame %llu after %llu (any %d)\n",
                         seed, static_cast<unsigned long long>(s),
                         static_cast<unsigned long long>(v.last), v.any);
            return 1;
          }
        }
        CHECK(!v.any || s > v.last);
        v.any = true;
        v.last = s;
        ++v.got;
      }
    }
    CHECK(keys_seen > 0);
    std::size_t got = 0;
    for (const Viewer& v : viewers) got += v.got;
    CHECK(got > 0);
  }
  return 0;
}

// Replaying, a viewer starts at the GOP's key, so none is counted wanting
// one, and joining and leaving change nothing it is handed.
int replay_counts_no_join_case() {
  GopLog log(1 << 20);
  CHECK(log.append(chunk(0, true), true));
  GopLog::Cursor viewer;
  log.join(&viewer);
  CHECK(log.key_wanted() == 0);
  CHECK(drain(log, &viewer) == (Seqs{0}));
  CHECK(log.append(chunk(1, false), false));
  log.leave(&viewer);
  GopLog::Cursor late;
  CHECK(drain(log, &late) == (Seqs{0, 1}));
  CHECK(log.key_wanted() == 0);
  return 0;
}

// Keys on demand: with no viewer nothing is kept; a joiner, counted, gets
// nothing until the next key, then every record; a second joiner mid-GOP
// waits for the next key, which the first gets in its turn; what every
// viewer took goes at the next record, and what only a leaver needed too.
int on_demand_join_case() {
  GopLog log(1 << 20, false);
  CHECK(!log.append(chunk(0, true), true));
  CHECK(!log.append(chunk(1, false), false));
  CHECK(log.bytes() == 0 && log.key_wanted() == 0);
  GopLog::Cursor first;
  log.join(&first);
  CHECK(log.key_wanted() == 1 && !log.open());
  CHECK(!log.append(chunk(2, false), false));
  CHECK(log.next(&first) == nullptr);
  CHECK(log.append(chunk(3, true), true));
  CHECK(log.open());
  CHECK(log.append(chunk(4, false), false));
  CHECK(drain(log, &first) == (Seqs{3, 4}));
  GopLog::Cursor second;
  log.join(&second);
  CHECK(log.key_wanted() == 2);
  CHECK(log.append(chunk(5, false), false));
  CHECK(log.next(&second) == nullptr);
  CHECK(drain(log, &first) == (Seqs{5}));
  CHECK(log.append(chunk(6, true), true));
  CHECK(log.append(chunk(7, false), false));
  CHECK(drain(log, &second) == (Seqs{6, 7}));
  CHECK(drain(log, &first) == (Seqs{6, 7}));
  CHECK(log.append(chunk(8, false), false));
  CHECK(log.bytes() == 16);
  CHECK(drain(log, &first) == (Seqs{8}));
  log.leave(&second);
  CHECK(log.append(chunk(9, false), false));
  CHECK(log.bytes() == 16);
  CHECK(drain(log, &first) == (Seqs{9}));
  log.leave(&first);
  CHECK(!log.append(chunk(10, false), false));
  CHECK(!log.append(chunk(11, true), true));
  CHECK(log.bytes() == 0 && log.key_wanted() == 2);
  return 0;
}

// Keys on demand, max_gop_bytes 3000: a viewer that stops taking chunks,
// holding its key under way, stays with exactly 3000 bytes behind it and
// falls back, counted, at the record that would leave it more; what only it
// needed goes; the chunk it holds stays whole; it gets no P-frame after it,
// then the next key. The viewer keeping up gets every record and is never
// counted. A record past the limit alone sends every viewer back, and a key
// past it lets none in.
int on_demand_fall_back_case() {
  GopLog log(3000, false);
  GopLog::Cursor slow;
  GopLog::Cursor fast;
  log.join(&slow);
  log.join(&fast);
  CHECK(log.append(chunk(0, true, 1000), true));
  const Chunk under_way = log.next(&slow);
  CHECK(under_way != nullptr && seq_of(under_way) == 0);
  CHECK(drain(log, &fast) == (Seqs{0}));
  const std::size_t sizes[] = {1000, 1000, 991, 9};  // 3000 behind it
  for (std::uint64_t seq = 1; seq <= 4; ++seq) {
    CHECK(log.append(chunk(seq, false, sizes[seq - 1]), false));
    CHECK(drain(log, &fast) == (Seqs{seq}));
  }
  CHECK(log.key_wanted() == 2 && log.bytes() == 3000);
  CHECK(log.append(chunk(5, false, 9), false));
  CHECK(log.key_wanted() == 3 && log.bytes() == 9);
  CHECK(under_way.use_count() == 1 && under_way->size() == 1000);
  CHECK(log.next(&slow) == nullptr);
  CHECK(drain(log, &fast) == (Seqs{5}));
  CHECK(log.append(chunk(6, false, 10), false));
  CHECK(log.next(&slow) == nullptr);
  CHECK(log.append(chunk(7, true, 500), true));
  CHECK(log.append(chunk(8, false, 500), false));
  CHECK(drain(log, &slow) == (Seqs{7, 8}));
  CHECK(drain(log, &fast) == (Seqs{6, 7, 8}));
  CHECK(log.key_wanted() == 3);
  CHECK(!log.append(chunk(9, false, 3001), false));
  CHECK(log.key_wanted() == 5 && log.bytes() == 0);
  CHECK(!log.append(chunk(10, true, 3001), true));
  CHECK(log.key_wanted() == 5);
  CHECK(log.next(&slow) == nullptr && log.next(&fast) == nullptr);
  CHECK(log.append(chunk(11, true, 3000), true));
  CHECK(drain(log, &slow) == (Seqs{11}));
  CHECK(drain(log, &fast) == (Seqs{11}));
  return 0;
}

// Keys on demand, a lost copy: every viewer in falls back, counted, a
// waiting one not again, and the next key lets both in.
int on_demand_lost_copy_case() {
  GopLog log(1 << 20, false);
  GopLog::Cursor in;
  GopLog::Cursor waiting;
  log.join(&in);
  CHECK(log.append(chunk(0, true), true));
  CHECK(drain(log, &in) == (Seqs{0}));
  log.join(&waiting);
  CHECK(log.key_wanted() == 2);
  log.break_gop();
  CHECK(log.key_wanted() == 3 && !log.open() && log.bytes() == 0);
  CHECK(!log.append(chunk(2, false), false));
  CHECK(log.next(&in) == nullptr && log.next(&waiting) == nullptr);
  CHECK(log.append(chunk(3, true), true));
  CHECK(drain(log, &in) == (Seqs{3}));
  CHECK(drain(log, &waiting) == (Seqs{3}));
  CHECK(log.key_wanted() == 3);
  return 0;
}

// Keys on demand, seeded runs of records (a key now and then, sizes past a
// small limit at times, a lost copy at times) and viewers that join, pull,
// fall behind and leave at random: the bytes held never pass the limit and
// are none while no viewer is in; a viewer's first chunk is a key, each
// P-frame follows the chunk it got before, and each jump is to a key after a
// fall-back the count took: the count is at least the joins and the jumps.
int on_demand_random_case() {
  for (std::uint32_t seed = 1; seed <= 40; ++seed) {
    std::mt19937 rng(seed);
    const auto pick = [&rng](std::uint64_t low, std::uint64_t high) {
      return std::uniform_int_distribution<std::uint64_t>(low, high)(rng);
    };
    const std::uint64_t limit = pick(200, 4000);
    GopLog log(limit, false);
    struct Viewer {
      GopLog::Cursor cursor;
      bool any = false;
      std::uint64_t last = 0;
    };
    std::list<Viewer> viewers;  // the log holds their cursors' addresses
    std::uint64_t seq = 0;
    std::uint64_t joins = 0;
    std::uint64_t jumps = 0;
    std::uint64_t got = 0;
    for (int step = 0; step < 4000; ++step) {
      const std::uint64_t what = pick(0, 99);
      if (what < 30) {
        const bool key = pick(0, 9) == 0;
        (void)log.append(chunk(seq++, key, pick(9, 600)), key);
        CHECK(log.bytes() <= limit);
        CHECK(log.open() || log.bytes() == 0);
      } else if (what < 31) {
        log.break_gop();
        ++seq;  // the record lost
        CHECK(!log.open() && log.bytes() == 0);
      } else if (what < 35 && viewers.size() < 6) {
        viewers.emplace_back();
        log.join(&viewers.back().cursor);
        ++joins;
      } else if (what < 37 && !viewers.empty()) {
        auto gone =
            std::next(viewers.begin(),
                      static_cast<std::ptrdiff_t>(pick(0, viewers.size() - 1)));
        log.leave(&gone->cursor);
        viewers.erase(gone);
      } else if (!viewers.empty()) {
        Viewer& v = *std::next(
            viewers.begin(),
            static_cast<std::ptrdiff_t>(pick(0, viewers.size() - 1)));
        const Chunk c = log.next(&v.cursor);
        if (c == nullptr) continue;
        const std::uint64_t s = seq_of(c);
        if (!v.any || s != v.last + 1) {
          if (!key_of(c)) {
            std::fprintf(stderr,
                         "FAIL seed %u: P-frame %llu after %llu (any %d)\n",
                         seed, static_cast<unsigned long long>(s),
                         static_cast<unsigned long long>(v.last), v.any);
            return 1;
          }
          jumps += v.any ? 1 : 0;
        }
        CHECK(!v.any || s > v.last);
        v.any = true;
        v.last = s;
        ++got;
      }
    }
    CHECK(log.key_wanted() >= joins + jumps);
    CHECK(got > 0);
  }
  return 0;
}

}  // namespace

int main() {
  if (before_any_key_case() != 0) return 1;
  if (join_mid_gop_case() != 0) return 1;
  if (newest_key_wins_case() != 0) return 1;
  if (overflow_case() != 0) return 1;
  if (failed_copy_case() != 0) return 1;
  if (never_without_key_case() != 0) return 1;
  if (replay_counts_no_join_case() != 0) return 1;
  if (on_demand_join_case() != 0) return 1;
  if (on_demand_fall_back_case() != 0) return 1;
  if (on_demand_lost_copy_case() != 0) return 1;
  if (on_demand_random_case() != 0) return 1;
  std::puts("networking gop log: OK");
  return 0;
}
