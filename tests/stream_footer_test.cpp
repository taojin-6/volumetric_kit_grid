// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// Host-only tests of opening a finished .vrst file from its footer index,
// through the public API: the counts and every key lookup equal the scan's of
// the same file with its TAIL cut off, over more than one block of key
// entries; each footer check refuses a footer forged to fail it alone --
// either CRC, a CRC claimed for the TAIL's empty body, B, T, the offset chain
// from the TAIL to the INDX and on to the TAIL, the key entries' order,
// alignment, range and tracks, a track's key pts, the summaries' key counts
// and K's bound of the file's size / 56 -- and the file is then scanned, its
// footer damaged; and, on a filesystem with sparse files (77, skipped,
// without), records past 4 GiB beyond a 5 GiB unknown chunk read on both
// paths, and an index claiming 2^38 keys is refused with nothing sized by
// them allocated.

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "stream_files.hpp"
#include "test_allocation_failure.hpp"
#include "test_check.hpp"
#include "volumetric_kit/recon/stream/stream_reader.hpp"
#include "volumetric_kit/recon/stream/stream_writer.hpp"

namespace vkc = volumetric_kit::core;
namespace stream = volumetric_kit::recon::stream;
namespace vrst = vr_test::vrst;

namespace {

using vrst::Bytes;
using vrst::info;
using vrst::open;
using vrst::pattern;

// CTest's skip code (tests/CMakeLists.txt): the core's test::kSkipExitCode,
// whose header needs the Vulkan tier this host-only test does not link.
constexpr int kSkip = 77;

stream::Session session_of(std::size_t count) {
  stream::Session s;
  s.writer = "footer";
  for (std::size_t i = 0; i < count; ++i) {
    stream::Track t;
    t.kind = stream::TrackKind::Geometry;
    t.clock = static_cast<std::uint16_t>(i);
    s.tracks.push_back(t);
  }
  return s;
}

// Write `records` records to a finished file of `track_count` tracks: record i
// on track i % `used` (the tracks past `used` stay empty), the n-th of its
// track a key when n % `key_every` is 0, at pts 1000 + 100 (n / 2) (+ the
// track), so pairs share a pts, seq n, set i, with a payload of (i * 397) %
// `sizes` bytes.
bool write_finished(const std::string& path, std::size_t track_count,
                    std::size_t used, std::size_t records,
                    std::size_t key_every, std::size_t sizes = 1500) {
  vkc::Result<std::unique_ptr<stream::StreamWriter>> w =
      stream::StreamWriter::create(path, session_of(track_count));
  if (!w.ok()) return false;
  for (std::size_t i = 0; i < records; ++i) {
    const auto track = static_cast<std::uint16_t>(i % used);
    const std::uint64_t n = i / used;
    const Bytes p = pattern((i * 397) % sizes, i);
    if (!w.value()
             ->append(info(track, n % key_every == 0,
                           1000 + 100 * (n / 2) + track, n, i),
                      p.data(), p.size())
             .ok()) {
      return false;
    }
  }
  return w.value()->finish().ok();
}

// Whether `x` and `y` are one lookup's answers: the same position, or the
// same refusal.
bool same_answer(const vkc::Result<stream::Position>& x,
                 const vkc::Result<stream::Position>& y) {
  return x.ok() ? y.ok() && x.value().offset == y.value().offset
                : !y.ok() && x.status().domain() == y.status().domain();
}

// Whether `a` and `b` agree on each track's stats and keys, and on the track
// past the last: key_count, key for every n and one past, and key_at at each
// of `pts` and either side of it.
bool same_lookups(const stream::StreamReader& a, const stream::StreamReader& b,
                  const std::vector<std::uint64_t>& pts) {
  const std::size_t count = a.session().tracks.size();
  if (b.session().tracks.size() != count || a.stats().size() != count ||
      b.stats().size() != count) {
    return false;
  }
  for (std::size_t i = 0; i <= count; ++i) {
    const auto t = static_cast<std::uint16_t>(i);
    bool same = (i == count || vrst::same(a.stats()[i], b.stats()[i])) &&
                a.key_count(t) == b.key_count(t);
    for (std::size_t n = 0; same && n <= a.key_count(t); ++n) {
      same = same_answer(a.key(t, n), b.key(t, n));
    }
    for (const std::uint64_t p : pts) {
      for (const std::uint64_t q : {p - 1, p, p + 1}) {
        same = same && same_answer(a.key_at(t, q), b.key_at(t, q));
      }
    }
    if (!same) {
      std::fprintf(stderr, "the readers differ on track %zu\n", i);
      return false;
    }
  }
  return true;
}

// Whether two reads of every record found the same records and payloads.
bool same_records(const vrst::ReadBack& a, const vrst::ReadBack& b) {
  if (a.records.size() != b.records.size() || a.payloads != b.payloads ||
      a.next_errors != b.next_errors || a.read_errors != b.read_errors ||
      a.ended != b.ended) {
    return false;
  }
  for (std::size_t i = 0; i < a.records.size(); ++i) {
    if (!vrst::same(a.records[i].info, b.records[i].info) ||
        a.records[i].offset != b.records[i].offset ||
        a.records[i].size != b.records[i].size) {
      return false;
    }
  }
  return true;
}

// Every record's pts, for key_at; or every `stride`-th record's and the
// last's.
std::vector<std::uint64_t> all_pts(const vrst::ReadBack& back,
                                   std::size_t stride = 1) {
  std::vector<std::uint64_t> pts = {0, ~std::uint64_t{0}};
  for (std::size_t i = 0; i < back.records.size(); i += stride) {
    pts.push_back(back.records[i].info.pts_ns);
  }
  if (!back.records.empty()) pts.push_back(back.records.back().info.pts_ns);
  return pts;
}

// `file` up to `index_at`, where its footer starts, then `parts`.
Bytes refooted(const Bytes& file, std::uint64_t index_at,
               const std::vector<Bytes>& parts) {
  Bytes out(file.begin(), file.begin() + static_cast<std::ptrdiff_t>(index_at));
  for (const Bytes& p : parts) out.insert(out.end(), p.begin(), p.end());
  return out;
}

// --- The footer against the scan. --------------------------------------------

int footer_equals_scan_case(const std::string& dir) {
  struct Shape {
    std::size_t tracks, used, records, key_every, sizes, stride;
  };
  // Three tracks, one empty, a key in five; one track all keys; no records;
  // and 70,000 keys, over two of the 32,768 key entries the reader reads at a
  // time. Those blocks depend on the keys alone, so its payloads are under 8
  // bytes (every padding still), and key_at is asked at every 97th record's
  // pts; key is still asked for every key.
  const Shape shapes[] = {{3, 2, 200, 5, 1500, 1},
                          {1, 1, 100, 1, 1500, 1},
                          {2, 2, 0, 1, 1500, 1},
                          {2, 2, 70000, 1, 8, 97}};
  for (const Shape& shape : shapes) {
    const std::string path = dir + "/same.vrst";
    const std::string cut = dir + "/same_cut.vrst";
    ::unlink(path.c_str());
    CHECK(write_finished(path, shape.tracks, shape.used, shape.records,
                         shape.key_every, shape.sizes));
    const Bytes file = vr_test::read_bytes(path);
    CHECK(vrst::write_bytes(cut, Bytes(file.begin(), file.end() - 40)));
    auto footer = open(path);
    auto scan = open(cut);
    CHECK(footer != nullptr && scan != nullptr);
    const stream::Recovery& f = footer->recovery();
    CHECK(f.finished && !f.footer_damaged && f.end == file.size());
    CHECK(f.torn_bytes == 0 && f.damaged_bytes == 0 && f.damaged_spans == 0);
    const stream::Recovery& s = scan->recovery();
    CHECK(!s.finished && !s.footer_damaged && s.end == file.size() - 40);
    CHECK(s.torn_bytes == 0 && s.damaged_spans == 0);
    const vrst::ReadBack back = vrst::read_all(*footer);
    CHECK(back.records.size() == shape.records);
    CHECK(same_records(back, vrst::read_all(*scan)));
    CHECK(same_lookups(*footer, *scan, all_pts(back, shape.stride)));
  }
  return 0;
}

// --- Footers that must not count. --------------------------------------------

// A finished file of three tracks, the last empty, 40 records alternating
// between the others, every fourth of a track a key: its key entries
// alternate between tracks 0 and 1, five each, pts above 0.
struct Sample {
  Bytes file;
  std::uint64_t begin = 0;     // the first record
  std::uint64_t index_at = 0;  // the INDX
  std::uint64_t tail_at = 0;   // the TAIL
  vrst::Index index;
  std::unique_ptr<stream::StreamReader> reader;  // the file, from its footer
  std::vector<std::uint64_t> pts;                // every record's
  std::vector<Bytes> payloads;                   // every record's
};

bool make_sample(const std::string& dir, Sample* out) {
  const std::string path = dir + "/sample.vrst";
  ::unlink(path.c_str());
  if (!write_finished(path, 3, 2, 40, 4)) return false;
  out->file = vr_test::read_bytes(path);
  out->begin = vrst::chunks(out->file).at(1).offset;
  out->index_at = vrst::index_offset(out->file);
  out->tail_at = out->file.size() - 40;
  out->index = vrst::read_index(out->file);
  out->reader = open(path);
  if (out->reader == nullptr || !out->reader->recovery().finished) {
    return false;
  }
  const vrst::ReadBack back = vrst::read_all(*out->reader);
  out->pts = all_pts(back);
  out->payloads = back.payloads;
  return out->index.entries.size() == 10 && out->index.entries[0].track == 0 &&
         out->index.entries[1].track == 1;
}

// Write `file`, which `sample`'s records begin, and open it: the footer
// refused, and `damaged` unless the scan finds no TAIL where the footer's
// starts; the file scanned to the sample's counts and keys, and every record
// read.
int refused(const std::string& dir, const char* what, const Sample& sample,
            const Bytes& file, bool damaged = true) {
  const std::string path = dir + "/forged.vrst";
  CHECK(vrst::write_bytes(path, file));
  auto r = open(path);
  CHECK(r != nullptr);
  const stream::Recovery& rec = r->recovery();
  if (rec.finished || rec.footer_damaged != damaged) {
    std::fprintf(stderr, "a footer with %s: finished %d, damaged %d\n", what,
                 rec.finished, rec.footer_damaged);
  }
  CHECK(!rec.finished && rec.footer_damaged == damaged);
  CHECK(same_lookups(*r, *sample.reader, sample.pts));
  const vrst::ReadBack back = vrst::read_all(*r);
  CHECK(back.ended && back.payloads == sample.payloads);
  return 0;
}

int forged_footer_case(const std::string& dir) {
  Sample sample;
  CHECK(make_sample(dir, &sample));
  const vrst::Index& good = sample.index;
  const std::uint64_t index_at = sample.index_at;
  const auto with = [&](const vrst::Index& index) {
    return refooted(sample.file, index_at,
                    {vrst::index_chunk(index), vrst::tail_chunk(index_at)});
  };
  // The builder rebuilds the writer's footer.
  CHECK(with(good) == sample.file);

  struct Forged {
    const char* what;
    Bytes file;
  };
  std::vector<Forged> forged;
  // Each CRC: a reserved byte of either header, a summary of the body.
  forged.push_back(
      {"a damaged INDX header", vrst::flipped(sample.file, index_at + 32, 0)});
  forged.push_back({"a damaged INDX body",
                    vrst::flipped(sample.file, index_at + 40 + 16, 0)});
  forged.push_back(
      {"a damaged TAIL", vrst::flipped(sample.file, sample.tail_at + 20, 0)});
  // A TAIL's empty body has CRC 0: one claiming another, its header's CRC
  // passing.
  forged.push_back(
      {"a TAIL claiming a body CRC",
       refooted(sample.file, index_at,
                {vrst::index_chunk(good),
                 vrst::header("TAIL", vrst::tail_fields(index_at), 0, 1)})});
  // B and the header's counts. B 8 over 40 T + 32 K, its CRC that of those
  // bytes alone: the size check alone refuses it.
  const Bytes whole = vrst::index_chunk(good);
  const Bytes body(whole.begin() + 40, whole.end());  // no padding
  Bytes longer =
      vrst::header("INDX", Bytes(whole.begin() + 20, whole.begin() + 36),
                   body.size() + 8, vrst::crc32(body));
  longer.insert(longer.end(), body.begin(), body.end());
  longer.resize(longer.size() + 8, 0);
  forged.push_back(
      {"B 8 over 40 T + 32 K",
       refooted(sample.file, index_at, {longer, vrst::tail_chunk(index_at)})});
  vrst::Index index = good;
  ++index.keys;
  forged.push_back({"K one over its entries", with(index)});
  index = good;
  index.tracks = 2;  // the body still sized for 3
  forged.push_back({"T not the session's", with(index)});
  // The key entries.
  index = good;
  std::swap(index.entries[0], index.entries[1]);
  forged.push_back({"key entries out of file order", with(index)});
  index = good;
  index.entries[1].offset = index.entries[0].offset;
  forged.push_back({"two key entries at one offset", with(index)});
  index = good;
  index.entries[1].offset += 4;
  forged.push_back({"a key entry off the 8-byte grid", with(index)});
  index = good;
  index.entries[0].offset = sample.begin - 8;
  forged.push_back({"a key entry in the session header", with(index)});
  index = good;
  index.entries.back().offset = index_at;
  forged.push_back({"a key entry at the index", with(index)});
  index = good;
  index.entries[0].track = 3;
  forged.push_back({"a key entry of no track", with(index)});
  index = good;
  index.entries[2].pts = index.entries[0].pts - 1;  // track 0's second key
  forged.push_back({"a track's key pts decreasing", with(index)});
  // The summaries' key counts.
  index = good;
  ++index.summaries[0].keys;
  --index.summaries[1].keys;
  forged.push_back({"key counts not the entries'", with(index)});
  index = good;
  ++index.summaries[2].keys;
  forged.push_back({"key counts summing above K", with(index)});
  index = good;
  --index.summaries[1].keys;
  forged.push_back({"key counts summing below K", with(index)});

  for (const Forged& f : forged) {
    if (refused(dir, f.what, sample, f.file) != 0) return 1;
  }
  return 0;
}

// The offset chain, the one check that the INDX and TAIL are a footer: the
// TAIL names the INDX, which ends where the TAIL starts.
int offset_chain_case(const std::string& dir) {
  Sample sample;
  CHECK(make_sample(dir, &sample));
  const std::uint64_t index_at = sample.index_at;
  const Bytes index = vrst::index_chunk(sample.index);
  const std::uint64_t size = sample.file.size();
  struct Named {
    const char* what;
    std::uint64_t offset;
  };
  const Named names[] = {
      {"the INDX's own fields", index_at + 8},
      {"the last record's end", index_at - 8},
      {"an offset off the 8-byte grid", index_at + 4},
      {"the session header", 0},
      {"the first record", sample.begin},
      {"the TAIL itself", sample.tail_at},
      {"the file's end", size},
      {"2^64 - 8", ~std::uint64_t{0} - 7},
  };
  for (const Named& n : names) {
    const Bytes file =
        refooted(sample.file, index_at, {index, vrst::tail_chunk(n.offset)});
    const std::string what = std::string("a TAIL naming ") + n.what;
    if (refused(dir, what.c_str(), sample, file) != 0) return 1;
  }
  // The TAIL names the INDX, which ends before the TAIL: an unknown chunk or
  // damage between them, or a second TAIL after a whole footer naming its
  // INDX.
  const Bytes tail = vrst::tail_chunk(index_at);
  const Bytes unknown = vrst::chunk("ZZZZ", {}, pattern(8, 1));
  Bytes again = sample.file;
  again.insert(again.end(), unknown.begin(), unknown.end());
  again.insert(again.end(), tail.begin(), tail.end());
  const std::pair<const char*, Bytes> between[] = {
      {"an unknown chunk before its TAIL",
       refooted(sample.file, index_at, {index, unknown, tail})},
      {"damage before its TAIL",
       refooted(sample.file, index_at, {index, Bytes(16, 0xEE), tail})},
      {"a TAIL after it naming the INDX before", again},
  };
  for (const auto& [what, file] : between) {
    if (refused(dir, what, sample, file) != 0) return 1;
  }
  // A TAIL claiming a body, which the file then lacks.
  const Bytes bodied = vrst::header("TAIL", vrst::tail_fields(index_at), 8, 0);
  if (refused(dir, "a TAIL claiming a body", sample,
              refooted(sample.file, index_at, {index, bodied})) != 0) {
    return 1;
  }
  // No footer the scan can find: an INDX 4 bytes off the 8-byte grid, which
  // ends where its TAIL starts; and a chunk of another tag where the TAIL
  // goes, with its fields.
  const Bytes not_tail = vrst::chunk("ZZZZ", vrst::tail_fields(index_at), {});
  const std::pair<const char*, Bytes> hidden[] = {
      {"an INDX off the 8-byte grid",
       refooted(sample.file, index_at,
                {Bytes(4, 0), index, vrst::tail_chunk(index_at + 4)})},
      {"another tag in the TAIL's place",
       refooted(sample.file, index_at, {index, not_tail})},
  };
  for (const auto& [what, file] : hidden) {
    if (refused(dir, what, sample, file, false) != 0) return 1;
  }
  return 0;
}

// --- K's bound. --------------------------------------------------------------

// A key record takes at least its 56-byte header, so K is at most the file's
// size / 56. Built by hand: one track, an unknown chunk of 512 bytes, and an
// index of K entries 8 bytes apart in it, every other check passing. The file
// is 104 + 512 + 80 + 32 K + 40 bytes: K = 30 is within the bound (1696 / 56
// = 30.3), K = 31 above it (1728 / 56 = 30.9).
int key_bound_case(const std::string& dir) {
  const std::string path = dir + "/bound.vrst";
  for (const std::uint64_t keys : {std::uint64_t{30}, std::uint64_t{31}}) {
    Bytes file = vrst::chunk("VRST", vrst::session_fields(0, 1, 0),
                             vrst::session_body(1));
    const std::uint64_t begin = file.size();
    const Bytes unknown = vrst::chunk("ZZZZ", {}, Bytes(488, 0));
    file.insert(file.end(), unknown.begin(), unknown.end());
    const std::uint64_t index_at = file.size();
    vrst::Index index{1, keys, {{keys, keys, 0, 0, 0}}, {}};
    for (std::uint64_t k = 0; k < keys; ++k) {
      index.entries.push_back({begin + 8 * k, k, k, 0});
    }
    for (const Bytes& c :
         {vrst::index_chunk(index), vrst::tail_chunk(index_at)}) {
      file.insert(file.end(), c.begin(), c.end());
    }
    CHECK(file.size() == 736 + 32 * keys);
    CHECK(vrst::write_bytes(path, file));
    auto r = open(path);
    CHECK(r != nullptr);
    const bool within = keys == 30;
    CHECK(r->recovery().finished == within);
    CHECK(r->recovery().footer_damaged == !within);
    CHECK(r->key_count(0) == (within ? keys : 0u));
  }
  return 0;
}

// --- Past 4 GiB, in a sparse file. -------------------------------------------

bool write_at(int fd, std::uint64_t offset, const Bytes& bytes) {
  return ::pwrite(fd, bytes.data(), bytes.size(), static_cast<off_t>(offset)) ==
         static_cast<ssize_t>(bytes.size());
}

// Whether a file in `dir` with a 64 MiB hole takes less than half of it.
bool sparse_files(const std::string& dir) {
  const std::string path = dir + "/probe";
  const int fd = ::open(path.c_str(), O_WRONLY | O_CREAT | O_EXCL, 0644);
  if (fd < 0) return false;
  const bool written = write_at(fd, std::uint64_t{64} << 20, {1});
  struct stat st{};
  const bool sparse = written && ::fstat(fd, &st) == 0 &&
                      static_cast<std::uint64_t>(st.st_blocks) * 512 <
                          (std::uint64_t{32} << 20);
  ::close(fd);
  ::unlink(path.c_str());
  return sparse;
}

// Records past 4 GiB, after a 5 GiB unknown chunk the scan skips by its
// size, read alike from the footer and by the scan.
int past_4_gib_case(const std::string& dir) {
  const std::string path = dir + "/sparse.vrst";
  const Bytes session =
      vrst::chunk("VRST", vrst::session_fields(0, 2, 0), vrst::session_body(2));
  const std::uint64_t begin = session.size();
  const Bytes r0 = vrst::record(0, 1, 10, 0, 0, pattern(100, 0));
  const std::uint64_t hole_at = begin + r0.size();
  constexpr std::uint64_t kHole = std::uint64_t{5} << 30;
  const Bytes hole = vrst::header("ZZZZ", {}, kHole, 0);
  const std::uint64_t r1_at = hole_at + hole.size() + kHole;
  const Bytes r1 = vrst::record(1, 1, 20, 0, 1, pattern(64, 1));
  const Bytes r2 = vrst::record(0, 0, 30, 1, 1, pattern(33, 2));
  const Bytes r3 = vrst::record(0, 1, 40, 2, 2, pattern(1000, 3));
  const std::uint64_t r2_at = r1_at + r1.size();
  const std::uint64_t r3_at = r2_at + r2.size();
  const std::uint64_t index_at = r3_at + r3.size();
  const vrst::Index index{
      2,
      3,
      {{3, 2, 100 + 33 + 1000, 10, 40}, {1, 1, 64, 20, 20}},
      {{begin, 10, 0, 0}, {r1_at, 20, 1, 1}, {r3_at, 40, 2, 0}}};
  Bytes after;
  for (const Bytes& c :
       {r1, r2, r3, vrst::index_chunk(index), vrst::tail_chunk(index_at)}) {
    after.insert(after.end(), c.begin(), c.end());
  }
  const int fd = ::open(path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
  CHECK(fd >= 0);
  const bool written = write_at(fd, 0, session) && write_at(fd, begin, r0) &&
                       write_at(fd, hole_at, hole) &&
                       write_at(fd, r1_at, after);
  ::close(fd);
  CHECK(written);
  const std::uint64_t size = r1_at + after.size();
  CHECK(r3_at > (std::uint64_t{4} << 30));

  const std::vector<Bytes> payloads = {pattern(100, 0), pattern(64, 1),
                                       pattern(33, 2), pattern(1000, 3)};
  const std::uint64_t offsets[] = {begin, r1_at, r2_at, r3_at};
  std::unique_ptr<stream::StreamReader> readers[2];
  for (const bool footer : {true, false}) {
    if (!footer) CHECK(vrst::cut(path, size - 40));
    auto r = open(path);
    CHECK(r != nullptr);
    const stream::Recovery& rec = r->recovery();
    CHECK(rec.finished == footer && !rec.footer_damaged);
    CHECK(rec.end == (footer ? size : size - 40) && rec.torn_bytes == 0 &&
          rec.damaged_spans == 0);
    for (std::size_t t = 0; t < 2; ++t) {
      CHECK(vrst::same(r->stats()[t], index.summaries[t]));
    }
    CHECK(r->key(0, 1).value().offset == r3_at);
    CHECK(r->key(1, 0).value().offset == r1_at);
    CHECK(r->key_at(0, 39).value().offset == begin);
    CHECK(r->key_at(0, 40).value().offset == r3_at);
    const vrst::ReadBack back = vrst::read_all(*r);
    CHECK(back.ended && back.next_errors.empty() && back.read_errors.empty());
    CHECK(back.payloads == payloads);
    for (std::size_t i = 0; i < 4; ++i) {
      CHECK(back.records[i].offset == offsets[i]);
    }
    readers[footer ? 0 : 1] = std::move(r);
  }
  CHECK(same_lookups(*readers[0], *readers[1], {10, 20, 30, 40}));
  return 0;
}

// An index claiming 2^38 keys, an 8 TiB body that is all hole, every other
// header check passing: refused by K's bound, and the file scanned, with no
// allocation of even 64 KiB -- none sized by K, not even a block of its
// entries.
int huge_index_case(const std::string& dir) {
  const std::string path = dir + "/huge_index.vrst";
  const Bytes session =
      vrst::chunk("VRST", vrst::session_fields(0, 1, 0), vrst::session_body(1));
  const std::uint64_t index_at = session.size();
  constexpr std::uint64_t kKeys = std::uint64_t{1} << 38;
  Bytes fields;
  vrst::put(fields, 1, 4);
  vrst::put(fields, kKeys, 8);
  vrst::put(fields, 0, 4);
  const std::uint64_t body = 40 + 32 * kKeys;
  const Bytes header = vrst::header("INDX", fields, body, 0);
  const std::uint64_t tail_at = index_at + header.size() + body;
  const int fd = ::open(path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
  CHECK(fd >= 0);
  const bool written = write_at(fd, 0, session) &&
                       write_at(fd, index_at, header) &&
                       write_at(fd, tail_at, vrst::tail_chunk(index_at));
  ::close(fd);
  if (!written) {
    std::puts("stream footer: no 8 TiB sparse file here; skipped");
    return kSkip;
  }
  CHECK(kKeys > (tail_at + 40) / 56);
  vr_test::allocation_failure.largest = 0;
  vr_test::allocation_failure.measure = true;
  auto r = open(path);
  vr_test::allocation_failure.measure = false;
  CHECK(r != nullptr);
  if (vr_test::allocation_failure.largest >= (std::size_t{64} << 10)) {
    std::fprintf(stderr, "the open allocated %zu bytes at once\n",
                 vr_test::allocation_failure.largest);
    return 1;
  }
  CHECK(!r->recovery().finished && r->recovery().footer_damaged);
  CHECK(r->stats()[0].records == 0 && r->key_count(0) == 0);
  return 0;
}

}  // namespace

int main() {
  vr_test::Scratch scratch("recon_stream_footer_");
  const std::string dir = scratch.path.string();
  if (footer_equals_scan_case(dir) != 0) return 1;
  if (forged_footer_case(dir) != 0) return 1;
  if (offset_chain_case(dir) != 0) return 1;
  if (key_bound_case(dir) != 0) return 1;
  if (!sparse_files(dir)) {
    std::puts("stream footer: no sparse files here; past 4 GiB skipped");
    return kSkip;
  }
  if (past_4_gib_case(dir) != 0) return 1;
  const int huge = huge_index_case(dir);
  if (huge != 0) return huge;
  std::puts("stream footer: OK");
  return 0;
}
