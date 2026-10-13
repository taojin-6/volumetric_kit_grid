// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// Host-only tests of what the .vrst reader recovers, through its public API:
// one bit flipped in a payload, a record header, the session header and the
// footer; a file cut at every byte of its last record and its footer, and at
// sampled bytes before; damaged spans, one longer than a resync's block;
// footers forged not to count, and "TAIL" in a payload where a footer's would
// start; a writer killed by SIGKILL; and a writer stopped by RLIMIT_FSIZE,
// whose failure sticks.

#include <signal.h>
#include <sys/resource.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

#include "stream_files.hpp"
#include "test_check.hpp"
#include "volumetric_kit/recon/stream/stream_reader.hpp"
#include "volumetric_kit/recon/stream/stream_writer.hpp"

namespace vkc = volumetric_kit::core;
namespace stream = volumetric_kit::recon::stream;
namespace vrst = vr_test::vrst;

namespace {

using Code = vkc::Status::Code;
using vrst::Bytes;
using vrst::info;
using vrst::open;
using vrst::pattern;

stream::Session two_tracks() {
  stream::Session s;
  s.writer = "recovery";
  for (std::uint16_t clock : {std::uint16_t{1}, stream::kHostClock}) {
    stream::Track t;
    t.kind = stream::TrackKind::Geometry;
    t.clock = clock;
    s.tracks.push_back(t);
  }
  return s;
}

// Write `sizes.size()` records alternating between two tracks, every third a
// key, finished or not; their payloads, in order.
std::vector<Bytes> write_sample(const std::string& path,
                                const std::vector<std::size_t>& sizes,
                                bool finish) {
  vkc::Result<std::unique_ptr<stream::StreamWriter>> w =
      stream::StreamWriter::create(path, two_tracks());
  if (!w.ok()) return {};
  std::vector<Bytes> payloads;
  for (std::size_t i = 0; i < sizes.size(); ++i) {
    payloads.push_back(pattern(sizes[i], i));
    const Bytes& p = payloads.back();
    if (!w.value()
             ->append(
                 info(static_cast<std::uint16_t>(i % 2), i % 3 == 0, i * 10, i),
                 p.data(), p.size())
             .ok()) {
      return {};
    }
  }
  if (finish && !w.value()->finish().ok()) return {};
  return payloads;
}

// The payloads but the ones at `skip`.
std::vector<Bytes> without(const std::vector<Bytes>& all,
                           const std::vector<std::size_t>& skip) {
  std::vector<Bytes> out;
  for (std::size_t i = 0; i < all.size(); ++i) {
    if (std::find(skip.begin(), skip.end(), i) == skip.end()) {
      out.push_back(all[i]);
    }
  }
  return out;
}

std::vector<vrst::ChunkAt> of_tag(const std::vector<vrst::ChunkAt>& chunks,
                                  const std::string& tag) {
  std::vector<vrst::ChunkAt> out;
  for (const vrst::ChunkAt& c : chunks) {
    if (c.tag == tag) out.push_back(c);
  }
  return out;
}

// --- One bit flipped. --------------------------------------------------------

int crc_mismatch_case(const std::string& dir) {
  const std::string path = dir + "/flip.vrst";
  const std::vector<Bytes> payloads =
      write_sample(path, {40, 0, 300, 17, 64, 9}, true);
  CHECK(payloads.size() == 6);
  const Bytes original = vr_test::read_bytes(path);
  const std::vector<vrst::ChunkAt> chunks = vrst::chunks(original);
  const std::vector<vrst::ChunkAt> records = of_tag(chunks, "RECD");
  const vrst::ChunkAt index = of_tag(chunks, "INDX").at(0);
  const vrst::ChunkAt tail = of_tag(chunks, "TAIL").at(0);
  const vrst::ChunkAt& third = records.at(2);

  // A payload: its read refuses it, naming the offset; the others read.
  CHECK(vrst::write_bytes(path, vrst::flipped(original, third.offset + 57, 4)));
  auto r = open(path);
  CHECK(r != nullptr);
  CHECK(r->recovery().finished && r->recovery().damaged_spans == 0);
  vrst::ReadBack back = vrst::read_all(*r);
  CHECK(back.records.size() == 6 && back.next_errors.empty());
  CHECK(back.read_errors == std::vector<Code>{Code::InvalidArgument});
  CHECK(back.payloads == without(payloads, {2}));
  Bytes payload;
  const vkc::Status refused = r->read(back.records[2], &payload);
  CHECK(refused.message().find(std::to_string(third.offset)) !=
        std::string::npos);

  // A record header: that record is a damaged span, the rest read.
  CHECK(vrst::write_bytes(path, vrst::flipped(original, third.offset + 26, 0)));
  r = open(path);
  CHECK(r != nullptr);
  CHECK(r->recovery().finished);  // the footer is intact
  CHECK(r->recovery().damaged_spans == 1 &&
        r->recovery().damaged_bytes == third.next() - third.offset);
  CHECK(r->stats()[0].records == 2 && r->stats()[1].records == 3);
  back = vrst::read_all(*r);
  CHECK(back.next_errors == std::vector<Code>{Code::InvalidArgument});
  CHECK(back.ended && back.payloads == without(payloads, {2}));

  // The last record's header: its span ends at the index, and the footer
  // still closes the file.
  const vrst::ChunkAt& last = records.back();
  CHECK(vrst::write_bytes(path, vrst::flipped(original, last.offset + 26, 0)));
  r = open(path);
  CHECK(r != nullptr);
  CHECK(r->recovery().finished && !r->recovery().footer_damaged);
  CHECK(r->recovery().damaged_spans == 1 &&
        r->recovery().damaged_bytes == index.offset - last.offset);
  back = vrst::read_all(*r);
  CHECK(back.ended && back.payloads == without(payloads, {5}));

  // The session header, its fields or its body: nothing to read.
  for (std::uint64_t at : {std::uint64_t{30}, std::uint64_t{60}}) {
    CHECK(vrst::write_bytes(path, vrst::flipped(original, at, 1)));
    CHECK(stream::StreamReader::open(path).status().domain() ==
          Code::InvalidArgument);
  }

  // The footer's index body, its header, and the tail: not finished, the
  // footer damaged, every record read. A damaged index header is a span,
  // which reading reports as it would a record's; a damaged tail ends the
  // file, torn.
  const std::uint64_t flips[] = {index.offset + 40 + 3, index.offset + 24,
                                 tail.offset + 24};
  for (std::uint64_t at : flips) {
    CHECK(vrst::write_bytes(path, vrst::flipped(original, at, 2)));
    r = open(path);
    CHECK(r != nullptr);
    const stream::Recovery& rec = r->recovery();
    CHECK(!rec.finished && rec.footer_damaged);
    const bool header = at == flips[1];
    CHECK(rec.damaged_spans == (header ? 1u : 0u));
    CHECK(rec.torn_bytes == (at == flips[2] ? 40u : 0u));
    back = vrst::read_all(*r);
    CHECK(back.next_errors.size() == (header ? 1u : 0u));
    CHECK(back.ended && back.payloads == payloads);
  }
  return 0;
}

// --- Truncation. -------------------------------------------------------------

int truncation_case(const std::string& dir) {
  const std::string path = dir + "/cut.vrst";
  const std::vector<Bytes> payloads =
      write_sample(path, {100, 33, 0, 250, 8, 7, 301}, true);
  CHECK(payloads.size() == 7);
  const Bytes original = vr_test::read_bytes(path);
  const std::vector<vrst::ChunkAt> chunks = vrst::chunks(original);
  const vrst::ChunkAt session = chunks.at(0);
  const std::uint64_t last = of_tag(chunks, "RECD").back().offset;

  // Every byte of the last record and the footer; every 11th before, and
  // every byte of the session header; longest first, so each is a cut of the
  // one before.
  std::vector<std::uint64_t> cuts;
  for (std::uint64_t at = 0; at < original.size(); ++at) {
    if (at >= last || at < session.next() || at % 11 == 0) cuts.push_back(at);
  }
  std::reverse(cuts.begin(), cuts.end());
  CHECK(vrst::write_bytes(path, original));
  for (std::uint64_t size : cuts) {
    CHECK(vrst::cut(path, size));
    if (size < session.end()) {
      CHECK(stream::StreamReader::open(path).status().domain() ==
            Code::InvalidArgument);
      continue;
    }
    // The chunks whole within `size` read; the first that is not is torn.
    std::size_t whole = 0;
    std::uint64_t end = size;
    for (const vrst::ChunkAt& c : chunks) {
      if (c.tag == "VRST") continue;
      if (c.offset >= size) break;
      if (c.end() > size) {
        end = c.offset;
        break;
      }
      if (c.tag == "RECD") ++whole;
    }
    auto r = open(path);
    CHECK(r != nullptr);
    const stream::Recovery& rec = r->recovery();
    if (rec.end != end || rec.torn_bytes != size - end) {
      std::fprintf(stderr, "cut at %llu: end %llu, torn %llu\n",
                   static_cast<unsigned long long>(size),
                   static_cast<unsigned long long>(rec.end),
                   static_cast<unsigned long long>(rec.torn_bytes));
    }
    CHECK(rec.end == end && rec.torn_bytes == size - end);
    CHECK(!rec.finished && !rec.footer_damaged && rec.damaged_spans == 0);
    const vrst::ReadBack back = vrst::read_all(*r);
    CHECK(back.ended && back.next_errors.empty() && back.read_errors.empty());
    CHECK(back.payloads ==
          std::vector<Bytes>(
              payloads.begin(),
              payloads.begin() + static_cast<std::ptrdiff_t>(whole)));
  }
  return 0;
}

// --- Damaged spans. ----------------------------------------------------------

int damaged_span_case(const std::string& dir) {
  const std::string path = dir + "/span.vrst";
  const std::vector<Bytes> payloads =
      write_sample(path, {64, 200, 3 << 20, 50, 70}, false);
  CHECK(payloads.size() == 5);
  const Bytes original = vr_test::read_bytes(path);
  const std::vector<vrst::ChunkAt> records =
      of_tag(vrst::chunks(original), "RECD");

  // Garbage over a header, with a RECD tag at an 8-byte step whose CRC fails;
  // and the header of the 3 MiB record, whose search runs through more than
  // a resync block of payload.
  Bytes file = original;
  std::fill_n(file.begin() + static_cast<std::ptrdiff_t>(records[1].offset), 40,
              std::uint8_t{0xAB});
  std::copy_n(
      "RECD", 4,
      file.begin() + static_cast<std::ptrdiff_t>(records[1].offset + 16));
  file = vrst::flipped(file, records[2].offset + 9, 5);
  CHECK(vrst::write_bytes(path, file));
  auto r = open(path);
  CHECK(r != nullptr);
  const stream::Recovery& rec = r->recovery();
  // One span: the search passes the second header, which fails its CRC.
  CHECK(rec.damaged_spans == 1 && rec.torn_bytes == 0);
  CHECK(rec.damaged_bytes == records[3].offset - records[1].offset);
  vrst::ReadBack back = vrst::read_all(*r);
  CHECK(back.next_errors.size() == 1);
  CHECK(back.payloads == without(payloads, {1, 2}));

  // A damaged header with no chunk after it: the rest is torn.
  file = vrst::flipped(original, records[4].offset + 30, 0);
  CHECK(vrst::write_bytes(path, file));
  r = open(path);
  CHECK(r != nullptr);
  CHECK(r->recovery().damaged_spans == 0 &&
        r->recovery().torn_bytes == file.size() - records[4].offset);
  back = vrst::read_all(*r);
  CHECK(back.next_errors.empty() && back.payloads == without(payloads, {4}));
  return 0;
}

// --- Footers. ----------------------------------------------------------------

int footer_case(const std::string& dir) {
  const std::string path = dir + "/footer.vrst";
  // One track and one key record, built by hand; then what each check adds.
  Bytes base =
      vrst::chunk("VRST", vrst::session_fields(0, 1, 0), vrst::session_body(1));
  const std::uint64_t record_at = base.size();
  const Bytes payload = pattern(24, 1);
  const Bytes record = vrst::record(0, 1, 5, 0, 0, payload);
  base.insert(base.end(), record.begin(), record.end());
  const std::uint64_t index_at = base.size();
  const auto opened = [&](const std::vector<Bytes>& parts) {
    Bytes file = base;
    for (const Bytes& p : parts) file.insert(file.end(), p.begin(), p.end());
    return vrst::write_bytes(path, file) ? open(path) : nullptr;
  };

  // An INDX naming `tracks` tracks and `keys` keys, its body sized for
  // `sized_for` tracks and the one key; a TAIL naming `index_offset`.
  const auto index = [&](std::uint32_t tracks, std::uint32_t sized_for,
                         std::uint64_t keys) {
    Bytes fields;
    vrst::put(fields, tracks, 4);
    vrst::put(fields, keys, 8);
    vrst::put(fields, 0, 4);
    Bytes body;
    const std::uint64_t summary[] = {1, 1, payload.size(), 5, 5};
    for (const std::uint64_t v : summary) vrst::put(body, v, 8);
    body.resize(body.size() + 40 * std::size_t{sized_for - 1}, 0);
    const std::uint64_t key[] = {record_at, 5, 0, 0};  // track 0, reserved
    for (const std::uint64_t v : key) vrst::put(body, v, 8);
    return vrst::chunk("INDX", fields, body);
  };
  const auto tail = [](std::uint64_t index_offset) {
    Bytes fields;
    vrst::put(fields, 0, 4);
    vrst::put(fields, index_offset, 8);
    vrst::put(fields, 0, 4);
    return vrst::chunk("TAIL", fields, {});
  };

  // The footer the writer writes: finished.
  const Bytes good = index(1, 1, 1);
  auto r = opened({good, tail(index_at)});
  CHECK(r != nullptr);
  CHECK(r->recovery().finished && !r->recovery().footer_damaged);

  // Footers that must not count: damaged, the record read all the same.
  const Bytes unknown = vrst::chunk("ZZZZ", {}, pattern(8, 2));
  const std::vector<std::vector<Bytes>> forged = {
      {index(2, 1, 1), tail(index_at)},         // T not the session's
      {index(1, 1, 2), tail(index_at)},         // K not the body's
      {good, tail(index_at + 8)},               // the tail names another
      {good, tail(0)},                          // offset
      {good, unknown, tail(index_at)},          // a chunk between
      {good, Bytes(16, 0xEE), tail(index_at)},  // damage between
  };
  for (const std::vector<Bytes>& parts : forged) {
    r = opened(parts);
    CHECK(r != nullptr);
    CHECK(!r->recovery().finished && r->recovery().footer_damaged);
    CHECK(r->recovery().torn_bytes == 0);
    const vrst::ReadBack back = vrst::read_all(*r);
    CHECK(back.ended && back.payloads == std::vector<Bytes>{payload});
  }
  // A record after the footer: the file does not end with it.
  r = opened({good, tail(index_at), record});
  CHECK(r != nullptr);
  CHECK(!r->recovery().finished && !r->recovery().footer_damaged);
  CHECK(vrst::read_all(*r).payloads == std::vector<Bytes>(2, payload));

  // "TAIL" in a payload, where a footer's would start, at the end of an
  // unfinished file, its record whole or cut short: no footer.
  for (const std::size_t size : {std::size_t{64}, std::size_t{200}}) {
    Bytes holder(size, 0);
    std::copy_n("TAIL", 4, holder.begin() + 24);
    const Bytes last = vrst::record(0, 0, 6, 1, 1, holder);
    Bytes file = base;
    file.insert(file.end(), last.begin(), last.begin() + 56 + 64);
    CHECK(std::equal(file.end() - 40, file.end() - 36, "TAIL"));
    CHECK(vrst::write_bytes(path, file));
    r = open(path);
    CHECK(r != nullptr);
    CHECK(!r->recovery().finished && !r->recovery().footer_damaged);
    CHECK(r->recovery().torn_bytes == (size == 64 ? 0u : 56u + 64u));
  }
  return 0;
}

// --- A killed writer. --------------------------------------------------------

int killed_writer_case(const std::string& dir) {
  const std::string path = dir + "/killed.vrst";
  std::vector<std::size_t> sizes;
  for (std::size_t i = 0; i < 25; ++i) sizes.push_back((i * 977) % 5000);
  std::fflush(nullptr);
  const pid_t child = ::fork();
  CHECK(child >= 0);
  if (child == 0) {
    vkc::Result<std::unique_ptr<stream::StreamWriter>> w =
        stream::StreamWriter::create(path, two_tracks());
    if (!w.ok()) ::_exit(2);
    for (std::size_t i = 0; i < sizes.size(); ++i) {
      const Bytes p = pattern(sizes[i], i);
      if (!w.value()
               ->append(info(static_cast<std::uint16_t>(i % 2), i % 3 == 0,
                             i * 10, i),
                        p.data(), p.size())
               .ok()) {
        ::_exit(3);
      }
    }
    ::kill(::getpid(), SIGKILL);
    ::_exit(4);
  }
  int status = 0;
  CHECK(::waitpid(child, &status, 0) == child);
  CHECK(WIFSIGNALED(status) && WTERMSIG(status) == SIGKILL);
  auto r = open(path);
  CHECK(r != nullptr);
  const stream::Recovery& rec = r->recovery();
  CHECK(!rec.finished && !rec.footer_damaged);
  CHECK(rec.torn_bytes == 0 && rec.damaged_spans == 0);
  const vrst::ReadBack back = vrst::read_all(*r);
  CHECK(back.payloads.size() == sizes.size());
  for (std::size_t i = 0; i < sizes.size(); ++i) {
    CHECK(back.payloads[i] == pattern(sizes[i], i));
  }
  return 0;
}

// --- A writer the file size limit stops. -------------------------------------

int file_size_limit_case(const std::string& dir) {
  const std::string path = dir + "/limited.vrst";
  constexpr std::size_t kPayload = 1000;  // a chunk of 1056 bytes
  constexpr std::uint64_t kChunk = 56 + kPayload;
  std::fflush(nullptr);
  const pid_t child = ::fork();
  CHECK(child >= 0);
  if (child == 0) {
    ::signal(SIGXFSZ, SIG_IGN);
    vkc::Result<std::unique_ptr<stream::StreamWriter>> w =
        stream::StreamWriter::create(path, two_tracks());
    if (!w.ok()) ::_exit(100);
    stream::StreamWriter& writer = *w.value();
    const auto limit =
        static_cast<rlim_t>(writer.size() + 3 * kChunk + kChunk / 2);
    const rlimit fsize{limit, limit};
    if (::setrlimit(RLIMIT_FSIZE, &fsize) != 0) ::_exit(101);
    int appended = 0;
    vkc::Status failure;
    for (std::uint64_t i = 0; i < 10; ++i) {
      const Bytes p = pattern(kPayload, i);
      failure = writer.append(info(0, true, i, i), p.data(), p.size());
      if (!failure.ok()) break;
      ++appended;
    }
    if (failure.domain() != Code::IoError) ::_exit(102);
    // It sticks: every later call returns it, and writes nothing.
    const std::uint64_t size = writer.size();
    const Bytes p = pattern(8, 99);
    const auto sticky = [&](const vkc::Status& s) {
      return s.domain() == failure.domain() && s.message() == failure.message();
    };
    if (!sticky(writer.append(info(1, true, 0, 0), p.data(), p.size()))) {
      ::_exit(103);
    }
    if (!sticky(writer.sync())) ::_exit(104);
    if (!sticky(writer.finish())) ::_exit(105);
    if (writer.size() != size) ::_exit(106);
    ::_exit(appended);
  }
  int status = 0;
  CHECK(::waitpid(child, &status, 0) == child);
  CHECK(WIFEXITED(status));
  const int appended = WEXITSTATUS(status);
  if (appended != 3) std::fprintf(stderr, "the child exited %d\n", appended);
  CHECK(appended == 3);
  auto r = open(path);
  CHECK(r != nullptr);
  const stream::Recovery& rec = r->recovery();
  CHECK(!rec.finished && rec.damaged_spans == 0);
  // The failed write may have written part of its record (Linux) or nothing
  // (macOS); either way that part is torn.
  CHECK(rec.end == r->begin().offset + 3 * kChunk);
  CHECK(rec.torn_bytes < kChunk);
  const vrst::ReadBack back = vrst::read_all(*r);
  CHECK(back.payloads.size() == 3);
  for (std::size_t i = 0; i < 3; ++i) {
    CHECK(back.payloads[i] == pattern(kPayload, i));
  }
  return 0;
}

}  // namespace

int main() {
  vr_test::Scratch scratch("recon_stream_recovery_");
  const std::string dir = scratch.path.string();
  if (crc_mismatch_case(dir) != 0) return 1;
  if (truncation_case(dir) != 0) return 1;
  if (damaged_span_case(dir) != 0) return 1;
  if (footer_case(dir) != 0) return 1;
  if (killed_writer_case(dir) != 0) return 1;
  if (file_size_limit_case(dir) != 0) return 1;
  std::puts("stream recovery: OK");
  return 0;
}
