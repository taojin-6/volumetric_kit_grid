// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// Allocation failures across the complete public stream calls, including
// validation, copies of a writer's sticky error, and a parser's failure,
// which sticks too; and the reads into caller memory, which allocate
// nothing. No GPU or camera.

#include <signal.h>
#include <sys/resource.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cstdint>
#include <cstdio>
#include <memory>
#include <new>
#include <string>
#include <utility>

#include "stream_files.hpp"
#include "test_allocation_failure.hpp"
#include "test_check.hpp"
#include "volumetric_kit/recon/stream/chunks.hpp"
#include "volumetric_kit/recon/stream/stream_writer.hpp"

namespace vkc = volumetric_kit::core;
namespace stream = volumetric_kit::recon::stream;
namespace vrst = vr_test::vrst;

namespace {

using Code = vkc::Status::Code;
using vrst::info;

stream::Session one_track() {
  stream::Session session;
  stream::Track track;
  track.kind = stream::TrackKind::Geometry;
  session.tracks.push_back(track);
  return session;
}

std::unique_ptr<stream::StreamWriter> create(const std::string& path) {
  auto result = stream::StreamWriter::create(path, one_track());
  if (!result.ok()) return nullptr;
  return std::move(result).value();
}

// Measure only the public call, then fail its last allocation at the same
// site. Returning the domain from each lambda avoids a caller-side Status
// copy consuming the injection. A separate measuring call also lets a
// successful operation be replayed on a fresh writer.
template <class Measure, class Call>
int expect_oom(const char* name, const Measure& measure, const Call& call,
               Code normal) {
  auto& failure = vr_test::allocation_failure;
  failure = {};
  failure.measure = true;
  const Code measured = measure();
  failure.measure = false;
  CHECK(measured == normal);
  CHECK(failure.bytes != 0 && failure.measured_site != nullptr);
  failure.site = failure.measured_site;
  failure.armed = true;
  Code returned = Code::Ok;
  bool escaped = false;
  try {
    returned = call();
  } catch (const std::bad_alloc&) {
    escaped = true;
  }
  const bool injected = failure.injected;
  failure = {};
  if (!injected || escaped || returned != Code::OutOfMemory) {
    std::fprintf(stderr, "FAIL %s: injected=%d escaped=%d returned=%d\n", name,
                 injected, escaped, static_cast<int>(returned));
    return 1;
  }
  return 0;
}

// Refusals and saved errors must still report their original code when
// allocations are available again.
template <class Call>
int expect_oom(const char* name, const Call& call, Code normal) {
  CHECK(expect_oom(name, call, call, normal) == 0);
  CHECK(call() == normal);
  return 0;
}

int writer_diagnostics(const std::string& dir) {
  const std::string path = dir + "/writer-diagnostics.vrst";
  auto writer = create(path);
  CHECK(writer != nullptr);
  CHECK(writer->append(info(0, true, 10, 1), nullptr, 0).ok());
  const auto size = writer->size();
  CHECK(expect_oom(
            "append invalid track",
            [&] {
              return writer->append(info(1, false, 10, 2), nullptr, 0).domain();
            },
            Code::InvalidArgument) == 0);
  CHECK(expect_oom(
            "append null payload",
            [&] {
              return writer->append(info(0, false, 10, 2), nullptr, 1).domain();
            },
            Code::InvalidArgument) == 0);
  CHECK(expect_oom(
            "append old sequence",
            [&] {
              return writer->append(info(0, false, 10, 1), nullptr, 0).domain();
            },
            Code::InvalidArgument) == 0);
  CHECK(expect_oom(
            "append old key timestamp",
            [&] {
              return writer->append(info(0, true, 9, 2), nullptr, 0).domain();
            },
            Code::InvalidArgument) == 0);
  CHECK(writer->size() == size);
  CHECK(writer->finish().ok());
  const auto finished_size = writer->size();
  CHECK(expect_oom(
            "append finished",
            [&] {
              return writer->append(info(0, false, 10, 2), nullptr, 0).domain();
            },
            Code::InvalidArgument) == 0);
  CHECK(expect_oom(
            "sync finished", [&] { return writer->sync().domain(); },
            Code::InvalidArgument) == 0);
  CHECK(expect_oom(
            "finish finished", [&] { return writer->finish().domain(); },
            Code::InvalidArgument) == 0);
  CHECK(writer->size() == finished_size);
  const auto reader = vrst::open(path);
  CHECK(reader != nullptr && reader->recovery().finished);
  CHECK(reader->stats()[0].records == 1);
  return 0;
}

int reader_diagnostics(const std::string& dir) {
  const std::string path = dir + "/reader-diagnostics.vrst";
  auto writer = create(path);
  CHECK(writer != nullptr);
  CHECK(writer->append(info(0, true, 10, 1), nullptr, 0).ok());
  CHECK(writer->finish().ok());
  const auto reader = vrst::open(path);
  CHECK(reader != nullptr);
  CHECK(expect_oom(
            "key invalid track",
            [&] { return reader->key(1, 0).status().domain(); },
            Code::InvalidArgument) == 0);
  CHECK(expect_oom(
            "key missing index",
            [&] { return reader->key(0, 1).status().domain(); },
            Code::NotFound) == 0);
  CHECK(expect_oom(
            "key_at invalid track",
            [&] { return reader->key_at(1, 10).status().domain(); },
            Code::InvalidArgument) == 0);
  CHECK(expect_oom(
            "key_at missing timestamp",
            [&] { return reader->key_at(0, 9).status().domain(); },
            Code::NotFound) == 0);
  stream::Record record;
  for (const auto offset : {std::uint64_t{0}, reader->begin().offset + 1}) {
    stream::Position pos{offset};
    CHECK(expect_oom(
              "next invalid position",
              [&] { return reader->next(&pos, &record).status().domain(); },
              Code::InvalidArgument) == 0);
    CHECK(pos.offset == offset);
  }
  auto pos = reader->key(0, 0);
  CHECK(pos.ok());
  const auto at = reader->key_at(0, 10);
  CHECK(at.ok() && at->offset == pos->offset);
  const auto more = reader->next(&pos.value(), &record);
  CHECK(more.ok() && more.value());
  CHECK(record.info.seq == 1 && record.info.pts_ns == 10);
  return 0;
}

int writer_retry(const std::string& dir) {
  auto measured = create(dir + "/measure-retry.vrst");
  const std::string path = dir + "/failure-retry.vrst";
  auto writer = create(path);
  CHECK(measured != nullptr && writer != nullptr);
  const auto initial_size = writer->size();
  CHECK(
      expect_oom(
          "append key index",
          [&] {
            return measured->append(info(0, true, 10, 1), nullptr, 0).domain();
          },
          [&] {
            return writer->append(info(0, true, 10, 1), nullptr, 0).domain();
          },
          Code::Ok) == 0);
  CHECK(writer->size() == initial_size);
  CHECK(writer->append(info(0, true, 10, 1), nullptr, 0).ok());
  const auto record_size = writer->size();
  CHECK(expect_oom(
            "finish footer", [&] { return measured->finish().domain(); },
            [&] { return writer->finish().domain(); }, Code::Ok) == 0);
  CHECK(writer->size() == record_size);
  CHECK(writer->finish().ok());
  const auto reader = vrst::open(path);
  CHECK(reader != nullptr && reader->recovery().finished);
  CHECK(reader->stats()[0].records == 1 && reader->key_count(0) == 1);
  return 0;
}

// The child changes only its own file-size limit and signal disposition.
int sticky_errors_child(const std::string& dir) {
  CHECK(::signal(SIGXFSZ, SIG_IGN) != SIG_ERR);
  auto writer = create(dir + "/sticky-io.vrst");
  CHECK(writer != nullptr);
  const auto size = writer->size();
  rlimit previous{};
  CHECK(::getrlimit(RLIMIT_FSIZE, &previous) == 0);
  const rlimit limited{static_cast<rlim_t>(size), previous.rlim_max};
  CHECK(::setrlimit(RLIMIT_FSIZE, &limited) == 0);
  const auto original = writer->append(info(0, false, 0, 0), nullptr, 0);
  CHECK(original.domain() == Code::IoError);
  CHECK(::setrlimit(RLIMIT_FSIZE, &previous) == 0);
  // These injections target copying the already-latched message, not
  // constructing the diagnostic for the initial failed write.
  CHECK(expect_oom(
            "append sticky error copy",
            [&] {
              return writer->append(info(0, false, 0, 0), nullptr, 0).domain();
            },
            Code::IoError) == 0);
  CHECK(expect_oom(
            "sync sticky error copy", [&] { return writer->sync().domain(); },
            Code::IoError) == 0);
  CHECK(expect_oom(
            "finish sticky error copy",
            [&] { return writer->finish().domain(); }, Code::IoError) == 0);
  CHECK(writer->append(info(0, false, 0, 0), nullptr, 0).message() ==
        original.message());
  CHECK(writer->sync().message() == original.message());
  CHECK(writer->finish().message() == original.message());
  CHECK(writer->size() == size);

  const std::string oom_path = dir + "/sticky-oom.vrst";
  writer = create(oom_path);
  CHECK(writer != nullptr);
  const rlimit partial_limit{static_cast<rlim_t>(size + 8), previous.rlim_max};
  CHECK(::setrlimit(RLIMIT_FSIZE, &partial_limit) == 0);
  // A non-key append allocates only when diagnosing this failed write.
  // Measure the initial diagnostic's string growth without relying on a
  // particular standard library's small-string capacity or errno wording.
  auto& failure = vr_test::allocation_failure;
  failure = {};
  failure.measure = true;
  std::string diagnostic("StreamWriter::append");
  diagnostic += ": cannot write ";
  failure.measure = false;
  CHECK(failure.bytes != 0);
  failure.armed = true;
  const auto refused = writer->append(info(0, false, 0, 0), nullptr, 0);
  const bool injected = failure.injected;
  failure = {};
  CHECK(::setrlimit(RLIMIT_FSIZE, &previous) == 0);
  CHECK(injected && refused.domain() == Code::OutOfMemory);
  const auto partial = vr_test::read_bytes(oom_path);
  // Some hosts reject the whole writev; others write up to the limit.
  CHECK(partial.size() == size || partial.size() == size + 8);
  // Once allocation and I/O can succeed again, the failed write must still
  // refuse later calls: retrying could append after a partially written record.
  CHECK(writer->append(info(0, false, 0, 0), nullptr, 0).domain() ==
        Code::OutOfMemory);
  CHECK(writer->sync().domain() == Code::OutOfMemory);
  CHECK(writer->finish().domain() == Code::OutOfMemory);
  CHECK(writer->size() == size);
  CHECK(vr_test::read_bytes(oom_path) == partial);
  const auto reader = vrst::open(oom_path);
  CHECK(reader != nullptr && !reader->recovery().finished);
  CHECK(reader->stats()[0].records == 0);
  CHECK(reader->recovery().torn_bytes == partial.size() - size);
  return 0;
}

int sticky_errors(const std::string& dir) {
  std::fflush(nullptr);
  const auto child = ::fork();
  CHECK(child >= 0);
  if (child == 0) ::_exit(sticky_errors_child(dir));
  int status = 0;
  CHECK(::waitpid(child, &status, 0) == child);
  CHECK(WIFEXITED(status) && WEXITSTATUS(status) == 0);
  return 0;
}

// An encoder that cannot allocate its chunk, or a refusal's diagnostic,
// returns OutOfMemory, and a refused record counts nothing either way.
int encoder_diagnostics() {
  CHECK(
      expect_oom(
          "encoder create",
          [] {
            return stream::ChunkEncoder::create(one_track()).status().domain();
          },
          Code::Ok) == 0);
  auto encoder = stream::ChunkEncoder::create(one_track());
  CHECK(encoder.ok());
  CHECK(encoder->frame(info(0, true, 10, 1), nullptr, 0).ok());
  CHECK(expect_oom(
            "frame invalid track",
            [&] {
              return encoder->frame(info(1, false, 10, 2), nullptr, 0)
                  .status()
                  .domain();
            },
            Code::InvalidArgument) == 0);
  CHECK(expect_oom(
            "frame null payload",
            [&] {
              return encoder->frame(info(0, false, 10, 2), nullptr, 1)
                  .status()
                  .domain();
            },
            Code::InvalidArgument) == 0);
  CHECK(expect_oom(
            "frame old sequence",
            [&] {
              return encoder->frame(info(0, false, 10, 1), nullptr, 0)
                  .status()
                  .domain();
            },
            Code::InvalidArgument) == 0);
  CHECK(expect_oom(
            "frame old key timestamp",
            [&] {
              return encoder->frame(info(0, true, 9, 2), nullptr, 0)
                  .status()
                  .domain();
            },
            Code::InvalidArgument) == 0);
  CHECK(encoder->frame(info(0, true, 10, 2), nullptr, 0).ok());
  return 0;
}

// A parser that cannot take what it is fed, or copy out a payload, keeps
// that failure: every later call returns OutOfMemory. One whose check
// failed keeps that failure too, and no more bytes.
int parser_failures(const std::string& dir) {
  const std::string path = dir + "/parsed.vrst";
  {
    auto writer = create(path);
    CHECK(writer != nullptr);
    const vrst::Bytes payload = vrst::pattern(100, 1);
    CHECK(writer->append(info(0, true, 10, 1), payload.data(), payload.size())
              .ok());
  }
  const vrst::Bytes bytes = vr_test::read_bytes(path);
  stream::Record record;
  vrst::Bytes payload;
  const auto stuck = [&](stream::ChunkParser& parser) {
    return parser.feed(bytes.data(), bytes.size()).domain() ==
               Code::OutOfMemory &&
           parser.next(&record, &payload).status().domain() ==
               Code::OutOfMemory;
  };
  stream::ChunkParser measured_feed;
  stream::ChunkParser failed_feed;
  CHECK(
      expect_oom(
          "parser feed",
          [&] {
            return measured_feed.feed(bytes.data(), bytes.size()).domain();
          },
          [&] { return failed_feed.feed(bytes.data(), bytes.size()).domain(); },
          Code::Ok) == 0);
  CHECK(stuck(failed_feed));

  stream::ChunkParser measured_next;
  stream::ChunkParser failed_next;
  CHECK(measured_next.feed(bytes.data(), bytes.size()).ok());
  CHECK(failed_next.feed(bytes.data(), bytes.size()).ok());
  vrst::Bytes measured_payload;
  CHECK(
      expect_oom(
          "parser next",
          [&] {
            return measured_next.next(&record, &measured_payload)
                .status()
                .domain();
          },
          [&] { return failed_next.next(&record, &payload).status().domain(); },
          Code::Ok) == 0);
  CHECK(measured_payload == vrst::pattern(100, 1));
  CHECK(stuck(failed_next));

  // A parser whose check failed takes no more bytes: feeding 1 MiB after
  // allocates no more than its message.
  stream::ChunkParser damaged;
  const vrst::Bytes bad = vrst::flipped(bytes, bytes.size() - 8, 0);
  CHECK(damaged.feed(bad.data(), bad.size()).domain() == Code::InvalidArgument);
  const vrst::Bytes more(std::size_t{1} << 20, 0);
  auto& failure = vr_test::allocation_failure;
  failure = {};
  failure.measure = true;
  const Code again = damaged.feed(more.data(), more.size()).domain();
  failure.measure = false;
  const std::size_t largest = failure.largest;
  failure = {};
  CHECK(again == Code::InvalidArgument && largest < more.size());
  return 0;
}

// A read into caller memory allocates nothing: the reader's and the parser's,
// a held record taken by a parser's next with no payload included.
int caller_memory(const std::string& dir) {
  const std::string path = dir + "/caller.vrst";
  const vrst::Bytes payload = vrst::pattern(5000, 3);
  {
    auto writer = create(path);
    CHECK(writer != nullptr);
    CHECK(writer->append(info(0, true, 10, 1), payload.data(), payload.size())
              .ok());
    CHECK(writer->finish().ok());
  }
  auto& failure = vr_test::allocation_failure;
  const auto allocates = [&](const auto& call) {
    failure = {};
    failure.measure = true;
    const bool ok = call();
    failure.measure = false;
    const bool none = failure.bytes == 0 && failure.measured_site == nullptr;
    failure = {};
    return ok && !none;
  };
  vrst::Bytes slot(payload.size() + 8, 0);
  const auto reader = vrst::open(path);
  CHECK(reader != nullptr);
  stream::Position pos = reader->begin();
  stream::Record record;
  CHECK(reader->next(&pos, &record).value());
  CHECK(!allocates(
      [&] { return reader->read(record, slot.data(), slot.size()).ok(); }));
  CHECK(vrst::Bytes(slot.begin(), slot.begin() + 5000) == payload);

  const vrst::Bytes bytes = vr_test::read_bytes(path);
  stream::ChunkParser parser;
  CHECK(parser.feed(bytes.data(), bytes.size()).ok());
  CHECK(!allocates([&] { return parser.next(&record, nullptr).value(); }));
  slot.assign(slot.size(), 0);
  CHECK(!allocates(
      [&] { return parser.read(record, slot.data(), slot.size()).ok(); }));
  CHECK(vrst::Bytes(slot.begin(), slot.begin() + 5000) == payload);
  return 0;
}

}  // namespace

int main() try {
  const vr_test::Scratch scratch("recon-stream-oom-");
  const auto dir = scratch.path.string();
  int failed = 0;
  failed |= writer_diagnostics(dir);
  failed |= reader_diagnostics(dir);
  failed |= writer_retry(dir);
  failed |= sticky_errors(dir);
  failed |= encoder_diagnostics();
  failed |= parser_failures(dir);
  failed |= caller_memory(dir);
  return failed;
} catch (const std::bad_alloc&) {
  vr_test::allocation_failure = {};
  std::fprintf(stderr, "FAIL: std::bad_alloc escaped a public stream call\n");
  return 1;
}
