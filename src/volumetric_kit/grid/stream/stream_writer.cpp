// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include "volumetric_kit/recon/stream/stream_writer.hpp"

#include <unistd.h>

#include <algorithm>
#include <utility>

#include "chunk.hpp"
#include "file.hpp"
#include "volumetric_kit/core/base/exception_status.hpp"
#include "volumetric_kit/core/base/file_io.hpp"

namespace volumetric_kit::recon::stream {
namespace {

constexpr std::uint8_t kZeros[8] = {};

}  // namespace

StreamWriter::StreamWriter(core::UniqueFd fd, std::string path,
                           ChunkEncoder encoder, std::size_t tracks)
    : fd_(std::move(fd)),
      path_(std::move(path)),
      encoder_(std::move(encoder)),
      size_(encoder_.session_chunk().size()),
      stats_(tracks) {}

StreamWriter::~StreamWriter() = default;

core::Result<std::unique_ptr<StreamWriter>> StreamWriter::create(
    const std::string& path, const Session& session) {
  constexpr std::string_view kApi = "StreamWriter::create";
  bool created = false;
  try {
    VKC_TRY(core::check_path(kApi, path));
    core::Result<ChunkEncoder> encoder = ChunkEncoder::create(session);
    if (!encoder.ok()) {
      return std::move(encoder).status().with_context(std::string(kApi) + ": " +
                                                      path);
    }
    VKC_ASSIGN(core::UniqueFd fd, detail::create_new_file(kApi, path));
    created = true;
    const std::vector<std::uint8_t>& header = encoder->session_chunk();
    const detail::Bytes part{header.data(), header.size()};
    core::Status written = detail::write_all(kApi, path, fd.get(), &part, 1);
    if (written.ok()) written = detail::flush_file(kApi, path, fd.get());
    if (!written.ok()) {
      fd.reset();
      ::unlink(path.c_str());
      return written;
    }
    detail::flush_directory_of(path);
    return std::unique_ptr<StreamWriter>(
        new StreamWriter(std::move(fd), path, std::move(encoder).value(),
                         session.tracks.size()));
  } catch (...) {
    core::Status failure =
        core::exception_status(kApi, core::Status::Code::IoError);
    if (created) ::unlink(path.c_str());
    return failure;
  }
}

core::Status StreamWriter::check_open(std::string_view api) const {
  if (!failure_.ok()) return failure_;
  if (finished_) {
    return core::Status::invalid_argument(std::string(api) + ": " + path_ +
                                          " is finished");
  }
  return {};
}

core::Status StreamWriter::append(const RecordInfo& info,
                                  const std::uint8_t* data, std::size_t size) {
  constexpr std::string_view kApi = "StreamWriter::append";
  try {
    VKC_TRY(check_open(kApi));
    // The index entry's room comes first, before the encoder counts the
    // record, so a refused allocation leaves the writer unchanged and a
    // record once written is always indexed.
    if (info.key &&
        index_keys_.capacity() - index_keys_.size() < detail::kKeyEntryBytes) {
      index_keys_.reserve(std::max<std::size_t>(2 * index_keys_.capacity(),
                                                64 * detail::kKeyEntryBytes));
    }
    core::Result<ChunkEncoder::RecordFraming> framed =
        encoder_.frame(info, data, size);
    if (!framed.ok()) {
      return std::move(framed).status().with_context(std::string(kApi) + ": " +
                                                     path_);
    }
    const ChunkEncoder::RecordFraming& framing = *framed;
    const detail::Bytes parts[3] = {
        {framing.header.data(), framing.header.size()},
        {data, size},
        {kZeros, framing.padding}};
    core::Status written = detail::write_all(kApi, path_, fd_.get(), parts, 3);
    if (!written.ok()) {
      failure_ = std::move(written);
      return failure_;
    }
    const std::uint64_t offset = size_;
    size_ += framing.header.size() + std::uint64_t{size} + framing.padding;
    if (info.key) detail::append_key_entry(index_keys_, offset, info);
    detail::count_record(stats_[info.track], info, size);
    return {};
  } catch (...) {
    return core::exception_status(kApi, core::Status::Code::IoError);
  }
}

core::Status StreamWriter::sync() {
  constexpr std::string_view kApi = "StreamWriter::sync";
  try {
    VKC_TRY(check_open(kApi));
    core::Status flushed = detail::flush_file(kApi, path_, fd_.get());
    if (!flushed.ok()) {
      failure_ = std::move(flushed);
      return failure_;
    }
    return {};
  } catch (...) {
    return core::exception_status(kApi, core::Status::Code::IoError);
  }
}

core::Status StreamWriter::finish() {
  constexpr std::string_view kApi = "StreamWriter::finish";
  try {
    VKC_TRY(check_open(kApi));
    const std::vector<std::uint8_t> footer =
        detail::encode_footer(stats_, index_keys_, size_);
    const detail::Bytes part{footer.data(), footer.size()};
    core::Status done = detail::write_all(kApi, path_, fd_.get(), &part, 1);
    if (done.ok()) done = detail::flush_file(kApi, path_, fd_.get());
    if (done.ok()) done = detail::close_file(kApi, path_, fd_);
    if (!done.ok()) {
      failure_ = std::move(done);
      return failure_;
    }
    size_ += footer.size();
    finished_ = true;
    return {};
  } catch (...) {
    return core::exception_status(kApi, core::Status::Code::IoError);
  }
}

std::uint64_t StreamWriter::size() const noexcept { return size_; }

}  // namespace volumetric_kit::recon::stream
