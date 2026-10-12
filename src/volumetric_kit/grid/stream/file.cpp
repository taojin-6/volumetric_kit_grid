// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// TODO(stream): this file's calls, chunk.hpp's CRC-32 and its little-endian
// words move to the family core's base tier when a second sibling needs them
// (core's rule for what it holds). Until then they are the stream's own; the
// core's file_io.cpp keeps a private errno mapping and flush of its own.

#include "file.hpp"

#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>  // IWYU pragma: keep (ssize_t on Linux)
#include <sys/uio.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <limits>
#include <system_error>
#include <utility>

#include "volumetric_kit/core/base/check.hpp"

namespace volumetric_kit::recon::stream::detail {
namespace {

static_assert(sizeof(off_t) == 8, "offsets past 2 GiB need a 64-bit off_t");

// One read() or write() moves at most this much: macOS refuses a count above
// INT_MAX.
constexpr std::size_t kMaxIo = std::size_t{1} << 30;
// The most runs one write_all takes: a record's header, payload and padding.
constexpr int kMaxParts = 4;

constexpr mode_t kNewFileMode =
    S_IRUSR | S_IWUSR | S_IRGRP | S_IWGRP | S_IROTH | S_IWOTH;

}  // namespace

core::Status system_failure(std::string_view api, std::string_view what,
                            const std::string& path, int error) {
  std::string message(api);
  message += ": ";
  message += what;
  message += ' ';
  message += path;
  message += ": ";
  message += std::generic_category().message(error);
  switch (error) {
    case ENOENT:
    case ENOTDIR:  // a component of the path is not a directory
      return core::Status::not_found(std::move(message));
    case EEXIST:
    case EISDIR:
      return core::Status::invalid_argument(std::move(message));
    case ENOMEM:
      return core::Status::out_of_memory(std::move(message));
    default:
      return core::Status::io_error(std::move(message));
  }
}

core::Result<core::UniqueFd> create_new_file(std::string_view api,
                                             const std::string& path) {
  for (;;) {
    const int fd = ::open(path.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC,
                          kNewFileMode);
    if (fd >= 0) return core::UniqueFd(fd);
    if (errno != EINTR)
      return system_failure(api, "cannot create", path, errno);
  }
}

core::Result<core::UniqueFd> open_file(std::string_view api,
                                       const std::string& path) {
  // O_NONBLOCK so a FIFO at the path is refused rather than waited on; it
  // changes nothing for a regular file.
  for (;;) {
    const int fd = ::open(path.c_str(), O_RDONLY | O_NONBLOCK | O_CLOEXEC);
    if (fd >= 0) return core::UniqueFd(fd);
    if (errno != EINTR) return system_failure(api, "cannot open", path, errno);
  }
}

core::Result<std::uint64_t> regular_file_size(std::string_view api,
                                              const std::string& path, int fd) {
  struct stat info{};
  if (::fstat(fd, &info) != 0) {
    return system_failure(api, "cannot inspect", path, errno);
  }
  if (!S_ISREG(info.st_mode)) {
    return core::Status::invalid_argument(std::string(api) + ": " + path +
                                          " is not a regular file");
  }
  return static_cast<std::uint64_t>(info.st_size);
}

core::Status write_all(std::string_view api, const std::string& path, int fd,
                       const Bytes* parts, int count) {
  VKC_CHECK(count >= 0 && count <= kMaxParts, "write_all takes 0 to 4 runs");
  int first = 0;         // the first part with bytes left
  std::size_t done = 0;  // its bytes written
  iovec iov[kMaxParts];
  for (;;) {
    while (first < count && done == parts[first].size) {
      ++first;
      done = 0;
    }
    if (first == count) return {};
    int n = 0;
    std::size_t total = 0;
    for (int i = first; i < count && total < kMaxIo; ++i) {
      const std::size_t skip = i == first ? done : 0;
      const std::size_t len = std::min(parts[i].size - skip, kMaxIo - total);
      if (len == 0) continue;
      iov[n].iov_base =
          const_cast<char*>(static_cast<const char*>(parts[i].data) + skip);
      iov[n].iov_len = len;
      ++n;
      total += len;
    }
    const ssize_t written = ::writev(fd, iov, n);
    if (written <= 0) {
      const int error = written == 0 ? EIO : errno;
      if (error == EINTR) continue;
      return system_failure(api, "cannot write", path, error);
    }
    for (std::size_t left = static_cast<std::size_t>(written); left > 0;) {
      const std::size_t room = parts[first].size - done;
      if (left < room) {
        done += left;
        left = 0;
      } else {
        left -= room;
        ++first;
        done = 0;
      }
    }
  }
}

core::Status flush_file(std::string_view api, const std::string& path, int fd) {
  // Apple's fsync stops at the drive's cache, so there F_FULLFSYNC flushes
  // that too; a filesystem that refuses it gets fsync.
#ifdef F_FULLFSYNC
  if (::fcntl(fd, F_FULLFSYNC) == 0) return {};
#endif
  if (::fsync(fd) != 0) return system_failure(api, "cannot flush", path, errno);
  return {};
}

void flush_directory_of(const std::string& path) {
  const std::size_t slash = path.rfind('/');
  const std::string directory = slash == std::string::npos ? "."
                                : slash == 0 ? "/"
                                             : path.substr(0, slash);
  const core::UniqueFd dir(
      ::open(directory.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC));
  if (!dir.valid()) return;
#ifdef F_FULLFSYNC
  if (::fcntl(dir.get(), F_FULLFSYNC) == 0) return;
#endif
  (void)::fsync(dir.get());
}

core::Status close_file(std::string_view api, const std::string& path,
                        core::UniqueFd& fd) {
  if (::close(fd.release()) != 0 && errno != EINTR) {
    return system_failure(api, "cannot close", path, errno);
  }
  return {};
}

core::Status read_at(std::string_view api, const std::string& path, int fd,
                     std::uint64_t offset, void* out, std::size_t size) {
  constexpr auto kMaxOffset =
      static_cast<std::uint64_t>(std::numeric_limits<off_t>::max());
  if (offset > kMaxOffset || size > kMaxOffset - offset) {
    return system_failure(api, "cannot read", path, EOVERFLOW);
  }
  auto* at = static_cast<unsigned char*>(out);
  while (size > 0) {
    const ssize_t n =
        ::pread(fd, at, std::min(size, kMaxIo), static_cast<off_t>(offset));
    if (n < 0) {
      if (errno == EINTR) continue;
      return system_failure(api, "cannot read", path, errno);
    }
    if (n == 0) {
      return core::Status::io_error(
          std::string(api) + ": " + path + " ends before byte " +
          std::to_string(offset + 1) + ", shorter than when it was opened");
    }
    at += n;
    offset += static_cast<std::uint64_t>(n);
    size -= static_cast<std::size_t>(n);
  }
  return {};
}

}  // namespace volumetric_kit::recon::stream::detail
