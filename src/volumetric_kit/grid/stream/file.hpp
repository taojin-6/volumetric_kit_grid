// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

// The POSIX file calls the stream writes and reads with, each failure a
// Status naming the caller, the path and the system's reason: a new file
// created exclusively, writes that loop until every byte is written, a flush
// to storage, and reads at an offset, which several threads may make at once.

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>

#include "volumetric_kit/core/base/result.hpp"
#include "volumetric_kit/core/base/unique_fd.hpp"

namespace volumetric_kit::recon::stream::detail {

// "<api>: <what> <path>: <the system's reason>", in the domain `error` says:
// NotFound for a missing file or directory, InvalidArgument for something
// already there or a directory, OutOfMemory, and IoError for the rest.
// Diagnostic allocation failures must still return a Status so a writer can
// latch the failed I/O, including after a partial write.
core::Status system_failure(std::string_view api, std::string_view what,
                            const std::string& path, int error) noexcept;

// A new file at `path`, opened for writing, created only if nothing is there
// (O_EXCL: not even a symbolic link), with 0666 less the umask.
core::Result<core::UniqueFd> create_new_file(std::string_view api,
                                             const std::string& path);

// The file at `path`, opened for reading.
core::Result<core::UniqueFd> open_file(std::string_view api,
                                       const std::string& path);

// The size of `fd`, which must be a regular file (InvalidArgument otherwise).
core::Result<std::uint64_t> regular_file_size(std::string_view api,
                                              const std::string& path, int fd);

// One run of bytes to write.
struct Bytes {
  const void* data = nullptr;
  std::size_t size = 0;
};

// Write every byte of `parts`, in order, at the file's end: one writev, and
// more while a write comes up short or is interrupted.
core::Status write_all(std::string_view api, const std::string& path, int fd,
                       const Bytes* parts, int count);

// Flush `fd` to the storage device.
core::Status flush_file(std::string_view api, const std::string& path, int fd);

// Flush the entries of the directory `path` is in, where the filesystem
// allows it. Best effort: what it follows has already happened.
void flush_directory_of(const std::string& path);

// Close `fd`, reporting a failure.
core::Status close_file(std::string_view api, const std::string& path,
                        core::UniqueFd& fd);

// Read exactly `size` bytes at `offset`; a file that ends first is IoError
// (it is shorter than when it was opened).
core::Status read_at(std::string_view api, const std::string& path, int fd,
                     std::uint64_t offset, void* out, std::size_t size);

}  // namespace volumetric_kit::recon::stream::detail
