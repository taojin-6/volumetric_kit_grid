// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

// What a StreamServer holds for its viewers: one GOP, the newest key record
// and every record after it, as whole chunks that every viewer still to send
// them shares, and each viewer's place in it. Newest key wins: a viewer that
// has not sent the GOP when a newer key arrives skips the rest, at a chunk
// boundary, and goes on from that key, so no P-frame is ever handed out
// without the records it is predicted from. No sockets, threads or clock: the
// server calls it under its lock.

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace volumetric_kit::recon::networking::detail {

// A record's whole chunk (stream::ChunkEncoder's header, the payload and its
// padding), shared by the log and every viewer sending it.
using Chunk = std::shared_ptr<const std::vector<std::uint8_t>>;

class GopLog {
 public:
  // A viewer's place: the GOP it is in, and the index of the next chunk of it
  // to hand out. A new viewer's, {}, is in no GOP: it starts at the key of the
  // GOP there is at its first chunk, or of the next.
  struct Cursor {
    std::uint64_t gop = 0;
    std::size_t next = 0;
  };

  // A log keeping a GOP of at most `max_gop_bytes` chunk bytes.
  explicit GopLog(std::uint64_t max_gop_bytes) noexcept;

  // Take one record's chunk. A key starts a new GOP, replacing the log. Any
  // other record joins the GOP, unless no key has come yet or the GOP is
  // broken. A chunk that would take the GOP past max_gop_bytes breaks it: the
  // chunk and the GOP's are dropped, and every record until the next key.
  // Returns whether the chunk was kept. A chunk the log has no room to hold
  // breaks the GOP too, and what its vector threw is thrown on.
  bool append(Chunk chunk, bool key);

  // Break the GOP, dropping its chunks: what a record that could not be
  // copied leaves, as it is lost.
  void break_gop() noexcept;

  // The next chunk for the viewer at `cursor`, which it moves on: from the
  // key of a newer GOP than the cursor's, else the GOP's next; null when the
  // viewer has every chunk there is. Called at a chunk boundary.
  Chunk next(Cursor* cursor) const noexcept;

  // Whether a GOP is open: a key has come, and the GOP is not broken.
  bool open() const noexcept;

 private:
  std::uint64_t max_gop_bytes_;
  std::uint64_t gop_ = 0;  // the GOP's number: the keys taken
  bool open_ = false;
  std::uint64_t bytes_ = 0;  // the GOP's chunk bytes
  std::vector<Chunk> chunks_;
};

}  // namespace volumetric_kit::recon::networking::detail
