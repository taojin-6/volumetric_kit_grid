// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

// What a StreamServer holds for its viewers, as whole chunks that every
// viewer still to send them shares, and each viewer's place in them. No
// sockets, threads or clock: the server calls it under its lock.
//
// Replaying the GOP (ServerOptions::replay_gop): one GOP, the newest key
// record and every record after it. Newest key wins: a viewer that has not
// sent the GOP when a newer key arrives skips the rest, at a chunk boundary,
// and goes on from that key, so no P-frame is ever handed out without the
// records it is predicted from.
//
// Keys on demand (replay_gop false): a viewer joins waiting for the next key,
// then gets every record after it, in order, while it stays within
// max_gop_bytes of the newest; one that would fall further behind waits for
// the next key again. The log keeps only the records some viewer still has
// to send. Each join and each fall-back counts in key_wanted(), which a
// sender reads to make its next frame a key.

#include <cstddef>
#include <cstdint>
#include <deque>
#include <memory>
#include <vector>

namespace volumetric_kit::recon::networking::detail {

// A record's whole chunk (stream::ChunkEncoder's header, the payload and its
// padding), shared by the log and every viewer sending it.
using Chunk = std::shared_ptr<const std::vector<std::uint8_t>>;

class GopLog {
 public:
  // A viewer's place. Replaying: the GOP it is in, and the index of the next
  // chunk of it to hand out; a new viewer's, {}, is in no GOP: it starts at
  // the key of the GOP there is at its first chunk, or of the next. Keys on
  // demand: whether it waits for a key, and once one let it in, the number
  // of its next record (the log numbers the records it keeps).
  struct Cursor {
    std::uint64_t gop = 0;
    std::uint64_t next = 0;
    bool waiting = true;
  };

  // A log keeping chunks of at most `max_gop_bytes`: one GOP while
  // `replay`, else what the joined viewers still need, as above.
  explicit GopLog(std::uint64_t max_gop_bytes, bool replay = true) noexcept;

  // Keys on demand: count `cursor`'s viewer in, waiting for the next key,
  // until it leaves. Throws what the viewers' vector throws, counting none.
  // Replaying: nothing, as a viewer starts at the GOP's key.
  void join(Cursor* cursor);
  // The viewer at `cursor` has left: forget it, if it joined.
  void leave(const Cursor* cursor) noexcept;

  // Take one record's chunk; returns whether it was kept. Replaying, a key
  // starts a new GOP, replacing the log, and any other record joins the
  // GOP, unless no key has come yet or the GOP is broken; a chunk that would
  // take the GOP past max_gop_bytes breaks it: the chunk and the GOP's are
  // dropped, and every record until the next key. Keys on demand, a viewer
  // the chunk would leave more than max_gop_bytes behind falls back to
  // waiting, a key lets every waiting viewer in, and the chunk is kept only
  // while a viewer is in; what no viewer still needs is dropped. A chunk the
  // log has no room to hold breaks the GOP too (break_gop), and what its
  // container threw is thrown on.
  bool append(Chunk chunk, bool key);

  // A record is lost (it could not be copied): replaying, break the GOP,
  // dropping its chunks; keys on demand, every viewer in falls back to
  // waiting, its chunks dropped.
  void break_gop() noexcept;

  // The next chunk for the viewer at `cursor`, which it moves on: replaying,
  // from the key of a newer GOP than the cursor's, else the GOP's next; keys
  // on demand, its next record unless it waits. Null when the viewer has
  // every chunk there is for it. Called at a chunk boundary.
  Chunk next(Cursor* cursor) const noexcept;

  // Replaying: whether a GOP is open, a key has come and the GOP is not
  // broken. Keys on demand: whether a viewer is in.
  bool open() const noexcept;

  // Keys on demand: the joins and fall-backs so far. Replaying: 0.
  std::uint64_t key_wanted() const noexcept;

  // The chunk bytes the log holds.
  std::uint64_t bytes() const noexcept;

 private:
  // A chunk kept on demand, and the bytes kept before it since the start.
  struct Held {
    Chunk chunk;
    std::uint64_t start = 0;
  };

  bool append_replay(Chunk chunk, bool key);
  bool append_on_demand(Chunk chunk, bool key);
  // Keys on demand: the number the next record kept takes.
  std::uint64_t head() const noexcept;
  // Keys on demand: the bytes kept from the viewer at `cursor`'s next record
  // on, none for a viewer that has them all.
  std::uint64_t behind(const Cursor& cursor) const noexcept;
  // Keys on demand: drop the records no viewer still needs.
  void trim() noexcept;
  void fall_back(Cursor* cursor) noexcept;

  std::uint64_t max_gop_bytes_;
  bool replay_;
  // Replaying.
  std::uint64_t gop_ = 0;  // the GOP's number: the keys taken
  bool open_ = false;
  std::uint64_t bytes_ = 0;  // the GOP's chunk bytes
  std::vector<Chunk> chunks_;
  // Keys on demand.
  std::vector<Cursor*> viewers_;  // the joined viewers' cursors
  std::deque<Held> held_;
  std::uint64_t first_ = 0;  // held_.front()'s number, or head() when empty
  std::uint64_t kept_ = 0;   // the chunk bytes kept since the start
  std::uint64_t key_wanted_ = 0;
};

}  // namespace volumetric_kit::recon::networking::detail
