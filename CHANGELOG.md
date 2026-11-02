# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project aims to
follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.9.0] - 2026-11-02

### Added

- **A sender that dials out** (`networking/dial.hpp`): with
  `ServerOptions::listen` false a `StreamServer` binds no socket, so no
  host that reaches the sender can connect to it; `dial(host, port, limit)`
  connects to the reader's numeric IPv4 address, and
  `StreamServer::add_viewer` serves the socket as an accepted one, in
  either mode, with the same sender setup (`TCP_NODELAY`, the 128 KiB
  low-water mark, BBR on Linux off loopback), counted in `viewers()` and,
  on demand, `key_wanted()` before it returns. The reader listens on one
  numeric address with `Listener`, never the wildcard, each accepted socket
  with a client's 2 MiB receive buffer, and reads each connection with
  `StreamClient::adopt`. `dial`'s refusals are sentences an address entry
  can show (DESIGN.md, networking; the 2026-11-02 entry in DECISIONS.md).

### Changed

- **A viewer counts in `key_wanted()` as it is admitted**, before its
  thread starts, rather than as its thread starts: earlier, never later.
  The bytes served are unchanged.

## [0.8.0] - 2026-11-01

### Added

- **`StreamServer::draining()`** counts, without `replay_gop`, the viewers
  still sending the newest key's records: each going on from the newest
  key that has not yet sent it and every record after it, a record counting
  until the viewer would take its next. A key published meanwhile would
  make one skip a record it has not begun, so a live sender may wait to
  make a key until the count is 0, read just before the key's publish.
  Safe from any thread (DESIGN.md, networking; the 2026-11-01 entry in
  DECISIONS.md).

## [0.7.1] - 2026-11-01

### Changed

- **A viewer that is not on loopback gets BBR on Linux**: `StreamServer`
  sets `TCP_CONGESTION` to `bbr` on each such accepted socket when
  `/proc/sys/net/ipv4/tcp_allowed_congestion_control` lists it, and
  otherwise keeps the kernel's default, said once a process at Info; the
  accept never fails for it. Loopback viewers and Apple senders are
  unchanged, as is every byte served (DESIGN.md, networking; the
  2026-11-01 entry in DECISIONS.md).

## [0.7.0] - 2026-10-30

### Added

- **`StreamServer::finish`** ends a stream without losing its tail: it
  refuses later records and connections, closes each viewer once it has
  sent every record queued for it, so the viewer reads the end right after
  the last, and closes any still behind after `kFinishLimit` (1 s). The
  destructor still closes at once (DESIGN.md, networking; the 2026-10-30
  finish entry in DECISIONS.md).

### Fixed

- **A viewer of a finishing `CodecStream` gets its last records**:
  `finish` writes the footer, then finishes the server, so a connected
  viewer gets the records `finish` writes past the last frame, which the
  file has, before its end. The bytes served are unchanged.

## [0.6.0] - 2026-10-30

### Changed

- **`StreamServer` serves a session of any tracks**: track 0's keys start a
  GOP, and a record of any other track must be a key (`InvalidArgument`
  otherwise, nothing sent) and joins the GOP as a P-frame does. A viewer
  that joins or skips to a newer key starts at a track-0 key and gets every
  record published after it, on any track. `start` no longer returns
  `Unsupported` for more than one track (DESIGN.md, networking; the
  2026-10-30 serving entry in DECISIONS.md).
- **`CodecStream` serves its Audio tracks**: with `CodecOutputs::serve`, a
  session with an Audio track is served as its file is written. A session
  with no Audio track writes and serves the bytes 0.5.0 did.
- **`serve_stream` replays every track** (`vgrid_example::ReplayLoop`): from
  track 0's first key, every track's records in file order, every pts moved
  by one span a pass and each track's seq by its own, so the tracks stay
  together across passes. A damaged payload skips its own track's records
  that are not keys up to that track's next key. `ReplayLoop::create` takes
  no track.

## [0.5.0] - 2026-10-30

### Added

- **`CodecStream` writes Audio tracks, muxed by pts**: `create` takes a
  session of a Geometry track 0 and Audio tracks after it, each with params
  a reader plays and on track 0's clock. `append` writes and serves an
  Audio record right after the first frame at or past its pts, drops one
  below the first frame and holds at most `kHoldLimitNs` (1 s) past the
  last frame; `finish` writes what is held; `appended(track)` counts each
  track's records held, dropped and written (DESIGN.md, codec_stream; the
  2026-10-30 mux entry in DECISIONS.md).

### Changed

- **A session `CodecStream` refuses** is still `InvalidArgument`, its
  message now naming the track. With `CodecOutputs::serve`, a session with
  an Audio track is the server's `Unsupported`, as `StreamServer` serves
  one track; its file works. A session with no Audio track writes and
  serves the bytes 0.4.0 did.

## [0.4.0] - 2026-10-30

### Added

- **Audio tracks in `.vrst`**: `TrackKind::Audio` (kind 2), interleaved
  samples as its params say, every record a key at its first sample's
  capture time. `stream/audio.hpp` holds the params: `AudioParams`,
  `AudioCodec::PcmS16`, `encode_audio_params` and `decode_audio_params`,
  8 bytes. The format stays version 0: a reader that does not know the kind
  passes it through, and a file without audio is byte for byte as before
  (DESIGN.md, stream; the 2026-10-30 entry in DECISIONS.md).

## [0.3.1] - 2026-10-29

### Changed

- **The `Encoder` holds each P-frame one quantization step against its
  reference**: a coefficient within one step of the reference's keeps the
  reference's value, so noise near a rounding boundary no longer flickers
  the decoded surface and P-frames take fewer bits; a larger change passes
  in the same frame. I-frames are quantized afresh and re-centre the held
  values, which drift in between, so a longer `intra_period` decodes
  rougher. The format and the `Decoder` are unchanged; a P-frame decodes to
  within one step a coefficient of an I-frame of its grid (DESIGN.md,
  codec; the 2026-10-29 entry in DECISIONS.md).

## [0.3.0] - 2026-10-29

### Changed

- **`vgrid_example::codec_outputs` serves `output.max_viewers` viewers**
  (core 0.30.0, default 4, the server's own) in place of the server's
  default alone, so recon's `fuse_orbbec` takes its viewer count from the
  scenario.
- **The core floor is 0.30.0** (`VGRID_CORE_MIN_VERSION`), whose config
  reads `output.max_viewers`; a minor, as every consumer moves its core pin
  with it.

### Migration

- A consumer that declares the core first pins core 0.30.0 or newer with
  this grid; one at 0.29.0 fails grid's configure (`vkc_require_core`).

## [0.2.1] - 2026-10-29

### Changed

- **Marching-cubes vertex normals**: each vertex carries the field's
  gradient at its own position, the same from every cell and both kernels,
  where it carried its cell's; positions and triangles are unchanged
  (DESIGN.md, mesh).

## [0.2.0] - 2026-10-29

### Added

- **`StreamServer::viewers()`**: the viewers a server is serving now, safe
  from any thread (DESIGN.md, networking).

## [0.1.0] - 2026-10-28

Extracted from `volumetric_kit_recon` at `5eeec098`; earlier history in that
repository. The commits here are recon's that touched grid's files, each file
at its path here and each `#N` written as `taojin-6/volumetric_kit_recon#N`;
a file's history from before an earlier move inside recon stays there, and
recon's CHANGELOG [0.1.0] lists the changes. The tiers, tests and examples
moved and took grid's names (the 2026-10-28 package entry in DECISIONS.md;
recon's CHANGELOG has the old → new table).
