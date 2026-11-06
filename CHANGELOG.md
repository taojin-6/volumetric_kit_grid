# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project aims to
follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.15.0] - 2026-11-05

### Added

- **A viewer's notes**: `networking::ViewerNote` (a tone and a line of at most
  `kMaxNoteLineBytes`, 200 bytes), `StreamClient::send_note`,
  `ServerOptions::read_notes` and `StreamServer::newest_note`. A viewer tells
  its sender one short line up the socket it reads from, as a small `.vrst`
  stream of the new `stream::TrackKind::Note`, which never reaches a file.
  `send_note` never waits, the newest wins, and no failure reading a note
  touches the stream. `read_notes` is off by default, which reads and drops
  what a viewer sends as ever, and a viewer that sends no note sends nothing.
  The wire down is unchanged either way. With `read_notes`, a note landing as a
  socket closes can cost that viewer the last records (`StreamServer::finish`;
  the 2026-11-05 notes entry in DECISIONS.md).

## [0.14.0] - 2026-11-05

### Added

- **`stream::Session::up`**: the world's up, a direction in the coordinates of
  the session's geometry and poses, which a viewer keeps up on screen; all
  zero, the default, when unknown. A session with one writes a `VRST` header
  of 72 bytes, still version 0, whose up a reader before this one ignores; a
  session without one writes the bytes it always did. The writer, the encoder,
  the server and the readers refuse an up neither all zero nor finite and of
  length 1 to within 1e-4 (the 2026-11-05 entry in DECISIONS.md).

### Changed

- **The examples' `codec_session`** states +Y, the placed mesh frame's up, for
  a session with geometry params, so a `codec_mesh` stream names its up. Other
  sessions leave it to their caller.

## [0.13.0] - 2026-11-03

### Removed

- **The two-step mesh conversion, a breaking change**:
  `VoxelHashMap::allocate_from_triangles`, `MeshIntegrator::integrate`, and
  `volume/triangle_candidates.hpp` (`triangle_candidate_offsets`).
  `MeshIntegrator::convert` replaces all three.

### Added

- **`MeshIntegrator::convert`**: makes a grid hold one mesh's distance field
  and nothing else in one call. It empties the grid, every attribute zeroed
  as `clear` zeroes them; counts the mesh's candidate blocks on the device;
  places the band's distinct blocks in one lock-free insert; and writes the
  band. It grows the grid itself, only when the insert meets a capacity
  limit (more band blocks than the grid holds, a full chain, or a band past
  about 7/8 of the grid), to twice its buckets or the band's need,
  whichever is more. A warm call makes two blocking waits. A refusal the
  host can make leaves the grid untouched; one only the device can find (a
  candidate total past 2^32, a bin past `kMaxBinTriangles`, the band
  index's own, a band past the largest grid) leaves it with no observed
  voxel (DESIGN.md, voxelize; the 2026-11-03 entry in DECISIONS.md).
- **`VoxelBlockGrid::record_clear`**: records `clear` into a caller's batch.
- **`MeshSdfParams::full_band_m`**: writes the band whole only within this
  distance of the field's zero level, plus a one-voxel apron, each voxel
  written with the whole band's bytes; 0, the default, writes it all.
  `MeshIntegrateStats::full_blocks` counts the blocks written whole.
- **`codec::true_sdf_keep_distance`**: the distance from zero within which
  every corner of every cell a true-SDF frame keeps lies.
  `mesh_codec_config` sets `full_band_m` to it, and `codec_mesh` writes the
  bytes 0.12.1 did.

### Changed

- **`clear()` is one submit**: `record_clear` and a submit.
- **The conversion's rows** are `mesh convert`, `  ..allocate` and
  `  ..write`.
- **A conversion is one tick**: its band blocks' `requested` and its written
  blocks' `changed` carry the same value.
- **`MeshIntegrateStats::blocks`** counts the blocks written, whole or in
  part.
- **The write measures a block in one round**, from per-triangle records,
  in a layout the device's facts pick: no voxel's bytes change.

Migration: replace "clear, allocate with retries and growth, integrate" with
one `convert`. It grows the grid itself, and zeroes every attribute the grid
declares.

## [0.12.1] - 2026-11-03

### Added

- **A mesh source's codec setup in the examples' helpers**
  (`codec_config.hpp`): `mesh_codec_config` makes the grid, the encoder's
  settings with `EncoderConfig::true_sdf` and the conversion's from a
  scenario, and refuses a band too narrow for a true SDF, on the host before
  any frame loads; `create_mesh_grid` makes the `tsdf` and `weight` grid a
  mesh converts into. `codec_mesh` calls both, and writes the bytes 0.12.0
  did.

## [0.12.0] - 2026-11-02

### Added

- **RGB-D tracks in `.vrst`**: `TrackKind::RgbdDepth` (kind 3), one RGB-D
  sensor's frames, each frame's depth half a record and every record a key,
  and `TrackKind::RgbdColor` (kind 4), the same sensor's colour, one coded
  picture a record. Their params and payloads are the sensor package's wire
  form, bytes grid carries and never parses. The format stays version 0: a
  reader that does not know a kind passes it through.
- **A record's payload in three runs**: `RecordPayload::of` takes a third
  run, so a depth record's prefix, rows and plane go out as they lie, with no
  joined copy: `StreamWriter::append` writes them in its one `writev`,
  `StreamServer::publish` makes its one copy from them, and both carry the
  CRC `of` computed (DESIGN.md, stream; the 2026-11-02 RGB-D entry in
  DECISIONS.md). A payload of one or two runs is framed as before.

## [0.11.0] - 2026-11-02

### Removed

- **`networking/dial.hpp`, a breaking move**: `dial` and `Listener` are
  the core's now, `core::net::dial` and `core::net::Listener` in
  `volumetric_kit/core/net/dial.hpp` (core 0.43.0, `volumetric_kit::core_net`),
  with the same arguments, results and messages. No consumer uses them
  yet: every consumer pins grid 0.8.0 or older (DESIGN.md, networking; the
  2026-11-02 net entry in DECISIONS.md).

### Changed

- **`grid_networking`'s sockets are the core's net tier**, which it links
  PRIVATE; grid's own socket calls are gone, and nothing under `src/` or
  `include/` includes a socket header (`tests/package_boundary.cmake`).
  `StreamServer` and `StreamClient` keep every declaration, option and
  message, and the wire is unchanged. The BBR note on Linux logs with
  source `"net"`, not `"vgrid"`.
- **The core floor is 0.43.0** (`VGRID_CORE_MIN_VERSION`), the first core
  with `core_net`, and the pin is core 0.43.0.
- **The test support's raw loopback sockets are the core's**
  (`core/testing/net_loopback.hpp`): `networking_loopback.hpp` names them in
  `vgrid_test::net` as before, its listener struct now `LoopbackListener`.

### Migration

- A consumer that calls `dial` or `Listener` includes
  `volumetric_kit/core/net/dial.hpp`, calls `core::net::dial` and
  `core::net::Listener`, and links `volumetric_kit::core_net`.
- A consumer that declares the core first pins core 0.43.0 or newer with
  this grid; an older one fails grid's configure (`vkc_require_core`).
- A test on grid's test support writes `net::LoopbackListener` for
  `net::Listener`.

## [0.10.0] - 2026-11-02

### Added

- **`volume::RayCaster`** (`volume/ray_caster.hpp`) casts K posed pinhole
  cameras into a grid's `tsdf` and `weight` in one dispatch: per pixel, the
  first observed surface's depth along the camera's +Z and the world unit
  normal there, into a stacked `ModelView` that later kernels read in place.
  Rays skip unallocated blocks whole, march free space in steps of the band,
  and refine a + to - crossing on the trilinear field. A `CastFilter` says
  which voxels answer: a positive, finite weight floor (at or above,
  `kObservedWeight` by default) and a 256-value allow set over a named 1-byte attribute; a
  trilinear sample answers only when its 8 corners do. The cameras may be
  written earlier in the batch, and a ring of `casts_per_batch` descriptor
  sets, every binding written at each cast, lets one batch recast at poses
  it moved (DESIGN.md, volume; the 2026-11-02 entry in DECISIONS.md).

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

### Fixed

- **On Linux a connection reset while it waited to be accepted is closed
  at accept**, as on macOS: Linux hands it out with its setup succeeding,
  and only naming its peer fails, so a server no longer admits it.

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
