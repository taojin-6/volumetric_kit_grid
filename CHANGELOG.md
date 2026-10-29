# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project aims to
follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
