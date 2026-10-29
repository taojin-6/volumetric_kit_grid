# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project aims to
follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.2] - 2026-10-29

### Changed

- **The `Encoder` holds each P-frame one quantization step against its
  reference**: a coefficient within one step of the reference's keeps the
  reference's value, so noise near a rounding boundary no longer flickers
  the decoded surface or costs P-frame bits; a larger change passes in the
  same frame. I-frames are quantized afresh. The format and the `Decoder`
  are unchanged; a P-frame decodes to within one step a coefficient of an
  I-frame of its grid (DESIGN.md, codec; the 2026-10-29 entry in
  DECISIONS.md).

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
