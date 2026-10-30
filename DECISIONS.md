# Locked decisions

The dated design record for `volumetric_kit_grid`: each rule's rationale,
measurements, review findings, and verification. Newest context wins.

[AGENTS.md](AGENTS.md) is the concise working guide; [DESIGN.md](DESIGN.md)
describes the implementation and detailed contracts. When a decision changes,
update its index entry and dated rationale here in the same commit, plus the
affected contract in DESIGN.md and any essential rule in AGENTS.md.

These entries governed grid's tiers while they were recon's, and moved here
with them, dates unchanged and text as it was but for a few words; a link to an entry recon kept goes to
[recon's record](https://github.com/taojin-6/volumetric_kit_recon/blob/main/DECISIONS.md#decision-index), which holds the rest of
the family's history and its measured lessons.

## Decision index

The index and dated entries are in the same chronological order. Read the
entries relevant to your task; later amendments supersede earlier rules.

- [**2026-06-21**](#2026-06-21--codec-ships-dct-only) —
  Codec ships DCT-only.
- [**2026-07-05**](#2026-07-05--shader-buffer-abi-is-scalar-block-layout-not-std430) —
  Shader buffer ABI is scalar block layout, not `std430`.
- [**2026-07-05**](#2026-07-05--per-voxel-storage-is-a-structure-of-arrays-attribute-store-voxelblockgrid-not-the-prior-engines-aos-voxel-in-the-hashmap) —
  Per-voxel storage is a structure-of-arrays attribute store
  (`VoxelBlockGrid`), not the prior engine's AoS `Voxel`-in-the-hashmap.
- [**2026-08-02**](#2026-08-02--meshvertex-is-the-renderers-vertex-layout) —
  `mesh::Vertex` *is* the renderer's vertex layout.
- [**2026-08-02**](#2026-08-02--the-mesh-arenas-extra-buffer-usage-is-declared-by-the-consumer-not-named-by-this-tier--and-usage-alone-does-not-reach-seam-b) —
  The mesh arena's extra buffer usage is declared by the *consumer*, not named
  by this tier — and usage alone does not reach seam B.
- [**2026-08-03**](#2026-08-03--the-mesh-arena-is-a-ring-of-slots-released-by-the-host-not-a-timeline-semaphore) —
  The mesh arena is a ring of slots released by the *host*, not a timeline
  semaphore.
- [**2026-08-03**](#2026-08-03--the-draw-command-is-written-by-the-kernel-that-counts-it-and-the-counters-unit-changes-from-triangles-to-indices) —
  The draw command is written by the kernel that counts it, and the counter's
  unit changes from triangles to indices.
- [**2026-08-03**](#2026-08-03--nothing-in-the-mesh-extractor-reads-the-current-slot-except-the-code-that-writes-it-the-capacity-plan-is-slot-independent-and-a-slot-is-marked-outstanding-only-where-a-devicemesh-is-handed-out) —
  Nothing in the mesh extractor reads "the current slot" except the code that
  writes it: the capacity plan is slot-independent, and a slot is marked
  outstanding only where a `DeviceMesh` is handed out.
- [**2026-08-08**](#2026-08-08--the-sparse-mesh-kernel-resolves-its-own-222-neighbourhood-by-probing-the-hash-table-on-device-and-mesh-therefore-reads-a-volume-buffer--a-coupling-paid-for-not-stumbled-into) —
  The sparse mesh kernel resolves its own 2×2×2 neighbourhood by probing the
  hash table on-device, and `mesh` therefore reads a `volume` buffer — a
  coupling paid for, not stumbled into. *Amended 2026-10-13:* the device
  shows the chained fixture spills; `HashDiagnostics` is gone. *Amended
  2026-10-22:* the extract splits its dispatch at the device's group limit.
- [**2026-08-08**](#2026-08-08--the-overflow-scan-stays-exhaustive-what-gets-bounded-is-its-cost-per-slot-not-its-length) —
  The overflow scan stays exhaustive; what gets bounded is its cost per slot,
  not its length. *Amended 2026-10-13:* `diagnostics()` is gone; the device
  confirms the chain fixtures. *Amended 2026-10-22:* the heap path refuses an
  overflow insert, unscanned, below an eighth of the blocks, and the retry
  loop ends at a capacity limit.
- [**2026-08-08**](#2026-08-08--in-block-vertex-sharing-is-a-second-compiled-kernel-not-a-branch-and-the-two-emitters-must-interpolate-an-edge-in-the-same-direction) —
  In-block vertex sharing is a *second compiled kernel*, not a branch, and the
  two emitters must interpolate an edge in the same direction.
- [**2026-08-09**](#2026-08-09--a-dirty-block-is-one-the-fuse-changed-the-flags-are-anchored-to-a-grid-the-library-checks-and-tracking-them-is-opt-in) —
  A dirty block is one the fuse *changed*, the flags are anchored to a grid the
  library checks, and tracking them is opt-in.
- [**2026-08-09**](#2026-08-09--incremental-mesh-extraction-is-worth-building-at-a-4x-ceiling-not-the-18x-one-window-suggested-and-the-worst-frame-rather-than-the-median-sizes-its-design-amends-the-dirty-block-decision-above) —
  Incremental mesh extraction is worth building at a ~4x ceiling, not the ~18x
  one window suggested, and the worst frame rather than the median sizes its
  design (amends the dirty-block decision above). It was removed on 2026-10-06.
- [**2026-08-11**](#2026-08-11--the-per-block-span-table-is-opt-in-is-retired-by-generation-rather-than-described-in-prose-is-anchored-per-block-slot-to-a-globally-unique-topology-token-and-is-one-table-for-the-whole-ring-rather-than-one-per-slot) —
  The per-block span table is opt-in, is retired by generation rather than
  described in prose, is anchored per block slot to a globally unique topology
  token, and is one table for the whole ring rather than one per slot. The table
  went on 2026-10-06.
- [**2026-08-11**](#2026-08-11--an-incremental-extract-trusts-one-struct-cleared-on-every-path-and-re-established-only-where-a-mesh-is-handed-out-every-refusal-is-a-silent-fallback-and-the-fallback-is-reported) —
  An incremental extract trusts one struct, cleared on every path and
  re-established only where a mesh is handed out; every refusal is a silent
  fallback, and the fallback is *reported*. Removed on 2026-10-06.
- [**2026-08-11**](#2026-08-11--incremental-extraction-runs-under-share_vertices-because-that-kernel-owns-its-index-run-and-so-retires-more-cheaply-not-less-reverses-the-share_vertices-clause-of-the-incremental-dispatch-decision-above) —
  Incremental extraction runs under `share_vertices`, because that kernel owns
  its index run and so retires *more* cheaply, not less (reverses the
  `share_vertices` clause of the incremental-dispatch decision above). Removed
  on 2026-10-06.
- [**2026-08-12**](#2026-08-12--meshing-a-cameras-view-is-a-caller-supplied-block-list-not-a-camera-the-mesh-tier-holds-and-it-stays-apart-from-incremental-extraction-rather-than-stacking-with-it) —
  Meshing a camera's view is a caller-supplied block list, not a camera the mesh
  tier holds; and it stays apart from incremental extraction rather than
  stacking with it. Incremental extraction went on 2026-10-06. *Amended
  2026-10-13:* the list is a `DeviceBlockList`; the host `BlockList` is gone.
- [**2026-08-31**](#2026-08-31--the-dense-extract-goes-extract-becomes-extract_host-so-the-two-workflows-are-named-rather-than-inferred) —
  The dense extract goes; `extract` becomes `extract_host`, so the two
  workflows are named rather than inferred.
- [**2026-08-31**](#2026-08-31--a-triangles-work-unit-is-the-candidate-block-not-the-triangle-and-the-band-it-allocates-is-measured-from-the-surface-not-dilated-from-a-point) —
  A triangle's work unit is the candidate *block*, not the triangle; and the
  band it allocates is measured from the surface, not dilated from a point.
- [**2026-09-26**](#2026-09-26--the-tsdf-codec-is-one-codec-tier-over-volume-with-separate-encoder-and-decoder-classes-a-geometry-only-intra-frame-of-per-block-dct-coefficients-an-observed-voxel-mask-and-sorted-block-coordinates-entropy-coded-by-chunked-static-table-rans) —
  The TSDF codec is one `codec` tier over `volume`, with separate `Encoder` and
  `Decoder` classes: a geometry-only intra frame of per-block DCT coefficients,
  an observed-voxel mask and sorted block coordinates, entropy-coded by chunked
  static-table rANS. *Amended 2026-10-13:* `check_block_list` is gone.
- [**2026-09-27**](#2026-09-27--the-v1-frame-is-independent-rans-segments-of-sorted-blocks-every-integer-a-class-plus-raw-bits-behind-fixed-per-frame-tables-and-a-section-table-the-decoders-end-check-is-for-consistency-not-integrity-amends-the-2026-09-26-entrys-interleaved-lanes) —
  The v1 frame is independent rANS segments of sorted blocks, every integer a
  class plus raw bits, behind fixed per-frame tables and a section table; the
  decoder's end check is for consistency, not integrity (amends the 2026-09-26
  entry's interleaved lanes). *Amended 2026-10-13:* a segment holds at most
  1,024 entries, a rule every writer and reader holds.
- [**2026-09-27**](#2026-09-27--encoder-and-decoder-are-the-codecs-public-api-encoding-drops-never-observed-blocks-and-sorts-the-rest-decoding-makes-a-callers-grid-hold-exactly-the-frame-by-diffing-its-block-set-everything-checkable-is-checked-before-the-grid-is-touched-and-a-grid-too-small-for-the-frame-is-refused-rather-than-grown) —
  `Encoder` and `Decoder` are the codec's public API: encoding drops
  never-observed blocks and sorts the rest, decoding makes a caller's grid hold
  exactly the frame by diffing its block set, everything checkable is checked
  before the grid is touched, and a grid too small for the frame is refused
  rather than grown. *Amended 2026-10-13:* the `Decoder` decodes P-frames
  too, against a reference it keeps. *Amended 2026-10-18:* a player's growth
  sits beside the decoder, as the `Fuser`'s does: `make_decode_grid` and
  `decode_growing`. *Amended 2026-10-21:* with `EncoderConfig::true_sdf`,
  observed blocks without surface are dropped too. *Amended 2026-10-22:*
  `make_decode_grid` and `decode_growing` take the caller's block ceiling,
  and refuse a frame or a grow past it as `InvalidArgument`.
  *Amended 2026-10-26:* the encoder takes a const grid.
- [**2026-09-27**](#2026-09-27--room0-sets-the-codecs-provisional-defaults-k--64-with-one-step-of-02-for-dc-and-ac-alike-the-coefficient-count-sets-the-quality-and-a-coarse-uniform-step-costs-almost-nothing-at-it-and-the-host-rans-coder-fits-a-frame-interval-at-1-cm-so-the-gpu-coder-waits) —
  Room0 sets the codec's provisional defaults, K = 64 with one step of 0.2 for
  DC and AC alike: the coefficient count sets the quality and a coarse uniform
  step costs almost nothing at it, and the host rANS coder fits a frame
  interval at 1 cm, so the GPU coder waits.
- [**2026-09-27**](#2026-09-27--quality-measurement-is-a-tier-of-its-own-eval-branching-off-mesh-mesh-to-mesh-accuracy-coverage-and-f-score-host-side-and-deterministic-and-production-infrastructure-rather-than-an-excluded-research-harness) —
  Quality measurement is a tier of its own, `eval`, branching off `mesh`:
  mesh-to-mesh accuracy, coverage and F-score, host-side and deterministic,
  and production infrastructure rather than an excluded research harness.
- [**2026-09-27**](#2026-09-27--a-mesh-becomes-a-tsdf-two-ways-in-the-tsdf-tier-signed-by-the-closest-triangles-normal-for-a-closed-mesh-or-as-an-unsigned-shell-for-any-mesh--binned-per-block-over-the-allocations-own-candidates-with-no-topology) —
  A mesh becomes a TSDF two ways, in the `tsdf` tier: signed by the closest
  triangle's normal, for a closed mesh, or as an unsigned shell, for any mesh
  — binned per block over the allocation's own candidates, with no topology.
  *Amended 2026-10-28:* in a tier of its own, `voxelize`.
- [**2026-09-30**](#2026-09-30--depth-allocation-works-a-16-x-16-pixel-tile-a-workgroup-and-dilates-each-distinct-block-of-the-tile-once-its-band-shared-out-over-the-lanes) —
  Depth allocation works a 16 x 16 pixel tile a workgroup and dilates each
  distinct block of the tile once, its band shared out over the lanes.
- [**2026-09-30**](#2026-09-30--with-the-spans-off-an-extracts-active-list-stays-on-the-device-and-the-map-hands-back-its-last-compaction-while-nothing-has-changed-since) —
  With the spans off, an extract's active list stays on the device, and the
  map hands back its last compaction while nothing has changed since. The spans
  went on 2026-10-06. *Amended 2026-10-24:* an extract records its own
  compaction into its batch when the last list does not hold, so only an
  extract after the encoder reuses one.
- [**2026-10-01**](#2026-10-01--every-block-slot-carries-stamps-ticks-of-one-clock-on-the-map-each-written-by-the-pass-that-knows-its-fact-the-grid-frees-the-blocks-no-allocation-has-asked-for-and-no-voxel-has-weighted-in-max_age-ticks) —
  Every block slot carries stamps, ticks of one clock on the map, each
  written by the pass that knows its fact; the grid frees the blocks no
  allocation has asked for and no voxel has weighted in `max_age` ticks.
- [**2026-10-01**](#2026-10-01--the-dirty-flags-become-the-changed-stamp-every-pass-that-writes-voxels-advances-the-maps-clock-and-stamps-what-it-changed-always-and-an-incremental-extract-keeps-the-tick-it-last-meshed-at-amends-the-2026-08-09-dirty-block-decision-and-the-stamps-entry-above) —
  The dirty flags become the `changed` stamp: every pass that writes voxels
  advances the map's clock and stamps what it changed, always, and an
  incremental extract keeps the tick it last meshed at (amends the
  2026-08-09 dirty-block decision and the stamps entry above). The incremental
  extract went on 2026-10-06; the stamp stays.
- [**2026-10-01**](#2026-10-01--the-codec-review-frame-v2-codes-a-partial-mask-a-plane-and-a-line-at-a-time-against-its-neighbours-and-a-sign-inside-its-mantissas-field-the-transform-finds-every-block-by-its-coordinate-one-commandbatch-a-call-over-device-memory-the-encoder-takes-the-maps-own-list-and-drops-never-observed-blocks-before-the-transform-and-the-decoder-stamps-only-what-it-changes-amends-the-changed-stamp-entry-above) —
  The codec review: frame v2 codes a partial mask a plane and a line at a
  time against its neighbours and a sign inside its mantissa's field; the
  transform finds every block by its coordinate, one `CommandBatch` a call
  over device memory; the encoder takes the map's own list and drops
  never-observed blocks before the transform; and the decoder stamps only
  what it changes (amends the `changed`-stamp entry above). *Amended
  2026-10-13:* the coordinate sort runs on the device too. *Amended
  2026-10-21:* with `EncoderConfig::true_sdf`, only the blocks holding
  surface are kept.
- [**2026-10-03**](#2026-10-03--garbage-collection-deletes-from-the-device-stale-list) —
  Garbage collection passes its device coordinate list directly to deletion,
  retaining host readback only for control counts. *Amended 2026-10-13:* it
  frees through the body `VoxelBlockGrid::remove` uses, which takes a device
  list alone.

- [**2026-10-02**](#2026-10-02--mesh-input-binning-keeps-counts-cursors-and-occupied-bins-on-the-device-retains-scratch-and-reads-only-validation-and-bounded-dispatch-control) —
  Mesh input binning keeps counts, cursors and occupied bins on the device,
  retains scratch, and reads only validation and bounded dispatch control.

- [**2026-10-03**](#2026-10-03--asset-loading-and-export-belong-to-io-with-assimp-as-an-optional-private-backend) —
  Asset loading and export move into the installed `io` tier; Assimp adds
  optional static mesh import, while dataset and normalization policy stay
  with consumers.

- [**2026-10-02**](#2026-10-02--per-basis-quantization-and-a-normalized-mesh-codec-fixture) —
  Per-basis quantization replaces the DC/AC split; frame v3 carries the kept
  bases' weights, and a normalized Rafa2 mesh joins room0 in the codec study.
  *Amended 2026-10-14:* the normalization's arithmetic is `io`'s opt-in
  helpers, and Rafa2's frames 1–150 are coded as a sequence.
- [**2026-10-03**](#2026-10-03--compact-the-codecs-observed-list-on-the-device-and-retain-transform-scratch) —
  Compact the codec's observed list on the device, predict its readback prefix
  to avoid an unconditional extra fence, and retain transform scratch.
  *Amended 2026-10-13:* the list stays on the device; only its count comes
  back.
- [**2026-10-03**](#2026-10-03--codec-mesh-evaluation-consumes-the-shared-io-loader) —
  The codec example uses Assimp-backed asset loading and shared PLY export;
  height normalization and topology checks remain example policy. *Amended
  2026-10-14:* they are opt-in `io` helpers; the example chooses the up
  vector and height, and codes a numbered range under one transform.
- [**2026-10-03**](#2026-10-03--rans-encoding-runs-on-the-device-byte-for-byte-the-hosts-frame-chosen-per-frame-by-segment-count) —
  rANS encoding runs on the device, byte for byte the host's frame, and the
  encoder picks the device per frame from 48 segments. *Amended 2026-10-13:*
  encoding runs on the device alone too, its sort and step layout included;
  the CPU coder is the tests' oracle.
- [**2026-10-03**](#2026-10-03--rans-decoding-runs-on-the-device-block-for-block-the-hosts-chosen-from-80-segments) —
  rANS decoding runs on the device, block for block the host's, in segments
  of at most 1,024 blocks, and the decoder picks the device per frame from 80
  segments. *Amended 2026-10-13:* decoding runs on the device alone, with no
  host fallback; the host reader is the tests' oracle.
- [**2026-10-06**](#2026-10-06--incremental-mesh-extraction-is-removed-the-changed-stamp-stays-and-the-mesher-no-longer-reads-it-amends-the-2026-08-09-2026-08-11-2026-08-12-2026-09-30-2026-10-01-and-2026-10-02-mesh-entries) —
  Incremental mesh extraction is removed; the `changed` stamp stays, and the
  mesher no longer reads it.
- [**2026-10-06**](#2026-10-06--marching-cubes-meshes-what-the-viewing-camera-sees-an-extract-over-the-device-frusta-list-the-viewers-re-meshing-on-a-fuse-or-a-view-change-and-a-culled-pass-records-its-density-amends-the-2026-08-12-view-meshing-entry-and-the-2026-10-06-incremental-removal-and-fusion-culling-entries) —
  Marching cubes meshes what the viewing camera sees: an extract over the
  device frusta list, the viewers re-meshing on a fuse or a view change, and
  a culled pass records its density. *Amended 2026-10-13:* the host
  `BlockList` it kept for the codec is gone.
- [**2026-10-08**](#2026-10-08--the-frames-models-and-block-grammar-are-one-glsl-text-that-the-host-compiles-as-c-and-golden-frames-pin-v3-amends-the-2026-10-03-encoding-entry) —
  The frame's models and block grammar are one GLSL text that the kernels
  include and the host compiles as C++, the device's walk runs in emit order,
  and golden frames pin v3's bytes on every machine.
- [**2026-10-08**](#2026-10-08--the-fuse-to-render-handoff-is-meshmeshexchange-one-gfx-free-library-type-owns-the-rings-release-a-parked-take-caps-the-mark-and-an-empty-mesh-draws-nothing-amends-the-2026-08-08-seam-b-and-2026-09-29-rig_viewer-entries) —
  The fuse-to-render handoff is `mesh::MeshExchange`: one gfx-free library
  type owns the ring's release, a parked take caps the mark, and an empty
  mesh draws nothing. *Amended 2026-10-19:* an empty mesh goes through
  `commit` with its payload. *Amended 2026-10-26:* `commit` may drop a mesh
  too.
- [**2026-10-10**](#2026-10-10--grid-validation-bounds-the-truncation-band-by-the-depth-kernels-integer-representation) —
  Grid validation rejects non-finite metrics and bounds the truncation band by
  the depth kernel's integer representation.
- [**2026-10-12**](#2026-10-12--recordings-are-vrst-files-of-our-own-verbatim-records-in-crc-framed-chunks-readable-from-a-killed-writer-in-recon_stream-on-the-cores-base-tier-alone-amends-the-2026-09-27-v1-frame-entry) —
  Recordings are `.vrst` files of our own: verbatim records in CRC-framed,
  8-byte-aligned chunks, readable from a killed writer, in `recon_stream` on
  the core's base tier alone. *Amended 2026-10-14:* a Geometry track's
  params may hold a mesh source's transform. *Amended 2026-10-21:* its
  chunks are made and read in memory too, publicly. *Amended 2026-10-30:*
  kind 2 is Audio, in version 0.
- [**2026-10-12**](#2026-10-12--p-frames-are-frame-type-1-of-v3-each-reference-block-gets-a-mode-in-update-segments-new-blocks-follow-in-intra-segments-and-a-reference-section-names-the-reference-by-xxh64-of-its-bytes-experimental-until-golden-p-frames-freeze-it) —
  P-frames are frame type 1 of v3: a mode per reference block in update
  segments, new blocks in intra segments, the reference named by XXH64 of its
  bytes; a missing reference is `NotFound`. Experimental until golden
  P-frames freeze it. *Amended 2026-10-13:* the `Decoder` decodes them; row
  13 is the device merge's, and the host merge is gone. *Amended
  2026-10-14:* the `Encoder` writes them. *Amended 2026-10-16:* golden
  P-frames pin type 1's bytes, but its freeze waits on a decision about
  interleaved rANS lanes. *Amended 2026-10-17:* no lanes: U defaults to R,
  and the freeze waits on the encode's count and a re-measure. *Amended
  2026-10-17:* frozen, so type 1's bytes change only on purpose.
- [**2026-10-13**](#2026-10-13--the-decoder-decodes-p-frames-against-the-frame-it-decoded-last-kept-on-the-device-as-decoded-in-two-slots-that-swap-only-when-a-decode-succeeds-one-integer-apply-makes-every-decoded-frame-intra-too-and-it-decodes-on-the-device-alone-amends-the-2026-09-27-encoderdecoder-and-2026-10-03-decoding-entries) —
  The `Decoder` decodes P-frames against the frame it decoded last, kept on
  the device as decoded and replaced only when a decode succeeds; one integer
  apply makes every decoded frame; and it decodes on the device alone.
  `FrameType` and `FrameInfo::type` are public. *Amended 2026-10-13:* the
  reference's coordinates live on the device, and the merge runs there.
  *Amended 2026-10-14:* the `Encoder` keeps its frames in the same store
  and writes P-frames. *Amended 2026-10-21:* the deblock, on request, joins
  the decode's last submit, and `rebuild` remakes a frame from the
  reference.
- [**2026-10-13**](#2026-10-13--the-codecs-coordinate-work-runs-on-the-device-the-encoder-sorts-its-blocks-with-a-stable-rank-merge-sort-on-coord_less-and-the-decoder-judges-merges-and-diffs-them-by-count-and-binary-search-never-sorting-only-counts-and-verdicts-cross-the-bus-amends-the-2026-10-13-decoder-2026-10-12-p-frame-2026-10-03-observed-list-and-garbage-collection-and-2026-10-01-codec-review-entries) —
  The codec's coordinate work runs on the device: the encoder sorts with a
  stable rank-merge sort, and the decoder judges, merges and diffs by binary
  search, never sorting; only counts and verdicts cross the bus. *Amended
  2026-10-13:* so does the writer's step layout, which leaves the codec's
  host only I/O and control; and the host `BlockList` it left without a
  caller goes, with the host-array `allocate` and `remove`. *Amended
  2026-10-14:* the encoder's split is its diff's own search, not a
  `BlockLists` pass. *Amended 2026-10-22:* the grid's zeroing rides the
  map's remove's first round, a fence fewer. *Amended 2026-10-24:* the
  verdicts no longer cross before the grid changes; a gate on the device
  judges them, and the counts are read once the frame's batch has run.
- [**2026-10-13**](#2026-10-13--block-lists-reach-the-volume-tier-on-the-device-alone-a-host-list-entry-point-that-only-tests-called-goes-amends-the-2026-08-08-sparse-mesh-and-overflow-scan-2026-08-12-view-meshing-2026-09-26-codec-2026-09-28-device-memory-2026-10-03-garbage-collection-2026-10-06-public-api-and-view-culling-and-2026-10-13-coordinate-entries) —
  Block lists reach the volume tier on the device alone; a host-list entry
  point that only tests called goes: the host `BlockList`, the host-array
  `allocate` and `remove`, and the table readbacks. *Amended 2026-10-17:*
  `HashEntry` and the fail slots are internal, the host sentinels go, and
  `hash_bucket` moves into the tests. *Amended 2026-10-28:* the entries
  accessor is internal too.
- [**2026-10-14**](#2026-10-14--the-encoder-writes-zero-motion-p-frames-against-the-frame-it-returned-last-kept-in-the-referencestore-as-its-forward-made-it-five-rules-make-an-i-frame-and-the-split-the-diff-and-a-static-price-for-zero-run-on-the-device-amends-the-2026-10-13-decoder-and-coordinate-and-2026-10-12-p-frame-entries) —
  The `Encoder` writes zero-motion P-frames at `intra_period` above 1,
  against the frame it returned last, kept in the `ReferenceStore` as its
  forward made it and moved only when `encode` returns one; five rules make
  an I-frame; the split, the diff and a static price for zero run on the
  device. *Amended 2026-10-16:* a sixth rule, an empty grid; U is
  `EncoderConfig::update_size`. *Amended 2026-10-17:* a P-frame is counted
  in the forward's submit. *Amended 2026-10-29:* a P-frame is held one
  quantization step against the reference before it is coded.
- [**2026-10-14**](#2026-10-14--mesh-normalization-and-the-topology-audit-are-opt-in-generic-io-helpers-for-a-mesh-sequence-bounds-measured-along-one-up-and-joined-over-frames-and-one-transform-for-every-frame-amends-the-2026-10-02-normalized-mesh-and-2026-10-03-codec-mesh-entries) —
  Mesh normalization and the topology audit are opt-in, generic `io`
  helpers for a mesh sequence: bounds measured along one up and joined over
  frames, and one transform applied to every frame. `codec_mesh` uses them,
  and its single frame keeps its bytes.
- [**2026-10-14**](#2026-10-14--codec_mesh-codes-a-numbered-mesh-sequence-under-one-transform-a-frame-number-in-sourcepath-an-exact-range-one-transform-fitted-to-every-frame-and-stored-in-the-stream-a-cleared-grid-and-one-record-per-frame-and-sourceallow_holes-amends-the-2026-10-02-normalized-mesh-2026-10-03-codec-mesh-2026-10-11-examples-and-2026-10-12-container-entries) —
  `codec_mesh` codes a numbered mesh sequence: a printf-style frame number in
  `source.path`, `run.first` and an exact `run.frames`, one transform fitted
  to every frame and stored in the Geometry track's params, a cleared grid
  and one record per frame, `source.allow_holes`, and its stream made before
  any work. Frame 150's conversion outlier is mostly the closest-face sign
  at a sliver fold. *Amended 2026-10-15:* require core 0.8.0's directory
  escaping, and refuse example intra periods other than 1 until C5b.
  *Amended 2026-10-16:* the examples code P-frames, on core 0.9.0.
  *Amended 2026-10-21:* it codes a true SDF and refuses a band too narrow
  for one; the templates convert in 16 voxels.
- [**2026-10-16**](#2026-10-16--the-codec-examples-code-p-frames-codecintra_period-and-codecupdate_size-reach-the-encoder-and-a-streams-rows-report-and-keys-go-by-frame-type-an-empty-grid-is-an-i-frame-golden-p-frames-pin-type-1s-bytes-and-its-freeze-waits-since-at-r--16-its-p-frames-code-slower-than-intra-amends-the-2026-10-12-p-frame-2026-10-14-encoder-and-codec-mesh-entries) —
  The codec examples code P-frames: `codec.intra_period` and
  `codec.update_size` reach the `Encoder`, a stream's rows, report and keys
  go by frame type, and a failure after the encode forces an I-frame. An
  empty grid is an I-frame, U is `EncoderConfig::update_size` (64,
  measured), and golden P-frames pin type 1's bytes. Type 1 stays
  experimental and the templates at period 1: at R = 16 its P-frames code
  slower than intra, and the freeze waits on a decision about interleaved
  rANS lanes. *Amended 2026-10-17:* U defaults to the segment size; no lanes,
  and the freeze waits on the encode's count and a re-measure. *Amended
  2026-10-17:* type 1 is frozen, though P encode at R = 16 stays slower than
  intra on the host, and the templates code at period 30.
- [**2026-10-17**](#2026-10-17--a-p-frames-update-segments-default-to-its-segment-size-encoderconfigupdate_size-0-the-default-takes-segment_size-resolved-once-in-encodercreate-the-format-and-the-goldens-are-unchanged-amends-the-2026-10-16-codec-examples-2026-10-14-encoder-and-2026-10-12-p-frame-entries) —
  A P-frame's update segments default to its segment size:
  `EncoderConfig::update_size` 0, the default, takes `segment_size`
  (`resolved_update_size()`, resolved once in `Encoder::create`); the format
  and the goldens are unchanged. No lanes: the freeze waits on the encode's
  count moving into the forward's submit, and a re-measure. *Amended
  2026-10-17:* the re-measure ran, and type 1 is frozen (the freeze entry).
- [**2026-10-17**](#2026-10-17--a-p-frame-is-counted-in-the-forwards-submit-its-count-and-step-layout-run-over-m-plus-the-frames-blocks-the-device-takes-off-the-diffs-survivors-and-an-encode-waits-on-one-fence-fewer-amends-the-2026-10-14-encoder-entry) —
  A P-frame is counted in the forward's submit: its count and step layout
  run over M plus the frame's blocks, the device takes off the diff's
  survivors, and an encode waits on one fence fewer (amends the 2026-10-14
  encoder entry).
- [**2026-10-17**](#2026-10-17--frame-type-1-is-frozen-so-its-bytes-change-only-on-purpose-and-the-codec-templates-code-at-period-30-p-frames-code-no-slower-than-intra-on-room0-at-1-cm-and-on-rafa2-at-r--64-and-at-r--16-decode-no-slower-but-encode-up-to-4-slower-on-the-host-alone-amends-the-2026-10-12-p-frame-2026-10-16-codec-examples-and-2026-10-17-update-size-and-count-entries) —
  Frame type 1 is frozen, so its bytes change only on purpose, and
  `codec_replica.json` and `codec_mesh_sequence.json` code at period 30.
  The re-measure missed one comparison, Rafa2's encode at R = 16, where
  P-frames take up to 4% longer than intra on the host alone; the user
  froze type 1 anyway, and trimming that overhead is next.
- [**2026-10-21**](#2026-10-21--the-vrst-chunks-in-memory-are-public-chunkencoder-frames-what-streamwriter-writes-by-the-record-rules-it-now-holds-and-chunkparser-reads-bytes-as-they-arrive-through-streamreaders-checks-amends-the-2026-10-12-container-entry) —
  The `.vrst` chunks in memory are public (`stream/chunks.hpp`), so a second
  tier, networking next, writes and reads the format without stream's
  detail: `ChunkEncoder` frames what `StreamWriter` writes and holds the
  record rules; `ChunkParser` copies bytes as they arrive and checks each
  chunk with the functions `StreamReader` uses, its header before waiting
  for its body; a failure is final.
- [**2026-10-21**](#2026-10-21--a-session-is-served-live-over-tcp-by-a-tier-of-its-own-recon_networking-the-wire-is-the-file-newest-key-wins-and-a-viewer-joins-at-the-cached-gop) —
  A session is served live over TCP by a tier of its own,
  `recon_networking` (the owner's: sockets apart from the format, testable
  alone). The wire is the `.vrst` file's bytes; a viewer behind skips to a
  newer key, only at a chunk boundary, so no P-frame goes without its
  references; a viewer joins at the cached GOP; `publish` never waits; a
  viewer types the sender's address. *Amended 2026-10-25:* a live sender
  serves keys on demand instead (`replay_gop` false), and the cached GOP
  stays the default.
- [**2026-10-21**](#2026-10-21--meshintegrator-finds-each-voxels-nearest-triangle-with-an-exact-early-exit-a-triangle-is-ruled-out-only-by-a-lower-bound-less-a-rounding-margin-of-its-own-strictly-above-the-voxels-limit) —
  `MeshIntegrator` finds each voxel's nearest triangle with an exact early
  exit: a workgroup per block sorts its bin by a lower bound, and a triangle
  is ruled out only by a bound, less a rounding margin of its own, strictly
  above the voxel's limit, so every voxel writes what its whole bin writes.
  Blocks past 16^3 voxels are refused.
- [**2026-10-21**](#2026-10-21--a-true-signed-distance-codes-without-the-observed-mask-encoderconfigtrue_sdf-cuts-each-corner-farther-than-3-voxels-from-zero-of-a-cell-whose-sign-changes-and-codes-only-the-blocks-that-hold-surface-whole-codec_mesh-alone-sets-it-in-a-band-of-at-least-15-voxels-so-fused-depth-keeps-its-mask-and-the-mesh-templates-convert-in-16-voxels-and-code-at-k-56-amends-the-2026-09-27-encoder-2026-10-01-codec-review-and-2026-10-14-codec-mesh-entries) —
  A true signed distance codes without the observed mask:
  `EncoderConfig::true_sdf` cuts each corner farther than √3 voxels from
  zero of a cell whose sign changes, and codes only the blocks that hold
  surface. `codec_mesh` alone sets it, in code, so multi-view RGB-D capture
  keeps its mask, byte for byte. The mesh templates convert in 16 voxels
  and code at K 56, about 2 Mbit/s on Rafa2; the format, the decoder and
  the goldens are unchanged.
- [**2026-10-21**](#2026-10-21--the-decoder-deblocks-on-request-a-runtime-switch-off-by-default-runs-the-studys-step-keyed-line-fit-and-gaussian-on-the-device-in-the-decodes-last-submit-keeping-each-block-with-an-unobserved-voxel-as-decoded-and-a-paused-player-rebuilds-its-frame-from-the-reference-amends-the-2026-10-13-decoder-entry) —
  The decoder deblocks on request: `Decoder::set_deblock`, off by default
  and read at each decode, runs the study's step-keyed line fit and
  Gaussian on the device in the decode's last submit, its filter picked
  from each frame's header; `Decoder::rebuild` remakes the last frame from
  the reference with the switch as it is, so a paused player shows a
  toggle. Streams are unchanged. *Amended 2026-10-22:* a frame with an
  unobserved voxel was skipped: on a live room capture the filter erodes
  surface at mask edges and in thin parts. *Amended 2026-10-23:* no frame
  is skipped; a block with an unobserved voxel keeps its exact decode, and
  a face is fitted only between two fully observed blocks, so a true-SDF
  frame is smoothed but for its cut blocks and a live frame, masked in
  nearly every block, decodes nearly exactly; `DeblockInfo` counts the
  blocks kept.
- [**2026-10-22**](#2026-10-22--streamclient-reports-progress-before-a-whole-record-is-ready) —
  `StreamClient::next` can report partial input and connection progress,
  allowing a consumer to drain it before waiting on its own controls.
- [**2026-10-24**](#2026-10-24--a-frame-decodes-and-meshes-in-the-callers-batch-every-count-stays-on-the-device-the-maps-rounds-k--3-the-extract-and-the-decoder-gather-their-results-into-ring-slots-and-the-blocking-calls-are-one-submit-amends-the-2026-09-28-device-memory-2026-10-13-coordinate-and-2026-10-23-breakdown-entries) —
  A frame decodes and meshes in the caller's batch: the map's removal and
  allocation run K = 3 device rounds, the extract sizes itself on the
  device, and the decoder judges rows 11 to 13 there; each gathers its
  results into a ring slot read after the batch. The blocking calls are
  record, one submit and finish.
- [**2026-10-25**](#2026-10-25--a-live-sender-makes-keys-on-demand-a-joiner-waits-for-the-next-key-the-server-keeps-what-its-viewers-still-need-and-a-viewer-that-falls-max_gop_bytes-behind-waits-for-a-key-alone-amends-the-2026-10-21-networking-entry) —
  Keys on demand: without `ServerOptions::replay_gop`, a joiner waits for
  the next key, the server keeps only what its viewers still need, and a
  viewer that falls `max_gop_bytes` behind waits for a key, alone; newest
  key wins still. `StreamServer::key_wanted()` counts both, and the codec
  examples' `CodecStream` codes an I-frame at its next front half when it
  rises. fuse_orbbec serves so; `serve_stream`, which cannot make a key,
  replays the GOP. *Amended 2026-10-26:* that stream is
  `sender::CodecStream`.
- [**2026-10-26**](#2026-10-26--meshexchanges-consumer-may-drop-a-mesh-it-took-a-third-outcome-rather-than-a-deadline-in-the-exchange-and-a-ring-slot-more-for-each-mesh-dropped-before-the-next-commit-amends-the-2026-10-08-meshexchange-entry) —
  `MeshExchange`'s commit may drop a mesh it took (`CommitDecision::kDrop`):
  it never becomes live, the next frame takes the newest mesh, and its slot
  goes with the generations below the next commit, so a ring needs a slot
  more for each mesh dropped before a commit. A third outcome, with no
  deadline or policy in the exchange. The ring's size counts one frame in
  flight as two.
- [**2026-10-26**](#2026-10-26--a-compaction-records-from-a-const-map-and-the-encoder-takes-a-const-grid-the-maps-compaction-output-is-scratch-any-compaction-rewrites-amends-the-2026-09-27-encoder-and-decoder-entry) —
  `VoxelHashMap::record_compaction` is `const`, so a caller holding a grid
  read-only records the active set's compaction, or a frustum's, into its
  own batch. It moves no block, stamp, heap count, epoch or tick; the list,
  its count, its kind's serial and the frusta are the map's compaction
  output, `mutable`, and a held list of that kind goes stale as before.
  `Encoder::encode` and `record_front` take a const grid. One call on a map
  at a time, const or not, and no resize or frusta grow until the batch has
  resolved; broken, a resize refuses the batch and a frusta grow is
  undefined behaviour.
- [**2026-10-28**](#2026-10-28--the-voxel-layer-is-camera-free-volume-owns-its-depth-camera-depthpinhole-and-the-one-line-of-unprojection-it-needs-tsdf-copies-its-camera-in-amends-the-2026-10-16-glsl-pinhole-entry) —
  The voxel layer is camera-free: `volume::DepthPinhole`, the fields and
  96 bytes of `DepthCameraParams`, is what depth allocation takes, and its
  kernel unprojects through a copy of the pinhole's one line; tsdf's
  `FrameInput` keeps `DepthCameraParams` and `to_depth_pinhole` copies it
  field by field. Allocation is main's, bit for bit.
- [**2026-10-28**](#2026-10-28--a-triangle-meshs-distance-field-is-a-tier-of-its-own-voxelize-on-volume-alone-amends-the-2026-09-27-mesh-to-tsdf-entry) —
  A triangle mesh's distance field is a tier of its own, `voxelize`
  (`recon_voxelize`), on `volume` alone: `MeshIntegrator` and its kernels
  leave `tsdf`, and the closest-point GLSL moves to `volume`.
- [**2026-10-28**](#2026-10-28--the-codec-stream-is-a-tier-of-its-own-codec_stream-on-codec-and-networking-and-its-frame-in-halves-is-public-amends-the-2026-10-26-sender-entry) —
  The codec stream is a tier of its own, `codec_stream`
  (`recon_codec_stream`), on `codec` and `networking`: `CodecStream`,
  `CodecOutputs` and `SentFrame` leave `recon_sender`, which keeps
  `SetSender` and `fuse_inputs`. The frame in halves is the stream's public
  calls (`record_front`, `front_ran`, `redo`, `finish_frame`), `SetSender`
  codes through them, and `coder()` is gone.
- [**2026-10-28**](#2026-10-28--the-voxel-grids-gpu-contract-has-one-home-volume-exports-grid_typesglsl-and-voxel_colorglsl-names-its-shared-attributes-and-keeps-the-hash-entries-internal-amends-the-2026-08-02-color-space-and-2026-10-13-block-list-entries) —
  The voxel grid's GPU contract has one home: `recon_volume` exports
  `grid_types.glsl` (the grid's structs, the in-block index, `vrTickAfter`)
  and `voxel_color.glsl` (the colour attribute's format), tsdf includes
  only those, `kTsdfAttribute` / `kWeightAttribute` / `kColorAttribute`
  name the shared attributes, and the hash entries' accessor is internal.
  Bit-identical to main.
- [**2026-10-28**](#2026-10-28--grid-is-a-package-of-its-own-the-voxel-grid-and-what-reads-it-under-grids-names-on-the-core-alone) —
  grid is a package of its own: the voxel grid and what reads it, under
  grid's names, on the core alone. It is now `volumetric_kit_grid`; recon
  builds on it.
- [**2026-10-29**](#2026-10-29--the-encoder-holds-each-p-frame-one-quantization-step-against-its-reference-i-frames-are-quantized-afresh-amends-the-2026-10-14-encoder-entry) —
  The `Encoder` holds each P-frame one quantization step against its
  reference: a coefficient within one step of the reference's,
  |q − q_ref| ≤ 1, keeps the reference's value, always, on the device;
  I-frames are quantized afresh, since holding them too lets the held
  values drift; they drift within a period too, so the intra period bounds
  the drift. Format and decoder unchanged.
- [**2026-10-30**](#2026-10-30--an-audio-track-is-kind-2-of-vrst-v0-interleaved-samples-as-its-8-params-bytes-say-every-record-a-key-at-its-first-samples-capture-time-no-version-change-and-no-new-record-flag-amends-the-2026-10-12-container-entry) —
  An Audio track is kind 2 of `.vrst` v0: interleaved samples as its 8
  params bytes say (`stream/audio.hpp`: codec, channels, rate; PCM s16
  first), every record a key at its first sample's capture time. No version
  change, as readers pass an unknown kind through, and no new record flag;
  a file without audio is byte for byte as before.

## Decision record

### 2026-06-21 — Codec ships DCT-only.

KLT (trained basis) and the iQuantizer
refinement are excluded as research.

### 2026-07-05 — Shader buffer ABI is scalar block layout, not `std430`.

The
host POD structs (`HashEntry`, …) embed `Vec3i` voxel-block coords, which
`std430` 16-byte-aligns — so a naive `std430` shader mirror does *not* match the
packed C/CUDA layout the host uploads (`HashEntry`: host `pos` at 4 in 20 B;
`std430` `pos` at 16 in 32 B). The compute shaders therefore declare
`layout(scalar)` (`GL_EXT_scalar_block_layout`; core in Vulkan 1.2, supported by
MoltenVK), under which the GLSL struct is byte-identical to the host struct for
every vector-bearing type — no per-field scalarization, one POD definition
across CPU / GLSL / CUDA. *Refines* the 2026-07-04 GLM decision
("`std430`-compatible" was imprecise for `vec3`-bearing structs); the host-side
`static_assert`s guard only the host packing.

### 2026-07-05 — Per-voxel storage is a structure-of-arrays attribute store (`VoxelBlockGrid`), not the prior engine's AoS `Voxel`-in-the-hashmap.

Following Open3D's `VoxelBlockGrid`, `VoxelHashMap` keys only a block *index*;
each per-voxel channel (`tsdf`, `weight`, `color`, …) is its own flat device
array of `num_blocks·voxels_per_block` elements, keyed by `BlockIndex::ptr`,
declared and allocated independently. *Diverges* from the monolithic prior
engine (which bundled a single AoS `Voxel{sdf,weight}` buffer inside the
hash-table struct): the win is à-la-carte + tiered — a `volume`-only user
(pure spatial hashing) allocates **zero** per-voxel memory, and each consumer
materialises only the channels it needs (TSDF writes `tsdf`+`weight`, a colour
pass writes `color`, meshing reads `tsdf`) instead of forcing the full SDF
volume (~6 GB at prod defaults) onto every map. The store lives in `volume`
(below every consumer tier); the AoS `Voxel`/`VoxelData` PODs in
`hash_types.hpp` become host-side read views.

### 2026-08-02 — `mesh::Vertex` *is* the renderer's vertex layout.

recon's
vertex was `{position, normal, color, uv0}` (48 B) and gfx's is
`{position, normal, tangent, uv0, color}` (64 B), so every mesh crossing the
seam was rebuilt field by field. recon now emits gfx's layout directly:
`mesh::Vertex` is byte-for-byte `gfx::assets::Vertex`, pinned by
`static_assert`s on both sides of the seam (`mesh/mesh.hpp` and the bridge),
and the marching-cubes kernels + the texturing kernel declare the matching
`layout(scalar)` mirror. The host converter collapses to a bulk copy, and —
the actual point — the bytes the kernel writes are **already in the shape the
renderer binds**, which is what interop seam B needs. gfx's vertex-input
description reads position/normal/uv0/color at exactly these offsets with this
stride; it does not bind `tangent` at all. *Necessary, not yet sufficient*:
the arena also has to be **bindable** (the usage bits — settled by the
2026-08-02 `MarchingCubesConfig` decision below) and then actually
**drawable** (lifetime, sharing mode, barrier scope, indirect counter — the
seam-B `TODO(mesh)` on that struct).
**The cost is paid knowingly, and it is not free.** Every vertex grows 48 → 64
bytes (+33%) for a `tangent` slot marching cubes cannot produce — it has no
surface parameterisation to derive one from, so the kernel writes the same
`(1, 0, 0, 1)` placeholder the host converter used to synthesize. A
recon-only consumer (`fuse_replica`, the codec tiers, a headless exporter)
pays that for a field it never reads: measured on the 400-frame room0,
**peak memory +11% and throughput −10%**, against a converter saving that
only a gfx-linked consumer sees. It also moves a limit: the arena is
`capacity · 3 · sizeof(Vertex)`, so the triangle count that fits under the
`maxStorageBufferRange` guard drops by a quarter. The
alternative — teaching gfx a
vertex-input variant for recon's tighter layout — keeps recon's bytes but
needs a change in the sibling repo plus a pin bump here, and leaves the
seam-B buffer still un-bindable without one. We chose the renderer's
convention because the seam is the point: the whole reason `uv0` and `color`
live on this vertex at all is the 2026-07-06 hybrid-colour path. Revisit if a
non-renderer consumer ever dominates the mesh tier's traffic — the tangent is
then 16 bytes of dead weight per vertex with no offsetting win.
(*2026-08-02:* that first consumer has arrived — the shared-device bootstrap
below — and the usage half is now settled by the `MarchingCubesConfig`
decision below.)

### 2026-08-02 — The mesh arena's extra buffer usage is declared by the *consumer*, not named by this tier — and usage alone does not reach seam B.

`MarchingCubes::create` takes a `MarchingCubesConfig{extra_vertex_usage,
extra_index_usage}`, OR-ed onto `STORAGE_BUFFER` for the vertex arena and the
index run on **every** grow, not just the first. The alternative — hardcoding
`VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | INDEX_BUFFER` in the `mesh` tier — was
rejected because it makes this tier guess at a sibling's vertex-input
requirements that it is deliberately not compiled against; the app knows both
and passes the union, the same shape as the create/adopt device seam. A
recon-only consumer (`fuse_replica`, the codec tiers, an exporter) defaults to
zero and its buffers are bit-identical to before.
**The verification half is what makes it a seam rather than a hope.** Vulkan
cannot be asked what a `VkBuffer` was created with, so `core::Buffer` records
its `usage()` (the reason `AdoptedDevice` carries its enabled extension list),
and `DeviceMesh` carries `vertex_usage` / `index_usage` — a consumer *checks*
that the binding it is about to make is permitted instead of trusting that
the flags it published reached the producer. Without it the app restates the
same flags independently at both ends and a mismatch is a validation-layer-only
diagnostic: undefined behaviour with layers off, which is the shipping
configuration and the only one on iOS. It is also what lets a test assert the
config *landed* — `marching_cubes_config_test.cpp` extracts through a
configured extractor and reads the bits back off the `DeviceMesh`, so dropping
`config_` from the grow path fails rather than passing silently for want of an
in-tree consumer.
**Bindable is not drawable, and the gap is recorded rather than implied.**
Review of the first cut found prose asserting that the usage bit was "the
whole of what stands between it and binding this arena in place." It is not.
Four things a create-time flag cannot supply remain, enumerated in a greppable
seam-B `TODO(mesh)` on the struct: (1) **lifetime** — one grow-only arena,
reused in place and freed *synchronously* on grow (`Buffer::operator=` runs
`vmaDestroyBuffer` with no fence), so the next extract overwrites or frees
memory an in-flight draw is reading; `DeviceMesh::generation` guards only the
host `download()` path, and the fix is the ring of slots + timeline semaphore
DESIGN.md's seam B already specifies. (*Superseded by the 2026-08-03 output-
ring decision below*, which takes the ring and deliberately declines the
semaphore.) (2) **sharing** —
`Allocator::create_buffer` hardcodes `VK_SHARING_MODE_EXCLUSIVE` and
`BufferDesc` has no knob, while on Apple recon and gfx sit on *different queue
families* (the bootstrap decision above), so a cross-family read needs
`CONCURRENT` or a release/acquire pair — and Metal has no ownership concept,
so this appears to work on the only platform CI runs. (3) **visibility** —
`core`'s shared `dispatch()` barrier reaches `COMPUTE_SHADER|HOST`, never
`VERTEX_INPUT`/`VERTEX_ATTRIBUTE_READ`/`INDEX_READ`; widening it was weighed
and declined, since every tier's dispatch would pay a broader scope for one
consumer, and a cross-queue handoff needs a semaphore regardless.
(*Both (2) and (3) are superseded by the 2026-08-03 decision below*, which
reverses that declining: the scope is a destination mask, so it costs an
execution dependency the driver already owed — the real cost turned out to be
a capability check, not a performance one.) (4)
**indirect draw** — `counter_` carries no `INDIRECT_BUFFER` usage and is not
on `DeviceMesh`, so the `vkCmdDrawIndexedIndirect` path seam B is specified
around cannot be expressed and the count still round-trips to the host.
None are implemented here: each needs a consumer to verify against, and
building them blind is how a two-field config grows four more fields nothing
passes. `create` does reject `SHADER_DEVICE_ADDRESS` (this repo's `Device`
never enables `bufferDeviceAddress`, so VMA asserts in debug and silently
drops the allocate flag in NDEBUG) — validated where the caller supplies it,
not inside the first extract's arena grow.

### 2026-08-03 — The mesh arena is a ring of slots released by the *host*, not a timeline semaphore.

Seam-B blocker (1) lifetime. `MarchingCubesConfig`
grows `slot_count`: one (the default) is byte-identical to the old single
grow-only arena, and more gives each outstanding extract its own arena and
index run. The consumer calls `MarchingCubes::release_through(generation)` as
its frames retire, and an extract only ever writes, grows, or **frees** a slot
that has been released — which is what makes the tier's existing synchronous
`vmaDestroyBuffer` (no fence wait; its own work is fence-blocked inside
`submit_single_time`) safe again for an external reader.
**This takes the ring DESIGN.md's seam B specifies and deliberately declines
its timeline semaphore.** A cross-library GPU wait is the one thing the
shared-queue arrangement forbids: a command buffer waiting on a value the
sibling has not signalled deadlocks against a swapchain rebuild, which drains
the queue while holding the submit mutex. Reporting completion *after the
fact* cannot deadlock, needs no fence queue in the library, and costs a
`std::uint64_t` store. The price is that the contract is now the consumer's to
honour: extracting with every slot outstanding is **refused**, not silently
serviced by overwriting a live draw.
Three things this got wrong before review caught them, all in the same place —
*the refusal path must cost the caller nothing*. (Not a closed set: six more
of the same family surfaced once a ring actually ran on device, including a
capacity plan that compounded 1.5× per extract. See the 2026-08-03
slot-independence decision below, which also moves the refusal above the
active-set compact — this decision's own "costs nothing" was still paying a
GPU round trip.) (1) The slot claim ran inside
`ensure_output_buffers`, i.e. **after** `++generation_`, so a refused extract —
the path whose entire purpose is to protect a slot the consumer is still
reading — advanced the generation and made `download()` retire the very mesh
it had just protected. Verified by probe before the fix (`download(gen2)` OK
before the refused extract, REFUSED after) and now pinned by a test that a
revert fails. The claim is its own `claim_output_slot()`, called once at the
top of each extract and before the bump; that also makes "once per extract,
not once per call" structural rather than conditional on a
generation-equality trick, which matters because a call that plans its
capacity low re-enters `ensure_output_buffers` to refit. (2) `release_through`
is **not atomic and not internally synchronized**, while the consumer it is
written for is on another thread — `fuse_viewer` fuses on a background thread
and retires frames on the render thread. Documented as a caller obligation
rather than made `std::atomic`, because an atomic is not movable and this
class's rule-of-zero defaulted moves are load-bearing; the consumer already
holds a mutex to hand the `DeviceMesh` across, and a `uint64_t` store under it
is free. (3) `download()` reads the *current* slot unqualified, which is
correct only because the generation check and the slot claim together make
"current generation" and "current slot" the same statement — the comment
there had been justifying the check with the single-arena argument, which a
ring **inverts** (a superseded view now names a *different* buffer). Widening
`download` to older generations therefore means indexing `slots_` by
generation, not relaxing a comparison.
Two further notes carried knowingly. The slots are a **fixed array, not a
`std::vector`**: this class promises a self-move leaves it intact and the
sparse test exercises `mc = std::move(mc)` directly, but `std::vector`'s
self-move-assignment is valid-but-unspecified and libc++ empties it, so the
extractor passed `valid()` and then indexed nothing (it segfaulted on the
first run). And the empty-mesh early return used to be the one path that
advanced the generation **without** claiming a slot, on the reasoning that an
empty mesh owns no bytes to protect. *Corrected by the decision below*: that
is true of the empty mesh and false of the **ring**, because
`released_through_` is a high-water mark — a consumer that retires the empty
generation (instantly, since it draws nothing) thereby marks every older slot
released too, including the one holding the mesh it is still drawing, which is
exactly what a viewer does on an empty frame. The empty path now claims and
stamps a slot like any other extract, so "every generation lives in exactly
one slot" **is** total. Still open for seam B at the time of writing: blocker
(4), the indirect command on `DeviceMesh`.

### 2026-08-03 — The draw command is written by the kernel that counts it, and the counter's unit changes from triangles to indices.

Seam-B blocker (4),
the last one. The marching-cubes append atomic no longer bumps a bare
`uint` triangle counter; it bumps `indexCount` of a real
`VkDrawIndexedIndirectCommand` by `kIndicesPerTriangle` per triangle, so the
bytes the kernel writes *are* the command `vkCmdDrawIndexedIndirect` reads.
The buffer carries `INDIRECT_BUFFER` beside the `STORAGE_BUFFER` the kernel
counts through, it is per **slot** rather than per extractor (it is part of
the mesh — a renderer reading slot N's command while N+1 is extracted is the
whole point of the ring), and `DeviceMesh` publishes it. The count still
round-trips to the host, but only because *this tier* needs it to refit an
undersized arena; a consumer no longer does, which is what closes the blocker.
**Changing what an atomic counts moves every site that reads it, and the first
cut moved only some.** The three that mattered, all found by review: (1) the
`uint32` overflow guard still bounded *triangles* while the counter had become
*indices*, leaving it **3× too loose** — and worse than a plain wrap, since
`2^32 ≡ 1 (mod 3)` desynchronises the `/3` that recovers the slot index, so a
wrapped dispatch overwrites triangles it already emitted and returns
`Status::ok` with a corrupt surface. (2) The safety clamp that bounds
`indexCount` by the arena ran only on the success path — where the loop's own
exit condition already makes it a **provable no-op** — and was skipped on the
two failure returns that genuinely leave the command naming more indices than
exist. The guarantee was exactly inverted; it now lives on the failure paths,
where at `slot_count == 1` an outstanding `DeviceMesh` names that very buffer.
(3) The GLSL contract sentence ("`index_count` always ends as the field's true
total") became false in the same edit, and it is not decoration — it is the
specification the refit protocol and the `continue`-not-`return` fix rest on,
and the CUDA port is required to stay numerically in lockstep with it. The
open-coded `3` is now `kIndicesPerTriangle`, named on both sides of the ABI,
because it is the conversion between the two units this tier trades in.
**Two adjacent bugs the same review turned up, neither about the command.**
The host `extract` overloads claim and stamp a ring slot but return
`Result<Mesh>` — no generation — and `download` does not release, so a
consumer at `slot_count > 1` exhausted the ring in `slot_count` calls and then
failed **permanently**, quoting generations the API never handed out. They now
release their own slot once the host copy is taken, on success *and* failure,
which keeps a host-only caller off the release contract entirely. And the
empty-extract path is corrected as described above.
**Verification is the point, and it was measured rather than assumed.**
Nothing pinned the command's contents: mutating `emitted * 3` to `emitted`
passed every mesh and texture test, and reverting the atomic to count
triangles passed the *sparse* suite — because dense and sparse share
`mcEmitCell`, so a unit error moves both sides of the equivalence equally. The
suite now reads the command back through a real `vkCmdCopyBuffer` (which makes
the test a genuine consumer of `extra_indirect_usage` — the readback and the
flag prove each other) and asserts `indexCount == triangle_count · 3` plus the
four fields that make it *drawable*; and the sparse test pins the counter's
**units** against the fixture's analytic surface area (`4πr²`), which no
counter participates in, so a 3× undercount fails by a mile. Both mutations
now fail. `DeviceMesh::valid()` covers all three buffers rather than two.
**Sharing and placement, the two remaining ways to get this wrong quietly.**
`MarchingCubesConfig` grows `queue_families` (held **by value** — the config
is stored and re-read on every arena grow, so a `BufferDesc`-style pointer
would dangle) applying to arena, index run *and* command alike, since a
renderer reads all three; `DeviceMesh` publishes the resulting
`sharing_mode`, which matters more than the usage bits it sits beside because
reading an `EXCLUSIVE` buffer from a non-owning family is *undefined* where a
missing usage bit is at least a validation diagnostic. Memory placement is
**recorded rather than fixed**: the command is host-visible because the refit
protocol reads it every extract, which is free on Apple's unified memory and
costs a PCIe fetch per draw on a discrete GPU — booked as a greppable
`TODO(mesh)` awaiting a discrete-GPU consumer to measure it, because the
hazard is not the cost but that the cost is invisible on the only hardware CI
runs. With this, **all four seam-B blockers are settled**; what remains is a
consumer that actually draws it.

### 2026-08-03 — Nothing in the mesh extractor reads "the current slot" except the code that writes it: the capacity plan is slot-independent, and a slot is marked outstanding only where a `DeviceMesh` is handed out.

Found on an iPad
Pro, where a ring reached a **1.1 GB arena for 36 904 triangles (0.59% full)**
in a few hundred remeshes and then lost the device. The cause was one line:
`plan_capacity` floored its estimate at `arena_capacity()` — right for a
single arena reused in place, and a **ratchet** across a ring, because it read
the slot just *written* while a different slot was about to be *grown*. Every
extract then asked for 1.5× the last, geometrically, with the measured
triangle density never getting a say.
**The floor is deleted, not repositioned.** Reordering the claim above the
plan fixes the symptom and leaves a purely positional invariant inside a
340-line function; the floor is also **provably inert** either way, since
`capacity` reaches only `ensure_output_buffers`, which keeps whatever the slot
holds and grows only on a larger request, and the dispatch is pushed
`arena_capacity()` rather than `capacity`. So the plan now reads *no slot at
all* — this call's active set and the last call's density — and cannot
compound in any call order. Confirmed by mutation: restoring the floor alone
changes nothing measurable; restoring it *together with* the old claim
position reproduces the runaway.
**Fixing that exposed the rest of the family, all the same shape — state that
was correct for recon reading its own results and silently wrong across the
ring.** (1) *The stamp says a consumer holds a mesh here, so only a path that
hands one out may write it.* It lived in `ensure_output_buffers`, so every
failure after it — the dispatch, the refit, the `out_of_memory` return, the
`maxStorageBufferRange` rejection — left a slot marked with a generation **no
`DeviceMesh` ever carried**, which nothing could then release: `slot_count`
such failures and the extractor refuses *permanently*, quoting generations the
API never handed out. It now sits beside `device_mesh.generation` on each
publishing return, which makes "every generation handed out lives in exactly
one slot" total by construction rather than by audit. (2) *`claim_output_slot`
scans instead of assuming.* "The next slot is always the oldest" holds only
while every claim goes on to publish; a failed extract and the host overloads
both leave a free slot behind the cursor, and assuming would refuse with one
in hand. (3) *The host overloads give back their own slot, not everything
below it.* They must hand it back — a `Result<Mesh>` carries no generation, so
a host-only caller cannot — but doing it through `release_through`, the
**consumer's** high-water mark, also retired every older slot, including one
an `extract_device` caller on the same extractor was still drawing out of;
the next grow then ran `vmaDestroyBuffer` under a live draw, undefined with
layers off, which is the shipping configuration and the only one on iOS.
`free_slot_of(generation)` clears one stamp and moves no mark. (4) *The claim
and `++generation_` are adjacent, and both move above the active-set compact.*
`download()` reads the current slot on the strength of a *generation*
comparison, so anything fallible between the two leaves a window where an
outstanding mesh passes the currency check and is copied out of a slot that
may never have been sized — a `memcpy` from a null mapping. Above the compact
because a refusal must cost the caller *nothing*, and the compact is a
dispatch, a fence wait and a full active-set readback that a consumer one
frame behind would pay on every frame. (5) *`ExtractTimings::arena_bytes` sums
the ring.* It reported the claimed slot alone against a header that promised
what the extractor holds, so at `slot_count = N` it under-reported resident
memory N-fold — and it is the instrument this bug was diagnosed with. (6)
*`download` gates on `valid()`* like both extract entry points; the defaulted
moves reset every owned member but copy the scalar `generation_`/`slot_`, so a
moved-from extractor passed the check and then read a null mapping.
**The regression test asserts a magnitude, not just a shape.** Pinning
per-slot periodicity alone stays green under a wholesale revert to worst-case
sizing — which is perfectly flat, just ~19× too big — so the ratchet block now
sandwiches resident arena bytes between 1× and 3× the emitted surface *per
slot* and holds them flat across the last two turns of the ring, with its
stride derived from `slot_count` rather than hardcoded beside it. Both halves
were confirmed to fail against a re-introduced ratchet, against worst-case
planning, and against reporting one slot instead of the sum. A separate case
holds a live `DeviceMesh` while a host extract runs on the same extractor and
asserts the ring then refuses — the handle-independent half of (3), which no
dedicated-extractor test could see.

### 2026-08-08 — The sparse mesh kernel resolves its own 2×2×2 neighbourhood by probing the hash table on-device, and `mesh` therefore reads a `volume` buffer — a coupling paid for, not stumbled into.

*Amended 2026-10-13 (the volume block-list entry, below):* `HashDiagnostics`
is gone, so the suite asks the device whether the chained fixture spills: a
table of its shape whose chains may not grow refuses some of its blocks for
chain room.
*Amended 2026-10-22:* the extract splits its dispatch at
`maxComputeWorkGroupCount[0]`, as the codec's and the block stamps' passes
do, so a smaller block costs more dispatches but no longer lowers the
reachable active-block count (the "two things deliberately not done"
below), and room0's ~107 k blocks at 5 mm mesh at the 65535 floor.

*Reverses* the "host-built
neighbour table … no device-side hash probe, and no coupling to the hash
table's internal buffers" that the sparse-extract description above asserted
since the tier landed. The host pass hashed every active block coordinate into
an `unordered_map` and did eight lookups per block: an `O(active·8)` serial
pass rebuilt every extract whether or not anything moved, measured through
`volumetric_kit_ios` on an M5 iPad Pro at 107 k blocks as **102.2 ms of a
132.7 ms extract** — against 25.9 ms for the marching-cubes dispatch it existed
to feed. The kernel now runs one **workgroup per block** (not a flat voxel
grid, since shared memory is per workgroup) and threads 0–7 each probe one
octant into `shared int s_neighbour[8]`; eight probes amortise over
`voxels_per_block` cells. Extract goes 132.7 → 42.0 ms. Both prior
implementations of this pipeline did it this way — the CUDA and Metal marching
cubes in `implicit_world_reconstruction` each fill a threadgroup
`s_neighbor_cache` from device-side `findVoxelBlockByCoord` probes — so the
host table was the outlier, not the design.
**What the coupling actually costs, stated because the reversed sentence used
to deny it.** `VoxelHashMap::entries_buffer` / `entries_buffer_size` are now
public, and `mesh` tracks `volume`'s entry layout, bucket geometry and chain
protocol. Three consequences are carried rather than waved at. (1) *The probe
is quiescent-only.* It takes neither the bucket lock nor the acquire
`memoryBarrierBuffer()` `block_exists` splits its `ptr`/`pos` test around, and
no barrier could fix that anyway — a walk can meet a chain mid-splice, which is
structural, not an ordering problem. So the precondition is stated on
`MarchingCubes::extract_device`, where a caller meets it, not only on the
`volume` accessor: a consumer that fuses and meshes on different threads must
serialise them. Within one thread it holds by construction, every `volume`
dispatch being fence-blocked. Per the 2026-08-04 rule this would be the
library's to *check* rather than document, and it is documented instead
knowingly: `VoxelHashMap` tracks no in-flight mutation, and a flag that a
second thread could race is not a check. (2) *The entries buffer is
host-visible* (`volume` books device-local + staging as a follow-up), so the
win is measured on unified memory only; on a discrete GPU each workgroup's
seven probes become up to `bucket_size + max_chain` dependent scattered reads
across PCIe — a greppable `TODO(mesh)` at the binding, the same shape as the
host-visible indirect command's. (3) *The binding names its real size and is
range-checked*, because this is the one buffer in the extract that **another
tier** sized and `resize` doubles: `VK_WHOLE_SIZE` would have shipped the
accessor added to avoid exactly that with zero call sites.
**One definition of the hash ABI, not a fourth mirror.** The first cut
restated `HashEntry`, the free/no-offset sentinels and the three hash primes
in `hash_lookup.glsl` under a `Vr` prefix, reasoning that the prefix would
force a compile error on drift. It does the **opposite**, and this was checked
against `glslc`: a duplicate `const int kFreeEntry` is a hard `redefinition`
error, so *identical* names are the compile-time guard and a prefix is
precisely what lets two divergent sets coexist in silence — with a divergence
resolving neighbours from the wrong bucket and dropping surface at block seams
under `Status::ok`. The stated blocker to sharing (`hash_common.glsl` declaring
its own `pc` block, which collides with any kernel that has push constants) is
now opt-out via `VR_HASH_COMMON_NO_PUSH_CONSTANTS`; only the table *shape* had
to arrive as arguments. `hash_lookup.glsl` and `hash_common.glsl` both carry
include guards, and the sparse kernel's own `BlockIndex` mirror is gone too.
**The chain half of the probe was covered by nothing, and that is the finding
that mattered.** Every sparse fixture used `bucket_size 8 / num_buckets 128`,
whose 216 block coords peak at bucket occupancy 4 — `allocate_in_overflow` is
never entered, so no entry is ever reachable only through a chain. Replacing
the chain walk with `return -1;` left **every suite green**. The suite now
meshes the identical field through `bucket_size 2 / num_buckets 512`, asserts
via `HashDiagnostics` that it *does* spill before trusting what it proves, and
requires the mesh to match the reference triangle for triangle; the mutation
fails it. Production is the overflow case — `VoxelGridParams::defaults()` is
50×30000 with `max_chain 128`, and `overflow_count` exists because buckets
fill.
**Two things deliberately not done.** The primary-bucket scan does **not**
early-exit on the first free slot, though a miss then costs `bucket_size`
scattered reads and misses are common (the truncation band dilates a shell, so
most outer blocks have unallocated `+neighbours`). Insertion takes the first
empty slot but `delete_primary` clears an arbitrary one *in place*, so a bucket
may hold a free slot ahead of an occupied one — `block_exists` scans in full
for that reason and so must this. And the workgroup stays a fixed 256: below
`block_size` 8 that idles most lanes *and* multiplies the group count by
`256/voxels_per_block`, lowering the reachable active-block count against
`maxComputeWorkGroupCount[0]` (floor 65535) by the same factor. Sizing it from
`voxels_per_block` needs a specialization constant through
`ComputePipeline::create`; every in-tree caller uses 8, where the shape is a
win, so both costs are booked in the kernel rather than paid.
**The +25% dispatch regression was misattributed, and the correction moves the
roadmap.** It was read as lost occupancy — "a workgroup whose cells all bail
early now idles where the flat dispatch packed work densely". At `block_size`
8, `voxels_per_block` is 512 = 2·256, so the old `gid / 512` was already
constant across each 256-thread group: **every workgroup already belonged to
one block**, and only the group count changed (2 per block → 1). No packing
existed to lose. The cost is the new prologue — 8 lanes walking buckets while
the other 248 wait at the `barrier()`, serialised ahead of every cell. That
matters because `dispatch` is now 77% of extract — of `extract_device`, on
desktop, the readback being most of the rest — and the levers are the probe
(spread the 8 across subgroups, or one lane per bucket slot with a reduction),
**not** the cell count (coarser voxels) or the block count (incremental
extraction) the first analysis pointed at. Left unmeasured rather than guessed
at, per this file's own lesson about profiling the phase before optimising it.
(*The block-count lever is reinstated by the 2026-08-09 incremental-extraction
decision below, on a device measurement this desktop one could not see.*)
`ExtractTimings::neighbour_lut_ms` is **removed**, not retained as a zeroed
phase: a permanently-0.00 row in the overlay that diagnosed this is worse than
an absent one.

### 2026-08-08 — The overflow scan stays exhaustive; what gets bounded is its cost per slot, not its length.

*Amended 2026-10-13 (the volume block-list entry, below):* `diagnostics()`,
`HashDiagnostics` and `volume_diagnostics_test.cpp` are gone;
`coords_in_bucket` lives in `volume_hash_map_test.cpp`, whose chain fixtures
the device confirms share their bucket.
*Amended 2026-10-22:* change (2)'s premise, that an empty heap is the
near-full state, was wrong. A dispatch that overfills the table leaves only
anchors free, so the heap never empties, and every failing insert swept the
whole table holding its home bucket's lock: Replica room0's first frame at
5 mm in 4 096 buckets spent 0.70–0.90 s of allocate device time on an M5 Max
(Release, three runs). On the heap path an overflow insert now reports
`heap` without scanning once fewer than an eighth of the blocks are left, the
anchors' share at the default bucket size; `table` remains a clean sweep's
proof, on the rehash path and at smaller buckets.
The retry loop also ends at the first round with a capacity limit, where it
re-dispatched until two rounds made no progress, each repeating the failed
inserts and the lock races of the threads queued behind them;
`AllocFailures::rounds` counts the rounds, and `recon_volume_allocate` tests
the exit. With both, that frame allocates in 13–22 ms, after one more grow
(resize 35–41 ms of host time, from 9–15), and meshes the same 6 585 513
vertices.

An M5 iPad Pro lost the GPU inside
`allocate_from_depth` at **31 480 of 32 768 blocks** —
`kIOGPUCommandBufferCallbackErrorHang` on recon's compute queue, taking the
renderer sharing the device down as collateral and reaching the app one frame
later as an unrelated `VK_ERROR_DEVICE_LOST`, with nothing in the chain naming
the kernel. `allocate_in_overflow` was the one loop that did not honour
`hash_common.glsl`'s own rule that a contended dispatch must not be able to
hang the GPU: it scanned all `num_buckets · bucket_size` entries for a free
non-anchor slot.
**The obvious fix — cap the scan — was tried and rejected**, because the
length was never the expensive part. Each slot cost a contended
`atomicCompSwap` **and two device-scope `memoryBarrierBuffer()`s taken before
the slot had even been read**: ~430 of each per attempt at 96% occupancy,
×5 `insert_block` attempts, ×27 band blocks per depth pixel — ~58 k atomics
for one pixel, still watchdog territory under any cap worth having. A cap also
spends the only thing `kFailTable` is good for. `hash.hpp` tells a caller to
answer it by growing and `fuse_replica` answers by doubling every attribute
array (768 → 1536 MiB each, ~2.3 GiB transient), so a reason that can *also*
mean "I stopped looking" converts lock contention and chain depth into an
unbounded memory spend over a table that was never full. And `resize`'s rehash
runs the same insert under the highest concurrency in the repo, so the
prescribed remedy for `kFailTable` could itself fail with `kFailTable` and
roll back to an equally full map — a permanent stall.
**Three changes, and the order of operations in the first is the whole fix.**
(1) The scan reads a candidate's `ptr` **unlocked first** and skips it when
occupied, so only a slot that looks free pays the atomic — ~25× fewer of them
at 96% occupancy. That is safe rather than lucky: alloc and free run in
separate dispatches, so within one dispatch a slot moves only free →
occupied, which makes the unlocked read conservative in the one direction that
matters — a stale "free" costs a wasted lock and is caught by the
authoritative re-test under it, and a stale "occupied" cannot happen at all,
so the filter never skips a genuinely free slot. Lock-then-look also
*manufactured* the contention it went on to report, seizing buckets it had no
use for. (2) An empty heap short-circuits the scan with **one atomic load**,
which is the state the iPad was actually in: `validate()` forces
`num_blocks == bucket_size · num_buckets == total_entries` and every occupied
slot on this path consumed exactly one heap block, so "no block left" and "no
slot left" are the same statement, and the sweep was running to exhaustion to
discover nothing. It fixes attribution too — that state now reports
`kFailHeap`, which is what `allocate_in_primary` already calls it, rather than
one physical condition getting two names according to which helper hit it.
Guarded to the heap path, so `kFailHeap` stays *provably impossible* on the
rehash preset. (3) `insert_block` stops re-running the scan on a reason it
cannot resolve: within a dispatch `kFailHeap` / `kFailTable` / `kFailChain`
are monotone and no other thread can insert the coord either, so only
`kFailLock` earns another of the five attempts.
**`kFailTable` therefore comes out *stronger*, not weaker**, which is the
point: a sweep that skipped a free-looking slot only because another thread
held its bucket now reports `kFailLock` — the retryable reason — so
`kFailTable` is reached only by a clean sweep and is genuine proof the table
is out of usable slots. It is still not the same as "every slot is taken":
free *anchor* slots are invisible to an overflow insert by design, and the
test fixture pins exactly that case.
**Growth also stops being purely reactive.** The occupancy signal existed but
was reachable only through `diagnostics()`, which scans every slot on the host
(1.5 M at the example defaults) and carries its own `TODO(volume)` — so no
caller could afford it per frame, every fuse loop grew only *after*
allocations had already failed, and that is how a device reached 96% in the
first place. `VoxelHashMap::load_factor()` is a 4-byte read of the
host-mapped heap counter, the same quantity `HashDiagnostics::load_factor`
reports without the scan. `fuse_replica` prints it beside the overflow reason;
adopting an actual grow-at-threshold policy is deliberately **not** taken
here, because it would move the peak-memory and throughput figures this file
quotes and deserves its own measurement.
**Verified by mutation, which the first cut of this work was not** — its test
passed against the unbounded original *and* against a cap half the size, so it
pinned the change in neither direction. Three independent assertions now, each
confirmed to fail against the code it exists to catch: re-introducing a
256-slot cap fails the 300-deep chain fixture (its 292nd overflow node sits
333 iterations out, chosen to exceed any window worth having, on a table that
stays ~85% free so nothing there is a capacity limit); deleting the heap
early-out fails the pair asserting that a full table reports `heap` while a
table whose only free slot is an anchor reports `table`, on a 2×2 fixture
small enough to sweep by hand; and breaking the chain head-insert fails the
one-dispatch re-allocation, the only assertion that traverses `offset` —
`compact_active_blocks` is a flat per-slot scan, so a broken link still yields
the right *count*, which is why the size check alone proved nothing. The
coord search is bounded (`coords_in_bucket`, mirroring
`volume_diagnostics_test.cpp`) rather than a loop whose only exit is success.
What stays untested is named rather than implied: the
`kFailLock`-on-contention path and the retry-loop early exit are both
concurrency properties, and every fixture here dispatches one coord at a time
precisely to keep contention out of what it measures. The hang itself remains
unreproducible in a unit test — it needs ~131 k entries and thousands of
concurrent invocations — so what is claimed here is the cost model and the
reason codes, not a re-measured device. (*Since 2026-09-28 a remove's
contention is tested: `recon_volume_delete` queues 1 024 coords on one
bucket's lock. The early exit now fires only on a capacity limit; see the
measured lesson "A flake read as a livelock".*)

### 2026-08-08 — In-block vertex sharing is a *second compiled kernel*, not a branch, and the two emitters must interpolate an edge in the same direction.

`MarchingCubesConfig::share_vertices` makes the cells that meet on an edge
index one vertex instead of emitting three private ones each: every cell emits
a vertex only on the three edges it **owns** (+x/+y/+z from its base corner,
derived from the tables by `mcEdgeOwner` rather than tabulated a fourth time),
and its neighbours look that up in `shared int s_edge_vtx[]`. Edges on a
block's `+face` are still duplicated — sharing them would need the
neighbouring workgroup's shared memory — which is why the saving is **~4x**
rather than the ~6x full sharing would give. Measured on Replica room0:
3.4x fewer vertices, 3.5% faster end to end. Two consumer-visible
consequences, both published on `DeviceMesh` rather than left to be inferred:
the index run stops being the identity `0,1,2,...` (`shares_vertices`), and
`vertex_count` stops being `3 * triangle_count`.
**Two kernels, chosen at `create`, and that is the decision.** Sharing needs
~8 KiB of `shared` arrays, and a `shared` array is reserved when the pipeline
is created whatever a push constant later says — so the first cut, one kernel
with a `share_vertices` push constant, made the **default** path pay
sharing's threadgroup budget (32 B → 8 224 B, verified by SPIR-V
disassembly; **44 B → 8 428 B** on today's kernels, re-verified the same way
after incremental extraction's stage 2 gave the default path a block
reservation and sharing an edge-owner table — the gap is the point, not the
endpoints), which bounds residency to 3 workgroups on Apple's ~32 KiB and
to **1** at Vulkan's guaranteed 16 KiB floor — on the kernel this file records
as 77% of an extract. It also cost the default path its memory *shape*:
routing both modes through the sharing emitter gave every corner its own
interleaved `atomicAdd(vertex_count, 1u)`, so a triangle's three 64-byte
writes landed at three arbitrary offsets in a multi-hundred-MB arena and the
index run came out non-monotonic — lost write coalescing in the kernel, lost
vertex-fetch locality in the seam-B draw that reads it. `fuse_viewer`,
`fuse_render`, the whole `texture` tier and the iOS scanner all take that
path for a feature none of them enables. So `marching_cubes_sparse.comp` is
the default and is byte-for-byte what this tier always did, and
`marching_cubes_sparse_shared.comp` is its sibling; what they do identically
(the neighbourhood probe, the corner gather, the block addressing) is
`marching_cubes_sparse_common.glsl`, the same shared-`.glsl` discipline
`marching_cubes_common.glsl` applies to the dense/sparse pair.
**The bug that made "sharing moves only the vertex count" false was the edge
DIRECTION.** `kEdgeToVert` lists edges 2/3/6/7 max-corner-first, so a cell's
owned +y edge (edge 3 = corners {3,0}) is its neighbour's edge 1 (corners
{1,2}) — the same segment, traversed opposite ways — and both `mix` and the
near-tangent guard are direction-dependent. Normally that is ulps; under the
guard it is a **whole voxel**, because at `sa = -1e-9, sb = +1e-9, iso = 0`
the guard forces `denom` to `+1e-6` one way and `-1e-6` the other and *both*
come out at ratio ~1e-3, hugging whichever endpoint was passed first.
`mcEdgeVertex` therefore orders the endpoints canonically (by corner shift —
the same rule `mcEdgeOwner` names the edge by, so the two cannot disagree),
and `mcEmitCell` now calls it instead of carrying a verbatim copy of its
body, which is what made the drift possible. That also makes the shared and
unshared meshes comparable **triangle for triangle on exact position floats**,
which is the invariant the suite asserts.
**`vertex_count` counts vertices, not claims.** With a per-vertex global atomic,
a phase-1 claim that overflowed the arena had no usable slot, so the cells
referencing that edge re-claimed a replacement and double-counted: an
overflowing dispatch reported `V_owned + ~3T` instead of the true ~0.75T —
*more* than not sharing at all — and the host refits a **grow-only** arena to
that number: ~95 MB pinned on room0 against the 15.9 MB the shared surface
needs, which is the entire thing the feature exists to save. That path is
taken by design on the first sharing extract of any real surface, since
`kSeedTrisPerBlock` deliberately plans low. It was first fixed with a
`kVertexDropped` sentinel distinct from `kNoVertex`; the stage-2 restructure in
the 2026-08-09 entry below **removes the sentinel**, because a block that counts
first and reserves once gives every vertex a deterministic slot whether or not
it lands inside the arena, so the totals are exact by construction rather than
by bookkeeping.
**Sizing two buffers means two budgets, and they must not read each other.**
The vertex arena and the index run are now grown **independently**: they used
to be released together because `arena_capacity()` came off the *arena*, which
stopped being true the moment sharing moved it to the index run — and keeping
the coupling grew whichever buffer *fitted* to 1.5x what it already held,
compounding on every such event. That is numerically the ring runaway the
2026-08-03 slot-independence decision exists to prevent, reintroduced one
buffer over, and it refaulted the arena — the phase measured at 49.8 ms of a
55 ms extract. Relatedly, `plan_capacity` bounds its triangle request by
**both** buffers' device limits (`max_triangles_for` covers only the index
run's 12 B/triangle; the arena's three private vertices are 16x tighter), and
`plan_vertex_capacity` clamps in *both* modes: a predicted quantity is
clamped, never rejected (the 2026-08-04 rule), and the unshared branch had no
device clamp at all, so on a driver at Vulkan's guaranteed 2^27 a plan could
be rejected by `ensure_output_buffers` — **permanently**, since the density is
recorded only on success and a scan's active set only grows.
**A failure disarms the draw command rather than clamping it.** Clamping
bounds how many indices a draw *reads* and says nothing about their *values*,
and on every failure path the values are the problem: a triangle whose vertex
claim overflowed is counted but never written, so its three index slots hold
whatever the VMA block last did, and the dense kernel never writes the index
run at all. Arbitrary `uint32`s into a 64-byte-stride arena with
`robustBufferAccess` enabled nowhere is a GPU fault, where "this slot has no
drawable geometry" is the honest statement — so `indexCount` goes to 0, on the
two sparse failure returns *and* at the end of the dense overload, which
publishes no `DeviceMesh` and used to leave a live-looking command behind.
**What the library checks rather than documents**, per the 2026-08-04 rule.
`ProjectiveTexturer::texture` **refuses** a mesh carrying `shares_vertices`:
it decides visibility per *triangle* and writes `uv0` per *vertex*, so where
an interior vertex is referenced by up to six triangles that disagree the last
writer wins nondeterministically, and a triangle left holding one sentinel
interpolates from `(-1,-1)` across its whole face — `Status::ok`, no
validation diagnostic, visible only as flicker. The real fix is a
per-primitive camera id, which the planned packed multi-camera atlas needs
anyway; this flag keeps the two decisions independent.
*Superseded 2026-08-11 (below):* the refusal is gone — the pass dispatches per
vertex, so there is one writer per vertex and nothing to disagree. What that
entry keeps from this one is the shape of the argument, not its conclusion: the
hazard was real and unseeable by the caller, so checking it rather than
documenting it was right while it existed. `shares_vertices` stays published for
**sizing**, and the per-primitive camera id is still what the packed
multi-camera atlas needs. And the sharing
kernel's compile-time `kMaxSharedCells` is mirrored by a host constant that
gates the per-*extract* refusal of a larger `block_size` (not `create`'s — the
block size arrives with the grid), with nothing but a comment tying the two
across languages; so the kernel **reports** whether it actually shared, in a
scratch word past the draw command, and the host fails on it. Raising the host
constant alone would otherwise admit a block the kernel silently declines.
**Verified by mutation**, since a test that cannot fail is worse than none.
`share_vertices` had **zero** coverage in the first cut — `bool sharing =
false;` left every suite green — and two of the assertions added beside it
were tautological restatements of what `download()` had just computed. Four
independent mutations each fail a named line now: dropping the canonical
endpoint order fails the shared-vs-unshared triangle-for-triangle equality;
removing the `kVertexDropped` sentinel failed a *magnitude* bound on the
refitted arena (the shape passes either way, because the second dispatch's
counter is the true total whatever the first reported — what changes is what
the grow-only arena is refit to), and that sentinel and that mutation are both
retired by the stage-2 restructure below, which carries its own; restoring the
coupled buffer growth fails a
case built from the measured capacities, where the dense overload's 3:1 vertex
demand grows the arena of a sharing extractor whose index run already fits;
and deleting the `sharing_applied` report while raising the host constant
turns a refusal into a silent empty mesh. What is **not** covered is named
rather than implied: the arena-overflow path is exercised only through a short
plan, not through concurrent contention, and no fixture measures the
threadgroup-occupancy claim above — it rests on the SPIR-V allocation and the
documented limits, not on a re-measured device. (The figure there moved to
8452 B when stage 2 added the sharing kernel's six reservation words; the
default kernel's 44 B is unchanged, and the gap is the decision.)

### 2026-08-09 — A dirty block is one the fuse *changed*, the flags are anchored to a grid the library checks, and tracking them is opt-in.

*Amended 2026-10-06 (below):* incremental mesh extraction is removed. The
`changed` stamp that replaced these flags stays; the mesher no longer reads
it.

`tsdf` can now
report which blocks a fuse invalidated — the input an incremental re-mesh
needs, and the first thing on the roadmap that the whole-volume extract's
`dispatch` phase (77% of an extract since the 2026-08-08 neighbour-probe
decision) actually points at. Four things about its shape are the decision,
and each was a defect in the first cut.
**(1) The flag means the field moved, not that a store happened.** The kernel
marked unconditionally at the top of the fuse. In **classic** mode — the
default, and what `fuse_replica` runs — free space ahead of the surface is
fused too (that is what classic *is*), so the only narrowing above the mark
was the occlusion return and the mark collapsed to *in-frustum ∧ valid depth
∧ not occluded*: a frustum survey wearing a different name, which is exactly
what `dirty_block_count`'s header said it was not. It now compares the new
`tsdf`/`weight`/`color` against the old and marks after the stores it
describes, so a voxel converged at `max_weight` — re-storing bit-identical
numbers, the steady state of any revisiting scan — marks nothing. Note the
asymmetry that gave it away: the dynamic-clear store was already guarded by
`weight > 0.0`; the fuse store was guarded by nothing.
**Booked honestly: this narrows the set but does not make it small.** On
room0 at 120 frames it reports **62% of active blocks changed** per 20-frame
window and 59% per *single* frame — because in classic mode a moving camera
keeps shifting the running average of everything it sees, which is a real
change, not a miscount. The dilated re-mesh set is 83%. So the honest
headline is that an incremental extract's ceiling here is ~1.2x, not the
order-of-magnitude the first cut's `speedup %.1fx` implied — and that figure
is **removed** rather than corrected, because it divided active blocks by
re-mesh blocks and so assumed extract cost is proportional to block count
with zero fixed cost, which this example's own phase table refutes (`compact`
walks every table slot regardless, `arena` is sized by the whole surface,
`readback` copies all of it; only `dispatch` scales).
**(2) A flag is keyed by block SLOT, so it means nothing without a grid — and
the library checks which one.** Slots come from a LIFO heap, so the same index
means a different block on a different grid, and a different *coordinate*
after a removal. Neither is knowable from the flags, and neither was knowable
to the caller, whose only instruction was a `@param` sentence — the shape the
2026-08-04 rule exists to close. `VoxelBlockGrid` therefore publishes
`topology_epoch()` (bumped by `remove`/`clear`, deliberately **not** by
`resize`, which preserves every index), and the integrator records the grid
pointer plus that epoch. A different grid **resets** the flags and re-anchors
— refusing instead would bind an integrator to one grid for life, and one
integrator over many grids is already how the tsdf suite is written, which is
how the silent OR-ing went unnoticed. A removal on the *same* grid **refuses**
until `reset_dirty()`, because the geometry that went away is stale and no
surviving flag can say so: "re-mesh everything" is the only true answer, and
only the caller can act on it. The pointer is held for identity and never
dereferenced.
**(3) A grow carries the flags forward.** The array was reallocated and zeroed
on every map grow under a comment asserting the opposite ("the surviving flags
stay correct"), so a window straddling a `resize` forgot every block fused
since the last reset — silently, and precisely on the frames a growing scan
brings in the most new surface, which are the frames that trigger the grow. It
now memcpy's and zeroes only the tail, mirroring `VoxelBlockGrid::resize`,
which grows the attribute arrays the same way and cites the same
index-preserving rehash.
**(4) It returns the block set, and tracking is off by default.** A count
cannot drive the incremental extract the API exists for, and the walk already
computes the coordinates, so `dirty_remesh_blocks` hands back `Vec3i`s and
`.size()` is the count. It takes the caller's already-compacted active set
rather than compacting its own — the flags carry slots, not coordinates, so
every caller has just compacted, and a second compaction inside was a
dispatch, a fence wait and a full readback (0.15–0.26 ms at the examples'
defaults) on a call already O(active). The walk also inverted: it gathers over
active blocks testing their `+{0,1}³` octant, which makes the
existence filter *structural* (only blocks that exist are considered) and
needs no dedup, against the two hash sets it replaced. And
`TsdfIntegratorConfig::track_dirty_blocks` defaults **off** — the 2026-08-01
bar is "nothing measured when the caller did not ask", and always-on cost
every integrator a `num_blocks * 4` host-visible array (6 MB at
`VoxelGridParams::defaults()`, doubling per grow) plus a store per voxel, paid
by `fuse_viewer`, `fuse_render`, the iOS scanner and the whole test suite for
a flag none of them reads. Off binds a 1-element dummy to the slot and pushes
`track_dirty = 0`, the `color_dummy_` pattern this file already uses.
The per-voxel store is now `atomicOr` and guarded by the change test, so the
atomic is paid only by voxels that moved (none once a block converges) — a
non-atomic race on a shared address is undefined rather than benign without
`coherent`/`NonPrivatePointer`, and the two doc sites claiming the kernel
"needs no atomics" are amended rather than left to read as whole-kernel.
**Verified by mutation, and the fixture is the reason.** The original
assertions could not fail: their dirty set was a contiguous 1-D run whose `-z`
dilation is a no-op, so dropping the dilation, dropping the existence filter,
inverting the octant to `+x/+y/+z` (which the PR body itself called "the
natural wrong guess") and `return active.size()` all passed, as did a
vacuous pre-integrate `== 0` that held with the guard deleted. The fixture is
now a 2×2×2 cube straddling the band plus one off-camera block, so changed=4,
re-mesh=8 and active=9 are three different numbers, and eight independent
mutations each fail a named line: the copy-forward, the change predicate, the
re-anchor reset, the dynamic-clear mark, the octant, the feature itself, the
opt-out, and the topology check. Fusion numerics are untouched — room0 at 120
frames gives 832 518 vertices / 277 506 triangles with tracking on *and* off,
identical to what this file records for main. What is **not** covered is named
rather than implied: the threading obligation on `dirty_block_count` /
`reset_dirty` / `dirty_remesh_blocks` is documented (mirroring
`MarchingCubes::release_through`) and **not** checked, because the reader they
race is a fuse thread that can free the mapping mid-read and a flag a second
thread could race is not a check; and the `atomicOr` contention path is
exercised by no fixture, since every fixture here dispatches one frame at a
time.

### 2026-08-09 — Incremental mesh extraction is worth building at a ~4x ceiling, not the ~18x one window suggested, and the worst frame rather than the median sizes its design (amends the dirty-block decision above).

*Amended 2026-10-06 (below):* incremental mesh extraction is removed. Its win
was never measured past this entry's ceiling estimate.

The dirty-block decision above measured coverage on **room0** — 62% of active
blocks changed per 20-frame window, 59% per single frame, **83%** once dilated
into the `-x/-y/-z` octant — and put an incremental extract's ceiling at
**~1.2x**, a clear "do not build it". It also *removed* the first cut's
`speedup %.1fx` read-out rather than correcting it, because dividing active
blocks by re-mesh blocks assumes extract cost is proportional to block count
with zero fixed cost, which this repo's own phase table refutes. Both of those
stand as **method**. The **verdict** does not: room0 is the wrong scene to have
measured coverage on, and this entry amends it. (Same date as the entry it
amends, so read this one as the later — file order is not the tie-breaker the
top of this file promises.)

**Measured on an iPad Pro M5**, one real handheld walk, map saturated at
111 942 blocks, sampling `dirty_remesh_blocks` / active per window:

| to re-mesh | ceiling (1 ÷ share) |
|---|---|
| 5.6% | 17.9x |
| 18.4% | 5.4x |
| 31.5% | 3.2x |
| 34.4% | 2.9x |

Only the left column is measured. The right one is that same invalid division,
kept because an upper bound is still worth having and labelled so it is not
read as a speedup: the realised factor is strictly below it by whatever share
of an extract does not scale with the block count, which is **not** measured on
this device.

**Median share 25% → a ~4x ceiling; worst sampled 2.9x, best 17.9x.** At the
ceiling the walk's `meshing` row goes 51.85 ms → **≥13 ms**, taking its ~64 ms
frame to **≥25 ms**; the real numbers land above both. So it is worth building
— but at roughly a *quarter* of what the best sampled window promised, and
quoting that window would have been the same error as quoting room0, one
optimism in place of one pessimism.

**room0 is a correctness fixture, not a coverage one**, and that is the whole
gap. It is one enclosed room shot through a 90° cone (`fx` 600 over 1200 px),
so the camera sees very nearly the entire surface every frame and *almost
everything it sees genuinely changes* — in classic mode a moving camera keeps
shifting the running average of every voxel in view, which is a real change,
not a miscount. Like for like, both sides dilated: **83% on room0 against the
25% device median**, so room0 is **~3.3x** pessimistic — and pessimistic
precisely because it is a good correctness fixture: total visibility is what
makes it reproduce triangle-for-triangle. Every conclusion drawn from it alone
about *coverage* was wrong. This is the same family as the
correct-on-the-machine-that-ran-it hazard the seam-B decisions kept turning up,
one level out — not wrong hardware but a wrong *scene*, and nothing in a green
test run says so.

**A proxy for the instrument was worse than either.** Before
`dirty_remesh_blocks` existed, a frustum survey stood in for it: it read 87% on
room0, and on device it was wrong in **both directions** — 4.5% against the
**18.7%** the instrument read on the same walk, off by ~4x in the optimistic
direction. (That 18.7% is a separate reading from the table above, whose
nearest row is 18.4%; which windows the two cover is not on record.) A survey
of what the camera *could* touch is not a measurement of what a fuse *did*
change, and the two do not even err consistently. That is why the dirty-block
decision above insists the flag mean the field moved rather than that a store
happened; a cheaper approximation of this quantity has already been tried and
it does not work.

**The variance, not the mean, is the design constraint.** A system that reaches
17.9x on its best sampled window and 2.9x on its worst still has to fit the
worst, and a frame-rate budget sized on the median tears exactly when the
camera sweeps into new geometry, which is when a scanning user is most likely
to be looking. So incremental extraction is planned **beside** a remesh
*cadence* cap rather than as a replacement for one: the cadence bounds the
worst frame, the incremental path buys back the median. Sizing anything on the
~4x alone would reintroduce the per-frame cliff that the whole-volume extract
at least had the virtue of being honest about.

**Dilation is cheap and needs no hedging.** The `-x/-y/-z` octant a
marching-cubes stencil reaches back through costs **1.16–1.31x on device** and
1.30–1.40x on room0 across three resolutions. It is a property of the stencil,
not of the scene, so the ratio is stable and no allocator has to carry a margin
against it. It is also already *inside* every figure above:
`dirty_remesh_blocks` returns the dilated set, so the ~4x ceiling is computed
on the blocks an incremental extract would actually redo and must not be
divided by the dilation a second time.

**The four stages, so "stage 2" has an antecedent.** Stage 1 is the input and
has landed: `TsdfIntegrator::dirty_remesh_blocks`, the decision above. Stage 2
is per-block contiguous emission. Stage 3 is dirty-only dispatch over that set.
Stage 4 is compaction of the retired spans. Each needs the one before it.

**Stage 2 has landed on both sparse kernels; stage 3 is next.** Both used to
append through *global* atomics, so one block's output interleaved with every
other block's and **per-block ranges could not exist**: the default
`marching_cubes_sparse.comp` bumped `index_count`
once per triangle (`marching_cubes_common.glsl`) and wrote that triangle's
three vertices at `tri * 3`, while `marching_cubes_sparse_shared.comp` bumped
two independent counters, `vertex_count` per vertex and `index_count` per
triangle. **Both** had to restructure to **count → reserve one span per block →
emit** — the default kernel to one `atomicAdd`, the sharing kernel to two —
because `share_vertices` selects one of the two at `create` and is fixed for
the object's lifetime, so a kernel stage 2 skipped would be permanently
ineligible for stages 3 and 4. Both now do.

**What the existing golden test covers, and what it does not.** It compares the
two meshes' triangle sets resolved through the index buffer, *sorted*, on
position alone (`canonical_triangles` in `marching_cubes_sparse_test.cpp`) —
because the kernel appends through an atomic, output has never been
byte-identical run to run, and this entry's first draft claiming stage 2's
output *is* byte-identical overstated both the property and the coverage. That
test does validate stage 2's **geometry**: the same triangles have to come out.
It cannot see stage 2's actual deliverable — that each block's output occupies
one contiguous, non-overlapping span — so stage 2 ships an assertion of its own:
attribute each triangle to a block by its centroid, walk the index buffer in
order, and require the owning block to change **exactly `distinct - 1`** times.
Exactly, not approximately, because the attribution is provably exact — the only
centroid that could fall into the neighbouring block is one lying on the far
face of its cell, which takes three sign changes around a 4-cycle, and parity
forbids it. It runs on a clean extract and on one that refit against an
undersized arena, where truncating a span shortens it without splitting it.
Without that assertion a span off by a triangle first surfaces in stage 3,
confounded with exactly the dirty-set machinery stage 2 is sequenced first to
avoid.

**It is a cost, not a win, and that corrects this entry's first draft**, which
claimed stage 2 "removes the interleaved per-vertex atomics the vertex-sharing
review measured as destroying write coalescing — a win even if stages 3 and 4
are never built". Measured on room0 at `--voxel 0.012`, Release, the extract
dispatch goes **1.17 → 1.28 ms, ~10%**, in four interleaved samples each way.
The draft conflated the two kernels: the per-*vertex* interleaving that review
measured is the **sharing** kernel's, while the default kernel — the one
measured here — already wrote each triangle's three vertices contiguously at
`tri * 3` and interleaved only whole triangles, so there was little coalescing
left to win and the second visit costs more than it recovers.

**What moved that number, and what did not.** The first cut of the counting
phase gathered every cell in full and measured **+13%** (1.167 → 1.315 ms).
Counting on the eight corner *signs* alone — no `sdf`/colour array copied out of
the gather and no sRGB decode, for a phase whose only consumer is the cube index
— brought it to the ~10% above. Nothing about *where* the per-cell triangle
count is kept has ever moved it: caching the count rather than the cube index
changed 1.318 → 1.315 ms, and moving that cache out of `shared` into a private
register changed 1.28 → 1.28 ms. The cost is the second visit itself, not the
lookup. The private cache was taken anyway, and on a different argument: it
returns the default path's threadgroup footprint to **44 B**. The 2026-08-08
decision above splits the two kernels precisely so this path does not carry
sharing's, and a two-phase emitter that spent 512 B of `shared` — plus an
`atomicOr` and a zeroing pass — on a cache no invocation but its own writer ever
reads would have handed part of that back for nothing.

**The sharing kernel's restructure costs nothing measurable, and that is not
the same argument.** Three interleaved samples each of the old and new kernels
span 1.77–1.82 ms on the extract dispatch and their means differ by less than
that spread (room0 at 120 frames, `--voxel 0.012 --share-vertices
--device-extract --preload`, Release). Two reasons it is free where the default
kernel's cost ~10%: this kernel already made two visits over a cached
classification, so the counting phases reuse a `s_case` that had to exist
anyway; and the pass that runs over 100% of the cells reads the eight corner
*signs* alone (`mcCellSigns`), the same lesson that took the default kernel from
+13% to +10%. Caching the per-cell triangle count beside the cube index — free,
in bits the cube index does not use — then lets phase four claim a cell's whole
run with one threadgroup atomic instead of one per triangle (~320k against
~766k at this scale), and lets both later passes walk a `tri_table` row by count
rather than each re-finding its `-1` terminator, so the terminator is walked in
exactly one place, `mcCellTriangleCount`, which is also where that walk is
bounded against a truncated host upload. Geometry is unchanged as a triangle set
and as a vertex count (766 117 / 668 792, byte-length-identical PLY with
different bytes — the arena order is precisely what moved). Stage 2 overall is
therefore still justified as the precondition, and the cost is the default
kernel's ~10% on the one phase that scales with the block count — `dispatch`,
which the 2026-08-08 neighbour-probe decision puts at 77% of a *desktop*
`extract_device` and which is unmeasured as a share on device.

**Each cursor is bounded by its own block's reservation, on both kernels.** The
default kernel's span bound was argued as making a count/emit disagreement fail
*safe*; the sharing kernel needs it more and did not get it in the first cut.
Its count and emit phases run *different* predicates over *two* ranges, and one
escape exists — the "cannot happen" re-gather in the owned-edge pass, which
fails in the **under**-reservation direction, because an owner that emits
nothing sends every cell referencing its edges down the duplicate path instead.
Forced on one cell of a `block_size` 3 block, that drove consumption past
reservation in 1070 of 3000 trials, writing into the next block's range with
`Status::ok` and exact counters. The bound turns it into dropped geometry.
Both bounds are guards on a state no fixture can reach, so **neither is
mutation-covered and that is stated rather than implied**: what is covered is
the arithmetic they guard.

**What stage 2's mutations cover on the sharing kernel.** Dropping the
duplicate-count branch — the one the count phase exists for — fails a named
index-range bound on the refit fixture, whose mesh the first cut discarded
while asserting magnitudes that passed either way. Putting the vertex claims
back on the global counter, leaving the triangle span intact, fails the
**vertex**-span assertion and nothing else: `block_layout` walks the index run,
so it is structurally blind to the second range this kernel reserves, and every
transition count in the file passes against a fully interleaved arena. Both
assertions run on a clean extract *and* on one that refit against an undersized
arena, and on a 27-block fixture rather than the single-block one — with one
block, a cursor that overran has no next block to land in, so the failure is
unobservable there in principle. The unshared extract of the same field is the
oracle for both.

**The tension with seam B is real and is *not* yet resolved.** Incremental
extraction wants **in-place mutation** of a persistent arena; the seam-B ring
wants **immutable snapshots** a renderer can still be drawing. Note what is
already true: `MarchingCubes::Slot` holds an `arena`, an `index_run` *and* an
`indirect` per slot, so both buffers are ringed today, and the ring is what
makes an extract's immediate `vmaDestroyBuffer` safe — a slot is only written,
grown or freed once `release_through` has covered it.

The direction under consideration splits them: keep the **index buffer** ringed
and rebuilt each extract, and make the **vertex arena** one persistent
allocation in which a re-meshed block *appends* a new span and its old span
retires once released. The attraction is memory — on the default kernel a
slot's arena is **16x** its index run (a 64-byte `Vertex` against a 4-byte
index, three of each per triangle), so ringing the run instead of the arena is
where nearly all of a ring's extra residency goes. Two things are unsettled,
and neither is a detail:

- Growing one persistent arena **frees the storage an in-flight draw is
  reading** — precisely the hazard `MarchingCubesConfig::slot_count > 1` exists
  to remove. A chunked arena that never frees while a generation is outstanding
  is the obvious answer and is not designed.
- `release_through` is a **whole-slot high-water mark**
  (`released_through_ = std::max(released_through_, generation)`). Retiring
  *ranges within* an arena needs a granularity it does not have.

So `LiveMesh` and gfx are untouched only in the sense that no API changes; until
both are answered the *guarantee* behind that API changes, and that is a cost to
the sibling rather than zero. The 16x is a ratio and not a measurement — no
arena/index pair for a named scene and slot count is on record — and it does not
hold under `share_vertices`.
*Amended 2026-08-11 (below):* that caveat used to be waived on the grounds that
"the only seam-B consumer cannot set `share_vertices` anyway, since
`fuse_viewer` runs the `texture` tier and that tier refuses sharing". The tier
refuses nothing now and `fuse_viewer` takes `--share-vertices`, so the ratio is
live again under a configuration this repo ships — and still unmeasured.

**What this does not establish, named rather than implied.** The ~4x is one
device, one walk, one scene — an M5 iPad Pro with unified memory, where the
host-visible arena the 2026-08-08 seam-B decision books as a discrete-GPU risk
is free. Nothing here re-measures that. Nor are the device figures recorded with
voxel size, integration mode or window length, against this repo's own rule that
a figure carries its configuration — so read them as a coverage sample and not
as a phase budget. Four windows are enough to establish that room0 is
unrepresentative and that the worst case is near 3x, and not enough to size a
cadence cap. And the instrument that produced the phase split is a *desktop*
one: `extract_device` is 2.1–2.7 ms against 20 ms for `extract`, i.e. **87–90%
of "the extract" was a readback no seam-B consumer pays**, which is why the
device figures above are quoted against the walk's `meshing` row rather than
against that 20 ms — and why that share and the 77% above are shares of
different denominators, 77% being `dispatch` within the 2.1–2.7 ms rather than
within the 20 ms. Related and worth not re-deriving: dispatch scales
**sublinearly** in cells (2.84x for 3.95x the blocks) with the emit rate flat at
~7.7% across a 4x resolution sweep — the kernel is healthy, it is simply handed
63 M cells — and steady-state `arena alloc` is **0.03 ms**, the earlier 4–17 ms
readings having been one-shot first-allocation noise that reversed two
conclusions before 200-extract means replaced them.

Provenance. The device numbers are instrumented by `volumetric_kit_ios` PR #14
(console read-out logging), where `meshing` is that scanner's own overlay row
and not an `ExtractTimings` phase this repo defines. On desktop the two are
separate instruments: `fuse_replica --dirty-every N` prints the changed and
dilated coverage, and `fuse_replica --device-extract` prints the phase split.

### 2026-08-11 — The per-block span table is opt-in, is retired by generation rather than described in prose, is anchored per block slot to a globally unique topology token, and is one table for the whole ring rather than one per slot.

*Amended 2026-10-06 (below):* the span table is removed with incremental
mesh extraction.

Stage 2 left both sparse kernels computing a block-to-range mapping and throwing
it away. Publishing it as `MarchingCubes::block_spans()` is what stage 3
(dirty-only dispatch) re-meshes against, and it is not derivable on the host: the
reservation atomics hand ranges out in workgroup **arrival** order, not block
order, so nothing outside the dispatch knows which range belongs to which block.
Four decisions came out of reviewing the first cut, and the last three are all
the same mistake — a fact only the library can know, written down for the caller
instead of checked.

**Opt-in, because it is sized by the grid rather than by the surface.**
`num_blocks * 16` is 24 MB at `VoxelGridParams::defaults()` and doubles with
every `VoxelHashMap::resize`, held for the extractor's lifetime — paid by
`fuse_viewer`, `fuse_replica`, the iOS scanner and the whole test suite, for a
table whose only reader today is a test. That is exactly the bargain
`TsdfIntegratorConfig::track_dirty_blocks` already struck for the same slot-keyed
table shape at a quarter the width, and the bar it set is verbatim: *nothing
measured for a caller who did not ask*. `MarchingCubesConfig::track_block_spans`
gates it; a 1-element dummy keeps the descriptor valid and a `write_spans` push
flag keeps the kernel off it, which is the mechanism `color_dummy_` already uses
one binding over. The table is also counted in `ExtractTimings::arena_bytes`,
which it was not: that instrument is how the ring's runaway growth was diagnosed,
and omitting a component of what stays resident is the same defect that folding
the index runs into it fixed.

**Retired by generation, because a borrowed pointer's staleness is not
documentable.** The accessor hands out mapped device memory, and a grow
move-assigns the buffer — `Buffer::operator=` destroys the current state first,
so the next extract over a grown grid frees the pages a cached pointer names.
That is the growing-scan path the feature exists to serve. `block_spans_generation()`
carries the same counter `DeviceMesh::generation` does, so the two are directly
comparable, and the accessor returns `nullptr` until an extract has actually left
a table describing its own output. This matters most on the failure paths: both
kernels publish a block's span **before** their capacity guard, deliberately —
the counters must carry the block's full total for the host's refit — so after a
failed extract the table names triangles at bases the arena never held.
`disarm_indirect_command()` stopped those being *drawn*; nothing stopped them
being *read*. One comparison does.

**Not per slot, and that is the ring collision the 2026-08-09 entry recorded as
unresolved — resolved in the conservative direction.** The arena, index run and
draw command are per `slot_count`; this table is one instance describing the
current dispatch. Making it per-slot would multiply the 24 MB by the ring depth
to serve a consumer that does not exist yet. So the table stays single and the
generation stamp makes the mismatch *visible* instead of silent: a consumer
holding generation N whose `block_spans_generation()` has moved to N+1 knows the
spans describe another slot's arena. When stage 3 needs to read a held mesh's
spans, that is when the per-slot cost is worth paying, and the check is already
the thing that would have to change.

**The grow carries the old spans forward and zeroes only the new tail.**
`VoxelHashMap::resize` preserves every block's index — which is precisely why
`VoxelBlockGrid::topology_epoch()` deliberately does not move across one — so a
slot means the same block on both sides and replacing the table wholesale would
discard every span on the one event the volume tier guarantees they survive.
`TsdfIntegrator::prepare_dirty_flags` grows its flags exactly this way, for
exactly this reason. Zeroing the tail is the other half: only blocks in an
extract's active set are written, which on a first extract is a few thousand of
1.5M entries, and the rest were being published as readable while holding
whatever VMA handed over.

**And the slot itself is anchored, not just the table.** The generation above
answers "which extract does this table describe"; it cannot answer "does this
slot still name the block its span was written for". A slot is meaningful only
against one grid at one topology — the block heap is LIFO, so after a `remove()`
a reused slot names a *different* block and its span reads as that block's
geometry under `Status::ok`. `block_span_valid(grid, slot)` is that check. It
takes the **grid** rather than trusting the caller to re-extract after a topology
change: between a `remove()` and the next extract the per-slot stamps are still
set, so a query that re-checked nothing would call a stale span live. A
`resize()` deliberately does not break it, for the same reason the buffer carries
its spans forward. The stamps are per slot so that "meshed and emitted nothing"
stays distinct from "this extract did not mesh this block", and they are written
from the active set the *host* dispatched rather than read back from the device,
which only knows where geometry went, not which blocks it was asked about. The
fixture had to build its own grid and extractor and run last, because removing
whichever block compaction happened to put last changes nothing when that block
carries no surface, and a test that fails on some runs is worse than one that
fails on all of them; and which slot is unmeshed is not guessable, so the
assertions count instead — exactly the active blocks are valid, no more and no
fewer.

**A stamp is compared for equality with the extract that published the table,
not for being set — and the anchor is one globally unique token, not a pointer
and a count.** The first cut got both halves wrong in the same direction, and
each was a fact the library could check written down as something the caller
would not do.

The stamp first. It was tested `!= 0`, which answers "has this slot ever been
meshed since the anchor" — and nothing clears a stamp on the way past, so a block
that drops out of the active set keeps its last extract's stamp while the span
under it goes on naming an arena range some other block has since been given.
Comparing against a per-extract serial makes the question "did the extract whose
table is published write this", which is the only reading a caller can act on. It
also *removes* work rather than adding it: with a serial, a re-anchor needs no
clearing pass at all, because a stamp written before it can never equal the
serial after it. The reachable failure today is narrow — every topology change
re-anchors — but dirty-only dispatch, the branch this landed on, makes meshing a
strict subset the steady state, and there `!= 0` is wrong on every frame.

The anchor second, and this one moved a member between tiers. It was a
`const VoxelBlockGrid*` beside the epoch, and it could not work: this class
hands out no way to un-anchor and must never dereference a borrowed grid, so a
grid destroyed after an extract leaves a dangling address that the *next* grid
built in that storage matches exactly — and a fresh grid's epoch was 0, the same
value the anchor had recorded. Both halves match, and a destroyed grid's spans
read live. The fix is to make the epoch itself identify the table: it moved from
`VoxelBlockGrid` down to `VoxelHashMap`, and each value is drawn once from a
process-wide counter at `create` and at every `remove()` / `clear()`. No two
tables and no two topologies ever share one, so equality already means "the same
grid, unchanged" and the pointer is gone. Moving it down a tier fixed a second
hole in the same stroke, one the previous entry above records as a known trap the
*test* routed around: `VoxelHashMap::remove` reached through
`VoxelBlockGrid::map()` frees an index, and a counter owned by the grid was
bumped only by the grid's wrapper — so the raw path defeated every anchor built
on it in silence. Owned by the object that frees the index, no path can miss it.
It also deleted the class of bug that `VoxelBlockGrid`'s hand-written
move-assignment had already fallen into: it assigned four members and not the
epoch, so a move-assigned grid took on another grid's blocks while still
reporting the destination's old epoch. There is no such member to forget now —
it rides inside `map_`. The cost is that the token counts nothing; it is
comparable for equality and that is all, which is all any of its readers wanted.

**What the table does not do is clear itself, and the accessor has to say so.**
A grow copies every existing span forward and zeroes only the tail, so `block_spans()`
publishes well-formed entries for slots the current extract never wrote. The
first cut's prose said the opposite — "a slot the last extract did not mesh reads
as an empty span here" — which is the wrong instruction to hand a reader, and
worse than no instruction: an empty span is what a block that meshed and emitted
*nothing* writes. There is no value in the table meaning "not mine", which is the
whole reason the stamp exists, so the accessor now documents itself as a fetch
mechanism for slots `block_span_valid` has already approved rather than as an
array to iterate. Zeroing the whole buffer on a re-anchor was considered and
rejected: it is a 24 MB write to protect a use the contract forbids, where the
tail zeroing it resembles guards *uninitialized* memory, which is a different
thing.

**And what the feature costs is reported, including the half that is not a
buffer.** The stamps are a host array of `num_blocks * 8` — 12 MB at the same
defaults, on top of the table's 24 MB — and `resident_output_bytes()` counted
only the table, so `ExtractTimings::arena_bytes` under-reported the feature by a
third on the instrument the ring's runaway growth was diagnosed with. Both are
counted now, and the gated-vs-ungated test asserts the difference as an equality
over both terms rather than one. The per-block stamping loop is under the same
gate as everything else about the table — it was running for every caller,
`track_block_spans` or not, ~107k divisions per extract to discard the result —
and its cost lands in `arena_alloc_ms`, because `total_ms()` sums the six phase
rows rather than measuring the call, so work outside a row is time nothing
reports.

**And the host/device mirror is pinned per field.** `BlockSpan` is four
same-typed `uint32`s, so every permutation is 16 bytes and `sizeof` alone cannot
see a transposition — while the GLSL side was an anonymous `uvec4` both kernels
filled positionally, with `s_ibase` (named for the *index* base) landing in
`triangle_base` and correct only because the atomic result had been divided by
`kIndicesPerTriangle`. The struct now has per-field `offsetof` asserts like every
other mirror in the repo, is declared once in
`shaders/marching_cubes_block_span.glsl` rather than copied into two kernels that
could disagree on field order and both compile, and both kernels assign it **by
name** — the same fix `SparsePushConstants` already applies to its own adjacent
same-typed scalars.

### 2026-08-11 — An incremental extract trusts one struct, cleared on every path and re-established only where a mesh is handed out; every refusal is a silent fallback, and the fallback is *reported*.

*Amended 2026-10-06 (below):* incremental mesh extraction and the arena
state this entry describes are removed.

**Stage 3 of the 2026-08-09 incremental-extraction plan.** Blocks whose
`+{0,1}³` neighbourhood a fuse did not change keep the triangles they already
have, at the offsets `block_spans()` names, and cost one workgroup that returns
before gathering a corner. The dilation from the *changed* set to the *re-mesh*
set runs on-device on the eight blocks `s_neighbour` already holds — no
readback, no set union, no upload — which is the same move the 2026-08-08
neighbour-table decision made and for the same reason. A changed block re-meshes
into the range it already owns when the new count fits and appends past the
watermark when it does not; the range it leaves behind is retired by writing
zero-area triangles over it, so no prefix sum and no index rebuild is needed and
the run stays the identity.

**What an incremental pass is allowed to trust is one struct, and it is cleared
before it is read.** The state is `{watermark, epoch, serial}` — how many
triangles the arena holds, the grid topology they were written against, and the
span serial of the extract that wrote them — and the three are only ever
meaningful together. It is snapshotted and then *cleared* at the top of both
extract paths, beside the span generation, and re-established only on the
publishing return. That shape is the fix for a class of defect the first cut had
four instances of, each with the same shape: a scalar cleared on one branch and
left standing on every other. The dense overload rewrites the same arena with a
kernel that knows nothing about blocks, and cleared the span generation but not
the watermark. An unrecoverable overflow, and a refit that had already freed the
old arena, both returned above the line that reset it. Cleared up front there is
no exit that can strand it, and no branch to audit.

**A guard compared after the thing it guards is not a guard.** The clause meant
to catch a topology change read `span_epoch_ == grid.topology_epoch()` *below*
`ensure_block_spans(grid)`, which assigns exactly that — so it compared a value
with itself and passed unconditionally. A `remove()` re-draws the epoch and puts
the freed indices back on a LIFO heap, so a slot then names a different block;
the pass would have re-meshed against another block's range under `Status::ok`.
The predicate now runs *above* the re-anchor and reads the snapshot rather than
the member. The general rule: a cache anchor is compared before the call that
maintains it, or it is not compared at all.

**Both ends of the flags carry the anchor, because only one end can check it.**
`tsdf` publishes the flag buffer for an on-device consumer, and `mesh` cannot
include `tsdf` to ask whether it is stale. So `TsdfIntegrator` refuses on
everything it *can* see — `dirty_flags_buffer()` returns null once the
topology-stale latch is set, and the capacity and epoch go null with it, since a
live capacity beside a null handle is a caller binding one against the other —
and publishes `dirty_epoch()` for the one thing it cannot: *which grid*. That is
the same globally-unique token the span table anchors on, so one comparison in
`mesh` covers both "the right grid" and "no blocks removed since". The flags are
also range-checked and bound with the byte range the push constant claims rather
than `VK_WHOLE_SIZE`: the kernel reads every slot below `dirty_capacity`, and
with `robustBufferAccess` enabled nowhere in this repo an over-stated capacity
would be an out-of-bounds device read. Bound to the stated range it is a
descriptor whose range exceeds its buffer, which validation names at the binding.

**A refit retries as a full pass.** The overflow path reallocates the arena
(discarding the clean blocks' triangles), and attempt 0 has already overwritten
the span table for every block it touched, so no block can still read the range
it held *before* the call. Three reasons, one fix: the flag is cleared on the
refit path, so the retry re-establishes the arena, the spans and the watermark
together. Relatedly, the draw command's seed became a *parameter* rather than a
member the reset consumes — both of `ensure_output_buffers`' range guards sit
above the reset, so a latched seed survived a failure and was inherited by a
later dense or empty extract over an arena it had rewritten, which is precisely
the staleness the reset exists to prevent. Passed explicitly, there is nothing to
strand. The same argument moved the pending flags from a member the wrapper armed
and disarmed around a delegated call to a parameter on a private impl: an
exception unwinding out of a function that allocates a vector, a 12 MB stamp
array and several `std::string`s left a caller-owned `VkBuffer` latched on the
extractor, which the *plain* `extract_device` would then bind.

**Occupancy is not density, and feeding one to the other ratchets.** The arena's
triangle count on an incremental pass is what it *holds* — clean blocks, appends,
and every retired degenerate — while the density estimate that sizes the next
plan wants what the surface *has*. Feeding occupancy back made the estimate
monotone non-decreasing, so each pass planned a larger arena than the last on
evidence its own dead triangles produced, grew geometrically to 1.5x that, and
never corrected. The live count is summed off the span table over the active set
— a clean block's entry is exactly its current range — inside the loop that was
already stamping serials. With the live count in hand the same measurement bounds
the memory: when occupancy passes `kMaxArenaOccupancy` (2) times live, the state
is *withheld*, which makes the next extract full, and a full pass rewrites the
arena from zero. The compaction is the fallback machinery, reused.

**Every refusal is a silent fallback, so the fallback is reported.** Falling back
rather than refusing is right — a caller fusing a live scan cannot predict a
topology change, and the correct response to one is to re-mesh everything — but
every clause is invisible from outside, and two of them (`share_vertices`, a
`slot_count` above one) turn the feature off *permanently*. `ExtractTimings::
dispatches` cannot carry it: it counts refit rounds and reads 1 on both paths.
*Amended 2026-08-11 (below):* `share_vertices` is no longer one of them — the
sharing kernel reuses its two ranges and retires more cheaply than the default
one, not less — so `slot_count` above one is the only clause left that turns
the feature off permanently. The reporting argument this entry makes is
untouched, and is what made the reversal measurable rather than hopeful.
So `incremental` says which pass it was and `remeshed_blocks` says how much it
saved, the latter counted on-device because the dilation happens in shared memory
the host never sees. This is the 2026-08-04 rule's second half: a library that
checks what the caller cannot see, and then declines to report the answer, has
converted a caller-visible error into an invisible perf regression.

**`slot_count == 1` only, and that is a real limitation rather than a
simplification.** A re-meshed block writes into the arena the *last* extract
filled, and a ring hands this one a different slot, so the retained triangles are
in the wrong buffer. `fuse_viewer` — the seam-B consumer this whole staged effort
names as the target — runs `frames_in_flight + 1` slots, so the feature is off
there today. Extending the ring (reuse a block's range only for slots released
through `released_through_`, or copy the retained run forward into the newly
claimed slot) is a `TODO(mesh)` on the class rather than a thing this slice does,
and `ExtractTimings::incremental` is what keeps it from being invisible
meanwhile.

**And the benchmark had to be fixed before it measured anything.** The fuse
kernel only ORs into the dirty flags, so without a reset every block reads dirty
within a few frames: `fuse_replica --incremental` re-meshed everything through
the incremental path while paying for the span table, the dilation and the
retirement, and the headline numbers (0.98 ms incremental against 0.94 ms full;
505 511 arena triangles for 277 506 live) were the 100%-dirty worst case rather
than the feature. The reset now happens immediately after the extract that
consumes the flags, `--dirty-every` is *refused* beside `--incremental` because
the two own the same flags on unrelated cadences, and `--incremental` turns on
`--device-extract` rather than being silently ignored beside it while still
allocating the table. The room0 figures above stand as the worst case they always
described; the ~4x lives at the iPad's 25% dirty rate and is still unmeasured
here.

### 2026-08-11 — Incremental extraction runs under `share_vertices`, because that kernel owns its index run and so retires *more* cheaply, not less (reverses the `share_vertices` clause of the incremental-dispatch decision above).

*Amended 2026-10-06 (below):* incremental mesh extraction is removed;
`share_vertices` stays.

The stage-3 entry above refuses `share_vertices`, on the grounds that a
relocated block cannot retire the range it leaves behind unless it owns its
vertices three-per-triangle. That argument is backwards, and the refusal cost
the one configuration this feature exists for.

**What the refusal cost.** The device measurement made this the blocker rather
than the seam-B ring. Measuring the incremental extract on the iPad meant
turning sharing off, which cost 3x the vertex arena on the one device where
that ceiling is real: **4 177 MB, against 33 MB for the same app in its normal
configuration**. Nothing else on the list came close in severity. The iOS
scanner runs sharing because the vertex arena is the term that binds there.

**Why the argument inverts.** Retirement is an order of magnitude *cheaper*
under sharing. This kernel writes its own index run, so a dead triangle is
retired by pointing its three indices at one vertex — 12 bytes, zero area,
culled before rasterisation — where the default kernel, whose run is the
identity it cannot touch, must overwrite 192 bytes of vertices to say the same
thing. The dead *vertices* need no writing at all: sharing is in-block and the
`+face` is duplicated, so once no triangle references them they are unreachable
rather than merely unused. The retirement cost that argued against relocation
in the default kernel mostly disappears here.

**What sharing actually adds is a coupling, and it is one test.** The two
ranges are reserved independently and a triangle indexes into the vertex range
beside it, so a block reuses in place only when **both** counts fit and
relocates both when either does not. Reusing one while relocating the other
would leave the kept range's triangles naming vertices that moved. That is the
whole of the difference from the sibling kernel, where `v = 3t` makes one test
serve both — and it is why `ArenaState` carries a `vertex_watermark` that is
its own number rather than `3 * watermark`, and why the vertex counter in the
scratch word past the draw command is *seeded* on an incremental pass exactly
as `indexCount` is.

**Occupancy is asked on both axes.** `kMaxArenaOccupancy` withholds the arena
state — which makes the next pass full, which is the compaction — when the
arena has drifted past 2x its live surface. Under sharing that is two questions
rather than one asked twice: a relocating block's triangles are retired as
degenerates that keep occupying index slots, while its vertices are simply left
unreachable and never written, so the two buffers drift apart and either can be
the one carrying the dead weight. Withholding on the worse of the two compacts
whichever it is. Without sharing the second test is the first restated and
changes nothing.

**Measured on room0 with `--share-vertices --incremental`:** the arena holds
**310 312 triangles for 277 506 live in 39.5 MB — 1.12x inflation, against
1.82x without sharing**. The extract's mean is 0.98 ms, unchanged, because
room0 re-meshes 81.67% of its blocks per window and so cannot show the win;
that number still lives at the iPad's 25% dirty rate and is still unmeasured.

**What the reversal is verified against, since a test that cannot fail is worse
than none.** The decisive fixture — the field changed under the extractor while
the flags still say nothing moved, so all-zero flags must return the OLD
surface and all-set flags must reproduce a full extract once retired
degenerates are dropped — now runs over **both** kernels. It did not before:
the loop that was supposed to select the emitter never read its own induction
variable, so `share_vertices` stayed false and the two iterations were the same
kernel run twice, with `-Werror` silent because the loop header used it. Wiring
it up failed immediately on `remeshed_blocks`, which this kernel declared and
never incremented — the counter had been added to the sibling alone, on the
assumption encoded in the very refusal this entry reverses.

**Two hazards the reversal had to close, both invisible to a caller.** A
triangle dropped for a vertex-claim overflow *inside a reused range* used to
keep the previous extract's index triple: still in range, so nothing faults,
but naming three unrelated vertices of the new surface and drawing a full-area
triangle across the block — and unreachable by the retire pass, which starts
past the live sub-range by construction. It is retired in place now. And the
dilation's `s_dirty` had no barrier between invocation 0 zeroing it and
invocations 0..7 OR-ing into it, so a lane that ORed early had its bit
clobbered and a changed block took the clean early-return, keeping stale
triangles for a surface that had moved. Latent where lanes 0..7 share one SIMD
group, as on Apple; live anywhere else. The sibling kernel had the barrier at
exactly that point.

**What this does not settle.** A reuse rewrites the span with the *live*
counts, which is the definition `block_spans()` consumers and the occupancy sum
both need, but it means a block that shrinks and grows back inside its original
reservation relocates where it could have stayed. Bounded rather than
unbounded — occupancy withholds and the next full pass compacts — so what is
unknown is the rate. Recording the reservation beside the live count (a private
table indexed per slot; not the four published span fields, whose ABI is pinned) would
remove it, and is worth building only once relocation is measured to be common
rather than rare. room0 cannot supply that measurement: at 81.67% dirty, nearly
everything relocates there for reasons unrelated to this. Marked `TODO(mesh)`
at the span write.

### 2026-08-12 — Meshing a camera's view is a caller-supplied block list, not a camera the mesh tier holds; and it stays apart from incremental extraction rather than stacking with it.

*Amended 2026-10-06 (view culling, below):* the caller's list is now the
device-resident `DeviceBlockList` from
`compact_active_blocks_in_frusta_on_device`, bound in place; the host
`BlockList` overload and the host `compact_active_blocks_in_frustum`
overloads are gone. A culled pass now records its density, reversing "A
culled pass also measures nothing" below. "No in-tree example culls yet" is
no longer true: `fuse_viewer` and `rig_viewer` mesh their render camera's
view. The rest stands: the set arrives and the camera does not, the cull edge
does not hole, refusals are rollbacks, and the win is still unmeasured.
*Amended 2026-10-06 (below):* incremental mesh extraction is removed, so the
"kept apart from incremental extraction" clause has nothing left to keep
apart. The culled overload and its density guard stay.
*Amended 2026-10-06 (below):* fusion now consumes the frustum compaction too:
`TsdfIntegrator::integrate` culls to the scanning cameras' frusta, on the
device and in one scan for a rig's set, which the header had claimed and no
code did.
*Amended 2026-10-13 (the volume block-list entry, below):* the host `BlockList`
this entry gave a job, and `VoxelBlockGrid::block_list`, which stamped its
epoch, go; `DeviceBlockList` carries the same epoch.

A scanning device renders a small part of a large volume — the iPad case, where
the viewport shows a room corner while the map holds the whole floor. Meshing
the whole active set to draw a fraction of it is the obvious waste, and two of
the three pieces to stop it were already here: the `volume` tier's
`compact_active_blocks_in_frustum` culls the active set on-device against six
planes, and `hash_compact_frustum.comp`'s conservative p-vertex AABB test has
been proven since the frustum decision landed. What was missing was a way *in*:
`MarchingCubes` compacted the whole map itself, on the first line of the one
funnel every extract entry point reaches.

**The set arrives; the camera does not.** The alternative — hand the extractor
the frustum planes, or a `DepthCameraParams`, and let it cull — would import a
camera, and with it a policy (which frustum, widened by how much, culled against
what near plane) into a tier whose job is turning voxels into triangles.
`MarchingCubes` has no camera vocabulary today and gains none here. A block list
is also strictly more general: a region of interest, a chunk queue and a
level-of-detail selection are the same call, and nothing in the extractor knows
or cares that a frustum produced this one. The unused `volume::BlockList`
declared in `hash_types.hpp` since the port is what it arrives as — the type was
right; it had never been given a job.

**The render camera has no pinhole, which is why `make_frustum_planes` grew a
second constructor.** The existing one takes intrinsics in pixels under the
OpenCV convention the fusion tiers pose a depth camera in. A viewport has
neither: it has `proj * view`, in whatever handedness the renderer chose. The
new overload reads the six planes off that matrix directly (Gribb-Hartmann), so
it holds for any convention the caller combined — with one hard requirement,
that depth maps to `[0, 1]`. gfx forces `GLM_FORCE_DEPTH_ZERO_TO_ONE`, so its
`Camera::view_proj` qualifies; a GL-convention matrix passed here reads the near
plane as `row2` where it should be `row3 + row2`, which normalizes to
`z >= 2nf/(n+f)` — the harmonic mean of the caller's near and far, always
between `n` and `2n`. So it does *not* clip at the eye, as this entry and the
header both claimed until the review: it over-culls a shell about one
near-distance thick in front of every camera. That is exactly the kind of wrong
that ships, so the test pins it — the same scene, the same frustum, built both
ways. The first cut of that test did not: rows 0, 1 and 3 are bit-identical
between the two conventions and the far plane normalizes the same under both, so
*only* the near plane moves, and no fixture block lay in the 0.1–0.196 m band it
moves through. Substituting `perspectiveRH_NO` left every assertion green. A
block at world z `[0.12, 0.16]` is the witness, and is now allocated for that
section alone.

**The margin is a distance in metres, and it is not decoration.** The pinhole
constructor widens its lateral planes ~10% by scaling the focal lengths — a
unitless nudge that means different distances at different depths and fields of
view, and that reaches only four of the six planes. The matrix overload takes
`margin_m` and applies it *after* normalizing, where `d += margin_m` moves a
plane outward by exactly that far. It defaults to 0, the exact frustum. Non-zero
is what an asynchronous consumer needs: fusion extracts on a background thread,
so the mesh a frame draws was culled against a pose several frames old, and an
exact frustum pops at the screen edge while the camera turns. Naming it in
metres lets a consumer size it from how fast its camera can move, which a focal
scale cannot express.

It reaches five of the six planes, **not the near one**, and that exclusion came
out of the review. Applied to all six, the near plane sits at world
`z = z_near - margin_m`, so any `margin_m > z_near` puts it behind the eye — and
because the *lateral* planes were widened by the same amount, what comes through
is not a sliver but a cone: measured against the shader's own p-vertex test on
the test camera, a 0.5 m margin on a 0.1 m near plane admitted blocks up to
0.36 m behind the camera and 0.56 m off-axis, and a 1.0 m margin reached 0.88 m
back. The doc that leads a consumer here points straight at metre-scale margins,
and `fuse_viewer` already builds its camera at `z_near = 0.05`. Excluding near
also matches the pinhole overload, which widens only its four lateral planes.
The cost is a block crossing `z_near` while the camera moves forward, which pops
at the near plane rather than at the screen edge — the smaller of the two
failures by a wide margin. The test pins it on both axes: a block 0.08 m behind
the eye stays culled at a 0.5 m margin, and every plane's `d` is read directly
to confirm five moved by exactly the margin and near moved by nothing.

**A degenerate matrix keeps blocks rather than dropping them.** The normalize
guard was `len > 0.0f`, which is false for a NaN length as well as a zero one,
so both fell through to the "keep it unnormalized" branch — and then still took
the margin, on a plane with no unit normal for a metre to mean anything against.
On an all-zero matrix that reduces the shader's test to `margin_m >= 0`: at any
negative margin it rejects everything, an empty mesh every frame under
`Status::ok`. None of those inputs is exotic. `Mat4f{}` *is* the all-zero matrix
(`vector_types.hpp` already warns that the identity must be spelled
`Mat4f(1.0f)`), `glm::infinitePerspectiveRH_ZO` emits an exactly-zero far row,
and a zero-width viewport puts `inf`/`nan` through the whole product — where
glslang's ordered comparisons make every NaN plane accept nothing. The guard is
now explicit and adds no margin in the degenerate case, so the cull fails toward
keeping blocks, which is the direction that shows up as a slow frame rather than
a missing surface.

**A bare scalar is a compile error.** glm's scalar-diagonal `mat4` constructor
is implicit unless `GLM_FORCE_EXPLICIT_CTOR` is defined, which this repo does
not, so `make_frustum_planes(0.5f)` compiled — as `Mat4f(0.5f)`, a unit box at
the origin — and `make_frustum_planes(100.0f, 100.0f)` compiled as a frustum
that keeps everything, both clean under `-Wall -Wextra -Wpedantic -Werror`. A
deleted single-argument template rejects exactly those and loses the overload
tie to the real `const Mat4f&` on a matrix. What it cannot catch is a
`cam_to_world` passed by mistake — same type, and the natural pattern-match
against the pinhole overload's ninth argument — so that one is a `@warning`.

**The cull boundary does not hole, and that is the 2026-08-08 probe decision
being paid back.** The sparse kernel resolves each block's 2×2×2 neighbourhood
by probing the hash table on-device, so a block on the edge of the supplied list
still samples correct corner values out of neighbours that were never
dispatched; the surface simply ends there. The host-built neighbour table this
replaced was fed from a compacted snapshot and would have produced a seam of
dropped cells all along the frustum edge. The coupling that entry called "paid
for, not stumbled into" is what makes this feature a parameter rather than a
project.

**Anchored by `topology_epoch`, the same token the spans and the dirty flags
use.** A `BlockIndex::ptr` indexes the attribute arrays directly and the block
heap is LIFO, so a `remove()`/`clear()` between the caller's compaction and the
extract has handed those pointers to *different* blocks. Every one of them is
still in range, so without the check the extract meshes whatever voxels now live
there and returns `Status::ok`. `BlockList` therefore carries the epoch it was
compacted at and the extractor refuses a mismatch — the 2026-08-04 rule, and the
third consumer of the token the 2026-08-11 table entry made globally unique.
What is deliberately *not* checked **on the host** is that each `ptr` is one this
grid handed out: that is O(count) of host work per frame — ~107k entries on
room0 — to catch a caller who fabricated a list rather than compacting one.

The review found that reasoning half right. The epoch does not catch a
**count** that outran its array: `topology_epoch` moves only on
`create`/`remove`/`clear`, never on allocate or resize, so a consumer that
re-compacts into a shorter vector while a cached `BlockList` keeps last frame's
larger count passes every check — and the list is *honestly obtained*, which is
the case the argument above assumed away. Nor is the consequence "meshes the
wrong voxels": the kernel stores `block_spans[ptr / vpb] = span`, so an
unbounded `ptr` is an out-of-bounds device **write**, with `robustBufferAccess`
enabled nowhere in this repo. The bound the deleted `compact_active_blocks()`
path gave for free (`collect_compacted` clamps its own count to `num_blocks`,
and a vector's `data()` and `size()` agree by definition) is now taken
explicitly, in the two places each costs nothing: `count <= num_blocks` is one
O(1) host comparison, and the per-block half is one comparison per *workgroup*
on-device, against a new `num_block_slots` push constant. The active-block
buffer also picks up the `check_storage_buffer_range` and the exact-range
binding that both other caller-sized bindings in the same function already had.

**Duplicates are the caller's to avoid, and now say so.** Nothing rejects a list
naming the same block twice — a duplicate is something the caller can see, and a
set-wise test is the O(count) cost declined above — but the consequence is
silent, so it is a `@warning` rather than an omission: one workgroup runs per
entry and each reserves its own arena range, so the block's surface is emitted
twice (measured: 132 triangles becomes 264) and the two race to write its span,
leaving the loser's range live but undescribed by the table. Neither internal
producer can emit one; a caller unioning two cameras' compactions can.

**An empty list is legal, including the default-constructed one.** `epoch`
defaults to 0 and `next_topology_epoch()` never returns 0, so an unconditional
epoch check made `BlockList{}` — the natural spelling of "the cull kept nothing"
— the one value that could never be accepted, refusing on exactly the frames a
camera sees nothing while the fully-spelled empty list worked. An empty list
names no block, so there is nothing about it that can be stale; the epoch check
is now skipped when `count == 0`.

**Every one of those refusals is a rollback**, and the contract now says so.
They are checked above `claim_output_slot()` and the generation bump, which the
implementation comment already called the whole point — but the `@return`
forwarded the caller to `extract`'s `@warning` that "a failure is not a
rollback", publishing the opposite of the guarantee the code provides. It
matters for exactly the consumer this feature exists for: culling a frame behind
means hitting the epoch refusal on *every* frame after a `remove()`, and
re-extracting and redrawing on each one costs more than the cull saves.

**And the `share_vertices` block-size refusal moved above the empty-set return.**
It was below it, which was harmless while an empty active set meant an empty
map. An empty *cull* is a routine per-frame outcome, so the misconfiguration
would have surfaced only on frames the camera happened to see geometry —
`Status::ok` while pointed at nothing, `InvalidArgument` a frame later, reading
as though the camera move caused it. It is a property of the config and the
grid, so it answers the same on every frame.

**The triple is built by `VoxelBlockGrid::block_list`.** Pointer, count and
epoch came from three separate expressions at every call site, including this
tier's own tests, while the doc asked for them to be paired "in the same breath"
— a discipline no signature enforced, on a seam whose only real consumer is
out-of-tree. One accessor takes the vector and stamps the grid's current epoch,
so the epoch cannot be fetched from a different grid, read a frame early, or
forgotten (and forgetting it is not a compile error: the default is a value no
live grid equals). What it does not fix is the *lifetime* — the list still
borrows the caller's vector — which is a `TODO(volume)` for an owning
`CompactedBlocks` returned straight from the compaction entry points.

**Kept apart from incremental extraction, deliberately.** The two are separate
answers to the same cost, and stacking them gives most of one back. An
incremental pass keeps the triangles of every block it does not re-mesh, and a
block outside a culled set is simply not dispatched — so it keeps its geometry
below the watermark and goes on being drawn. Correct, but it surrenders the
arena and draw savings and leaves only the cheaper dispatch, while the arena
converges on the union of everything ever visible. A full culled extract seeds
the draw command at zero and rebuilds the arena from the visible set alone,
which is what a memory-bound device actually wants. So the overload is offered
on `extract_device` and not on `extract_device_incremental`, and
`blocks == nullptr` is a clause of the incremental predicate rather than an
invariant left to the call sites — a future entry point that offers both gets
this tier's documented answer to a combination it cannot serve (a full extract,
reported as `ExtractTimings::incremental == false`) instead of a surprise.

**That clause guarded the wrong scope, and the review found it by running it.**
Keeping the two apart *within one call* is not the hazard; the hazard is the
*sequence*. A culled extract published `arena_state_` exactly as a full one
does — `over_occupied` is provably false off the incremental path, since `live`
is initialised to `emitted` — while stamping and rewriting spans for only the
blocks it was handed. Every other block kept a range describing the arena that
pass had just rebuilt from zero. The next `extract_device_incremental` then
passed every clause: watermark live, epoch unmoved, and `serial` equal, because
the serial it compares is the culled call's own (`ensure_block_spans` bumps it
*below* the predicate). Reproduced on MoltenVK by two reviewers independently:
a culled extract over half a 216-block sphere followed by an **all-clean**
incremental returned 3 806 of 7 388 triangles with `Status::ok` and
`ExtractTimings::incremental == 1` — half the surface silently gone — and with
the culled-out half marked dirty it wrote over 2 472 *live* triangles belonging
to blocks that pass had promised not to touch, plus 1 751 triangles the surface
does not contain. The kernel reads `block_spans[slot]` blind: `span_stamp_` is a
host-side array with no device mirror, so nothing on the GPU can tell a stale
span from a current one.

So the invariant belongs on the *published state*, not on a per-call boolean:
**a culled extract withholds `arena_state_`**, exactly as the dense `extract()`
path clears it and for the same stated reason ("nothing in the span table says
'not mine'; this is what does"). The next incremental request then falls back to
a full extract and reports it, which is the documented behaviour for everything
else this tier cannot vouch for — so alternating the two entry points is *safe*
and costs one full extract per switch, rather than being a combination a caller
must know to avoid. The `blocks == nullptr` clause on the predicate stays, now
as belt-and-braces rather than as the guard.

**A culled pass also measures nothing.** `tris_per_block_` is a density, and the
next call multiplies it by *its* active set, so it is sound only while the set
it was measured over is drawn from the same population as the set it will scale.
A frustum cull is systematically denser per block — it keeps camera-facing
surface blocks and drops the far, back-side and empty truncation-band ones — so
feeding it back poisons the next full extract's plan. Measured: one culled
extract over a single dense block took the following full extract from a 20 736
triangle plan and a 4.2 MB arena to 74 196 and 15.1 MB, a 3.6x blow-up held for
the extractor's lifetime, since the arena is grow-only. Both density figures are
now recorded on the full path only.

**And `live_sum` only counts spans the previous extract vouched for.** It was
guarded by capacity alone, so it summed the stale ranges of blocks the last pass
never dispatched — inflating the occupancy denominator, which is the wrong
direction twice: `over_occupied` is what forces the compacting full pass, so an
inflated `live` is precisely what stops the recovery from firing. It compares
against `prev_arena.serial`, read before the stamp loop overwrites it;
comparing against `span_serial_` in that position would be a tautology.

**What it does not make cheaper.** The frustum compaction still scans every
hash-table slot; its dispatch is sized by the table, not by the survivors. What
shrinks with the visible fraction is the upload, the marching-cubes
dispatch, the arena, and whatever draws or textures the result —
`ExtractTimings::compact_ms` reads 0 on the new path because no compaction
happened *there*, not because one got faster. Two rows named in the first cut do
**not** shrink and have been struck from the claim: `readback_ms` is the 20-byte
draw command alone and `descriptor_ms` is a fixed set of descriptor writes, both
per-call constants. In a repo whose standing rule is that three bottleneck
guesses have already been wrong, advertising a fixed-cost row as scaling is the
claim most likely to be acted on. For the same reason the arena wording is now
"no live bytes" rather than "no bytes": the arena is grow-only, so one full
extract — a warm-up frame, a pose that is not ready, any documented fallback —
sizes it for the whole active set and it never shrinks back.

**Verified against a partition, not against a smaller number.** Meshing a subset
and observing that it is smaller passes for a kernel that drops the seam. The
test splits the 6×6×6-block sphere's active set into two complementary halves,
meshes each through the new overload, and requires the two canonical triangle
sets to merge back into the full extract *exactly*: every cell's base corner
lies in one block, so the halves must be disjoint and exhaustive. A dropped seam
triangle is a short merge, a duplicated one a long merge. Handing over the whole
active set is checked to reproduce the internal compaction bit for bit, and the
epoch refusal is exercised by removing a block under a list already taken.

**What is unmeasured.** Everything above is correctness; the win is not
quantified in-tree, because no in-tree example culls yet. `fuse_viewer` is the
natural place and needs its render camera published across the fusion-thread
boundary — a viewer-architecture change, not a library one — and the consumer
that motivates the feature is the iOS scanner, in `volumetric_kit_ios`. The
figure to expect is roughly linear in the visible fraction on every row below
`compact_ms`, and nothing here should be quoted until a run says so; three
guesses at this pipeline's bottleneck have already been wrong (see "Measured
lessons").

### 2026-08-31 — The dense extract goes; `extract` becomes `extract_host`, so the two workflows are named rather than inferred.

Two things had ended up sharing the name `extract`, and they shared nothing
else. One took a caller-supplied flat array of `Voxel{sdf, weight}` over a
`DenseGrid` and ran a kernel of its own; the other meshed a sparse
`VoxelBlockGrid`. Different input model (AoS versus the SoA attribute arrays),
different kernel, different sizing policy. On top of that the host/device pair
was asymmetric — `extract` beside `extract_device` — so the unsuffixed name read
as the default when it is the *export* path, and the device path is what the
live consumer uses.

**The dense path was never ported heritage; it was rewrite scaffolding.** The
proven engine this repo re-implements has no dense entry point at all:
`MarchingCubesExtractor` exposes exactly two overloads, both taking a
`VoxelHashMap` — one a dirty-block subset, one a caller-supplied `BlockList`,
which are precisely `extract_device_incremental` and the culled
`extract_device` here. The only `DenseGrid` anywhere in that tree is an
unrelated mesh-to-SDF stress test. So the dense overload existed to prove the
per-cell kernel before the sparse path landed, and nothing outside the tests
ever called it.

**What it was actually costing.** It shared the ring and arena with the sparse
path, which its own comment flagged: a large dense call grows whichever slot it
lands on to the dense worst case (5 triangles/cell, three private vertices
each, no refit), and that slot keeps it for the extractor's lifetime — so a
process touching both entry points forfeits the fitted arena. It also generated
its own test surface: of the four dense uses in the sparse suite, **three**
existed only to pin what happens when a dense call mixes with sparse work
(retiring the span table, forcing an incremental fallback, growing one output
buffer). Those are problems that exist because dense exists.

**Removing it does not cost the marching-cubes oracle**, which was the first
argument for keeping it and was wrong. The sparse suite writes the analytic
sphere SDF straight into a real block grid — the attribute buffers are
host-visible and mapped — and checks radius, unit outward normals, winding, and
total surface area against `4*pi*r^2`. The prior engine validated its MC with
strictly less than that (radius tolerance, outward-normal ratio, index range)
and shipped.

**Nor does it cost the cross-block correctness proof.** That was the second
argument, and the better one, but the check that carries it does not involve
dense: the collision-chain case meshes the *same* field through a table shaped
to spill (2-entry buckets, `overflow_count > 0` asserted before it is trusted)
and requires the result match the reference **triangle for triangle** through
`canonical_triangles`. Its own comment names the failure mode — "surface
silently missing at block seams" — which is exactly what the dense equivalence
was there for. It is also the sharper instrument of the two: a bug in the
shared per-cell body moved both sides of the dense equality at once and left it
green, which the suite records as a real event.

**Nor is the independent-growth guard lost, and the first cut of this change
was wrong to say so.** "Growing one output buffer must not resize the other"
defends a real past bug — the two were reallocated together, which grew the
buffer that already *fitted* to 1.5x on every event, compounding into the ring
runaway the slot-independence decision exists to prevent. The argument for
parking it was that the case was reachable only through dense, since under
`share_vertices` a surface holds ~0.75 vertices per triangle and a sparse
extract always asks in that proportion. That is false: the arena is planned
from `verts_per_1000_tris_`, **learned from the previous extract**, so a first
extract over a thin shell teaches a ratio a dense field then breaks. The sparse
test now arranges exactly that on one `share_vertices` extractor with no second
entry point anywhere, and asserts both bounds of the window before it asserts
the growth, so the case cannot go vacuous if those capacities drift.

The lesson generalises past this guard: **a property is not dense-only because
the dense caller is the one that used to reach it.** Three of the four dense
uses in the sparse suite really were about dense mixing with sparse work and
went with it; this fourth one was about the two budgets being out of
proportion, and dense was merely the most convenient way to arrange that.

**The rename.** `extract` becomes `extract_host`, symmetric with
`extract_device` and matching the `Mesh` / `DeviceMesh` type pair, so the shape
is `extract_<destination>[_<mode>]` and every device-side mode clusters under
one prefix. `extract_host` is not a convenience wrapper and could not be
deleted in favour of `extract_device` + `download`: it also calls the private
`free_slot_of`, and a host-only caller has no generation to release with
(`Result<Mesh>` carries none) while the public `release_through` is the
*consumer's* high-water mark and would retire slots an `extract_device` caller
is still drawing from. Keeping a host entry point is what keeps a PLY writer
off the ring contract entirely.

### 2026-08-31 — A triangle's work unit is the candidate *block*, not the triangle; and the band it allocates is measured from the surface, not dilated from a point.

Putting a mesh into the volume — the mesh-to-SDF pass that comes next — first
needs the blocks the mesh's truncation band covers. (This was first planned as
the codec's input stage; the 2026-09-26 codec entry has the codec encode a
fused grid instead, so nothing here is on the codec's path.)
`allocate_from_points` over the vertices looks like it already does this and
does not: it dilates each point into the `(2*tb+1)^3` cube, which is **one
block** wide at the defaults (40 mm), so any triangle wider than that leaves an
unallocated hole through its middle — and a triangle wider than 40 mm is not an
edge case, it is what every mesh that is not a dense scan is made of. The test
pins the difference on a 0.6 m triangle: the block at its centroid is allocated
by the triangle path and absent from the point path.

**Why the work unit is a block and not a triangle.** The obvious kernel is one
thread per triangle, looping its own bounding box, and it is unusable here for
a reason that has already cost this repo a device: *triangle size is unbounded*.
A single 20 m floor quad at 5 mm voxels covers ~250 000 blocks, so one lane
would own a quarter-million bucket-locked inserts while its neighbours idle —
the same shape of dispatch that hung an M5 iPad in the 2026-08-08 overflow-scan
entry, and striding inside the lane does not fix it, because the lane still owns
every insert. So the host walks the triangles once, counts each one's candidate
blocks, and uploads the exclusive prefix sum; a work item binary-searches that
array for its triangle and handles exactly **one** block. Work per lane is then
a search, one point-triangle distance, and at most one insert, whatever the mesh
looks like — a property of the decomposition rather than of the input.

The host pass is not overhead reluctantly accepted; it is the only place two
other things can happen. It is where an index is **bounds-checked** — the kernel
indexes `vertices` with it directly and `robustBufferAccess` is enabled nowhere
in this repo, so an out-of-range index is an out-of-bounds read, not a wrong
answer (the 2026-08-04 rule: a limit the caller cannot see is the library's to
check). And it is where a **zero-area triangle** is dropped, which matters
because that is the one input that makes the closest-point solve divide by zero:
each of its three edge branches divides by a squared edge length, so excluding
degenerate triangles at the source is what lets the GLSL carry no per-lane
guard. A non-finite vertex is dropped in the same pass, for the plainer reason
that a NaN turns a bounding box into a block range of garbage extent. Neither is
an error — a mesh file routinely carries a few, and one bad face should not fail
the allocation.

**The band is measured, not dilated.** A block is allocated when its centre lies
within `trunc_dist` plus the block's half-diagonal of some triangle. This is a
deliberate divergence from what `allocate_from_depth` and `allocate_from_points`
do, and the divergence is in the input: those dilate a *point*, where the cube
is both the cheapest and the tightest thing available, while a triangle has a
real extent and its bounding box is mostly empty space. Without the distance
test a large slanted triangle allocates the solid interior of its box; with it,
the same triangle allocates a sheet. The test pins both directions — a far
corner of a diagonal triangle's box stays out, while the surface itself is
covered — because a prune that is merely conservative and a prune that does
nothing both pass a containment check. The half-diagonal is measured off
`(block_size - 1) * voxel_size`, the span of a block's *voxels*, since voxels
are node-centred; the nominal block extent would still be conservative, just
looser by a voxel per axis.

Two follow-on notes. The kernel **recomputes** each triangle's box rather than
reading back the host's, which is sound because `worldToBlock` mirrors
`world_to_block` bit-for-bit (the invariant `allocate_from_depth` already rests
on) and min/max/subtract are exact in IEEE — but the decode is guarded against
`local` exceeding the recomputed extent anyway, so a future divergence
under-allocates rather than decoding onto an unrelated coordinate. And the
closest-point primitive itself landed in **`core`**
(`core/shaders/triangle_common.glsl`) rather than in `volume`, for the reason
`core/math/vector_types.hpp` is there: it carries no tier's concepts, and the
mesh-to-SDF pass will evaluate the field with the same function, so the blocks
allocated here and the voxels written there cannot drift apart.

### 2026-09-26 — The TSDF codec is one `codec` tier over `volume`, with separate `Encoder` and `Decoder` classes: a geometry-only intra frame of per-block DCT coefficients, an observed-voxel mask and sorted block coordinates, entropy-coded by chunked static-table rANS.

*Amended 2026-10-13 (the volume block-list entry, below):* the transform takes a
device list, and `VoxelBlockGrid::check_block_list` and `block_list`'s
deleted rvalue overload, added here, are gone with `BlockList`.

The roadmap listed two later tiers here, `compress` (DCT) and `codec`/`stream`, a split
that came from the prior engine: an `ISdfCompressor` family for one volume and a
`FrameCompressor` for a sequence on top of it. What gets built first is a
**per-frame** codec. Each frame's volume is encoded on its own (intra only), and
a player decodes it into a `VoxelBlockGrid` that the mesh tier extracts unchanged.
The prior engine's working version (`src/compression/` in
`implicit_world_reconstruction`) is the reference for the algorithm. Its CUDA
kernels, its adaptive range coder and its nvcomp zstd streams are not ported
as-is (see below).

**One tier, named `codec`.** The prior split made the lower layer expose a
transform-only API (`transformForward` / `transformInverse`) purely so the upper
layer's temporal predictor could reach raw coefficients. In one tier, P-frames
are a later `FrameType` in the same bitstream, and they use the transform through
a private header rather than a public API made for one caller. `compress` leaves
the roadmap. `stream` stays separate, because transport and the RGB that travels beside the geometry
(below) are not the codec's business. The tier links **`recon_volume` alone**.
It reads a grid's `tsdf` and `weight` and, on decode, re-creates blocks through
`VoxelHashMap::allocate`. It does not link `tsdf`, which produced the field, or
`mesh`, which consumes the decoded one.

**`Encoder` and `Decoder` are separate concrete classes**, sharing a private
`DctTransform`. They run on different machines (the capture rig, the player).
Each will hold its own reference-frame state once P-frames exist, and a player
never needs the forward kernel. There is no virtual base: the codec is DCT-only
(2026-06-21), so an interface would have one implementation. `TsdfIntegrator`'s
runtime `IntegrationMode` flag is the precedent for a lean concrete class over a
strategy hierarchy.

**The frame is geometry only.** It holds a header, the block coordinates, a
1-bit-per-voxel observed mask and the coefficients. Color is excluded: the
player colors the decoded mesh by projective texturing from RGB frames that
travel beside the codec, and per-voxel RGB would be 1.5 KB per block raw, far more
than the geometry it would sit next to. The cost is named rather than left to
be discovered. A surface that none of the transmitted frames sees falls back to
the mesher's per-vertex color, and a grid with no observed color renders that
white (`marching_cubes_sparse_common.glsl`). The header carries a stream table,
so a later stream is a format version, not a break.

**Why a mask and not weights.** The mesher skips any corner whose weight is under
its threshold, and a freshly allocated block reads all zeros. So a decoded
volume must say *which* voxels were observed, or nothing meshes. It need not say
*how much*: weights matter only to a later fuse, and a player does not fuse. The
old engine sent uint8 weights, 512 bytes per block before zstd, and the mask is
64. The decoder writes weight 1.0 where the bit is set and leaves an unobserved
voxel as a fresh block holds it, with zero `tsdf` and zero `weight`. "Observed"
is `weight >= volume::kObservedWeight` (1e-6). The mesher's threshold is
defined from the same constant, since both tiers link `volume`, so the mask
cannot drift from the voxels the mesher uses.

**The transform.** It is the prior engine's, re-implemented in GLSL:
- **Shape:** an orthonormal 8³ DCT-II, separable, over each block's SDF
  normalized by `trunc_dist` (so every input is in [-1, 1]).
- **Coefficient order:** the first K in 3-D zigzag order, which sorts by
  `x + y + z` and then by `(x, y, z)`. The order is generated rather than
  pasted, and it reproduces the prior engine's table, all 512 entries, pinned
  by a hash of that table in `codec_params_test`.
- **Quantization:** separate DC and AC steps, expressed as fractions of
  `trunc_dist` so one setting holds across scenes. Rounding is half-to-even,
  and the result is clamped to ±32767. The params and `trunc_dist` travel with
  the coefficients in one `DctBlocks`, and the inverse compares that
  `trunc_dist` exactly against its grid's: a grid with another band would
  rescale every SDF and report success.
- **Unobserved voxels are filled, not zeroed.** The forward never reads an
  unobserved voxel's `tsdf`. Each one takes the value of the nearest observed
  voxel along x, then y, then z (the lower index on a tie), and a block with
  none stays 0. The decoder masks these voxels out, so their value is free to
  choose, and zero is the worst choice: a fused block's unobserved voxels read
  0, the iso level, and the step from the observed band down to it is energy
  the first 32 coefficients cannot hold. It leaks into the observed voxels
  beside the mask edge, which is where the surface is. A double-precision
  simulation at the defaults, RMS error on the observed voxels in units of
  `trunc_dist`:

  | block | zero | observed mean | nearest, x/y/z |
  |---|---|---|---|
  | plane, observed z < 5 | 0.073 | 0.086 | 0.031 |
  | tilted plane, observed in front + half a band | 0.153 | 0.116 | 0.055 |
  | 3³ observed corner | 0.109 | 0.120 | 0.010 |
  | 70% of voxels observed at random | 0.130 | 0.088 | 0.056 |
  | fully observed (control) | 0.010 | 0.010 | 0.010 |

  The observed mean, the fill this entry first listed as an open compression
  question, is worse than zero on the plane and the corner and never close to
  the line fill, so it was not taken. A
  harmonic fill beats the line fill only on the random mask and needs an
  iterative solve. The line fill's three passes use the same lane-to-line
  mapping as the transform. On the GPU the tilted plane decodes at 0.055
  against zero's 0.153. The format does not change, and neither does the
  decoder.
- **Every entry is checked on the device.** Lane 0 of each workgroup probes
  the hash table through `hash_lookup.glsl`, the mesher's probe, and requires
  the entry's coord to resolve to exactly its ptr. An entry that does not is
  neither read nor written, and the call is refused with the count. The probe
  replaces the O(count) host ptr scan this entry first had, which `mesh`
  declines to pay, and it catches two things a range check cannot. One is a
  free slot, which the inverse would write weight 1.0 into, so the next
  `allocate` would hand out a block that meshes as observed. The other is a
  coord paired with another block's ptr. The host keeps only the O(1) half,
  `VoxelBlockGrid::check_block_list`, which `mesh` now calls too. An inverse
  refused this way has already written every live entry.
- **Why the clamp never fires:** an orthonormal coefficient of inputs in
  [-1, 1] is bounded by √512 ≈ 22.63, so `CodecParams::validate` refuses a step
  below √512 / 32767. The clamp is then a guard against float drift, never a
  silent loss. That is the 2026-08-04 rule: a limit the caller cannot see is
  the library's to check.
- **Defaults:** K = 32 with a DC step of 0.25 and an AC step of 0.05. These are
  the prior engine's 0.01 m / 0.002 m at its 40 mm band, carried over until the
  room0 measurement tunes them.
- **Block size:** 8 only. The zigzag order and the cosine table are 8-specific,
  and another block size is refused rather than transformed wrong.
- **Dispatch:** one workgroup of **64** invocations per block, each owning one
  8-voxel row per pass. That stays under Vulkan's guaranteed 128 invocations,
  where the prior engine's 512 threads per block would not. Dispatches are
  batched past `maxComputeWorkGroupCount[0]` rather than refused, since room0's
  ~107 k blocks already exceed the 65535 floor.

**Entropy coding: chunked, static-table, interleaved rANS.** This reverses the
first plan, which was to port the prior engine's range coder. That coder is
adaptive (a Fenwick-tree frequency model with an escape symbol) and codes one
stream per coefficient index, so it is sequential within a stream and at most
K-way parallel. At about 107 k blocks × 32 coefficients, room0 is about 3.4 M
symbols a frame, roughly 100 M symbols/s at 30 fps. That is more than one
adaptive host coder can plausibly sustain. That is an estimate, and the room0
run is where it gets measured. The replacement:
- **Tables:** static per-frame, per-band frequency tables in the header, over a
  small alphabet (a magnitude class plus raw bits, JPEG-style).
- **Chunks:** fixed runs of blocks, each coded by interleaved rANS lanes, with a
  chunk offset table so decode is parallel too. *(Amended 2026-09-27: each run,
  a segment, is one independent stream instead of interleaved lanes.)*
- **Arithmetic:** integer-only, so this stage is bit-exact across devices.
- **No new dependencies:** the same coder handles the coordinates and the mask,
  so neither nvcomp (CUDA-only) nor libzstd is needed.

A host reference coder lands first. GPU encode and decode kernels follow once
room0 measures the host coder, and they must produce bytes identical to that
reference, which is their test.

**Determinism, defined.** The encoder sorts blocks by coordinate, because the
GPU compaction orders them by atomics and the same volume would otherwise give
different bytes. The entropy stage is exact. The transform is float, so the
same device reproduces its bytes, but another device may round a coefficient
at a bin edge the other way. With intra-only frames that cannot build up. So a
bitstream is **reproducible per device and decodable on any**, which is what
the 2026-06-21 "deterministic" means here. A fixed-point transform would buy
cross-device byte equality, and nothing needs that yet.

**The sequence**, each PR with its own tests:
1. This entry, the tier, `CodecParams`, and the forward and inverse transform
   kernels.
2. The host rANS reference and the frame format.
3. `Encoder` / `Decoder` end to end.
4. `examples/codec_replica` and the room0 measurement that tunes the defaults.
5. The GPU rANS kernels.

**What this PR lands.** It lands the `recon_codec` target, the public
`codec/codec_params.hpp`, and the private `DctTransform`. The transform turns a
`volume::BlockList` into a `DctBlocks` and back: K quantized coefficients per
block plus the observed mask, beside the params and `trunc_dist` they were made
with. It refuses a list anchored to another topology epoch, one holding more
blocks than the heap, and one null with a count. On the device it refuses an
entry that is not a live block of the grid. It also refuses a grid without
float `tsdf` and `weight` (for an empty list too), a block size other than 8,
invalid `CodecParams`, and on the inverse a `DctBlocks` whose `trunc_dist` or
size does not match. In `volume` it adds `kObservedWeight` and
`VoxelBlockGrid::check_block_list`, both shared with `mesh`, and deletes
`block_list`'s rvalue overload, since the natural
`block_list(map().compact_active_blocks().value())` left the list pointing at a
vector gone by the next statement. `codec_dct_test` compares the kernels
against a double-precision host reference of the same fill and transform and
checks the reconstruction bounds orthonormality guarantees.

**Verified** on macOS (Apple M5 Max, MoltenVK 1.4.2), in Release and in Debug
under ASan + UBSan. The full suite passes, 29 of 29 against 27 before, in
both builds, re-run after the review follow-up. `recon_codec_dct` checks:
- every forward coefficient against the reference, rounded half-to-even
  exactly. The one exception is within 0.05 of a half-way point, where either
  neighbour passes; exact comparisons must outnumber those 20 to 1.
- per-block RMS error within the orthonormal bound at K = 512.
- the truncation error at every zigzag band size (1, 4, 10, 20, 35, 84, 512),
  within 1e-3 of what the reference's dropped energy predicts.
- the mask below, at and above the threshold.
- the decoder's clamp, on a step edge whose truncated reconstruction rings
  to ~1.26.
- a constant block coming out DC-only, and the floor step landing on the
  clamp's edge.
- output following list order.
- batching at 3 blocks per dispatch, checked against the reference.
- the fill: a mask of every fourth voxel pair, and a block with nothing
  observed, which yields all-zero coefficients.
- a partially observed tilted plane at the defaults: its coefficients match
  the reference fill; junk written into its unobserved voxels changes no
  coefficient; and it decodes within 5e-3 of the reference round trip, at under
  half the error of a zero fill.
- every refusal: a free slot and a mis-paired coord among the dead entries,
  with the free slot left zeroed by the refused inverse and the live entries
  decoded; another `trunc_dist`; a grid without a weight for an empty list.
  And the moves.

Each assertion was shown to catch the bug it guards against, by planting that
bug and watching the test fail. The planted bugs were:
- the inverse using the forward basis
- the inverse ignoring the zigzag order
- `>` for `>=` in the mask
- either kernel ignoring its batch offset
- `floor` for round-half-even
- a mis-packed mask word
- weight written on unobserved voxels
- the DC step dropped
- no decode clamp
- no epoch check
- a `ptr` past the heap

and, for the review follow-up:
- no fill
- the fill searching only one direction
- the fill skipping its z pass
- the forward reading an unobserved voxel's `tsdf`
- the probe reduced to a range check
- the host ignoring the rejected count
- the count never reset between calls
- no `trunc_dist` check
- the decoded weight not pushed
- the attribute check after the empty-list return
- the forward dropping `trunc_dist` from its output
- two zigzag entries swapped

A fill planted in the kernel and the reference together is caught by the
accuracy check alone (0.153 against its own 0.153). Make compares mtimes at
one-second resolution, so a planted file restored within the second of its
build left a stale binary that failed on correct code. Each plant therefore
deletes the codec's objects and SPIR-V before rebuilding.

Two of these needed the test strengthened before it caught them. The batched
forward kept passing when compared with the single run, because a same-size
allocation came back still holding that run's output (VMA reusing the freed
block is the likely mechanism), so it is now judged against the reference.
And no fixture rang past ±1 until the step edge was added. A removed barrier
between passes is **not** caught: that is a race, and a race the M5 Max happens
to win cannot be pinned by a test.

**Declined in review: grow-only per-call buffers.** Each call allocates its
list, coefficient and mask buffers. That looked like 650 MB/s of churn at
30 fps, but the three allocations measure 0.027 ms steady state (0.23 ms on the
first call) against a 1.4 ms forward over 97 774 blocks, Apple M5 Max, Release.
That is 2%, the same per-call pattern the integrator and the mesher use for
their inputs. The buffers change anyway with the device-resident output below.

**Open.**
- P-frames (frame type 1). *The format landed 2026-10-12 (the P-frame
  entry, below), and the decoder 2026-10-13; the encoder follows.*
- The GPU rANS kernels, and device-resident coefficients so the GPU coder
  reads them without a host round trip. *Both landed 2026-10-03 (the device
  encoding and decoding entries, below).*

### 2026-09-27 — The v1 frame is independent rANS segments of sorted blocks, every integer a class plus raw bits, behind fixed per-frame tables and a section table; the decoder's end check is for consistency, not integrity (amends the 2026-09-26 entry's interleaved lanes).

*Amended 2026-10-12 (the `.vrst` entry, below):* no CRC section is added; a
frame travels in a `.vrst` record, whose CRC-32 checks it. *Amended
2026-10-13 (the device-only encoder):* a segment holds at most
`kMaxDeviceDecodeSegmentSize` (1,024) entries. R, and a P-frame's U, are
each from 1 to it: every reader refuses a frame past it as
`InvalidArgument`, before any device work, and every writer refuses such a
size (`check_segment_size`). One GPU invocation decodes a whole segment, and
the codec decodes on the device alone, so this is what makes every valid
frame decodable; before, the format took any size of at least 1, and the
device reader alone refused a frame past the limit. No frame outside the
repository uses a larger size, and the golden frames keep theirs.

The second of the 2026-09-26 entry's five PRs: the host rANS reference coder
(`codec/rans.hpp`) and the v1 intra frame (`codec/bitstream.{hpp,cpp}`), both
internal. Nothing public changes. `Encoder` / `Decoder` are the next PR.

**Segments, not interleaved lanes.** The 2026-09-26 entry planned chunks of
blocks, each coded by interleaved rANS lanes. What landed is simpler: a
**segment** of R consecutive blocks (`segment_size`, default 64, stored in the
header) is one fully independent rANS stream. It holds everything about its
blocks, starts with its first block's coordinate in full, and is listed by
length in a SEGMENTS section.
- **Why:** on the GPU that is one thread per segment in both directions, with
  no coordination over where each lane's words go. True interleaving into one
  stream needs that coordination.
- **Cost:** a segment carries its state (4 bytes), its first coordinate (12)
  and its length (4). That is 20 bytes per 64 blocks, against an estimated 30+
  bytes per block of payload. *Measured 2026-10-11 (v3, defaults, M5 Max):*
  18.8 bytes a segment, 2.2% of room0's final frame at 2 cm, 2.6% at 1 cm and
  2.7% at 5 mm (`codec_replica --encode-every 0 --entropy host`, the frame at
  `--segment-size 64` against the same frame as one segment).
- **Parallelism:** room0's ~107 k blocks come to ~1.7 k segments. *Measured
  2026-10-11:* 101 766 blocks at 5 mm make 1 591.

**The coder.** Giesen's word-oriented rANS:
- **Parameters:** a 32-bit state in `[2^16, 2^32)`, 16-bit output words, and
  12-bit probabilities. One renormalization step always suffices, and nothing
  needs more than 32-bit arithmetic, so the GPU kernels can mirror it in GLSL.
  The encoder's bound is tested as `(x >> 20) >= f`. The first cut wrote
  `x >= f << 20` as a 64-bit product: right on the host, but a literal port
  computes `f << 20` in a 32-bit `uint`, where it is 0 at a probability of one,
  and would emit a word for every probability-one symbol. The two forms agree
  exactly, and the writer's output did not change by a byte.
- **The writer:** `RansWriter` takes symbols in decode order and runs the coder
  backwards itself, so no caller reverses anything. A symbol its table cannot
  encode (frequency 0, or past the alphabet) or a raw field wider than 32 bits
  is recorded as a step of frequency 0, and `finish` refuses the stream at the
  division that step would have made. The failure is recorded in band because
  a sticky flag stored in `put` made a room0-sized frame's write 47.5 ms
  against this form's 42.8, which is within noise of no check at all.
  `finish` appends to the caller's vector, so the frame writes every segment
  onto one payload.
- **The reader:** `RansReader` is bounds-checked and its failures are sticky,
  so garbage input costs a wasted loop, never a read past the buffer. A raw
  read wider than 32 bits fails it too; it was a shift past the word.
- **Measured:** 100 000 symbols of probability one cost nothing past the
  4-byte state. A 200 000-symbol skewed source with interleaved 7-bit raw
  fields came to 1 087 232 bits against an ideal 1 087 459 under its table
  (−0.02%).

**The symbol model.** Every integer is a **class** (its bit length) drawn from
a table, plus the bits below its leading one (and a sign) as raw bits:
- **Coefficients:** 16 classes (±32767 < 2^15), one table per coefficient
  index, as the prior engine had.
- **Coordinates:** blocks are sorted by (z, y, x). Deltas go through five
  context tables, each for a case the sort already constrains: a z step; a y
  step within a z slice; a y step after a z step; a run step x − 1 within a
  row, so a run is a stream of zeros; and a free x step. All five have 33
  classes, since a delta can reach 2^32 − 1.
- **Masks:** a 3-way class per block (all observed, none, partial). A partial
  mask then sends its 64 bytes through a 256-symbol table.
- **Tables:** counted over the frame and normalized on the host, with integer
  arithmetic only. The GPU path will hand its histograms to the same function,
  so the tables cannot differ between the two.

**The layout.** Little-endian throughout:
- A 44-byte header: magic `VRTC`, version, frame type, voxel size,
  `trunc_dist`, block size, K, the two steps, block count, segment size.
- A section table of `{id, flags, length}`, then the bodies: TABLES, SEGMENTS,
  PAYLOAD. *Amended 2026-10-12:* a P-frame adds REFERENCE, section id 4,
  first (the P-frame entry, below).
- The section rules: an unknown section flagged required stops the read with
  `Unsupported`, and an unknown optional one is skipped. The sections may come
  in any order. The frame's length must be exact.
- Every other flag bit is reserved: zero in v1, and a known section that sets
  one is refused as `Unsupported`. A later version may define a flag that
  changes how a body reads, and a v1 reader must not parse that body as v1.
- A table's varints must be canonical, so each value has one spelling. That
  does not make every accepted frame canonical, since the sections may come in
  any order, so a consumer that hashes frames should hash what the writer
  made.

**Sorted coordinates are a format rule.** They must be strictly increasing, and
the writer refuses anything else. That is what keeps the deltas small, and it
means a decoded frame is duplicate-free, which is the precondition the inverse
transform documents and does not check. Within a segment that holds by
construction, since the delta code cannot step backwards. A segment's first
coordinate is raw bits, though, which no end check sees, so the reader checks
each segment against the one before. The first cut did not, and review found
it. A frame whose second segment's first x was edited down onto the first
segment's block decoded cleanly, with one block twice. The writer then refused
its own decoded frame, and a decoder would have raced two workgroups on one
block.

**The reader takes the caller's capacity.** A frame's size does not bound its
block count: a block whose every symbol has probability one costs no bits, so
640 all-zero blocks in one run fit in 415 bytes. `read_intra_frame` therefore
takes `max_blocks` and refuses a frame that claims more, rather than
allocating whatever the header asks for. It bounds bytes as well as blocks:
the reader allocates `12 + 64 + 4K` bytes per block, at most ~2.1 KB at
K = 512, which is half the 4 KB of `tsdf` + `weight` the grid it decodes into
holds per block. Every size is computed in 64 bits and checked against the
vector's `max_size` before anything is allocated. On a 32-bit build
(armeabi-v7a is a target) `n * K` used to wrap, sizing the coefficients short
and then writing past them, and it is now refused as `OutOfMemory`. The writer
likewise refuses more than 2^30 − 1 segments or 4 GiB of payload, since its
u32 lengths cannot hold them.

**The end-of-stream check is for consistency, not integrity.** The first draft
of `RansReader::finish` called it "a free integrity check". The test showed
otherwise:
- **Truncation:** caught every time.
- **Single-bit flips, 4000 of them on a random stream:** 2232 were caught.
  1714 landed in raw bits, where they are invisible by construction: raw bits
  leave the state as they are decoded, and every value of them is valid. The
  other 54 changed a table symbol and still finished. They moved the slot into
  another symbol of equal frequency, so the next state was identical and the
  decoder resynchronized one substituted symbol later.
- **A realistic frame:** 2990 of 3000 random one-to-three-byte corruptions were
  refused, against 2967 before the review's checks. The rest decode to a
  different, valid frame.

So the no-checksum rule stands, and **integrity is the transport's**. If a
consumer ever keeps frames where nothing checks them, a CRC can be added as an
optional section. A v1 reader skips it, so that needs no new version.

**Verified** on macOS (Apple M5 Max), in Release and in Debug under ASan +
UBSan. The full suite passes, 31 of 31 against 29 before. The two new
host-only tests also pass under GCC 16, with identical figures to Clang's.
That took one fix: the frame fixture drew three random numbers as one
constructor's arguments, whose evaluation order GCC and Clang choose
differently, so CI's GCC legs had tested a different frame. The two tests
are `recon_codec_rans` and `recon_codec_bitstream`:
- `recon_codec_rans` covers the normalization invariants (including the path
  where bumping rare symbols to 1 overshoots M), round trips from 0 to 100 000
  operations with raw fields 1 to 32 bits wide, the zero cost, the size bound,
  every truncation, the flips above, and a reader that stops early.
- `recon_codec_bitstream` covers round trips across block counts, K up to 512
  and segment sizes from 1 to larger than the frame, with decode then
  re-encode giving identical bytes. It also covers coordinates at both ends of
  int32 and deltas of 2^32 − 1, every writer refusal, and every header,
  section and table refusal (each made by editing one field of a valid frame).
  Further: a delta stepped past int32 in a hand-patched stream, truncation at
  every length, 3000 random corruptions, and the all-zero frame's cost.
- For the review follow-up, `recon_codec_rans` adds every put and get either
  side refuses and `finish` appending. `recon_codec_bitstream` adds a
  duplicate and a step backwards across segments (and a step forwards, which
  is just another valid frame), a reserved flag on each known section,
  overlong varints, and a reader that fails partway through a delta off
  `INT32_MAX` being reported as corrupt. The first cut reported that as
  leaving int32.

The review follow-up was also checked three ways the tests cannot pin:
- The writer's output is byte-identical to the first cut's over 84 frames
  (2.6 MB, K up to 512, segment sizes 1 to larger than the frame), so the
  32-bit bound and the rewritten writer changed no byte of the format.
- Timed on a room0-sized frame (107 000 blocks, K = 32, 64-block segments,
  3.05 MB; Apple M5 Max, Release, `-O3`, best of 35), the writer takes
  42.8 ms against the first cut's 40.7. The copies and per-segment allocations
  review asked to remove were not that cost: the first cut, and the first cut
  appending into one payload with no per-segment vector, measure within noise
  (40.5 and 40.3 ms). Reverting any one change of the follow-up leaves the gap
  in place, so it is the writer function's code generation rather than any
  one change. It is 5% of a host reference, and the GPU-coder measurement
  under Open is the number that matters. Two things this timing caught
  first: the sticky flag above (47.5 ms), and a `reserve` per appended
  segment. `reserve` allocates exactly what it is asked for, so that version
  copied the frame once per segment (83.7 ms). `finish` now appends with
  `push_back`, which grows geometrically. Its first replacement,
  `resize`-then-index, failed CI's ubuntu-24.04 Release leg. GCC 13 at `-O3`
  cannot see that `resize` grew the caller's vector, so `-Wstringop-overflow`
  fails the build. It reproduced in an `ubuntu:24.04` container, and neither
  Clang nor GCC 16 warns. `push_back`, a pointer taken after `resize`, and a
  staged copy all compile cleanly under GCC 13 and time the same.
- The frame code and both tests compile under `-Werror -O3` with GCC 13.3 in
  CI's `ubuntu:24.04` image.
- The frame code and both tests compile under `-Werror` for a 32-bit `size_t`
  (`arm64_32`, the watchOS SDK) and under GCC 16.
- `bitstream.cpp` now includes none of the 17 Vulkan and GPU headers it used
  to. `DctBlocks` moved into `dct_blocks.hpp`, which needs only `CodecParams`,
  so the host-only format no longer reaches the transform.

Mutation-checked: 15 planted bugs, each caught by the tests. They span the
renormalization bounds on both sides, the state check, the normalization's
excess path, the table-sum, trailing-byte, `max_blocks`, required-section and
int32-overflow checks, the writer's sort check, a run step coded one way and
read another, and mask bytes read reversed. One check is not pinned:
`read_intra_frame`'s segment-length check (at least 4 bytes, even). It is
redundant with `RansReader`'s own, which still refuses the frame, and it is
kept for its message.

The review follow-up planted 12 more bugs, in the Debug build under ASan +
UBSan, and each was caught:
- no cross-segment order check
- a reserved flag accepted on a known section
- an overlong varint accepted
- the int32 check ahead of the reader's failure
- a segment's length off by one word
- `put` reading past the alphabet (caught by ASan)
- `finish` dividing by a zero frequency (caught by UBSan and the test)
- `finish` keeping the ops of a refused stream
- `put_bits` accepting more than 32 bits
- `get_bits` accepting more than 32 bits
- `finish` replacing rather than appending
- the renormalization bound off by one

One test needed strengthening before it caught its bug. Reading 33 bits from
a stream of one 32-bit field ran off its end with or without the width check.
The test now reads 36 bits from three whole 12-bit chunks, which only the
check refuses.

**Declined in review.**
- A byte budget beside `max_blocks`. It would bound nothing the grid the caller
  decodes into does not already bound (see above).
- Catching `std::bad_alloc`. The library builds with `-fno-exceptions` on
  mobile, so the reader checks every size before it allocates instead.
- Writing the payload straight into the frame and back-patching its lengths.
  It removes the last full copy, and it measured no faster than appending into
  one payload (see the timing above). The simpler structure stayed.

**Open.**
- `Encoder` / `Decoder`, the next PR. *Landed 2026-09-27 (the public-API
  entry, below).*
- The room0 throughput measurement that decides whether the GPU coder is
  needed. *Measured 2026-09-27 (the room0 defaults entry, below).*
- The optional CRC section above, a `TODO(codec)` in `parse_frame`.
  *Dropped 2026-10-12 (the `.vrst` entry, below).*
- The reader accepts a "partial" mask that decodes all-full or all-empty. Our
  writer never makes one, and it decodes correctly. A `TODO(codec)` in the
  reader's block decode.

### 2026-09-27 — `Encoder` and `Decoder` are the codec's public API: encoding drops never-observed blocks and sorts the rest, decoding makes a caller's grid hold exactly the frame by diffing its block set, everything checkable is checked before the grid is touched, and a grid too small for the frame is refused rather than grown.

*Amended 2026-10-10 (the fusion entry, below):* `tsdf::Fuser` grows for
fusion through the stateless `volume::grow_grid` helper, replacing
`allocate_band`. The decoder still refuses a grid too small for its frame;
a player grows it with `grow_grid`.

*Amended 2026-10-13 (the P-frame decoding entry, below):* the `Decoder`
decodes P-frames too, against the frame it decoded last, which it keeps on
the device. Everything here holds for them, and no refusal or failure
changes that reference.

*Amended 2026-10-18:* the decoder still refuses a grid too small for its
frame; a player's growth sits beside it in the codec, as the `Fuser`'s sits
in `tsdf`. `make_decode_grid` and `decode_growing` (`codec/decode_grid.hpp`)
moved from the examples, so a player outside recon, the next app, decodes a
`.vrst` into a growing grid from one copy. The grid has the examples'
layout, the volume tier's default (`volume::kDefaultBucketSize`,
`volume::kDefaultMaxChain`). A grow shows in `num_buckets`, so the call
counts none.
`codec_replica` (room0 at 1 cm, period 30) and `codec_mesh` (Rafa2's frames
1–150) write byte-identical streams, and CSV frame, type, bytes and blocks
columns, against main.

Both helpers contain host allocation failures. `make_decode_grid` includes
header diagnostics and grid creation in its exception boundary. Once a
decode succeeds, aggregating its optional metrics cannot turn it into a
failure: the reference has committed, so reporting failure would invite a
retry against the wrong reference. A merge that runs out of host memory
leaves the caller's prior metrics intact and returns decode success.
`recon_codec_decode_grid_oom` injects the diagnostic and merge failures and
checks that the following P-frame still decodes to the expected grid.

*Amended 2026-10-22:* both helpers take the caller's ceiling, `max_blocks`.
They sized the grid from the frame's own header, so the v1-frame entry's
bound, the caller's capacity, was gone once the 2026-10-21 networking entry
fed them TCP bytes: a forged 50-byte header, or a frame of 8 bytes per 1024
blocks, made a grid of gigabytes that was never shrunk. A frame past the
ceiling, or a grow past `make_decode_grid`'s grid for it, is refused before
the decode or the grow, as `InvalidArgument`, a frame a player skips rather
than an `OutOfMemory` that stops it. `make_decode_grid` checks the block
count against the segment table through the decoder's own parse, where a
sound P-frame stops at its reference (`NotFound`). The ceiling is a required
argument, so a player states its budget.

The third of the 2026-09-26 entry's five PRs, and the first public API the
codec has had: `codec/encoder.hpp` (`Encoder`, `EncoderConfig`) and
`codec/decoder.hpp` (`Decoder`, `FrameInfo`, `read_frame_info`). Both hold the
private `DctTransform` through a `std::unique_ptr` to a forward declaration, so
the installed headers name nothing under `src/`.

**Encoding.** `Encoder::encode(grid)` works in five steps:
1. Compact the active set.
2. Sort it by (z, y, x).
3. Run the forward transform.
4. Drop every block whose mask is empty.
5. Write the frame.

The sort puts the blocks in the order the frame format requires, and it is also
what makes the output independent of hash order. A test pins this: two grids
with the same content, allocated in opposite orders into tables of different
sizes, give identical bytes.

**Why drop never-observed blocks.** A fused grid allocates the whole
truncation band, and much of it is never integrated. Such a block costs a
coordinate and a mask class in the frame and contributes nothing to the mesh,
since the mesher skips unobserved corners. The filter runs after the transform
because the mask is what says which blocks those are. As a result a decoded
grid can hold fewer blocks than the source grid, with the same surface.
*Amended 2026-10-21:* with `EncoderConfig::true_sdf` the encoder drops the
observed blocks that hold no surface too, and its cut changes the surface
on purpose, past sign errors.

**Decoding into the caller's grid, by diffing.** The grid is the caller's, as
it is for every tier. After `Decoder::decode(frame, grid)` succeeds, the grid
holds **exactly** the frame's blocks. The decoder does not clear it for that:
at playback rate, clearing a large grid would zero every attribute array each
frame. Instead it merges two sorted lists, the grid's current active set and
the frame's coordinates:
- Blocks the frame lacks are removed; `VoxelBlockGrid::remove` zeroes them.
- Blocks the grid lacks are allocated.
- Blocks in both are kept in their slots.

The inverse transform then rewrites every voxel's `tsdf` and `weight`, so a
kept block carries nothing over from the previous frame. A test pins that a
kept block keeps its slot, which is what makes the sort matter. An unsorted
merge still gives the right grid, but it removes and re-allocates blocks it
should have kept. When the merge allocates nothing, its kept blocks are the
frame's blocks in the frame's order, in the slots the first compaction read,
since a removal frees other blocks' slots and moves none. So that pass skips
the second compaction and its sort. A grid that declares any attribute beyond
`tsdf` and `weight` is refused. A kept block would otherwise carry that
attribute's previous values (a `color`, say) under the new geometry, and the
2026-08-04 rule makes a staleness the caller cannot see the library's to
check. `VoxelBlockGrid::attribute_count()` exists for that check.
`read_frame_info` reads the header alone, which is what a player uses to
build a grid of the stream's geometry before its first decode.

**Checked before the grid is touched:** the grid's block size and attributes,
the whole frame parsed and entropy-decoded, the grid's geometry matched
exactly, and the frame fitting its heap. Geometry is compared exactly, not
approximately: the coordinates are in the grid's voxels and the coefficients
are fractions of its band, so a frame from a grid of other geometry would
otherwise decode to the wrong place at the wrong scale and report OK. A
refusal at any of these checks leaves the grid bit for bit as it was, which
the tests check for truncation, a flipped byte, another geometry, a heap too
small, a missing attribute and an extra one. The inverse transform's
`maxStorageBufferRange` checks come after the grid has changed, but they
cannot refuse there. Once the frame fits the heap, every buffer the inverse
binds is at most the grid's 4-byte `tsdf` array:
- K ≤ 512 coefficients per block
- 16 mask words per block
- one 16-byte list entry per block
- 20 bytes of hash entry per slot

`VoxelBlockGrid::create` and `resize` bound that array to the limit, and
`attribute()` confirms it still covers the grid. Only three failures can come
after the grid has changed: a hash table that cannot place the blocks, bucket
locks that keep losing races, or a failed dispatch. The grid is then valid but
holds **neither** frame. Blocks both frames share still hold the previous
frame's voxels, and blocks new to this frame are empty. The next successful
decode restores it exactly, because each decode diffs against whatever the
grid holds.

**Refused, not grown.** A grid too small for the frame gets `OutOfMemory`,
whether its heap has fewer slots than the frame has blocks or its hash table
cannot place them. The message names the fix (`resize`), and one recovery
serves both cases. Growing is policy the library leaves to its callers: the
examples grow (`allocate_band`), and no tier does. The heap case comes from
the frame reader, which checks `max_blocks` only after the block count agrees
with the segment table. A corrupt count is therefore still `InvalidArgument`,
and only a sound frame is refused for its size. Lock contention is not a
capacity limit, so the decoder retries it itself, removal and allocation
alike, up to four rounds each, since both calls already re-dispatch
internally. Contention that outlasts the rounds is `IoError` ("decode
again"), never `OutOfMemory`: growing on it would double every attribute
array over a table with room, which is the misreading `AllocFailures` exists
to prevent. A removed block the free heap refuses is `IoError` too. The heap
only takes back what it handed out, so that grid's accounting was already
broken. Tests build both too-small grids, get `OutOfMemory`, resize, and
decode the same frame exactly. The first grid has 32 slots for a larger
frame. The second has as many slots as blocks, two per bucket and chains of
one.

**Metrics from the start.** The public API would otherwise change in the next
PR, whose room0 run measures exactly these rows:
- `"codec encode"` over `"  ..active set"`, `"  ..sort"`, `"  ..forward"`
  and `"  ..rans encode"`.
- `"codec decode"` over `"  ..rans decode"`, `"  ..active set"` (the
  compactions), `"  ..apply"` (the merge, the removal and allocation, and
  ordering and checking the placed set) and `"  ..inverse"`.

Rows merge by name, so the two sides share no name of their own: an encode
and a decode timed into one `StageMetrics` stay apart, and a test pins it.
The one row they share is `"  ..active set"`. The map itself publishes that
row for every caller that compacts inside a stage, `tsdf` included.

The parent row's device half is the transform, timed through a `GpuStageScope`
that `DctTransform::forward` and `inverse` now take. The encoder takes the grid
by non-const reference because compaction runs the map's kernels, as
`MarchingCubes::extract_host` does. It reads the grid and never writes it.
*Amended 2026-10-26 (the const compaction entry, below):* a compaction
records from a const map, so the encoder takes a const grid.

**Measured**, on an analytic sphere (radius 0.12 m, 5 mm voxels, 40 mm band,
298 coded blocks) meshed with `MarchingCubes` and judged against the true
surface, not against the source mesh:

| | size | triangles vs. source | worst / mean vertex off the sphere |
|---|---|---|---|
| source grid (the analytic SDF) | — | 21 472 | 0.005 / 0.002 voxels |
| defaults (K = 32) | **35.8 B/block** | ×1.0067 | 0.27 / 0.06 voxels |
| near-lossless (K = 512, steps 0.002) | 289 B/block | ×1.0007 | 0.014 / 0.003 voxels |

At near-lossless settings the worst voxel error is 0.12 mm, against the
0.9 mm the orthonormal bound allows. The room0 run in the next PR is the
real-content version of this table, and it is where the defaults get tuned.

**Verified** on macOS (Apple M5 Max), in Release and in Debug under ASan +
UBSan. The full suite passes, 34 of 34 against 32 before.
`recon_codec_encoder` and `recon_codec_decoder` are new; they share
`tests/codec_fixture.hpp`, and the decoder test links `recon_mesh`, which the
tier itself may not. Mutation-checked with eight planted bugs, each caught:
- keeping unobserved blocks
- sorting by `ptr`
- never removing
- no geometry check
- a capacity failure reported as the wrong code
- the attribute check skipping `weight`
- the heap bound dropped
- the merge left unsorted (caught by the slot-keeping check)

**What the review changed.** A review found the gaps above, and each fix is
in the text: the heap case's code, the contention and heap-refusal codes, the
removal's count, the extra-attribute refusal, the post-failure state, and the
row names. It also found two costs:
- `VoxelBlockGrid::remove` found each removed block by scanning the whole
  active set, which is O(k·n) for k of n blocks. It now sorts the snapshot
  once and binary-searches, so a decoder removing thousands of blocks a frame
  pays O((k + n) log n).
- The encoder copied every observed block's coefficients into a second set of
  vectors, so it held the readback twice. It now slides kept blocks down in
  place and moves the result into the frame. The sphere figures above are
  unchanged by it.

Two smaller fixes: `sort_by_coord` now lives beside `coord_less` in
`bitstream.hpp`, so the encoder and decoder share one frame order, and a
`static_assert` ties `EncoderConfig::segment_size` to the writer's
`kDefaultSegmentSize`. A moved-from `Encoder` now reports an all-zero
configuration, consistent with `valid()`. The reader's messages now say
`codec frame:`, not the private `read_intra_frame:`, since `read_frame_info`
and `decode` both surface them.

Nine more planted bugs, each caught:
- the heap limit refused as `InvalidArgument`
- the heap limit checked before the segment table
- no extra-attribute refusal
- the no-allocation pass skipping the removal
- the no-allocation pass using the pre-merge set
- the in-place filter moving masks only
- a defaulted `Encoder` move
- a decoder row sharing the encoder's name
- `VoxelBlockGrid::remove` matching on z alone

Two of them needed a test first. The in-place filter was invisible, because no
test grid had a dropped block sorting *before* a kept one, so nothing ever
slid. The round trip now plants one. The z-only match was invisible because
the volume test removed a single block, and it now removes two of four in one
z plane. Verified in Release and in Debug under ASan + UBSan, 34 of 34. The
changed sources and tests also compile at `-O3 -Werror` under GCC 13.3 in an
`ubuntu:24.04` container, the CI leg local toolchains have missed before.

**Declined in review.**
- Moving the inverse's buffer-range checks ahead of the grid change. They
  cannot fire there (see above), so a second copy of them would guard
  nothing.
- Renaming `"  ..active set"`. It is the map's row, published by
  `compact_active_blocks` for every caller, so the codec reporting under it is
  the design, not a collision. Timing the codec's compactions under a name of
  its own would lose their device half.
- A test of the contention and heap-refusal paths. Neither can be provoked
  from outside the map without planting a fault in its kernels.

**Open.**
- The room0 measurement and tuning of the defaults (the next PR).
- The GPU coder, if that measurement calls for it.
- P-frames. Blocks kept in their slots across frames are what an incremental
  re-mesh of a decoded stream would need.

### 2026-09-27 — Room0 sets the codec's provisional defaults, K = 64 with one step of 0.2 for DC and AC alike: the coefficient count sets the quality and a coarse uniform step costs almost nothing at it, and the host rANS coder fits a frame interval at 1 cm, so the GPU coder waits.

The fourth of the 2026-09-26 entry's five PRs, where the codec meets real data:
- **`examples/codec_replica`** fuses a Replica sequence exactly as
  `fuse_replica` does. Every `--encode-every` frames it encodes the fusion grid
  and decodes it into a player grid, which it builds from the first frame's
  `read_frame_info` as a player would.
- **Its report:** bytes per frame and per block, the bitrate at the coded
  frame rate (30 fps divided by `--encode-every`), and the stage rows of both
  calls, timed on the attempt that decoded. At the end it meshes both grids
  and measures one against the other.
- **`--sweep`** encodes the final grid under a list of configurations and prints
  the rate–distortion table below.

**The metric is mesh to mesh, not SDF sampling.** Both directions are named as
the codec's ground-truth test (#94) names them:
- **accuracy:** each decoded vertex's distance to the source surface;
- **coverage:** each source vertex's distance to the decoded surface.

Each is a point-to-triangle distance through a cell hash, up to a reach of
`max(trunc_dist, 2 cm)`, sampled at every fourth vertex. A point with nothing
within reach is counted apart, not averaged, so accuracy's RMS, p95 and max
cover only the points within it; the count beyond it is reported beside them
(`acc>r`, `cov>r`). The code was `examples/common/mesh_distance.hpp`,
header-only, with a test pinning it against meshes a known distance apart and
against a scan of every triangle. It moved into the `eval` tier the same day
(the next entry). The first plan was to sample the source grid's
SDF at the decoded vertices, which would be simpler. It was dropped because a
fused TSDF is **projective**: distance along the camera ray, not Euclidean. So
on real scans the simpler metric would have measured the fusion's bias as well
as the codec's error.

**Measured** on room0: 400 frames, every frame coded, M5 Max, Release. Encode
and decode times are host ms per coded frame, and the rANS share of each is in
brackets.

| | voxel | blocks/frame | B/block | Mbit/s @ 30 fps | encode ms | decode ms | accuracy RMS | coverage RMS / beyond reach |
|---|---|---|---|---|---|---|---|---|
| prior engine's (K = 32, DC 0.25 / AC 0.05) | 1 cm | 21.9 k | 18.7 | 98.4 | 13.1 (rANS 10.1) | 10.7 (7.0) | 0.886 mm | 1.226 mm / 104 |
| **new (K = 64, 0.2 / 0.2)** | 1 cm | 21.9 k | **17.4** | 91.5 | 17.4 (13.8) | 15.7 (11.5) | **0.641 mm** | 0.835 mm / 31 |
| prior engine's | 2 cm | 5.4 k | 21.6 | 28.0 | 3.9 (2.9) | 3.6 (2.0) | 2.264 mm | 3.414 mm / 37 |
| **new** | 2 cm | 5.4 k | **20.5** | 26.6 | 4.6 (3.7) | 4.6 (3.1) | **1.629 mm** | 2.382 mm / 24 |

**The sweep over the final 1 cm grid** (24.9 k coded blocks; steps are
fractions of `trunc_dist`, distances in mm). The 2 cm sweep orders the configs
the same way.

| config | B/block | accuracy RMS / p95 / max | encode / decode ms |
|---|---|---|---|
| split, K = 8 / 16 / 32 / 64 / 128 | 13.3 / 15.5 / 18.5 / 23.0 / 28.3 | RMS 2.646 / 1.374 / 0.884 / 0.574 / 0.381 | 10.6 → 33.0 |
| split, K = 32, steps ×0.5 / ×2 / ×4 | 20.9 / 16.5 / 14.7 | RMS 0.882 / 0.902 / 0.978 | ~15 |
| uniform, K = 32, step 0.025 → 0.4 | 21.1 → 13.5 | RMS 0.879 → 0.995 | ~15 |
| uniform, K = 64, step 0.1 / 0.2 / 0.4 | 19.9 / 17.1 / 14.8 | RMS 0.586 / 0.642 / 0.796 | ~20.5 / 16.3 |
| uniform, K = 128, step 0.2 / 0.4 / 0.8 | 19.1 / 15.8 / 13.4 | RMS 0.496 / 0.700 / 1.074; max 34 / 19 / 28 | ~29.5 / 25.5 |
| uniform, K = 256, step 0.4 / 0.8 | 16.7 / 13.7 | RMS 0.671 / 1.073 | ~47 / 46 |
| uniform, K = 512, step 0.8 / 1.6 | 13.9 / 11.9 | RMS 1.078 / 1.751 | ~85 / 137 |
| reference, K = 512, step 0.002 | 147.1 | 0.012 / 0.023 / 2.2 | 122 / 131 |

**What it says.**
1. **The error's tail is the quantizer's.** The near-lossless reference decodes
   to 0.012 mm RMS, with nothing beyond reach. So the 20–37 mm worst vertices
   of every practical config are ringing from dropped coefficients, a few
   floaters near thin structures, and not a pipeline defect. Every config's
   p95 is under 2 mm. The max cannot exceed the 4 cm reach, but nothing hides
   past it: no decoded vertex of any config is beyond reach of the source,
   save 5 of ~980 k at K = 8 (and at 2 cm, 2 at K = 512 with a step of 1.6).
2. **K sets the quality, and the step barely does at low K.** At K = 32, going
   from a step of 0.025 to one of 0.4 cuts the bytes by 36% and raises the
   error by 13%. Each doubling of K costs about a quarter more bytes and takes
   about a third off the error.
3. **A DC step coarser than the AC step buys nothing.** At the same AC step,
   the split's 5× DC step saves 2% of the bytes for 7% more error (×4:
   14.7 B/block at 0.978 mm, against uniform 0.2 at 15.0 and 0.915). That
   matches the theory: the transform is orthonormal, so a unit of quantization
   error costs the same in any coefficient.
4. **Many coefficients coarsely beats few finely.** Past K = 128, though, a
   coarse step zeroes the extra coefficients anyway: at a step of 0.8,
   K = 128, 256 and 512 all come to ~13.5 B/block and ~1.07 mm, and only the
   time grows. The best K rises with the bitrate. At the low end, K = 32 at
   0.4 (13.5, 0.995) edges out K = 128 at 0.8 (13.4, 1.074). This is what the
   per-band quantization study is for (see Open).

**The choice: K = 64 with a uniform 0.2.** It beats the prior engine's
defaults on bytes and on error at both resolutions: 7% smaller and 28% more
accurate at 1 cm, 5% smaller and 28% more accurate at 2 cm. It also keeps both
calls inside a 30 fps frame interval on the host at 1 cm (encode 17.4 ms,
decode 15.7 ms).

The two better points traded speed for size:
- **K = 128 at 0.4:** 15.8 B/block and 0.700 mm.
- **K = 128 at 0.2:** 19.1 B/block and 0.496 mm.

Both roughly double the host coding time, which would put 1 cm room0 at the
edge of real time.

The defaults are **provisional**, and are recorded as such in
`codec_params.hpp`. They are to be re-chosen by the per-band study on these
same tables.

**Where the new defaults lose.** On the analytic sphere of `codec_decoder_test`
they are 16% smaller than the old ones (30.2 against 35.8 B/block) but **17%
less accurate** on the mean (0.070 against 0.060 voxels off), and the worst
vertex is **2.4x as far off** (0.66 against 0.27 voxels). A smooth surface
keeps all its energy in the low bands, which the defaults' coarse AC step
quantizes as coarsely as the high ones. So the test's worst-vertex bound
moved from 0.5 to 0.8 voxels, and it gained a mean bound of 0.085 --
1.2x what an M5 Max measures, as tight as another device's rounding allows,
so the regression is pinned rather than hidden in slack. The sphere is kept as the per-band study's second fixture, beside room0.
A table can be fine in the low bands and coarse in the high ones, and so could
win on both.

**The GPU coder waits.** At the new defaults, host rANS takes 42% of a frame
interval to encode and 35% to decode at 1 cm, single-threaded. That is inside
the interval, so PR 5 is not needed for this content at this resolution. It
would be at finer voxels, at K = 128, or on a slower host. Even then, the
cheaper first step is to code the segments on several host threads, since
they are independent by construction (the 2026-09-27 format).

**Bitrate.** Intra-only room0 at 1 cm is ~90 Mbit/s at 30 fps. Most blocks do
not change between frames, which is the case for P-frames.

**Verified.** The full suite passes, 35 of 35 against 34 before; the new test
is `recon_mesh_distance`. The metric was mutation-checked with five planted
bugs, each caught:
- a closest-point vertex region returning the wrong vertex
- a query that skips neighbour cells
- the reach counted as within it
- coverage measured the wrong way round
- p95 read as the median

**Review fixes** (all in this PR, before merge):
- **Degenerate triangles.** The first closest point was Ericson's region test,
  which divides by an edge's length and by the area. Over 1 cm triangles at
  room coordinates, it gave NaN for half of all queries when `a == b`, NaN
  for 4% and a point up to 4.7 mm off when `b == c`, and NaN for one in 2000
  on a micron sliver; `std::min` silently dropped each NaN from the search.
  It is now the face projection when that lands inside, with the face trusted
  only at a sine above 1e-4, and otherwise the nearest of the three edges,
  within 0.2 µm of a double-precision reference on all of those. On room0 no
  figure moved: its reference mesh has no triangle with coincident corners,
  and on seven of the sweep's configs the two agree on every sampled
  distance to 1 µm.
- **The search.** Cells are half the reach, a query scans its own cell first,
  and then only the neighbours nearer than its best, on squared distances.
  The sweep hashes the reference once rather than once per config. Comparing
  one room0 mesh both ways went from 8.7 s to 2.2 s, and matches a scan of
  every triangle exactly.
- **p95** is the nearest rank, `ceil(0.95 n)`; it had been read one rank
  high, which for 20 points or fewer is the max.
- **The player grid** is sized from its first frame's block count and the
  sweep decodes into the stream's, where each had been sized from the
  source's grown table (131 k slots, 512 MiB of `tsdf` + `weight`, for 25 k
  blocks). A decode that runs out of room grows the grid and always decodes
  again. IoError is retried twice, and then returned as itself, since the
  Decoder also uses it for a broken heap, which no retry mends; it had been
  retried 16 times and then reported as OutOfMemory.
- **The partial-block DCT test** is pinned to the prior engine's K = 32, the
  case the unobserved-voxel fill was built for, rather than following the
  defaults to K = 64.

The sweep's first cut timed each config's first call on a fresh `Encoder`,
which varied from 5 to 35 ms on the same grid. Each config now runs one
untimed round first.

**Open.**
- **The per-band quantization study,** its own PR, before P-frames. It judges
  a step table per `x + y + z` band on room0 and on the sphere, with these
  tables as the baseline (a `TODO(codec)` on `CodecParams`).
- **P-frames,** for the bitrate above.
- **Host-threaded segment coding,** if a finer voxel or K = 128 is wanted in
  real time (a `TODO(codec)` in `bitstream.cpp`).
- **The GPU coder,** after that, if threading is not enough.

### 2026-09-27 — Quality measurement is a tier of its own, `eval`, branching off `mesh`: mesh-to-mesh accuracy, coverage and F-score, host-side and deterministic, and production infrastructure rather than an excluded research harness.

The mesh-distance metric came in with the room0 measurement as a header in
`examples/common`. It had three consumers, or would have soon:
- **`codec_replica`**, whose defaults it chose;
- **the codec's ground-truth test (#94)**, which carries its own copy of the
  closest-point code;
- **the per-band quantization study**, which will be judged on it.

A metric that picks defaults and gates tests belongs in the library, where it
is documented, versioned and tested like the rest.

**Named `eval`, not `metrics`.** `core` already has `StageMetrics`, the timing
vocabulary every tier reports in, and a `metrics` tier would read as more of
that. The target is `recon_eval`, the namespace `volumetric_kit::recon::eval`.

**Off `mesh`.** It reads `mesh::Mesh`, so it depends on `mesh` and branches off
it, as `texture` does. Nothing in the pipeline links it. The codec's tests and
example depend on it, never the reverse, so `codec` stays on `volume` alone. A
later grid-to-grid metric would fit too, `volume` being to its left.

**Not an excluded harness.** The exclusion list names "Python research/eval
harnesses and any learned/neural or paper-experiment code". This is none of
those: it is host-side C++ with no dependency beyond the mesh container, and
it is deterministic up to the inputs it is given. The repo's own tests and
examples use it to hold the reconstruction and the codec to a number, so it
meets the same bar as any tier: full Doxygen, `Status` on every refusal, a
test with analytic answers. A paper's evaluation protocol, a dataset loader or
a plotting script would still be excluded.

**What v1 holds** (`eval/mesh_distance.hpp`):
- **`MeshDistance`:** point-to-surface distance up to a reach, through a hash
  of cells half the reach on a side. A query scans its own cell first, then
  only the neighbours within two cells whose box is nearer than its best so
  far, on squared distances (from #95's review, below).
  - It **copies** the triangles, so the source mesh need not outlive it. The
    header-only version kept a reference, a lifetime the caller could not see.
  - `create` refuses a reach that is not finite and positive, an index count
    that is not a multiple of 3, and an index past the vertices, where the
    header-only version read out of bounds.
  - It also refuses a triangle corner that is not finite or lies past the cell
    keys' range (about half a million reaches out), and a reach so small for
    the triangles that they would average more than `kMaxCellsPerTriangle`
    (64) cells each. A triangle no larger than half the reach touches at most
    8, and one no larger than the reach 27. Both
    are limits the caller cannot see: the first overflowed the float-to-int
    cell conversion, which saturates on ARM and left the filing loop without
    an end, and the second grew memory as `(size / reach)^3` (one 1 cm
    triangle at a micron took 8.1 GB).
  - A query that is not finite, or past the range, reads as the reach.
- **`closest_point_on_triangle`** is the projection onto the face when it
  lands inside, and otherwise the nearest of the three edges; a triangle
  whose angle at its first corner has a sine under 1e-4 is measured by its
  edges alone. Ericson's regions assume an area, and with the first two
  corners coinciding every point fell in edge ab's region, whose parameter
  is 0 / 0. The NaN lost to `std::min`, so the triangle silently dropped
  out, and marching cubes makes one wherever a crossing lands on a voxel
  corner. (This PR first kept Ericson's test behind a guard at a sine of
  1e-5; #95's review fixed the same bug with the projection, which is the one
  kept, as below.)
- **What counts as surface:** every triangle but one collapsed to a point,
  and the points measured are the vertices those triangles use.
  `extract_device_incremental` retires a triangle to a point (the sharing
  kernel's indices all at vertex 0, the default kernel's three vertices at
  the origin) and leaves the sharing kernel's dead vertices unreferenced at
  stale positions. So comparing an incremental extract with a full one, the
  obvious use of this tier, would have measured its garbage.
- **`summarize`:** a distance distribution over the points within reach, plus
  a count of those beyond it. The p95 is the nearest rank, `ceil(0.95 n)`.
- **`compare_meshes`:** accuracy (test vertices to the reference surface) and
  coverage (reference vertices to the test surface), the directions the
  ground-truth test named. It also computes the **F-score** at a threshold,
  since precision and recall fall out of the same distances.
  - The threshold must lie in `[0, reach]`. Past the reach, every distance
    reads as the reach, and the score would be meaningless.
  - `codec_replica` reports it at half a voxel. That is inside the
    reconstruction's own resolution, so the score moves with the codec rather
    than saturating: 0.993 on the 20-frame 2 cm smoke run.
  - It checks both meshes before indexing either, and drops each index before
    building the next, so one comparison holds one.
- **`ReferenceMesh`:** a reference indexed once, for many meshes to be judged
  against. The sweep compares 24 decoded meshes against one source and
  rebuilt the source's index for each. It gives exactly what `compare_meshes`
  gives for the same pair, and it pairs the index with the points taken from
  the same mesh, so the two cannot be mismatched.

**A stride picks vertices by a hash of their position.** `compare_meshes` can
measure about one vertex in `stride`, and `codec_replica` uses 4 on room0's
million vertices. The first cut took every `stride`-th vertex of the array.
But marching cubes emits vertices in the order its atomics hand out ranges,
so that subsample changed from run to run. The same grid's figures moved by
about half a percent (accuracy mean 1.208 against 1.214 mm on two runs of the
smoke test, the max 44.8 against 39.1 mm), which hid any codec change
smaller than that. A hash of the position depends only on the geometry, so
three runs now agree in every printed digit (accuracy mean 1.214 mm, F
0.9926). The earlier entry's room0 tables were measured under the array
stride. This is a subsample of the same size, so they should stand to within
that half percent. One was re-run: the 1 cm defaults' accuracy RMS reads
0.640 mm under the hash, against the table's 0.641.

**The example's entry point stays concise** (this PR's base,
`refactor(examples): keep codec_replica's entry point concise`). The player
policy, the report printers and the sweep live in `examples/common`
(`codec_stream.hpp`, `codec_sweep.hpp`). `main.cpp` is the example's story:
options, setup, the fuse loop, the report. *Amended 2026-10-08 (below):*
every example's, its command line, flags and stage table included.

**Verified.** The full suite passes, 35 of 35. `recon_eval_mesh_distance`
replaces the example-header test: the closest point in every region of a
triangle and on every degenerate one, the reach clamp, the statistics,
accuracy / coverage between planes a known distance apart, the F-score, what
counts as surface, figures independent of the mesh's order, the reusable
reference, and every refusal.
- **F-score checks:** it is exactly 1 above a 3 mm shift and exactly 0 below
  one, and not 0/0. Half a plane gets precision 1 and the exact recall of the
  extent it kept. That test first assumed the cut fell at 0.1 m, but
  `20 * 0.005f` rounds below `0.1f`, so the mesh keeps a 21st column of
  squares. The test now reads the extent off the mesh.
- **A copy made at `create`:** the index is built over a temporary mesh and
  queried after it is gone, which ASan would catch if it were not a copy.

**Review (PR #96).** Fixed as described above:
- the degenerate triangle whose first two corners coincide;
- the unbounded cell loop on a non-finite or distant corner;
- the unbounded cells a small reach cost;
- the unreferenced and retired vertices counted as surface;
- the order-dependent stride;
- a p95 one rank high, which read the max for 20 samples;
- the index rebuilt for every configuration of the sweep;
- both indices held at once.

Four smaller fixes:
- The reach is checked before the threshold, so a bad reach is not reported
  as a threshold left at 0.
- `codec_replica` writes its PLYs before comparing, so a refusal leaves the
  meshes to look at.
- The sweep prints `-` for an F-score nobody asked for, and names its
  threshold.
- The `eval` mark in DESIGN.md's diagram now sits under `mesh`.

Each fix has a test, and taking it back fails that test: the degenerate
check, the p95, the unreferenced vertices, the point triangles, the stride
(the reorder test rotates the vertices and bumps the surface, since
reversing 41^2 vertices maps every seventh index onto every seventh index,
and a flat grid gives many subsets one distribution), the order of the
option checks, and the cell budget (the test hangs without it). The two
range checks are caught by the sanitizer leg, as the float-to-int overflow
they prevent; the cell budget alone refuses the far vertex on ARM. (The
degenerate guard measured here, 2% per call, was on Ericson's test, which
the merge below replaced.)

Kept on purpose: `positive_finite` and the cell key stay local duplicates, not
`core` helpers. The first is one line. The second differs from
`tsdf_integrator.cpp`'s `block_key` in encoding (offset rather than
two's-complement masking) and in its range check, which only `eval` needs,
since block coordinates are bounded by the grid.

**Merged with #95's review.** #95's review fixed the same header in
`examples/common` while this PR's review fixed its copy here, each without
the other. When this PR was rebased onto the merged #95, `eval` took the
better of each, measured rather than argued:
- **The closest point is #95's.** Both versions were run against a
  double-precision reference on 1 cm triangles at room coordinates, about
  4.2 million queries (coincident corners, collinear corners, 1 µm needles,
  and slivers with sines from 1e-6 to 1e-2). The projection is within
  0.17 µm on every one, which is float rounding at 3 m. Ericson behind the
  guard is off by more than 0.5 µm on 12 of them, all slivers or collinear,
  by up to 89.5 µm. Rare, so no room0 figure moves, but never is better than
  rarely, and it is simpler.
- **The search is #95's**, with this PR's range checks moved onto the finer
  cells. Room0 at 1 cm, the same grids, compared both ways at stride 4:
  8.25 s before, 1.86 s after, and every printed figure identical (accuracy
  RMS 0.640 mm, F 0.9981 at 5 mm).
- **The rest is this PR's:** the refusals, the surface-only sampling, the
  position-hashed stride, and `ReferenceMesh`, which covers what #95's
  hash-once `compare_meshes` overload was for. #95's `acc>r` column and its
  sweep decoding into the stream's player carry over into the examples.
- **Tests from #95**, ported: thin triangles against the double-precision
  answer (50 000 draws, enough to meet Ericson's rare misses; reverting to
  it fails; 5 000 under the sanitizers), and the pruned search against a
  scan of every triangle (probing one cell fewer at either end along x, or
  leaving the box gap unsquared, fails at 500 points). One
  change survives: trusting the face down to any nonzero area passes every
  case too. The 1e-4 guard is kept as a margin, not a measured need.

**Threaded** (2026-10-22): `compare` measures each direction on every core,
a slice of the points per thread, each writing only its own entries, so the
figures are one thread's to the bit. Room0 fused at 2 cm with each triangle
split in four (1.3 M triangles) against a copy moved up to 1 mm, stride 4,
both ways, on the shared M5 Max in Release: 1.6 to 1.9 s on one core, 0.25
to 0.32 s threaded, every figure identical.

### 2026-09-27 — A mesh becomes a TSDF two ways, in the `tsdf` tier: signed by the closest triangle's normal, for a closed mesh, or as an unsigned shell, for any mesh — binned per block over the allocation's own candidates, with no topology.

*Amended 2026-10-28 (below):* `MeshIntegrator` is
`voxelize::MeshIntegrator`, in a tier of its own on `volume`.

The second stage of mesh → TSDF (S1), on top of the 2026-08-31 block
allocation (S0). It lands `tsdf::MeshIntegrator`, which writes a mesh's
truncated distance field into a grid's `tsdf` and `weight`, in one of two
`MeshSdfMode`s. Both modes are the prior engine's `MeshToSDF::computeSigned` /
`computeShell`, and the research codec's two conversion rules. The
acceleration structure is not.

**Two modes.**
- **Shell** is `d − σ`, the unsigned distance minus a half-thickness (1.5
  voxels by default): the unsigned offset field of Xu & Barbič, *Signed
  Distance Fields for Polygon Soup Meshes* (GI 2014), before their interior
  components are removed. It needs nothing from the mesh: open, non-manifold,
  self-intersecting or inconsistently wound meshes are all fine. Around a
  closed solid it extracts two walls, one σ outside and one σ inside.
- **Signed** is ±`d`, signed by the closest triangle's face normal. That is the
  research codec's rule for a watertight mesh, and it is meant for a closed,
  outward-wound one. Nothing about the mesh is checked.

**Why no topology: the field is the codec's input, not a reconstruction of the
mesh.** The first cut signed by the angle-weighted pseudonormal of the closest
feature (Bærentzen & Aanæs 2005), which is correct past sharp edges and corners
where the face rule is not. On the test's regular tetrahedron (faces at
70.5°), **973 of 10 822** observed voxels are ones the face rule can sign
wrongly. It paid for that with a host topology pass: a weld by exact position,
edge adjacency, a union-find over each vertex's fans, and pseudonormals in
double. That pass was ~9.7 ms of a 22.6 ms signed call on the sphere below. It
would need a sort and a scan the repo does not have to move to the GPU. And it
brought refusals the research codec never needed: non-manifold edges and
vertices, flipped winding, a mesh wound inside out. The conversion exists to
feed the codec, which takes a block's SDF, transforms it with the DCT, drops the
high frequencies, quantizes and entropy-codes what is left. The research codec
produced the paper's results with the closest-face rule and no topology at all.
So the pass went, with its weld, its refusals, `MeshIntegrateStats`'
`boundary_edges` and the feature-reporting closest point. `Signed` now costs
what `Shell` does. Xu & Barbič's robust mode (shell, marching cubes, drop the
nested components, then signed) went from the plan for the same reason, since
it exists to make the sign exact on meshes the simple rule gets wrong.

**What it gives up,** stated on `MeshSdfMode::Signed`:
- Past an edge or corner whose faces meet at under 90°, the faces tied for
  nearest disagree about the side, and the lowest-indexed one decides.
- An open mesh grows a skirt out to `trunc_dist` past its rim, since nothing
  finds the rim. `Shell` is the mode for a mesh that is not closed.
- A mesh wound inside out comes back inside out.
- At exactly 90°, off the grid's axes. Beyond a corner, on the line that
  continues an edge, two of the tied faces' normals are perpendicular to the
  offset, so rounding picks the side. A plain cube turned off the axes
  extracted 72 stray vertices, up to 26 mm off. Aligned with the axes and off the voxel
  lattice, it puts no voxel on those lines.

The test checks the sign against the winding number wherever the tied faces
agree, and only the distance where they do not.

**Triangles are binned per block, over S0's own candidates.** The prior
engine's signed pass measured every voxel against every triangle of the mesh,
as the research codec's kernel does. That is the long-dispatch shape the
2026-08-08 overflow-scan entry records hanging an M5 iPad. The prior engine's
shell pass used a dense grid over the mesh's bounding box, which at an 80 mm
cell is ~16 M cells for a 20 m cube whatever the surface looks like. Here,
`mesh_bin.comp` runs the (triangle, candidate block) work items S0 allocates
over twice, first counting and then filling each block's list of triangles.
The bins are keyed by heap slot, 16 bytes per `num_blocks` (a count and a
coordinate). Then `mesh_integrate.comp` runs one thread per voxel over its
block's bin.
- **One decomposition.** S0 and S1 share it in two places, so the blocks
  allocated and the blocks binned cannot drift apart:
  `volume::triangle_candidate_offsets` (host; it moved out of
  `allocate_from_triangles`) and `volume/shaders/triangle_candidates.glsl`
  (the decode and the band test).
- **Exact where observed.** A bin holds every triangle within `trunc_dist` plus
  the half-diagonal of the block's centre, so it holds every triangle within
  `trunc_dist` of any of the block's voxels. The minimum over it is the true
  minimum wherever the result is observed.
- **A missing block is refused.** The count pass also counts the (triangle,
  block) pairs whose band block the table does not hold, and the host refuses
  a non-zero count before anything is written. A missing block would
  otherwise leave that part of the surface holding stale data.

**What is written.** Every voxel of every band block is overwritten, not
blended. **Observed** means within `trunc_dist` of the mesh in both modes,
which is the set the band test guarantees complete. An observed voxel gets
weight 1 and its distance clamped to ±`trunc_dist`; every other voxel gets
`tsdf = 0, weight = 0`. Those are the codec inverse's fresh-block zeros, so a
mesh-derived grid reads exactly like a decoded frame. Blocks outside the band,
and `color`, are untouched. The shell's thickness is refused below √3/2 voxels
(a point can be that far from every voxel, so a thinner shell can fall between
them) and at or past `trunc_dist` (nothing observed would read as outside).

**Ties break on the triangle index.** The fill pass writes each bin in
atomic-arrival order, and two triangles tie along every shared edge and at
every shared vertex. The per-voxel minimum takes the lower index, for one
comparison, so the same mesh writes the same bytes and a soup writes the bytes
of its indexed twin. The test checks both.

**Review: fourteen findings.** A code review of the first cut returned
fourteen, each checked against the code before it was fixed. Seven are fixed,
four went with the topology pass, two are fixed in part and one is documented.
- **A bin could be read short.** The fill pass re-derived each work item's
  block with the count pass's own probe and band test, and only dropped an
  overflow. Any disagreement between the two dispatches left some other bin
  one short, and the integrate pass then read an entry nobody wrote as a
  triangle index: out of bounds, with `robustBufferAccess` off. The count pass
  now records each item's slot and the fill replays it, so every bin fills
  exactly. That removed the fill's repeated work too. The count pass also
  stores each binned slot's coordinate, which retired the whole-map compaction,
  its readback and the host's second copy of the counts, since the counts are
  rewritten in place as the fill's cursors. With the compaction went the check
  it happened to give, that the table had not changed mid-call, which the
  class's quiescence requirement already rules out, as for the mesh tier's own
  probe. What stays O(`num_blocks`) per call is one clear and one scan of
  those counts, 4 bytes a block, which the prefix sum needs.
- **Nothing bounded a bin.** A thread's work is its bin's length, and the host
  knew the largest bin and checked nothing. A bin past `kMaxBinTriangles`
  (2^16, about 230 times the sphere's average below) is refused, and the write
  is split into dispatches of at most `kMaxDispatchBinEntries` (2^20) bin
  entries, so no one submission grows with the mesh. Both are named on the
  class, as the 2026-08-10 rule asks of a ceiling the library knows.
- **The closest point had a size threshold.** The face-interior branch
  collapsed onto vertex `a` whenever `|ab × ac|²` fell below 1e-12 m⁴, which
  is every triangle with edges under about a millimetre, and a dense scan has
  those everywhere. The guard is now against zero, and the barycentrics are
  clamped onto the triangle, so a sliver's rounding cannot put the closest
  point somewhere the triangle is not. The research codec's kernel has the
  same problem harder: it skips any triangle whose `|cross|` is at most 1e-6 m²,
  for its distance as well as its sign.
- **Four went with the topology pass:** a NaN pseudonormal from a triangle
  collinear in double, a sliver closing a T-junction read as three rim edges,
  a rim vertex refused as a pinch, and an inside-out closed mesh accepted. Each
  was fixed and fixture-checked, and then the code it fixed was removed.
- **Smaller ones.** The missing count is (triangle, block) pairs, not blocks,
  and the message now says so. The O(1) binding checks run before the host
  passes. `triangle_candidate_offsets` refuses a null mesh before it allocates,
  and takes a null `who`. `voxel_hash_map.cpp` lost two includes the refactor
  left behind. The bins and the per-item slots are device-local now. The
  inputs stay host-visible with a `TODO(tsdf)`, as the grid's own attributes
  do, since staging them needs a copy path `core` does not have yet.
- **Documented rather than built: dirty blocks.** The integrator does not
  report the blocks it writes as dirty. Dirty flags belong to the
  `TsdfIntegrator` fusing into a grid, and the only caller who can mix the two
  on one grid is the one making both calls, so it is a `@note` on the class
  rather than a mechanism nothing consumes.

**Verified by mutation.** Seven planted reversions of the current code, each
failing a fixture:
- the face normal reversed, and the sign taken from a closest point other than
  the minimum's: the tetrahedron
- the 1e-12 threshold: a triangle 0.87 mm across, under one voxel column
- the chunk offset and the dispatch budget: a metre-square sheet of 320 000
  triangles, whose 2^20-plus bin entries split across dispatches
- the bin cap: an 80 000-triangle patch 30 mm square
- a fill that replays no slots: the tetrahedron

Not caught: the tie-break, since no fixture can force an atomic arrival order;
and the barycentric clamp, since no fixture can make float rounding produce an
out-of-range barycentric on demand. The first cut's own sixteen mutants were
mostly pseudonormal ones, and went with them.

**Open.**
- **No codec round trip test.** One was built here (#94): mesh → TSDF →
  frame → decode → mesh, measured against the source mesh. It was dropped
  before landing. `codec_decoder_test` already takes the codec end to end
  against ground truth: an analytic sphere, the observed set exactly, the
  field within the transform's bound, and the decoded mesh against the
  sphere. `recon_tsdf_mesh_integrate` covers the conversion. All the round
  trip added was about five minutes under the sanitizers. Two of its findings
  stay:
  - `Signed`'s second limit, at 90° off the grid's axes, now under *What it
    gives up* above.
  - The research codec's published figures appear to score its output against
    the marching-cubes extraction of its uncompressed conversion
    (`ObjMeshingOp` writes it to the run's `ground_truth/`), while Draco was
    scored against the source meshes, so conversion error never counted
    against it. A mesh-sourced comparison with this codec has to measure both
    against the source.
- **What it costs**, on an 81 920-triangle sphere 1 m across, its triangles
  about 1.6 voxels on a side like a scan's (Apple M5 Max, Release, 5 mm voxels,
  the 40 mm band, a table of 2 048 buckets of 8; 6 861 blocks, 1.97 M bin
  entries, ~287 triangles a bin; medians of ten runs):

  | call | wall | GPU |
  |---|---|---|
  | `allocate_from_triangles` | 14.2 ms | 12.6 ms |
  | `integrate`, `Shell` | 12.8 ms | 10.8 ms |
  | `integrate`, `Signed` | 12.8 ms | 10.8 ms |

  The first cut quoted 8.6 ms for the allocation on a table this run did not
  reproduce; that code did not change, and the difference was not chased. A
  voxel's cost is its bin, which a band as wide as a block keeps in the
  hundreds; a mesh far denser than the grid is where the dispatch grows, up to
  the two ceilings above.

### 2026-09-30 — Depth allocation works a 16 x 16 pixel tile a workgroup and dilates each distinct block of the tile once, its band shared out over the lanes.

**The rule.** `hash_allocate_depth.comp` maps each workgroup onto a 16 x 16
tile of the depth image. The host dispatches one group a tile and pushes the
tile count along x, so it alone places the tiles. Every lane unprojects its
pixel to a centre block. In shared memory, a lane keeps its block only when
no earlier lane of the tile has it, scanning back from the lane before: the
left neighbour settles most lanes, and only the first of each block reads
every lane before it. Then every distinct block's `(2·tb+1)³` band is shared
out over the workgroup's 256 lanes, offset by offset across the centres,
rather than walked by one. The allocated
set is the per-pixel one exactly: `allocate_block` is idempotent, and a
block's band is the same whoever dilates it. This is the prior engine's
leader election (`hash_ops.metal`), in shared memory, so it needs no
subgroup feature, and it retires the `TODO(volume)` the kernel carried
(PERF.md's P3).

**Why.** Neighbouring pixels nearly always land in one block: a 1 cm-voxel
block at 1.5 m spans a few hundred pixels. Dilating per pixel probed the same
27 blocks hundreds of times, about 22 M probes for a 1200 x 680 frame. It also
set those pixels racing for one bucket's lock, and every lost race meant
another round and another submit. After the 2026-09-30 batching entry,
allocation was most of a set's fusion, and its retries re-dispatched every
camera.

**Measured.** The batching entry's bench, room0, sets of four 1200 x 680
frames, 1 cm, Release, metrics off, interleaved with that entry's branch,
median per set in set mode:

| | before | after |
|---|---|---|
| M5 Max, per set | 5.77–5.97 ms | 2.87–2.94 ms |
| M5 Max, allocate device (30 sets) | 105–111 ms | 25.3 ms |
| RTX 5090, per set | 2.68–3.07 ms | 1.07–1.09 ms |
| RTX 5090, allocate device (30 sets) | 73–74 ms | 14.0 ms |

Retries, from a round counter on the same bench (30 sets, three runs):

| | before | after |
|---|---|---|
| M5 Max, sets needing a second round | 18–20 | 1 |
| RTX 5090, sets needing a second round | 3 | 2–3 |

On the Mac the retry cost the batching entry recorded is gone with the
contention. The set left on both machines is the first, into an empty map; a
second round settles it. The 5090 had little to remove: its first set loses
about 14 000 races in round 1 to either kernel, and a few later sets lose
1–41, since NVIDIA's independent thread scheduling lets a spin lock make
progress. Allocation's device time is now the same whether the frames come one at a
time or as a set: 24.9 against 25.3 ms on the Mac. Against fusing one frame at
a time before either change, a set takes 2.9 ms rather than 6.9 on the Mac.

**Verified.**
- `recon_volume_allocate` adds a frame three tiles by two, ragged at both
  edges, with holes, one smooth half (many pixels a block) and one jagged
  half (many blocks a tile). It must allocate exactly the host's union of
  every pixel's band. Its values are dyadic (focal length 128, a 1/128 m
  voxel, depths in 32nds, a power-of-two pose), so the host and the shader
  unproject identically.
- It fails two mutants: a lane that defers to its left neighbour without
  comparing blocks, and a band loop that allocates only the centres. The old
  4 x 4 frame passes the first, since it is one partial tile.
- The suite passes on the M5 Max and the RTX 5090.

**Review.** The PR's review changed three things:

- **The scan walks back.** One backward scan from the lane before replaced
  the left and upper shortcuts and the forward scan, and the band work goes
  offset by offset rather than centre by centre. Over the 30 sets,
  allocation's device time fell from 25.4 to 22.8 ms on the Mac and from
  14.1 to 12.2 ms on the 5090; the scan alone gave 7% on the Mac. The review
  proposed the offset order to cut lock races, and it did not change them; it
  stays for its 2%.
- **The host places the tiles.** The kernel derived its tile count along x
  from the camera. It now takes the host's, in the push constant's spare
  argument, and `kTile` sizes its workgroup and shared arrays.
- **The tests cover the wide band and the retry.** The multi-tile frame also
  runs at tb = 2, whose 125-block band outnumbers the lanes. A one-bucket
  table puts every block of the 4 x 4 frame behind one lock: on the Mac its
  first round loses 41–43 races and it settles by its fifth. The 4 x 4 case
  claimed that contention before, but the dedup had taken it away.

After P3, integrate is the larger part of fusion's device time on the Mac:
41.4 ms over the 30 sets against allocation's 22.8. On the 5090 it is 10.3
against 12.2. The batching entry's camera-looping kernel waits on that
figure.

Not changed:
- A 2-D dispatch. Padding to whole tiles can push a frame of about 16.7 M
  pixels past a min-spec device's 65 535 groups, which `CommandBatch`
  refuses cleanly. That is sixteen times a 1024 x 1024 depth frame, and the
  batch dispatches in one dimension.
- A test that the dedup happens. Allocation is idempotent, so the allocated
  set cannot show it; the bench does.

**Verified (review).**
- After every set of the room0 run, the sorted block set hashes the same as
  main's, on both machines.
- `recon_volume_allocate` fails four more mutants: a scan that ignores the
  block, the failure report dropped (the one-bucket case), a band decode that
  assumes a side of 3 (tb = 2), and the shader's `kTile` changed alone.
- The suite passes on the M5 Max (56 tests) and the RTX 5090 (43).

**Not taken.** Deduplicating the bands themselves, which overlap 18 of 27
blocks between adjacent centres, a `TODO(volume)` at the band loop. The rest
of allocation's device time has not been broken down.

### 2026-09-30 — With the spans off, an extract's active list stays on the device, and the map hands back its last compaction while nothing has changed since.

*Amended 2026-10-06 (below):* the spans are removed, so a whole-map extract
always binds the map's device list.
*Amended 2026-10-06 (below):* a fuse culls to its frusta and leaves no full
list, so an extract after a fuse compacts the full list itself, and the reuse
serves whoever follows the extract (the encoder, the grid's block pass);
`recon_tier_stage_metrics` now expects a fuse's row every call. The frustum
list has a buffer of its own and never counts as the active set.

**The rule.**
- **No host list.** With `track_block_spans` off, `extract_device` and
  `extract_host` no longer bring the active list to the host and upload it
  again. They compact onto the device (`compact_active_blocks_on_device`) and
  bind the map's own list in place, so only its count comes back. This is the
  `TODO(mesh)` the extract carried.
- **The fuse's list, reused by the map.** `compact_active_blocks_on_device`
  keeps the list it last returned. While that list still holds, the call
  returns it again, and dispatches and reports nothing. So an extract after
  a fuse reuses the fuse's compaction, with no new API, for every caller:
  `rig_viewer`'s sets, and `fuse_viewer`'s frames.
- **What "holds" means.** The list is stamped with the map's topology epoch
  (moved by a remove or clear), its compaction serial (moved by any
  compaction or resize) and the heap's free count. Within one epoch the free
  count only falls, so a change in it means a block was allocated since.
  `check_device_block_list` makes the same test, so a list held across an
  allocation is now refused too.

**Not taken: a caller-held list.** The first cut added an
`extract_device(grid, iso, DeviceBlockList)` overload, fed by an out-param
on `TsdfIntegrator::integrate` and `fuse_set`. Its review found it
returning the wrong mesh with `Status::ok`:
- An empty list, as `integrate` hands back when it fuses nothing, meshed
  nothing on a full grid.
- A list held across `allocate_from_triangles` and `MeshIntegrator`
  passed the check and left out the new surface, since allocation moved no
  stamp.

It also refused outright with the spans on, made `extract_device(grid, iso,
{})` ambiguous, and reached only the set overload's callers. Keeping the list
in the map removes the overload and both out-params.

**Measured.** The batching entry's bench with an extract after each set
(room0, sets of four 1200 x 680 frames, 1 cm, Release, a three-slot ring,
unshared), median per extract:

| | host list | device list, compacted here | the fuse's list |
|---|---|---|---|
| M5 Max | 1.77 ms | 1.69 ms | 1.51 ms |
| RTX 5090 | 0.91–0.96 ms | 0.74 ms | 0.60 ms |

The Mac's last two columns are the map's reuse, interleaved in one binary:
an untimed host compaction before the extract makes it compact again. The
5090's are the overload's, which did the same device work. They were not
re-run, because the box was busy with other work.

**Verified.**
- `recon_volume_hash_map`: a second compaction with nothing changed returns
  the same list. An allocation that takes a block makes it stale. One that
  takes none does not.
- `recon_mesh_marching_cubes_sparse` meshes the map's list to the same
  triangles as a spans-on extractor's host list, and leaves the fuse's list
  current. After one more block is allocated, it compacts again.
- `recon_tier_stage_metrics` reports no `"  ..active set"` row for a reused
  list.
- Mutants that drop the free-count clause or the reuse each fail two or
  three of those tests.

### 2026-10-01 — Every block slot carries stamps, ticks of one clock on the map, each written by the pass that knows its fact; the grid frees the blocks no allocation has asked for and no voxel has weighted in `max_age` ticks.

**The rule.**
- **One clock, one record a slot.** `VoxelHashMap` keeps a `uint32` tick,
  which the caller advances (`rig_viewer` once a set), and a device buffer
  of one `BlockStamp` per block slot beside the heap (`stamps_buffer()`).
  The init kernel zeroes every record, at create and clear and for a grow's
  new slots, and the delete kernel zeroes a slot's as it frees the block.
  `resize` copies the records forward, since it keeps every slot, and the
  rehash never stamps.
- **Each stamp is written by the pass that knows it.**
  - `requested` is written by every allocation kernel, through
    `allocate_block`, for every block it asks for, inserted or found. The
    tick rides the shared push block, and the store is an `atomicExchange`
    since a band's lanes ask for one block together. Ages are differences of
    ticks, so they hold across the clock's wrap.
  - `weighted` is written by the grid's block pass (`stamp_blocks`): one
    workgroup per active block, where any voxel is observed
    (`weight >= kObservedWeight`, the mesher's and the codec's threshold).
  - A consumer compares ticks against a tick of its own and never resets
    anything. So the dirty flags, which a consumer must reset, are the next
    to move onto a stamp: `changed`, written by the integrator.
- **The first consumer is the GC.** `VoxelBlockGrid::free_stale_blocks`
  frees the blocks whose newer stamp is at least `max_age` ticks old, under
  a `"block stamps"` row. A second kernel zeroes their attributes where the
  pass listed them, a dispatch an attribute, rather than through `remove`,
  which resolves coordinates on the host and fills block by block; the list
  is then read back for the map's `remove`. Both kernels are built on the
  first pass. `rig_viewer` runs it every `--free-after` sets (30), with
  `max_age` the same.
- **Why the map, not the grid.** The map hands slots out and takes them
  back, so its kernels are the ones that know a block was asked for or
  freed. A record kept a tier up would need the allocation kernels to bind
  a buffer they do not own, or the host to zero a record per freed block.

**Not taken: freeing on weight alone.** The first cut kept the records on
the grid, stamped a block `allocated` when a pass first saw it, and freed
one empty for `max_age` ticks. The synthetic bench below liked it. The lab
rig did not:
- In a static room about 60% of the active set holds no weight at any
  moment. It is most likely the corners of the (2tb+1)³ cube depth
  allocation dilates every surface block into, which the allocator asks
  for every set and Dynamic integration never weights.
- So every other pass freed 33 000 blocks (the heap 44% → 18% of 131 072),
  and the sets after it asked for every one again.
- Each such pass took 140 ms, nearly all of it the zero fills: MoltenVK runs
  each `vkCmdFillBuffer` as a dispatch of its own. Removing 33 000 scattered
  blocks of 60 000 spent 80–97 ms in its one submit, against 6 ms sorting
  and searching on the host and 1 ms for the map's delete. A fill per block
  for the stamps was about four in ten of the fills, and the delete kernel
  now does that instead.

`requested` is what tells a block the allocator still wants from one a
surface has left.

**Measured.** A throwaway bench: four cameras at the rig's calibrated
poses, Femto-like 640 x 576 depth ray-cast on the host, a static 0.3 m
sphere and a 0.25 m one walking a 0.8 m loop, then the same loop shifted
0.5 m. 600 sets fused through `fuse_set`, Dynamic, 1 cm, Release; GC every
30 sets at `max_age` 30.

| | blocks at set 600 | blocks holding weight | integrate, device ms/set at the end |
|---|---|---|---|
| M5 Max, no GC | 7 348 | 1 947 | 0.41 |
| M5 Max, GC | 2 909 (2 100–3 200 throughout) | 1 947 | 0.25 |
| RTX 5090, no GC | 7 345 | 1 947 | 0.08–0.09 |
| RTX 5090, GC | 2 909 | 1 947 | 0.04 |

The weighted count matches the no-GC run at every 50-set checkpoint, so no
block holding weight was freed. A pass costs about 2 ms of host time on the
M5 Max and 1 ms on the RTX 5090, once every 30 sets.

- **What the stamp costs allocation.** Interleaved A/B with GC off, three
  rounds of 300 sets, device ms a set: 0.228–0.235 on the base against
  0.220–0.232 on the M5 Max; 0.088–0.090 against 0.092–0.093, +4%, on the
  RTX 5090.
- **The lab rig, static room** (M5 Max, four cameras at 1280 x 720, 1 cm,
  300 sets): the heap stays at 43–44% of 131 072. Each pass frees 1 100–1 800
  blocks the allocator stopped asking for, depth noise most likely, in
  4.6–7.4 ms, nearly all of it zero fills, which the review moved into a
  kernel (below).
- **So the GC bounds history, not the band.** A room nobody moves through
  keeps its size. Shrinking the band is a separate item, PERF.md's P9.

**Verified.**
- `recon_volume_block_stamps` (new):
  - each allocation path stamps what it asks for: coords, points,
    triangles and depth, each through a binding of its own;
  - asking again restamps;
  - the pass stamps only the blocks holding weight;
  - `free_stale_blocks` frees exactly the blocks neither stamp has touched
    in `max_age` ticks, and zeroes their records;
  - `resize` keeps every record, so the rehash does not restamp;
  - `clear` zeroes them all;
  - a `max_age` of 0, and a grid without `weight`, are refused.
- Four mutants each fail it: allocation not stamping, delete keeping the
  record, GC on weight alone, and `resize` dropping the records.
- The volume tests and `recon_tsdf_integrate` run clean under
  synchronization validation.
- The full suite passes on the M5 Max (57) and the RTX 5090 (44), and
  `rig_viewer` ran 300 sets on the lab rig.

**Review.** The PR's review changed seven things:
- **The move-assignment names every member.** It moved none of the block
  pass's, so a grid assigned into a moved-from one kept an empty kernel and
  aborted on its next pass.
- **Zeroing is a kernel.** `block_zero.comp` zeroes one attribute of every
  listed block, a workgroup a block, reading the pass's list where it lies.
  A throwaway bench freed every other block of N, so each freed block was a
  fill of its own before (M5 Max, Release, interleaved, the `"block stamps"`
  row):

  | freed blocks | fills | kernel |
  |---|---|---|
  | 1 500 | 5.2–8.3 ms | 0.74–0.78 ms |
  | 33 000 | 94–100 ms | 1.9–2.8 ms |

  The kernel needs each attribute's block to be whole 4-byte words, as at
  block size 8 it always is, and `free_stale_blocks` refuses the rest.
  `remove` still fills, a `TODO(volume)` on `zero_blocks`.
- **Ages hold across the wrap.** `requested` was an `atomicMax` and the age
  `tick - max(requested, weighted)`, so once the clock wrapped a block's old
  `requested` won, and a block weighted that very pass could be freed.
- **One threshold for observed.** The pass counted any weight above 0, so a
  block of weights under `kObservedWeight`, empty to the mesher and the
  codec, was never freed.
- **A large active set splits.** Both kernels dispatch a group a block in
  dispatches of at most `maxComputeWorkGroupCount[0]`, which a min-spec
  device sets at 65 535 and the lab rig's 58 000 blocks nearly reach.
- **The pass is built on first use**, so a grid that never frees builds no
  pipeline, pool or query pool for it, and its two buffers are named for a
  capture.
- **`rig_viewer` reports a failed pass and fuses on**, rather than ending
  the session over housekeeping.

**Verified (review).**
- `recon_volume_block_stamps` now also checks that a weight under
  `kObservedWeight` leaves a block to be freed, that a freed block's voxels
  are zeroed in every attribute while a kept block's stay, and two
  move-assignments: into a moved-from grid, and over one that ran its pass
  over fewer attributes.
- It fails each of these mutants: a zeroing kernel that writes nothing, the
  threshold back at 0, the move-assignment without the new members (a
  crash), and either kernel ignoring its base with the split forced to 2
  groups. The split forced to 2 alone passes.
- A throwaway run wrapped the clock: the old logic freed a block weighted
  that pass, and the new keeps it.
- The volume tests run clean under synchronization validation, and the
  suite passes on the M5 Max (57). The lab rig was not re-run.

**Not taken.**
- Removing on the device. The list comes back in the zeroing's submit and
  goes up again in `remove`'s, 16 bytes a block each way and no submit of
  its own; a device-list `remove` would save those bytes and nothing
  measured.
- Skipping the `weight` fill, which the review suggested since the pass had
  found it 0. Under the new threshold it need not be, and the kernel leaves
  little to save.

### 2026-10-01 — The dirty flags become the `changed` stamp: every pass that writes voxels advances the map's clock and stamps what it changed, always, and an incremental extract keeps the tick it last meshed at (amends the 2026-08-09 dirty-block decision and the stamps entry above).

*Amended 2026-10-06 (below):* the stamp and every writer of it stay, and
`read_block_stamps` serves the iOS scanner; its mesh reader, the incremental
extract, is removed.

**The rule.**
- **Writers advance the clock.** Three calls write voxels, and each advances
  the map's tick before it writes and stamps `changed` with the new one:
  - `TsdfIntegrator::integrate`, one tick a call, so a rig's set is one. It
    stamps where a stored value differs, Dynamic's clear included, exactly
    where the flags were set.
  - `MeshIntegrator::integrate` and the codec's inverse, on every block they
    write, since they overwrite whole blocks.

  The caller no longer advances the tick, and `rig_viewer`'s call goes. So
  an age in ticks is an age in writes: in sets for the rig, in frames for a
  single camera.
- **Readers keep a tick.** `extract_device_incremental(grid, iso, timings)`
  re-meshes the blocks whose `changed` stamp is newer than the tick its last
  publishing extract ran at, kept in its arena state beside the watermark.
  Nothing is reset. `fuse_replica --dirty-every` keeps a tick of its own and
  dilates on the host, so it and `--incremental` no longer contend for one
  set of flags.
- **Gone:** `DirtyBlocks`, `TsdfIntegratorConfig` (whose only field was the
  flag), `dirty_block_count`, `reset_dirty`, `dirty_flags_buffer`,
  `dirty_flags_capacity`, `dirty_epoch` and `dirty_remesh_blocks`. So is
  their anchoring: grid identity, an epoch latched per integrator, and
  re-arming. The stamps live in the grid's own map, which zeroes a slot's
  record as it frees the block. A remove still forces a full extract,
  through the arena state's own epoch.
- **Always on.** The integrator stamps on every fuse, with no opt-in. One
  lane of each subgroup stamps for the lanes beside it, since every lane of
  a workgroup is one block's when a block is a whole number of workgroups.
  It reads the stamp first, so subgroups after the first skip the atomic.

**Why the writers advance it, not the caller.** A tick is a period, and a
fuse and an extract both happen within one. With the caller advancing, an
extract records "meshed through T", and a fuse later in the same tick also
stamps T, which the next extract reads as already meshed: a stale surface
under `Status::ok`. A writer that advances first stamps a tick newer than
every cursor taken before it, however the calls interleave.

**Why every writer.** The extractor now reads the stamps instead of a
struct the caller hands it. So a writer that stamped nothing would leave
its blocks' triangles stale, which is the library's to prevent (2026-08-04).
The flags had the same blind spot, since `MeshIntegrator`'s notes said a
grid it wrote had to be meshed in full. But there the caller had to bring
the flags, so the extractor never meshed against a writer it could not
see.

**Measured.**
- **Equivalence**, on room0 at 2 cm in Release, on the M5 Max and the RTX
  5090, against #132:
  - `--incremental --mesh-every 10`: 38 of 40 extracts incremental and 71.0%
    of blocks re-meshed on both builds, with the same final mesh;
  - `--dirty-every 10`: 53.30% changed and 71.44% to re-mesh, 3 329 → 4 530
    of 8 692, on both;
  - both flags at once give the same numbers as each alone.
- **Integrate's device time**, interleaved A/B against #132 (ms a frame):

  | | 2 cm | 1 cm, mean |
  |---|---|---|
  | M5 Max, before / after | 0.128–0.130 / 0.127–0.134 | 0.480 / 0.488 |
  | RTX 5090, before / after | 0.033 / 0.034 | 0.219 / 0.207 |

  It took three cuts. An atomic for each changed voxel read 0.046 against
  0.033 at 2 cm on the 5090. One elected lane a subgroup still read 0.225
  against 0.197 at 1 cm. Reading the stamp first brought both to parity.

**Verified.**
- `recon_tsdf_integrate`'s stamps fixture:
  - a fuse is one tick, and stamps the four band blocks but neither the four
    ahead of them nor the off-camera one;
  - a reader's tick sees both fuses since, or only the later one;
  - a grow keeps the stamps;
  - Dynamic's clear stamps;
  - an identical re-fuse stamps nothing;
  - a second grid's fuse leaves the first's stamps and clock alone.
- `recon_tsdf_integrate_set`: one frame at a time and the set stamp the
  same blocks, at a tick a frame against one for the set.
- `recon_mesh_marching_cubes_sparse`, incremental passes driven by stamps:
  - nothing stamped keeps the old surface;
  - one stamped block re-meshes its neighbourhood alone;
  - a pass after it, with nothing new stamped, re-meshes nothing;
  - all stamped gives the new surface;
  - the culled-extract and remove fallbacks still hold.
- `recon_tsdf_mesh_integrate` and `recon_codec_decoder`: every block written
  is stamped at the call's tick.
- `recon_volume_block_stamps`: `changed` is kept across a resize and zeroed
  by the delete kernel and by clear.
- Seven mutants each fail one of those: no stamp on a fuse, none on the
  clear, no tick advanced, an inclusive cursor, a cursor not kept, and no
  stamp from `MeshIntegrator` or from the decoder.
- The changed tests run clean under synchronization validation, and the
  full suite passes on the M5 Max (57) and the RTX 5090 (44).

**Review.** The PR's review changed five things:
- **The GC frees a block older than `max_age`, not one that old.** With
  integrate advancing the clock, a set's allocation is a tick old by the
  time the set is fused. So `age >= max_age` freed blocks a set early, and
  at `--free-after 1` it freed the whole unweighted band every set, about
  60% of a static room. Freeing above `max_age` gives `rig_viewer` its old
  behaviour exactly: a block goes `max_age` sets after it was last asked
  for. This amends the stamps entry's "at least `max_age` ticks old".
- **The integrate stamp is an exchange, and its read an atomic load.**
  `atomicMax` compares unsigned, so once the clock wrapped, a stamp from
  before the wrap would outrank every new tick and its block would never
  re-mesh. The plain read before it raced the other groups' atomics, which
  the memory model leaves undefined. A relaxed `atomicLoad`
  (`GL_KHR_memory_scope_semantics`) keeps the skip and is well defined.
  Interleaved A/B on the M5 Max, room0, 300 frames, device ms a frame:
  0.127–0.150 before and 0.124–0.131 after at 2 cm; a mean of 0.515 before
  and 0.520 after at 1 cm. Not re-measured on the RTX 5090, which was busy
  with other jobs.
- **The arena records its iso.** An incremental pass at another iso kept
  every clean block's triangles on the old surface. It now falls back to a
  full extract. The flags' version had the same gap.
- **One wrap-aware comparison.** `volume::tick_after` and its GLSL twin in
  `block_stamp.glsl` replace five hand-written copies.
- **Smaller fixes:**
  - the extractor binds the stamps on every pass, not a dummy on full ones;
  - `fuse_replica`'s `changed_since` refuses a zero block size and counts a
    slot past the stamps as changed, as the kernel does;
  - two stale comments and a stale fallback in `ExtractTimings::incremental`
    are fixed.

**Verified (review).**
- `recon_volume_block_stamps`: a block asked for a tick before a writer
  stays at `max_age` 1, and goes a tick later.
- `recon_tsdf_integrate`: with every stamp set above the clock, as after a
  wrap, a fuse still stamps what it changed.
- `recon_mesh_marching_cubes_sparse`: an incremental request at another
  iso falls back, and gives that iso's surface.
- Each fails its mutant: `>=` in the GC, `atomicMax` in the integrate, and
  the iso clause dropped.
- room0 at 2 cm gives this entry's numbers again: 38 of 40 extracts
  incremental at 71.0% re-meshed, and 53.30% changed, 3 329 → 4 530 of
  8 692.
- The changed tests run clean under synchronization validation, and the
  suite passes on the M5 Max (57).

**Not taken.**
- **Shims for the removed dirty API.** `volumetric_kit_ios` fetches recon
  at `main`, and its scanner (`apps/scanner/Bridge/Fusion.mm`) calls
  `TsdfIntegratorConfig`, `dirty_remesh_blocks`, `dirty_block_count`,
  `reset_dirty` and the `DirtyBlocks` overload. So it stops compiling at
  this merge, and it migrates in a PR of its own:
  - drop the config and the resets;
  - call `extract_device_incremental(grid, iso, &timings)`;
  - survey from `read_block_stamps()` against a kept tick, as
    `fuse_replica`'s `changed_since` does.

  Shims would keep two mechanisms for one fact.
- **A public call that stamps blocks changed, for other writers.** Nothing
  in the family writes voxels outside the three writers. `attribute()` now
  says that a write through it stamps nothing, so the next extract must be
  a full one. The test that drives the extractor without fusing stamps by
  hand.
- **A narrower `--dirty-every` readback.** It reads 12 bytes a slot, against
  the flags' 4. That is about 1.5 MB at room0's 131 072 slots, once a window,
  outside every timed row.

### 2026-10-01 — The codec review: frame v2 codes a partial mask a plane and a line at a time against its neighbours and a sign inside its mantissa's field; the transform finds every block by its coordinate, one `CommandBatch` a call over device memory; the encoder takes the map's own list and drops never-observed blocks before the transform; and the decoder stamps only what it changes (amends the `changed`-stamp entry above).

A review of the codec tier found fifteen things, five of correctness and ten
of cost. All fifteen are fixed here. One of them changes the frame, so it is
version 2, and a v1 frame is `Unsupported`; nothing in the family keeps
frames.

**The mask was half the frame.** Measured before anything changed, on room0
at 1 cm coding every fourth frame (2.2 M blocks, 41% of them partially
observed), v1's partial masks were **51.4% of the bits** and 25.3% of the
coder steps: 64 raw bytes a partial block through one order-0 table. The mask
edge of a fused band is a smooth surface, so its lines repeat. The models
were judged offline on the last frame's 24 885 masks, by the empirical
entropy of each context (bytes a coded block):

| v1, one table | a line's byte by its predecessor's class | per line: same, empty, full or a byte | **per plane, then per line** |
|---|---|---|---|
| 8.25 (64 steps a partial block) | 4.88 | 2.80 (68 steps) | **2.23 (28 steps)** |

So v2 codes a partial mask a z plane at a time. Each plane is one symbol:
the same as the plane before it, all empty, all full, or coded line by line.
Each line (one byte, eight voxels along x) is the same as its predictor,
empty, full, or a byte. Its predictor is the line before it in the plane, or
the same line of the plane before. Its symbol and byte are drawn from one of
three tables, chosen by whether the predictor is empty, full or neither. The
plane before the first, and the predictor of the first line, are unobserved.
The writer tries the symbols in that order, so a value two of them name has
one spelling from it. A plane-model context of the previous plane's symbol
measured 2.17 and was not taken for three more tables.

**A sign rides its field.** v1 sent a sign as a raw field of its own beside
the bits below the leading one. v2 sends one field of c bits: the sign is the
low bit and the mantissa sits above it. A value of class c <= 12 is one coder
step, not two, and the bits are the same.

**Measured**, room0, every frame coded, M5 Max, Release, three interleaved
pairs at 1 cm and one at 2 cm (host ms a coded frame):

| | B/block | Mbit/s @ 30 fps | encode (rANS) | decode (rANS) | accuracy RMS |
|---|---|---|---|---|---|
| 1 cm, before | 17.4 | 91.5 | 17.6 (13.7–13.8) | 15.9–16.1 (11.1–11.3) | 0.640 mm |
| **1 cm, after** | **11.0** | **57.7** | **14.2–14.4** (11.1–11.3) | **13.9–14.1** (10.3–10.5) | 0.640 mm |
| 2 cm, before | 20.5 | 26.6 | 4.70 (3.69) | 4.58 (2.96) | 1.635 mm |
| **2 cm, after** | **13.6** | **17.6** | **3.83** (2.96) | **4.10** (2.72) | 1.635 mm |

The decoded surface is the same to the triangle: 1 312 202 at 1 cm on both.
On the decoder test's sphere the defaults go from 30.2 to 21.6 B/block with
the same error. The rest of the encode's time went as follows:
- `..sort`: 1.36 to 1.00 ms, sorting only the kept blocks.
- `..forward`: 2.03 to 1.17 ms.
- `..active set`: gone, since the fuse's list is reused.
- `..observed`: 0.75 ms, new.

On the decode side, `..active set` went from 0.98 to 0.67 ms, one
compaction instead of two, and `..apply` from 2.70 to 1.70 ms. The device
half rose: the encode's from 0.92–0.96 to 0.95–1.05 ms, and the decode's
from 0.59–0.60 to 0.78–0.82 ms, since the inverse now reads every voxel
before it writes it (below). Nothing here was measured on a discrete GPU.

**The transform is resident.** Its coefficients, masks, list, tables and
reject count were host-visible, and each batch of up to
`maxComputeWorkGroupCount` blocks was its own fence-waited `dispatch()`.
That is what the 2026-09-28 rule forbids. They are device-local now. Each
call is one `CommandBatch`: the inputs staged up, the dispatches, and the
results and the reject count read back. The coefficients travel as 16 bits,
two to a word, each entry starting a word of its own. So an odd K pads its
last word, and the host drops the pad on the way out and restores it on the
way in. `DctBlocks` holds them as `int16_t`, which halves the transform's
largest buffers and the host's copy. The reader's allocation bound drops
from ~2.1 KB a block to ~1.1 KB.

**Every block is found by its coordinate.** The kernels used to check that an
entry's coord resolved to exactly its ptr, and refused it otherwise. Now they
take the block the hash table holds for the coord and never read the ptr. A
coord the table does not hold is refused, and nothing is written for it. That
is the same single probe as before. It is what lets the decoder list the
frame's blocks without reading back the slots its allocation drew, which had
cost it a second compaction and a sort of the whole grid on every frame that
added a block. The check that the grid held exactly the frame went with that
compaction. The inverse's refusal still catches a frame block the grid
lacks. A block the grid gained could only come from a writer during the
call, which `Decoder` already forbids.

**The encoder takes the map's own list and keeps only observed blocks.**
*Amended 2026-10-03:* the observed kernel now appends a compact list on the
device, with a predicted-prefix readback instead of the full input and flags;
see the dated entry below. The coordinate sort and CPU entropy boundary stay.
*Amended 2026-10-13:* neither stays; the coder runs on the device alone (the
2026-10-03 encoding entry's amendment), and so does the sort (the 2026-10-13
coordinate entry).
*Amended 2026-10-06 (below):* a fuse culls to its frusta and leaves no full
list, so the encoder takes back an extract's list, and an encode straight
after a fuse compacts the full list itself.

`compact_active_blocks_on_device` returns a fuse's list while it still
holds, and its compaction is kept for the extract after. Before, the
encoder's host compaction invalidated the fuse's list, so an extract after
an encode compacted again. A new kernel, `dct_observed.comp`, reads each
listed block's weights and writes one word back per entry. The encoder drops
the unobserved blocks there, then sorts and transforms the rest. Before, it
transformed every block and read the zeros of the dropped ones back: room0
codes one block in five of those it allocates.

**The decoder stamps only what it changes.** The `changed`-stamp entry above
had the inverse stamp every block it wrote. That made an incremental extract
of a decoded stream re-mesh the whole grid every frame, though the blocks
kept across frames are what such an extract needs (the 2026-09-27 decoder
entry's Open list). The inverse now reads each voxel before writing it, and
stamps a block only where a value differs, as the integrator does. That read
is the decode's device time above. Decoding a frame twice into one grid
stamps nothing the second time.

**`VoxelBlockGrid::remove` resolves on the device.** It found each removed
block's slot from a compaction and a sort of the whole active set, on every
call and so on every decoder retry. Now `block_zero.comp` finds each coord's
block itself, and zeroes it, so the cost is the count's and not the grid's.
`free_stale_blocks` uses the same kernel. An attribute whose blocks share
4-byte words cannot be zeroed by that kernel. That needs an odd block size
and an element under 4 bytes, and `free_stale_blocks` already refuses it.
`remove` keeps the old snapshot and fills for that case.

**The smaller fixes.**
- **A step ceiling.** `kMaxStep = 64`, the power of two past
  `2 sqrt(512)`, where every coefficient quantizes to 0. A corrupt header's
  finite step of 1e38 made decoded coefficients infinite, and the inverse's
  sums NaN. `validate` refuses it, and so does the frame reader.
- **Geometry before size.** The decoder compares the frame's geometry with
  the grid's from the header, before decoding anything. A frame of other
  geometry that was also larger than the grid came back `OutOfMemory`,
  telling the player to grow a grid that could never take it.
- **A broken heap is `InvalidArgument`.** A free heap that refuses a removed
  block shared `IoError` with lock contention, whose answer is to decode
  again. `decode_growing` could not tell them apart.
- **Other fixes:**
  - `kMinStep` is a literal, which the params test holds to its formula,
    since `std::sqrt` made it a dynamic initializer;
  - `bit_length` counts leading zeros in one instruction rather than a
    shift loop;
  - `volume::coord_less` is the one (z, y, x) order, where the codec, the
    grid's remove and the test fixture each had a copy.

**Verified.**
- New cases:
  - `recon_codec_dct`:
    - an odd K over four blocks, both ways, against the reference;
    - ptrs that are forged and ignored, an absent coord refused, and nothing
      written for it;
    - the observed pass, in the list's order, refusing a stale list;
    - stamps only on a change.
  - `recon_codec_decoder`: a second decode of one frame stamps nothing, and
    another geometry larger than the grid is `InvalidArgument`.
  - `recon_codec_encoder`: it takes back a fuse's list, leaves its own for
    the next caller, and has the `..observed` row.
  - `recon_codec_bitstream`: slab-shaped partial masks, the v2 table count,
    and the step ceiling.
  - `recon_volume_block_grid`: an absent coord zeroes nothing, and the
    shared-word fallback.
- Ten planted bugs, each caught:
  - every block stamped;
  - every block observed;
  - no geometry check;
  - the odd-K pad dropped and restored wrong;
  - the zero kernel trusting the ptr;
  - "same" before the first plane read as full;
  - the sign read from the wrong bit;
  - no step ceiling;
  - the coefficient halves swapped.
- The changed tests run clean under synchronization validation. The changed
  sources and tests compile at `-O3 -Werror` under GCC 13.3 in an
  `ubuntu:24.04` container. The codec and volume tests pass under ASan and
  UBSan, and the suite passes on the M5 Max, 44 of 44.

**Not taken.**
- **Refusing a second spelling.** A mask whose plane is coded line by line
  but repeats the plane before it, or a line coded as a byte that is empty,
  decodes correctly and is never written. It joins v1's partial-mask-all-full
  case in the same `TODO(codec)`.
- **The GPU coder, and a device-resident forward output.** Host rANS is now
  34% of a 30 fps interval to encode and 31% to decode at 1 cm. That is less
  than before, so the GPU coder waits on the same terms as the room0 entry.
  *Both landed 2026-10-03 (the device encoding entry, below).*

### 2026-10-03 — Garbage collection deletes from the device stale list.

*Amended 2026-10-13 (the coordinate entry, below):* the grid's own `remove`
takes a device list too (`VoxelBlockGrid::remove(const Buffer&, count,
...)`, beside the map's device `allocate`), and one private body -- zero the
listed blocks, then the map's device `remove` -- serves both grid overloads
and garbage collection: the host overload stages its coordinates once, where
they went up twice. The device overload checks its input first, so a refused
list zeroes nothing and keeps the topology epoch.
*Amended 2026-10-13 (the volume block-list entry, below):* the host
overloads are gone; the one body serves the grid's device `remove` and
garbage collection.

The 2026-10-01 block-stamp decision downloaded the stale list after zeroing
its attributes, then uploaded the same coordinates for `VoxelHashMap::remove`.
The new `remove(const Buffer&, count, ...)` overload binds that list where the
GPU produced it. Both overloads share the same input validation, topology
epoch invalidation, per-entry completion flags, and retry/failure protocol.
The map still does not clear attributes itself: the grid completes their
zeroing before it gives the map the device list.

For S stale blocks this removes 16*S bytes of download, 16*S bytes of upload,
the host coordinate vector, and the temporary device coordinate allocation.
The small stale count and deletion failure/heap counters are still read back;
the existing synchronous submit boundaries remain. This is a transfer-volume
reduction, not a claimed latency improvement without measurement.

Regression coverage exercises device-list deletion with duplicate and absent
coordinates, a live trailing coordinate beyond the supplied count, repeated
deletion, heap reuse, cached-list invalidation, and invalid buffer sizes/usage.
The contention fixture alternates host and device inputs. Repeated garbage
collection grows then reuses the stale list with a shorter live prefix, while
the existing stamp test checks zeroed attributes and preservation of observed
or recently requested blocks.

Release builds with warnings as errors pass all 44 tests on both the M5 Max
and RTX 4090. The six volume allocation, deletion, resize, and block-stamp
tests also pass with the Khronos layer and synchronization validation forced
on both devices. These checks validate correctness and synchronization; no
garbage-collection latency comparison was made.

### 2026-10-02 — Mesh input binning keeps counts, cursors and occupied bins on the device, retains scratch, and reads only validation and bounded dispatch control.

Amends the 2026-09-28 residency entry's mesh-input count readback. The two
triangle-candidate passes were device-local, but each call allocated eight
bulk buffers, read `4 * num_blocks` count bytes to the host, scanned them,
and uploaded the same bytes as cursors plus 12 bytes per occupied bin. The
new path removes that internal round trip; mesh vertices, indices and
candidate offsets still enter from the host once per call.

**A hierarchical exclusive scan and ordered compaction.** Each 256-slot
workgroup scans entry counts and occupied counts while reducing the largest
bin. Group summaries are scanned recursively, then their prefixes are added
back down. A sticky carry flag follows every addition; uint32 overflow is
refused before a wrapped prefix can become a cursor. Compaction emits the
same slot-ordered `MeshBin { ptr, begin, count }` records as the old host
loop. Counts become fill cursors on-device. No distance, sign, tie-break,
observation or changed-stamp rule changes. Missing blocks, excess bin size,
overflow, binding ranges and dispatch limits are still checked before the
first grid write.

**The watchdog still bounds a submission.** Let `B` be
`kMaxDispatchBinEntries`, `M` be `kMaxBinTriangles`, and `W = B - M + 1`.
Group bins by the window containing their exclusive entry prefix. Bin starts
in a window span at most `W - 1` entries; adding the last bin contributes at
most `M`, so every range has at most `B` entries. GPU binary searches find
the range endpoints in the ordered compact list. The host reads these
8-byte pairs and may split them further for the device thread limit. Each
write remains a separate submission, with the fill in the first. The
partition can use more submissions than the previous greedy packing; it
never increases a submission's entry bound.

The readback is 20 bytes of validation/size control (entry total, occupied
count, largest bin, overflow and missing blocks), plus
`8 * ceil(candidate_count / W)` bytes of range control: at most 34,960
range bytes at the uint32 candidate limit. Usually only a few ranges are
needed. This is independent of unused grid capacity. Kernels never consume
mapped host memory or a CPU-generated cursor/bin list.

**Scratch is retained.** All eight prior bulk allocations and the new scan
and range scratch grow only when needed, with 1.5x headroom bounded by
`maxStorageBufferRange`. Every call refreshes the input arrays, zeroes the
logical counts/missing value, and rebinds the current grid. A compact list
starts with at most 4096 entries; if insufficient, compaction leaves the
local prefixes intact, the host grows the list, and only compaction/range
planning retries. Steady calls keep the previous submission count apart from
the bounded partition difference. No geometry, topology or pointer identity
is used as a cache-validity shortcut.

**Validation.** The full Release suite passes 45/45 on both Apple M5 Max
(warnings as errors) and RTX 4090. On both GPUs,
`recon_tsdf_mesh_integrate`, `recon_tsdf_mesh_bins` and
`recon_volume_allocate_triangles` also pass 3/3 with forced Khronos
synchronization validation. The independent distance/winding oracle still
covers signed/shell conversion, deterministic bytes, missing blocks,
refusals and split submissions. A new 65,544-slot partial hierarchy grows the
compact list past 4096, repeats, changes the mesh in place and reverses its
winding, then returns to a smaller grid. Synthetic GPU counts are compared
with a uint64 host oracle at exact watchdog boundaries, `UINT32_MAX`,
overflow in leaf and summary levels, oversized bins, empty/padded groups,
and guarded compact retries.

**Initial timing evidence, not a universal speed claim.** Apple M5 Max,
Release, baseline `e4db453`, three interleaved baseline/change pairs per
workload; each process warms three integrations then reports nine samples.
Allocation, object creation and mesh generation are outside the measured
integration. The same metre-square +Z sheet, 1 cm voxels, 8³ blocks and 4 cm
truncation is two triangles at 393,216-slot capacity for the sparse case,
and 320,000 triangles at 16,384-slot capacity for the dense case.

| case | baseline host medians, ms | changed host medians, ms | baseline device medians, ms | changed device medians, ms |
|---|---|---|---|---|
| sparse capacity | 1.055 / 1.527 / 0.908 | 1.294 / 0.772 / 0.783 | 0.069 / 0.047 / 0.033 | 0.281 / 0.274 / 0.275 |
| dense triangles | 21.346 / 20.930 / 20.855 | 21.717 / 20.940 / 21.238 | 12.866 / 12.722 / 13.053 | 13.541 / 12.468 / 13.493 |

The scan adds GPU work; sparse host time is variable and the dense case is
roughly flat/slightly slower on this shared Mac.

**RTX 4090 evidence.** The same C++ harness and Release `-O3` workload,
baseline `e4db453` against source `13adbb3`: one excluded warmup pair, then
three interleaved baseline/change pairs, each with three warmups and nine
measured integrations. Allocation and object creation remain outside timing.

| case | baseline host medians, ms | changed host medians, ms | baseline device medians, ms | changed device medians, ms |
|---|---|---|---|---|
| sparse capacity | 0.920386 / 0.955022 / 0.906461 | 0.124897 / 0.124780 / 0.124343 | 0.009824 / 0.009376 / 0.009504 | 0.036864 / 0.036736 / 0.035904 |
| dense triangles | 16.065795 / 16.326386 / 16.322679 | 16.962439 / 16.474617 / 16.198409 | 8.807296 / 8.781312 / 8.792128 | 8.853760 / 9.141056 / 9.090464 |

The median of those three host medians is about 7.4x faster for the sparse
case (0.920386 → 0.124780 ms), while dense host time is 0.9% slower
(16.322679 → 16.474617 ms) and dense device time is 3.4% slower
(8.792128 → 9.090464 ms). Sparse device time also rises, from 0.009504 to
0.036736 ms: removing the CPU work and transfers saves wall time there even
though the GPU now performs the scan. This is a capacity-bound win with a
measured limiting case, not a general frame-rate improvement.

Both versions on both GPUs report 392 blocks; sparse has 472 bin entries
and one write submission, dense has 2,281,214 entries and three submissions.
The RTX logs are `mesh-{sparse,dense}-bench-{1,2,3}.log` under the remote
`.worktrees/gpu-residency-base-e4db453/` and
`.worktrees/gpu-mesh-fix-13adbb3/` checkouts. The demonstrated structural
change is bounded control transfer and retained bulk storage.

### 2026-10-03 — Asset loading and export belong to io, with Assimp as an optional private backend.

The codec mesh example exposed a missing library boundary: image decoding
and PLY writing lived in `examples/common`, and loading one OBJ required a
new example parser. Asset I/O is useful independently of reconstruction or
that example. Move the existing JPEG/PNG readers and PLY writer, plus the
viewer's PNG export, into the installed `recon_io` target. It depends on
`mesh` for the host mesh container; no GPU tier depends on it. This supersedes
the architecture's earlier assignment of future mesh file export to `mesh`.
Future OBJ/glTF exporters belong in `io`; they are not implemented here.

The public namespace is `volumetric_kit::recon::io`, with headers under
`io/`. APIs return owned host data or `Status`, validate file-boundary
inputs, and keep backend types private. Encoded image bytes remain encoded;
depth import requires an actual 16-bit grayscale PNG and explicit finite
units per metre. PLY's RGB bytes are sRGB-encoded from the mesh's linear
colors. RGBA8 PNG export expects already-encoded pixels. These file operations
do not create GPU resources or initiate readback; a caller explicitly chooses
to export a host mesh or image. Replica's frame selection, trajectory parsing
and camera contract remain in its capture adapter. Sensor/video decoding and
rig calibration retain their existing contracts.

Use Assimp for reusable static mesh parsing, behind a separate optional
`recon_io_assimp` target. `VR_WITH_ASSIMP` defaults OFF; ON finds an installed
Assimp CMake package, as the repository does for other compiled optional
backends. No Assimp source download or change to the CMake 3.21 floor is
needed. The installed package re-finds the Assimp that backend was built
against (same major, no older), including the private link requirement of a
static archive. A cross-compile
must provide a target-platform package. Existing stb/tinyply pins move from
the examples to private library build dependencies. stb symbols and settings
are translation-unit-local; tinyply uses a private namespace, so static
consumers can provide their own independent copies of either backend.

`load_mesh` produces positions and triangle indices, discarding materials and colour.
Assimp decodes the asset and triangulates polygons; our conversion traverses
the scene and applies every mesh instance's composed node transform. A
negative determinant reverses indices to preserve orientation. Singular,
nonfinite and non-affine transforms of mesh instances are refused; nodes
without meshes are not checked. Animated, skinned or morphed
assets require an explicitly evaluated static asset and return Unsupported.
Remaining point/line primitives are refused rather than silently dropped.

The loader joins exactly coincident finite positions, including material/UV
seams. It does not use Assimp's approximate vertex-joining or topology-repair
passes, and preserves duplicate and degenerate triangles for the consumer's
audit. It does not impose a target height, rotate to a guessed up direction,
or claim unknown source coordinates are metres. Importer unit and up-axis
conversions are switched off; Assimp before 5.3 still applies Collada's
`<unit>`, and 5.4.x converts FBX regardless. Rafa2's explicit 1.7 m
normalization remains a codec-example policy; adopting this loader there is
a separate change from this standalone module review.

Validation on Apple M5 Max with installed Assimp 6.0.5: Release builds are
warning-clean with warnings as errors, and the Assimp/FFmpeg-enabled full
suite passes 52/52. Host tests cover the PNG round trip and color packing,
16-bit depth units, 8-bit depth refusal, and the PLY byte layout. Mesh import
has no model-file tests; CI builds the loader against Assimp 5.2 and 6.0.
Both viewer executables compile with the new PNG API; no physical display
check is claimed.

A host-only import of the actual Rafa2 OBJ retains 24,998 vertices, 50,000
triangles and 75,000 consistently paired edges, positive signed volume
0.00809495816868, and projected head-up extent 0.795690825727 source units.
The loader leaves that source extent intact; the consumer must explicitly
normalize it to 1.7 m. This verifies asset ingestion, not a new codec or
end-to-end timing result.

### 2026-10-02 — Per-basis quantization and a normalized mesh codec fixture.

**Contract.** `CodecParams` replaces `dc_step` and `ac_step` with
`quantization_scale` and 512 `quantization_weights`, indexed by canonical
frequency `(u,v,w)` as `u + 8*v + 64*w`. The effective step is their product,
in fractions of `trunc_dist`, for the existing round-half-to-even scalar
quantizer. DC is entry zero. All blocks share the same table. Only the K bases the
zigzag cutoff keeps use their weights, so only those are checked and carried.
A table's DC weight of one is a useful normalization convention for comparing
global scales; it is not a format restriction. The scale and each kept weight
must be positive and normal, and each kept basis's effective step must satisfy
the existing step bounds. Normal factors keep validity independent of a host
or device that flushes subnormals to zero; the first draft accepted subnormal
factors with a normal product, and checked weights beyond K too.

The host computes effective steps once and uploads them to device-local
memory in the transform's `CommandBatch`. Both directions use the same
canonical table through the zigzag map. The DCT, observed mask and entropy models are
unchanged; each coefficient already had its own rANS probability model.

**Frame v3.** The former two steps become the global scale and a weight
count equal to K. The kept bases' float weights follow the 44-byte prefix in
zigzag order, so the fixed header is `44 + 4K` bytes: 300 at K = 64 and
2,092 at K = 512, with no per-block table overhead. Weights beyond K never
apply, are not carried, and read back as 1. The first draft carried all 512
weights, 2,048 bytes in every frame and about a tenth of a 1 cm Rafa2 frame.
`read_frame_info` recovers the parameters without entropy decoding.
All intra frames remain self-contained; v1 and v2 are `Unsupported`.

**The study.** The C++ examples share three candidate tables, all with DC
weight one and axis-permutation symmetry:

- `uniform`: `Q(u,v,w) = 1`;
- `band`: `Q(u,v,w) = 1 + 0.25*(u+v+w)`;
- `radial`: `Q(u,v,w) = 1 + 0.125*(u*u+v*v+w*w)`.

These are deterministic hypotheses, not fitted tables. The radial table
can distinguish bases within a total-frequency band. K = 64 is the prior
baseline and cuts shell 6; 84 and 120 retain complete shells through degree
6 and 7; 512 retains every mode. Each is measured at scales 0.05, 0.1, 0.2,
0.4 and 0.8, plus a uniform K = 512 / scale = 0.002 near-lossless control.
The complete 61-row study uses a warm-up and three timed encode/decode
rounds. Host and device timings are separate; quality measurement and mesh
extraction are outside them. Decoding repeatedly measures steady state on
an already-sized player grid, excluding resize/retry cost. The table shape
is judged against total frame bytes and mesh error, not equal scale values
across differently weighted tables. Default weights remain uniform while
the new candidates are evaluated.

**Rafa2 units and orientation.** The supplied `Rafa2/Frame_00001_textured.obj`
declares VologramsAPI, 24,998 vertices and 50,000 triangles, but no physical
unit or up axis. Its raw bounds are X [-0.28273, 0.47853],
Y [-0.10953, 0.054242], Z [0.67838, 1.1272]. Projection inspection shows a
complete person tilted in XZ, with the head towards -X/-Z. The explicit
head-up vector `(-0.9120591159, 0.0661250017, -0.4046920474)` is the negative
long principal axis of the vertex positions, visually checked for sign.
Measured on the float vertices `codec_mesh` reads, its full projected
extent is 0.795690825737 source units; scaling by 2.13650823286 gives a
bounding height of 1.7 m. This is the normalization convention, not a claim
of a recovered source unit or measured stature. `codec_mesh` rotates this
direction to +Y with a proper rotation, centres X/Z and places the minimum Y
at zero, without changing the source asset. *Amended 2026-10-14 (below):*
that arithmetic is `io`'s opt-in `up_axes`, `fit_transform` and
`apply_transform`, and `codec_mesh` prints the same two figures.

The indexed mesh has one connected component, 75,000 edges and no boundary,
nonmanifold or inconsistently oriented edges. Its signed volume is positive
(0.008094958217 source units cubed), with no degenerate triangles. This
supports the signed closest-face conversion; self-intersections were not
checked. The actual path is `allocate_from_triangles` →
`tsdf::MeshIntegrator` → `codec::Encoder` → `codec::Decoder` into a separate
grid → `mesh::MarchingCubes`. The example reports conversion against the
normalized original, codec loss against the uncompressed extracted surface,
and total decoded error against the normalized original separately. PLYs
for all three surfaces and the actual compressed frame can be saved.

**Measured locally, M5 Max, Release, Vulkan/MoltenVK.** Room0 fuses all 400
available frames and measures the final grid (24,885 encoded blocks), not
the stream-average workload of the earlier decisions. Rafa2 is one supplied
mesh frame, at both 1 cm (600 encoded blocks) and 5 mm (2,395). Every dataset
runs the complete 61-configuration sweep. Distances below are sampled
mesh-to-mesh accuracy RMS / coverage RMS in mm; the metric samples every
fourth vertex. Reach is 4 cm at 1 cm voxels and 2 cm at 5 mm voxels; samples
beyond reach are counted separately, never folded into RMS. Times are
means over three warmed codec calls, host ms and device ms respectively.

| Content | Table / K / scale | Frame bytes | RMS acc / cov (mm) | Host encode / decode (ms) | Device encode / decode (ms) |
|---|---|---:|---|---|---|
| Room0, 1 cm | uniform / 64 / 0.2 | 273,871 | 0.640 / 0.814 | 17.11 / 15.46 | 1.19 / 0.90 |
| Room0, 1 cm | radial / 64 / 0.1 | 271,663 | 0.654 / 0.815 | 17.73 / 15.59 | 1.86 / 1.13 |
| Rafa2, 1 cm | uniform / 64 / 0.2 | 19,986 | 0.889 / 0.838 | 1.28 / 1.42 | 0.08 / 0.05 |
| Rafa2, 1 cm | radial / 64 / 0.1 | 19,454 | 0.983 / 0.893 | 1.11 / 1.11 | 0.07 / 0.05 |
| Rafa2, 5 mm | uniform / 64 / 0.2 | 68,737 | 0.416 / 0.392 | 3.45 / 3.21 | 0.18 / 0.13 |
| Rafa2, 5 mm | radial / 64 / 0.1 | 66,971 | 0.434 / 0.406 | 3.24 / 3.38 | 0.20 / 0.12 |

Frame bytes here were measured with the first draft's 512-weight header. The
final header carries K weights, so each K = 64 frame is 1,792 bytes smaller
with an identical payload (1 cm Rafa2 uniform: 18,194 bytes); the K = 512
rows are unchanged.

These paired rows are close in rate, not exactly matched. Their rate/error
tradeoff does not establish a consistent improvement, so the default table
remains uniform. The band table at K = 64 / scale = 0.1 gives 69,686 bytes
and 0.401 / 0.375 mm on 5 mm Rafa2: slightly better than the baseline and
slightly larger. A fitted table should be compared at explicitly matched
rates before being promoted. K = 512 removes the cutoff but remains costly:
room0's band / 512 / 0.2 is 259,494 bytes at 0.666 / 0.659 mm, costing
76.50 / 125.30 host ms; the low error alone does not make it a live default.
All three listed room0 rows (uniform/radial K = 64 and band K = 512) have
zero accuracy samples and 29 coverage samples beyond reach; all listed
Rafa2 rows have zero in either direction. The near-lossless control's
accuracy RMS is 0.019 mm at 1 cm Rafa2 and 0.010 mm at 5 mm Rafa2.

For Rafa2 the separate mesh-to-TSDF conversion accuracy / coverage RMS is
0.091 / 0.403 mm at 1 cm and 0.030 / 0.163 mm at 5 mm. With default codec
parameters the total decoded-to-original error is 0.891 / 0.857 mm and
0.415 / 0.381 mm, respectively. These errors are measured independently,
not added. Exported input Y is [0, 1.70000004768] m. A double-precision
inspection of the OBJ text gives 0.795690789501 and 2.136508330159 instead;
the difference is rounding into the float vertex format.

**Reproduce** with the Release executables and an absolute `dataset_root`:

```sh
"$recon_root/build/examples/codec_replica/codec_replica" \
  "$dataset_root/replica_room0/room0" --max-frames 400 \
  --voxel 0.01 --encode-every 0 --preload --sweep
"$recon_root/build/examples/codec_mesh/codec_mesh" \
  "$dataset_root/Rafa2/Frame_00001_textured.obj" \
  --height 1.7 --up-vector -0.9120591159,0.0661250017,-0.4046920474 \
  --voxel 0.005 --mode signed --sweep -o "$recon_root/build/rafa2"
```

Repeat Rafa2 with `--voxel 0.01` for its coarser row. To export the measured
nonuniform example, use `--quant-table radial --step 0.1 --k 64` and a
different output prefix. Sweeps print the entire table; sizes include all
headers, masks, coordinates, entropy tables and coefficients. Timings do
not establish capture-to-display throughput or discrete-GPU performance.
Rafa2 has only one frame, so it supplies no temporal quality evidence.
*Amended 2026-10-14 (the mesh-sequence entry, below):* this file is frame 1
of V-SENSE's Rafa2, whose frames 1–150 `codec_mesh_sequence.json` codes
under one transform fitted to them all.

The exported 5 mm radial / K = 64 / scale = 0.1 frame was independently
parsed: 66,971 bytes, 2,395 blocks, all 512 weights of the first-draft
header, and section
lengths consuming the exact file. Its marching-cubes output has 199,864
triangles (198,220 before coding). A shared-view rendering of normalized
input, uncompressed TSDF and decoded output retains the body silhouette,
but shows fine rippling/faceting after coding. Sub-millimetre positional
RMS is not visual losslessness; smooth-surface shading remains a reason to
tune tables and examine more than the aggregate positional metric.

**Validation.** The Release build is warning-clean with
`VR_WARNINGS_AS_ERRORS=ON`; all 45 tests pass. The three GPU codec tests also
pass with the Khronos layer explicitly enabled and synchronization
validation on, as does the full 5 mm radial Rafa2 conversion/codec/meshing
example, with no validation messages. Added regressions cover nonuniform and directional basis
weights, A/B/A table changes on a transform and public decoder, malformed
and truncated v3 parameters, effective-step bounds, subnormal factors,
input-unit invariance, oblique height normalization and
winding preservation, and OBJ/topology refusals. No discrete GPU or new
sanitizer run was performed for this change.

### 2026-10-03 — Compact the codec's observed list on the device and retain transform scratch.

*Amended 2026-10-13 (the coordinate entry, below):* the encoder sorts the
list on the device, so it never leaves it: the filter reads back its count
and reject tally alone, and the predicted prefix and its tail submit go.

The GPU memory audit found two avoidable costs inside an otherwise staged
codec. The observed filter downloaded every allocated `BlockIndex` (16 bytes)
plus a four-byte flag, then discarded unobserved entries on the CPU. Forward
and inverse also recreated their list, coefficient and mask buffers on every
call. This change is stacked on the per-basis v3 codec; it changes neither the
frame format nor the CPU coordinate sort and rANS reference implementation.

**Compact before readback.** One workgroup still checks one block's weights
and resolves its coordinate through the hash table. Its leader atomically
appends the input `BlockIndex` when any voxel is observed. A two-word counter
buffer holds rejected and observed counts, reset before the call's dispatches.
Output order is unspecified; `Encoder` still sorts by (z, y, x), preserving
the same content's exact frame bytes regardless of scheduling.

Downloading the exact list only after learning its count would require two
fence waits every time. Instead the filter follows the hash map's
`collect_compacted`: download the count and a predicted list prefix in the
same batch, then download a missing tail only if the result grew past the
prediction. A call with no previous non-zero count (the first, or one after an empty
result) predicts the whole input; later calls predict the last observed count
plus 25%, capped by the current input count. The first draft predicted zero
after an empty result, so the next non-empty call always paid the second
batch. Empty inputs submit nothing. The returned observed
count is checked against input capacity before resizing or reading a tail.

For A input blocks, N observed blocks and prefix G, total observed readback is
`8 + 16*max(G,N)` bytes, bounded by `8 + 16*A`, against the old `4 + 20*A`.
The steady one-in-five case predicts about A/4 and transfers about `4*A + 8`
bytes, with one submission. The first/all-observed case transfers `16*A + 8`.
These are byte counts, not a latency claim; atomics add device work and growth
can still add a fence. Device compaction/sorting of coordinates and GPU rANS
remain separate work. The host coder still requires the coefficient/mask
readback on encode and upload on decode.

**Retain scratch.** Each transform owns grow-only device buffers for the
observed output, sorted input list, packed coefficients and masks. A fitting
call makes no new allocation for these buffers. A grow releases the old
buffer before creating its replacement, with 1.5x headroom, so a slowly
growing input does not reallocate on every frame and never holds both;
`ensure_device_scratch` in `core/compute_util.hpp` does this for the codec
and the mesh integrator alike. Descriptors bind logical
ranges, and transfers use logical counts, so a smaller list or K cannot expose
an old suffix. Scratch remains at its largest size until destruction; at A
input blocks, N coded blocks and padded coefficient count Kp the high-water
sizes are up to 1.5x 16A, 16N, 2NKp and 64N bytes respectively. The obsolete flags buffer
is removed. Batch staging allocations remain per-call; the per-basis 2 KiB
effective-step upload and its barriers are unchanged.

**Validation.** Release with `VR_WARNINGS_AS_ERRORS=ON` builds and all 45 tests
pass. New GPU regressions cover mixed/all/none observed results, zero-count
regrowth, input growth/shrinkage, empty input, original coordinates and ptrs,
and forced two-workgroup dispatch chunks. Forward/inverse reuse through count
and K changes, odd padding and emptiness is compared with the independent CPU
DCT reference, including unchanged blocks outside the current list. Existing
tests retain deterministic frame bytes, stale-list refusal and A/B/A
quantization-table coverage. The three GPU codec tests also pass with the
Khronos layer explicitly enabled and synchronization validation on, with no
validation messages. The same full 45-test suite and three forced-Khronos
synchronization-validation codec tests pass on the RTX 4090 in Release.

**Apple timing check.** A one-off C++ harness linked
the unchanged `23c6e36` baseline and this change separately, both Release/O3 on
Apple M5 Max through Vulkan/MoltenVK. It allocated 32,768 blocks in a 65,536-slot
grid, marked one voxel in each selected block, and timed only `observed` with
five warmups and 50 measured calls per density. Allocation, weight upload and
active-list construction are outside the measurement. Host spans include
recording, submission, waiting and readback; device spans cover dispatches.

Two alternating baseline/changed runs showed device warm-up drift in the first
baseline run, so these are the second pair, not an aggregated speedup claim:

| Observed blocks | Baseline host mean / median / p95 (ms) | Changed host mean / median / p95 (ms) | Baseline / changed device mean (ms) |
|---:|---|---|---|
| 0 | 0.408 / 0.380 / 0.488 | 0.348 / 0.305 / 0.404 | 0.164 / 0.147 |
| 6,554 | 0.333 / 0.328 / 0.365 | 0.294 / 0.292 / 0.327 | 0.117 / 0.118 |
| 32,768 | 0.381 / 0.379 / 0.395 | 0.313 / 0.312 / 0.324 | 0.116 / 0.120 |

The old pass reads 655,364 bytes at every density. The warmed new pass reads
8, 131,080 and 524,296 bytes respectively. The all-observed row illustrates
the small extra atomic device work despite less host traffic. This is a
synthetic filter-only check; it does not measure total codec latency, scratch
allocation savings, real capture-to-display throughput, or discrete-GPU gains.

**RTX 4090 timing check.** The same `/tmp/codec_observed_bench.cpp` harness ran
on the idle RTX 4090 host `taojin-desktop`, both variants built Release/O3:
baseline `23c6e36` and implementation `525bdf7` (before this documentation-only
amendment). One interleaved baseline/changed pair was excluded as warm-up;
three subsequent pairs each used five warmups and 50 measured calls per case.
The workload and timing scopes are exactly those above. Each table entry is
the **median of the three per-run medians**, with the minimum and maximum
per-run median in brackets; these are not pooled-sample percentiles.

| Observed blocks | Baseline host (ms) | Changed host (ms) | Baseline device (ms) | Changed device (ms) |
|---:|---|---|---|---|
| 0 | 0.1241 [0.1221–0.1254] | 0.0511 [0.0507–0.0512] | 0.0245 [0.0244–0.0247] | 0.0240 [0.0239–0.0243] |
| 6,554 | 0.1643 [0.1643–0.2315] | 0.0695 [0.0693–0.0697] | 0.0246 [0.0244–0.0246] | 0.0246 [0.0246–0.0247] |
| 32,768 | 0.1428 [0.1427–0.2113] | 0.1120 [0.1120–0.1148] | 0.0248 [0.0248–0.0250] | 0.0334 [0.0334–0.0335] |

The second baseline run's host times rose without a corresponding device-time
increase, consistent with host scheduling noise; the ranges retain that run.
Host filter latency is lower at all three densities. With every block observed,
the append atomics increase device time from about 0.0248 to 0.0334 ms even
though less readback and host work reduce total filter time. This establishes
a discrete-GPU improvement for this synthetic filter workload, not total codec
or live-pipeline throughput, and does not isolate retained-transform-scratch
savings. Raw runs 1–3 are retained on `taojin-desktop` under
`/home/taojin/ws/volumetric_kit/volumetric_kit_recon/.worktrees/`, in
`gpu-codec-base-23c6e36/observed-bench-{1,2,3}.log` and
`gpu-codec-fix-525bdf7/observed-bench-{1,2,3}.log`.

### 2026-10-03 — Codec mesh evaluation consumes the shared io loader.

The codec change is stacked above the standalone asset I/O module. Its
`codec_mesh` example now calls `io::load_mesh` directly and passes the owned
`io::TriangleMesh` into its normalization and topology checks. The bespoke
OBJ parser and duplicate geometry container are removed. PLY exports use
`io::write_ply`, matching the room example. This completes the codec adoption
left separate by the asset I/O decision above.

`examples/codec_mesh/mesh_normalization.*` contains the explicit height/up
convention, topology audit and conversion to the evaluator's host mesh. Rafa2
is a test and demo asset, so this is demo policy, compiled into `codec_mesh`
alone and not tested as library code. *Amended 2026-10-14 (below):* the
normalization and the audit, which were never Rafa2's, are opt-in `io` helpers
tested there; the example keeps its host-mesh conversion (`geometry_mesh.*`),
and Rafa2's up vector and height stay in its configuration. The audit runs
after normalization, on the float metres the conversion reads, so a triangle
that collapses there counts as degenerate. *Amended 2026-10-14 (the
mesh-sequence entry, below):* `codec_mesh` codes a numbered range under one
transform, and audits each frame on its source coordinates first, to refuse
before any GPU work, and again on the placed ones, which decides. Both codec
examples share `parse_number.hpp` for their numeric flags, and `codec_mesh`
grows its grid only for a capacity limit, retrying lock contention. These are
consumer choices, so they stay outside the format loader. *Amended 2026-10-08
(below):* `parse_number` is `cli.hpp`'s, and both examples declare their codec
flags through `codec_flags.hpp`.
Quantization-table candidates likewise remain codec-example policy.
`codec_mesh` requires `VR_WITH_ASSIMP=ON`; the room codec, production codec
library and tests build without it.

The 1.7 m Rafa2 convention and signed-mode checks are unchanged. Assimp's
format parsing, triangulation and exact position joining are now inherited
from `io`; successful import still does not certify a closed manifold.
Earlier rate-distortion and latency tables describe their recorded builds,
not a new performance measurement of this integration.

Validation of the stacked integration on Apple M5 Max: Release with warnings
as errors, Assimp and FFmpeg passes all 54 tests. The three GPU codec tests
and the actual 5 mm Rafa2 radial/K64/0.1 pipeline pass with Khronos
synchronization validation forced on, without validation diagnostics. The
import retains 24,998 vertices and 50,000 triangles with closed, consistently
oriented edges; explicit normalization produces Y bounds [0, 1.70000005] m.
The pipeline emits all three PLYs and a 66,973-byte frame, decodes 2,395
blocks, and extracts 199,864 decoded triangles. This is a functional smoke
run with validation enabled, not a latency benchmark. An Assimp-disabled
Release/Werror build also succeeds, and its image/PLY host tests pass.

**Review follow-up.** Rebased onto #156 and #158. On Apple M5 Max, Release
with warnings as errors and Assimp, all 51 tests pass, and an Assimp-off build
passes its codec and IO tests. The codec DCT, encoder and decoder tests and
the mesh-integrate test report no messages with the Khronos layer forced on
and synchronization validation. `codec_mesh` on a box OBJ writes a
15,179-byte frame whose 300-byte header carries 64 weights. No new rate or
latency measurement was made. The normalization's own test, listed in the
2026-10-02 validation, is removed with it from the shared code.

### 2026-10-03 — rANS encoding runs on the device, byte for byte the host's frame, chosen per frame by segment count.

*Amended 2026-10-08 (the frame grammar entry, below):* the walk is the
host's own `emit_block`, in emit order, from the GLSL both sides compile, and
`rans_ops.comp` writes each block's steps down from the end of its run.
*Amended 2026-10-13 (the device-only encoder):* encoding runs on the device
alone too, by the rule the decoding entry's amendment below gives, so
`EncoderConfig::entropy`, `EntropyCoding`, `kMinDeviceEncodeSegments` and the
`kAuto` policy (`entropy_choice.hpp`) go. `Encoder::create` builds the device
writer and fails if it cannot, and a frame the device cannot hold is refused
with the device's code. The CPU coder -- `RansWriter` and `RansReader`,
`write_frame`, `read_frame` and the transform's host forward and inverse --
moves under `tests/` (`codec_oracle*.hpp`, the `recon_codec_oracle` test
library): the independent oracle the codec tests hold the device to, byte
for byte, the one place the user kept a CPU coder. The library keeps what the
device path uses -- the rANS tables and their normalization, the frame's
layout, parsing and shared checks, `frame_hash` and the merges -- and no
function that only tests call. The examples' `codec.entropy` takes `auto`
and `device` alike and refuses `host`; core drops the key in its next config
change. *Amended 2026-10-14:* core 0.7.0 dropped it. Two host steps remain
in the encoder, each a `TODO(codec)`: the coordinate sort, whose observed
list crosses the bus both ways, and the scan of each entry's coder steps into
the segments' step and slot offsets. A frame of few segments, which `kAuto`
coded on the host, pays for it. Hand
check on the M5 Max, Release, the median of six runs before and six
after, interleaved:
- room0 at 1 cm, Classic, 40 I-frames of about 21,900 blocks in 343
  segments of 64, which `kAuto` already coded on the device: 4.12 → 3.94 ms a
  frame of host time and 1.97 → 1.95 ms of device time, `..rans encode`
  1.93 ms both, within noise;
- Rafa2 at 5 mm, 2,395 blocks in 38 segments of 64, which `kAuto` coded on
  the host, as `codec_mesh`'s sweep times a warm frame: 2.85 → 2.97 ms of
  host time at K = 64 and step 0.2 (+0.12 ms, +4%), its device time
  0.07 → 2.15 ms.

Every frame's bytes are unchanged. *Amended 2026-10-13 (the coordinate
entry, below, and its amendment):* neither host step remains; the sort and
the step layout run on the device.

The 2026-09-26 decision's fifth PR, for the encoder: the v3 format is
unchanged, and every frame the device writes is `write_intra_frame`'s.

**The pipeline.** The forward output stays in device memory
(`DctTransform::record_forward`). In the forward's own batch, one invocation
per block walks its frame symbols (`rans_walk.glsl`, the reverse of
`emit_block`) to count every model's symbols and the block's coder steps.
Only those counts cross to the host, which builds the tables with the host
writer's `frame_tables`, lays each segment's blocks last first so a
segment's steps are one run, and sizes each segment's output slot at one
word per step plus the state. A second batch writes every block's steps in
parallel (`rans_ops.comp`, `start | freq << 16`), runs one invocation per
segment over its run with `RansWriter::finish`'s integer arithmetic
(`rans_encode.comp`), packs the streams with a scan and a gather sized from
the payload by an indirect dispatch, and reads back the segment lengths and
the payload. The payload is read up to the last frame's bytes per block plus
25% (32 before any frame) with the lengths, and any rest after. The
container comes from `assemble_intra_frame`, which the host writer now uses
too and which checks the header fields and sections it is given.

**Memory, discrete first.** Every bulk buffer is device-local and retained
(`ensure_device_scratch`), and the host reaches them only through
`CommandBatch`, write-combined staging up and cached staging down. Per frame
the bus carries the symbol counts (8 KB at K = 64), 4 bytes per block of step
counts up and step offsets down, 8 bytes per segment of offsets, and the
lengths and payload (about 12 bytes per block) up. The host path instead
reads back 64 + 2K bytes per block of masks and coefficients, 192 at K = 64.

**Exactness.** Integer arithmetic only. The device frame test compares the
device's bytes with `write_intra_frame`'s over 0 to 700 blocks, K of 1, 20,
64 and 512, segments of 1, 16, 64 and 1,024 blocks, and int32 coordinate
extremes. Five planted kernel bugs each failed it: chunk order, mask symbol
order, coordinate order, a gather index and the scan's carry. The encoder's
contract cases run again with device coding, and an automatic choice past the
threshold writes the host's bytes. The tests pass on Apple M5 Max through
MoltenVK and on an RTX 5090 under Ubuntu 24.04, GCC 13.

**Why the walk is a pass of its own.** A segment is one serial rANS chain,
and one GPU invocation takes about 0.2 µs a coder step on the M5 Max against
about 8 ns on a CPU core. Rafa2 at 1 cm on the M5 Max, R = 64 (10 segments),
coding ms: the walk inside the coding loop with 64 segments a workgroup,
12.1; the same with the tables in shared memory, 12.1; one segment a
workgroup with whole-word writes, 2.15; the walk in its own pass, 2.07; steps
loaded four ahead, 1.96. Dividing by libdivide's reciprocal instead of `/`
measured 2.60 through MoltenVK, so it was reverted. On the RTX 5090 the
separate walk took room0 at R = 64 from 3.95 to 1.50 ms.

**Measured**, Release, host / device `..rans encode` ms, frames identical in
each pair. The device's counting pass shares the forward's batch, so it is
timed in `..forward`, not here; the whole-encode figures below include it. Room0 at 1 cm codes every frame of 60 (14,728 blocks); Rafa2 is
the 1.7 m normalized mesh, coded 21 times in a row by a temporary loop in
`codec_mesh`, not committed:

| Content | Machine | R = 64 | R = 32 | R = 16 | R = 8 |
|---|---|---|---|---|---|
| room0 | M5 Max | 7.15 / 1.74 | | 7.25 / 0.84 | |
| room0 | RTX 5090 | 8.41 / 1.50 | | 8.45 / 0.72 | |
| Rafa2 5 mm, 2,395 blocks | M5 Max | 2.38 / 2.12 | | 2.28 / 0.84 | |
| Rafa2 5 mm | RTX 5090 | 3.01 / 1.70 | 3.00 / 1.00 | 3.01 / 0.64 | 3.02 / 0.45 |
| Rafa2 1 cm, 600 blocks | M5 Max | 0.59 / 1.96 | | 0.59 / 0.79 | |
| Rafa2 1 cm | RTX 5090 | 0.80 / 1.48 | 0.80 / 0.90 | 0.80 / 0.57 | 0.80 / 0.33 |

The whole room0 encode at R = 64 goes from 8.98 to 3.12 ms on the M5 Max and
from 10.67 to 3.06 ms on the RTX 5090. Its forward drops from 0.68 to 0.39 ms
and from 1.30 to 0.40 ms, since the coefficients no longer come back. Smaller
segments buy the device parallelism at some size: room0 is 2.5%, 7.5% and
17% larger at R = 32, 16 and 8, and 1 cm Rafa2 0.9%, 2.8% and 6.7%.

**Chosen per frame.** The device's time follows its longest segment and the
host's the whole frame, so the device loses on a frame of few segments: it
breaks even near 20 segments on the RTX 5090 and 45 on the M5 Max. The
choice changes only the time, so `EntropyCoding::kAuto`, the default, codes
on the device from `kMinDeviceSegments` (48; `kMinDeviceEncodeSegments` since
the decoding decision below) and on the host below.
`kHost` and `kDevice` force one. `kAuto` builds the device coder at its
first device frame, and codes on the host any frame the device could not:
one past `maxStorageBufferRange` or free memory, or kernels that do not
build. Only `kDevice` reports those failures.

**Not done.** Decoding stays on the host (moved to the device by the next
entry). The tables are still normalized on the host, which costs the second
batch's round trip. Lanes within a segment (format v4) are the next step if
frames of few segments need the device; host threads per segment remain a
`TODO(codec)` in the writer.

**Validation.** Apple M5 Max, Release with warnings as errors and Assimp: all
48 tests pass, and the four GPU codec tests report no messages with the
Khronos layer forced on and synchronization validation. RTX 5090, Ubuntu
24.04 container: the seven codec tests pass. No sanitizer run was made
locally.

### 2026-10-03 — rANS decoding runs on the device, block for block the host's, chosen from 80 segments.

*Amended 2026-10-13 (the P-frame decoding entry, below):* decoding runs on
the device alone. The host does I/O -- a mesh or a camera stream read in, a
model saved, a frame's encoded bytes to and from a `.vrst` file or the
network -- and the control around it: a frame's header and tables, and small
counts and status words read back. Every other step runs on the GPU, with no
host fallback (the user's rule, 2026-10-13). So `DecoderConfig` and
`kMinDeviceDecodeSegments` go, and `EntropyCoding` is the encoder's alone.
`Decoder::create` builds the device reader and fails if it cannot, and a
frame the device refuses (a segment past `kMaxDeviceDecodeSegmentSize`, a
buffer past `maxStorageBufferRange` or free memory) is refused with the
device's code; `Encoder::create` refuses a `segment_size` past that limit,
so the library writes no frame it cannot decode. The host reader stays, as
the tests' oracle only. A frame of
few segments, which `kAuto` decoded on the host, pays for it. Hand check on
the M5 Max, Release, the median of six runs before and six after,
interleaved:
- room0, 400 I-frames at 1 cm of about 340 segments each, which `kAuto`
  already decoded on the device: unchanged, 4.06 → 4.03 ms a frame of host
  time and 1.93 ms of device time;
- Rafa2 at 5 mm, 2,395 blocks in 38 segments of 64, which `kAuto` decoded on
  the host, as `codec_mesh`'s sweep times a frame the player holds: 3.29 →
  3.99 ms of host time at K = 64 and step 0.2 (+0.70 ms, +21%), its device
  time 0.15 → 2.17 ms; 3.29 → 4.11 ms over all the K = 64 rows; K = 512
  gains, 13.92 → 8.91 ms. At R = 16 the table below has the device's rANS
  decode of this frame at 0.87 ms against the host's 1.85; lanes within a
  segment, a format change, would keep R = 64.

The 2026-09-26 decision's fifth PR, for the decoder. The v3 format is
unchanged. The device decodes every frame it can hold to `read_intra_frame`'s
blocks, and refuses every corrupt segment it refuses, with the same message.

**The pipeline.** The host parses the frame: the header, sections, tables,
segment lengths and the heap check (`parse_intra_frame`, which
`read_intra_frame` now calls too). It uploads the tables (8 KB at K = 64) and
the payload. One batch expands each model's table into a 4096-slot lookup on
the device (`rans_slots.comp`, about 0.02 ms). It then decodes one segment
per workgroup (`rans_decode.comp`: `RansReader` and `read_block` in the same
32-bit arithmetic) into the inverse transform's layout. Only the coordinates,
16 bytes a block for the decoder's merge, and each segment's fault come
back. `check_segment` judges them in segment order, the one place both
readers' refusals come from. The inverse reads the coefficients and masks
where they were decoded (`DctTransform::inverse` over `ResidentBlocks`); the
host path uploads 64 + 2K bytes a block instead.

**Exactness.** The device frame test decodes every frame of the writer
comparison on both sides (0 to 700 blocks, K of 1 to 512, segments of 1 to
1,024 blocks, int32 extremes): the same coordinates, masks and coefficients.
Of 800 payloads with one to three random bytes changed, the two sides agree
on every one, with the same message: 40 decode, 179 step a coordinate out of
int32, 511 are corrupt and 70 start out of order. The decoder's contract
cases run again with device decoding, and host, device and automatic
decoding leave identical grids.

**Measured**, Release, host / device `..rans decode` ms, parsing included.
Room0 at 1 cm decodes every frame of 60 (14,728 blocks); Rafa2 is coded 21
times in a row by the same uncommitted loop in `codec_mesh` as before:

| Content | Machine | R = 64 | R = 32 | R = 16 | R = 8 |
|---|---|---|---|---|---|
| room0 | M5 Max | 6.11 / 1.79 | | 6.08 / 0.76 | |
| room0 | RTX 5090 | 6.09 / 2.07 | | 5.98 / 0.77 | |
| Rafa2 5 mm, 2,395 blocks | M5 Max | 1.94 / 2.48 | 2.03 / 1.51 | 1.85 / 0.87 | 1.78 / 0.61 |
| Rafa2 5 mm | RTX 5090 | 1.89 / 2.45 | 1.90 / 1.36 | 1.90 / 0.77 | 1.88 / 0.47 |
| Rafa2 1 cm, 600 blocks | M5 Max | 0.46 / 2.83 | 0.47 / 1.18 | 0.46 / 0.84 | 0.52 / 0.61 |
| Rafa2 1 cm | RTX 5090 | 0.50 / 2.32 | 0.50 / 1.31 | 0.50 / 0.73 | 0.50 / 0.39 |

The whole room0 decode goes from 8.54 to 3.29 ms on the M5 Max and from 8.07
to 4.08 ms on the RTX 5090 at R = 64, and to 2.27 and 2.75 ms at R = 16.

**One segment per workgroup.** Segments diverge at every mask and coordinate
branch: 32 of them a workgroup took room0 at R = 64 to 4.62 ms, 8 to 2.91,
and one to 2.06 (M5 Max, an earlier and busier run than the table's).
Loading the next stream word ahead changed nothing (2.04 ms), and a binary
search of each model's 8 KB table in place of the 1.2 MB lookup was slower
(3.10 ms); both were reverted.

**Chosen per frame, later than encoding.** Decoding cannot take the symbol
walk out of the serial chain, as encoding did: each symbol's model depends on
the symbols before it. A block costs the device 30 to 45 µs of its segment's
chain and the host 0.4 to 0.8 µs, and the device's batch adds a fixed 0.1 to
0.4 ms. At R = 64 the device breaks even near 48 segments on Rafa2 on both
machines, and on room0 near 70 on the M5 Max and 80 on the RTX 5090, so `EntropyCoding::kAuto` decodes on the
device from `kMinDeviceDecodeSegments` (80); the encoder's threshold is
renamed `kMinDeviceEncodeSegments`. Smaller segments need more of them, since
the fixed cost stays and each covers fewer blocks: 1 cm Rafa2 wins only at
R = 8 on the RTX 5090. Like the encoder, `kAuto` builds the device reader at
its first device frame and decodes on the host any frame the device could
not; `kHost` and `kDevice` force one. The examples' `--entropy` sets both.

**What the device refuses.** The frame sets the segment length, and one
invocation decodes a whole segment, so a stream of long segments could run
one invocation past a GPU watchdog and lose the device shared with the
renderer. The device therefore decodes segments of at most
`kMaxDeviceDecodeSegmentSize` (1,024 blocks, about 50 ms at the measured 30 to
45 µs a block). `kDevice` refuses a longer segment, and a frame past
`maxStorageBufferRange` or free memory, which the host decodes; every buffer
size is checked before any grows. Both coders now share the `kAuto` policy
(`entropy_choice.hpp`): the host takes a frame the device refused as
`InvalidArgument` or `OutOfMemory`, and every frame after a failed kernel
build, which is tried once; any other device failure is reported, where the
encoder's `kAuto` hid it before.

**Not done.** Lanes within a segment (format v4) would shorten the chain,
which is what decoding few segments on the device needs; host threads per
segment remain a `TODO(codec)` in the writer.

**Validation.** Apple M5 Max, Release with warnings as errors and Assimp: all
tests pass, and the four GPU codec tests report no messages with the Khronos
layer forced on and synchronization validation. RTX 5090, Ubuntu 24.04
container: the seven codec tests pass. No sanitizer run was made locally.

### 2026-10-06 — Incremental mesh extraction is removed; the `changed` stamp stays, and the mesher no longer reads it (amends the 2026-08-09, 2026-08-11, 2026-08-12, 2026-09-30, 2026-10-01 and 2026-10-02 mesh entries).

*Amended 2026-10-06 (view culling, below):* the caller's list is a
device-resident `DeviceBlockList`, and the `BlockList` overload and its
density guard went.

**The rule.** `MarchingCubes` meshes the whole active set (`extract_host`,
`extract_device`) or a caller's `BlockList`, and every extract rewrites its
slot from zero. There is no incremental mode.

**Why.** Nothing live used it. `fuse_viewer`, `rig_viewer` and the shipping
iOS scanner run with `slot_count` 3 and the spans off, a configuration the
incremental path refused by falling back to a full extract; only
`fuse_replica --incremental` and an iOS benchmark build turned it on. Its win
was never measured: room0 caps it at ~1.2x (83% of blocks re-mesh per
window), and the iPad's ~4x is the 2026-08-09 entry's ceiling estimate, not a
timed extract. And it is what made view culling complicated: the culled
overload had to withhold arena state, the span table needed a generation, a
per-slot stamp and a topology anchor, and retirement left degenerates that
needed an occupancy ceiling. View culling, a later change, makes an extract
cost the visible region instead.

**What goes.** `extract_device_incremental`;
`MarchingCubesConfig::track_block_spans`; `BlockSpan`, `block_spans()`,
`block_span_capacity()`, `block_spans_generation()`, `block_span_valid()` and
the span table behind them (device buffer, host copy, per-slot stamps, serial
and epoch anchors, stale-span clearing, span readbacks); the arena state
(watermark, tick, iso) and `kMaxArenaOccupancy`; `ExtractTimings::incremental`
and `remeshed_blocks`. In both sparse kernels: the span writes, the stamp
dilation, in-place range reuse and the retire pass, the span and stamp
bindings (the default kernel now binds 8, the sharing one 9), the
`write_spans`, `incremental` and `since` push constants (the block is now 52
bytes) and the `remeshed_blocks` scratch word (the command buffer is now 28
bytes). `fuse_replica --incremental` and its report; the GLSL `tick_after`,
left with no caller. With no spans, a whole-map extract always binds the map's device list.

**What stays.** `share_vertices`; the slot ring (`slot_count`,
`release_through`, generations); the culled `BlockList` overload and its
density guard; the per-block range reservation in both kernels (the default
kernel's ~10% dispatch cost was taken for incremental extraction, and going
back to the per-triangle append is a measured follow-up, a `TODO(mesh)` in the
kernel); the sharing kernel's degenerate triple for a dropped triangle; the
`changed` stamp, every writer of it (`TsdfIntegrator`, `MeshIntegrator`, the
codec's inverse) and `read_block_stamps`, which the iOS scanner reads; and
`fuse_replica --dirty-every`, which measures the changed fraction from the
stamps.

**Downstream.** The iOS scanner's `incremental_benchmark` mode is a runtime
flag compiled into every scanner build: `apps/scanner/Bridge/Fusion.mm` sets
`track_block_spans` and calls `extract_device_incremental`, and `Readout.mm`
reads `ExtractTimings::incremental` and `remeshed_blocks`. The scanner must
remove the mode before it re-pins this recon.

**How it could return.** With a measurement on a large scan: a scene whose
map dwarfs the region a frame changes, which room0 replicated several times
over can stand in for. Measure the extract's time per frame against a full
extract with view culling, phase by phase, before building anything; and
design it with the ring and culling from the start rather than as a mode they
refuse.

**Validation.** Apple M5 Max, macOS, Release with warnings as errors, the
viewers, Orbbec and FFmpeg: the 55 tests pass with
`VR_TEST_HEVC_BACKEND=videotoolbox`, and `recon_mesh_marching_cubes_sparse`,
`recon_mesh_marching_cubes_config` and `recon_texture_device_mesh` run clean
under the Khronos layer's synchronization validation. Not measured: the
sparse kernels' dispatch time (each lost one barrier per workgroup).

### 2026-10-06 — Marching cubes meshes what the viewing camera sees: an extract over the device frusta list, the viewers re-meshing on a fuse or a view change, and a culled pass records its density (amends the 2026-08-12 view-meshing entry and the 2026-10-06 incremental-removal and fusion-culling entries).

*Amended 2026-10-13 (the volume block-list entry, below):* `volume::BlockList`,
`block_list` and `check_block_list`, kept here for the codec, go.

**The rule.** `MarchingCubes::extract_device(grid, iso, const
volume::DeviceBlockList&, timings)` meshes exactly the listed blocks, binding
the list in place as the whole-map path binds the map's own; only its count
reaches the host. The list must be the grid's own map's, from
`compact_active_blocks_in_frusta_on_device` or
`compact_active_blocks_on_device`, and still hold: the new
`VoxelHashMap::check_device_block_subset` refuses a list from another or a
moved map, and one that a later compaction of the same kind, a resize, an
allocation, a remove or a clear has made stale, before the extract claims a
ring slot. `check_device_block_list` still refuses a frusta list as the
active set. The frusta list gained a serial of its own for this: before, a
second frustum compaction rewrote the buffer and left the first list looking
current.

**The viewers.** `fuse_viewer` and `rig_viewer`'s render threads publish
`view_proj` under the shared mutex, with a serial that moves only when the
view does. The fuse thread meshes the blocks inside
`make_frustum_planes(view_proj, 0.25 m)` — the margin covers the frames the
view moves between the cull and the draw; the near plane takes none — and
re-meshes after fusing, at `--remesh-every`, as before, and whenever the view
moved while nothing fused: a live source between frames, `fuse_viewer`'s
replay after fusion, an idle rig or one a `--sets` run stopped, and a rig
whose capture failed with the map intact. So both fuse threads now run until
the window closes. Before the render thread has published a view, the fuse
thread meshes the whole map. A view that sees nothing meshes nothing. No
option was added. The block box the compaction tests covers the cells a block
meshes, which reach a half-voxel past its voxels to the next block's first:
with the voxels' box alone, a block whose voxels lay behind the near plane,
which takes no margin, lost its in-view boundary triangles (all 162 of a
plane at 1 cm, found in review).

**What goes.** The host-list `extract_device(grid, iso, const BlockList&)` and
the host `compact_active_blocks_in_frustum(FrustumPlanes)` and
`(DepthCameraParams)`, which only it and tests used (the 2026-10-06 public
API entry). `volume::BlockList`, `VoxelBlockGrid::block_list` and
`check_block_list` stay: the codec uses them. `make_frustum_planes(view_proj,
margin_m)` stays: the viewers call it. `ExtractTimings::input_upload_ms`
goes: nothing is uploaded on any path.

**Density.** A culled pass records `tris_per_block_` and
`verts_per_1000_tris_` as a full one does. The 2026-08-12 guard kept a culled
set, denser per block, from inflating the next full extract's plan (3.6x,
measured then) while one extractor alternated the two. The viewers now only
cull, and under the guard their plan stayed on the 64-triangle seed for
good: a dense view refit (`dispatches == 2`) on every call. A full extract
after a culled one may now plan high once, and its grow-only arena keeps
that; the viewers extract the whole map only before a view exists, which is
before any culled pass.

**Verified.** `recon_mesh_marching_cubes_sparse` splits the sphere's 216
blocks with two box frusta into disjoint halves of 108, meshes each through
the new overload, and requires the two triangle multisets to merge into the
full mesh exactly; the whole map as a frusta list reproduces the internal
compaction. It refuses a list a later frustum compaction rewrote (and the
mesh handed out before stays valid), a list from another grid, and one an
allocation made stale; an empty list meshes nothing. One dense block meshed
through a list refits from the seed, and the next culled extract, into the
ring's other unsized slot, does not. Each fails under its mutation: meshing
the whole map whatever the list, skipping the subset check, and not
recording a culled pass's density. `recon_volume_frustum` runs its cases
through the device compaction and checks the subset check across two frustum
compactions.

**Measured.** A scratch bench (not in the repo) replicated room0 along x,
12 m apart, streamed 60 frames into room 0 and meshed the view of the last
frame's camera (far 12 m), on an Apple M5 Max, Release, alone on the machine,
one run each; fusion is the integrate row's device median, meshing the wall
median of 15 extracts including the compaction. "Before" is the same bench
built at the incremental-removal commit, without either cull.

| map | blocks | fusion culled | before | meshing view | whole map |
|---|---|---|---|---|---|
| 1 room, 1 cm | 32 819 | 0.91 ms | 1.14 ms | 1.77 ms (22 530 blocks) | 2.10 ms |
| 4 rooms, 1 cm | 131 276 | 1.12 ms | 1.87 ms | 1.78 ms | 7.57 ms |
| 8 rooms, 1 cm | 262 552 | 0.90 ms | 2.58 ms | 1.78 ms | 14.86 ms |
| 16 rooms, 2 cm | 138 736 | 0.44 ms | 1.75 ms | 0.79 ms (5 964 blocks) | 7.67 ms |

Both culls hold their cost to what the cameras see while the whole-map cost
grows with the map. The integrate row's host time is noisier (one room: 2.3
culled against 1.8 ms; eight: 1.9 against 4.1). Not run: the viewers (built
only; no interactive session), and the RTX 5090.

### 2026-10-08 — The frame's models and block grammar are one GLSL text that the host compiles as C++, and golden frames pin v3 (amends the 2026-10-03 encoding entry).

The model layout was written twice, as `enum Model` and
`frame_model_alphabet` in `bitstream.cpp` and as constants with hand-summed
offsets in `rans_models.glsl`, and a block's grammar four times: the host's
`emit_block` and `read_block`, the device's walk, reversed, and the device's
`read_block`. Three of the five kernel bugs planted for the 2026-10-03
encoding entry were ordering bugs in the reversed walk. P-frames bring a
grammar of their own, and v4's lanes touch every reader and writer.

**One text.** `shaders/frame_models.glsl` holds the models, each one's
alphabet and first entry in the per-symbol arrays (`model_base`, which the
host reads as `frame_model_base`), and the mask's symbols.
`shaders/frame_grammar.glsl` holds a block's grammar: a writer half,
`emit_block`, and a reader half, `read_block`, side by side, each chosen by
a define. The kernels include them as GLSL. `bitstream.cpp` defines `uint`,
`ivec3`, `uvec2` and `findMSB`, includes the models at namespace scope, and
includes each half inside a struct whose members are the names that half
calls: the sinks or the coder, the block's mask (`g_mask`) and coefficients,
and a reader's coordinate (`g_coord`). The host's writer and reader are the
device's lines. A table that generated both sides would hold the layout but
not the grammar, which is control flow.

**The shared text keeps to what both languages mean alike**, by the rules
its header states; a signed read, for one, returns a `uvec2` of magnitude
and sign rather than taking an `out` parameter. An includer takes one half,
since it defines only that half's names.

**The walk runs in emit order.** `rans_ops.comp` writes each block's steps
down from one past the last step of its run (`step_ends`, in place of
`step_offsets`), so the steps buffer is unchanged and the walk is
`emit_block` itself.

**Golden frames.** `tests/codec_frames.hpp` pins five: 700 blocks at K = 64
and R = 64, 40 at K = 512 and R = 16, 65 at K = 21 and R = 7, and
int32-extreme coordinates at R = 64, which steps up to 2^32 - 1 on every
axis, and at R = 1, which writes each in full. Each has its content hash,
frame size and FNV-1a. The host writer must write them and read them back;
the device writer must write them, and the device reader must read them as
the host does. Every CI leg asserts the same constants, so a host or device
that codes differently on one machine fails too. A lockstep edit of the
shared grammar, swapping the line context's first two models, passed every
round trip and the host/device comparison and failed the golden frames on
both sides. The constants change only with `kFrameVersion`.

**Measured**, Release, room0 at 1 cm for 60 frames on an M5 Max shared with
other jobs, before and after in alternation: frames identical (173,606 bytes
a frame), and `..rans encode` / `..rans decode` within the machine's noise:
device 2.05–3.58 / 2.01–3.85 ms before and 1.91–3.56 / 1.95–3.77 after,
host 7.56–8.81 / 6.64–9.70 before and 7.46–8.63 / 6.43–7.65 after.

**Not done.** The coder itself is still two texts: `rans.hpp`'s
`RansWriter::put_bits` and `RansReader` against `rans_ops.comp`'s raw
chunks, `rans_count.comp`'s step count and `rans_decode.comp`'s reader. The
golden frames hold them together.

**Validation.** Apple M5 Max, Release with warnings as errors: all 57 tests
pass, and the four GPU codec tests report no messages with the Khronos layer
and synchronization validation. GCC 16 builds the changed sources with
`-Werror`, and its libstdc++ build of `recon_codec_bitstream` passes the
same golden frames; that test passes under ASan and UBSan too.

### 2026-10-08 — The fuse-to-render handoff is `mesh::MeshExchange`: one gfx-free library type owns the ring's release, a parked take caps the mark, and an empty mesh draws nothing (amends the 2026-08-08 seam-B and 2026-09-29 `rig_viewer` entries).

*Amended 2026-10-08 (the streamed-atlas entry, below):* the atlas a mesh is
drawn with is gfx's `StreamedAtlas`, and `viewer_atlas.hpp`'s `Atlas`,
`upload_atlas`, `white_atlas` and `bind_atlas` are gone.

*Amended 2026-10-19:* an empty mesh is handed to `commit` with its payload,
like any other, and still skips the bindable check; `commit` returning
`false` parks it. Committed at once, it dropped its payload, so a consumer
whose payload names what the mesh shows (the player's `FrameStamp`) stayed
on the frame before, and a stream that meshed nothing never left loading.
The viewers' payloads hold an empty atlas job for an empty mesh, which
copies nothing; `fuse_viewer`'s extract rows now describe it.
`recon_mesh_exchange` covers the payload reaching the commit, an empty mesh
replacing a drawn one, and a parked empty mesh capping the mark and holding
newer meshes back until it is committed. Under the old rule, all three and
the seeded runs fail.

*Amended 2026-10-26 (the drop entry, below):* `commit` may return a
`CommitDecision`, and `kDrop` lets the mesh go undrawn.

The protocol the 2026-08-08 entry derived was example code. `fuse_viewer`
and `rig_viewer` each held the take, commit and park state machine and the
producer's release, `viewer_common.hpp` held the mark and the bindable
check, and the iOS bridge held a third copy that had drifted: it refuses an
empty mesh, which then stays uncollected and stops meshing for the session.
Each new consumer had to re-derive obligations that were each first found
as a field bug.

**The rule.** `mesh::MeshExchange<Payload>` (`mesh/mesh_exchange.hpp`) owns
the handoff:
- Producer: `release_and_may_publish(ring)` applies the consumer's mark to
  anything with `release_through`, on the calling thread, and says whether a
  published mesh is still uncollected; `publish(mesh, payload)`; and
  `wait_collected(timeout, cancel)`, the bounded wait before a final extract.
- Consumer: `begin_frame(slot, commit)`, once a frame after the renderer's
  fence wait for that slot. It retires the slot, computes the mark, and takes
  the newest mesh unless one is parked or the exchange has latched, all under
  one lock. It then commits an empty mesh at once (until the 2026-10-19
  amendment), refuses an unbindable one (latching), and commits any other
  once `commit` (the consumer's atlas, made from the payload) accepts it,
  parking it when it does not. `live()` is what to draw.
- The payload is whatever must stay with the mesh: what its `uv0` index
  into (`fuse_viewer`'s keyframe pixels, `rig_viewer`'s atlas job), so the
  mesh and its atlas stay one value by construction, and in `fuse_viewer`
  the extract's timings, which its panel shows beside the mesh. `commit`
  runs outside the lock while the producer may be extracting the next mesh,
  so anything read from the producer then can describe a newer one.
  `unbindable_reason` checks a mesh against the usage bits and
  `cross_family` of a `MeshExchangeConfig`, which default to an indexed
  indirect draw's.

**A parked take caps the mark.** Before anything was committed, the mark
fell back to the newest generation taken, and a parked take is that
generation. A first mesh whose atlas failed was released while it waited.
An extract that failed after claiming then moved the ring's cursor toward
its slot, a final or last-set extract forced over an uncollected mesh
reclaimed it, and the retried commit drew freed buffers. The mark now stops
one below a parked generation.

**An empty mesh draws nothing.** It is taken like any other, committed
without the bindable check or (until the 2026-10-19 amendment) the payload's
commit, and released with the rest.

**Why the mesh tier, not an `interop` target.** The protocol belongs to the
ring and names no gfx type, so it sits beside `MarchingCubes` in
`recon_mesh`, header-only, and a host model of the ring tests it without a
device. The `DeviceMesh` to `LiveMesh` adapter needs gfx's type, so it stays
in `examples/viewer/recon_gfx_bridge.hpp` (`to_live_mesh`) beside the
vertex-layout asserts. An `interop` target linking gfx would make recon's
package depend on its renderer sibling, which the 2026-07-07 viewer decision
keeps behind `VR_BUILD_VIEWER`.

**What moved.** `fuse_viewer` and `rig_viewer` each hold a `MeshExchange`
and a `SharedView` (`viewer_common.hpp`: the render camera, published with a
serial, and the view-culled extract) in place of their copies.
`retire_and_release_mark` and `unbindable_reason` leave `viewer_common.hpp`.
The iOS bridge adopts the type in its I1 port. The atlas a mesh is drawn
with, an image with a pool and set of its own, is `viewer_atlas.hpp`'s
`Atlas` in `fuse_render`, `fuse_viewer` and `rig_viewer` alike, made by
`upload_atlas`, `white_atlas` or, for an image filled on the device,
`bind_atlas`.

**Verified.** Apple M5 Max, macOS, Release with warnings as errors, Orbbec
SDK 2.10.6, FFmpeg and the viewers. `recon_mesh_exchange` runs on the host
and covers: the take order; a commit running while the producer publishes,
which sees only its own payload; a generation released once the frames
drawing it retire; no refused extract with fusion outrunning the renderer;
the parked take under a failed claim and a forced extract; an empty
null-handle mesh; the latch; the bounded wait; generations at the top of
the range; and 500 seeded runs with failed claims, empty extracts, failed
atlases, skipped frames and forced extracts, in which no slot the consumer
holds is reclaimed and the ring keeps publishing once the failures stop.
Without the cap, the parked test, the top-of-range test and the seeded runs
fail, and with the commit under the lock, the commit test does. The viewers
build, and `recon_example_viewer_atlas` makes an uploaded, a white and a
bound atlas on a headless device; neither viewer was run against a window
or the rig.

### 2026-10-10 — Grid validation bounds the truncation band by the depth kernel's integer representation.

`VoxelGridParams::validate` is the configuration boundary for the coordinate
and hash helpers. Positive infinity previously passed its positivity checks:
an infinite `trunc_dist` or a finite-input division overflow then reached
`int(ceil(trunc_dist / block_extent))`, undefined on the host. An infinite
`voxel_size`, or overflow in `block_size * voxel_size`, instead collapsed the
radius to its one-block minimum.

Both metric fields and the derived block extent must now be finite. The
float ratio, evaluated in the same order as `truncation_blocks` and GLSL
`truncationBlocks`, must be at most 127 before either helper casts it to int.
That bound follows the depth allocator's existing arithmetic:
`uint items = s_centres * side * side * side`, where
`side = 2 * tb + 1` and a 16 x 16 tile has at most 256 distinct centres.
Radius 127 yields 4,244,832,000 items and leaves room for the final lane
increment; radius 128 yields 4,345,495,808, beyond `UINT32_MAX`. This is a
representation limit, not a runtime or memory-budget policy. Changing that
enumeration requires revisiting this bound.

Host tests reject infinities, both finite-input overflow paths, and the first
float above radius 127, while accepting 127 itself. A small shader runs the
real shared radius helper and the depth count expression at that boundary,
checked against host math in 64 bits; it does not enumerate the candidate band.

### 2026-10-12 — Recordings are `.vrst` files of our own: verbatim records in CRC-framed chunks, readable from a killed writer, in `recon_stream` on the core's base tier alone (amends the 2026-09-27 v1-frame entry).

*Amended 2026-10-14 (the mesh-sequence entry, below):* a Geometry track's
params are empty or a mesh source's transform (`io::encode_transform`), an
addition to the kind that keeps version 0.

*Amended 2026-10-21 (the chunks entry, below):* the chunks are also made and
read in memory, publicly (`stream/chunks.hpp`), and the writer frames
through the encoder.

*Amended 2026-10-30 (the audio entry, below):* kind 2 is Audio, interleaved
samples as its params say, in version 0 and with no new record flag.

A recording holds tracks of several kinds, each on its own clock, the rig's
calibration, and per record a pts, a sequence number, a capture set and
whether decoding can start there. It must read up to its last whole record
after the writer is killed. `stream/session.hpp` holds the layout; DESIGN.md
(`### stream`) the behaviour.

- **Our own format, payloads verbatim.** MP4 and MKV need a muxer
  dependency, a finalising step or fragments to survive a kill, and a private
  sample entry for geometry anyway. Storing each payload byte for byte keeps
  a later MP4/MKV export a repackaging with no re-encode, and the planned
  P-frames name their reference by a hash of its exact bytes, which the
  container must not change.
- **Two CRCs per chunk, zlib's CRC-32.** The header's lets a reader trust a
  body size before it allocates, and lets the scan read one small header per
  record. The body's is the integrity the codec leaves to the transport (its
  rANS end check is for consistency). zlib's polynomial lets Python's
  `zlib.crc32` check any chunk.
- **8-byte alignment.** Every payload starts aligned, so a later mapped
  reader can read 16-bit depth in place, and the search past damage tests one
  offset in 8, for at most 7 bytes a record. The search accepts only `RECD`,
  `INDX` and `TAIL` with a valid header CRC, not any printable tag, so damage
  must forge one of three tags and a CRC to stop it early.
- **A clock per track.** A device clock is comparable only within its
  frames (`sensor::ClockDomain::Device`), so one clock per session cannot
  describe a rig on device clocks: a track is on the host's `system_clock`
  (0) or a numbered domain.
- **The calibration stored as bytes; the core's base tier alone.** The
  camera tier's document is the family's one serialization of models and
  poses, and a geometry record is a codec frame, also bytes, so
  `recon_stream` links neither tier, and a sibling reads a recording with the
  core's base tier and nothing else.
- **The footer accelerates; it is not needed.** Every file can be read by
  its scan; a finished file's index is there so an open can skip the scan.
- **Kinds, extensions and the version.** A kind comes with its first
  writer: Geometry now, colour and depth with the capture recorder. An
  unknown tag is skipped by its size, a larger header's trailing fields are
  ignored, reserved fields are zero, and a flag bit a reader does not know is
  `Unsupported`, for the file in `VRST` and for that record in `RECD`. The
  version changes only for what an old reader must not ignore.
- **`O_EXCL`, not the core's `WriteMode::CreateNew`.** That writes a
  temporary file and hard-links it into place complete: exFAT, the cards and
  drives a capture writes to, has no hard links, and a recording must be
  visible and recoverable while it is written.
- **The file calls, CRC-32 and little-endian words are `recon_stream`'s
  own.** The core holds code two siblings need, and calib, gfx and ios have
  no append, offset reads, CRC or byte-order helpers, so moving them there
  would raise every recon consumer's core floor and wait on a core release.
  The cost: about 20 lines repeat the core's private errno mapping and
  flush, io's PNG writer keeps stb's CRC, and the codec keeps its own byte
  reader and writer, which work by cursor and varint. A `TODO(stream)` in
  `file.cpp` moves them to the core when a second sibling needs them. stream
  uses recon's existing core 0.2.0 floor, for `check_path`, `exception_status`
  and `UniqueFd` in the base tier; the pin does not move.
- **The 2026-09-27 v1-frame entry's CRC section is not added:** a frame
  travels in a `.vrst` record, whose CRC checks it.

**Verification.** Apple M5 Max, Release with warnings as errors, under the
Khronos validation layer: the full suite passes, 90 of 90, two of them new
and host-only. `recon_stream_container` pins golden bytes whose CRCs Python's
`zlib.crc32` computed, a 65,543-byte payload's included; the CRC of every
length to 64 at each of 8 alignments against a bitwise one; round trips of
the session, every record field and payloads of 0 to 300,000 bytes; seeking
by key; an unknown kind, chunk and header field; the payload limit and a
body size of 2^64 - 8; every refusal; and two threads on one reader.
`recon_stream_recovery` flips a bit in a payload, a record header, the
session header and each part of the footer; cuts the tail at every byte of
the last record and the footer; damages spans, one of 3 MiB; forges footers
that must not count as finished; and reads the files of a writer killed by
SIGKILL and of one `RLIMIT_FSIZE` stopped, whose failure stuck. Both also pass under ASan and UBSan (Debug) and built with
GCC 16; the library and the container test compile for `arm64_32` (32-bit
`size_t`) with `-Wshorten-64-to-32`, which nothing here can run. The
measured rates are DESIGN.md's.

### 2026-10-12 — P-frames are frame type 1 of v3: each reference block gets a mode in update segments, new blocks follow in intra segments, and a REFERENCE section names the reference by XXH64 of its bytes (experimental until golden P-frames freeze it).

*Amended the same day (the device coders, at the end):* the device writer
and reader code type 1 too; `read_frame_info` and `Decoder::decode` still
refuse it. *Amended 2026-10-13 (the P-frame decoding entry, below):* both
take it. *Amended 2026-10-13 (the coordinate entry, below):* row 13, a new
block on a reference block, is the decoder's merge kernel's, worded by
`check_new_block`; `merge_by_coord`, `merge_frame` and the one walk they
shared are gone, since the decoder never walks two lists and the encoder's
split will be a `BlockLists` pass. `check_segment(s, fault)` and
`check_survivors(survivors, n, a)` take what the device reader reads back,
the order across segments a fault of its own,
`SegmentFault::kOutOfOrder`. *Amended 2026-10-14 (the `Encoder`'s
P-frames, below):* the `Encoder` writes type 1, and its split is its diff's
own search. *Amended 2026-10-16 (the codec examples' entry, below):* golden
P-frames pin type 1's bytes, but its freeze waits: at R = 16 its P-frames
code slower than intra, and interleaved rANS lanes would change the format.
*Amended 2026-10-17 (the update-size entry, below):* no lanes: U defaults to
R, and the freeze waits on the encode's count and a re-measure. *Amended
2026-10-17 (the freeze entry, below):* frozen, so type 1's bytes change only
on purpose.

The host writer and reader of type-1 frames, internal. `bitstream.hpp` holds
the layout and the order of the refusals; this entry holds why. Nothing public
changes but one refusal: `read_frame_info`, `Decoder::decode` and the device
reader refuse type 1 as `Unsupported` through `check_frame_type(type, newest,
who)`, now after the header's other checks, so a type-1 header that also
fails one of those is `InvalidArgument`, as any frame's. The device writer
still writes intra only. The device coders take P-frames next, then the
`Decoder` and the `Encoder`; golden P-frames freeze the bytes after those,
and until then even the worked example in `recon_codec_inter_bitstream` may
change.

**Step 0 chose the design.** A throwaway probe (never committed) priced
P-frame layouts over v3 intra dumps on the M5 Max, Release, at K 64, uniform
weights and scale 0.2, each frame against the one before:

| Run | Input | Frames | Voxel | R | Mean I-frame |
|---|---|---|---|---|---|
| c1cm | room0, Classic | 0–399 | 1 cm | 64 | 242,699 B |
| c2cm | room0, Classic | 0–399 | 2 cm | 64 | 74,320 B |
| d1cm | room0, Dynamic | 0–399 | 1 cm | 64 | 231,082 B |
| dn1cm | room0, Dynamic, depth noise σ = 0.002 z | 0–399 | 1 cm | 64 | 478,157 B |
| rafa64 | Rafa2 meshes, signed TSDF, grid cleared each frame | 1–150 | 5 mm | 64 | 62,050 B |
| rafa16 | the same | 1–150 | 5 mm | 16 | 65,078 B |

Its rules, the numbers in the order c1cm / c2cm / d1cm / dn1cm / rafa64:

| Rule | Test | Numbers | Verdict |
|---|---|---|---|
| G0 | best mean P/I ≤ 0.6 | 0.050 / 0.062 / 0.153 / 0.430 / 0.426 | go |
| D1 | G-iii saves ≥ 10% of median P bytes over G-i | −34.2 / −26.7 / −13.0 / −2.5 / −5.2% | G-iii |
| D2 | skip covers ≥ 1% of kept blocks, or saves ≥ 1% | covers 92.2 / 90.0 / 77.8 / 45.5 / 9.0% | in |
| D2 | zero for a kept block, the same | covers 0.39 / 0.38 / 0.68 / 4.13 / 2.56%, saves 2.3 / 1.1 / 1.3 / 1.1 / 1.2% | in |
| D3 | a second coefficient table set saves ≥ 3% | costs 1.12 / 3.05 / 0.70 / 0.12 / 0.93% more | one set |
| D3 | reusing t − 1's tables saves ≥ 5% on Rafa2 | 1.95% (rafa16 1.71%); all tables are 3.4% of a median P-frame | no reuse |
| D3b | a mode context saves ≥ 0.1 bit a block | 0.02 / 0.02 / 0.06 / 0.21 / −0.01 | none |
| D5 | Dynamic content: exact skips < 50% of kept and max\|dq\| ≤ 1 for ≥ 80% | d1cm 77.8 / 94.6%, dn1cm 45.5 / 68.1%, rafa64 9.0 / 69.3% | the encoder's threshold skip waits |
| D6 | 128 or 256 entries an update segment save ≥ 3% | −9.0 / −13.1, −5.5 / −8.3, −2.1 / −3.2, −0.4 / −0.6, −0.8 / −1.1% | U = 64, a frame field |

This format is the probe's V7 at U = 64: mean P/I 0.058 / 0.069 / 0.158 /
0.433 / 0.431 (rafa16 0.411), and open-ended 3.07 / 1.13 / 8.42 / 48.63 /
6.41 Mbit/s at 30 fps against 58.25 / 17.84 / 55.46 / 114.76 / 14.89
all-intra. The probe priced a symbol at −log2(f / 4096) under the frame's own
tables plus its raw bits, a stream at 4 + 2⌊bits / 16⌋ bytes plus its 4-byte
length, and the tables as `write_tables` writes them. Priced intra frames came
within 0.024% of their files, and priced G-i frames, which it also wrote,
within 0.31% of their bytes; an independent Python v3 decoder reproduced its
churn counts on 30 frame pairs. Caveats:
- G-iii was priced, not written, until this change.
- The noise stand-in is independent Gaussian noise per pixel, not the rig. No
  rig recording exists, so the Femto rig is still the deciding input for D3b,
  D5 and D6.
- Rafa2 includes 30 open frames, coded signed anyway; dropping every pair that
  touches one moves no Rafa2 input by more than 0.3 points. Its frame 150 has
  a 37 mm conversion outlier.

**Why G-iii.** G-i lists every block with its coordinate, so it resends the
coordinates of blocks the reference already placed: 38 / 31 / 11 / 3 / 5% of
its bytes. G-iii codes the reference's blocks in reference order, a mode each
and no coordinate, so an update segment decodes from (s, U, M, the tables)
alone: one device invocation per segment at fixed positions, with no prefix
sum. Removing a block costs its mode symbol. New blocks keep the intra
grammar, and keep, zero and new blocks share one table set.

**Why integer residuals.** A residual is the difference of quantized
coefficients, wrapped to int16, from the q the reference decoded to, and a
mask goes by XOR. Both are lossless, so a P-frame decodes to exactly the intra
decode of the same content, on any device, with no drift. −32768 is the one
wrapped residual the 16 classes cannot carry; the encoder codes such a block
with zero, and the writer refuses it in a keep entry.

**The reference is named by XXH64 of its exact bytes.** A frame number (as in
H.264) needs a session to be unique in, and with no name at all the P-frame
after a lost frame would decode against the wrong reference. A hash of the
bytes keeps I-frame bytes identical at every intra period, cannot be fooled
by a recycled id (a P-frame's bytes contain its own reference's hash), and
catches a lost, reordered or re-serialized frame. Readers accept sections in
any order, so transports must store frames verbatim. It need not be
cryptographic: integrity stays the transport's (2026-09-27). Timed on the M5
Max (`-O3`, best of 50):

| Frame | XXH64 | FNV-1a-64 |
|---|---|---|
| 242,699 B (c1cm I-frame mean) | 0.009 ms | 0.227 ms |
| 478,157 B (dn1cm I-frame mean) | 0.018 ms | 0.446 ms |

FNV-1a-64 would cost 7–14% of a frame's encode on each side. `frame_hash` is
written from the published algorithm, with no dependency, and its known
answers were checked against the reference library (python-xxhash 4.0.1,
xxHash 0.8.3).

**A missing reference is `NotFound`, checked in `parse_frame`.** A P-frame
whose reference is not held, or that names another hash or M, is sound and
decodes once its I-frame arrives. `Unsupported` would say never,
`InvalidArgument` corrupt, and `IoError` decode again, which
`decode_growing` would retry for nothing. The check comes after every format
check, so a corrupt frame stays `InvalidArgument` and a motion frame
`Unsupported` whatever is held; outside the device decode, so `kAuto` never
retries it; and before anything is allocated, so M, a count the caller holds,
bounds the reader's work. Bounding M by `max_blocks` instead would refuse a
sound P-frame decoded into a fresh grid sized from its N, whenever N < M.

**A P-frame's coding parameters are its reference's.** Its `voxel_size`,
`trunc_dist`, K, scale and kept weights must equal its reference's bit for
bit, or it is `InvalidArgument`, after `NotFound`: a residual means nothing
across quantizers, and another K would read the reference at the wrong
stride. `parse_frame` and `assemble_frame` share the check, so both writers
make it.

**Allowed, not refused.**
- M = 0. A P-frame on an empty reference is the intra frame plus 29 bytes;
  refusing it would force an I-frame after every empty frame.
- Second spellings among the modes. Keep with zero residuals and an unchanged
  mask is skip, and zero can spell any keep; each decodes to the same block,
  as the mask's second spellings do (2026-09-27). Choosing is the encoder's.
  A new block on a reference block, whatever that block's mode, is refused
  all the same (`merge_frame`): a reference block is recoded with zero, never
  removed and added, so survivors and new blocks merge with no ties.

**Motion's rule.** Mode 4 is reserved for motion compensation. `read_tables`
refuses a mode table that gives it a frequency as `Unsupported` right after
reading it, before the trailing-bytes check. A motion design may therefore
append models, or a required section, only when that frequency is nonzero,
and today's P-frame readers then still read every frame without motion. A
static table never yields a symbol of frequency 0, so no decoder needs a
motion branch.

**One merge.** `merge_by_coord` is the codec's one walk of two sorted
lists: the Decoder's grid update moves onto it unchanged, `merge_frame` gives
a P-frame's blocks in frame order with their sources (`ApplySource`, laid out
as the device's apply will read it), and the encoder's split will use it. The
library has no host apply: both entropy paths will apply on the device,
against a reference kept there, so the tests check against an oracle of
their own (`tests/codec_frames.hpp`).

**Validation.** Apple M5 Max, macOS, Release with warnings as errors: the 74
tests pass, the GPU ones under the Khronos validation layer, and the codec
tests under synchronization validation too. Intra bytes are unchanged: the
five golden frames pass on both writers. The new
`recon_codec_inter_bitstream` covers the hash's known answers; the 161-byte
worked example, read and written byte for byte, and at U = 1 against streams
built from the coder by the format text; round trips of every mode at K 1, 64
and 512, U and R of 1, 7, 64 and past the frame, M = 0, A = 0, N = 0, all
removed, all skipped and extreme coefficients; I → P chains on real hashes
against the oracle; `merge_by_coord` against `std::set`'s algorithms; each
reader refusal, one edit each, with its code and message, and the order
across the reader's stages, REFERENCE's body before TABLES included; each
writer refusal, with its code and message; every truncation; and 3,000
random corruptions (2,973 refused). Twenty-seven planted bugs each failed a
test, among them an update range off by one, update and new segments
swapped, a payload written or read for skip, the motion check after the
trailing-bytes check, TABLES read before REFERENCE, the coding check before
`NotFound`, no A ≤ N, no M + A bound in the reader or either writer, no
survivor count, no check against reference coordinates, and a mode model's
base off by one. The host codec tests pass under ASan and UBSan (Debug), and
the GPU codec tests under them too. GCC 16 compiles the changed sources and
tests with `-O3 -Werror`, and its libstdc++ build of `recon_codec_bitstream`
and `recon_codec_inter_bitstream` passes with the same figures. The frame
code and both host tests pass a `-Werror` syntax check for a 32-bit `size_t`
(`arm64_32`, the watchOS SDK), not a build. Not run: GCC 13.3 in
`ubuntu:24.04` (no Docker daemon on the machine).

**The device coders (amended the same day).** `DeviceFrameWriter` and
`DeviceFrameReader` code P-frames byte for byte as `write_frame` and
`read_frame` do, the `Encoder` and the `Decoder` unchanged. Both work on
entries, the stream positions, which `segment_range` places on the device as
on the host:
- The rANS push constants grow from 20 to 32 bytes: `frame_type`,
  `reference_count` and `update_size`, with `num_entries` (M + A) for the
  block count.
- A mode is a u32 on the device, one a reference entry, which the kernels
  read and write whole; the host keeps bytes. The reader reads back 4 bytes
  a reference entry and 16 a new block, where an intra frame's 16 a block
  were all.
- The writer predicts the payload it reads back from the last frame of the
  same type: a P-frame's bytes per entry are a fraction of an I-frame's, so
  one prediction would read an I-frame after a P-frame in two batches, and a
  P-frame after an I-frame many times over.
- `kMaxDeviceDecodeSegmentSize` counts entries, as an update segment's cost
  does: `max(min(M, U), min(A, R))`.
- The count pass flags what `write_frame` refuses of the entries: the largest
  mode past the alphabet, and a written −32768, clamped as before so no count
  indexes past its model. `finish` refuses them, and a counted motion mode,
  through `check_entries`, the host writer's own last check (after the arrays'
  sizes and the coordinates' order). It makes the coding parameters' refusals
  (`check_coding`) before them, as the host writer does, so both writers refuse
  with one code and message.

Validated on the M5 Max, Release with warnings as errors: the 94 tests pass
under the validation layer, and again under synchronization validation. In
`recon_codec_device_frame` the device writer's bytes equal `write_frame`'s,
and the reader's modes, new coordinates and keep, zero and new payloads
`decode_frame`'s, on the worked example (at U = 1 too), the 336 frames of the
round-trip matrix and a P-frame of no entries. The reader refuses each of the
43 edits of the refusal list (6 judged after a device decode) and 800 random
payload corruptions of two P-frames (13 decoded, 11 out of int32, 762 corrupt,
14 out of order) with the host's code and message. The writer refuses motion,
modes past the alphabet and a −32768 in a keep, zero, new and intra entry as
`write_frame` does, and a frame that also has a bad voxel size or `trunc_dist`
for that. An update or new segment of 1,025 entries is refused while the host
decodes it, and one of 1,024 decodes, as does a frame of 1,026 blocks in short
segments. Thirteen planted bugs each failed a named case: update and new
segments swapped, an update range off by one, the tables of slots without the mode
model, a skip that reads a payload, a new segment's start counted in stream
positions, no entry check in `finish`, an unflagged bad mode, an unflagged
−32768, no survivor check, the limit counted in blocks (whole, or in its
new-block term), `finish` trusting another reference count than was counted,
and the entries judged before the coding parameters. The codec tests pass
under ASan and UBSan (Debug). Not run: the RTX 5090 (not reachable from this
machine).

### 2026-10-13 — The `Decoder` decodes P-frames against the frame it decoded last, kept on the device as decoded in two slots that swap only when a decode succeeds; one integer apply makes every decoded frame, intra too; and it decodes on the device alone (amends the 2026-09-27 `Encoder`/`Decoder` and 2026-10-03 decoding entries).

*Amended 2026-10-21 (the deblock entry, below):* with
`Decoder::set_deblock` on, the decode's last submit also smooths a frame's
fully observed blocks after the inverse, and `Decoder::rebuild` remakes the
last frame from the reference without reading it again.

*Amended 2026-10-13 (the coordinate entry, below):* the reference's
coordinates live on the device, in each slot's block list, and `merge_frame`
is gone: the decoder's merge kernel writes each block's sources into the
spare in the decode's submit, and the apply reads them there. The apply
counts a source past what is bound, which no frame that passes rows 11 to
13 makes, and the store refuses the frame after the submit. *Amended
2026-10-14 (the `Encoder`'s P-frames, below):* the `Encoder` writes type 1
too, from a store of its own: its forward fills the spare, and `encode`
commits it.

The `Decoder` takes type 1; the `Encoder` still writes intra frames only,
and type 1 stays experimental until golden P-frames freeze it. The format,
the reference hash and why a missing reference is `NotFound` are the
2026-10-12 entry's. Public: `FrameType` moves into `codec_params.hpp`, where
the internal frame code takes it from, `FrameInfo` gains `type`, and
`read_frame_info` reads a P-frame's header, its `block_count` the decoded
frame's.

**The reference is what the decoder decoded, kept on the device.** A residual
adds to the quantized coefficient its reference decoded to, which the grid,
holding dequantized and clamped SDF, no longer has. So `detail::ReferenceStore`
keeps the decoded coefficients and masks in the inverse transform's layout,
and the frame's `FrameReference` and sorted coordinates on the host, which
`parse_frame` and `merge_frame` take. The reference is the decoder's, not a
grid's: a P-frame decodes into any grid of its geometry, a fresh one built
from `read_frame_info` included, and a moved decoder takes it along.

**One apply makes every frame.** `merge_frame` gives each of a frame's N
blocks an `ApplySource`, and `inter_apply.comp`, one workgroup a block,
writes the block into the store's spare slot: keep adds each residual to the
reference mod 2^16, a 16-bit half at a time, and XORs the mask; skip copies
the reference; zero and a new block copy the payload. An absent side reads as
zero, so one integer formula serves all three, and an odd K's pad half is
written 0. An intra frame's sources are `{none, i}`, so it lands in the store
by the same path. The apply reads the payload where the device reader
decoded it, so no entry crosses the bus: decoding runs on the device alone
(the 2026-10-03 decoding entry's amendment says why). The list and sources'
upload, the apply and the inverse share one submit
(`DctTransform::inverse` takes the caller's batch).

**The reference changes only when a decode succeeds.** The store's two slots
swap at `commit`, after the inverse's submit and reject check pass, with
`{frame_hash(bytes), header}` and the coordinates. A refusal, which comes
before the grid changes, or a failure after it leaves the reference as it
was: a resize, or decoding again, decodes the same frame against the same
reference, and a P-frame on a frame that failed is `NotFound`. `reserve`
grows the spare, the list and the sources before the grid changes; each is
at most the grid's `tsdf` array once the frame fits the heap, as the
inverse's buffers are.

**Memory and cost.** The two slots, the list and the sources take about 10 MB
at 25,000 blocks of K = 64 (`decoder.hpp` gives the bytes a block), all
retained as the transform's scratch is.
Every decode now copies its frame through the apply and hashes it. Hand check
on the M5 Max, Release: room0, 400 frames at 1 cm, Classic, every frame coded
as an I-frame and decoded on the device, the median of six runs before and
six after, interleaved: 3.95 → 4.09 ms a frame of host time, 1.91 →
1.94 ms of device time; `..rans decode` +0.05 ms (the coordinates taken off
the reader), `..apply` +0.03 ms (`merge_frame`, which takes an intra frame's
blocks as they are), `..inverse` +0.06 ms (the list and sources staged, and
the apply). Bytes and F-score are unchanged.

**Validation.** Apple M5 Max, Release with warnings as errors: the 96 tests
pass under the validation layer, and again under synchronization validation.
The new `recon_codec_inter` covers:
- the apply against plain uint16 arithmetic, every source kind at K 1, 21, 64
  and 512, from device entries whose pad is not 0, in one dispatch and in
  several (a test's cap on blocks a dispatch); the slots growing under a held
  reference; `commit` alone changing what is held; the apply's refusals;
- frames leaving the grid bit for bit as the CPU oracle decodes their content
  (the host reader's entries through the inverse alone): the `Encoder`'s at
  K = 37 in segments of 64 and 3; chains of spheres from the `Encoder` that
  move and grow; synthetic chains with every mode and new blocks, then
  A = 0, N = 0 and M = 0, at K 21, 64 and 512; and coefficients that wrap or
  differ by 32768;
- a P-frame into a fresh grid built from `read_frame_info`;
- `NotFound` with no reference, with a later I-frame held, and on a second
  decoder; `InvalidArgument` for a scale not the reference's, a new block on a
  reference block, and a segment, intra or update, one entry past
  `kMaxDeviceDecodeSegmentSize`, where segments of the limit decode;
  `OutOfMemory` at the parse and at the hash table, each recovered by a
  resize, with a P-frame on the failed frame `NotFound`; each refusal leaving
  the grid and the reference as they were;
- a decoder moved mid-chain.

The decoder tests run their cases once, on the device; their two interim
type-1 refusals from the 2026-10-12 change check type 2. Nine planted bugs
each failed one of the two: a carry between the halves, an unmasked pad, OR
for XOR, no slot swap at commit, the wrong bytes hashed, a commit after a
failed allocation, the segment limit off by one or doubled, and the reader's
verdict ignored. Not run: the RTX 5090.

### 2026-10-13 — The codec's coordinate work runs on the device: the encoder sorts its blocks with a stable rank-merge sort on `coord_less`, and the decoder judges, merges and diffs them by count and binary search, never sorting; only counts and verdicts cross the bus (amends the 2026-10-13 decoder, 2026-10-12 P-frame, 2026-10-03 observed-list and garbage-collection, and 2026-10-01 codec-review entries).

*Amended 2026-10-22 (the remove's first round):* `VoxelHashMap::remove`
takes the public first-round hook **What crosses, and the fences** names,
`first_round`, and `VoxelBlockGrid::remove` records its zeroing there,
ahead of the delete in the same submit. A decode that drops blocks, and a
garbage collection that frees any, waits on one fence fewer: 4 for a
decode with adds and drops.

*Amended 2026-10-14 (the `Encoder`'s P-frames, below):* the split the
**Next** item plans as a `BlockLists` pass is the encoder's diff's own first
search (`inter_diff.comp`), on these searches and the reader's survivor
ranks.

*Amended 2026-10-13 (the device step layout):* the **Next** step layout is
done, so the codec's host does I/O and control alone. `detail::StepLayout`
(`rans_layout.comp`, one workgroup on `scan.glsl`) records into the
forward's submit after the count: an exclusive scan E of the entries' steps
in stream order, from which entry p of segment `[x, y)` ends its steps at
`E[x] + E[y] - E[p]` and the segment takes `E[x]` to `E[y]`, then a scan of
the segments' slot sizes. `rans_ops.comp` and `rans_encode.comp` read E, so
the per-entry step ends and per-segment step offsets go, and only the two
totals come back to size the coding's buffers. The step total stops at
2^30, the first whose steps reach 4 GiB, so the writer refuses such a frame
with its code and words as before; the slot total stays exact below it. An
encode reads back 4N - 8 bytes fewer and uploads 4N + 8(S + 1) fewer (N
entries, S segments), exactly as the scratch counting core measured on two
sphere frames: 176,896 bytes a frame at room0's 21,771 blocks, 19,464 at
Rafa2's 2,395. It waits on no more fences (3 an encode with the map's list
held). `frame_segment_range`, whose one caller was the host loop, goes. Hand
check, M5 Max, Release, the median of six runs before and six after,
interleaved: room0 at 1 cm, 400 frames of 21,771 blocks, encode 2.96 →
2.97 ms of host time and 1.98 → 2.03 ms of device time (`..forward`, which
now holds the layout's dispatch, +0.03 ms; `..rans encode` 1.88 ms both);
Rafa2 at 5 mm, over the sweep's 90 K = 64 rows, 2.92 → 3.01 ms and 2.11 →
2.16 ms. On unified memory the bytes saved cost no copy, so the dispatch is
what shows; a discrete GPU was not measured. The `.vrst` files are the same
byte for byte. `recon_codec_step_layout` holds every start, slot offset and
total to the oracle's layout of the oracle's step counts
(`codec_oracle::step_layout`, `entry_steps`): intra frames of 0 to 65,537
blocks at K of 1, 64 and 512 in segments of 1 to 1,024, int32's extremes,
the worked example and C2's matrix at every update and segment size, any
steps up to the bound, and the cap, also past a u32. Fifteen planted bugs
each failed a test: an inclusive scan, a lane's run short of its last value,
no cap, slots a word short or long, either total missing, the slot scan not
reset or not after the barrier, the wrong A, the ops' or the encoder's run,
`segment_of` without the update segments, slots sized by steps, and the step
total read one entry short.

*Amended 2026-10-13 (the volume block-list entry, below):* the host
`BlockList`, `block_list` and `check_block_list` that **Next** names go, and
so do the host-array `allocate` and `remove` the device overloads here sat
beside.

The rule the 2026-10-03 decoding entry's amendment records -- only I/O and
control cross the bus -- applied to the codec's work on block coordinates.
Before, the encoder read its observed list back, sorted it and uploaded it
again; the decoder read back the new blocks' coordinates, a P-frame's modes
and each segment's fault, judged rows 11 and 12 and merged the frame (row
13) on the host, then compacted the grid, read it back, sorted it, merged it
with the frame into the remove and allocate lists, and uploaded those and
the apply's list and sources. On room0 at 1 cm (21,771 blocks a frame) the
host sort was 0.99 ms of a 3.89 ms encode, and the decoder's compaction
0.21 ms and its active-set sort most of a 1.33 ms `..apply` (the hand check
below); a frame moved about 0.8 MB over the bus encoding and 1.3 MB
decoding.

**The encoder sorts; the decoder never does.** The observed filter appends
in hash-table order, so the frame's (z, y, x) order takes a sort. Three
candidates for (z, y, x) int32 keys:

| | Full-range LSD radix | Packed 64-bit keys | Stable rank-merge sort (chosen) |
|---|---|---|---|
| Key | 96 bits, sign bits flipped: 12 digits of 8 bits | each axis less its minimum, at most 21 bits | `coord_less` itself |
| `extreme_frame`'s int32 extremes | 12 passes | do not fit: a second, full-range path | free |
| Dispatches | 3 a pass: 36, or 9 at room0 after a min/max readback, one more fence | radix's, a pack kernel and the fallback | a tile pass and one a merge level (`block_lists.hpp`): 8 at room0, 5 at Rafa2's 2,395 blocks |
| New code | a histogram, a ranking and a 256-counter scan a pass | radix's, and `shaderInt64` or its emulation | two small kernels on `coord_order.glsl` |
| Work | O(N) a pass | O(N) a pass | O(N log² N): about 2.5 M comparisons at room0, 200 M at a million blocks |

Placing each entry by a search of its sibling run (`block_lists.hpp` has the
mechanism) keeps ties in order, makes the output a permutation, and fixes the
pass count by N alone, which the host holds; a tile's two arrays of 256
entries take 8 KB of the 16 KB of shared memory Vulkan guarantees. A hash
table's keys are unique, so any correct sort gives the host's order: every
frame's bytes are unchanged. If frames reach about a million blocks, a
merge-path partition on the same searches is the upgrade. The sort reads its
input and writes two retained buffers of its own, so the transform's observed
list is never rewritten; it records into the forward's batch, and the `..sort`
row goes. `VR_COORD_SEARCH` (`coord_order.glsl`) generates the searches for
each array they run over, since GLSL passes no buffer to a function. The
decoder's frame is a merge of two sorted lists, and its grid diff searches
that frame, so it sorts nothing.

**Rows 11 and 12 stay the reader's, on the device.** After `rans_decode`,
`rans_verdict.comp` takes, an invocation a segment, its fault, else -- for a
block segment with a block segment before it -- whether its first coordinate
follows that segment's last (`SegmentFault::kOutOfOrder`, which the decode
never writes), and keeps the smallest failing `s << 2 | kind` by `atomicMin`;
segments number below 2^30, so it fits a u32. The smallest is the segment
the CPU oracle's walk stops at: every one before it decoded clean, so the
coordinates it compares are real, and a later segment's comparison against
a corrupt one never wins. `survivor_ranks.comp`, one workgroup on
`codec/shaders/scan.glsl` (the scan `rans_scan.comp` ran, now shared, its
outputs unchanged), writes the survivors before each reference entry, and
their count at M. The reader reads back those two words and `check()` keeps
its contract: `check_segment(s, fault)` for the first failing segment, then
a P-frame's `check_survivors(survivors, n, a)`, codes and words as before.
The tests' CPU reader computes `kOutOfOrder` itself. The verdict is reset in
every decode's batch, a frame of no segments included.

**The merge is a count.** Once row 11 passes, the new blocks strictly
increase -- the grammar's deltas are strictly positive within a segment, and
the verdict orders segments -- as the committed reference's do, and row 13
refuses a tie between the two, so a block's place is the survivors below it
plus the new blocks below it, which `frame_merge.comp` counts from the ranks
and a binary search (its header has the formulas). Row 13 takes `atomicMin`
of q and of e: the colliding pairs are monotone, so the two minimums are the
pair a walk in frame order meets first, which `check_new_block` words as
`merge_frame` did. On a corrupt frame, refused before anything reads the
outputs, writes stay below N and every search ends.

**The grid diff: the hash table against the frame** (`grid_diff.comp`), the
table quiescent as it is for the inverse and fetched each call, since a resize
replaces it. Neither list is ordered, since `remove` and `allocate` draw heap
slots atomically anyway, and its two passes are independent, with no flag
buffer between them; the slot pass shares `vrEntryActive` with the
compactions. That replaces the decoder's compaction, a fence and a read back,
and its active-set sort. Gone is at most `num_blocks`, since an occupied slot
holds one block and `validate` makes slots blocks: 16 bytes a slot, 1/128 of
the `tsdf` array. Missing is at most N. The grid removes and allocates
straight from them, through `VoxelBlockGrid::remove` and
`VoxelHashMap::allocate`'s device-list overloads (the 2026-10-03
garbage-collection entry's amendment), with `kRounds` and the failure handling
as before. The encoder keeps its compaction: its observed filter runs a
workgroup per listed block, not per slot, and often reuses an extract's list.

**The store keeps the reference on the device.** Each slot holds its block
list, and `commit` takes no coordinates. The merge writes the spare's list
and sources in the decode's submit, so `record_apply` takes neither. It
checks on the host what it still can -- the room reserved, the payload's
buffers, a P-frame's M and K against the held reference's -- and
`inter_apply.comp` reads a source past what is bound (a reference entry past
the frame's M, a payload entry past its count) as absent and counts it,
which the store refuses after the submit. No frame that passes rows 11 to 13
makes one; this replaces host checks that cannot see sources the device
made.

**Every size before anything is recorded.** `reserve` -- the store's spare,
list and sources, and the diff's lists -- comes before the decode's batch.
Its range checks cannot refuse a frame the parse passed: each buffer is at
most the grid's 4-byte `tsdf` array, and the ranks, 4 bytes a reference
entry, fit under the current slot's 64 bytes of masks an entry. So only a
device out of memory comes before rows 11 to 13, which is no refusal of the
frame, and a refused frame touches neither the grid nor the reference: the
merge writes only the spare.

**Gone from the library:** `sort_by_coord`, `kAbsent`, `merge_by_coord`,
`MergedFrame` and `merge_frame`. Nothing calls them, and the 2026-10-06 rule
keeps no function only tests call; the tests hold the device merge to the
oracle's own map (`codec_frames::merge_oracle`), row 13 to a walk, and the
diff to `std::set_difference`.

**What crosses, and the fences.** The encoder reads back 8 bytes before its
forward (the observed count and reject tally), and the decoder 24 in its
decode's submit (the verdict and survivor count, row 13's first collision,
the gone and missing counts). Fence waits a call, counted against a scratch
core that counts `vkWaitForFences`, sphere frames of about 400 blocks on the
M5 Max:

| Call | Before | After |
|---|---|---|
| encode, compacting | 4 | 4 |
| encode, the map's list held | 3 | 3 |
| decode into a fresh grid (adds only) | 4 | 3 |
| decode, the same block set | 4 | 2 |
| decode, adds and drops | 6 | 5 |

The encoder also loses the tail submit an outgrown prefix took, and the
decoder its compaction and that read back's tail. Fewer would need volume
changes this does not make: a one-submit decode needs record-into-batch
variants with device counts, and would have a refused decode move
`topology_epoch` and `tick`; folding the zeroing into the remove's first
round needs a public first-round hook, as recon declares no `friend`.

**Hand check.** M5 Max, Release, the median of six runs before (#237's
head) and six after, interleaved:
- room0 at 1 cm, Classic, 400 frames each encoded and decoded, 21,771 blocks
  a frame: encode 3.89 → 2.96 ms a frame of host time (`..sort`'s 0.99 ms
  gone, `..forward` +0.09 ms), its device time 1.96 → 2.00 ms; decode 4.10 →
  2.86 ms (`..active set`'s 0.21 ms gone, `..apply` 1.33 → 0.30 ms,
  `..rans decode` +0.03 ms), its device time 1.94 → 1.96 ms. The `.vrst` is
  the same byte for byte, and the F-score 0.9980.
- Rafa2's frame 1 at 5 mm, 2,395 blocks, over the sweep's K = 64 rows:
  encode 3.01 → 3.15 ms (+0.14 ms: the host's sort took 0.10 ms, and the
  device's adds five dispatches to the forward's submit), decode 2.88 →
  2.71 ms. The frame's bytes are the same, and the F-score 1.0000.

**Validation.** Apple M5 Max, Release with warnings as errors: the 97 tests
pass under the validation layer, and the codec and volume tests under
synchronization validation too. The new `recon_codec_block_lists` covers the
sort against `std::stable_sort` (0 to 200,000 entries: all of int32,
clusters, every kind of tie, sorted and reversed); the merge against the
oracle's map (every mode mix, M = 0, A = 0, N = 0, int32's extremes, the
worked example, 1,000 random frames and one of 70,000 blocks); row 13
against a walk; the diff against `std::set_difference` (overflow chains,
empty sides, a resize); and garbage writing nothing past an output.
`recon_codec_device_frame` adds the verdict's first failure, a segment
starting on the last coordinate of the one before, and the verdict's reset,
and the ranks at M of 0 to 65,537; `recon_codec_inter` the apply's faults,
a chain whose new blocks sit at int32's extremes, buffers grown and shrunk,
and a P-frame that fails row 11 or 12 and also collides, refused by the
first; `recon_volume_hash_map` and `recon_volume_block_grid` the
device-list overloads. Twelve planted bugs each failed a test: ties to the
wrong side, the last partial run skipped, inclusive ranks, `atomicMax` for
the collision, the verdict not reset, the order checked against the next
segment, an equal coordinate accepted across segments, an unsigned
`vrCoordLess`, the table's first pass over `num_buckets` alone, the lookup
with `max_chain` 0, and the decoder judging row 13 before rows 11 and 12 or
ignoring them. The codec and
volume tests pass under ASan and UBSan too (Debug, Apple clang). Not run:
the RTX 5090 (not reachable from this machine), and GCC, which CI covers.

**Next.** The writer's per-entry coder steps still cross the bus, 4 bytes an
entry each way, laid out in an O(N) host loop (`device_frame_writer.cpp`'s
`TODO(codec)`); a segmented layout on `scan.glsl` is the next PR (done, the
amendment above). The encoder's split (C5a) becomes a `BlockLists` pass in
the forward's submit, on these searches and ranks. Cutting fences further
waits for a measurement.
`volume::BlockList`, `VoxelBlockGrid::block_list` and `check_block_list`
now have no library caller, only tests; removing them is a breaking volume
change, its own PR (done, the volume block-list entry below).

### 2026-10-13 — Block lists reach the volume tier on the device alone; a host-list entry point that only tests called goes (amends the 2026-08-08 sparse-mesh and overflow-scan, 2026-08-12 view-meshing, 2026-09-26 codec, 2026-09-28 device-memory, 2026-10-03 garbage-collection, 2026-10-06 public-API and view-culling, and 2026-10-13 coordinate entries).

*Amended 2026-10-17 (the table's internals):* by the user's decision, the
host mirrors this entry kept public leave the public headers, under the
2026-10-06 rule. `HashEntry` and the fail-tally slots are internal
(`src/volumetric_kit/recon/volume/hash.hpp`). The host slot sentinels,
which nothing read, go, leaving `hash_common.glsl`'s. `hash_bucket` and its
primes become the tests' oracle (`tests/volume_hash_oracle.hpp`).
`BlockIndex` and `BlockStamp` stay public. Other tiers still bind
`entries_buffer()` and read it through `hash_common.glsl` or
`hash_lookup.glsl`. `recon_package_headers` fails if `volume/hash.hpp` is
installed again, or if an installed volume header, or an installed header
that includes one, declares any of these names in the volume namespace.
*Amended 2026-10-22 (the install entry):* `recon_subproject_headers` makes
that check on the public headers, as a fetching sibling sees them.
*Amended 2026-10-28 (the grid-contract entry):* `entries_buffer()` and
`entries_buffer_size()` leave the public header for
`VoxelHashMap::Internal`; only codec, mesh and voxelize bind the entries,
and the check fails if either is public again.

**The rule** is the user's (2026-10-13): only I/O takes a host path --
reading a mesh model, a camera stream (Orbbec, ARKit) -- and what it reads
goes straight to the GPU; gfx shows processed results without a copy, and
they return to the host only to be saved. Control may read back counts and
verdicts. The volume tier's block lists are processing, so they live on the
device, and no library code stays that only tests call. Here that
withdraws the 2026-10-06 public-API entry's `diagnostics()` exception; its
others lie outside the volume tier (**Next** 5).

**What goes.** `volume::BlockList`, `VoxelBlockGrid::block_list` (both
overloads) and `VoxelBlockGrid::check_block_list`. Since the coordinate
entry nothing in `src/`, `include/` or `examples/` called them, nor in
ios, gfx or calib (`git grep` on each `origin/main`): the mesher binds a
`DeviceBlockList` since 2026-10-06 and the codec its own device lists. A
`DeviceBlockList` is the one block list, stamped with the epoch, a
compaction serial and the heap's free count, which
`check_device_block_list` and `check_device_block_subset` check. The codec
oracle's `forward` and `inverse` took a `BlockList` only to carry a pointer
and a count; they take a `std::vector<BlockIndex>`, a test's own host list,
and upload it as the device list the transform binds.
`recon_codec_dct` keeps every list refusal, now on the device list: none
with a count, one shorter than the count, one not a storage buffer, more
blocks than the heap, and a list compacted before a remove.

**The host-array `allocate` and `remove` go too:**
`VoxelHashMap::allocate(const BlockIndex*, …)`,
`VoxelHashMap::remove(const BlockIndex*, …)` and
`VoxelBlockGrid::remove(const BlockIndex*, …)`. Since the coordinate entry
the decoder removes and allocates from its device lists and garbage
collection frees from its own, so no library code staged coordinates from
the host; the only caller outside recon's tests is the iOS compute smoke
test's block stage (`apps/compute_smoke/smoke_report.cpp:420`), which must
upload its eight blocks when it re-pins. The device overloads keep every
check, round, tally and failure split; the grid's still zeroes, then frees,
in two submits. A test hands its own coordinates over as a producer kernel
would, through `tests/device_blocks.hpp` (upload, then the device
overload), at about 90 call sites. The cases that compared the two
overloads go with the host one: the hash-map contention fixture runs four
fresh maps on the device list, the capacity split it repeated is the main
case's, and the delete test's host/device alternation and host zero-count
call are gone.

**And the host table readbacks:** `VoxelHashMap::read_entries`,
`VoxelHashMap::diagnostics` and `HashDiagnostics`. The 2026-10-06
public-API entry kept `diagnostics()` as the tests' one window on the
device heap counter; the rule now has no such exception, and the tests
check what they read by behaviour:
- **The heap counter.** Every allocate and remove round reads it back into
  the copy `load_factor()` returns, so after one the copy is the device's.
  `recon_volume_block_grid` compares it with the compacted block count after
  every call that moves the counter, and `recon_volume_resize` after the
  allocations that follow a grow and a rolled-back grow, and checks that no
  index is handed out twice. Two planted bugs each failed both tests: a
  rebuilt device counter one short of the host copy, and a heap rebuilt
  without excluding the live blocks.
- **Chains.** `recon_volume_hash_map` already found chained blocks again
  only by walking their chains, and drops the anchor-offset reads. Those
  reads also showed that coordinates the host's `hash_bucket` puts in one
  bucket share it on the device, so a probe asks the device instead
  (`chain_refusals`, `tests/device_blocks.hpp`): a table of the fixture's
  shape whose chains may not grow (`max_chain` 1) refuses each coordinate
  past the bucket's slots. The 7- and 300-coordinate fixtures expect 3 and
  292 refusals; a GLSL Z prime drifted from the host's fails the second,
  and failed no test without the probe. `recon_mesh_marching_cubes_sparse`
  probes its whole block set, so its chained fixture spills on the device;
  8-entry buckets fail it.
- `recon_volume_diagnostics` goes with the API.

The host `HashEntry`, `hash.hpp`'s slot sentinels and `hash_bucket` with
its primes stay, as the table's layout and hash that the GLSL mirrors;
`entries_buffer_size()` sizes the table by `HashEntry`, and only tests call
`hash_bucket`. No host code reads a slot. Whether these mirrors stay public
or move into the tests awaits the user's decision.

**What stays, and why.** The volume tier's other host entry points, with
their callers in `src/`, `include/` and `examples/` (and ios's, at its
older pin):

| Entry point | Host side | Callers | Verdict |
|---|---|---|---|
| `allocate_from_depth(const float*, camera, …)` | the image staged | none in recon; ios's ARKit fusion (`Fusion.mm:731`, `:849`) and compute smoke test | camera-stream I/O: stays |
| `allocate_from_depth(const std::vector<DepthInput>&, …)` | a host frame's depth staged | `tsdf/fuser.cpp:87` | camera-stream I/O: stays |
| `allocate_from_triangles(vertices, …, indices, …)` | mesh staged; candidates counted on the host | `examples/codec_mesh/main.cpp:115` | mesh-model I/O stays; the count is **Next** 3 |
| `triangle_candidate_offsets` | the candidate count | `tsdf/mesh_integrator.cpp:284`, `allocate_from_triangles` | **Next** 3 |
| `compact_active_blocks()` | the active set read back | `examples/common/remesh_report.hpp:97`, `VoxelHashMap::resize`; ios's `Fusion.mm:1290` | **Next** 1, 2 |
| `read_block_stamps()` | every slot's stamps read back | `examples/common/remesh_report.hpp:50`; ios's `Fusion.mm:102` | **Next** 1 |
| `load_factor()` | none: the copy of a counter the rounds read back | `tsdf/fuser.cpp:53`, `:106`; the viewers | control: stays |

**Next**, each its own PR, since each changes another tier, an example or
a sibling:
1. The remesh report (`examples/common/remesh_report.hpp`) counts changed
   blocks on the device, leaving `compact_active_blocks()` to `resize` and
   the iOS scanner, and `read_block_stamps()` to the scanner alone.
2. `VoxelHashMap::resize` rehashes from the device list and rebuilds the
   heap on the device (its `TODO(volume)`), with no snapshot read back and
   no host pass over `num_blocks`.
3. `allocate_from_triangles` and tsdf's `MeshIntegrator` count triangle
   candidates on the device; the vertices and indices stay the mesh model's
   I/O.
4. ios, when it re-pins: the compute smoke test uploads its eight blocks,
   and the scanner's survey and stamps follow item 1. The smoke test also
   rewords its two comments that name `HashEntry` (the amendment above).
5. The 2026-10-06 public-API entry's other exceptions, outside the volume
   tier, are audited under this rule: the host `TextureView` `texture`
   overloads, `FrequencyTable::empty`, and the camera tier's
   `distort_rational` and Rodrigues helpers, which calib now calls
   (`examples/01_project_point/main.cpp:44`, `cv_pose.cpp:40`).

**Verified** on an Apple M5 Max, macOS, Release with warnings as errors,
Orbbec, FFmpeg and Assimp: all 96 tests pass under
`VKC_TEST_VALIDATION=1`, and the codec, volume, tsdf, mesh and texture
tests under `VKC_TEST_SYNC_VALIDATION=1` too. Not run: the RTX 5090,
ASan/UBSan, and GCC, which CI covers.

### 2026-10-14 — The `Encoder` writes zero-motion P-frames against the frame it returned last, kept in the `ReferenceStore` as its forward made it; five rules make an I-frame, and the split, the diff and a static price for zero run on the device (amends the 2026-10-13 decoder and coordinate, and 2026-10-12 P-frame entries).

Library only: the examples keep period 1 until they read
`codec.intra_period`, and type 1 stays experimental until golden P-frames
freeze it. The format, the reference hash and `NotFound` are the 2026-10-12
entry's; the store and the apply the 2026-10-13 decoder entry's.

**Public.** `EncoderConfig::intra_period` (u32, default 1, so every frame's
bytes are as before; `create` refuses 0) and `Encoder::force_intra()`.

**A frame is an I-frame when:**
- the encoder holds no reference (its first frame);
- the period's count is 0: it counts the frames returned since the last
  I-frame, so any I-frame restarts it;
- `force_intra()` was called since the last frame returned;
- the grid's `voxel_size` or `trunc_dist` is not the reference's, bit for
  bit: `coding_difference`, the rule the writers and the reader already
  shared, now declared in `bitstream.hpp` (the encoder's params never
  change);
- the reference holds no block, where the I-frame is 29 bytes smaller;
- *amended 2026-10-16:* the grid holds no observed block (the 2026-10-16
  entry).

Otherwise it is a P-frame. Only a frame `encode` returns moves the
reference and the count; a refusal keeps both, and a pending
`force_intra()`. A frame returned and then lost breaks the chain at the
decoder (`NotFound`) until the next I-frame, which `force_intra()` brings
forward.

**The reference is what the decoder will hold.** The forward writes into
the store's spare and the sorted list is copied beside it, so either frame
type leaves there the q and masks the decoder's apply makes of the frame: a
skip only when every word is equal, keep's residuals mod 2^16, zero and new
blocks as they are. `record_forward` takes its output buffers, and the
transform's own coefficient and mask scratch go, since only tests would
have used them. `encode` commits `{frame_hash(bytes), header}` once it has
the bytes, after checking that the frame holds every block the forward made
(a bug otherwise, which would leave the store's list short of M). One store
class serves both sides: the encoder's never applies, the decoder's never
diffs.

**The split and the diff run in the forward's submit**
(`ReferenceStore::record_diff`, `inter_diff.comp`, whose header has the
rules):
1. Pass 0, an invocation a reference entry, finds its block in the frame's
   sorted list (`VR_COORD_SEARCH`) and writes its mode and payload.
2. `survivor_ranks.comp`, the reader's scan, counts the survivors below
   each entry.
3. Pass 1, an invocation a frame block, finds it in the reference; one the
   reference lacks is new block `q = t - ranks[e]`, its coordinate and
   payload at `M + q`.

The coordinate entry planned the split as a `BlockLists` pass. It is the
diff's own first search instead: both need the store's two slots, and a
split of its own would write a target an entry only for the diff to read.
8 bytes come back, the survivor count, from which the host takes A for the
count, so a P-frame counts its symbols in a second submit: 4 fence waits
an encode with the map's list held, against an I-frame's 3. *Amended
2026-10-17:* counted in the forward's submit, 3 (the 2026-10-17 count
entry). U is `kDefaultUpdateSize`, 64. *Amended 2026-10-16:* U is
`EncoderConfig::update_size`, default 64. *Amended 2026-10-17:* default the
segment size.

**Zero over keep.** A residual of −32768, which no class carries, codes
zero. Otherwise the kernel prices both payloads by walking `emit_payload`
with a sink that charges a symbol `bit_length(s) + 1` bits, a byte model's
symbol 8, and a raw field its bits, and codes zero only when strictly
cheaper: a static stand-in for the frame's tables, which do not exist
until the count. It changes the encoder's bytes, never what a frame decodes
to. Step 0 put greedy zero-for-kept at 1–2% of P bytes (its V7).

**Memory.** The store's two slots take `80 + 4 ceil(K/2)` bytes a block
each and the sources 8, as in the decoder; a P-frame's entries
`80 + 4 ceil(K/2)` a frame block and `72 + 4 ceil(K/2)` a reference block:
about 18 MB at room0's 21,771 blocks of K = 64, all retained, and up to
1.5x that once grown (`ensure_device_scratch`'s headroom).

**Hand check.** M5 Max, Release, room0's 400 frames at 1 cm: each frame
coded at period 30 (14 I-frames, 386 P-frames) and, by a second encoder, at
period 1, each chain decoded by a fresh decoder of its own, and the two
players compared after every frame (an uncommitted edit of
`codec_replica`). Means per frame; step 0's predictions are its V7 u64.

| Fusion | I bytes | P bytes | P/I (step 0) | Mbit/s (all-intra) | Encode I / P, host ms | Device ms | Same grid |
|---|---|---|---|---|---|---|---|
| Classic | 239,825 | 12,900 | 0.058 (0.058) | 5.00 (58.25) | 3.66 / 2.87 | 2.22 / 1.24 | 400 of 400 |
| Dynamic | 228,003 | 35,319 | 0.159 (0.158) | 10.10 (55.46) | 3.05 / 2.54 | 2.13 / 1.43 | 400 of 400 |

P/I is each P-frame over the intra frame of the same grid, and I-frame
times leave out frame 0's warm-up. A P-frame decodes in 2.09 / 2.63 ms of
host time against an I-frame's 2.88 / 3.26. The F-scores, 0.9980 and
0.9984, are the all-intra runs'. At period 1, the median of six runs before
and six after, interleaved (room0 Classic, 1 cm): encode 2.99 → 3.07 ms of
host time and 2.04 → 2.06 ms of device time (`..forward`, which now copies
the sorted list into the store, +0.04 ms), decode unchanged, and the bytes
and F-score the same.

**Validation.** Apple M5 Max, Release with warnings as errors: the 98 tests
pass under the validation layer, and the codec and volume tests under
synchronization validation too.
- `recon_codec_inter` holds every frame of chains of a sphere that moves
  and grows beside one that stays -- at period 4 and open-ended, at K 37 in
  segments of 3 and at the defaults -- to the CPU oracle's bytes: its intra
  frame of the grid, or `write_frame` of the oracle's P-frame with the
  price (`codec_frames::encoder_frame`, `codec_oracle::payload_price`); each
  decodes to its I-frame's grid. It makes each of the diff's choices from
  one edit: all skip, one block changed, a mask alone (keep, residuals 0),
  a block gone and one added, unrelated content zero and a small change
  keep, an empty frame all removed and then an I-frame; and a DC step of
  32768 zero where the price would keep. It also holds the diff's
  refusals.
- `recon_codec_encoder`: an intra period of 0 refused; each I-frame rule at
  period 3, rows included; refusals (another block size, no weight) that
  keep the reference, the count and a pending `force_intra()`; and a chain
  a fresh encoder repeats.
- `recon_codec_decoder`: chains of 32 frames at period 30 and open-ended,
  each decoded by its own decoder to the all-intra chain's grid at every
  frame; and a lost frame, `NotFound`, then `force_intra()`.
- `recon_codec_dct`: the forward's output buffers refused when short.

Thirteen planted bugs each failed a test: skip never chosen, the price
ignored, a wrap in the low half missed, a new block's rank off by one, a
borrow carried between halves, a skip blind to the mask, the ranks not
dispatched, the sorted list not copied, the count not restarted by an
I-frame, `force_intra()` never cleared, the empty-reference or the geometry
rule dropped, and the period off by one. Not run: the RTX 5090, ASan and
UBSan (CI runs them), and Rafa2, which waits for `codec_mesh` to code a
sequence.

### 2026-10-14 — Mesh normalization and the topology audit are opt-in, generic `io` helpers for a mesh sequence: bounds measured along one up and joined over frames, and one transform for every frame (amends the 2026-10-02 normalized-mesh and 2026-10-03 codec-mesh entries).

**Why.** The user, 2026-10-14: "keep in mind the normalization is sequence
specific, we have Rafa2 now but we can change the data sequence any time. it
should be designed to handle generic temporal mesh sequences", and "scaling
can be a helper in the IO module that we can opt in to use". `codec_mesh`'s
copy fitted a transform to one frame and was written around Rafa2. A
sequence needs one transform, fitted to all its frames and applied to each,
so a point that does not move keeps its place, and any dataset needs the
same arithmetic. So the helpers are `io`'s, opt-in (DESIGN, io), and the
AGENTS.md rule now lets `io` offer them while no tier applies them on its
own.

**Choices.**
- The up direction is fixed before measuring, and the bounds carry their
  axes: a box in input coordinates cannot give the exact extent along a
  tilted up, and Rafa2's is tilted.
- The origin stays in the rotated frame, in source units; one in input
  coordinates would change the rounding.
- A component with holes is capped before its volume is taken. Taken about
  its union-find root vertex, an open component's volume depended on that
  vertex, so on index order: the open tetrahedron {0,1,3}, {0,3,2},
  {1,2,3} gave 4/3, the same faces written from vertex 3 gave 0, and
  `allow_holes` passed one and refused the other.
- The axes must be a proper rotation, to the camera tier's 1e-6, because
  the sequence will read a transform back from its `.vrst`.
- `mesh_normalization.cpp` compiles with `-ffp-contract=off`. In #242's
  review, Apple clang fused 32 multiply-adds in it and GCC 16 (aarch64) 35,
  and with contraction off the last bit of the origin's x and z changed, so
  a stored transform would have depended on the compiler. Rafa2 frame 1's
  placed floats are the same either way.
- The helpers use recon's `Vec3d` (glm's `dvec3`) and glm's functions, as
  the rest of recon does. Normalizing with `glm::normalize` moved five axis
  components and the scale by one ulp, by the same review's count; frame
  1's placed floats did not change, and no transform had been stored yet.

**Same bytes.** `codec_mesh` on Rafa2 frame 1 at 5 mm, on main and on this
change (Apple M5 Max, Release, validation on): the `.vrst` is 67,272 bytes,
sha256 `b53eb950…`, on both; `_input.ply` and the printed normalization are
identical. `_source.ply` and `_decoded.ply` differ in vertex order between
two runs of main itself, as marching cubes emits through atomics; as sorted
triangle sets they are identical on every run, and so is every non-timing
line of the report and its sweep.

**Validation.** `recon_io_mesh_normalization` tests every helper on
synthetic meshes, holes reordered, shifted and rotated among them. All 51
mutations planted in the helpers fail it, 16 of them the review's. Release
with warnings as errors and Assimp passes all 84 tests under the validation
layer.

**Next.** `codec_mesh`'s sequence extends the bounds over its range, applies
one transform to every frame and stores it in the `.vrst`.

### 2026-10-14 — `codec_mesh` codes a numbered mesh sequence under one transform: a frame number in `source.path`, an exact range, one transform fitted to every frame and stored in the stream, a cleared grid and one record per frame, and `source.allow_holes` (amends the 2026-10-02 normalized-mesh, 2026-10-03 codec-mesh, 2026-10-11 examples and 2026-10-12 container entries).

**Why.** The user, 2026-10-14: "keep in mind the normalization is sequence
specific, we have Rafa2 now but we can change the data sequence any time. it
should be designed to handle generic temporal mesh sequences". The temporal
codec needs consecutive frames of one geometry. Core 0.7.0 (core #33) gives
`codec_mesh` the keys and checks only their types and ranges; the frame
number, the placement and the conversion's rules are recon's.

**Choices.**
- **The frame number is in the path.** printf's `%d` or `%0Nd` (1 ≤ N ≤ 9),
  once, and `%%` for a literal `%`: the part ffmpeg's image sequences read,
  for any naming that carries a frame number, where stepping a file's last
  digits would step `_v2` or `_lod0`. `%5d` is refused, since printf pads it
  with spaces and ffmpeg with zeros. Only `codec_mesh` reads it
  (`numbered_path`), so recon owns it; a bad pattern fails at run time, before
  any GPU work, with exit 1.
- **`run.frames` is an exact count from `run.first`.** A dataset's is a limit
  that stops at its last frame; here a gap is an error, and "every frame"
  would code Rafa2's mirrored replay (frames 151–300 replay 150–1). Every
  file is checked before the first load: missing is `NotFound`, present but
  not a regular file `InvalidArgument`, as `io::load_mesh` answers. A range
  key on a plain path is refused, naming it, as its number would select no
  file.
- **One transform for the sequence.** A first pass loads each frame, joins
  its bounds along the up axes and drops it, so memory does not grow with the
  range; one transform is fitted to the union (`fit_transform` to
  `height_m`, or `scale_transform` by `metres_per_unit`) and places every
  frame unchanged, so a still point stays still and a persisting block keeps
  its coordinates. Fitting frame 1 alone, as step 0 did, sizes the range by
  one frame. The cost is loading every frame twice.
- **The transform is stored** in the Geometry track's params, 112 bytes:
  "VRMT", version 1 and 13 little-endian binary64 (axes x, up, z; origin;
  metres per unit). `io::encode_transform` and `decode_transform` are its one
  home, beside `MeshTransform`, since a player needs it and `recon_stream`
  links only the core's base tier. This is an addition to the v0 Geometry
  kind, whose params were empty: an old reader passes them through, so the
  version holds. A single mesh stores its transform too.
- **The audit runs twice.** The first pass audits each frame on its source
  coordinates. In exact arithmetic a proper similarity changes no count,
  hole, orientation or volume sign, so a frame signed mode cannot take is
  refused there, before any GPU work. Rounding can still make a placed
  triangle degenerate or flip a volume's sign, so the second pass audits the
  placed coordinates and decides.
- **`source.allow_holes`.** Off by default and signed mode only: it admits a
  mesh whose only defect is boundary edges, each hole growing a skirt to the
  truncation past its rim (`MeshSdfMode::Signed`). Rafa2's frames 74–99 have
  8 boundary edges each, two 4-edge loops 13 and 21 mm across, each missing
  2 triangles; frames 128–131 have 16, loops of 9 and 7 edges at most 13 mm
  across. No frame has another defect. The sequence template sets it.
- **A cleared grid per frame.** `MeshIntegrator` overwrites only the blocks
  its band reaches, so an earlier frame's blocks the new band misses would
  stay: each frame is converted into the one source grid after a `clear`.
  The player grid is never cleared, as the `Decoder` leaves exactly a
  frame's blocks.
- **Records.** Set the frame's number, pts that number at 30 fps (as
  Replica's frames, which carry no time either), seq from 0, all key until
  P-frames are coded. A single mesh is set 0 at pts 0, as before. Quality,
  the sweep and the PLYs are the last frame's; the CSV schema holds.
- **The stream before any conversion.** `codec_mesh` creates its stream after
  the GPU setup and before the first conversion or PLY, so a rerun onto an
  existing stream is refused before any GPU work on a frame or any output,
  and a mesh signed mode refuses leaves no `_input.ply`. A later failure
  leaves a `.vrst` that reads as unfinished, as any mid-run failure of a
  sequence does. `codec_replica` keeps its order.
- **The shell bound is the conversion's.** Core 0.7.0 checks only that
  `shell_voxels` is finite and positive. `MeshSdfParams::validate(grid)`
  holds the `[√3/2, trunc_dist)` rule; `MeshIntegrator::integrate` calls it,
  and `codec_mesh` calls it before any GPU work with the resolved truncation.
- **The band.** `fusion.truncation_m` sets it, and an absent value is still
  4 voxels, so `codec_mesh.json` is unchanged. `codec_mesh_sequence.json`
  sets 0.04 m, 8 voxels: the paper's band and step 0's. *Amended
  2026-10-21:* `codec_mesh` refuses a band too narrow for a true SDF, and
  both templates set 0.08 m, 16 voxels (the 2026-10-21 true-SDF entry).
- **`codec.entropy` is gone** from core 0.7.0, and recon stops reading it.
- **One home for a mesh file's checks.** `io::check_mesh_file` holds what
  `io::load_mesh` checks of a path before it reads; the sequence calls it
  for every frame before the first load.
- **A placement past float range is the fit's refusal.** The fits refuse
  bounds they would place past float range, which `apply_transform` would
  refuse frame by frame, so `codec_mesh` refuses such a scale before any GPU
  work instead of after coding a frame.
- **A `%` in the scenario's directory.** *Amended 2026-10-15:* pin core
  0.8.0 (core #34, `c072a23`) and require that version for examples and
  their config tests. Core 0.7.0 prepended the directory without escaping:
  a scenario under `50%off/` was refused, and one under `take%d/` could
  silently read `take0/`. Core now escapes the directory's `%` as `%%`
  for CodecMesh's `source.path`; `numbered_path` reads it as a literal
  `%` in plain and numbered paths. Tests load real scenario files through
  `config::load` and compare paths and geometry with a different mesh
  present in `take0/`; the executable's inspection counts catch the same
  wrong-file selection. Both regressions fail against the previous pin.
- **The new intra-period key.** Core 0.8.0 accepts `codec.intra_period` for
  both codec profiles, but the examples still write all-key records. Their
  shared `encoder_config` refuses any value other than 1, naming the key,
  before GPU setup. C5b will connect the period and record frame types;
  accepting and ignoring it would silently change the requested coding.
  *Amended 2026-10-16:* they map it (the 2026-10-16 entry).

**Frame 150's conversion outlier.** Kept: the sequence stays 1–150. Under
the range's transform, frame 150's conversion accuracy is mean 0.024, rms
0.536, p95 0.063 and max 38.358 mm (F 0.9997 at 2.5 mm); step 0 measured
37.144 mm under frame 1's. Every vertex measured (the report samples 1 in
4), 340 of 550,470 extracted vertices lie more than a voxel from the placed
input, at most 39.444 mm, in three clusters within 8 cm of
(−0.25, 1.06, −0.08) m placed, (0.0054, −0.1147, 0.8006) in source
coordinates. Of the 289 observed voxels within two voxels of them, 18 take
the side the input's generalized winding number does not. At 14, faces
within 1 µm of the nearest point disagree on the side and the conversion's
pick takes the wrong one: the documented failure of the closest-face sign
past an acute edge (`MeshSdfMode::Signed`), here at a sliver fold whose
faces meet their neighbours at 0.1–1.8°. At the other 4, one face's
interior is nearest and puts the voxel inside, though the winding number
puts it outside; that needs a self-intersection, which no audit checks, so
that part is unverified. The cause is frame 150's own geometry, not the
sequence.

**Same bytes, one frame.** `codec_mesh` on Rafa2 frame 1 at 5 mm
(`codec_mesh.json` with `output.stream`; Apple M5 Max, Release): its one
record is byte for byte main's, and its `.vrst` is 67,384 bytes, main's
67,272 plus the 112-byte params. The `VRST` body holds them, so its size and
both its CRCs change, and every later chunk starts 112 bytes later, so the
`INDX` key entry's offset and the `INDX` and `TAIL` CRCs and index offset
change. Stripping the params and shifting those offsets back gives main's
file, sha256 `b53eb950…`. `_input.ply` and every non-timing comparison and
sweep line are identical.

**Rafa2, frames 1–150** (`codec_mesh_sequence.json` pointed at V-SENSE's
`references/Rafa2_obj`, with outputs; intra period 1; Apple M5 Max, Release):
- 150 records, sets 1–150, seq 0–149, pts 33,333,333 to 5,000,000,000 ns,
  all key; the stream finished, the CSV 151 lines.
- 61,869 bytes a frame on average (58,056 at frame 148 to 65,217 at frame
  125), 3,426 blocks (3,233–3,546), 18.1 bytes a block. Against step 0's
  frames under frame 1's transform (scale 2.13650823286 against the range's
  2.13436903472), each frame is 0.9971 times as large on average
  (0.9916–1.0047), correlation 0.9915.
- 12.2 s wall for the run, both passes included. Per frame, the conversion
  takes 8.7 ms host and 6.6 ms device, encoding 2.7 and 1.6, and decoding
  3.1 and 1.7 (all from one run).
- A fresh `Decoder` decodes every record into a fresh grid to the run's
  player grid bit for bit (blocks, `tsdf` and `weight`); the last frame
  extracts to the run's 184,708 triangles.
- Frames 74–99 and 128–131 are admitted open. Inspection over the range
  lists the same 30 without a GPU in 6.9 s, and without `allow_holes` the
  run is refused at frame 74 after 1.7 s, before any GPU work or file.

```sh
# $dataset_root/Rafa2_obj is V-SENSE's references/Rafa2_obj: point the
# template's source.path at "$dataset_root/Rafa2_obj/Frame_%05d_textured.obj"
# and add output.stream and output.csv.
"$recon_root/build/examples/codec_mesh/codec_mesh" \
  --config "$recon_root/config/examples/codec_mesh_sequence.json"
```

**Validation.** `recon_example_mesh_sequence` covers the pattern and its
refusals, every file checked before any load, one transform over a growing
sequence where frame 1 alone fits wrong, a plain path placed as one mesh was,
signed mode's rule with and without `allow_holes`, inspection over a range,
each frame in a cleared grid, and `code_frame`'s records and stored
transform. `recon_io_mesh_normalization` round-trips the transform bit for
bit; `recon_tsdf_mesh_integrate` holds `validate` to `integrate`'s
refusals; `recon_codec_mesh_run` runs the executable on a device, an
existing stream refused before any conversion or output, then two frames;
the config-entry test inspects a range and counts its open and refused
frames, and with no Vulkan driver to find, refuses twelve coding scenarios,
each for its own reason before any GPU work. `recon_io_mesh_file` holds
`load_mesh` to `check_mesh_file`.

### 2026-10-16 — The codec examples code P-frames: `codec.intra_period` and `codec.update_size` reach the `Encoder`, and a stream's rows, report and keys go by frame type; an empty grid is an I-frame; golden P-frames pin type 1's bytes, and its freeze waits, since at R = 16 its P-frames code slower than intra (amends the 2026-10-12 P-frame, 2026-10-14 encoder and codec-mesh entries).

*Amended 2026-10-17 (the update-size entry, below):* `update_size` 0, the
default, takes the segment size, and `kDefaultUpdateSize` is gone; "U stays
64" now holds at R = 64. The user chose neither option under "Open, for the
user": no lanes, and the freeze waits on the count moving into the
forward's submit, then a re-measure.

*Amended 2026-10-17 (the freeze entry, below):* type 1 is frozen, though
r16's P encode stays slower than intra on the host, and the templates code
at period 30. Type 1's rows and the worked example, which is
`codec_frames::example_bytes` in the tests, not in `bitstream.hpp`, follow
that entry's rule, not the golden rule below: they change only on purpose.

**Type 1 stays experimental,** and the templates at period 1. Its freeze
needed P-frame encode and decode no slower than intra on room0 at 1 cm and
on Rafa2's frames 1–150. Room0 meets it and Rafa2 at R = 16 does not, so
the freeze waits on the user's decision about interleaved rANS lanes (both
below).

**An empty grid is an I-frame,** the sixth rule: a grid with no observed
block codes intra whatever the reference. The P-frame it replaces removed
every reference entry at 8 bytes an update segment (its length and final
rANS state), a few KB at room0's sizes, where the intra frame of no blocks
has no segment. With the fifth rule, an empty frame on either side makes
an I-frame. The format still admits a P-frame of no blocks, and the
decoder decodes one.

**U is the caller's.** `EncoderConfig::update_size` sets a P-frame's update
segment, 1 to `kMaxDeviceDecodeSegmentSize` (`check_segment_size`, the
format's one segment rule); its default restates `kDefaultUpdateSize`, held
equal by a `static_assert`, as the segment size's is. `EncoderConfig::validate`
holds the config's checks: `Encoder::create` makes them first, and the
examples before any GPU work. Core 0.9.0 (core #35) reads
`codec.update_size`, 0 being the codec's own, and refuses one given at a
period of 1, where no P-frame uses it, by core's rule that an ignored
explicit setting is an error: only core can tell an explicit key from its
default.

**The examples.** `encoder_config` maps both keys, and the refusal of a
period other than 1 goes. `CodecStream`:
- keeps each frame's type in its row and the CSV's `type` column;
- makes a record a key exactly when it holds an I-frame, where a reader
  starts or seeks;
- reports each type's bytes (mean, and nearest-rank p50, p95 and max),
  stage rows and rANS share, and at a period above 1 a GOP's bitrate: an
  I-frame and the period's P-frames at their types' means, the rate a long
  stream settles at whatever I-frames its own run forced;
- calls `force_intra()` on any failure after the encode: the append, the
  decode or the CSV. The encoder codes its next P-frame against every frame
  it returns, so a frame the stream or the player lacks would leave the next
  P-frame `NotFound` until an I-frame. A frame index or pts out of order is
  refused before the encode, so it needs no recovery.

The sweep codes every row intra (`sweep_encoder_config`): a row re-encodes
one grid, which a P-frame would code as all skips.

**Golden P-frames.** `kGoldenInterFrames` (`tests/codec_frames.hpp`) pins
type 1 as `kGoldenFrames` pins type 0. Each row is a P-frame against a
golden I-frame row's frame and carries that row, its mode pattern, R and U,
its content hash (which now covers the modes, leaving every intra value as
it was), its size and the FNV-1a of its bytes, which pins `frame_hash` too
through REFERENCE. The oracle writes each and reads it back against its
reference; the device writes the same bytes and reads them as the oracle
does.

| Reference row | K | R | U | Covers |
|---|---|---|---|---|
| 700 blocks | 64 | 64 | 64 | every mode, 90 new blocks spread through |
| 65 blocks | 21 | 7 | 6 | a partial last segment of each kind |
| int32 extremes | 4 | 1 | 1 | every mode, new blocks at the extremes |

U is 6 on the second row so that its 65 entries end in a partial update
segment. The golden rule is now "a new frame type adds rows; nothing else
changes a value" (a new `kFrameVersion` is a new format, with goldens of its
own). It binds type 1's rows and the worked example (`bitstream.hpp`'s
161-byte P-frame) from type 1's freeze; until then only a change to type
1's format, such as the lanes below, may change them.

**Measured.** Apple M5 Max, MoltenVK, Release; a warm-up, then the median
of 3 runs; host ms is a codec call's wall clock and device ms its timed
device spans, each a mean over every frame of a type. Room0's frames 0–399
through `codec_replica` (`encode_every` 1), at 1 cm (4 cm band) and 2 cm
(8 cm), Classic (c1, c2) and Dynamic (d1, d2), R = 64; Rafa2's frames 1–150
through `codec_mesh` with `codec_mesh_sequence.json`'s settings (5 mm),
R = 64 and 16 (r64, r16). K = 64, the uniform table, scale 0.2, U = 64.
Open-ended is a period of 1000. Bytes a frame, with Mbit/s at 30 fps:

| Input | Period 1 | 8 | 30 | Open | I p95 / max | P at 30: mean / p95 / max |
|---|---|---|---|---|---|---|
| c1 | 242,699 (58.25) | 41,498 (9.96) | 20,843 (5.00) | 13,230 (3.18) | 274,780 / 275,004 | 12,900 / 21,283 / 47,856 |
| c2 | 74,320 (17.84) | 13,432 (3.22) | 7,180 (1.72) | 4,877 (1.17) | 83,670 / 83,690 | 4,773 / 7,771 / 15,041 |
| d1 | 231,082 (55.46) | 59,683 (14.32) | 42,063 (10.10) | 35,571 (8.54) | 268,319 / 269,182 | 35,319 / 41,221 / 51,805 |
| d2 | 69,502 (16.68) | 17,575 (4.22) | 12,225 (2.93) | 10,261 (2.46) | 79,818 / 80,014 | 10,177 / 12,480 / 16,082 |
| r64 | 61,869 (14.85) | 31,696 (7.61) | 28,419 (6.82) | 27,451 (6.59) | 64,025 / 65,217 | 27,255 / 45,742 / 53,047 |
| r16 | 64,888 (15.57) | 32,089 (7.70) | 28,533 (6.85) | 27,483 (6.60) | 67,082 / 68,341 | 27,267 / 45,796 / 53,155 |

Ms a frame at period 30, host / device:

| Input | I encode | P encode | I decode | P decode | P no slower |
|---|---|---|---|---|---|
| c1 | 3.248 / 2.180 | 2.170 / 1.097 | 2.901 / 1.976 | 1.951 / 1.149 | yes |
| c2 | 2.745 / 1.880 | 1.551 / 0.568 | 2.371 / 1.602 | 1.171 / 0.557 | yes |
| d1 | 3.210 / 2.223 | 2.516 / 1.446 | 3.477 / 2.010 | 2.487 / 1.397 | yes |
| d2 | 3.081 / 2.198 | 1.802 / 0.810 | 2.765 / 1.793 | 1.677 / 0.808 | yes |
| r64 | 2.632 / 1.621 | 2.557 / 1.370 | 2.750 / 1.678 | 2.468 / 1.404 | yes |
| r16 | 1.554 / 0.547 | 2.534 / 1.366 | 1.555 / 0.523 | 2.438 / 1.387 | no |

Each run agrees: r16's P decode is 0.88–0.95 ms slower than its I decode,
and r64's P encode about 3% faster than its I encode. Dropping each type's
first frame, which allocates, changes no verdict. By stage (host ms; the
breakdown rows the examples print), a P-frame's forward carries its second
count submit, and its rANS stages follow its segment count:

| Stage | c1 I | c1 P | r16 I | r16 P |
|---|---|---|---|---|
| `..active set` | 0.172 | 0.161 | 0.231 | 0.227 |
| `..observed` | 0.359 | 0.352 | 0.192 | 0.196 |
| `..forward` | 0.621 | 0.908 | 0.390 | 0.654 |
| `..rans encode` | 2.093 | 0.747 | 0.732 | 1.467 |
| `..rans decode` | 1.965 | 1.108 | 0.801 | 1.662 |
| `..apply` | 0.290 | 0.284 | 0.478 | 0.518 |
| `..inverse` | 0.576 | 0.557 | 0.264 | 0.258 |

The real coder lands on step 0's price (the 2026-10-12 entry; mean P/I,
open-ended): c1 0.058 against 0.058, c2 0.070 against 0.069 and d1 0.159
against 0.158, on the same grids (the I-frame means equal step 0's to the
byte); r64 0.440 against 0.431 and r16 0.420 against 0.411, on near grids
(`codec_mesh` fits one normalization to all 150 frames and grows
`allow_holes` skirts, where step 0 placed each frame by frame 1's). Every
chain measured, the periods above on each input and the U sweep below, 26
in all, decodes with a fresh `Decoder` bit for bit to the all-intra
stream's grids, and its I-frames are the all-intra stream's bytes; a chain
checked against another input's stream fails every frame. Accuracy,
coverage and F-score are the same at every period, U and run, and every
run writes the same bytes. Not run: the RTX 5090 (unreachable) and E1's
rig dumps (none exist yet).

**U stays 64.** The rule: the U of the fewest total bytes over c1, d1 and
r64 at period 30, among those whose P-frame encode and decode host ms are
within 5% of U = 64's on each. Bytes, and P encode / decode host ms:

| Input | U = 64 | 128 | 256 |
|---|---|---|---|
| c1 | 8,337,022; 2.170 / 1.951 | 7,862,096; 2.325 / 2.094 | 7,631,050; 2.605 / 2.425 |
| d1 | 16,825,140; 2.516 / 2.487 | 16,532,546; 2.960 / 2.994 | 16,386,974; 3.878 / 4.029 |
| r64 | 4,262,910; 2.557 / 2.468 | 4,235,630; 3.556 / 3.624 | 4,222,312; 5.086 / 5.922 |

U = 256 saves 4.03% of the total and U = 128 2.70%, but their P-frames
take 20–140% and 7–47% longer: fewer update segments, less parallelism.

**The freeze waits on a decision about rANS lanes.** Rafa2's frames hold
about 3,430 blocks, and the device codes one segment an invocation. An
I-frame at R = 16 has about 216 segments, but a P-frame's reference entries
go in ceil(M / U) update segments, about 54 at U = 64, so its rANS stages
take twice an I-frame's (the stage table). U = 16 gives 215 segments, and
r16's rANS stages and P decode come back to intra's, but not its P encode,
whose forward carries the second count submit. Host ms at U = 16 over two
sets of runs, against the I-frames of the same runs and of period 1 (decode
and encode with each type's first frame dropped, the stages as the reports
print them):

| r16 | P | I |
|---|---|---|
| decode | 1.51–1.61 | 1.56–1.75 |
| encode | 1.83–1.88 | 1.52–1.72 |
| `..rans decode` | 0.74–0.77 | 0.79–0.88 |
| `..rans encode` | 0.66 | 0.70–0.76 |
| `..forward` | 0.73 | 0.38–0.52 |

U = 16 also costs 3.8% of r16's stream (4,442,994 against 4,279,898 bytes)
in each segment's length and final state. Interleaved rANS lanes within an
update segment would give the device that parallelism without the segments,
but change type 1's format, and encode at R = 16 would still need the
second count submit gone, which this entry leaves to a later performance
change. Open, for the user: add the lanes before the freeze, which then
waits on that change too, or freeze type 1 as it is and accept P-frames
slower than intra at small R.

**Validation.** Apple M5 Max, Release with warnings as errors, core
`138914f`: the 87 tests pass under the validation layer, and the codec,
example and volume tests under synchronization validation too. Fifteen
planted bugs each failed a test: the period not mapped, an update size of
0 passed through, the sweep's period not pinned, every record a key, no
record a key, no `force_intra()` after a refused decode, none after a
failed CSV row (the test stops the CSV with the file size limit; a failed
append takes the same branch, but the writer's failure sticks, so no later
frame decodes to show it), the median one rank high, a GOP weighting its
P-frames by the period, every frame's stage rows filed as the I-frames',
every row typed intra, the empty-grid rule dropped, a P-frame counted at
the default U, `validate` blind to the update size, and `frame_hash`'s last
shift changed, which the device's golden P rows catch past the worked
example. By hand, `codec_replica` at period 30 prints
period 1's accuracy, coverage and F-score. Not run: the RTX 5090, ASan and
UBSan (CI runs them).

### 2026-10-17 — A P-frame's update segments default to its segment size: `EncoderConfig::update_size` 0, the default, takes `segment_size`, resolved once in `Encoder::create`; the format and the goldens are unchanged (amends the 2026-10-16 codec-examples, 2026-10-14 encoder and 2026-10-12 P-frame entries).

**The user's choice.** Of the 2026-10-16 entry's two options the user took
neither: no interleaved rANS lanes, a format change, and no freeze with
P-frames slower than intra at small R. U defaults to R instead; a P-frame's
count moves into the forward's submit next (the `TODO(codec)` in
`encoder.cpp`), and type 1's freeze follows a re-measure. *Amended
2026-10-17:* the count moved (the next entry). *Amended 2026-10-17:* the
re-measure left r16's P encode slower than intra, on the host alone, and the
user froze type 1 anyway (the freeze entry).

**The rule.** `EncoderConfig::update_size` 0, the default, takes
`segment_size`; an explicit 1 to `kMaxDeviceDecodeSegmentSize` still wins.
`EncoderConfig::resolved_update_size()` is the rule's one home: `validate`
checks it after the segment size, so a segment size of 0 is named as one,
and `Encoder::create` stores it, so the writer sees only a concrete U. The
format does not change: every P-frame carries its U in REFERENCE, and every
writer and reader still refuses 0. `kDefaultUpdateSize` goes; the oracle's
default U becomes `kDefaultSegmentSize`, so the goldens and the worked
example keep their bytes. Core is unchanged: its 0 "selects the codec's own
size", which `encoder_config` now passes through.

**Why.** At U = R a P-frame's M reference entries take ceil(M / R) update
segments, as many as its reference has coded intra, so the device codes and
decodes it with at least that parallelism at any R. Nothing changes at
R = 64. At R = 16 the 2026-10-16 entry's U = 16 runs bring P decode level
with intra for 3.8% of the stream.

**Rejected:** `std::optional` (core already uses 0, and every use would
need `value_or`), a default member initializer (it would not follow a later
`segment_size`), a mapping in the examples alone (a library caller at
R = 16 would keep 64), and U always R (it loses the U sweep, the odd-U
tests and core's key).

**Validation.** Apple M5 Max, Release with warnings as errors, core
`138914f`: the 91 tests pass under the validation layer, and the codec,
volume and example tests under synchronization validation too; eight
planted bugs each failed a test. By hand, on the 2026-10-16 entry's
protocol: Rafa2 at R = 16 and period 30 with no `update_size` prints update
segments of 16 and writes the U = 16 runs' bytes; P decode comes level with
intra (1.467 against 1.499 host ms) while P encode stays slower (1.697
against 1.485), with device ms level; r64 and c1 keep their bytes; and
every chain decodes bit for bit.

### 2026-10-17 — A P-frame is counted in the forward's submit: its count and step layout run over M plus the frame's blocks, the device takes off the diff's survivors, and an encode waits on one fence fewer (amends the 2026-10-14 encoder entry).

*Amended 2026-10-17 (the freeze entry, below):* the re-measure ran, and type
1 is frozen.

**The bound.** `ReferenceStore::record_diff` returns the P-frame's entries
as its batch leaves them, `ResidentBlocks::bounded`: M + n positions, of
which the `ranks[M]` survivors are counted twice, as a reference entry and
as a frame block, so the frame's M + A entries are the first
M + n − `ranks[M]`. `DeviceFrameWriter::record_count` copies that word into
a 4-byte surplus (binding `ranks` at offset 4M would break
`minStorageBufferOffsetAlignment`); exact entries zero it. The count kernel
and `StepLayout` take N from `counted_entries` (`rans_push.glsl`): the
bound less the surplus, clamped to the new blocks, so a wrong word reads
nothing past the bound. The layout's segment count follows from N through
`segment_of`, and lane 0 writes both totals to a buffer of their own, read
back at offset 0 whatever N the device took.

**Direct over the bound, not indirect.** The `TODO(codec)` asked for an
indirect count dispatch. Core's `dispatch_indirect` would serve, but its
arguments need a kernel of their own or `survivor_ranks.comp` tied to the
writer's width, I-frames would keep a second, direct path, and the layout
would still need N on the device. Over the bound, a survivor's invocation
reads one word and returns.

**Where A is read.** The writer reads the surplus back with the counts, and
`finish` takes the same bounded entries: it refuses a surplus above
min(M, n), which only a bug makes, and codes the first count − surplus
entries. `diffed()` and the store's 8-byte readback go, so one frame has one
entry set. A comes from the device's survivor count and the frame's block
count from the mode counts, so the encoder's check that the frame holds
every block the forward made still compares two results.

**Bytes and refusals.** The count walks the M + A entries it walked before,
so every frame's bytes, the goldens', and every reachable refusal are
unchanged. `record_count`'s refusals of the bound cannot fire from the
encoder. An allocation failure in the count now comes before a forward
reject, as for an I-frame, and the survivor check moves from `diffed()` to
`finish`, still `InvalidArgument`.

**Fence waits an encode,** counted against a scratch core that counts every
`vkWaitForFences`, frames of two spheres (433 and 494 blocks), M5 Max:

| Encode | Before | After |
|---|---|---|
| I-frame, the map's list held | 3 | 3 |
| I-frame, compacting | 4 | 4 |
| P-frame, the map's list held | 4 | 3 |
| P-frame, compacting | 5 | 4 |

A payload that outgrows `finish`'s read-ahead adds one. Decode is unchanged.

**Hand check.** M5 Max, Release, period 30 at the default U, before
(`a25fc759`) and after: a warm-up, then three runs of each in shuffled,
interleaved order; the median of per-type means, host and device apart,
each type's first frame dropped (`..forward` is the report's P row).

| Input | I encode host | P encode host | P `..forward` | P encode device |
|---|---|---|---|---|
| c1, room0 1 cm Classic | 2.979 → 3.034 | 2.204 → 2.011 | 0.933 → 0.740 | 1.117 → 1.097 |
| d1, room0 1 cm Dynamic | 3.085 → 3.025 | 2.585 → 2.379 | 0.842 → 0.655 | 1.465 → 1.453 |
| r64, Rafa2 frames 1–150 | 2.553 → 2.567 | 2.530 → 2.327 | 0.670 → 0.491 | 1.382 → 1.358 |
| r16 | 1.499 → 1.518 | 1.704 → 1.534 | 0.672 → 0.509 | 0.558 → 0.546 |

A P encode's host time falls by 0.17–0.21 ms, `..forward` by 0.16–0.19 ms,
about twice the 0.10 ms the 2026-09-28 entry's step 5a measured for one
batch and fence wait. I encode moves by at most 0.06 ms either way, and P
device time falls by 0.01–0.02 ms though the count covers M + n. Every
`.vrst` is the same byte for byte, and so are the CSV's frame, type, bytes
and blocks in every run. At r16 a P-frame now encodes in 1.534 ms against an
I-frame's 1.518; the re-measure decides the freeze. Not run: the RTX 5090,
and ASan and UBSan (CI runs them).

**Validation.** Apple M5 Max, Release with warnings as errors, core
`138914f`: the 91 tests pass under the validation layer, and the codec and
volume tests under synchronization validation too.
`recon_codec_step_layout` lays out every frame again with garbage steps
past N; `recon_codec_device_frame` writes the matrix, the golden P-frames
and the writer's refusal edits from bounded entries too, over a tail whose
−32768s a count past N would flag; `recon_codec_inter` counts the diff's
all-remove frame in its batch. Twelve planted bugs each failed a test: the
surplus never copied, the layout's segment count from the push, the totals
read at the bound's end, a device segment count one more at A = 0 (the
step layout's test alone, as the bytes do not change), the surplus not
zeroed for exact entries, `finish` coding the bound, the survivor refusal
dropped, the clamp dropped, the count's exit at the bound, `record_diff`
leaving the bound unmarked, the surplus copied from `ranks[0]`, and
`finish` blind to the bound. An encoder that still submits before
`record_count` passes every test; only the fence count catches it.

### 2026-10-17 — Frame type 1 is frozen, so its bytes change only on purpose, and the codec templates code at period 30: P-frames code no slower than intra on room0 at 1 cm and on Rafa2 at R = 64, and at R = 16 decode no slower but encode up to 4% slower, on the host alone (amends the 2026-10-12 P-frame, 2026-10-16 codec-examples, and 2026-10-17 update-size and count entries).

**Frozen** means a change to type 1's bytes is deliberate, never
incidental; for type 1, this replaces the 2026-10-16 entry's golden rule.
Its goldens, `kGoldenInterFrames` and the worked example
`codec_frames::example_bytes`, fail on any change. A deliberate one amends
this entry and updates them in the same PR, and bumps `kFrameVersion` only
if files written before must stay readable. Mode 4 stays reserved for
motion, which its rule (the 2026-10-12 entry) lets a later design define
without changing a frozen byte.

**The re-measure,** its protocol declared before it ran, on main `d71fe849`
(core `7a1219e`), Apple M5 Max, MoltenVK, Release. The scenarios are the
committed templates with `source.path` and `output.csv` set and
`codec.sweep` off, and `output.stream` in the first measured run; d1 adds
`fusion.dynamic` true, and r16 `codec.segment_size` 16. None sets
`update_size`, so U = R. Each input also ran at period 1. A warm-up, then
10 runs, each pass over the 8 configurations in a shuffled order; a type's
time is the median over the runs of its mean host ms (the printed
`codec encode` and `codec decode` rows), device ms beside it. The rule: P
no slower than I, with no tolerance, encode and decode on each input,
against the same run's I-frames with (a) every frame and (b) each type's
first frame dropped. Under (b), host / device ms:

| Input | I encode | P encode | I decode | P decode |
|---|---|---|---|---|
| c1, room0 1 cm Classic | 3.069 / 2.092 | 2.070 / 1.106 | 2.863 / 1.978 | 1.996 / 1.172 |
| d1, room0 1 cm Dynamic | 3.078 / 2.147 | 2.380 / 1.449 | 3.258 / 2.011 | 2.504 / 1.405 |
| r64, Rafa2 frames 1–150 | 2.560 / 1.605 | 2.358 / 1.362 | 2.748 / 1.673 | 2.446 / 1.403 |
| r16, Rafa2 at R = 16 | 1.501 / 0.554 | 1.558 / 0.551 | 1.546 / 0.525 | 1.489 / 0.468 |

It holds on c1, d1 and r64, encode and decode, under both baselines and in
every run, and on r16's decode, where P is slower in 1 of the 10 runs. It
fails on r16's encode: by 9 µs under (a), 1.585 against 1.577 ms, and by
56 µs under (b), about 4%, with P slower in 9 of the 10 runs. Against the
period-1 runs' I-frames, for information, every comparison holds (r16's
encode 1.558 against 1.634). An earlier gate of 3 runs, on the update-size
and count changes before they merged, failed the same comparison under (b)
by 13 µs.

**The gap is the host's.** r16's device time is level, 0.551 against
0.554 ms, so the 56 µs is host time, recording and submit (a P-frame's
`..forward`, which waits on its diff too, takes 0.503 ms against an
I-frame's 0.415, every frame). Nothing the format decides is in it, so no
format change, interleaved rANS lanes included, would remove it, and the
user froze type 1 anyway.

**Bytes and accuracy.** c1, d1 and r64's period-30 streams are the
2026-10-16 entry's byte for byte, and r16's frames have its U = 16 run's
bytes (4,442,994 in all). Every configuration's CSV frame, type and bytes
columns match in all 10 runs, and accuracy, coverage and F-score at period
30 equal period 1's on each input.

**The templates.** `codec_replica.json` codes every frame at period 30
(`encode_every` 1; its 0 coded only the final grid), then sweeps;
`codec_mesh_sequence.json` codes at 30; `codec_mesh.json`, one frame,
keeps 1. None sets `update_size`, so each codes at the measured U = R = 64.
r64 and r16 above measure `codec_mesh_sequence.json` as it stood before the
2026-10-21 entry moved it to a true SDF at K 56 in a 16-voxel band.
`EncoderConfig::intra_period` and core's `codec.intra_period` stay 1, so no
caller's bytes change.

**Next.** Trim a P-frame's host overhead at small R, until r16's encode is
no slower than intra's.

**Validation.** Apple M5 Max, Release with warnings as errors and Assimp,
core `7a1219e`: the 93 tests pass under the validation layer, and the codec
and example tests under synchronization validation too.
`recon_example_config` checks each template's settings above, through
`encoder_config`. Five planted bugs each failed it: `codec_replica.json` at period 1 or at `encode_every` 0, the
sequence without its `codec` section or at a fixed `update_size` of 64, and
the single mesh at 30. The golden rows pass unchanged
(`recon_codec_bitstream`, `recon_codec_inter_bitstream`,
`recon_codec_device_frame`). By hand, each period-30 template, with its
paths set, prints period 1's accuracy, coverage and F-score and writes the
measured c1 or r64 stream byte for byte, which decodes with a fresh
`Decoder` bit for bit to the all-intra stream's grids. Not run: the RTX
5090, ASan and UBSan (CI runs them).

### 2026-10-21 — The `.vrst` chunks in memory are public: `ChunkEncoder` frames what `StreamWriter` writes, by the record rules it now holds, and `ChunkParser` reads bytes as they arrive, through `StreamReader`'s checks (amends the 2026-10-12 container entry).

A live view of the rig sends a session's chunks over TCP as the file holds
them, and the sockets go in a tier of their own, `recon_networking`, by the
owner's decision, testable apart from the format. That tier makes and checks
chunks, so the format's in-memory half is public in `recon_stream`, where
the format already is, rather than reached in stream's private `detail`.

- **One home for the format.** The writer is the encoder and the file: the
  session chunk, each record's header and padding, and the rules (a track
  the session has, seq rising, key pts never back), whose state moves into
  the encoder. The parser and the reader check a chunk with the same pure
  functions; the reader keeps its `pread`, index and search past damage.
- **The parser copies what it is fed.** A buffer handed out to read into
  would save one copy at a stream's rate, for a second call and a size
  contract. Copying takes pieces of any size, a byte at a time included.
  Drained between pieces, it holds about one chunk.
- **Checked before a body is waited for.** Each check runs as the bytes it
  needs arrive: the magic and H at 8, the CRC and fields at H, the body's
  CRC at its end. B is refused from the header: a session's above
  `kMaxSessionBytes` as `InvalidArgument`, as the reader refuses it, since
  the format caps it; any other above 64 MiB (`ChunkParser::kMaxBodyBytes`)
  as `OutOfMemory`, the parser's own limit, as
  `ReaderOptions::max_payload_bytes` is the reader's.
- **Every body's CRC, a skipped one's too.** A skipped body arrives
  anyway, and bytes that fail a CRC cannot be trusted past it.
- **Nothing resyncs.** A stream has no file to search past damage, so a
  failed check ends the parse, and every later call returns it.
- **A writer's refusal carries the encoder's** after the call and the path
  (`Status::with_context`), and `create`'s name the path now too. To count
  nothing for a record it cannot index, the writer reserves a key's index
  room before the encoder frames it.

**Verification.** Apple M5 Max, Debug and Release with warnings as errors:
`recon_stream_chunks` compares the encoder's bytes with the writer's file
before its footer and with an unfinished one on 1 and 2 tracks; refuses
every session and record the writer refuses, with the writer's message the
encoder's after the path; feeds finished files, a killed writer's cut in its
last padding, one with skipped chunks and one with a wider session header
whole, a byte at a time and in 12 seeded splits, each giving the reader's
records, offsets and payloads; meets each of 20 refusals at the byte that
completes it, bodies of 1 GiB before any arrives; and returns a failure
again after more bytes. `recon_stream_oom`
fails the encoder's and the parser's allocations. The existing stream tests
pass unchanged, and all of them under ASan and UBSan.

### 2026-10-21 — A session is served live over TCP by a tier of its own, `recon_networking`: the wire is the file, newest key wins, and a viewer joins at the cached GOP.

The player on the laptop, over the VPN to the lab, and later an iPhone show
the rig's reconstruction as it is fused: fuse_orbbec, or `serve_stream`
replaying a recording, sends; the player reads.

- **TCP.** A P-frame names its reference by hash, and the decoder refuses
  one without it (`NotFound`), so every record must arrive, in order, until
  the next key. A datagram transport would rebuild TCP's retransmission, or
  freeze to the next key on every loss. TCP's cost is head-of-line
  blocking, about an RTT (22 ms over the VPN) per lost segment. Measured from
  the lab, the path lost nothing and carried 4 to 11 MB/s from a Python
  sender with the 128 KiB mark below, at least 3 times a stream's 6.8 to
  10 Mbit/s at an intra period of 30. A chunk is one write with
  `TCP_NODELAY`: header and payload in two writes cost a P-frame 16 ms more
  under Nagle (measured).
- **The wire is the file.** The `VRST` chunk, then `RECD` chunks as the
  writer writes them, without the footer: a killed writer's file, which
  every reader takes. Each chunk carries H, B and both CRCs, so there is no
  framing of our own, and a viewer could save its bytes as a `.vrst`. The
  viewer sends nothing yet; a WebSocket would carry the same chunks, one a
  message.
- **Newest key wins.** A viewer that falls behind must lose records, but a
  P-frame lost alone breaks every frame to the next key. So the server keeps
  one GOP, and a viewer skips the unsent rest of it only once a newer key
  exists, and only at a chunk boundary: the outcome of dropping P-frames to
  the next I-frame, with the GOP log the one queue. The skip must happen in
  the sender, not behind the kernel's buffers: a peer that did not read took
  2.05 MB on Linux and 0.69 MB on macOS before pushing back, and with a
  128 KiB `TCP_NOTSENT_LOWAT` 0.20 MB on Linux, where the mark gates `send`
  too, and 0.59 MB on macOS, where it gates `poll` alone (loopback,
  measured). A viewer takes its next chunk only once its socket polls
  writable, and the client's receive buffer is 2 MiB, set before the
  connect so the window scale fits it. A stream carries at most one buffer
  a round trip: 256 KiB held it to about 12 MB/s over the VPN, where the
  rig at 1 cm needs 9.4 MB/s; 2 MiB holds that rig's key (0.78 MB) and
  three times its bytes a round trip (0.62 MB) (2026-10-22). Its stall is
  counted from what it acknowledges too, not from the socket polling
  writable alone: one draining under 64 KiB (Linux) or a chunk (macOS) a
  `stall_limit` stays above the mark as it reads (measured).
- **Joining at the cached GOP.** A viewer gets the GOP from its key, up to
  about 1.3 MB and 0.2 to 0.4 s over the VPN, and shows a frame at once.
  Waiting for the next key instead sends nothing extra but shows nothing for
  up to a second, and a replayed file cannot force a key. A viewer's first
  record, after the session, is the GOP's key, and the log keeps every
  record since, so once it has the session it misses no record but by
  newest key wins.
- **A tier of its own.** The sockets are a module apart from the format,
  "to separate concern and make it testable" (the owner).
  `recon_networking` links `recon_stream`, the format's in-memory half
  having been made public for it (the chunks entry, above), so it reaches
  into no private detail of stream. Its socket calls
  and its GOP log, the drop rule with no socket, thread or clock, are tested
  alone, under the loopback tests. The core gets no TCP helper until a
  second sibling needs one.
- **`publish` never waits.** fuse_orbbec's fusion thread calls it, and a
  slow iteration drops camera sets. It copies the record once, takes one
  lock and wakes the viewers; one thread per viewer sends, and a viewer's
  failure is logged at Info (the default sink drops it), never returned.
- **A typed address.** mDNS does not cross the routed WireGuard tunnel (the
  lab's `.local` name does not resolve), so a viewer names the sender's
  `host:port`; Bonjour is deferred, as are authentication and encryption.
  IPv4 alone, a name's lookup included: the lab's name has an A record
  only, and an IPv6 address could only be refused, or stall the connect
  where IPv6 is broken.
- **One track.** Newest key wins is per GOP; a second track would need a GOP
  of its own.

**Verification.** Apple M5 Max, Debug and Release (117 tests each, all
passing), and the stream and networking tests under ThreadSanitizer and
under ASan with UBSan, as CI now runs them on the Mac Debug leg and in the
sanitizer job. On loopback (127.0.0.1, port 0): a raw viewer's bytes are the
writer's file less its footer, and a client's records carry the file's
offsets; a viewer joining mid-GOP gets the GOP from its key, and before any
key nothing; one that does not read while 3 GOPs of 1 MiB records are
published gets the key it was sending, then the newest GOP whole, each
`publish` back at once; past `max_gop_bytes` nothing comes until the next
key; a viewer taking no byte is closed after a 200 ms `stall_limit`;
connection `max_viewers` + 1 ends before any byte; viewers leaving by a close
or a reset never fail `publish`; the destructor returns within 1 s with no
viewers, idle ones and a stalled one; 100 SIGUSR1 to a no-op handler leave
every record whole; the client's refusals and errors each come once and
again. Mutations each fail a test: the newest-key jump left out (GOP log,
server, stress), a viewer started at the log's end rather than its key
(GOP log, server, stress), a chunk abandoned mid-way (server), the
low-water mark not set (the socket test, reading it back; on macOS the
loopback tests pass without it, as there it gates `poll` alone), the stall
limit never reached (server), `EINTR` not retried (socket, server), the log
read outside the lock (server and stress under ThreadSanitizer, stress
without it), and a lookup of any family kept unfiltered (socket). Taking a
chunk before the socket polls writable fails none: it sends one stale
chunk, which is whole and in order.
`serve_stream`'s loop, on loopback, replays Rafa2 at 29.99 records a second
and 6.84 Mbit/s for two passes, seq unbroken. Over the VPN (the lab's
`serve_stream`, built with g++ 15.2, to a client on the laptop): a fast
reader got Rafa2 at 30.00 records a second and 6.9 Mbit/s of payload, its
session 43 to 48 ms and its first record, a key, 83 to 92 ms after the
connect by address (55 to 74 and 92 to 117 ms by name); two at once got
the same, though in one of two runs each skipped once to a key in its first
2 s; a reader taking one record per 200 ms for 20 s stayed within 4 s of
live, each of its 20 jumps to a key, and reading fast again drained
0.28 MB of stale records in 21 ms and was live 0.31 s later; five
reconnects each began at a key.

### 2026-10-21 — `MeshIntegrator` finds each voxel's nearest triangle with an exact early exit: a triangle is ruled out only by a lower bound, less a rounding margin of its own, strictly above the voxel's limit.

A mesh sequence converts each frame with `MeshIntegrator`, whose write
measured every voxel against every triangle of its block's bin. In the
16-voxel band the mesh templates move to with the K56 point (#262), Rafa2
at 5 mm bins 4.65 M (triangle, block) pairs a frame, 3.9 times the 8-voxel
band's 1.21 M, and the write took 29.57 / 27.17 ms (host / device) a
frame. Almost none of a bin can be a given voxel's nearest triangle.

- **A workgroup per block.** It sorts the bin into 256 buckets of each
  triangle's support-plane bound at the block's centre (a counting sort in
  shared memory) and stages it 64 triangles at a time with each triangle's
  bounds. Each brick of 32 voxels keeps the staged triangles that a support
  plane or a ball does not rule out anywhere in its box, and each voxel
  measures those its own support-plane or cylinder bound does not. A voxel
  stops at the first chunk whose bucket floor, less its distance from the
  centre, rules out everything after. Past its first 2,048 entries a bin
  goes on unsorted, every voxel scanning it to the end.
- **Exact.** A triangle is ruled out only when a bound, taken less
  2^-14 (s + band) -- s its largest coordinate or the block centre's, band
  `trunc_dist` plus the block's half-diagonal -- is positive with its square
  strictly above the voxel's best d^2 or `trunc_dist`^2. By the count in
  `mesh_integrate.comp`, that margin is about ten times what the closest
  point and the bounds can round by, so such a triangle can neither beat nor
  tie the best, nor be observed. Every triangle measured goes through the
  same routine on the same floats, and the minimum over (d^2, index) does not
  depend on the order: the bytes are the brute force's.
- **Limits.** Shared memory is 16,132 bytes, inside the 16,384 every Vulkan
  device offers; the shader's preprocessor refuses to compile past it.
  `kMaxVoxelsPerBlock` (16^3) bounds a workgroup's rounds of 256 voxels at
  16, refused past it where a dispatch's thread limit used to refuse.

**Measured** (M5 Max, MoltenVK, Release; Rafa2 frames 1–150 at 5 mm, the
`"mesh integrate"` row's mean a frame over three runs each, interleaved, each
run on an idle GPU):

| band | origin/main | this |
| --- | --- | --- |
| 8 voxels (`codec_mesh_sequence.json` before #262) | 8.55 / 6.89 ms | 3.65 / 2.21 ms |
| 16 voxels (#262's K56 template) | 29.57 / 27.17 ms | 7.95 / 5.95 ms |

origin/main's three 16-voxel runs gave 25.26 to 28.20 ms on the device (#262
measured 25.52), this branch's 5.94 to 5.96.

**Validation.** `codec_mesh` on the 8-voxel template writes origin/main's
`.vrst` byte for byte (4,272,320 B), and every frame's `tsdf` and `weight`
hash the same as origin/main's in both bands, signed and shell.
`recon_tsdf_mesh_integrate_exact` compares every voxel's bits with a brute
force over the same bins, rebuilt by replaying the binning's work items
through its own GLSL, on 19 fixtures. Planted bugs each fail it: a margin of
0 (15 fixtures), the tie order reversed (11), the unsorted tail skipped (2),
and `>=` in place of both of the rule's `>` (the fan at zero distance, whose
faces all measure 0 at the origin's voxel and whose lowest index signs it
-0). Two do not, as the argument says: `>=` on the square alone, since the
margin already keeps ties, and the margin of the centre's alone, which only a
triangle hundreds of metres long through the blocks near the origin would
tell apart. At 1/128 of the margin every fixture still passes; at 1/1024 four
fail.

**Not done**: the conversion's smaller costs around the write, in DESIGN's
Next work.

### 2026-10-21 — A true signed distance codes without the observed mask: `EncoderConfig::true_sdf` cuts each corner farther than √3 voxels from zero of a cell whose sign changes, and codes only the blocks that hold surface, whole; `codec_mesh` alone sets it, in a band of at least 15 voxels, so fused depth keeps its mask, and the mesh templates convert in 16 voxels and code at K 56 (amends the 2026-09-27 encoder, 2026-10-01 codec-review and 2026-10-14 codec-mesh entries).

The mask is 43% of an I-frame's bytes and 51% of a P-frame's on Rafa2 (the
2026-10-21 quantization study), and it says which voxels a fused grid
observed. A mesh's distance is exact wherever it is computed, so its blocks
need no mask, only a band wide enough to observe them. The owner's rule:
code without the mask only where a true signed distance exists; masks stay
the default, and multi-view RGB-D capture keeps the true mask.

- **One switch, in the encoder.** `EncoderConfig::true_sdf`, false by
  default, which codes as before, byte for byte. No scenario or config key
  sets it, so no RGB-D scenario can. Its doc states the precondition, a true
  distance. `CodecParams` is untouched: it is in every header and the
  decoder's checks, and the rule is neither.
- **The cut.** In each cell whose 8 voxels, across block boundaries, are observed
  and whose sign changes, every corner with |tsdf| > √3 h + 1e-6 m is coded
  unobserved. A distance is 1-Lipschitz, so a cell a real surface crosses
  has every corner within √3 h of it: only sign errors are cut, such as the
  sheets the closest-face sign leaves past a hole's rim or a sharp edge (the
  2026-09-27 mesh entry). One pass, from the field as it stands, so no cut
  changes which cells another tests.
- **The surface blocks.** Only the blocks holding a corner of a cell whose 8
  voxels are observed and uncut, and that changes sign or has a corner
  within h of zero, are coded. Without the margin a hole opened in the
  entropy study (frame 1 in a 4-voxel band: its coverage maximum, 1.75 mm,
  became 3.55). A coded voxel then lies within (1 + 8√3) h = 14.86 h of the
  zero level: a near cell's corner within h, the cell's others within √3 h
  of it, the block's within 7√3 h. So a band of `kTrueSdfBandVoxels` (15)
  observes every one, and every mask is all ones but the cuts; past a
  narrower band, voxels stay unobserved.
- **The true distance comes from the conversion.** The encoder cannot drop
  the mask alone: over a narrow band an all-ones mask codes the forward's
  fill as data (the study, frames 1–30: −79.7% BD-rate, against −82.7% for
  the true distance and −84.3% with the surface blocks).
- **On the device, in `"  ..observed"`'s batch** (DESIGN has the passes).
  The cut set and the keep flags are encoder scratch, so the caller's grid
  is never written, and only the count crosses, as before. The mesher's
  neighbourhood probe and corner addressing moved to
  `volume/shaders/block_neighbourhood.glsl`, since codec may not include
  mesh.
- **`codec_mesh` sets it, in code,** the one caller, and refuses before any
  GPU work a narrower band, with the shell's half-thickness added in shell
  mode, whose zero level lies that far from the mesh. So a mesh stream
  always codes without the mask, and nothing else does.
- **The templates.** `codec_mesh_sequence.json` converts in 16 voxels
  (`fusion.truncation_m` 0.08), the smallest power-of-two multiple of 8
  voxels past 15, and codes at K 56, scale 0.040877, period 30: the study's
  measured point, scale 0.081754 at 0.04 m. The scale is a fraction of the
  band, so twice the band takes half the scale for the same 3.27 mm step,
  and with a power-of-two ratio the quantized coefficients are the study's
  own. The study's advice to keep 8 voxels is about the band its candidate
  blocks came from, which changes no kept block; the band a frame codes in
  is the conversion's. `codec_mesh.json`, one frame, moves to the same band,
  K and scale.
- **The stream records nothing new.** The format is frozen, and no consumer
  reads a decoded weight as "seen": a decoded grid is meshed, never fused
  into. A decoded voxel of a coded block has weight 1 unless cut, and the
  header's `trunc_dist` is the band. A mesh stream is marked already, by its
  writer, `codec_mesh`, and the Geometry track's placement; a flag waits for
  a consumer that needs one.
- **Not taken:** an all-ones mask over the fill (above); a second, narrower
  conversion for the candidate blocks, which the kept set does not need; a
  mask flag in the stream; a separate switch for the cut, without which the
  conversion's curtains are coded whole.

**Measured** by the study's harness on Rafa2 frames 1–150: its total chamfer
to the input over the 150 frames, at a 0.2 m reach, and its face jump, one
method for every row.

| Stream | Mbit/s | Total chamfer | Face jump |
|---|---|---|---|
| The old template (masked; K 64, scale 0.2, 8 voxels) | 6.82 | 0.543 mm | 0.920 mm |
| The K56 template | 1.99 | 0.294 mm | 0.466 mm |
| The K56 template and the study's LF deblock, not shipped | 1.99 | 0.262 mm | 0.193 mm |

The K56 template's largest GoP is 1.74x its mean (3.47 Mbit/s, frames
121–150) and its largest I-frame 29,655 B; at K 120 (6.78 Mbit/s) the
largest GoP is 1.66x its mean, and the old template's 1.47x.

**This PR's streams carry those numbers over by identity.** `codec_mesh` on
the K56 template writes 1,246,265 B (1.994 Mbit/s) against the study's
1,246,126, with its block count on all 150 frames and its bytes on frames
1–73. Decoded by one decoder, the two streams' grids are bit-identical in
metres on 138 frames, frames 1–73 among them. On the 12 others, between 75
and 141, every difference is one of 53 voxels this PR cuts and the study did
not, or the fill of their blocks (at most 2.1 mm). Each of the 53 lies
33.9 mm or more from the surface, by a near-lossless code of the same
conversion (masked, K 512, every frame intra); 6 more frames differ in bytes
alone, P-frames against a reference that differed. This PR's cut examines
every cell of voxels observed in 16 voxels, the study's only the cells of
blocks it observed in 8, and the two differ past 31 mm alone, in no cell
that keeps a block. At K 120 and scale 0.007036 the same 12 frames differ
the same way, by the same 53 voxels, at 4,236,145 B against 4,235,824.

**Validation.** Apple M5 Max, MoltenVK, Release with warnings as errors,
Assimp, FFmpeg and Orbbec, core `b992f12`: the 111 tests pass under the
validation layer and synchronization validation, the goldens unchanged.
`recon_codec_encoder` holds `true_sdf` frames to the CPU oracle, whose
blocks and cut set come from the study's rule on the host, on a curtain (a
sphere's distance negated in a slab 20 mm out) and a speck only the near
rule keeps, at three segment sizes: cuts across block boundaries, partial masks,
the same bytes whatever the hash order, heap growth between frames, and no
stale cut after a cleared grid; on two spheres, nothing is cut.
`recon_codec_dct` holds the passes to the same host rule in one dispatch and
in chunks of 3 and 7 blocks, and `recon_example_config` holds the band check
at every voxel size from 1 to 50 mm. Planted bugs each failed their test: no
cut pass (`true_sdf_oracle_case`, `curtain_case`), the anchor block alone
kept or every observed block appended (`surface_blocks_case`), a forward
that ignores the cut (`cut_case`), no margin (`true_sdf_oracle_case`), no
zeroing of the cut set or the keep flags (`stale_cut_case`), passes that
rerun the first chunk (`surface_chunks_case`), and an absolute tolerance on
the band (`recon_example_config`). By hand: `codec_replica` on room0 writes
origin/main's `.vrst` byte for byte (8,361,488 B) and its CSV's frame, type,
bytes and blocks on all 400 frames; an untouched origin/main build decodes
the K56 and K120 streams (`decode_growing`) to this build's grids on all 150
frames. Time, each run gated on an idle GPU with no other test or player
running, the median of 3 runs each, host / device ms, origin/main on its
template against this build on the new one: `codec encode` takes
2.66 / 1.60 an I-frame and 2.41 / 1.37 a P-frame against 2.48 / 1.37 and
2.23 / 1.13, the fewer blocks outweighing the two passes. The mesh
conversion in each band is timed in the `MeshIntegrator` entry, above. Not
run: the RTX 5090, ASan and UBSan (CI runs them).

### 2026-10-21 — The decoder deblocks on request: a runtime switch, off by default, runs the study's step-keyed line fit and Gaussian on the device in the decode's last submit, keeping each block with an unobserved voxel as decoded, and a paused player rebuilds its frame from the reference (amends the 2026-10-13 decoder entry).

*Amended 2026-10-23 (the owner), superseding the 2026-10-22 frame gate and
that day's earlier amendment, which skipped a frame once more than half its
blocks were masked (#273):* the deblock skips no frame; it falls back to the
exact decode block by block. A masked block, one holding an unobserved
voxel, is never written. A line is fitted across a face only when both its
blocks are fully observed, so neither side of a face with a masked block is
line-fitted. The Gaussian runs on fully observed blocks alone; a masked
neighbour's observed voxels keep their decoded values through all three
passes and enter its taps as such, its unobserved ones dropping out as
before. Why: a frame gate is all or nothing, and the frames sit at both
ends -- the K56 template's 71 frames with a cut have 1–29 masked blocks of
about 1,900, the live room captures 99.8–100% -- while the room evaluation
below put the unguarded filter's damage at mask edges and in thin parts,
where the masked blocks are. Keeping exactly those blocks exact keeps the
filter wherever a block is fully observed, so a true-SDF mesh frame is
smoothed but for its cut blocks, and a live frame decodes as with the
switch off but for its few fully observed blocks; the player's switch
defaults on, the library's stays off. `DeblockInfo` keeps the rule's
filter and T and gains the frame's `block_count` and `exact_blocks`, the
blocks the fallback kept at their exact decode, those holding an unobserved
voxel (a fully observed block the filter happens to leave unchanged is not
counted), so a fully masked frame, which decodes as off, says so. On the
device a first dispatch, an invocation a block, flags each fully observed
block by slot and counts the masked blocks, whose count the decode's last
readback carries; the later kernels return at once for a masked block, by
its own mask. The inverse's count, the skip word and the kernels'
comparison are gone. The amendment's validation closes this entry.

The 2026-10-21 quantization study found the steps quantization leaves at
block boundaries the largest error a decoded mesh shows, and a decoder-side
filter that removes most of them for no bits: on Rafa2 the K56 template's
face jump falls from 0.466 to 0.193 mm under its LF filter, and its chamfer
from 0.294 to 0.262 mm (the true-SDF entry, above). Its prototype ran the
filters on the GPU in 0.15–0.52 ms a frame at 30 fps on the M5 Max, in a
submit and fence of its own. The owner wants it in the player, toggled
live.

- **A switch on the `Decoder`, off by default, read at each decode.**
  `set_deblock` / `deblock()`. Off is the exact decode, which the decoder's
  contract, the bit-for-bit oracle and inter tests and the examples'
  quality numbers rest on; the player turns it on. The filter is out of
  loop -- a P-frame decodes from its reference's coefficients, never from a
  grid -- so it may change between any two frames; a construction-time
  option could not be toggled live. No `DecoderConfig`: nothing else needs
  one.
- **The filter from the header alone.** K and T, the step of DC and the
  zigzag's next three coefficients at their basis peaks, pick a row of
  `kDeblockRule`, the study's rule A for its rounding streams; T takes the
  study's arithmetic, so its thresholds fall where it measured them. *To
  confirm:* K 36–55, which the study did not fit, takes the rows of K 56
  and above. An RDOQ stream would need a signal the frozen header lacks;
  none ships.
- **Not where a voxel is unobserved** (the owner, 2026-10-22, on the live
  rig's result below; the 2026-10-23 amendment, above, makes it a block's
  rule). The 2026-10-22 decision skipped every frame with an unobserved
  voxel, so the K56 template's frames with a cut were skipped too, against
  the decision's premise that true-SDF streams stay as they were. Finer
  than the block, at 1 cm, a host emulation leaving every voxel within a
  voxel of an unobserved one exact improved chamfer by 2.2% (the filter
  alone, 1.9%) and left 69% fewer of the input's vertices 10 mm or more
  from the decoded surface, though it removed only 24% fewer triangles.
- **In the decode's last submit.** The inverse records without submitting
  (`DctTransform::record_inverse`), the deblock follows it, and the
  inverse's reject count is read back last, so no copy splits the compute.
  Nothing of its own but the count of the blocks it kept, read back in the
  same batch just before the inverse's reject count: no submit, fence,
  compaction or indirect dispatch; the frame's list and its count come from
  the decode.
- **The kernels,** a 64-lane workgroup a block (Vulkan guarantees 128,
  WebGPU 256). The line passes run in place, a dispatch an axis: a line
  reads three voxels a side and writes two, so a voxel lies on one line of
  a pass, read before it is written. The Gaussian is fused, one dispatch
  over the block and the 1-voxel halo of its 26 neighbours in 4 KB of
  shared memory, each pass's outputs held in registers until its inputs
  are read; it writes the interior, which no other workgroup reads, in
  place, and the shell into a snapshot packed to its 296 voxels a block,
  by list position, which a store dispatch copies back. LF is 4 dispatches
  (the flags, then an axis each), LG and R 6, against the prototype's 4
  and 7. Whether a block is fully observed comes from its own mask words
  or, for a neighbour, its flag; a masked neighbour's observed voxels come
  from its weights; every block is found by coordinate, never by a list's
  `ptr`.
- **Stamps** follow what it writes: with it on they may name a block whose
  values end as they were, never miss one that changed.
- **Its device time is part of `"codec decode"`'s** (*to confirm*): one
  submit cannot split its host half, and the player sums device halves.
- **A paused frame is rebuilt from the reference.** The store holds the
  frame's list, masks, quantized coefficients and header, so `rebuild`
  runs the inverse and the deblock again under the switch as it is now,
  with no bitstream read and no GoP re-decoded, once the grid diff has
  confirmed the grid holds exactly the frame's blocks: a block it lacked
  would be no neighbour, and one too many a neighbour the frame lacks.
- **The filter is reported by name** (`DeblockInfo`, with T): a filter
  picker for A/B comparisons would need a public enum, which waits until
  one is asked for.
- **Not taken:** a construction-time option; a per-frame neighbour table,
  which saves no probe (the Gaussian needs its 26 neighbours either way);
  a copy of the exact decode to compare against at the end, whose cost
  buys only that a voxel moved and moved back keeps its bits, not 1 ulp
  of metres; per-axis ping-pong snapshots (more dispatches and bytes); a
  whole-block snapshot with the interior in place, which saves no memory
  sector over the shell's.

**Validation.** Apple M5 Max, MoltenVK, Release with warnings as errors,
core `b992f12`: the 101 tests pass under the validation layer and
synchronization validation, and the four codec tests it touches in Debug.
So do the 100 of a Linux build, GCC 15.2 with no Assimp, FFmpeg, CUDA or
Orbbec, on an NVIDIA RTX 5090 (Ubuntu 26.04, driver 615.71), its ICD the
loader's only one. `recon_codec_deblock` holds the
kernels to a CPU oracle ported from the study's filters, which reads a
snapshot of each pass whatever order the workgroups run in, within 2e-6 of
the band: LF, LG, R and each stage alone over partial masks, missing face,
edge and corner neighbours and an unobserved plane; lines set by hand for
the 3-sample candidacy, a sample at the clamp's kink and an ulp below it,
the [-1, 1] clamp and the clip; an unobserved Gaussian tap against a hand
value, a spike losing three clips, and a corner reaching its diagonal block.
Every workgroup alone, after the last one's writes, in list order and in
reverse, and in chunks of 2 and 3, gives the one-dispatch run's bits, so a
voxel one workgroup writes and another reads in a pass fails in one order or
the other. The fixtures' blocks sit past slot 2n, their lists carry other
blocks' `ptr`s, and their masks are by list position. The Decoder's cases:
off is the exact decode; on, every mask-free frame is the oracle's filter of
the exact decode under the row the rule picks (none, LF, LG, R by T, R at
K 20), and not the next row's; I- and P-frames toggled between frames, and
four filters in one decoder over growing frames; `rebuild` equal to the
decode with the switch so, its refusals leaving the grid as it was, and the
next P-frame unchanged by it; a block only the deblock writes stamped; the
rows unchanged. 26 bugs planted by hand each failed their case, among them
a shell voxel written in place on either face (the serialized orders), p
and q swapped, candidacy over two samples, the kink at `>`, either clip or
the clamp dropped, a 6-face halo, no chunk base, the snapshot or the masks
by slot, a trusted `ptr`, the axes reversed, the deblock before the
inverse, an ignored switch, `rebuild` without the diff, either stamp
dropped, and a deblock on by default (`recon_codec_inter` and
`recon_codec_deblock`; `recon_codec_decoder`'s bounds are loose enough to
pass it).

By hand, against the study's own CPU code, its T computed its own way, on
Rafa2 frames 1–150 of the prototype's four streams and the K56 template's,
every frame: the rule's filter as the decoder ran it with the switch on
before the gate, and LF, LG and R each forced on the exact decode. All 20
pass the study's rule: every
voxel within 0.030 µm (none over 1 µm), the marching-cubes triangle count
equal on every frame (five sign flips, each within 0.00012 µm), no weight,
unobserved voxel or block changed. T matched the study's to the bit on
every frame, and so did the row. The table is the M5 Max's; on the RTX 5090
all 20 pass too, every voxel within 0.030 µm, the triangle counts equal and
T and the row the study's on every frame.

| Stream | T | Rule | Largest difference, the rule's / LF / LG / R |
|---|---|---|---|
| stock (K 64, s 0.2) | 0.490 mm | LG | 0.019 / 0.011 / 0.019 / 0.011 µm |
| stk682a | 0.021 mm | none | exact / 0.007 / 0.026 / 0.015 µm |
| stk2 | 0.112 mm | LF | 0.011 / 0.011 / 0.030 / 0.015 µm |
| stk1 | 0.335 mm | LG | 0.030 / 0.011 / 0.030 / 0.022 µm |
| K56 template | 0.200 mm | LF | 0.011 / 0.011 / 0.030 / 0.015 µm |

**Time,** measured before any gate or fallback, every block filtered,
stock's masked frames included (the fallback's own cost follows the
room-scale result).
Release, on the M5 Max and on the RTX 5090 above (with an Intel
Core Ultra 9 285K; clocks not locked), each unit run only once the GPU had
been idle for 3 s with no other test, player or scratch GPU binary running
(nor, on the RTX 5090's host, a CI job or another compute app) and other
processes under 400% CPU, and re-run if such a binary, job or app, or that
CPU load, appeared while it ran. The run's own processes are exempt
throughout, and on the M5 Max the CPU bound also skips desktop apps such as
WindowServer and the browser. The chain's device interval, barriers
included, back to back: 20 chains recorded into one batch over each frame's
blocks, against 20 over one block (the same dispatches and host encoding),
the wall difference over 20; frames 1–150, ms, mean / p95 on the M5 Max. On
the RTX 5090 a frame's interval is within the method's noise (11 of its 15
cells hold a negative one), so the mean alone.

| Stream (blocks a frame) | M5 Max: LF | LG | R | RTX 5090: LF | LG | R |
|---|---|---|---|---|---|---|
| stock (3,426) | 0.043 / 0.055 | 0.107 / 0.120 | 0.109 / 0.123 | 0.009 | 0.029 | 0.037 |
| stk682a (1,925) | 0.021 / 0.030 | 0.053 / 0.074 | 0.050 / 0.075 | 0.005 | 0.013 | 0.013 |
| stk2 (1,925) | 0.020 / 0.029 | 0.053 / 0.077 | 0.052 / 0.077 | 0.005 | 0.014 | 0.014 |
| stk1 (1,925) | 0.020 / 0.029 | 0.053 / 0.076 | 0.051 / 0.072 | 0.005 | 0.010 | 0.015 |
| K56 template (1,924) | 0.021 / 0.029 | 0.052 / 0.076 | 0.052 / 0.076 | 0.009 | 0.013 | 0.013 |

The prototype's, back to back on the same streams on the M5 Max, one
timestamp span around its whole chain (which this repeat method matched
within 7%): LF 0.052–0.097 ms and LG 0.099–0.186 ms. Building the three
pipelines, before the fallback's flags kernel made four, takes 2.5 ms
(4.3 ms the first time) of `Decoder::create`'s 11.8 ms on the M5 Max, and
0.33 ms (1.8 ms the first time, the driver's shader cache warm) of 1.3 ms
on the RTX 5090.

Paced at 30 fps, as a player runs it: 150 frames, each `decode_growing`
one host interval, with the deblock off and on, each with and without
`StageMetrics`; every mode on a fresh decoder and grid after an unrecorded
warm-up, the modes' order rotated by round. On − off is paired by frame and
round, mean ± a 95% interval, in ms; "device" is the `"codec decode"`
row's device half, and the wall is without metrics unless marked. Both
machines pool rounds 0 and 1 (stk2 on the M5 Max round 1 alone); in each,
off with metrics runs just before on with metrics. The RTX 5090's round 2,
which ran off with metrics first, read that mode's device row 0.03–0.09 ms
above rounds 0 and 1 on every stream and is left out. stk682a is the
control: its rule picks no filter, so on and off do the same work.

| Stream (the filter run) | M5 Max: device | wall | wall, metrics on | RTX 5090: device | wall | wall, metrics on |
|---|---|---|---|---|---|---|
| stock (LG) | +0.12 ± 0.27 | +0.60 ± 0.27 | +0.47 ± 0.29 | +0.054 ± 0.001 | +0.007 ± 0.054 | −0.089 ± 0.054 |
| stk682a (none) | +0.24 ± 0.30 | +0.17 ± 0.32 | +0.55 ± 0.33 | +0.000 ± 0.001 | −0.093 ± 0.056 | +0.048 ± 0.054 |
| stk2 (LF) | +0.12 ± 0.02 | +1.25 ± 0.19 | +0.40 ± 0.15 | +0.017 ± 0.001 | +0.070 ± 0.055 | +0.021 ± 0.053 |
| stk1 (LG) | +0.29 ± 0.03 | +0.54 ± 0.11 | +0.13 ± 0.12 | +0.034 ± 0.001 | −0.070 ± 0.051 | −0.206 ± 0.049 |
| K56 template (LF) | +0.16 ± 0.05 | +0.13 ± 0.10 | +0.18 ± 0.12 | +0.015 ± 0.000 | +0.108 ± 0.051 | −0.045 ± 0.078 |

With the deblock off, the device row averages 3.2–5.4 ms a frame on the M5
Max and 0.8–1.8 ms on the RTX 5090, the wall 5.3–8.4 and 2.3–3.4 ms. The
wall resolves no deblock on either machine: the control moves it by up to
0.55 ms on the M5 Max and by −0.09 ms on the RTX 5090, whose walls move
−0.21 to +0.11 ms, of either sign, so the modes' order and the GPU's power
state move it more than the filter does. On the M5 Max the device row
resolves LF at 0.12–0.16 ms and LG at 0.29 ms on the 1,925-block streams,
five to eight times the back-to-back interval; on stock and the control its
noise is ±0.3 ms. On the RTX 5090 it resolves every filter: +0.054 (stock,
LG), +0.034 (stk1, LG), +0.017 and +0.015 (LF), each within 0.002 ms of the
chain's span sum below, and the control +0.000; its GPU sat in P0 for a
third of the 1 Hz samples during these units, and mostly in P5 or P8
otherwise. There the rule's chain (LG for stk682a), recorded on its own
after each paced exact decode, sums over its dispatches' timestamp spans to
0.015–0.016 ms for LF, 0.034 ms for LG at 1,925 blocks and 0.054 ms at
3,426 (mean over 150 frames, the GPU mostly in P5; not run on the M5 Max).

**On the live rig, room scale,** the reason for the gate and then the
fallback. Two captures of
the four-camera Orbbec rig, coded by `fuse_orbbec` under the rig's capture
config (K 64, uniform scale 0.2, intra period 30, the default truncation of
4 voxels) and filtered as before the gate; every 10th frame (31 a capture,
11 I and 20 P) judged against the grid the encoder read, at a 0.2 m reach,
the face and interior jumps the study's.

| Capture | Blocks a frame | T (filter) | RMS error near the surface | Face / interior jump | Face jump, on vs off |
|---|---|---|---|---|---|
| 1 cm voxels, 305 frames | 21,168 | 0.490 mm (LG) | 3.25 mm, 6.6 T | 0.98 | −10.5% |
| 5 mm voxels, 302 frames | 82,030 | 0.245 mm (LF) | 2.08 mm, 8.5 T | 0.94 | −4.4% |
| Rafa2 stock, for scale | 3,426 | 0.490 mm (LG) | 0.55 mm, 1.1 T | 2.43 | −79% (LF −73%) |

T halves at 5 mm only because the truncation is 4 voxels. The room's error
lies inside blocks, not at their faces, so the filter T sizes is a light
denoiser there. At 1 cm LG improves accuracy (−8.7%), coverage (−3.1%) and
chamfer (−1.9%) on every frame, yet 53k more of the input's vertices end
10 mm or more from the decoded surface (+8.9% over off), 93% of them within
a voxel of an unobserved voxel, and parts 3 voxels thin or less lose 1.0%
of theirs. About half of that count is surface the sign flips remove
(12.7k fewer triangles a frame), the rest vertices moved past the 10 mm
threshold, one voxel at 1 cm. At 5 mm LF is worse than off in coverage
(mean +0.8%, p95 +2.5%) and chamfer (+0.8%) on all 31 frames, mostly by
eroding parts 1.5 cm thin or less, and neither LG, R nor a host-emulated
mask guard beats off there. Paced at 30 fps on the lab's RTX 4090, rounds 0
and 1 pooled as above, the filter adds +1.49 ± 0.15 ms of decode wall at
1 cm, its rounds +0.95 and +2.03 (device +1.26 ± 0.14), and +1.40 ± 0.28 ms
at 5 mm, +1.15 and +1.65 (device +1.16 ± 0.27), the device time 3.2 and 1.8
times its chain back to back (LG 0.39 ms, LF 0.64 ms), the GPU clocking down
between frames. Rafa2 stock on that GPU, an LG reference rather than a
control, adds +0.17 ± 0.09 ms (device +0.12 ± 0.03); no no-filter control
ran there.

**The fallback's cost.** Its own work runs only with the switch on: two
fills, a dispatch an invocation a block that probes the fully observed
blocks alone, and a word read back; a masked block's workgroups return on
its own mask words before any probe. On the lab's RTX 4090, the
`"codec decode"` row's device half with the switch on minus off, paired by
frame back to back with each grid read back between decodes (indicative,
not PERF's method; frames 2 on): the K56 template +0.025 ms against main's
(221ac91f, the same LF chain) +0.022; the 1 cm room +0.01 ± 0.07 ms against
main's, which skips every room frame, +0.05 ± 0.06; the 5 mm room
+0.58 ± 0.66 against +0.15 ± 0.57 (medians +0.28 and +0.14), so not
resolved at room scale.

**The 2026-10-23 amendment's validation.** The lab's RTX 4090 (Ubuntu
26.04, driver 615.71.10, GCC 15.2), Release with warnings as errors and the
CI ubuntu-26.04 leg's options: the 95 tests pass under the validation layer
(1.4.341) and synchronization validation. `recon_codec_deblock` holds the
kernels to the oracle, which implements the same rule. On a 5^3 cube with
one block masked by a single unobserved voxel, under LF, LG and R, in list
order and reversed, in one dispatch and a workgroup a dispatch (the masked
block in the flags kernel's second workgroup), that block keeps its bits
and its stamp, the rest is the oracle's filter, the same bits every run,
and against the filter without the fallback the grid differs only within
2 voxels of the block (5 with a Gaussian); under LF the voxels 1 and 2 from
each of its faces that only a line across that face reaches keep their
bits. A single unobserved voxel anywhere in either of two blocks leaves
every line of their face unfitted, both sides to the bit; a masked
neighbour's observed tap enters the Gaussian as decoded and its unobserved
one drops out (hand values), and a y pass reads a masked neighbour's spike
as decoded, not as the x pass would have left it; every block masked, the
grid and its stamps stay as they were, all 125 counted; a block the frame
does not list reads as masked. Decoded, each frame is the oracle's filter
of its exact decode, its masked blocks kept to the bit and their stamps the
exact decode's, reported with the rule's filter and its masked blocks as
`exact_blocks`, and rebuilt to the same: mask-free frames as before, the
masked sphere chains (84–86% of their blocks masked), interleaved with
mask-free frames, a frame of 250 blocks with 1 masked block (a single
voxel), 125 and all 250, the last the exact decode, and the true-SDF
curtain, 20 of 130 blocks cut. 14 planted bugs each failed it: a masked
block written by the Gaussian, the store or the line pass from either side;
p moved beside a masked neighbour, q kept; a masked neighbour's voxels
filtered in the Gaussian's halo; the flags never zeroed, ignoring the chunk
base, or a block read by its first mask word; the count of full blocks, of
masked words, never zeroed, or 0 with no filter; and the decoder reporting
no block kept. With the cases before the cube left out, the five kernel
bugs of the fallback and of its count each failed the cube, and with every
kernel case left out, they and the decoder's report each failed the
decoder's cases.

On the same GPU, every frame decoded with the switch on and off, each grid
hashed, against main (221ac91f) and the unguarded deblock (27bfada4, #264
before its gate): the exact decode is main's on every frame of the K56
template and both room captures. The K56 template's 79 mask-free frames
decode on to main's hash; its 71 cut frames keep 1–29 blocks exact
(363 in all, each frame's masked blocks), and per voxel they differ from
the unguarded deblock only in those blocks (131,122 voxels) and within
2 voxels of them (158,651), none farther. The rooms leave few blocks
eligible, fully observed: at 1 cm 0–2 of 17,592–21,797 a frame, none on
285 of 305 frames, which decode as off; at 5 mm 1–158 of 64,382–85,017,
0.12% on average. On every frame of the three the masked blocks hold the
exact decode, `exact_blocks` counts them, and `rebuild` equals the decode.

**Room quality.** The 2026-10-22 evaluation, re-run on nunga's second
RTX 4090 for off, the fallback and the unguarded deblock (27bfada4): every
10th frame of both captures (31 each) against the encoder's input S_enc at
a 0.2 m reach, the exact decode identical in both runs. At 1 cm (LG) the
fallback decodes as off: 30 of the 31 frames equal it in every measure,
the 31st within 0.0003 mm of it in each distance (coverage p95 the
farthest), and the triangle count and the S_enc vertices more than a voxel
(10 mm) from the decoded surface are off's on all 31. The unguarded
deblock, as before, adds 53,368 such vertices (26,373 removed by sign
flips, 26,995 moved past the threshold), 1.0% of thin parts' vertices and
93% at mask edges, and 12,743 fewer triangles a frame, for chamfer −1.9%.
At 5 mm (LF) the fallback adds 169 vertices more
than a voxel (5 mm) from the surface over the 31 frames, 58 by sign flips
and 111 past the threshold, 166 of them in thin parts, and 17 fewer
triangles a frame of 5.24 million; chamfer moves +0.000002 mm on average
and +0.00003 mm at most. The unguarded deblock adds 963,436 (475,814 and
487,622), 1.2% of thin parts' vertices, and 66,993 fewer triangles a frame,
for chamfer +0.8%. So on live room frames the fallback is the exact decode
at 1 cm and erodes a trace at 5 mm, about 1/5,700 of the unguarded
filter's loss, in the few fully observed blocks it filters; nor does it
gain anything there that these measures resolve.

### 2026-10-22 — StreamClient reports progress before a whole record is ready.

The player waits on its consumer's take when a decoded frame cannot yet be
published. A zero-wait `StreamClient::next` takes at most one receive, and
can return false while more of a partial record is already in the socket.
Treating that false as idle adds a consumer wait per receive, slowing the
drain even when the sender and decoder can keep up.

The four-argument overload adds an optional `bool* progressed`. Each call
resets it, then sets it after a successful lookup/connect step, bytes
received, or a record returned from the parser. Once true it stays true for
that call, including when the next step is idle. Callers can drain with
zero-wait calls while it reports progress, checking their controls between
calls, then wait on those controls once no progress was possible. On error
it covers only earlier successful steps; the error remains terminal. The
three-argument overload remains a forwarding symbol, so existing callers
retain their API and behavior.

The loopback regression sends a 512 KiB record in pieces, distinguishes
partial progress from an idle call, and checks records sent together,
connect progress, sticky errors and an allocation failure while copying
one. The player's consumer-wakeup policy stays with the player.

### 2026-10-24 — A frame decodes and meshes in the caller's batch: every count stays on the device, the map's rounds (K = 3), the extract and the decoder gather their results into ring slots, and the blocking calls are one submit (amends the 2026-09-28 device-memory, 2026-10-13 coordinate and 2026-10-23 breakdown entries).

**Why.** The owner asked for timing that is "comprehensive and correct" with
"no unnecessary copy or waits", "everything ... GPU with zero-copy", on
"regular DRAM/VRAM split devices" as well as unified memory, and "if the new
design is beneficial everywhere, we do not need the older pipelines". A
rafa2 P-frame took 6 submits, 6 blocking waits, 6 readback buffers and 7
staging buffers (M0, on the RTX 4090 and the M5 Max): every count a step
needed crossed to the host before the next step was recorded. Now the whole
frame, decode and mesh, is one batch the caller records and submits, and
the host reads the frame's results once it has run. The old per-frame paths
are gone; `decode`, `decode_growing`, `extract_device`, `allocate`,
`remove` and the vector reads are record, one submit and finish.

**The map's rounds run on the device.** `record_allocate` and
`record_remove` record K rounds over one list. A one-thread kernel
(`hash_rounds.comp`, one instance a kind) sizes each round from the tally
the last one left: a round runs only while the last lost bucket-lock races
and met no capacity limit, as the host loop did. After the last round it
writes the outcome whole into a result slot and closes the caller's gate on
any failure, so what is gated on it after the rounds runs nothing. `settle`
reads the slot and refreshes the heap's host copy. K = 3
(`kRecordedRounds` says why). The blocking calls repeat record, submit and
settle while lost
races alone remain, up to the 10 rounds they always had: one submit for
every K rounds, and no readback buffer.

**The extract sizes itself.** `record_extract` records the compaction, a
one-thread kernel that sizes the marching cubes from the compaction's count
(zero groups behind a closed gate) and resets the slot's draw command, the
marching cubes, and a finalize that gathers the results and zeroes the
command's `indexCount` when the surface outgrew the slot. `finish` reads
them and, on an overflow, grows the slot and meshes again in a blocking
batch of its own (`"  ..refit"`, a breakdown of the call's
`"marching cubes"`). The 1.5x growth stays, so a density rise under it
costs no refit. `extract_device` binds the map's last list while it holds,
as after the encoder, and otherwise records the compaction into its one
batch, so an encoder after it compacts its own (amends the 2026-09-30
device-list entry). `DeviceMesh::ready` names the extractor's timeline
value the batch set: a consumer's GPU wait on it waits on a value already
reached, which the shared queue allows (the header's D10). A producer that
records its next extract before the consumer takes the last mesh needs
`slot_count` of frames in flight plus two (D5).

**The decoder judges a frame on the device (D4).** The host parses the
record in place in its input slot and checks what it can before anything
is recorded: the grid, the header, the tables, the reference, N against the
heap. Rows 11 to 13, which crossed the bus before the grid changed, are now
a gate kernel's: it opens the removal and allocation rounds, which close it
on their own failure, and the apply, the inverse and the deblock are
indirect slices of N behind it. The removal is recorded whatever the diff
holds, since the host no longer knows it, so every frame moves
`topology_epoch`; a frame refused on the device moves `tick` too, though the
grid and the reference stay untouched, as the 2026-10-13 coordinate entry
foresaw. The payload is bound where it lies
in the slot, at a byte offset the kernels take. Every update of the batch
(the tables, the segment words, the steps) comes before its first dispatch,
so MoltenVK opens no blit encoder mid-batch. `finish` settles both rounds
and reads the results slot. `rebuild` is one batch, one submit and a wait,
its results in the same slot. `record_growing` grows a heap the frame
outgrows before recording; `finish_growing` decodes the frame again from
the pending frame's own slot, after a grow (once) or as it is (twice), so a
recovery needs no other input slot. A recovery's host time, its decode
again included, is one `"  ..grow"` or `"  ..retry"` row within
`"codec decode"`, so the totals count it.

**Gathers write whole structs.** Each gather assigns its results with one
constructor, and a `static_assert` ties the host struct's size to its
fields. recon's GPU tests switch on core's ring-output poison
(`tests/gpu_test.hpp`), so a field a kernel skipped shows as the poison.

**The slots are leases (the free rule's user side).** A result or input
slot is claimed at record and held by the token -- `RecordedRounds`,
`PendingExtract`'s claim, `PendingDecode::input` -- until the token is
dropped and its batch has resolved. A refused recording claims nothing.
The map has `kResultSlots` (4), the decoder two input and two result slots,
the extractor two result slots. The map's record and `claim_input` refuse a
claim past them (`InvalidArgument`); the decoder's and the extractor's
result claims cannot fail while their `@pre` holds, and check it
(`VKC_CHECK`).

**No mappable-memory switch (amends the 2026-09-28 entry).** Core's
`HostRing` places a slot in place where `DeviceMapped` memory is cached
(Apple, lavapipe) and copies the used range through a device-only mirror
elsewhere (NVIDIA), from the device's facts alone. The 2026-09-28 reason,
that a Mac must exercise the NVIDIA path, is met by the NVIDIA CI legs,
which run the copied path under synchronization validation; its condition,
a staging measurement on Apple, by the M5 Max and the iPhone, where staged
minus in place was zero or more within noise.

**What a frame costs now.** One submit and no staging buffer, recorded or
through `decode()` and `extract_device()` (`recon_codec_recorded`,
`recon_mesh_recorded_extract`, `recon_volume_recorded`). A rafa2-like
P-frame's batch -- K 56, the LG deblock, blocks removed, kept and new,
decoded and meshed -- records 39 dispatches and 47 barriers, 48 where the
ring copies (its output copies' barrier); `recon_codec_recorded` pins them.
That is half the Apple step near 80 dispatches a batch, so none of its
levers is needed.

### 2026-10-25 — A live sender makes keys on demand: a joiner waits for the next key, the server keeps what its viewers still need, and a viewer that falls `max_gop_bytes` behind waits for a key, alone (amends the 2026-10-21 networking entry).

*Amended 2026-10-26 (below):* the stream that reads `key_wanted()` is
`sender::CodecStream`, in `code` and in `SetSender::send`.

**The rule.** `ServerOptions::replay_gop`, true by default, is the
2026-10-21 server: a joiner gets the GOP since the newest key. False, a
viewer joins waiting for the next key and from it gets every record, in
order. One that a record would leave more than `max_gop_bytes` behind the
newest finishes its chunk under way and waits for the next key again,
alone; a lost copy sends every viewer back so. The server keeps only the
records a viewer still has to send. Newest key wins in both modes.
`StreamServer::key_wanted()` counts each join and each fall-back, and the
viewers waiting for a key that is lost; a key within `max_gop_bytes` counts
none, as every viewer goes on from it; 0 with `replay_gop`. A record past
`max_gop_bytes` on its own, which no viewer can take, is logged at Warning
in both modes. The codec examples' `CodecStream` reads the count before
each frame's front half, in `code` and in `record` (so in
`SetSender::send`), and calls `Encoder::force_intra()` when it rose since
the last read. `codec_outputs` serves without `replay_gop`, so
fuse_orbbec's viewers that join or fall behind get a key at the next set;
`serve_stream` replays a file, cannot make a key, and keeps it.

**Why.** The 2026-10-21 entry joined viewers at the cached GOP because
waiting for the next key showed nothing for up to a second, and a replayed
file cannot force one. A live sender can: the joiner waits one set, 33 ms
at 30 fps, and is sent the key alone, not the GOP since the last one (up to
1.3 MB). And a GOP past `max_gop_bytes`, at any
`EncoderConfig::intra_period`, no longer freezes every viewer to the next
key: the server holds what the slowest viewer still needs, and a viewer
that falls that far behind within a GOP waits for a key alone. Newest key
wins still bounds the lag of one slower than the stream to about a GOP:
simulated on the GOP log for 600 s at 30 fps, at the rig's rate at 1 cm
(the 2026-10-21 entry's 0.78 MB key every 30 records and 9.4 MB/s, so
0.3 MB P-frames) and the default limit, a viewer taking 95% of the stream
fell 7 s behind before a fall-back without it, and stays within 2 records
with it, as when replaying. So a key counts none: a viewer it would take
past the limit goes on from it, and a second key made for that viewer
would only add to its backlog (without the skip, at a 2 MiB limit, the
simulated sender forced 239 keys in the 600 s, 189 with the viewer not
waiting; with it, none).

**Where the key is read.** The encoder picks I or P as the front half is
recorded (`Encoder::record_front`), so the count is read there, before the
fusion's batch is submitted, and a rise during a set keys the next. The
request stands until a frame is returned (the encoder's `finish`), so a
frame refused or abandoned, as when a fusion done again records the front
half again (`CodecStream::redo`), keeps it; joins between two frames make
one key; and the intra period counts from it. There is no limit past one
key a frame: a host that rejoins every set keys every set, and since a
viewer is unauthenticated (the 2026-10-21 entry), a limit would not stop a
determined one.

**Tests.** The GOP log, with no socket: replaying counts no join; on demand,
a joiner gets nothing until the next key, then every record; a viewer
holding its key under way stays at exactly `max_gop_bytes` behind and falls
back, counted, at the next record, the viewer keeping up getting every record;
a viewer behind skips to a newer key, which counts none though the viewer
would be past the limit with it; a record or key past the limit alone; a
lost copy, and a lost key counting a lone joiner again; and seeded runs of
records, lost copies and keys, and viewers joining, pulling, falling behind
and leaving, the bytes held never past the limit and none with no viewer in,
the count exactly the joins, fall-backs and waits a lost key ends, and each
viewer's records its key's run, broken only by the newest key. On loopback:
a joiner counted, then the next key and every record after; a raw viewer
that stops reading falls back, counted, once 512 KiB are behind it, then
drains a run from the first key, the new key and the P-frames after it,
while a client reading each record gets every one and is never counted; one
that stops reading over 3 GOPs of 1 MiB gets the newest GOP, in both modes;
a key past the limit alone logged at Warning, its joiner not counted again,
and a P-frame past it sending the viewer back, counted, at Warning; a copy
failing (an allocation failure injected, in `recon_networking_server_oom`,
apart so the rest link under TSan on Linux too) counting the viewer it sends
back, and a lost key the one waiting, before `publish` returns; replaying,
nothing counted. The stress run in both modes, its publisher keying on the count,
under TSan on the Mac leg. On a device (`recon_example_set_codec`), served
as fuse_orbbec serves, through `SetSender::send` and through `code`, each
viewer reading each set as it is published: keys at the intra period of 4
with no joiner; three joins before set 2 and one before set 7 key 0, 2, 6
and 7, each viewer's first record the key at its set; and, on 32 buckets
with the cameras seeing a patch until set 3,
set 3's fusion done again and its frame a key with a joiner, a P-frame with
none. `recon_example_config` pins fuse_orbbec's serve options to
`replay_gop` false.

**Validation.** RTX 5090, Ubuntu 26.04, GCC 15.2, Release with warnings as
errors and the ubuntu-26.04 leg's options, under the validation layer: the 108
tests pass, and the sender, config and networking tests under sync
validation too. RTX 4090, Ubuntu 26.04, Release with the ubuntu-24.04 leg's
Orbbec, FFmpeg, CUDA and Assimp: the 127 tests pass; the keyed sender,
server and stress tests 20 runs of 20 each; and the GOP log, socket,
server, stress and stream container tests under ThreadSanitizer (clang,
Debug). Apple M4
(Mac mini), macOS 27.0.1, Release with Orbbec, Assimp and the viewers: the
133 tests pass under the validation layer, the iOS check links, and the 12
threaded tests pass under ThreadSanitizer (Debug). Built Debug with clang
under ASan and UBSan on the RTX 5090, the networking tests and the three
example tests pass. The raw viewer fell back
after 10 records of 64 KiB on Linux, 2 of them sent, and after 15 on macOS,
7 sent. The on-demand stress run published 127,618 records in 1.5 s and
forced 34 keys for 44 joins and fall-backs, its viewers jumping 403
times. Mutations each fail a test: the count read only in `code` or only in
`record`, or after the front half (a key one set late); a key forced while
any viewer is counted, or one per join; `abandon` dropping the request (the
fusion done again); a joiner let in at once rather than at the next key, and
no fall-back (GOP log and server each); a fall-back or a join not counted;
the count not published after `publish`, or after a lost copy; the log never
trimmed; no newest-key skip on demand (GOP log, server); a key counting the
viewers it would take past the limit; a lost key counting no waiting viewer;
no Warning on demand; and fuse_orbbec's options replaying the GOP.

### 2026-10-26 — `MeshExchange`'s consumer may drop a mesh it took: a third outcome rather than a deadline in the exchange, and a ring slot more for each mesh dropped before the next commit (amends the 2026-10-08 `MeshExchange` entry).

**The rule.** `MeshExchange::begin_frame`'s `commit` returns a
`CommitDecision`. `kCommit` and `kPark` are what `true` and `false` were,
and a `commit` may still return `bool`; `kDrop` is new. A dropped mesh
gives `ExchangeOutcome::kDropped` and never becomes live: the live mesh
stays drawn. The drop frees the place a parked mesh holds, so the next
`begin_frame` takes the newest mesh published, one that superseded another
meanwhile included. A mesh that cannot be bound is still refused before
`commit` sees it, and an empty mesh may be dropped like any other.

**A third outcome, not a deadline.** A consumer that judges a mesh it took
stale, such as a viewer that has fallen behind and catches up by skipping
it, or one it parked that newer meshes have since overtaken, could only
draw it anyway or keep it parked, holding back every later one. Whether a
mesh is stale is the consumer's judgement, which the exchange cannot make,
so the exchange takes no clock and no policy: the consumer decides, and the
exchange releases what it lets go.

**The ring.** The mark is one monotone value and cannot skip a generation.
A dropped mesh therefore keeps its slot, as a superseded one does, until the
frames drawing the meshes before the next one committed retire; dropped
before any commit, it is released at the next `begin_frame`, as every take
then is. Each mesh dropped before the next commit holds a slot, so the
producer's ring needs one more for each a consumer may drop
(`MeshExchangeConfig::frames_in_flight`, `MarchingCubesConfig::slot_count`).
Parking needs none: a parked mesh holds the next take back, and the
producer, with a published mesh waiting, extracts no further. The exchange
does not cap drops, since a cap would either refuse the consumer's decision
or hold the producer back while nothing is left to commit; a consumer that
drops without bound exhausts any ring, as `release_through` says.

**One frame in flight counts as two**, for both producers and with or
without drops, which corrects the size the headers gave before: the mark is
computed before the take, so the mesh a commit replaces keeps its slot until
the committing frame retires. With two or more frames in flight, the frame
before draws that mesh until then anyway, so only a single frame needs the
extra slot.

**What stays.** The release, the take and the latch are unchanged, so a
`bool` commit behaves as before, and `fuse_viewer` and `rig_viewer` keep it.

**Tests.** `recon_mesh_exchange` adds: a drop's outcome, the live mesh
unchanged and the dropped generation never live; the take after a drop,
with a mesh parked and then dropped, which takes the mesh that superseded
another; the mark, dropped before any commit and after one, two drops in a
row held until the next commit's frames retire; an empty mesh dropped; a
dropped mesh's payload destroyed unless the commit moved from it; seeded
runs under one and two frames in flight, of commits, parks and none, one
or two drops before each commit, empty meshes included, in which no slot a
frame draws or the consumer has parked is reclaimed and, at the documented
ring size, no extract is refused, while one slot fewer some are; and the
`bool` form against `kCommit` and `kPark`, step for step over seeded runs
with failed claims and forced extracts. Its earlier cases are unchanged.
Its producer and consumer are two threads, so it now runs in the macOS
ThreadSanitizer leg too.

**Validation.** Release with warnings as errors, under the validation
layer. RTX 5090, Ubuntu 26.04, GCC 15.2, with FFmpeg, CUDA and Assimp, and
synchronization validation: the 120 tests pass, the ThreadSanitizer leg's
target and tests pass (GCC), and `recon_mesh_exchange` passes ten more
times under ThreadSanitizer and ten under ASan and UBSan (clang 21). Apple
M4 (Mac mini), macOS 27.0.1, with Orbbec, Assimp and the viewers, which
build unchanged: the 135 tests pass, the ThreadSanitizer leg's 13 tests
pass, `recon_mesh_exchange` among them, and the iOS check links.

### 2026-10-26 — A compaction records from a const map, and the encoder takes a const grid: the map's compaction output is scratch any compaction rewrites (amends the 2026-09-27 `Encoder` and `Decoder` entry).

**The rule.** `VoxelHashMap::record_compaction` is `const`. A caller that
holds a grid read-only, one it only reads (the encoder) or one handed to it
as `const`, records the active set's compaction, or a frustum's, into its
own batch, and runs its own kernels over the list, the count left on the
device. Before, such a caller had no way to get the active set on the
device: every compaction was non-const, and the map holds no list after a
recorded compaction, as a fuse's or an extract's leaves none.

**What const means here.** The call moves no block, voxel, `BlockStamp`,
heap count, topology epoch or tick. What it writes is the map's compaction
output, which every compaction rewrites: the list and its count, device
scratch written through the batch, which C++ constness does not see; the
serial that marks an earlier list of the same kind stale; and the frusta it
uploads, in a buffer it may grow. The two serials and the frusta buffer are
`mutable` and the frusta upload is `const`; no function body changed, and
one comment in the upload names the rule below. A list of the kind recorded
goes stale as before, so `held_device_list` and `check_device_block_list` /
`_subset`, const queries, can answer differently after a const call. The
map has no lock: no other call may overlap this one, const or not, and a
const reference does not make a map shareable across threads. A reader's
batch on the map's queue is ordered after the batches before it by the
core's barriers; another queue needs a semaphore, as before.

**Until the batch resolves.** The batch binds the map's own kernel sets, so
until it has resolved the map must not be resized, nor its frusta grown (a
frustum compaction of more frusta than any before): the owner of a grid
grows it only once a reader's batch has resolved. This held for every
recorded compaction already; a reader's batch can outlive the call that
recorded it while the owner goes on. A resize submits and waits before it
replaces a buffer, a wait that follows the batch on the map's queue, so the
batch has run, and the core refuses it at its wait, its readbacks
unwritten. A frusta grow waits for nothing: under a batch still running it
rewrites the set that batch bound and frees its buffer, which is undefined
behaviour. Handing the old buffer to the growing batch
(`CommandBatch::retain`) would keep the memory but not the set, which
Vulkan forbids rewriting under a pending command buffer (with the buffer
kept, the validation layer still reports the rewrite), so the rule stays a
precondition.

**The encoder takes a const grid.** `Encoder::encode` and `record_front`
take `const volume::VoxelBlockGrid&`. The 2026-09-27 entry took the grid
non-const only for the compaction; the encoder reads it and never writes
it. Every caller compiles unchanged.

**Why not the caller's own buffers.** A const compaction into buffers the
caller owns would bind them in the map's kernel set, which the core refuses
to rewrite while a batch that bound it is pending. The map would then need
a ring of sets, a limit on pending calls and a reclaim call, itself mutable
state, and each reader a list of `num_blocks` entries of 16 B. A compaction
object of the caller's own meets the same set problem at every resize.

**Why not the extract's recorded list.** The list an extract records has no
stamp, and its count lives only on the device, so a reader could not tell
whether it still names the active set. A reader pays one table scan in its
own batch instead, with no submit more; reusing the extract's list waits
for a measurement in which that scan shows.

**What stays.** The fuse, the free pass, the extract and the blocking
compactions call the same function and record the same commands: no copy,
submit or dispatch more. `MarchingCubes::record_extract` and the extract's
device path are, as the encoder was, non-const only for this call; they
keep their signatures until a caller needs a const extract.

**Tests.** `recon_volume_recorded` adds a const case: a `static_assert`
that a const map records one; then, through a const map, the whole set and
a frustum's subset, each the blocking compaction's list, pointers included,
in that list's buffer, in the caller's one submit with one dispatch and no
staging buffer; a held list of the kind recorded goes stale while the other
kind's holds; and no block, pointer, stamp, heap count, epoch or tick
moves. A resize case grows the grid under a reader's pending batch of each
kind, recorded through a const grid: the batch is refused at its wait, its
count unwritten, the validation layer sees nothing in use freed or
rewritten, and the reader's next batch counts the blocks again.
`recon_codec_encoder` encodes through a const grid in its device-list case
(the same bytes, the `"  ..encode set"` row only when no list holds, no
list held after), and the helper its halves, rules and tail cases record a
front half through takes one. Mutation-checked with four planted bugs,
each caught: the serial left unbumped, the other kind's serial bumped, the
frustum list returned for the whole set, and a resize that leaves the
kernel sets on its old buffers.

**Validation.** Release with warnings as errors, under the validation
layer. RTX 5090, Ubuntu 26.04.1, GCC 15.2, with FFmpeg, CUDA and Assimp,
and synchronization validation: the 120 tests pass. RTX 4090, the same
system and options: the 120 tests pass (`recon_sensor_video_jpeg` skips,
with no JPEG engine). Apple M4 (Mac mini), macOS 27.0.1, with Orbbec,
Assimp and the viewers: the 135 tests pass, the two changed tests pass
under synchronization validation too, and the iOS check links.

### 2026-10-28 — The voxel layer is camera-free: volume owns its depth camera, `DepthPinhole`, and the one line of unprojection it needs; tsdf copies its camera in (amends the 2026-10-16 GLSL pinhole entry).

**The rule.** Nothing in `volume`, headers, sources or shaders, includes
the camera vocabulary (`core/camera_params.hpp`, `core/shaders/pinhole.glsl`);
`recon_volume_camera_free` fails naming any such include. Depth allocation
takes `volume::DepthPinhole` (`voxel_hash_map.hpp`): `DepthCameraParams`'
fields in its 96-byte layout, uploaded verbatim as before. Its kernel's
camera block is `volume/shaders/depth_pinhole.glsl`'s mirror of it, with
`vrUnprojectDepth`, `vr_unproject_pinhole`'s one line copied operation for
operation.

**Why.** The voxel layer leaves with the codec for a repository of its own,
and the camera vocabulary goes with the sensors to another. A volume that
took `DepthCameraParams` would make the voxel layer depend on the sensors'
repository, and every consumer of the codec would pull sensor code. The
coupling was thin: the kernel already declared its own camera struct and
borrowed one line of `pinhole.glsl`, and the host only copied the bytes.

**tsdf keeps the camera vocabulary.** `tsdf::FrameInput` no longer derives
from `DepthInput`: it holds the depth, its `DepthCameraParams` and its
colour, built as `{depth, camera, color}`, so fusion, the sender and the
sensor pass are unchanged. `tsdf::to_depth_pinhole` copies a camera field by
field, beside `static_assert`s that the two types share every offset and
their size; the `Fuser` allocates through it. A list of `FrameInput`s no
longer slices into `DepthInput`s.

**Bit-identical, so measured once.** The same bytes reach the same kernel
arithmetic: allocation gives main's blocks, heap count and stamps. The
kernel's cost is unchanged, so no timing was run. Rejected: a matrix
unprojection (`world = M (u d, v d, d, 1)`), which moves a sample by a
rounding and so a block at a face, for a generality nothing needs; moving
`allocate_from_depth` up into tsdf, which would publish the hash table's
insert protocol; and volume taking the sensors' frame contract, which the
codec's consumers would then pull. The copy is the published formula in a
second place, pinned to the first by a test.

**Tests.** `recon_volume_allocate` pins three posed frames' blocks, heap
count and stamps (FNV-1a over the sorted coordinates and each block's
stamp), one frame alone from a host array and a device buffer and two as a
set, blocking and recorded, to values recorded on main at d8b8bda1, the
same on an Apple M5 Max (MoltenVK), an RTX 5090 and lavapipe: every
sample is at least a hundredth of a voxel from a block boundary, so no device's
rounding moves a block. `recon_volume_depth_unproject` holds
`vrUnprojectDepth` to `vr_unproject_pinhole` bit for bit and to the host's
double within float rounding on 9780 cases; it is the one test that reads
both copies. `recon_tsdf_depth_pinhole` checks the copy field for field and
byte for byte. The volume tests take `DepthPinhole`, and `sphere_scene.hpp`
is camera-free, its camera type a template parameter. Mutation-checked: a
swapped principal point fails the golden and parity tests, a field copied
from its neighbour the converter test, a reordered field the converter's
asserts at compile time, and a restored camera include the include check.
Reassociating the unprojection, `(px - c) (z / f)`, fails the parity test
on lavapipe; on MoltenVK and the RTX 5090 both forms gave the same bits on
every case, so there it passes.

**Validation.** Apple M5 Max, MoltenVK, Release with warnings as errors,
under the validation layer: the 119 tests pass. RTX 5090 and lavapipe,
Ubuntu, Release: the new and rewritten volume and tsdf tests pass. Replica
`room0` (400 frames, 2 cm), fused on main and on this change on the M5 Max:
every frame's allocated block count and block-set hash are the same, and so
is the final mesh.

### 2026-10-28 — A triangle mesh's distance field is a tier of its own, `voxelize`, on `volume` alone (amends the 2026-09-27 mesh-to-TSDF entry).

**The rule.** `MeshIntegrator`, `MeshSdfMode`, `MeshSdfParams` and
`MeshIntegrateStats` are `voxelize::`, in `voxelize/mesh_integrator.hpp`
and the target `recon_voxelize`, which links `recon_volume` alone and sits
right after it in the tier order. Its seven kernels and `mesh_scan.glsl`
move with it, so `recon_tsdf` builds only the fusion kernels. The
closest-point GLSL both mesh-driven passes share moves from `core` to
`volume/shaders/triangle_common.glsl`, the lowest tier either is in. No
header forwards and no namespace aliases the old names.

**Why.** The conversion fuses no depth. Its header and source include no
`tsdf` header and its kernels only `volume`'s GLSL and the closest point;
it needs the grid, the triangle allocation's candidates and `recon_core`'s
dispatch helper. In `tsdf` it made every fusing caller build seven mesh
kernels it never runs, and every mesh caller -- `codec_mesh` and the
consumers' stream writers -- link the fuser. It is the prep step that lets
the voxel layer leave recon without depth fusion.

**What stays.** The code, the kernels' bytes and every field it writes; the
2026-09-27, 2026-10-02 and 2026-10-21 rules hold as written. Only the
debug names change, `tsdf_mesh_*` and `tsdf.mesh_*` to `voxelize_mesh_*`
and `voxelize.mesh_*`. The tests move with it: `recon_voxelize_mesh_integrate`
and `recon_voxelize_mesh_integrate_exact`, unchanged but for their names.

### 2026-10-28 — The codec stream is a tier of its own, `codec_stream`, on `codec` and `networking`, and its frame in halves is public (amends the 2026-10-26 sender entry).

**The rule.** `CodecStream`, `CodecOutputs` and `SentFrame` are
`codec_stream::`, in `codec_stream/codec_stream.hpp` and the target
`recon_codec_stream`, which links `recon_codec`, `recon_stream` and
`recon_networking` and sits right after `networking` in the tier order.
`recon_sender` keeps `SetSender` and `fuse_inputs` and links it. The frame
in halves is the stream's own public calls: `record_front` into the
caller's batch, `front_ran` once that batch has resolved, `redo` for a grid
changed after it, and `finish_frame`; `valid()` tells a moved-from stream.
`coder()` is gone, and `stream_coder.hpp` is the stream's implementation,
included by its source alone. No header forwards and no namespace aliases
the old names.

**Why.** The stream needs the codec, the container and the server, and
nothing that fuses or reads a sensor. In `recon_sender` every user of the
stream, `codec_mesh` and a consumer's stream writer among them, linked the
fuser and the frame prep. And `coder()` handed a private type across a
tier: `SetSender`'s source completed it from a header beside the stream's
source. It is the prep step that lets the codec stream leave recon with the
codec, without depth fusion.

**The halves public, on the stream.** The 2026-10-26 entry kept them
internal, so that no caller could misuse four record calls. `SetSender`
now reaches them from another tier, through a public API only. A second
public class would need the accessor this removes, or repeat the stream's
calls. They are refused where they can be, as the encoder's own halves
are: a pending frame that a later front half or `code` superseded, a frame
out of order (its pending frame dropped), and every half of a finished or
moved-from stream but `front_ran`, which then does nothing. The rest is a
precondition, as the encoder's is: the front batch resolved before
`front_ran` and `finish_frame`.

**What stays.** Every frame, record and row. `SetSender::send` makes the
calls it made, through the stream's public calls rather than the internal
class: one call and one null check more per half, three a coded set. A set
due a frame of a moved-from stream is still refused before anything is
recorded, now through `valid()`.

**Tests.** `recon_sender_codec_stream` is `recon_codec_stream`
(`tests/codec_stream_test.cpp`), which links `recon_codec_stream` and no
fuser, mesh or sensor tier. Its case through `SetSender` moves, unchanged
but for its names, to `recon_sender_set_sender`. In its place, the halves
alone: frames in halves around a caller's timed batch are the file `code`
writes, byte for byte, the batch's device rows split between the frame and
the caller's own scope; a front half on another grid, recorded again on the
source, makes `code`'s frame of the source, its wait timed, and the pending
frame it replaced is refused; a frame out of order is refused; a viewer
joining between two frames makes the next frame in halves an I-frame and a
key record; and a finished or moved-from stream refuses the halves, a frame
whose front half ran before `finish` included.

**Validation.** Apple M5 Max, macOS, Release with warnings as errors,
Assimp and the Orbbec SDK, under the validation layer with a device
required: the tree builds, and the 19 affected tests pass. Mutations each
fail a test: `front_ran` giving the frame no device rows, or every row;
`valid()` true on a moved-from stream; `redo` keeping the first front half;
`finish_frame` dropping `sent`; `redo` dropping its rows; `record_front`
not reading the viewers' key requests (`recon_sender_set_sender` too);
`finish_frame` taking a finished stream, or refusing it as out of order;
all in `recon_codec_stream`.

### 2026-10-28 — The voxel grid's GPU contract has one home: `volume` exports `grid_types.glsl` and `voxel_color.glsl`, names its shared attributes, and keeps the hash entries internal (amends the 2026-08-02 color-space and 2026-10-13 block-list entries).

**The rule.** What a kernel outside the voxel grid's tiers may include is
what `recon_volume` exports (`vkc_export_glsl`, prefix
`volumetric_kit/recon/volume/shaders`), two files:
- `grid_types.glsl`: `VoxelGridParams`, `BlockIndex` and `BlockStamp`, the
  host PODs' scalar-layout mirrors; the in-block voxel index
  `x + bs (y + bs z)` (`vrLocalIndex`) and its inverse (`vrLocalBase`); and
  `vrTickAfter`, `volume::tick_after`. It absorbs `block_stamp.glsl`;
  `hash_common.glsl` includes it, and `tsdf_common.glsl`'s copies of the
  structs are gone.
- `voxel_color.glsl`: the `color` attribute's format. The exact sRGB curve
  and its packed decode and running-mean encode (from
  `core/shaders/color_common.glsl`, which is gone), `kVoxelColorUnobserved`
  (0), `vrVoxelColorFromObserved` (a first observation's R, G, B with alpha
  0xFF) and `vrVoxelColorToLinear` (unobserved reads opaque white). Its
  host half is `volume/voxel_color.hpp`.

tsdf's kernels include these, and texture's `pinhole.glsl`, through
`INCLUDE_TARGETS` (`vr_embed_shaders`), never `src/`, so they reach the
grid as a separate repository's kernels will. The hash table's GLSL stays
internal to codec, mesh and voxelize, and so does the entries buffer they
bind: `VoxelHashMap::entries_buffer()` and `entries_buffer_size()` move to
`VoxelHashMap::Internal`, a nested class the public header declares and
`src/volumetric_kit/recon/volume/hash_map_internal.hpp` defines (a member,
so it reads the map's private state without a `friend`).
`kTsdfAttribute`, `kWeightAttribute` and `kColorAttribute`
(`voxel_block_grid.hpp`, beside `kObservedWeight`) name the attributes the
tiers share, `float`, `float` and `uint32_t`. The PLY writer encodes sRGB
with a copy of its own, so `recon_mesh_io` includes no colour vocabulary.

**Why.** The voxel grid leaves recon for a repository of its own, and what
it publishes must be small and versioned: the grid's shape and block
records, and the colour format fusion writes and the mesher reads. The slot
layout, probe order and sentinels stay private, as the 2026-10-13 entry
meant them to. Before this, tsdf re-declared two of the structs, the colour
format lived in `core` beside the camera vocabulary that goes with the
sensors, three attribute names were string literals in five tiers, and
any tier could bind the hash table. The GLSL `tick_after`, dropped with
incremental meshing for want of a caller, returns as part of what a
consumer's kernel reads stamps with.

**Bit-identical.** The kernels compute what they computed: the structs
are the same declarations, the colour functions are the expressions they
replace, and the in-block index tsdf and voxelize wrote inline is
`vrLocalIndex`'s and `vrLocalBase`'s integer arithmetic.

**Tests.** `recon_volume_grid_types` runs `grid_types.glsl` on the device,
built through `INCLUDE_TARGETS`: `vrTickAfter` against `tick_after` across
the wrap, the in-block index for every voxel of five block sizes, and every
field of the three structs read from the host PODs. `recon_volume_voxel_color`
(was `recon_core_color_space_gpu`) runs `voxel_color.glsl` over every grey
code, the unobserved word and 255 others against `voxel_color.hpp`, and holds
`voxel_color.hpp` to `color_space.hpp` bit for bit on the host, the
unobserved word to opaque white.
`recon_io_ply` writes 5,892 linear channels, every code's rounding edge
among them, and reads back `color_space.hpp`'s codes. The subproject
consumer checks that each export root holds exactly its exported files,
compiles both kernels through `INCLUDE_TARGETS`, and fails if
`VoxelHashMap` names either entries accessor or `Internal` is complete in a
public header. Mutation-checked: of 20 mutants 19 fail a test; the one
left is equivalent (the decode toe moved from 0.04045 to 0.04, between
which no 8-bit code lies).

**Validation.** Apple M5 Max, MoltenVK, Release with warnings as errors
and Assimp, under the validation layer: the 128 tests pass, the codec's
golden frames unchanged. Main (7e93e85e) and this change, twice each:
`codec_replica` on Replica `room0` (400 frames, 1 cm, every frame coded)
and `codec_mesh` on Rafa2 frames 1–12 write the same stream bytes;
`fuse_replica` on `room0` (400 frames, 2 cm, colour) ends at 8,692 blocks
and 330,324 triangles, and its mesh and `codec_mesh`'s uncompressed and
decoded meshes hold the same vertices, colour bytes included, and the same
triangles. Their PLY bytes differ only in vertex order, as between any two
runs of main. The kernels' cost is unchanged, so no timing was run.

### 2026-10-28 — grid is a package of its own: the voxel grid and what reads it, under grid's names, on the core alone.

**What moved.** From `volumetric_kit_recon`, at
[`ac1b792e`](https://github.com/taojin-6/volumetric_kit_recon/commit/ac1b792e),
with their tests, shaders, examples and templates: `volume`, `voxelize`,
`mesh`, `io`'s mesh files (now the `mesh_io` tier), `eval`, `codec`,
`codec_stream`, `stream` and `networking`, and recon_core's split dispatch
(`dispatch.hpp`, its kernel and `dispatch_slices.glsl`), now `base`'s and
public, as recon's fusion records with it. `base` also holds grid's own
`device_requirements()` (`scalarBlockLayout`; the decoders' extensions stay
recon's), log source (`"vgrid"`), GLM aliases and version: per package, so
neither carries the other's. The examples `codec_mesh` and `serve_stream`,
the examples' scaffolding and codec helpers (`grid_example_common`) and the
test helpers recon's tests share too (`grid_test_support`) came along.

**Why one package, not voxel hashing apart from the codec.** The codec reads
the hash table on the GPU, probing slots through `hash_lookup.glsl`, and
`grid_diff.comp` walks every slot; apart, `HashEntry`, the probe order, the
primes and the sentinels would become a versioned public GPU ABI. Together,
grid exports three GLSL files (`grid_types.glsl`, `voxel_color.glsl`,
`dispatch_slices.glsl`) and keeps the slot layout private. That seam is also
the busiest: most PRs that crossed tiers in the two weeks before touched both.

**Names, renamed once, with no forwarding header or alias left behind.**
Namespace `volumetric_kit::grid` (`vgrid::` in docs; `vg::` is gfx's),
headers `volumetric_kit/grid/<tier>/`, targets
`volumetric_kit::grid_<tier>` under the umbrella `volumetric_kit::grid`,
macros and options `VGRID_`, GLSL functions `vgrid…` and macros `VGRID_…`,
tests `grid_…`. recon's CHANGELOG ([0.1.0](https://github.com/taojin-6/volumetric_kit_recon/blob/main/CHANGELOG.md)) holds the old
→ new table a consumer applies at its re-pin. Entries above this one name
things as they were then (`recon_<tier>`, `volumetric_kit::recon`, `VR_`).
Bytes do not move with names: the PLY comment stays `volumetric_kit_recon`,
`.vrst`, `VRMT` and the codec frames are unchanged, and the codec's golden
frames pass as they were.

**The boundary.** grid builds on volumetric_kit_core alone. No file of it
includes recon's or sensor's (`grid_package_boundary`), and CI configures,
builds and tests `grid/` alone (`cmake -S grid`). In recon's repository, recon
declares it after the core, by `FetchContent_Declare(volumetric_kit_grid
SOURCE_DIR …/grid)`, and checks it with `vkc_require_sibling`; there grid's
tests, examples and `-Werror` follow recon's, and it builds outside the
cached `_deps`. A project that fetches grid itself declares it first, by URL,
SHA and `SOURCE_SUBDIR grid` until the repository is its own. grid starts at
0.1.0, and recon moves to 0.1.0 with it.

### 2026-10-29 — The `Encoder` holds each P-frame one quantization step against its reference; I-frames are quantized afresh (amends the 2026-10-14 encoder entry).

**Why.** The forward quantized every frame from scratch, so sensor noise
that moves a coefficient to and fro across a rounding boundary flipped it
between two steps frame after frame: the decoded surface flickered where
fusion's did not, and every flip cost P-frame bits.

**The rule.** In the front's batch, after the forward and before the diff,
`inter_hold.comp` (an invocation a reference entry, the frame's block found
by binary search) gives each coefficient of a block the reference holds the
reference's value when |q − q_ref| ≤ 1, in the store's spare in place. So
the frame codes, the decoder holds and the next frame is coded against the
held values, and a block held whole, its masks unchanged, costs a skip; a
held coefficient in a block kept otherwise codes a zero residual. A change
of more than one step
passes in its frame, and every coefficient stays within one step of the
forward's. Integer only, so every device holds alike; masks and the blocks
the reference lacks are left as they are. Always on, with no switch: the
format, the golden frames and the `Decoder` are unchanged.

**I-frames are not held.** Held the same way, an I-frame still carries its
values in full and decodes on its own, and the flicker of the transition
into it goes. But nothing then re-centres a held value: it moves only when
the forward is two steps off it, and then to that outlier, so period after
period the held values drift toward the ends of the noise. On take1 (below)
that raised the decoded mesh's interior roughness p90 to 15.5° (Dynamic) and
13.7° (Classic), against 12.2° and 9.8° with P-frames alone and 11.6° and
9.8° on main (median 4.1° and 3.3°, against 2.9° and 1.9°), and its steady
flicker and rate were higher than with P-frames alone; only Classic's
interior flicker over a period was lower, 0.0009 against 0.0033. Held
I-frames decode exactly as P-frames held at an open-ended period, each frame
after the first held against the one before, so these figures are also what
an open-ended period costs. An I-frame quantized afresh re-centres once a
period, at one transition of flicker.

**The intra period bounds the drift.** The held values drift within a
period too, and only the next I-frame re-centres them: Rafa2's error climbs
from 0.26 mm over a period's first 5 frames to 0.34 mm over its last 5
(main's at the same rate is 0.37 mm throughout), so the hold's 14% lower
error at that rate is 29% at a period's start and 7% at its end; take1
Dynamic's interior roughness p90 climbs from 11.9° just after an I-frame to
12.3° just before the next (main's stays at 11.5–11.6°). A longer period
lets the drift grow toward the held-I-frame figures above.

**Measured** (Release; take1, a lab recording from 4 Orbbec Femto cameras:
1 cm, K 64, scale 0.2, period 30, 300 sets of recon's fusion through this
encoder; flicker the share of the decoded mesh's band triangles unmatched
across a transition, steady over sets 150–165, into the I-frame the single
transition 179 → 180, and over a period of 30 that transition with 29 at
the mean of 175–185's other nine; roughness over 150–165, and 175–179 and
180–185 for its climb; Rafa2: frames 1–150 on `codec_mesh`'s path, 5 mm,
K 56, scale 0.040877, period 30, error the decoded surface's against the
uncompressed TSDF's, mean over frames; one run per configuration):

| | main | held P-frames |
| --- | --- | --- |
| take1 Dynamic: Mbit/s; interior / edge flicker | 74.8; 0.155 / 0.355 | 51.3; 0.061 / 0.335 |
| take1 Dynamic, into the I-frame | 0.156 / 0.365 | 0.210 / 0.462 |
| take1 Classic: Mbit/s; interior / edge flicker | 17.0; 0.0213 / 0.0247 | 12.8; 0.0002 / 0.0011 |
| take1 Classic, into the I-frame | 0.022 / 0.024 | 0.092 / 0.119 |
| Rafa2: Mbit/s; error mm | 1.99; 0.212 | 1.26; 0.316 |
| Rafa2 at scale 0.09, main's at the same rate | 1.26; 0.368 | — |
| encode device ms, P median: 4090 take1 D / C | 1.65 / 0.93 | 1.46 / 0.92 |
| encode device ms, P median: M5 Max take1 D / C / Rafa2 | 2.24 / 1.52 / 1.20 | 1.98 / 1.42 / 1.15 |

Fusion is untouched: its mesh's flicker matches main's to every digit (its
vertex order varies run to run). Over a period of 30 the transition
into the I-frame leaves interior flicker 58% (Dynamic) and 84% (Classic)
below main's. On Rafa2, a clean mesh sequence, the hold trades error for
rate: at the same scale 37% fewer bits and a larger error, at the same rate
14% less error than main's coarser step, and decoded flicker down to the
source's own. The hold itself costs at most 0.01 ms a P-frame (the
forward's stage, which holds it, moves by at most 0.014 ms on both
devices) and no device memory, as it writes the store's spare in place;
P-frames encode faster because fewer entries survive to rANS (take1 Dynamic
0.92 → 0.75 ms on the 4090, 1.44 → 1.17 ms on the M5 Max). I-frames encode
in the same time (take1 medians within 0.05 ms on both devices). The 4090's
Rafa2 pair is left out: its I-frames, byte for byte the same, encoded 27%
apart.

### 2026-10-30 — An Audio track is kind 2 of `.vrst` v0: interleaved samples as its 8 params bytes say, every record a key at its first sample's capture time; no version change and no new record flag (amends the 2026-10-12 container entry).

**The kind.** `TrackKind::Audio` (2) holds interleaved samples as its
params say (`stream/audio.hpp`): 8 bytes, little-endian, codec `u16`,
channels `u16` (1 to 64) and sample rate `u32` (frames a second). The one
codec, `AudioCodec::PcmS16` (1), is signed 16-bit little-endian samples, a
record whole frames. Every record is a key, as each decodes alone; each key
costs 32 B of footer index, 5.76 MB an hour for a track of 20 ms records,
which the writer holds until `finish`. A record's pts is its first sample's
capture time on the track's clock, its seq the source's chunk count. Width
and height are 0, the sensor is the source's id, and the rate is nominal:
48 kHz in chunks of 960 is `{48000, 960}`.

**No version change, no record flag.** Version 0's readers pass an unknown
kind through, so a file with an Audio track reads wherever it did, and the
version changes only for what an old reader must not ignore. A new `RECD`
flag would make every audio record `Unsupported` to an old reader, and the
key bit already says what it would. A file without an Audio track is byte
for byte what 0.3.1 wrote, on disk and on the wire: `grid_stream_container`
pins a geometry-only session and records as `d1fcbc4`'s writer wrote them.

**The params refuse what no reader can play.** `decode_audio_params`
refuses fewer than 8 bytes, 0 or more than 64 channels and a rate of 0 as
`InvalidArgument`. It reads the codec first: one it does not know is
`Unsupported` whatever the other fields hold, so a reader skips that track
and plays the rest. Bytes past the 8 are ignored, so a later codec (Opus,
with a trailing pre-skip field) is a codec value, not a new kind.
`encode_audio_params` writes what it is given; the writer stores params
unread, as every kind's.

**PCM first.** Raw s16 needs no codec library and no compute: a mono
48 kHz track in 20 ms records is 790 kbit/s with their headers, against
take1's 13 to 51 Mbit/s of geometry (the 2026-10-29 entry).

**Not here.** `CodecStream` and `StreamServer` still take one Geometry
track; writing and serving an Audio track beside it comes with their next
changes.

**Verification.** Apple M5 Max, Release with warnings as errors, under the
Khronos validation layer: the full suite passes, 71 of 71. The new
`grid_stream_audio`, host-only, checks the params' golden bytes, read back
from bytes made by hand too; round trips, trailing bytes ignored; each
refusal; and a file of a Geometry track and an Audio track of a synthetic
440 Hz tone, opened from its footer and by a scan: every payload, each
track's stats and keys alike both ways, seeking on the Audio track, and
the Geometry track seeking as in a file of its records alone. Four planted
bugs fail the tests: the sample rate big-endian in both directions, the
more-than-64-channels check dropped, the channels checked before the
codec, and a track entry's reserved word written nonzero (both goldens).
