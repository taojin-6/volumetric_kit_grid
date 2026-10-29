# Design

This is the architecture and implementation reference for
`volumetric_kit_grid`. [AGENTS.md](AGENTS.md) holds the working
instructions; [DECISIONS.md](DECISIONS.md#decision-index) holds the dated
rules, rationale, measurements and review findings. Read the sections the
task needs, and update them with the code when a contract changes.

- [Tiered architecture](#tiered-architecture) and [names](#naming-conventions)
- Implementation: [base](#base), [volume](#volume), [voxelize](#voxelize),
  [mesh](#mesh), [mesh_io](#mesh_io), [codec](#codec), [eval](#eval),
  [stream](#stream), [networking](#networking), [codec_stream](#codec_stream)
- [Examples](#examples) and [next work](#next-work)

## Purpose and scope

`volumetric_kit_grid` is the family's sparse voxel grid and what reads it:
voxel hashing on the GPU, a mesh's distance field written into a grid,
marching cubes and the mesh containers, mesh files, mesh-to-mesh quality
measurement, the per-frame TSDF geometry codec, the `.vrst` session
container, a session served live over TCP, and the codec stream that joins
them. It fuses no depth and knows no camera: depth fusion, texturing and the
live sender are [volumetric_kit_recon](https://github.com/taojin-6/volumetric_kit_recon)'s,
which builds on grid. It is not a renderer; `volumetric_kit_gfx` draws the
meshes it extracts, on one shared `VkDevice` ([recon's interop
seam](https://github.com/taojin-6/volumetric_kit_recon/blob/main/DESIGN.md#the-interop-seam)).

Compute runs as Vulkan compute shaders (GLSL → SPIR-V), with MoltenVK on
Apple, one code path on Linux, Android, macOS and iOS. Every buffer a kernel
reads is declared `layout(scalar)`, so a device needs `scalarBlockLayout`
(`base/device_requirements.hpp`). The Vulkan foundation (device, allocator,
buffers, `CommandBatch`, `ComputeKernel`, timers) and the `Status` /
`Result` idiom are [volumetric_kit_core](https://github.com/taojin-6/volumetric_kit_core)'s,
used under its names; recon's design describes them where its tiers meet
them ([core](https://github.com/taojin-6/volumetric_kit_recon/blob/main/DESIGN.md#core), [gotchas](https://github.com/taojin-6/volumetric_kit_recon/blob/main/DESIGN.md#key-gotchas-verified)).

## Naming conventions

- Package: `volumetric_kit_grid`; namespace `volumetric_kit::grid`, nested
  per tier except `base`, which stays in `volumetric_kit::grid`. Docs write
  `vgrid::` for it (`vg::` is gfx's).
- Headers: `include/volumetric_kit/grid/<tier>/…`; the version header is
  `volumetric_kit/grid/version.hpp`.
- Macros: `VGRID_` (`VGRID_VOLUME_API`, `VGRID_WITH_ASSIMP`); GLSL functions
  `vgrid…` and macros `VGRID_…`. The error and contract macros are the
  core's, used under its names (`VKC_TRY`, `VKC_ASSIGN`, `VKC_CHECK`).
- CMake: fetched by commit (`FetchContent`); targets
  `volumetric_kit::grid_base`, `…_volume`, `…_voxelize`, `…_mesh`,
  `…_mesh_io`, the optional `…_mesh_io_assimp` (`VGRID_WITH_ASSIMP`),
  `…_codec`, `…_eval`, `…_stream`, `…_networking` and `…_codec_stream`, under
  the umbrella `volumetric_kit::grid`; for a sibling's tests and examples,
  `volumetric_kit::grid_test_support` and `grid_example_common`.
- Logging: the core's sink, with the source `"vgrid"` (`base/log.hpp`).

## Tiered architecture

A tier depends only on tiers to its left; no upward includes, and nothing
of recon's or sensor's (`tests/package_boundary.cmake`).

```text
base → volume → mesh → mesh_io
         │       └→ eval
         ├→ voxelize
         └→ codec ─┐
stream → networking → codec_stream
```

- **`base`** — what every tier is written against: the device requirements,
  the log source, the GLM aliases, the version and the split dispatch.
- **`volume`** — the sparse voxel hash map in Vulkan buffers and the
  attribute store over it (`VoxelBlockGrid`); allocate / compact / rehash as
  compute shaders, a stateless `grow_grid`, a default hash layout
  (`kDefaultBucketSize`, `kDefaultMaxChain`) and block layout
  (`block_grid_params`). Camera-free: depth allocation takes its own
  `DepthPinhole` (2026-10-28).
- **`voxelize`** — a triangle mesh's distance field written into a grid
  (signed, or as a shell), `MeshIntegrator`, on `volume` alone.
- **`mesh`** — marching cubes over a grid, the host and device mesh
  containers and the fuse-to-render handoff (`MeshExchange`).
- **`mesh_io`** — mesh files: PLY export, the host triangle containers and the
  opt-in normalization helpers in `grid_mesh_io`, static mesh import in the
  optional Assimp-backed `grid_mesh_io_assimp`. Host-side.
- **`eval`** — mesh-to-mesh distance, accuracy / coverage and the F-score,
  for the tests and examples that judge a reconstruction or a codec;
  host-side, and linked by nothing in a pipeline.
- **`codec`** — the per-frame TSDF geometry codec, `Encoder` and `Decoder`,
  on `volume` alone.
- **`stream`** — the `.vrst` session container, on the core's base tier
  alone: no Vulkan, no other tier.
- **`networking`** — a session served and read live over TCP, on `stream`.
- **`codec_stream`** — grids coded as frames into a `.vrst` file and to live
  viewers, decoded back, on `codec` and `networking`.

## Packaging

A consumer fetches grid by commit (`FetchContent`) and links its targets;
grid has no install rules. Its version, `project(VERSION)`, is the umbrella's
`VKC_PACKAGE_VERSION`, which a consumer checks with the core's
`vkc_require_sibling(volumetric_kit::grid <min>)`. The exported GLSL is the
grid's GPU contract: `grid_base`'s `base/shaders/dispatch_slices.glsl` and
`grid_volume`'s `volume/shaders/grid_types.glsl` and `voxel_color.glsl`,
under roots that hold those alone (`tests/glsl_consumer.cmake`); the hash
table's GLSL stays internal.

## Implementation reference

[DECISIONS.md](DECISIONS.md) carries the *why* behind anything here that
looks arbitrary.

### base

`device_requirements()` is what grid's kernels need of a device:
`scalarBlockLayout`, the buffer ABI every grid shader declares; every
`create` that builds kernels first calls `check_device_requirements`, which
refuses a device without it. A consumer that shares its device merges its
own requirements with these (`core::merge`). `log_message` logs through the
core's sink with the source `"vgrid"`, and `math/vector_types.hpp` holds the
GLM aliases the headers trade in (`Vec3f`, `Vec3i`, `Mat4f`, …).

`base/dispatch.hpp` is the split dispatch every tier records with (its names
are in `detail`, outside the versioned API, public because recon's fusion
records with it too). Vulkan guarantees only 65535 groups along x, which a
room's blocks pass, so `dispatch_items` records a pass whose size the host
knows in as many dispatches as it takes, each told its first item.
A pass whose size a kernel finds on the device is sized there:
`record_dispatch_args` runs the one count-to-indirect-slices kernel
(`base/shaders/dispatch_args.comp`), which writes `IndirectSlices` a later
`dispatch_slices` reads, so no count reaches the host. A kernel that sizes
its own passes writes them through `base/shaders/dispatch_slices.glsl`
(`vgridGroups`, `vgridWriteSlices`), the split's one home on the device.
`set_test_groups_per_dispatch` caps the split for a test, so a small input
takes the path a device's limit gives a large one.

### volume

`VoxelGridParams::validate` rejects non-finite metric fields and block extents,
and bounds `ceil(trunc_dist / (block_size * voxel_size))` to 127. This is the
representation limit of the depth allocator's `uint32` work-item count:
256 distinct tile centres times `(2 * 127 + 1)^3` is 4,244,832,000, whereas
radius 128 overflows. It also keeps the shared host/GLSL radius conversion
within its integer range. Validation happens before creating a map; the
coordinate helpers retain that precondition. `block_grid_params` is the
default block layout: 8³-voxel blocks in the default hash layout, as many
blocks as bucket entries, at the caller's resolution, band and table; the
fusion grid, the decode grid and codec_mesh build with it (2026-10-25).

`VoxelHashMap` drives init / allocate-from-coords, -depth,
-triangles / remove / compact / compact-in-frusta / resize as GLSL
kernels
(`volume/shaders/hash_*.comp`) over the scalar-block-layout ABI. Depth
allocation unprojects a posed frame and dilates each surface block into the
`(2·tb+1)³` truncation band — a solid cube, not a ray march — a 16 × 16
pixel tile a workgroup, each distinct block of the tile dilated once and its
band shared out over the lanes (2026-09-30);
`allocate_from_triangles` deliberately does **not** dilate that cube
(2026-08-31): a triangle wider than the band — which is one block, 40 mm, at
the defaults — would hole through the middle, so a block is allocated when its
centre is within `trunc_dist` + the block's half-diagonal of some triangle,
and the work is split per *candidate block* rather than per triangle so one
lane cannot own a 250 000-block quad; as a depth tile does, a workgroup asks
for each distinct block its lanes name once. The host pass that counts those
blocks is also where an index is bounds-checked and a zero-area or non-finite
triangle is dropped — the former being the one input that divides by zero in
the closest-point solve, which lives in `volume/shaders/triangle_common.glsl`
so mesh-to-SDF evaluates the field with the same function. That pass is the
public `triangle_candidate_offsets`, and the decode and band test are
`volume/shaders/triangle_candidates.glsl`: `voxelize`'s mesh integrator bins
over both, so the blocks allocated and the blocks binned cannot drift apart.
Frustum-culled compaction gives the per-frame working set, from frusta built
by `make_frustum_planes` off a depth camera's pinhole
intrinsics or — since 2026-08-12 — off a *render* camera's `view_proj`, whose
planes are read off the matrix itself and so hold for any handedness, provided
depth maps to `[0, 1]` (gfx's convention; a GL matrix puts the near plane at
the harmonic mean `2nf/(n+f)`, over-culling a shell about one near-distance
thick — pinned by a block that lies inside it). That
overload's `margin_m` widens every plane **except the near one** by a distance
in metres, applied after normalizing, for a consumer whose cull runs a few
frames behind its draw; widening near too would move it behind the eye at
`margin_m > z_near` and admit a cone of geometry behind the camera. A
degenerate or non-finite matrix leaves its planes unnormalized and unwidened,
so the cull degrades to keeping blocks rather than dropping them.
`compact_active_blocks_in_frusta_on_device` keeps the blocks inside any of
several frusta in one scan, on the device in a list of its own, which
`check_device_block_list` refuses as not the active set and
`check_device_block_subset` accepts until the next frustum compaction
rewrites it or the map allocates, resizes, removes, clears or moves; fusion
culls to the scanning cameras with it and marching cubes to the viewing one
(2026-10-06). A block list is a `DeviceBlockList`, on the device
(2026-10-13). A sorted block list is in `coord_less` order (z, then y,
then x); `volume/shaders/coord_order.glsl` mirrors it for kernels
(`vgridCoordLess`), with the binary searches they run over one. `resize`
preserves block indices, so per-voxel data survives a grow.
`VoxelBlockGrid` composes the map with independently-allocated SoA
attribute arrays (`tsdf`, `weight`, `color`, …), each
`num_blocks·voxels_per_block`, so a consumer materialises only what it
needs; the three the tiers share are named by `kTsdfAttribute`,
`kWeightAttribute` and `kColorAttribute`. Every buffer of both is
device-local, reached through a `CommandBatch`: `create`, `clear` and
`remove` zero on the device (`clear` and `remove` before any index is freed,
`remove` by a kernel that finds each coord's block itself, so its cost is
the count's, not the grid's), `resize` copies there, and a call's inputs
are uploaded in its batch.
`VoxelHashMap::allocate` and `remove`, and `VoxelBlockGrid::remove`, take
a device buffer of `BlockIndex`, bound where its producer wrote it; a test
uploads its own coordinates first (`tests/device_blocks.hpp`). Their work
is recorded into a caller's batch by `record_allocate` and `record_remove`
(the 2026-10-24 one-batch entry): K = 3 rounds, each sized on the device
from the last one's tally, the count a `DeviceCount` a kernel wrote and an
optional gate word, the outcome gathered into a result slot that `settle`
reads once the batch has run. The blocking calls are record, one submit and
settle, repeated while lost races alone remain, 10 rounds at most. Each also takes
a caller's `GpuStageScope`, whose tile every round counts to, as the
decoder's `"  ..apply"` does. A grid's `create` builds the kernels its removal
and block pass run, so the first remove compiles nothing. The grid's
zeroes the listed blocks and frees them through the map's, the zeroing
recorded into the map's first round (`remove`'s `first_round`) ahead of
the delete, so it takes no submit of its own.
One body serves it and garbage collection, so the stale-block list stays on
the GPU through deletion, its count sizing the removal there; only the
removal's outcome reaches the host.
A compaction still reads its list back, in the count's own submit while the
set stays within a quarter past its last count (the 2026-09-28 residency
decision), except `compact_active_blocks_on_device`'s: its
`DeviceBlockList` stays on the device, stamped with the epoch, a
compaction serial and the heap's free count, and `check_device_block_list`
refuses one that a full compaction, allocation, resize, remove, clear or
move has made stale. While the last one still holds, the call returns it
again, dispatching and reporting nothing, so an extract after the encoder
reuses the encoder's compaction (2026-09-30); an extract records its own
into its batch when none holds (2026-10-24). A fuse culls instead and leaves
none: `record_compaction` records its compaction into the fuse's own batch,
the list and its count left on the device for the dispatches after it.
It is `const` (2026-10-26), so a caller holding the grid read-only, such as
the encoder or a caller handed a const grid, records one into its own batch:
it moves no block, stamp, heap count, epoch or tick, and writes only the
map's compaction output (the list, its count, its kind's serial and the
frusta, `mutable`), which any compaction rewrites. Its batch binds the
map's kernel sets, so the map is neither resized nor its frusta grown until
that batch resolves: a resize waits for the queue and the core then refuses
the batch, while a frusta grow under a running batch is undefined.
`topology_epoch()` lives on the *map* — the object that frees a block
index — and is a globally unique token re-drawn at `create`, at every
`clear` and at every `remove` that accepts its input, never at `resize`: a
`DeviceBlockList` anchors on it, so no path may free an index without
moving it and no two grids may ever share a value. `load_factor()` is the
occupancy read a per-frame caller can afford (a host copy of the heap
counter, which every call that moves it refreshes, `settle` for a recorded
one), and `kGrowThreshold` is
the occupancy it says to grow at — named here so a UI or an embedder cannot
draw a ceiling that disagrees with it. No host call reads the table back;
a kernel of the grid's own tiers (codec, mesh, voxelize) binds the entries,
`VoxelHashMap::Internal::entries_buffer` in
`src/volumetric_kit/grid/volume/hash_map_internal.hpp` (2026-10-28), and
reads them through `hash_common.glsl` or `hash_lookup.glsl` (2026-10-13),
whose layout is internal: the host `HashEntry` and fail slots are in
`src/volumetric_kit/grid/volume/hash.hpp`, and the slot sentinels and hash
are the GLSL's alone in the library (2026-10-17). What any kernel may
include is what `grid_volume` exports (`vkc_export_glsl`, 2026-10-28):
`grid_types.glsl` (`VoxelGridParams`, `BlockIndex`, `BlockStamp`, the
in-block index `vgridLocalIndex` / `vgridLocalBase` and `vgridTickAfter`) and
`voxel_color.glsl`; tsdf's kernels include them through `INCLUDE_TARGETS`,
not `src/`.
`grow_grid` centralizes the growth arithmetic: double, or reach a larger
requested minimum, within `VoxelBlockGrid::max_num_buckets` (block pointers
within int32, each attribute array within one binding). It retains no policy
or retry state and reports resize failures through `Status`. `bytes_at`
reports the attributes and grid-sized table's storage footprint. It excludes
transient rehash/staging buffers, host scratch and allocator/driver overhead,
so it is not a peak-memory budget for resize (2026-10-10).
`allocate_from_depth` takes a frame's camera as volume's `DepthPinhole`, the
fields and 96-byte layout of `DepthCameraParams`, which tsdf fills by
`to_depth_pinhole`. It also takes a list of frames (`DepthInput`), every
frame dispatched in each round's one submit, on a set of its own; a round
that retries dispatches them all again, and the rounds are the call's, not
each frame's. `record_allocate_from_depth` records one such round into a
caller's batch, its outcome in a result slot and an optional gate closed on
any failure; the blocking call is that round, one submit and `settle`,
repeated while lost races alone remain. `settle` moves the heap's host copy
only to a recording newer than every one settled and every blocking call
since, so recordings settle in any order. Opt-in
`StageMetrics*` on `allocate_from_depth` (an `"allocate"` row summing every
retry round) and on both compaction entry points (an `"active set"` row,
breakdown-prefixed when the caller already has a stage open).
Every block slot carries a `BlockStamp` (`hash_types.hpp`): ticks of the
map's clock (`tick()`), in a device buffer beside the heap
(`stamps_buffer()`), each written by the pass that knows its fact
(2026-10-01). Every allocation kernel stamps `requested` on each block it
asks for, inserted or found, and the grid's block pass (in `free_stale_blocks`)
stamps `weighted` on each holding an observed voxel. Every pass that
writes voxels -- `TsdfIntegrator`, `MeshIntegrator`, the codec's
inverse -- advances the clock first and stamps `changed` on what it
changed, so the clock counts writes (one a set, for the rig) and a
reader's recorded tick is older than every later change, however the calls
interleave. A fuse its gate closed on wrote nothing, so `Fuser::finish`
takes its tick back (`rewind_tick`) before fusing the set again. The init kernel zeroes the records, the delete kernel zeroes a
freed slot's, and `resize` copies them forward, the rehash stamping
nothing. A consumer compares ticks and resets nothing:
`free_stale_blocks(max_age)` frees the blocks whose newer stamp is older
than that, through the grid's remove body over the pass's list, so the band the
allocator still asks for stays. `record_free_stale` records it into a
caller's batch, at the tick then: the compaction, the stamp pass sized from
its count, and `record_remove` of the list it writes, one recording of
K rounds, so a block a lost race keeps waits for the next pass;
`free_stale_blocks` is that, one submit and `settle`, its result the heap's
gain (2026-10-25). The live examples run it every
`fusion.free_after` sets (`free_stale_blocks_every`, `fuse_frame.hpp`); the
iOS scanner reads `changed` through `read_block_stamps`.

### voxelize

`MeshIntegrator` writes a triangle mesh's distance field into a grid
(2026-09-27), **overwriting** every voxel of every block the band reaches:
weight 1 within `trunc_dist` of the mesh, the codec inverse's fresh zeros
elsewhere, so a mesh-derived grid reads like a decoded frame. `Signed` signs
by the closest triangle's face normal, the research codec's rule, and builds
no topology and checks nothing: the field is the codec's input, not a
reconstruction of the mesh, so it gives up the sign past edges sharper than
90° and grows a skirt past an open mesh's rim. `Shell` is `d − σ` and takes
any mesh. The blocks must be
allocated by `allocate_from_triangles` — a missing one is refused before
anything is written — and triangles are binned per block over that
allocation's own candidates, so no voxel measures the whole mesh. The fill
replays the slots the count pass recorded, so no bin comes up short; a bin
past `kMaxBinTriangles` is refused, and the write splits into dispatches of
at most `kMaxDispatchBinEntries` bin entries, each submitted on its own.
Counts, fill cursors and occupied bins stay on the device: a hierarchical
256-way prefix scan carries entry totals, occupied counts, largest bin and
overflow, then compacts bins in slot order. The host reads 20 bytes of
validation/size control plus 8 bytes per bounded dispatch range, at most
34,960 bytes of ranges even for the uint32 candidate limit. Prefix windows
of `kMaxDispatchBinEntries - kMaxBinTriangles + 1` bound every range; the host
only splits them further for the device thread limit. Bulk scratch grows
with demand and stays allocated. If the compact list outgrows its retained
capacity, only compaction/range planning retries after growth; normal calls
need one planning submission plus one per write. No grid write precedes
missing-block, overflow, bin-size and binding-range validation
(2026-10-02). Ties break on the triangle index, so the same mesh writes the same
bytes. One workgroup writes each block, and a voxel measures only the
triangles no lower bound rules out; a bound, less a rounding margin of the
triangle's own, rules a triangle out only when it is strictly farther than
the voxel's best so far, or than `trunc_dist` where that is nearer, so every
voxel writes what measuring its whole bin writes (2026-10-21;
`mesh_integrate.comp` has the bounds and the rounding count). Blocks past
`kMaxVoxelsPerBlock` (16^3 voxels) are refused. Each call is a tick, and
every block it writes is stamped `changed`.

### mesh

`MarchingCubes` over a sparse `VoxelBlockGrid`, and only that
(the dense analytic entry point was removed 2026-08-31; the prior engine
never had one). Two workflows, named by their entry points: `extract_host`
returns an owned host `Mesh` for export — the complete transaction, giving
its ring slot back so a PLY writer never learns the ring exists — and
`extract_device` returns a borrowed `DeviceMesh` a renderer draws directly,
released by generation. One workgroup per active block, with the cross-block 2×2×2
neighbourhood resolved by probing the hash table on-device. A block counts
its output, reserves one range for all of it with a single atomic, and
only then writes, so **a block's output is contiguous in the arena** (both
sparse kernels; `share_vertices` reserves two ranges, since a shared vertex
breaks `v = 3t`). Each cursor is bounded by its block's own reservation, so a
count that disagreed with the emit would drop geometry rather than write over
the next block's range. Every extract meshes the whole active set or a
caller's list and rewrites its slot from zero; incremental extraction was
removed on 2026-10-06. The vertex arena
is fitted to the surface, grow-only, and held as a **ring of slots** the
consumer releases by generation; the kernel writes a real
`VkDrawIndexedIndirectCommand`. `extract_device` returns a borrowed
`DeviceMesh` (valid until the next extract, enforced by a generation stamp),
`download` takes the single host copy and bridges the two workflows. The
arena, index run and draw command are device-only. An extract is recorded
into a caller's batch (`record_extract`, the 2026-10-24 one-batch entry): a
one-thread kernel sizes the marching cubes from the compaction's count,
gated, and a finalize gathers the results into a result slot that `finish`
reads, refitting in a blocking batch of its own when the surface outgrew
the slot; `DeviceMesh::ready` names the timeline value the batch set.
`extract_device` records the same over the map's device list (its last
compaction while that holds) and submits it once. Given `StageMetrics`, it
reports a
`"marching cubes"` row over the compaction's `"  ..compact"`, the submit's
`"  ..dispatch"` and a refit's `"  ..refit"`; its timer is made by the
first such call. A whole-map extract never
brings the active list to the host: it compacts onto the device, or takes
the map's last list back from it, and binds it in place (2026-09-30). An
`extract_device` overload meshes a `volume::DeviceBlockList` the caller
compacted instead — the viewers' frustum-culled set of what their camera
sees (2026-10-06), bound in place like the map's own. The arena is rebuilt
from that dispatch alone, so a block outside the list costs no triangles and
no *live* bytes — the arena is grow-only, so one full extract sizes it for
the whole set and it never shrinks back; the surface does not hole at the
cull edge, since the on-device probe still resolves neighbours that were
never dispatched; a list from another map, or one a later compaction,
allocation, resize, remove or clear has made stale, is refused by
`VoxelHashMap::check_device_block_subset` above the slot claim, so a refusal
is a **rollback** and an outstanding `DeviceMesh` survives it; and
`compact_ms` reads 0 while every row that scales with the active set shrinks
with it (`readback_ms` and `descriptor_ms` are per-call constants and do
**not**). A culled pass records its density like a full one: a viewer only
ever culls, and a plan left on the seed refits on every call.
`share_vertices` selects a second
compiled kernel that indexes in-block vertices — 3.4x fewer on room0, and
textured like any other mesh since the `texture` tier moved to a per-vertex
verdict (2026-08-11). `DeviceMesh::shares_vertices` still publishes it,
because `v = 3t` no longer holds and a consumer sizing an arena cannot derive
that from the buffers, and because the `texture` tier's several-view atlas
chooses per triangle and so refuses a shared mesh (2026-09-28).
A vertex on a corner whose distance is the iso-value is that corner,
`fl(k · voxel_size)` on each axis: it is selected, not interpolated, so it
is the same bits on every device and from both kernels. Every corner is that
rounded product before anything interpolates from it (`precise`, with the
voxel size splatted for MoltenVK; NVIDIA's compiler otherwise fuses it into
the interpolation). The edge ratio and `mix()` are still the device's, so an
interpolated vertex may differ across devices: on NVIDIA and Apple by an ulp
of the ratio times the voxel along its edge (an ulp of a coordinate, unless
the coordinate is small against the voxel), and in every coordinate on a
device whose `mix()` rounds all three, as lavapipe's does.
`MeshExchange<Payload>` (`mesh/mesh_exchange.hpp`) hands `extract_device`'s
meshes from the extracting thread to a renderer drawing them in place, each
with a payload that stays with it (its atlas), and owns the ring's release
(2026-10-08): the renderer's `begin_frame` retires a frame slot, computes the
mark below every generation a frame in flight, the live mesh or a parked
take holds, and takes the newest mesh, under one lock; the producer applies
the mark on its own thread before each extract. The consumer's commit
commits, parks or drops each mesh taken (2026-10-26). Header-only and
gfx-free; it takes any ring with `release_through`, so a host model of the
ring tests it. The `DeviceMesh` →
`LiveMesh` adapter is recon's viewers' `recon_gfx_bridge.hpp`.

### mesh_io

`write_ply` exports a `mesh::Mesh` as binary PLY: it converts the linear
colours to canonical sRGB bytes (clamped, NaN as 0) with the files' own copy
of the published curve, which `grid_io_ply` holds to the voxel colour
format's, and preserves the triangle indices. It opens its output only
after encoding succeeds. Errors name the file and the backend's reason.
Public functions validate their input and translate backend exceptions that
reach them into `Status`/`Result` (tinyply's writer is `noexcept`); private
third-party types never appear in public headers.

`grid_mesh_io_assimp` adds `mesh_io::load_mesh`, which returns the owned
`mesh_io::TriangleMesh` position / index container of
`mesh_io/triangle_mesh.hpp`. Assimp handles format decoding and
triangulation. grid walks every node instance with composed transforms and
corrects triangle winding when the determinant is negative. Exactly
coincident positions join across UV/material seams, with no tolerance-based
merging. This is geometry import, not topology repair: degenerate and
duplicate triangles are retained, and manifold/sign checks are the
consumer's, through `mesh_io::audit_mesh_topology` if it opts in.
`mesh_io::check_mesh_file` holds the path checks `load_mesh` makes before it
reads, for a caller that checks a sequence's files before loading any.
Non-triangle primitives, animation, skinning, morph targets and invalid
transforms on mesh instances are refused; nodes without meshes are not
checked.

`mesh_io::load_textured_mesh` (`mesh_io/textured_mesh.hpp`) runs the same
traversal with each vertex keyed on its position and first UV channel;
`load_mesh` keys on the position alone, so its output is unchanged. An OBJ's
diffuse map path is relative to its material library, which the loader finds
through an `Assimp::IOSystem` that notes each `.mtl` Assimp opens. Assimp 6
already prefixes the path with the library's directory relative to the OBJ,
even an absolute path; the loader removes that prefix and resolves the rest
against the library, as it does on earlier releases, which add none.

Mesh coordinates retain the imported scene's scale; neither a metre unit nor
a target height is guessed. No Assimp normalization or handedness flags are
enabled, and importer unit/up-axis conversions are switched off where the
installed Assimp allows: before 5.3 it still applies Collada's `<unit>`, and
5.4.x converts FBX regardless. This is a host boundary for file input and
output, separate from GPU-resident work. The default build has no Assimp
dependency; `VGRID_WITH_ASSIMP=ON` finds an installed Assimp and builds a
separate target. The PLY backend is the pinned tinyply source, a private
implementation dependency of `grid_mesh_io`.

`mesh_io/mesh_normalization.hpp`, in `grid_mesh_io`, holds opt-in helpers for a
caller that knows its mesh's unit or up direction; nothing in grid's tiers
calls them. They serve one mesh or a temporal sequence alike:

- `up_axes` turns a head-up direction into the proper rotation that takes it
  to +Y. `mesh_bounds` measures positions along those axes, and
  `extend_bounds` joins another frame's, exactly and in any order, so a
  sequence is measured one frame at a time.
- `fit_transform` (to a height in metres) or `scale_transform` (a given
  `metres_per_unit`) makes one `MeshTransform` from the bounds, centring X/Z
  and putting the lowest Y at zero. `apply_transform` maps every frame
  through it in double precision, rounded to float, so a still point stays
  still. The fits and `apply_transform` refuse axes that are not a proper
  rotation, and a refusal leaves the mesh unchanged; the fits also refuse
  bounds they would place past float range. The source compiles
  with `-ffp-contract=off`, so a transform's doubles do not depend on the
  compiler.
- `encode_transform` and `decode_transform` are a transform's one byte
  form, bit for bit (the layout is in the header): what a `.vrst` Geometry
  track's params hold, so a player can place decoded geometry back in the
  source's coordinates.
- `audit_mesh_topology` counts the edges, triangles and components the
  signed conversion depends on, capping each hole before it takes a
  component's volume; `supports_signed(allow_holes)` gives the verdict.

### codec

The codec is implemented, intra frames and P-frames (frame type 1), and it
codes and decodes on the device alone (2026-10-13): the host does the I/O
around it (a frame's bytes, and their hash) and the control (a frame's
header, sections and tables, and small counts and status words, a decode's
and an encode's read from results slots once their batches have run).
The CPU rANS coder is the tests' oracle, under `tests/`
(`codec_oracle*.hpp`), which pins the format's bytes.
Quantization is one scalar quantizer with a step per DCT basis (2026-10-02):
`step[u,v,w] = quantization_scale * quantization_weights[u + 8*v + 64*w]`.
DC is the table's first entry. All blocks share the same 512-entry table, but
only the K kept bases use their weights: the global scale and those K weights
travel once in the self-contained v3 frame.
The defaults retain room0's K = 64 and uniform effective step 0.2, expressed
as scale 0.2 and weights of one. Frequency-dependent tables are measured
candidates, not a claim that high frequencies always deserve fewer bits.
The public API
is `CodecParams`, **`Encoder`** (`encoder.hpp`), **`Decoder`** with
`read_frame_info` (`decoder.hpp`), and a player's `make_decode_grid` and
`decode_growing` (`decode_grid.hpp`). A frame is two batches, the caller's
(the 2026-10-25 entry): `Encoder::record_front` records, after what the
batch holds (a sender's fusion), every step whose count only the device
knows, and `record_back` the rANS coding, sized on the host from the
front's results; `finish` lays the frame out, its payload left in an
encoder slot (`EncodedFrame`), and `encode(grid)` is those, a submit and a
wait each, the head and payload joined: 2 submits. The front takes a
const grid and the map's own active list -- the one it holds
(`held_device_list`), or one compacted into the batch
(`record_compaction`), which leaves the map none, so an extract after
compacts its own -- keeps the blocks with an observed voxel, or with
`EncoderConfig::true_sdf` those holding surface (`DctTransform::record_observed`,
their count left on the device; the true-SDF passes below share it), sorts
them by (z, y, x) on the device with a stable rank-merge sort on
`coord_less` itself, over all of int32 (`detail::BlockLists`, whose header
has the mechanism), its last level writing the reference store's spare list,
and transforms them into the spare's masks and coefficients (below). Every
step is sized for the grid's block slots and runs for the device's count:
the observed pass and the forward from indirect slices, the sort's levels
recorded for the bound, a level the count does not need copying its entries
through. Keys are unique, so the same content gives the same bytes whatever
the hash table's order. A frame whose front finds no block to code is an
I-frame of none (below), with no back batch.
At `EncoderConfig::intra_period` N above 1 (default 1, 0 refused) the
`Encoder` writes P-frames (the 2026-10-14 entry): an I-frame first, then
every N coded frames, after `force_intra()`, when the grid's `voxel_size` or
`trunc_dist` is not the reference's (`coding_difference`, the writers' and
reader's rule), when the reference holds no block, and when the grid holds
none (the 2026-10-16 entry, decided by the device's count, so a P-frame
planned at record finishes as an I-frame of none); a P-frame otherwise,
against the frame returned last. Its reference is what the decoder
will hold of that frame: the forward's q and masks and the sorted list, left
in `detail::ReferenceStore`'s spare and swapped in when `finish` returns the
frame, so a refusal or `abandon` keeps the reference and the period's count.
In the front's batch, `inter_diff.comp` splits the frame against the reference
and codes it, an invocation an entry: each reference entry finds its block
by binary search (remove, skip when every word is equal, else keep or zero),
`survivor_ranks.comp` counts the survivors, and each block the reference
lacks takes its place among the new blocks from those ranks. A residual of
−32768 codes zero, and so does a block whose target payload a static price
(`emit_payload` walked at `bit_length + 1` bits a symbol, 8 a byte symbol,
plus raw bits) puts strictly below its residual's. Only the device knows the
frame's blocks and its new blocks' count, so the front counts a P-frame, at
`EncoderConfig::update_size` entries an update segment
(`EncoderConfig::resolved_update_size()`: the segment size unless the caller
sets one), over M plus the bound's blocks, and the device takes the blocks
the frame lacks and the survivors off (`counted_entries`, from two frame
words). `EncoderConfig::validate` holds the config's checks, which `create`
makes first.
The frame's rANS coding runs on the device (`detail::DeviceFrameWriter`,
built by `Encoder::create`, which fails if its kernels do not build): the
forward output stays in VRAM, the front also counts every model's symbols
and each block's coder steps and lays the steps out (`detail::StepLayout`,
whose header has the formulas: one workgroup scans the steps into each
entry's place, and the segments' slot sizes into their offsets), and its
last command (`encode_results.comp`) gathers what the host needs -- the
symbol counts, the layout's two totals, the frame words and the reject and
refusal words -- into a results slot (`core::HostRing`), from which the
host builds the tables (`frame_tables`) and sizes the coding's buffers. The
back half uploads the tables inline first, writes every block's steps in
parallel, codes one segment per invocation into its slot, scans the lengths
into the payload's offsets and a second results slot, and packs the streams
into a payload slot: in place on unified memory and lavapipe, the slot
itself; copied on a discrete GPU, the payload's likely size (the last
frame of its type's bytes per entry plus 25%), any rest gathered again into
its own output range by `finish` (a tail, `"  ..tail"`). The frame's
reference hash runs over the head and the payload unjoined (`frame_hash`'s
two-run form). Bulk buffers are device-local and retained. A segment is one
serial chain, so smaller segments give the device parallelism at a few
percent of size. A frame the device cannot hold is refused, never coded on
the host.
Both device coders work on entries, a P-frame's stream positions
(`ResidentBlocks`): its reference entries, each with a u32 mode, then its new
blocks, which alone the block list holds; an intra frame is all new blocks.
The count pass also flags what every writer refuses of the entries (a mode
past the alphabet, a written −32768), which `record_back` refuses in their
words (`check_entries`).
The `Decoder` decodes on the device (`detail::DeviceFrameReader`): the host
parses the frame (`parse_frame`) and uploads only its tables and payload.
One batch expands each table into a 4096-slot lookup, decodes one segment per
workgroup into the inverse transform's layout, judges the segments
(`rans_verdict.comp`: the smallest failing segment, by its fault or its first
coordinate against the block segment before's last) and scans the survivor
ranks (`survivor_ranks.comp`, on `scan.glsl`), and reads back 8 bytes, the
verdict and the survivor count, which `check_segment` and `check_survivors`
word exactly as the oracle's reader does. Every decoded entry stays in VRAM
for the merge, the apply and the inverse (`DctTransform::record_inverse` over
`ResidentBlocks`). One invocation
decodes a whole segment, so the format holds a segment to at most
`kMaxDeviceDecodeSegmentSize` (1,024) entries, keeping an invocation near
50 ms: every reader refuses a frame whose segment size or update size is
larger, before any device work, and every writer, `Encoder::create`
included, refuses such a size (`check_segment_size`), so every valid frame
decodes on the device. A frame past `maxStorageBufferRange` or free memory
is refused with its code (`InvalidArgument` or `OutOfMemory`); nothing
decodes on the host.
`Decoder::decode(frame, grid)` leaves the caller's grid holding exactly the
frame. It is `Decoder::record` into a batch, one submit and `finish` (the
2026-10-24 one-batch entry): the record is parsed in place in an input
slot, whose payload the reader binds at its byte offset, and the whole
decode is recorded with every count on the device. `detail::BlockLists`
merges a P-frame's surviving reference blocks with its new ones into the
frame's list and each block's sources, finds row 13, a new block on a
reference block, and diffs the hash table against the frame's list into the
blocks the grid lacks (missing) and those the frame lacks (gone), by count
and binary search (the header has the mechanism). Nothing sorts. A gate
kernel (`detail::FrameGate`) judges rows 11 to 13 and opens the grid's
removal of the gone blocks and allocation of the missing ones from those
device lists (`VoxelBlockGrid::record_remove`,
`VoxelHashMap::record_allocate`), whose rounds close it on failure; behind
it, sized by N on the device, the frame keeps shared blocks in their slots,
rewrites every voxel's `tsdf` and `weight`, and stamps `changed` only on a
block it leaves different. A frame the gate refuses moves the map's epoch
and tick and leaves the grid untouched. The counts are gathered into a
results slot, which `finish` reads. Attributes, geometry (exact
`voxel_size` / `trunc_dist`, off the header), parsing and the heap are all
checked before the grid is touched. The attributes must be `tsdf` and `weight` and
nothing else (`VoxelBlockGrid::attribute_count`), since a kept block would
carry any other one stale. A grid too small for the frame is `OutOfMemory`,
whether its heap has too few slots or its hash table cannot place the
blocks. Both are recovered by growing the grid and decoding again; the
decoder never grows one. Beside it, as `tsdf::Fuser` grows for fusion,
`decode_grid.hpp` holds a player's growth: `make_decode_grid` builds a
grid from a frame's header in `volume::block_grid_params`' layout, its heap
twice the frame's blocks, and `decode_growing` decodes into it, growing it
once on `OutOfMemory` and
decoding again twice at most on `IoError`; `record_growing` grows a heap
the frame outgrows before it records, and `finish_growing` decodes a frame
the device refused again from its own input slot. Both take the caller's ceiling,
`max_blocks`: a frame past it is refused before it is decoded, and a grow
past `make_decode_grid`'s grid for it before it is made, each as
`InvalidArgument`; the factory checks the header's block count against the
segment table as the decoder does. The examples pass the source grid's
slots. The factory contains host allocation
failures from header parsing through grid creation. The growing decode merges
only its successful attempt's metrics; if that merge runs out of host memory,
the caller's prior metrics stay unchanged and the decode still succeeds, since
its reference has committed. Lock contention that
outlasts the device rounds is `IoError`, never `OutOfMemory`, and a free heap
that refuses a removed block is `InvalidArgument`. A failure after the
grid has changed leaves it holding neither frame until a decode
succeeds. A P-frame decodes against the frame the `Decoder` decoded last,
which `detail::ReferenceStore` keeps as decoded: its quantized coefficients,
masks and block list on the device, in the inverse's layout, and its
`FrameReference` on the host, never rebuilt from a grid. The merge writes
each frame block's sources (the 2026-10-12 entry) into the store's spare,
`inter_apply.comp` makes the blocks there with integers alone, counting any
source past what is bound, and the inverse reads the spare in the same
batch; the spare becomes the reference only when the whole decode
succeeds. An intra frame takes the same apply from its payload alone (the
2026-10-13 entry). With `Decoder::set_deblock` on (off by default; the
2026-10-21 deblock entry), the same batch then smooths the frame across its
block boundaries (`detail::Deblock`, whose header has the filters): a line fit across
each face, and at coarser steps a Gaussian after it. The host picks the
filter per frame from the header alone: K and T, the step of DC and the
zigzag's next three coefficients at their basis peaks, in millimetres,
select a row of `kDeblockRule`, the study's rule; a fine step runs none. A
masked block, holding an unobserved voxel, keeps its exact decode (that
entry's 2026-10-23 amendment): a first dispatch, an invocation a block,
flags each fully observed block by slot in a word a slot zeroed earlier in
the batch and counts the masked blocks; every later kernel returns,
uniformly and before any probe or barrier, for a block whose own mask words
are not all ones; a line is fitted across a face only when the neighbour's
flag is set too, so neither side of a face with a masked block is
line-fitted; the Gaussian holds a masked neighbour's voxels at their
decoded values through all three passes, its unobserved ones dropping out;
and the count joins the results slot, beside the inverse's reject count.
The line passes run in place on `tsdf`, a dispatch an axis, a
64-lane workgroup a block and a lane a face line, probing the block and its
+axis neighbour through `vgridResolveOctants` (`block_neighbourhood.glsl`).
The Gaussian is one dispatch a block over the block and the 1-voxel halo of
its 26 neighbours, in shared memory; it writes the interior in place and
the shell, the voxels a neighbour's halo reads, into a snapshot by list
position, which a store dispatch copies back. A neighbour's observed voxels
come from its weights, and each block is found by coordinate. Only fully
observed blocks' `tsdf` change, and a block the deblock writes is stamped
`changed`. P-frames decode from the reference's coefficients, so the switch
may change between any two frames.
`Decoder::rebuild` reconstructs the last decoded frame from the held
reference with the switch as it is now, into the grid that decode left
(the grid diff checks its blocks first, on the device, gating the rest), in
one batch and one submit, so a paused player shows a toggle;
`Decoder::last_deblock` reports the filter, T, the frame's blocks and those
the fallback kept at their exact decode. Both classes report `StageMetrics`
(`"codec encode"` / `"codec decode"`). Their breakdown rows share no name;
the encoder's compaction is `"  ..encode set"`, apart from a fuse's
`"  ..active set"`. The private pieces under
`src/volumetric_kit/grid/codec/` are the `DctTransform`, the rANS tables,
the frame's layout and checks, its device coders, the block lists and the
reference store. The transform's forward takes a device block list to
`ResidentBlocks` — K
quantized 16-bit coefficients per block in 3-D zigzag order and a 16-word
observed mask, left on the device for the writer — and its inverse takes
them back, each recorded into the caller's `CommandBatch` over device-local
buffers. The forward writes into its caller's buffers, the
encoder's reference store; the compacted observed list and the sort's two
halves are retained and grow only when a call needs more capacity, releasing
the old buffer first and taking 1.5x headroom (`ensure_device_scratch`);
descriptor ranges and transfers use the current logical sizes.
They remain allocated until their owner is destroyed. The per-call staging
inside `CommandBatch` remains transient. The SDF is normalized by
`trunc_dist` before the transform and the steps are fractions of it, so the
decoder checks a frame's `trunc_dist` against its grid's before the inverse.
`CodecParams::validate` checks the scale,
each kept basis's weight and its effective step; weights beyond K are unused.
The scale and kept weights must be normal, so a flush-to-zero host or device
agrees on validity. It refuses a step
small enough for the ±32767 clamp to engage (√512 / 32767), and one past
`kMaxStep` (64), where every coefficient is 0 and a decoded one could
overflow to infinity. The forward
never reads an unobserved voxel's `tsdf`: it fills each one from the nearest
observed voxel along x, then y, then z, since a fused block's zeros there
are a step K = 32 cannot hold, and they decoded a partially observed block
with 2.8x the error. The transform refuses any block size but 8; more
entries than the grid's heap, or a list buffer too small for them; and, on
the device with one hash probe per block, any entry whose
coord the grid does not hold, which the inverse then writes nothing into.
Each block is found by that probe and its ptr never read, so the decoder
lists blocks it has just allocated by coordinate alone. "Observed" is
`volume::kObservedWeight`, the threshold the mesher reads too. The inverse
writes weight 1.0 on observed voxels and a fresh block's zeros elsewhere.
With `EncoderConfig::true_sdf` (the 2026-10-21 entry), `dct_surface.comp`
runs twice over the active list before the observed filter, a workgroup a
block, each lane taking the 8 cells anchored on its x-line, their corners
reached across the block's +x/+y/+z faces through
`volume/shaders/block_neighbourhood.glsl`, the mesher's neighbourhood probe
and corner addressing. A cell counts when its 8 voxels are observed. The
first pass sets, in each counted cell whose sign changes, the cut bit of
every corner with |tsdf| > √3 h + 1e-6 m; the second flags each block
holding a corner of a counted cell with no corner cut, whose sign changes or
that has a corner with |tsdf| < h; the filter appends the flagged blocks.
Metres on the raw `tsdf`, strict comparisons, below 0 inside, and every
write an `atomicOr`, so the order the workgroups run in changes nothing. The
forward reads the cut set: a cut voxel is unobserved, its mask bit 0 and its
value filled. The cut set (a bit a voxel slot) and the keep flags (a word a
block slot) are retained scratch too, sized to the grid's heap, zeroed in
each batch and bound on every call, since a regrown one is a new buffer.
The v3 frame (`bitstream.hpp`, the 2026-10-02 entry) carries the K kept
weights in its parameter header, `44 + 4K` bytes (300 at K = 64); older
versions are refused. `read_frame_info` reads them without entropy decoding.
Header byte 6 is the frame type (`FrameType`, which `read_frame_info`
reports): intra (0), or a P-frame (1), coded against the frame before it
as `bitstream.hpp` lays out and the 2026-10-12 entry explains. The device
writer and reader, and the oracle's, code both, and the `Decoder` decodes
both; every reader refuses a later type as `Unsupported`.
The payload is segments of R sorted
blocks (default 64). Each segment is one independent rANS stream
(`rans.hpp`: a 32-bit state, 16-bit words and 12-bit probabilities, integer
only, which the kernels code and the oracle pins byte for byte). Every integer
is coded as a class from a fixed per-frame table plus one raw field, its
sign the low bit. A partial mask is coded a z plane at a time and a line
at a time, each against the one before it (v2, 2026-10-01: the mask had
been half of room0's bits). The models and a block's grammar, its writer
and its reader, are one GLSL text that the kernels include and the oracle
compiles as C++ (`shaders/frame_models.glsl`, which `bitstream.cpp` compiles
too, and `shaders/frame_grammar.glsl`, 2026-10-08), and golden frames in the
tests pin v3's bytes for the device writer and the oracle's on every
machine. The format
requires strictly increasing coordinates, so a decoded list is duplicate-free:
the delta code cannot step backwards within a segment, and the reader checks
each segment's raw first coordinate against the one before. A flag bit other
than `kSectionRequired` is reserved, so a known section that sets one is
refused. `parse_frame` never reads outside its buffer, and it takes the
caller's `max_blocks`, because a frame's size cannot bound its block count
(probability-one blocks cost no bits). It checks that limit only once the
count agrees with the segment table, so a sound frame past it is
`OutOfMemory` and a corrupt count stays `InvalidArgument`. Every size is
checked in 64 bits so a 32-bit build refuses rather than wraps. A segment's
end check (every word consumed, the state back at its start) is a
**consistency** check, not an integrity one. It cannot see a flipped raw
bit, nor a symbol swapped for one of equal frequency, so integrity is the
transport's: the CRC-32 of the `.vrst` record that carries the frame
(`stream`).

### eval

`MeshDistance` (point-to-surface distance up to a reach,
through a hash of cells half the reach on a side, searched nearest first
and pruned by distance, over a **copy** of the triangles),
and `ReferenceMesh`, which indexes a reference once so a sweep can judge
many meshes against it, giving accuracy, coverage and an optional F-score. They refuse, with `Status`, what would read out of
bounds, overflow or mean nothing:
- a bad reach, or indices out of range;
- a corner that is not finite or past the cell keys' range;
- a reach so small that the triangles would average more than
  `kMaxCellsPerTriangle` cells each;
- a threshold past the reach.

The surface is every triangle but one collapsed to a point. The points
measured are the vertices those triangles use. The closest point is the face projection
when it lands inside, else the nearest edge, so a degenerate triangle
counts as the segment it collapses to and a thin one is measured to float
rounding, where Ericson's region test lost it now and then. A `stride` picks
vertices by a hash of their position, so the figures reproduce whatever
order marching cubes' atomics emitted the mesh in.

### stream

`grid_stream` writes and reads `.vrst` files, version 0: a `VRST` session
header, the records appended after it in `RECD` chunks, and on `finish` a
footer index (`INDX`, `TAIL`). The byte layout, the CRC and the rules a
record follows are written down once, in `stream/session.hpp`. Payloads are
stored and returned byte for byte. `TrackKind::Geometry`, a codec frame per
record with key meaning intra, its params empty or a mesh source's
transform, is the only kind defined; the writer takes any nonzero kind and
the reader passes an unknown one through.

**Writing** (`StreamWriter`, one thread). `create` makes the file with
`O_CREAT | O_EXCL`, so nothing at the path is touched, a symbolic link
included, then writes the session header and flushes the file and its
directory. Each `append` checks the record rules, then is one `writev` with
no buffering, so a killed process loses nothing it appended. `sync` flushes
to storage (`F_FULLFSYNC` on Apple); a power loss can lose what came after
the last one. A failed write or flush sticks: every later call returns it,
and the file reads as a killed writer's. `finish` writes the footer, flushes
and closes.

All fallible stream calls contain host allocation failures, including
validation diagnostics and copies of a saved failure, as `OutOfMemory`.
Validation or preparation allocation failures leave the writer retryable;
a failed write or flush remains sticky even if its diagnostic cannot be
allocated.

**Reading** (`StreamReader`). `open` checks the session header, then takes
each track's counts (`stats`) and key records (`key`, and `key_at` for the
last key at or before a pts) from a finished file's footer index, reading no
record. The checks a footer must pass are `Recovery::finished`'s: the
offset chain, the `TAIL` naming an `INDX` that ends where the `TAIL` starts,
is what makes the two a footer, and the key count is bounded before anything
is sized by it. Any other file `open` scans, one chunk header per chunk and
never a payload. `next` returns record headers in file order from a
`Position` the caller holds, skipping other chunks, and `read` checks the
header again and the payload's CRC, into caller memory such as a decoder's
input slot (allocating nothing) or a vector it sizes. Every method is `const` over `pread`, so
threads share one reader. A payload above `ReaderOptions::max_payload_bytes`
is `OutOfMemory` before any allocation.

**Recovery.** A whole header whose body runs past the end starts the torn
tail. A header that fails its CRC or holds what no chunk can starts a
search, in 8-byte steps, for the next `RECD`, `INDX` or `TAIL` with a valid
header; the bytes skipped are a damaged span, or with no chunk after them
the torn tail. `Recovery` reports them, and whether the file was opened from
its footer (`finished`) or the scan found a `TAIL` where the footer's starts
(`footer_damaged`). `next` passes damage the same way as the scan,
returning `InvalidArgument` once; a geometry consumer resumes at the next
key record.
A session header that is cut, out of range or fails a CRC is
`InvalidArgument`, and one of another version or with a flag set
`Unsupported`.

**In memory** (`stream/chunks.hpp`, the 2026-10-21 entry). `ChunkEncoder`
makes the session chunk and frames each record by the record rules, whose
state it holds; `StreamWriter` writes through one. `ChunkParser` takes bytes
as they arrive (`feed` copies them) and returns the session, then each
record, through the checks the reader's chunks pass. Each check runs once
the bytes it needs are in, a header's before its body is needed: B is at
most `kMaxSessionBytes` for the session (`InvalidArgument`) and
`ChunkParser::kMaxBodyBytes` (64 MiB) for any other chunk (`OutOfMemory`).
Both CRCs are checked on every chunk, a skipped one's too. Other tags, the
footer included, and a record setting an unknown flag are skipped. Nothing
resyncs: every call after a failure returns it. Drained between pieces, it
holds about one chunk. A `next` with no payload holds the record, its CRC
checked, for `read` into caller memory, as `StreamClient`'s does live.

Measured on an Apple M5 Max (Release, APFS, warm page cache): CRC-32 at
2.5 GB/s, 26 times the roughly 95 MB/s of a four-camera recording; appends of
737,280-byte (depth-sized) payloads at 1.95 GB/s; reads, CRC checked, at
2.1 GB/s. Opening 864,000 records of 1 KiB on 8 tracks (an hour at 240 a
second, 934 MB) takes 0.33 to 0.36 s by the scan, and from the footer 0.5 ms
with a key in 30 records or 14 ms with every record a key (a 27.6 MB index).

### networking

`grid_networking` serves one session live over TCP and reads it
(`networking/stream_server.hpp`, `networking/stream_client.hpp`, the
2026-10-21 entry). The wire is the file: the `VRST` chunk, then `RECD`
chunks as `StreamWriter` writes them, made by `ChunkEncoder` and read by
`ChunkParser`, with no footer and no framing of its own. A viewer sends
nothing; the server reads and drops what one does send, and closes on its
end. The port is 47801 (`kDefaultPort`).

**Serving** (`StreamServer`). `start` listens on a numeric IPv4 address, or
every interface, and starts an accept thread; one thread per viewer sends,
at most `max_viewers` (4) at once, first come, and a connection past them is
closed at accept. `publish` frames the record by the writer's rules, copies
its chunk once, takes one lock to append it to the GOP log and wakes the
viewers: it never waits on a socket. A viewer takes its next chunk only once
its socket polls writable, under 128 KiB unsent (`TCP_NOTSENT_LOWAT`), and
a P-frame never goes out without the records it is predicted from. At a
chunk boundary a viewer goes on from a newer key if one has come: newest key
wins, in either mode, so a lagging viewer's seq jumps to a key. By
default (`replay_gop`) the log holds one GOP, the newest key and every
record after it, as chunks shared by the viewers still to send them. A
viewer's first record, after its `VRST` chunk, is the GOP's key, and the
log keeps every record since. A record that would take the GOP past
`max_gop_bytes` (64 MiB) drops it and every record until the next key, as
a copy that fails does (`OutOfMemory`). Without `replay_gop` (the
2026-10-25 keys-on-demand entry), a viewer joins waiting for the next key, then
gets every record, in order, but by newest key wins; one that a record would
leave more than `max_gop_bytes` behind finishes its chunk under way and
waits for the next key again, alone, as every viewer does after a lost copy.
The log keeps only the records a viewer still has to send, none before the
newest key. Each join and each fall-back counts in `key_wanted()`, an
atomic any thread reads, which a live sender answers with a key at its next
frame; a lost key counts its waiting viewers again, a key within
`max_gop_bytes` counts none, and with `replay_gop` it stays 0. A fall-back
is logged at Info, and a record past `max_gop_bytes` on its own, which no
viewer can take, at Warning. A viewer whose end acknowledges no
byte for `stall_limit` (10 s) with bytes waiting is closed: a slow reader
stays while its end acknowledges a byte within the limit (one behind a large
receive buffer may not), as does one with nothing waiting. `viewers()`, an
atomic any thread reads, counts each viewer from its accepted connection,
before its first byte and while it waits for a key, until the server closes
it; a refused connection never counts. Viewers that join, leave or fail are
logged at Info; a refusal at Warning, the first and then one in 10 s at
most, the rest at Debug. The destructor stops every thread within a poll
slice (20 ms) and the write under way. One track is served (`Unsupported`
otherwise), and `publish` is called from one thread.

**Reading** (`StreamClient`). `connect` never blocks: a numeric IPv4 address
is parsed at once, and a name is looked up on a detached thread, which the
client abandons if it is destroyed first. `next(wait, …)` finishes the
lookup and the connect, trying each address in turn, then returns each
record, both CRCs checked, its `offset` that of the bytes received, waiting
at most `wait`. The receive buffer is 2 MiB, set before the connect. No
byte for `idle_limit` (10 s), counted from `connect`, ends the connection
with `IoError`, as do a refusal at every address, a reset and the sender's
end, which comes after every whole record; a failed check ends it with the
parser's error, which every later call returns again. Every error reads
`StreamClient: <host>:<port>: <reason>`, the address once, and the idle
limit's reason names the stage it ended in (the header lists the reasons).
The four-argument `next` overload also writes whether this call made
progress: a record returned, a successful lookup/connect step, or bytes
received. It resets that output on every call and keeps it true across a
later idle step. `next(0, ...)` reads at most one receive, so a false result
alone does not mean the input is drained; callers waiting on their own
controls drain while progress is true. On an error the output covers only
earlier successful steps, and the error remains terminal. The original
three-argument overload forwards without requesting progress.

**Sockets** (private `socket.hpp`, which gives the reasons). IPv4 only:
the server listens on it, and a name is looked up for it. Every socket is
non-blocking and close-on-exec, accepted ones included, and never raises
SIGPIPE; every call is retried on `EINTR`. The sender sets `TCP_NODELAY`
and the listener `SO_REUSEADDR`. The GOP log (`gop_log.cpp`), in either
mode, has no socket, thread or clock, and is tested alone.

### codec_stream

`grid_codec_stream` codes grids as frames into a `.vrst` file and to live
viewers (`codec_stream/codec_stream.hpp`, the 2026-10-26 and 2026-10-28
entries). It links `codec` and `networking`, and no fusion or sensor tier.

**The stream** (`CodecStream`). `create` takes the encoder's settings, the
session and `CodecOutputs`: a new `.vrst` file, the server's options, and
whether to decode. It refuses a session that is not one Geometry track,
then makes the server, so a port in use leaves no file, then the
file, which must be new, then the encoder and, with `decode`, the decoder,
so a refused output costs no kernel. A codec error removes the new file
again, as does destroying the stream unfinished before its first frame. The
file and the viewers get that session. `code(grid, set, pts_ns)`
encodes the grid, appends a record (a key exactly when an I-frame), serves
it once it is in the file, so no viewer gets a frame the file lacks, then
decodes it with `decode_growing` from the decoder's input buffer into a
player grid `make_decode_grid` built from the first frame, and returns a
`SentFrame`: its type, bytes, blocks, whether its payload took a tail, and
the encode's and the decode's rows. A set or pts out of order is refused
before it is encoded, so the writer never refuses one the encoder has
taken. Any other failure after the encode, of the append or the decode,
calls `force_intra()`, so the next frame never references one the file or
the player lacks; a frame the decoder refuses stays in the file and counts
in `coded()`. A record the server refuses is logged at Warning and fails no
frame, as failing would stop the recording. Before each frame's front half,
where the encoder picks I or P, the stream reads the server's
`key_wanted()`: a rise calls `force_intra()`, which stands until a frame is
returned, so joins between two frames make one key, a refused frame or a
front half recorded again keeps it, and the period counts from it.
`finish` closes the viewers and writes the footer, in one attempt.

**A frame in halves.** A caller with work of its own on the grid codes the
frame around its batch: `record_front` records the front half into that
batch, after what it holds, reading the server's key requests as `code`
does; `front_ran`, once the batch has resolved, gives the frame the device
rows its front half placed and the caller the rest; `redo` records the
front half again, in a batch of its own and waited on (`"wait: redo"`), for
a grid changed after the batch ran; and `finish_frame` codes the back half
in a batch of its own, waits (`"wait: frame"`), and writes, serves and
decodes the frame as `code` does, running the caller's `sent` once it is
written and served. The halves make `code`'s frames of the same grids. One
frame is in flight: a later front half or `code` supersedes it, and the
encoder refuses its pending frame. A moved-from stream (`valid()` false)
refuses the halves too.

## Examples

(`examples/`.) Every example reads a scenario with `--config path.json`;
`--help` prints that entry point. The optional core `config` tier owns the
JSON format, typed settings, application defaults and static input
validation, and `examples/common/config.hpp` handles startup and
diagnostics. `codec_config.hpp` only copies validated settings into runtime
API types and attaches resources, except that the codec's rules stay the
codec's: `encoder_config` maps `codec.quantization_table` to the weights
(`codec_quantization.hpp`, the tables' one home), and before GPU setup
refuses what `EncoderConfig::validate` refuses: a step out of the codec's
bounds, or a segment or update size past the longest. Stage rows print
through one table, `stage_table.hpp`; `info_log.hpp` prints Info lines a
sender's server logs, and `interrupt.hpp` turns Ctrl+C into a flag. These
headers are `grid_example_common`, which recon's examples link too.

The codec examples (`codec_mesh`, and recon's
`codec_replica`) stream through `CodecStream` ([codec_stream](#codec_stream)); their
report and the sweep are `examples/common/codec_report.hpp` and
`codec_sweep.hpp`. These and `codec_config.hpp` include no fusion, so
`codec_mesh` includes no fuser or sensor header; a fusing sender's config
and the figures of each set it sends (`sender_config`, `report_sent`) are
recon's (its `fuse_sender.hpp`). The report (`CodecReport`) keeps a
row per frame that went out whole, from its `SentFrame` (source index, frame
type, bytes, blocks, and encode and decode host and device ms), sums each
type's stage rows, and writes `output.csv` as each row comes; the core cannot
append, so the file is the report's own. A stream that does not decode
(`CodecOutputs::decode` off) has rows with empty decode columns.
`codec_outputs` maps `output.stream` and `output.serve` to `CodecOutputs`,
and `codec_session` gives the `.vrst` one Geometry track at the source's
nominal rate (the rig's `source.fps`, else 30 a second), with no session id
or start time, so a rerun writes the same bytes; a record's set is the
source index and its pts that index at 30 fps, as Replica frames carry no
time. `codec_outputs` serves without `replay_gop`, so the viewers of recon's `fuse_orbbec`
that join or fall behind get a key at the next set; `serve_stream`, which
cannot make a key, keeps it.

The report opens the CSV once the stream's file is made, and refuses one
that is that file under another name, through symlinks or case-insensitive
names (`InvalidArgument`); the stream, dropped unfinished before its first
frame, then removes its file for a retry.

**`codec_mesh`** codes one mesh or a numbered sequence of them (README,
Codec evaluation, has the keys). The core
escapes literal `%` characters in the scenario's directory before joining a
relative `source.path` (0.8.0), so only the user's path carries a
frame-number pattern.
Before any GPU work it refuses a malformed
frame-number pattern, a range on a plain path and any frame file that is
missing or not a regular file, checks a shell against the truncation
(`MeshSdfParams::validate`), and refuses a band too narrow for a true SDF
(`check_true_sdf_band`). It codes with `EncoderConfig::true_sdf` (the
2026-10-21 entry), the one caller that sets it. Its first pass loads each
frame, joins its bounds along `source.up_vector` (`mesh_io::extend_bounds`),
audits it and drops it, so memory does not grow with the range. In exact
arithmetic a proper similarity changes none of the audit's counts, so a
frame signed mode cannot
take is refused there; rounding can still change the degenerate count or a
volume's sign, which the second pass's audit decides. One transform for the range follows (`fit_transform` or
`scale_transform`) and places every frame unchanged, so a still point stays
still; it is printed once and stored as the Geometry track's params
(`mesh_io::encode_transform`). The stream is created after the GPU setup and
before any frame is converted, so a rerun onto it is refused before any
conversion or output. Each frame (`code_frame`) is loaded, placed, audited again on the
placed coordinates (the audit that decides, as rounding can make a triangle
degenerate), converted into the one source grid after a `clear`, so no block
of an earlier frame stays, and coded as one record: set the frame's
number, pts that number at 30 fps, seq from 0. The last frame alone is
extracted from the source and player grids with `mesh::MarchingCubes` and
compared, conversion error against the placed input apart from codec error
against the uncompressed surface. `codec.inspect_only` runs both passes on
the host. File parsing, placement and PLY export belong to `io`; the example
keeps its sequence (`mesh_sequence.*`) and the evaluator's host mesh
(`geometry_mesh.*`).

Both examples take `codec.quantization_table` (`uniform`, `band`, `radial`) and `codec.quantization_scale` for a
single configuration. Their common `codec.sweep` compares all three table
families at K = 64, 84, 120 and 512. The middle two retain complete frequency
shells; K = 64 cuts a shell and is kept as the existing baseline. Each
configuration has a warm-up and three timed codec rounds of one grid, all
intra whatever `codec.intra_period` (`sweep_encoder_config`); mesh
extraction and quality evaluation are outside those timings. Full frame
bytes include the quantization table. These are deterministic candidate
tables, not fitted JPEG image tables: the criterion is decoded geometry at
a given total size.

**`serve_stream <in.vrst> [--port N]`** serves a recording live: a sender
with no camera or GPU, so it builds wherever the examples do. It replays one
track's records from its first key read whole, each due at its pts from its
pass's start, in a loop until Ctrl+C (`ReplayLoop`, `stream_replay.hpp`).
Each pass is restamped, seq past the passes before and pts a pass's span (the
records' pts span and one mean step) later, so seq keeps rising and key pts
never go back, as `publish` requires. A record whose payload fails its CRC
is skipped with the P-frames after it, up to the pass's next key, and
printed in the first pass. It listens on every interface, on 47801 or N (0
takes a free one, which it prints). Bad arguments exit 2 with the usage; a
failure, a missing file or no key whose payload reads whole included,
exits 1.

## Next work

**Incremental mesh extraction was removed** (2026-10-06): nothing live
used it and its win was never measured. It comes back only with a
measurement on a large scan, against a full extract with view culling.
View-culled meshing has landed (2026-10-06); its win is correct and
unmeasured, so quote nothing until a run on a large scan says so. On
`mesh_io`, first-class glTF/GLB export. On `mesh`, the greppable
`TODO(mesh)`s: cross-block vertex sharing, per-vertex normals, measuring a
return of the default kernel to the per-triangle append (the per-block
reservation cost ~10% of the dispatch and was taken for incremental
extraction).

**On `codec`**, trimming a P-frame's host overhead at small R, where its
encode stays slower than intra's on the host alone (the 2026-10-17 freeze
entry). For true-SDF streams (the 2026-10-21 entry): the step-keyed deblock
in the player; one 0.2 m comparison reach in `codec_compare_options`, which
reaches max(truncation, 2 cm) today; and a sweep ladder in metres, since
fractions of a 16-voxel band are coarse steps. For the deblock (the deblock
entry's 2026-10-23 amendment): a guard finer than the block, at mask edges,
before the masked blocks of live RGB-D frames, nearly all of theirs, are
filtered.

**On `voxelize`**, mesh → TSDF has its two modes (2026-09-27), a second way
into the grid the codec encodes: a mesh sequence converts frame by frame. A
winding-number sign would fix at the root the sign errors a true-SDF stream
cuts (the 2026-10-21 true-SDF entry), and the folds within √3 voxels that it
cannot. Both modes cost the same. On Rafa2 the band's allocation costs what
recon's PERF.md MESH3 row gives, and the write what the 2026-10-21
`MeshIntegrator` entry gives. Smaller costs around them remain: writing only
the blocks the codec will code, chaining the write's submissions, and
computing `triangle_candidate_offsets` and uploading the mesh once a frame
rather than once to allocate and again to write.
