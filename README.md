# volumetric_kit_grid

[![ci](https://github.com/taojin-6/volumetric_kit_grid/actions/workflows/ci.yml/badge.svg)](https://github.com/taojin-6/volumetric_kit_grid/actions/workflows/ci.yml)

The **sparse voxel grid** of the `volumetric_kit` family and what reads it:
voxel hashing on the GPU, a triangle mesh's distance field, marching cubes and
the mesh containers, mesh files, mesh-to-mesh quality measurement, the
per-frame TSDF geometry codec, the `.vrst` session container, and a session
served live over TCP, all on Vulkan compute (MoltenVK on Apple).

It builds on [`volumetric_kit_core`](https://github.com/taojin-6/volumetric_kit_core)
alone. [`volumetric_kit_recon`](https://github.com/taojin-6/volumetric_kit_recon)
fuses depth into its grid and sends what it codes;
[`volumetric_kit_gfx`](https://github.com/taojin-6/volumetric_kit_gfx) draws
the meshes it extracts. grid was extracted from recon's `grid/`, with its
history (see [CHANGELOG.md](CHANGELOG.md)).

> Start with [AGENTS.md](AGENTS.md) for working conventions and commands.
> [DESIGN.md](DESIGN.md) describes the architecture and implementation;
> [DECISIONS.md](DECISIONS.md#decision-index) holds the dated decisions.

## Building

Requires CMake ≥ 3.21, a C++17 compiler and the Vulkan toolchain (MoltenVK on
Apple: `brew install glm molten-vk vulkan-loader vulkan-headers shaderc
spirv-tools`).

```sh
cmake -S "$grid_root" -B "$grid_root/build" -DCMAKE_BUILD_TYPE=Release
cmake --build "$grid_root/build" --parallel
ctest --test-dir "$grid_root/build" --output-on-failure
```

The GPU tests follow the core's test policy (its README, "Build and test"):
`VKC_REQUIRE_VULKAN_DEVICE`, `VKC_TEST_VALIDATION` and
`VKC_TEST_SYNC_VALIDATION`.

Consume it from another CMake project with `FetchContent`, pinned by commit
(grid has no install rules), then link a tier (e.g. `volumetric_kit::grid_codec`) or
the umbrella `volumetric_kit::grid`. Fetched, grid builds its library tiers
alone: its tests, examples and `-Werror` (`VGRID_WARNINGS_AS_ERRORS`) default
on only when it is the top-level project. `VGRID_BUILD_TEST_SUPPORT` builds
`volumetric_kit::grid_test_support`, the helpers its tests share, for a
sibling's tests. A consumer that needs a grid at least as new as a given
version calls the core's `vkc_require_sibling(volumetric_kit::grid
<min-version>)` after `FetchContent_MakeAvailable(volumetric_kit_grid)`.

A consumer's kernels include grid's exported GLSL as grid's own do, through
the core's `INCLUDE_TARGETS`: `volumetric_kit::grid_volume` exports the voxel
grid's contract (`volumetric_kit/grid/volume/shaders/grid_types.glsl`) and its
colour attribute's format (`.../voxel_color.glsl`), and
`volumetric_kit::grid_base` the split dispatch
(`volumetric_kit/grid/base/shaders/dispatch_slices.glsl`). The hash table's
GLSL is not exported.

```cmake
vkc_compile_shaders(my_kernels INCLUDE_TARGETS volumetric_kit::grid_volume
                    SPIRV_VAL_ARGS --scalar-block-layout SHADERS my.comp)
```

grid needs the core at `VGRID_CORE_MIN_VERSION` (0.30.0) or newer, with its
vulkan tier, and checks it with the core's `vkc_require_core`. A project that
also fetches another sibling declares `volumetric_kit_core` first and sets
`VKC_WITH_VULKAN ON` before its first `FetchContent_MakeAvailable`; pass
`-DFETCHCONTENT_SOURCE_DIR_VOLUMETRIC_KIT_CORE=<path>` to build against a
local core checkout. Examples and tests also need the core's config tier.

### Mesh files

`volumetric_kit::grid_mesh_io` exports binary PLY (`mesh_io/ply_writer.hpp`)
and holds the host triangle containers and the opt-in normalization helpers
(`mesh_io/mesh_normalization.hpp`). Static mesh import is a separate optional
target, `volumetric_kit::grid_mesh_io_assimp`, with Assimp as a private
implementation dependency. Install Assimp and enable it explicitly:

```sh
brew install assimp                       # macOS
sudo apt install libassimp-dev zlib1g-dev   # Debian/Ubuntu
cmake -S "$grid_root" -B "$grid_root/build" -DCMAKE_BUILD_TYPE=Release \
    -DVGRID_WITH_ASSIMP=ON
```

`mesh_io::load_mesh(path)` in `mesh_io/mesh_io.hpp` returns owned positions
and triangle indices. It triangulates polygons, applies scene-node transforms
and instances, preserves winding under reflections, and joins exactly
coincident positions. OBJ, PLY and glTF are covered by tests; other formats
depend on the installed Assimp build. Materials/textures are discarded and
animated/skinned/morphed assets are refused. Import does not impose physical
units or a target height: normalization such as Rafa2's 1.7 m convention is a
caller operation, with the opt-in normalization helpers.
`mesh_io::load_textured_mesh(path)` in `mesh_io/textured_mesh.hpp` loads the
same triangles with one UV per vertex, a position repeated at UV seams, and
the path of the diffuse map. The default build does not need Assimp.
Cross-compiles must supply Assimp for the target platform, not a host
package.

## Codec evaluation

`codec_mesh` codes one triangle mesh or a numbered sequence of them: it loads
each through `mesh_io::load_mesh`, uses `voxelize::MeshIntegrator` to convert
it to a TSDF, compresses and decodes that grid, then extracts the decoded
surface with marching cubes. Build it with `VGRID_WITH_ASSIMP=ON` after
installing the [mesh file prerequisites](#mesh-files); recon's
`codec_replica` codes a fused room sequence the same way.
Its `codec` section selects `quantization_table` (`uniform`, `band`,
`radial`), `quantization_scale`, `segment_size`, `intra_period` (an I-frame
every N coded frames, zero-motion P-frames between; 1, the default, codes
every frame intra), `update_size` at a period above 1 (a P-frame's entries
per update segment; 0, the default, takes the segment size), and the shared
rate-distortion `sweep`, which codes intra.
`output.stream` records every coded frame in a new `.vrst` file, each I-frame
a key (see [the stream container](DESIGN.md#stream)), and `output.csv` writes
a row per decoded frame: its type (`I` or `P`), bytes, blocks, and encode and
decode host and device ms.

A mesh scenario gives the head-up direction (`source.up_vector`) and either
the input's unit (`source.metres_per_unit`) or a height to fit
(`source.height_m`). The example rotates head-up to +Y, centres X/Z and puts
the lowest point at Y=0, with one transform fitted to every frame it codes,
printed once and stored in the stream; the files are left unchanged.
Rafa2's OBJs declare no unit and its person is tilted, so its scenarios give
the visually checked long principal axis and fit 1.7 m: a height
normalization, not an assertion about the source's unit.

A sequence's `source.path` holds a printf-style frame number, such as
`Frame_%05d_textured.obj` (`%d` or `%0Nd` once; `%%` for a literal `%`), and
`run.first` and `run.frames` select an exact range, every file checked
before any work. Each frame is converted alone and coded as one record whose
set is its number, at 30 fps. A mesh's field is a true distance, so
`codec_mesh` codes it without the observed mask
(`EncoderConfig::true_sdf`), in a band of at least 15 voxels from the
surface (`fusion.truncation_m`), plus the shell's half-thickness in shell
mode; it refuses a narrower one before any GPU work.
`codec_mesh_sequence.json` codes Rafa2's frames 1–150 at 5 mm in a 16-voxel
band, at K 56 and scale 0.040877; its frames 151–300 replay 150–1, so stop
at 150. Frames 74–99 and 128–131 have holes, so it sets
`source.allow_holes: true`: signed mode then takes a mesh whose only defect
is boundary edges, each hole growing a skirt past its rim, which the coding
cuts where it flips sign farther than √3 voxels out.

```sh
cmake -S "$grid_root" -B "$grid_root/build" -DCMAKE_BUILD_TYPE=Release \
  -DVGRID_WITH_ASSIMP=ON
cmake --build "$grid_root/build" --parallel
# Edit source.path in these files to point at the installed datasets.
"$grid_root/build/examples/codec_mesh/codec_mesh" \
  --config "$grid_root/config/examples/codec_mesh.json"
"$grid_root/build/examples/codec_mesh/codec_mesh" \
  --config "$grid_root/config/examples/codec_mesh_sequence.json"
```

The comparisons, sweep and `output.prefix` PLYs are the last frame's:
`_input.ply` (normalized original), `_source.ply` (uncompressed TSDF surface)
and `_decoded.ply`. Conversion error, codec-only error and total error are
reported separately. `codec.inspect_only` checks every frame's input and
normalization without a GPU ([config/README.md](config/README.md)).
The signed mode checks indexed topology and winding; it does not establish
freedom from self-intersections. `source.sdf_mode: "shell"` supports open or
inconsistently wound meshes, with an intentional surface offset.

Use a Release build for measurements. The current format is v3, with a
global scale and the K kept bases' quantization weights in each frame; older
versions are refused. See [the codec contract](DESIGN.md#codec) and
[the measurement record](DECISIONS.md#2026-10-02--per-basis-quantization-and-a-normalized-mesh-codec-fixture).

## Live streaming

`volumetric_kit::grid_networking` serves a session to viewers over TCP as it
is recorded (`networking/stream_server.hpp`) and reads it as it arrives
(`networking/stream_client.hpp`), byte for byte the `.vrst` file's chunks;
see [the networking contract](DESIGN.md#networking).
`volumetric_kit::grid_codec_stream` codes grids into a `.vrst` and serves
them ([the codec stream contract](DESIGN.md#codec_stream)); recon's
`fuse_orbbec` serves a live rig through it. `serve_stream` serves a
recording live, paced by its pts and looping until Ctrl+C, with no camera or
GPU; a viewer connects to the sender's `host:port`:

```sh
"$grid_root/build/examples/serve_stream/serve_stream" rafa2.vrst  # port 47801
"$grid_root/build/examples/serve_stream/serve_stream" rafa2.vrst --port 0
```

Both listen on every interface, with no authentication or encryption: any
host that reaches the port can watch, and the first four to connect keep
their slots while they stay (`max_viewers`; `fuse_orbbec` takes its count
from the scenario's `output.max_viewers`, 4 by default). Both print each
viewer joining and leaving on stderr; one the sender closes, or whose
connection fails, ends with `closed: <why>` or `failed: <why>` in place of
`left`:

```text
[vgrid info] StreamServer: viewer 192.168.1.20:52311 joined, 1 of 4
[vgrid info] StreamServer: viewer 192.168.1.20:52311 left
```

## License

MIT — see [LICENSE](LICENSE).
