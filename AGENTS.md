# Shared repository instructions

This is the canonical working guide for all agents, including Codex and Claude
Code. Keep shared instructions here. `CLAUDE.md` imports this file; it must not
duplicate these rules.

## Project and scope

`volumetric_kit_grid` is the sparse voxel grid of the `volumetric_kit` family
and what reads it: voxel hashing, a triangle mesh's distance field, marching
cubes and the mesh containers, mesh files, mesh-to-mesh quality measurement,
the per-frame TSDF geometry codec, the `.vrst` session container, a session
served live over TCP, and the codec stream that joins them.

- It builds on [`volumetric_kit_core`](https://github.com/taojin-6/volumetric_kit_core)
  alone (fetched, pinned by commit), used under the core's names
  (`core::Status`, `core::Device`). Code two or more siblings need belongs
  there.
- [`volumetric_kit_recon`](https://github.com/taojin-6/volumetric_kit_recon)
  builds on grid (depth fusion, texturing, the live sender); grid includes
  nothing of recon's or sensor's (`tests/package_boundary.cmake`). Until it is
  a repository of its own, grid lives in recon's `grid/`, and a change that
  needs both lands as one recon PR.
- Vulkan compute (GLSL → SPIR-V) on every platform, with MoltenVK on Apple.
- Production only: no neural or research code. The `eval` tier is
  production infrastructure.
- The tiers work on any data: a tier never applies dataset preprocessing on
  its own; the caller opts in, and a tier's tests use synthetic data.

## Read what the task needs

| Task | Read |
| --- | --- |
| Build, options, consuming grid | [README.md](README.md#building) and the CMake files |
| Architecture, tier APIs, implementation | [DESIGN.md](DESIGN.md#tiered-architecture), then the tier's section |
| Changing a contract or a constraint | [Decision index](DECISIONS.md#decision-index), then the dated entries; newest context wins |
| What changed per version | [CHANGELOG.md](CHANGELOG.md) |

## Architecture and conventions

```text
base → volume → mesh → mesh_io
         │       └→ eval
         ├→ voxelize
         └→ codec ─┐
stream → networking → codec_stream
```

- No upward includes. `stream` links the core's base tier alone,
  `networking` links `stream`, `codec` and `voxelize` link `volume`, `eval`
  and `mesh_io` link `mesh`, and `codec_stream` links `codec` and
  `networking`. Assimp is an optional private backend of
  `grid_mesh_io_assimp`.
- C++17, no compiler extensions. Namespace `volumetric_kit::grid` (`vgrid::`
  in docs), nested per tier except `base`. Headers:
  `include/volumetric_kit/grid/<tier>/…`. Macros `VGRID_`; GLSL functions
  `vgrid…` and macros `VGRID_…`; CMake targets `volumetric_kit::grid_<tier>`.
- Fallible APIs return `core::Status` / `core::Result<T>` and propagate with
  `VKC_TRY` / `VKC_ASSIGN`; no exception crosses the API boundary, and grid
  builds with exceptions on for its handlers. Programmer errors use
  `VKC_CHECK`. grid logs with source `"vgrid"` (`base/log.hpp`).
- Include Vulkan through the core's `volumetric_kit/core/vulkan/vulkan.hpp`.
  Keep host PODs and GLSL `layout(scalar)` definitions byte-identical. A
  `create` that builds kernels first calls `check_device_requirements`.
- Kernel memory is device-local, reached through `CommandBatch`; the host
  reads small results back through the batch or a ring slot it binds.
- The exported GLSL is grid's GPU contract (`grid_types.glsl`,
  `voxel_color.glsl`, `dispatch_slices.glsl`); the hash table's GLSL and
  entries stay internal.
- Formats are versioned surface: codec frame v3 (type 1 frozen), `.vrst` v0
  and PLY. A rename never changes bytes.
- Full Doxygen on public classes and functions. `@ref` only grid's names;
  write the core's in backticks. No `friend` declarations; test through
  public APIs. Mark deferred work with a greppable `TODO:`.
- RAII handle owners follow recon's rules ([its AGENTS.md](https://github.com/taojin-6/volumetric_kit_recon/blob/main/AGENTS.md#raii-resource-types)):
  copy deleted, the move pair resetting every member, self-move guarded; an
  aggregate of owners deletes copy and move and `create()` returns a
  `unique_ptr`.

## Versions

Before 1.0 an API, ABI, exported-GLSL or format change bumps the minor in the
PR that makes it, and behaviour a consumer relies on bumps the patch; each
version gets a CHANGELOG section. Consumers pin SHAs and check floors with the
core's `vkc_require_sibling(volumetric_kit::grid <min>)`.

## Build and validation

```sh
# grid_root: grid's checkout, or grid/ in recon's
cmake -S "$grid_root" -B "$grid_root/build" -DCMAKE_BUILD_TYPE=Release
cmake --build "$grid_root/build" --parallel
ctest --test-dir "$grid_root/build" --output-on-failure
```

- Always set the build type; use Release for measurements and report the
  build type, hardware, workload and host/device timing apart.
- Tests are `main()`s of their own on the core's test support and grid's
  (`grid_test_support`); GPU tests run on the core's policy:
  `VKC_TEST_VALIDATION=1` fails a test on any error the validation layer
  reports, and `VKC_REQUIRE_VULKAN_DEVICE=1` fails rather than skips without
  a device.
- `VGRID_WITH_ASSIMP` builds the optional mesh loader; Assimp is an installed
  prerequisite, never fetched.
- Keep first-party code warning-clean with `VGRID_WARNINGS_AS_ERRORS=ON`.
- Work in a branch and a git worktree; use Conventional Commits.

## Keeping the guidance current

Keep this file concise. Implementation detail goes in DESIGN.md, decision
evidence in DECISIONS.md. When changing a locked decision, update its index
entry and dated rationale in the same commit, and the DESIGN.md contract with
it.
