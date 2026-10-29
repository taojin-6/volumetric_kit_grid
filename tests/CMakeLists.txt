# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Tao Jin

# Every test is a main() of its own on the core's test support: CHECK,
# VKC_TEST_SANITIZED and the scratch directory (core_test_base), and grid's
# shared helpers (grid_test_support, support/). A GPU test also links
# core_test_policy, and one that fails or measures allocations
# core_test_allocation_failure.
link_libraries(volumetric_kit::core_test_base volumetric_kit::grid_test_support)

# A sibling that fetches grid (subproject/) takes each dependency this build
# fetched as a local source, so it needs no network.
file(
  GLOB _vgrid_fetched
  LIST_DIRECTORIES true
  "${FETCHCONTENT_BASE_DIR}/*-src")
set(_vgrid_fetched_sources)
foreach(_vgrid_source IN LISTS _vgrid_fetched)
  get_filename_component(_vgrid_name "${_vgrid_source}" NAME)
  string(REGEX REPLACE "-src$" "" _vgrid_name "${_vgrid_name}")
  string(TOUPPER "${_vgrid_name}" _vgrid_name)
  list(APPEND _vgrid_fetched_sources
       "-DFETCHCONTENT_SOURCE_DIR_${_vgrid_name}=${_vgrid_source}")
endforeach()
# A dependency this build took from a local source instead, as given.
get_cmake_property(_vgrid_cache CACHE_VARIABLES)
foreach(_vgrid_var IN LISTS _vgrid_cache)
  if(_vgrid_var MATCHES "^FETCHCONTENT_SOURCE_DIR_")
    if(NOT "${${_vgrid_var}}" STREQUAL "")
      list(APPEND _vgrid_fetched_sources "-D${_vgrid_var}=${${_vgrid_var}}")
    endif()
  endif()
endforeach()

# The base tier's own pieces: the version and the log seam's "vgrid" source.
add_executable(grid_base_smoke_test base_smoke_test.cpp)
target_link_libraries(grid_base_smoke_test PRIVATE volumetric_kit::grid_base)
vgrid_target_warnings(grid_base_smoke_test)
add_test(NAME grid_base_smoke COMMAND grid_base_smoke_test)

# The package boundary (package_boundary.cmake): no file under grid/ includes a
# header or GLSL file of recon or sensor, the packages above it.
add_test(NAME grid_package_boundary
         COMMAND "${CMAKE_COMMAND}" "-DSOURCE_DIR=${PROJECT_SOURCE_DIR}" -P
                 "${CMAKE_CURRENT_SOURCE_DIR}/package_boundary.cmake")

# volume coordinate math: pure host transforms (world <-> voxel <-> block,
# negative-coordinate block indexing, truncation band), the params' validation
# and the block layout (block_grid_params); and the tests' host oracle of the
# spatial hash (volume_hash_oracle.hpp). CPU-only (no device), so these always
# run -- not gated on a Vulkan driver.
add_executable(grid_volume_coords_test volume_coords_test.cpp)
target_link_libraries(
  grid_volume_coords_test PRIVATE volumetric_kit::grid_volume
                                  volumetric_kit::core_test_allocation_failure)
vgrid_target_warnings(grid_volume_coords_test)
add_test(NAME grid_volume_coords COMMAND grid_volume_coords_test)

add_executable(grid_volume_hash_test volume_hash_test.cpp)
target_link_libraries(grid_volume_hash_test PRIVATE volumetric_kit::grid_volume)
vgrid_target_warnings(grid_volume_hash_test)
add_test(NAME grid_volume_hash COMMAND grid_volume_hash_test)

# The volume tier is camera-free: no header, source or shader of it includes the
# camera vocabulary, core/camera_params.hpp or core/shaders/pinhole.glsl
# (volume_camera_free.cmake).
add_test(NAME grid_volume_camera_free
         COMMAND "${CMAKE_COMMAND}" "-DSOURCE_DIR=${PROJECT_SOURCE_DIR}" -P
                 "${CMAKE_CURRENT_SOURCE_DIR}/volume_camera_free.cmake")

# The grid's exported GLSL, built through INCLUDE_TARGETS
# volumetric_kit::grid_volume as a consumer's kernels are, against volume's host
# types on the device: grid_types.glsl's tick comparison, in-block index and
# struct layouts (shaders/grid_types_cases.comp), and voxel_color.glsl's colour
# format, decode, running-mean encode, first observation and the mesher's read
# (shaders/voxel_color_cases.comp). The colour test also holds
# volume/voxel_color.hpp to the published sRGB curve, its own reference, on the
# host, bit for bit; recon's recon_tsdf_voxel_color_space holds it to the sensor
# boundary's copy. Skip the device half where no device is present.
add_executable(grid_volume_grid_types_test volume_grid_types_test.cpp)
vgrid_embed_shaders(
  grid_volume_grid_types_test INCLUDE_TARGETS volumetric_kit::grid_volume
  SHADERS "${CMAKE_CURRENT_SOURCE_DIR}/shaders/grid_types_cases.comp")
target_link_libraries(
  grid_volume_grid_types_test
  PRIVATE volumetric_kit::grid_volume Vulkan::Vulkan
          volumetric_kit::core_test_policy)
vgrid_target_warnings(grid_volume_grid_types_test)
add_test(NAME grid_volume_grid_types COMMAND grid_volume_grid_types_test)

add_executable(grid_volume_voxel_color_test volume_voxel_color_test.cpp)
vgrid_embed_shaders(
  grid_volume_voxel_color_test INCLUDE_TARGETS volumetric_kit::grid_volume
  SHADERS "${CMAKE_CURRENT_SOURCE_DIR}/shaders/voxel_color_cases.comp")
target_link_libraries(
  grid_volume_voxel_color_test
  PRIVATE volumetric_kit::grid_volume Vulkan::Vulkan
          volumetric_kit::core_test_policy)
vgrid_target_warnings(grid_volume_voxel_color_test)
add_test(NAME grid_volume_voxel_color COMMAND grid_volume_voxel_color_test)

# volume GPU hash map: build a VoxelHashMap, allocate a cube of block coords,
# compact, and verify the active set round-trips on the real driver (MoltenVK /
# NVIDIA). The table's entry layout is internal (hash.hpp), reached by include
# path. Skips where no device is present.
add_executable(grid_volume_hash_map_test volume_hash_map_test.cpp)
target_link_libraries(
  grid_volume_hash_map_test PRIVATE volumetric_kit::grid_volume Vulkan::Vulkan
                                    volumetric_kit::core_test_policy)
target_include_directories(
  grid_volume_hash_map_test
  PRIVATE ${PROJECT_SOURCE_DIR}/src/volumetric_kit/grid/volume)
vgrid_target_warnings(grid_volume_hash_map_test)
add_test(NAME grid_volume_hash_map COMMAND grid_volume_hash_map_test)

# volume block grid: declare independent SoA per-voxel attributes, verify they
# are distinct + correctly-sized buffers keyed by the block pool, the composed
# map allocates, and the error / move paths. Skips with no device.
add_executable(grid_volume_block_grid_test volume_block_grid_test.cpp)
target_link_libraries(
  grid_volume_block_grid_test
  PRIVATE volumetric_kit::grid_volume Vulkan::Vulkan
          volumetric_kit::core_test_policy)
vgrid_target_warnings(grid_volume_block_grid_test)
add_test(NAME grid_volume_block_grid COMMAND grid_volume_block_grid_test)

# volume block stamps: every allocation path stamps the blocks it asks for with
# the map's tick, the block pass stamps those holding weight, and
# free_stale_blocks frees those neither for more than max_age ticks; resize
# keeps the records, remove and clear zero them; a pass split into several
# dispatches (base/dispatch.hpp's test seam). Skips with no device.
add_executable(grid_volume_block_stamps_test volume_block_stamps_test.cpp)
target_link_libraries(
  grid_volume_block_stamps_test
  PRIVATE volumetric_kit::grid_volume Vulkan::Vulkan
          volumetric_kit::core_test_policy)
vgrid_target_warnings(grid_volume_block_stamps_test)
add_test(NAME grid_volume_block_stamps COMMAND grid_volume_block_stamps_test)

# voxelize mesh integrate: write a mesh's truncated distance field (signed
# through pseudonormals, or as a shell) and check every voxel of every written
# block against a brute-force host reference -- an independent closest point,
# and the winding number for inside/outside -- on a sharp tetrahedron, a dented
# cube, an open quad and a triangle soup, plus determinism (a split binning
# among it) and every refusal. Skips with no device.
add_executable(grid_voxelize_mesh_integrate_test
               voxelize_mesh_integrate_test.cpp)
target_link_libraries(
  grid_voxelize_mesh_integrate_test
  PRIVATE volumetric_kit::grid_voxelize Vulkan::Vulkan
          volumetric_kit::core_test_policy)
vgrid_target_warnings(grid_voxelize_mesh_integrate_test)
add_test(NAME grid_voxelize_mesh_integrate
         COMMAND grid_voxelize_mesh_integrate_test)

# voxelize mesh integrate, exactly: the kernel's early exit writes, bit for bit,
# what a brute force (shaders/mesh_integrate_reference.comp) writes measuring
# every triangle of each voxel's bin -- on shared vertices with opposite
# windings, duplicates, near-coincident sheets, slivers, metre-long triangles, a
# far sphere, bins past the kernel's sort, three block sizes, both modes. The
# brute force binds the hash entries (volume/hash_map_internal.hpp, by src/).
# Skips with no device.
add_executable(grid_voxelize_mesh_integrate_exact_test
               voxelize_mesh_integrate_exact_test.cpp)
target_link_libraries(
  grid_voxelize_mesh_integrate_exact_test
  PRIVATE volumetric_kit::grid_voxelize Vulkan::Vulkan
          volumetric_kit::core_test_policy)
target_include_directories(grid_voxelize_mesh_integrate_exact_test
                           PRIVATE ${PROJECT_SOURCE_DIR}/src)
vgrid_embed_shaders(
  grid_voxelize_mesh_integrate_exact_test SHADERS
  "${CMAKE_CURRENT_SOURCE_DIR}/shaders/mesh_integrate_reference.comp")
vgrid_target_warnings(grid_voxelize_mesh_integrate_exact_test)
add_test(NAME grid_voxelize_mesh_integrate_exact
         COMMAND grid_voxelize_mesh_integrate_exact_test)

# The core parses and validates scenarios. This test pins the conversion of the
# codec_mesh templates and of written scenarios into the runtime codec options
# (the codec's steps checked here, not by core), without opening a GPU.
add_executable(grid_example_config_test example_config_test.cpp)
target_link_libraries(grid_example_config_test PRIVATE grid_example_common)
# codec_config.hpp maps the outputs into the codec stream's CodecOutputs and a
# mesh source into voxelize's MeshSdfParams, and the sweep's report compares
# with eval's. grid_codec_stream brings the codec tier: naming it too would
# repeat its archive on the link line.
target_link_libraries(
  grid_example_config_test
  PRIVATE volumetric_kit::core_config volumetric_kit::grid_eval
          volumetric_kit::grid_codec_stream volumetric_kit::grid_voxelize)
target_compile_definitions(
  grid_example_config_test
  PRIVATE VGRID_CONFIG_FIXTURES="${PROJECT_SOURCE_DIR}/config/examples")
vgrid_target_warnings(grid_example_config_test)
add_test(NAME grid_example_config COMMAND grid_example_config_test)

# The live examples' Ctrl+C (examples/common/interrupt.hpp): a raised SIGINT
# sets the flag and restores its default handler, SIGTERM's still caught. Host
# only.
add_executable(grid_example_interrupt_test example_interrupt_test.cpp)
target_link_libraries(grid_example_interrupt_test PRIVATE grid_example_common)
vgrid_target_warnings(grid_example_interrupt_test)
add_test(NAME grid_example_interrupt COMMAND grid_example_interrupt_test)

# The senders' log (examples/common/info_log.hpp): Info, Warning and Error in
# the default sink's line form, Debug dropped, and concurrent lines whole. Host
# only.
add_executable(grid_example_info_log_test example_info_log_test.cpp)
target_link_libraries(grid_example_info_log_test PRIVATE grid_example_common)
target_link_libraries(grid_example_info_log_test
                      PRIVATE volumetric_kit::core_base Threads::Threads)
vgrid_target_warnings(grid_example_info_log_test)
add_test(NAME grid_example_info_log COMMAND grid_example_info_log_test)

# The examples' one stage table (examples/common/stage_table.hpp). Host only.
add_executable(grid_example_stage_table_test example_stage_table_test.cpp)
target_link_libraries(grid_example_stage_table_test PRIVATE grid_example_common)
target_link_libraries(grid_example_stage_table_test
                      PRIVATE volumetric_kit::grid_base)
vgrid_target_warnings(grid_example_stage_table_test)
add_test(NAME grid_example_stage_table COMMAND grid_example_stage_table_test)

# The codec examples' report of a stream (examples/common/codec_report.hpp): on
# the host, a row's type and host and device milliseconds, its CSV line, a frame
# type's sizes, a GOP's bytes and bitrate, a frame's pts, the comparison options
# the codec examples share, and a path that is the stream's or the CSV's file
# refused; on a device, a CSV refused after the stream is made or because it is
# the stream's file, a row per frame matching the frame and each type's stage
# rows its own, the CSV replaced on create and written as rows come, and a CSV
# that stops taking rows refused. The device part skips with no device.
add_executable(grid_example_codec_report_test example_codec_report_test.cpp)
target_link_libraries(grid_example_codec_report_test
                      PRIVATE grid_example_common)
target_link_libraries(
  grid_example_codec_report_test
  PRIVATE volumetric_kit::core_config volumetric_kit::core_test_policy
          volumetric_kit::grid_eval volumetric_kit::grid_codec_stream
          volumetric_kit::grid_voxelize Vulkan::Vulkan)
vgrid_target_warnings(grid_example_codec_report_test)
add_test(NAME grid_example_codec_report COMMAND grid_example_codec_report_test)

# codec_mesh's sequence (examples/codec_mesh/mesh_sequence.hpp), where the mesh
# example is built. On the host: the frame-number pattern, every file checked
# before any load, one transform for the whole range, signed mode's rule and
# source.allow_holes, and inspection. On a device: each frame converted into a
# cleared grid, and code_frame's records and the transform in the stream.
# examples/ defines grid_example_mesh after this directory. The device part
# skips with no device.
if(VGRID_BUILD_EXAMPLES AND TARGET grid_mesh_io_assimp)
  add_executable(grid_example_mesh_sequence_test example_mesh_sequence_test.cpp)
  target_link_libraries(
    grid_example_mesh_sequence_test
    PRIVATE grid_example_mesh volumetric_kit::core_test_policy Vulkan::Vulkan)
  vgrid_target_warnings(grid_example_mesh_sequence_test)
  add_test(NAME grid_example_mesh_sequence
           COMMAND grid_example_mesh_sequence_test)
endif()

# volume delete: allocate a cube, remove a subset, verify the survivors + heap
# reuse (chain splice / successor pull-up) on the real driver. Skips where no
# device is present.
add_executable(grid_volume_delete_test volume_delete_test.cpp)
target_link_libraries(
  grid_volume_delete_test PRIVATE volumetric_kit::grid_volume Vulkan::Vulkan
                                  volumetric_kit::core_test_policy)
vgrid_target_warnings(grid_volume_delete_test)
add_test(NAME grid_volume_delete COMMAND grid_volume_delete_test)

# volume recorded rounds: record_allocate, record_remove and settle in a
# caller's batch against the blocking calls, the device count and gate, the
# rounds' bound under planted contention, one batch holding both kinds with the
# grid's zeroing first, the split dispatch (base/dispatch.hpp's test seam),
# staleness, the steady-state counters and the result slots. Skips with no
# device.
add_executable(grid_volume_recorded_test volume_recorded_test.cpp)
target_link_libraries(
  grid_volume_recorded_test PRIVATE volumetric_kit::grid_volume Vulkan::Vulkan
                                    volumetric_kit::core_test_policy)
vgrid_target_warnings(grid_volume_recorded_test)
add_test(NAME grid_volume_recorded COMMAND grid_volume_recorded_test)

# volume resize: allocate, grow the table to more buckets, and verify the active
# set survives + further allocation past the old capacity works. Skips with no
# device.
add_executable(grid_volume_resize_test volume_resize_test.cpp)
target_link_libraries(
  grid_volume_resize_test PRIVATE volumetric_kit::grid_volume Vulkan::Vulkan
                                  volumetric_kit::core_test_policy)
vgrid_target_warnings(grid_volume_resize_test)
add_test(NAME grid_volume_resize COMMAND grid_volume_resize_test)

# Stateless volume growth: doubling, an explicit minimum, content retention, and
# rejection of invalid or unreachable sizes. The table's entry layout is
# internal (hash.hpp), reached by include path. Skips (exit 0) with no device.
add_executable(grid_volume_grid_growth_test volume_grid_growth_test.cpp)
target_link_libraries(
  grid_volume_grid_growth_test
  PRIVATE volumetric_kit::core_test_policy volumetric_kit::grid_volume
          Vulkan::Vulkan)
target_include_directories(
  grid_volume_grid_growth_test
  PRIVATE ${PROJECT_SOURCE_DIR}/src/volumetric_kit/grid/volume)
vgrid_target_warnings(grid_volume_grid_growth_test)
add_test(NAME grid_volume_grid_growth COMMAND grid_volume_grid_growth_test)

# volume allocate-from-depth: unproject a synthetic one-pixel depth frame and
# verify exactly the expected (2*tb+1)^3 truncation-band block cube is allocated
# on the real driver -- the camera unprojection + band dilation, plus the
# DepthPinhole scalar-layout upload -- and pin the blocks, heap count and stamps
# three posed frames allocate. Skips where no device is present.
add_executable(grid_volume_allocate_test volume_allocate_test.cpp)
target_link_libraries(
  grid_volume_allocate_test PRIVATE volumetric_kit::grid_volume Vulkan::Vulkan
                                    volumetric_kit::core_test_policy)
vgrid_target_warnings(grid_volume_allocate_test)
vgrid_embed_shaders(grid_volume_allocate_test SHADERS
                    shaders/grid_truncation.comp)
add_test(NAME grid_volume_allocate COMMAND grid_volume_allocate_test)

# volume triangle-mesh allocation: allocate from a small mesh, then verify on
# the real driver that every block holding a voxel within trunc_dist is present
# -- brute-forced on the host against an independent closest-point -- that a
# triangle wider than the band leaves no hole (where the point path does), that
# the distance prune keeps the bounding box's far corners out, and that a split
# dispatch (base/dispatch.hpp's test seam) allocates the same blocks. Skips
# where no device is present.
add_executable(grid_volume_allocate_triangles_test
               volume_allocate_triangles_test.cpp)
target_link_libraries(
  grid_volume_allocate_triangles_test
  PRIVATE volumetric_kit::grid_volume Vulkan::Vulkan
          volumetric_kit::core_test_policy)
vgrid_target_warnings(grid_volume_allocate_triangles_test)
add_test(NAME grid_volume_allocate_triangles
         COMMAND grid_volume_allocate_triangles_test)

# volume frustum-culled compaction: allocate blocks at known world positions
# (one dead-centre, others behind / beyond-far / off to the side), then verify
# compact_active_blocks_in_frusta_on_device keeps only the in-view blocks while
# plain compaction returns all -- proving the plane math + the on-device AABB
# cull -- and that several frusta keep their union. Skips where no device is
# present.
add_executable(grid_volume_frustum_test volume_frustum_test.cpp)
target_link_libraries(
  grid_volume_frustum_test PRIVATE volumetric_kit::grid_volume Vulkan::Vulkan
                                   volumetric_kit::core_test_policy)
vgrid_target_warnings(grid_volume_frustum_test)
add_test(NAME grid_volume_frustum COMMAND grid_volume_frustum_test)

# Sparse marching cubes: mesh an analytic sphere written into a real multi-block
# VoxelBlockGrid, exercising cross-block neighbour sampling, and prove the same
# field meshes identically however it is divided into blocks or dispatches
# (base/dispatch.hpp's test seam). Skips where no device is present.
add_executable(grid_mesh_marching_cubes_sparse_test
               marching_cubes_sparse_test.cpp)
target_link_libraries(
  grid_mesh_marching_cubes_sparse_test
  PRIVATE volumetric_kit::grid_mesh Vulkan::Vulkan
          volumetric_kit::core_test_policy)
vgrid_target_warnings(grid_mesh_marching_cubes_sparse_test)
add_test(NAME grid_mesh_marching_cubes_sparse
         COMMAND grid_mesh_marching_cubes_sparse_test)

# Marching cubes at corners whose distance is exactly the iso-value: lattice
# planes at 1 cm (k = -248, -1, 0; both orientations; per-column slopes, and
# slopes below the near-tangent guard) put every vertex on its corner, bit for
# bit, with that corner's attribute, from both kernels, at iso 0 and 0.0025.
# Planes 1e-8 off the iso-value keep their vertices off the lattice. Skips where
# no device is present.
add_executable(grid_mesh_marching_cubes_corner_test
               marching_cubes_corner_test.cpp)
target_link_libraries(
  grid_mesh_marching_cubes_corner_test
  PRIVATE volumetric_kit::grid_mesh Vulkan::Vulkan
          volumetric_kit::core_test_policy)
vgrid_target_warnings(grid_mesh_marching_cubes_corner_test)
add_test(NAME grid_mesh_marching_cubes_corner
         COMMAND grid_mesh_marching_cubes_corner_test)

# mesh recorded extract: record_extract, finish and abandon against
# extract_device, sharing on and off; the refit and its later ready value, the
# closed gate, the empty set, abandoned and dropped tokens, the skipped signal,
# the superseded pending, ready across a move, the steady-state counters and the
# sharing refusal. Skips with no device.
add_executable(grid_mesh_recorded_extract_test mesh_recorded_extract_test.cpp)
target_link_libraries(
  grid_mesh_recorded_extract_test
  PRIVATE volumetric_kit::grid_mesh Vulkan::Vulkan
          volumetric_kit::core_test_policy)
vgrid_target_warnings(grid_mesh_recorded_extract_test)
add_test(NAME grid_mesh_recorded_extract
         COMMAND grid_mesh_recorded_extract_test)

# mesh MarchingCubesConfig: prove a consumer's extra buffer usage reaches the
# vertex arena and index run -- on the first allocation AND across a grow -- and
# that an unsupported bit is rejected at create(). grid itself never sets one,
# so this is the only place the non-zero path runs. Skips with no device.
add_executable(grid_mesh_marching_cubes_config_test
               marching_cubes_config_test.cpp)
target_link_libraries(
  grid_mesh_marching_cubes_config_test
  PRIVATE volumetric_kit::grid_mesh Vulkan::Vulkan
          volumetric_kit::core_test_policy)
vgrid_target_warnings(grid_mesh_marching_cubes_config_test)
add_test(NAME grid_mesh_marching_cubes_config
         COMMAND grid_mesh_marching_cubes_config_test)

# mesh MeshExchange: the fuse-to-render handoff against a host model of the ring
# -- the mark, the take, a parked take never released, an empty mesh drawing
# nothing, the latch, a seeded run with failures in which no slot the consumer
# holds is reclaimed, and a dropped mesh, with the ring a consumer that drops
# needs. Host only, so it always runs.
add_executable(grid_mesh_exchange_test mesh_exchange_test.cpp)
target_link_libraries(grid_mesh_exchange_test PRIVATE volumetric_kit::grid_mesh)
vgrid_target_warnings(grid_mesh_exchange_test)
add_test(NAME grid_mesh_exchange COMMAND grid_mesh_exchange_test)

# codec tier, host half: CodecParams validation, EncoderConfig's sizes and the
# DCT's two tables (basis orthonormality, the zigzag order against the prior
# engine's). CPU-only, so it always runs. The tables are internal
# (non-installed), reached by include path.
add_executable(grid_codec_params_test codec_params_test.cpp)
target_link_libraries(grid_codec_params_test PRIVATE volumetric_kit::grid_codec)
target_include_directories(
  grid_codec_params_test
  PRIVATE ${PROJECT_SOURCE_DIR}/src/volumetric_kit/grid/codec)
vgrid_target_warnings(grid_codec_params_test)
add_test(NAME grid_codec_params COMMAND grid_codec_params_test)

# codec tier, the CPU oracle (codec_oracle*.hpp): the CPU rANS coder and frame
# coder that pin the format's bytes, which the library codes on the device
# alone, and their ends on the device -- a host frame's entries uploaded, the
# forward read back, the inverse fed from the host, and the frame an Encoder
# must write -- and the decoder-side deblock's filters in double
# (codec_deblock_oracle.hpp). Test-only; the codec tests that hold the device to
# it link it, and reach the internal headers through it.
add_library(grid_codec_oracle STATIC codec_oracle.cpp codec_oracle_device.cpp
                                     codec_deblock_oracle.cpp)
target_link_libraries(grid_codec_oracle PUBLIC volumetric_kit::grid_codec)
target_include_directories(
  grid_codec_oracle PUBLIC ${PROJECT_SOURCE_DIR}/src/volumetric_kit/grid/codec
                           ${PROJECT_SOURCE_DIR}/src)
vgrid_target_warnings(grid_codec_oracle)

# codec tier, GPU half: the internal DctTransform's forward kernel against a
# double-precision reference, the orthonormal reconstruction bounds, the
# observed mask, batching past one dispatch, the refusals and the moves. Skips
# with no device.
add_executable(grid_codec_dct_test codec_dct_test.cpp)
target_link_libraries(
  grid_codec_dct_test PRIVATE grid_codec_oracle Vulkan::Vulkan
                              volumetric_kit::core_test_policy)
vgrid_target_warnings(grid_codec_dct_test)
add_test(NAME grid_codec_dct COMMAND grid_codec_dct_test)

# codec tier, the decoder-side deblock: detail::Deblock's chains and every rule
# of its filters against the CPU oracle, workgroups one at a time in either
# order and in chunks to the same bits, its stamps, recording without a submit,
# its refusals and the step-keyed rule; and the Decoder's switch -- off exact,
# on the oracle's filter of each frame under the rule, I- and P-frames, toggled
# between frames, the filter changing in one decoder, rebuilt while paused, its
# stamps and rows. Internal headers through the oracle. Skips without a device.
add_executable(grid_codec_deblock_test codec_deblock_test.cpp)
target_link_libraries(
  grid_codec_deblock_test PRIVATE grid_codec_oracle Vulkan::Vulkan
                                  volumetric_kit::core_test_policy)
vgrid_target_warnings(grid_codec_deblock_test)
add_test(NAME grid_codec_deblock COMMAND grid_codec_deblock_test)

# codec tier, the device frame writer and reader: every frame the writer writes
# is the CPU oracle's, byte for byte, across block counts, K, segment sizes and
# int32 coordinate extremes, the golden frames' pinned bytes among them, intra
# and P, and every frame the reader reads, or corrupt payload it refuses, is the
# oracle's. Skips without a device.
add_executable(grid_codec_device_frame_test codec_device_frame_test.cpp)
target_link_libraries(
  grid_codec_device_frame_test PRIVATE grid_codec_oracle Vulkan::Vulkan
                                       volumetric_kit::core_test_policy)
vgrid_target_warnings(grid_codec_device_frame_test)
add_test(NAME grid_codec_device_frame COMMAND grid_codec_device_frame_test)

# codec tier, P-frames on the device: the reference store's apply against plain
# uint16 arithmetic, and the Decoder's frames -- the Encoder's, and I-then-P
# chains -- leaving the grid as the CPU oracle's decode of the same content
# does, bit for bit, with NotFound, a segment longer than the format allows, the
# refusals and failures that keep the reference, and moves; and the Encoder's
# P-frames, the CPU oracle's bytes, with the diff's every choice and refusal,
# and its one-step hold of every frame against oracle::held. Internal headers
# through the oracle (the public decoder test stays public-only). Skips without
# a device.
add_executable(grid_codec_inter_test codec_inter_test.cpp)
target_link_libraries(
  grid_codec_inter_test PRIVATE grid_codec_oracle Vulkan::Vulkan
                                volumetric_kit::core_test_policy)
vgrid_target_warnings(grid_codec_inter_test)
add_test(NAME grid_codec_inter COMMAND grid_codec_inter_test)

# codec tier, the block lists on the device: the encoder's sort against
# std::stable_sort, over all of int32 and every kind of tie; the decoder's merge
# against the oracle's map and row 13's first collision against a walk; the grid
# diff against std::set_difference; garbage writing nothing past an output; and
# the refusals. Internal headers through the oracle. Skips without a device.
add_executable(grid_codec_block_lists_test codec_block_lists_test.cpp)
target_link_libraries(
  grid_codec_block_lists_test PRIVATE grid_codec_oracle Vulkan::Vulkan
                                      volumetric_kit::core_test_policy)
vgrid_target_warnings(grid_codec_block_lists_test)
add_test(NAME grid_codec_block_lists COMMAND grid_codec_block_lists_test)

# codec tier, the device frame writer's step layout: every entry's start and
# segment's slot against the CPU oracle's, for real frames' steps (every K and
# segment size, int32 extremes, the worked example, the P-frame matrix) and any
# steps up to the bound, the step total's cap, and the refusals. Internal
# headers through the oracle. Skips without a device.
add_executable(grid_codec_step_layout_test codec_step_layout_test.cpp)
target_link_libraries(
  grid_codec_step_layout_test PRIVATE grid_codec_oracle Vulkan::Vulkan
                                      volumetric_kit::core_test_policy)
vgrid_target_warnings(grid_codec_step_layout_test)
add_test(NAME grid_codec_step_layout COMMAND grid_codec_step_layout_test)

# codec tier, entropy coding: the table normalization and the CPU oracle's rANS
# coder (round trips, a probability-one symbol's zero cost, size against the
# ideal code length, truncation and bit flips), the v3 frame through the oracle
# (round trips, every writer and reader refusal, the frame type, the section
# rules, corruption, and an all-zero frame's cost), and its P-frames (the
# reference hash, the worked example, round trips and chains against an
# independent model, the block merge, every refusal in order, corruption). All
# host-only, so they always run.
foreach(_vgrid_codec_host_test rans bitstream inter_bitstream)
  add_executable(grid_codec_${_vgrid_codec_host_test}_test
                 codec_${_vgrid_codec_host_test}_test.cpp)
  target_link_libraries(grid_codec_${_vgrid_codec_host_test}_test
                        PRIVATE grid_codec_oracle)
  vgrid_target_warnings(grid_codec_${_vgrid_codec_host_test}_test)
  add_test(NAME grid_codec_${_vgrid_codec_host_test}
           COMMAND grid_codec_${_vgrid_codec_host_test}_test)
endforeach()

# codec tier, public API: the Encoder (an empty grid, the CPU oracle's bytes,
# independent of the hash table's order, unobserved blocks left out, its rows,
# refusals, moves, and the rules for an I-frame or a P-frame) and the Decoder
# with the codec end to end -- a decoded grid within the orthonormal bound of
# its source, meshing to the analytic sphere it came from, frame after frame
# into one grid, untouched on every early refusal, OutOfMemory that a resize
# recovers, and P-frame chains decoding to the all-intra chain's grids. Public
# headers only, plus the shared codec_fixture.hpp and the oracle's public-typed
# codec_oracle_encode.hpp; the decoder test links grid_mesh, which the tier may
# not. Both skip with no device.
add_executable(grid_codec_encoder_test codec_encoder_test.cpp)
target_link_libraries(
  grid_codec_encoder_test PRIVATE grid_codec_oracle Vulkan::Vulkan
                                  volumetric_kit::core_test_policy)
vgrid_target_warnings(grid_codec_encoder_test)
add_test(NAME grid_codec_encoder COMMAND grid_codec_encoder_test)

# Allocation refusal preserves the predictor, GOP and retryable front results.
add_executable(grid_codec_encoder_oom_test codec_encoder_oom_test.cpp)
target_link_libraries(
  grid_codec_encoder_oom_test
  PRIVATE volumetric_kit::grid_codec Vulkan::Vulkan
          volumetric_kit::core_test_policy
          volumetric_kit::core_test_allocation_failure)
vgrid_target_warnings(grid_codec_encoder_oom_test)
add_test(NAME grid_codec_encoder_oom COMMAND grid_codec_encoder_oom_test)

add_executable(grid_codec_decoder_test codec_decoder_test.cpp)
target_link_libraries(
  grid_codec_decoder_test
  PRIVATE volumetric_kit::grid_codec volumetric_kit::grid_mesh Vulkan::Vulkan
          volumetric_kit::core_test_policy)
vgrid_target_warnings(grid_codec_decoder_test)
add_test(NAME grid_codec_decoder COMMAND grid_codec_decoder_test)

# codec tier, a player's grid (decode_grid.hpp): make_decode_grid's geometry,
# layout and attributes off a frame's header, and decode_growing's one grow, its
# second OutOfMemory, its timing of the attempt that succeeds, and the decoder's
# refusals passed through; the caller's ceiling, against frames forged with the
# internal frame writer. Skips with no device.
add_executable(grid_codec_decode_grid_test codec_decode_grid_test.cpp)
target_link_libraries(
  grid_codec_decode_grid_test PRIVATE volumetric_kit::grid_codec Vulkan::Vulkan
                                      volumetric_kit::core_test_policy)
target_include_directories(
  grid_codec_decode_grid_test
  PRIVATE ${PROJECT_SOURCE_DIR}/src/volumetric_kit/grid/codec)
vgrid_target_warnings(grid_codec_decode_grid_test)
add_test(NAME grid_codec_decode_grid COMMAND grid_codec_decode_grid_test)

# codec tier, a frame decoded and meshed in the caller's batch: the recorded
# decode against decode(), deblock on and off; the host refusals, which record
# nothing; a device refusal gating the extract after it; superseded and
# abandoned tokens and the input slots; the rows; the steady-state counters; the
# pinned dispatches and barriers of a rafa2-like P-frame's batch; and
# record_growing and finish_growing. Links grid_mesh, which the codec tier
# itself may not. Skips with no device.
add_executable(grid_codec_recorded_test codec_recorded_test.cpp)
target_link_libraries(
  grid_codec_recorded_test
  PRIVATE volumetric_kit::grid_codec volumetric_kit::grid_mesh Vulkan::Vulkan
          volumetric_kit::core_test_policy)
vgrid_target_warnings(grid_codec_recorded_test)
add_test(NAME grid_codec_recorded COMMAND grid_codec_recorded_test)

# Allocation injection stays in its own executable: factory diagnostics and a
# metrics merge after a committed P-frame, followed by the next P-frame.
add_executable(grid_codec_decode_grid_oom_test codec_decode_grid_oom_test.cpp)
target_link_libraries(
  grid_codec_decode_grid_oom_test
  PRIVATE volumetric_kit::grid_codec Vulkan::Vulkan
          volumetric_kit::core_test_policy
          volumetric_kit::core_test_allocation_failure)
vgrid_target_warnings(grid_codec_decode_grid_oom_test)
add_test(NAME grid_codec_decode_grid_oom
         COMMAND grid_codec_decode_grid_oom_test)

# A pending-frame allocation failure stays inside record and both blocking
# decode APIs, preserves the grid/reference, and permits the same P-frame retry.
add_executable(grid_codec_decoder_oom_test codec_decoder_oom_test.cpp)
target_link_libraries(
  grid_codec_decoder_oom_test
  PRIVATE volumetric_kit::grid_codec Vulkan::Vulkan
          volumetric_kit::core_test_policy
          volumetric_kit::core_test_allocation_failure)
vgrid_target_warnings(grid_codec_decoder_oom_test)
add_test(NAME grid_codec_decoder_oom COMMAND grid_codec_decoder_oom_test)

# eval tier: closest points, the reach clamp, the summary statistics, accuracy /
# coverage and the F-score between meshes a known distance apart, and every
# refusal. Host-only.
add_executable(grid_eval_mesh_distance_test eval_mesh_distance_test.cpp)
target_link_libraries(grid_eval_mesh_distance_test
                      PRIVATE volumetric_kit::grid_eval)
vgrid_target_warnings(grid_eval_mesh_distance_test)
add_test(NAME grid_eval_mesh_distance COMMAND grid_eval_mesh_distance_test)

add_executable(grid_io_ply_test io_ply_test.cpp)
target_link_libraries(grid_io_ply_test PRIVATE volumetric_kit::grid_mesh_io)
vgrid_target_warnings(grid_io_ply_test)
add_test(NAME grid_io_ply COMMAND grid_io_ply_test)

# The opt-in mesh normalization and topology helpers, on synthetic meshes.
add_executable(grid_io_mesh_normalization_test io_mesh_normalization_test.cpp)
target_link_libraries(
  grid_io_mesh_normalization_test
  PRIVATE volumetric_kit::grid_mesh_io
          volumetric_kit::core_test_allocation_failure)
vgrid_target_warnings(grid_io_mesh_normalization_test)
add_test(NAME grid_io_mesh_normalization
         COMMAND grid_io_mesh_normalization_test)

# mesh_io::check_mesh_file, the one home of a mesh path's checks, and load_mesh
# answering through it. Host only; built with the Assimp backend.
if(TARGET grid_mesh_io_assimp)
  add_executable(grid_io_mesh_file_test io_mesh_file_test.cpp)
  target_link_libraries(grid_io_mesh_file_test
                        PRIVATE volumetric_kit::grid_mesh_io_assimp)
  vgrid_target_warnings(grid_io_mesh_file_test)
  add_test(NAME grid_io_mesh_file COMMAND grid_io_mesh_file_test)

endif()

# stream: the .vrst container through its public API -- golden bytes whose CRCs
# Python's zlib.crc32 computed, round trips, seeking, unknown kinds and chunks,
# the limits, every refusal, two threads on one reader -- its chunks in memory:
# the encoder's bytes and refusals against the writer's, the parser fed a file
# in any pieces against the reader, each refusal at its byte -- what its reader
# recovers: flipped bits, the tail cut at every byte, damaged spans, a killed
# writer and one stopped by RLIMIT_FSIZE -- and its open from the footer: the
# scan's counts and keys, each footer check refusing a forged footer, and
# records past 4 GiB (skipped, 77, without sparse files). Host-only.
find_package(Threads REQUIRED)
add_executable(grid_stream_container_test stream_container_test.cpp)
target_link_libraries(grid_stream_container_test
                      PRIVATE volumetric_kit::grid_stream Threads::Threads)
vgrid_target_warnings(grid_stream_container_test)
add_test(NAME grid_stream_container COMMAND grid_stream_container_test)

add_executable(grid_stream_chunks_test stream_chunks_test.cpp)
target_link_libraries(grid_stream_chunks_test
                      PRIVATE volumetric_kit::grid_stream)
vgrid_target_warnings(grid_stream_chunks_test)
add_test(NAME grid_stream_chunks COMMAND grid_stream_chunks_test)

add_executable(grid_stream_recovery_test stream_recovery_test.cpp)
target_link_libraries(grid_stream_recovery_test
                      PRIVATE volumetric_kit::grid_stream)
vgrid_target_warnings(grid_stream_recovery_test)
add_test(NAME grid_stream_recovery COMMAND grid_stream_recovery_test)

add_executable(grid_stream_footer_test stream_footer_test.cpp)
target_link_libraries(
  grid_stream_footer_test PRIVATE volumetric_kit::grid_stream
                                  volumetric_kit::core_test_allocation_failure)
# A 64-bit off_t, as the library's, for pwrite and truncate past 4 GiB.
target_compile_definitions(grid_stream_footer_test PRIVATE _FILE_OFFSET_BITS=64)
vgrid_target_warnings(grid_stream_footer_test)
add_test(NAME grid_stream_footer COMMAND grid_stream_footer_test)

# Public stream calls contain allocation failures in validation, diagnostics,
# saved errors, and index/footer growth. Host-only, including retry behavior.
add_executable(grid_stream_oom_test stream_oom_test.cpp)
target_link_libraries(
  grid_stream_oom_test PRIVATE volumetric_kit::grid_stream
                               volumetric_kit::core_test_allocation_failure)
vgrid_target_warnings(grid_stream_oom_test)
add_test(NAME grid_stream_oom COMMAND grid_stream_oom_test)

# networking: the server's GOP log with no sockets (join, newest key wins at a
# chunk boundary, the GOP limit, a lost copy, no P-frame without its key; keys
# on demand: joins and fall-backs counted, the bytes held bounded) and the
# socket calls (each socket's settings read back, no SIGPIPE, the bytes a peer
# has not acknowledged, a wait a signal interrupts) -- internal and unexported,
# so the tests compile their sources themselves -- then the server and the
# client on loopback, 127.0.0.1 port 0 only: the wire against the writer's file,
# joins, newest key wins, keys on demand, the limits, the rules, teardown,
# signals, the client's errors against raw senders, and a stress run in both
# modes for ThreadSanitizer. Host-only.
set(_vgrid_networking_src
    ${PROJECT_SOURCE_DIR}/src/volumetric_kit/grid/networking)
add_executable(grid_networking_gop_log_test
               networking_gop_log_test.cpp ${_vgrid_networking_src}/gop_log.cpp)
target_include_directories(grid_networking_gop_log_test
                           PRIVATE ${_vgrid_networking_src})
vgrid_target_warnings(grid_networking_gop_log_test)
add_test(NAME grid_networking_gop_log COMMAND grid_networking_gop_log_test)

add_executable(grid_networking_socket_test networking_socket_test.cpp
                                           ${_vgrid_networking_src}/socket.cpp)
target_include_directories(grid_networking_socket_test
                           PRIVATE ${_vgrid_networking_src})
target_link_libraries(grid_networking_socket_test
                      PRIVATE volumetric_kit::core_base Threads::Threads)
vgrid_target_warnings(grid_networking_socket_test)
add_test(NAME grid_networking_socket COMMAND grid_networking_socket_test)

foreach(_vgrid_networking_test server client stress)
  add_executable(grid_networking_${_vgrid_networking_test}_test
                 networking_${_vgrid_networking_test}_test.cpp)
  target_link_libraries(
    grid_networking_${_vgrid_networking_test}_test
    PRIVATE volumetric_kit::grid_networking Threads::Threads)
  vgrid_target_warnings(grid_networking_${_vgrid_networking_test}_test)
  add_test(NAME grid_networking_${_vgrid_networking_test}
           COMMAND grid_networking_${_vgrid_networking_test}_test)
endforeach()

# The client also checks its progress output across an allocation failure while
# copying a saved error.
target_link_libraries(grid_networking_client_test
                      PRIVATE volumetric_kit::core_test_allocation_failure)

# The server, keys on demand, when a record's copy fails: the viewers it sends
# back counted before publish returns. Apart from the server's other tests,
# which link no replaced operator new, so they build under ThreadSanitizer on
# Linux too.
add_executable(grid_networking_server_oom_test networking_server_oom_test.cpp)
target_link_libraries(
  grid_networking_server_oom_test
  PRIVATE volumetric_kit::grid_networking Threads::Threads
          volumetric_kit::core_test_allocation_failure)
vgrid_target_warnings(grid_networking_server_oom_test)
add_test(NAME grid_networking_server_oom
         COMMAND grid_networking_server_oom_test)

# codec_stream tier: the codec stream (codec_stream/codec_stream.hpp) on a grid
# holding a sphere, linking no fuser and no sensor tier -- outputs it cannot
# make and a session that is not one Geometry track refused on create, none
# written without outputs, each frame as it went out, the .vrst's records (every
# encoded frame, byte for byte, a frame the decoder refuses included and
# counted, a key exactly when an I-frame, the same bytes on a rerun), an I-frame
# after a frame not decoded, a frame out of order or any call once finished or
# moved from refused, an unfinished stream read whole and an empty one removed,
# frames in halves around a caller's batch the file code() writes, the batch's
# device rows split between the frame and the caller, the front half again,
# decode off, and the frames served live to a client on 127.0.0.1 port 0 as the
# file records them. The device part skips with no device.
add_executable(grid_codec_stream_test codec_stream_test.cpp)
target_link_libraries(
  grid_codec_stream_test
  PRIVATE volumetric_kit::core_test_policy volumetric_kit::grid_codec_stream
          Vulkan::Vulkan)
vgrid_target_warnings(grid_codec_stream_test)
add_test(NAME grid_codec_stream COMMAND grid_codec_stream_test)

# serve_stream's replay (examples/common/stream_replay.hpp): the records from
# the first key, seq and key pts across passes by the writer's rules, the dues,
# damage left out and an allocation failure past it refused, and a damaged
# payload skipped to the next key. Host only.
add_executable(grid_example_stream_replay_test example_stream_replay_test.cpp)
target_link_libraries(grid_example_stream_replay_test
                      PRIVATE grid_example_common)
target_link_libraries(
  grid_example_stream_replay_test
  PRIVATE volumetric_kit::grid_stream
          volumetric_kit::core_test_allocation_failure)
vgrid_target_warnings(grid_example_stream_replay_test)
add_test(NAME grid_example_stream_replay
         COMMAND grid_example_stream_replay_test)
# The tests whose code runs threads -- a reader shared by two, the mesh
# handoff's producer and consumer, the sockets', the senders' log -- which CI
# builds and runs again under ThreadSanitizer (.github/workflows/_build.yml):
# one target to build them.
add_custom_target(grid_tsan_tests)
add_dependencies(
  grid_tsan_tests
  grid_stream_container_test
  grid_mesh_exchange_test
  grid_networking_gop_log_test
  grid_networking_socket_test
  grid_networking_server_test
  grid_networking_server_oom_test
  grid_networking_client_test
  grid_networking_stress_test
  grid_example_info_log_test)

# What a sibling that fetches grid gets (subproject/): its configure, from a
# clean tree, with each dependency this build fetched as a local source, so it
# needs no network. The defaults hold, the sibling's vkc_require_sibling accepts
# this grid and refuses it under a floor above its version, the export roots
# hold exactly the exported GLSL, the public headers hold none of the hash
# table's internals, and grid refuses a core the sibling made available first
# that is older than VGRID_CORE_MIN_VERSION or lacks the vulkan tier. The old
# core is this one stamped 0.0.1, standing in for one below a later floor: a
# real core older than 0.1.0 has no vkc_require_core at all.
#
# TODO: move this requirement harness -- the message helper and subproject/'s
# version stamp, which recon, gfx and calib repeat -- into the core's test
# support, as a function each sibling calls with its package and floor.
set(_vgrid_subproject "${CMAKE_CURRENT_BINARY_DIR}/subproject")
add_test(NAME grid_subproject_clean COMMAND "${CMAKE_COMMAND}" -E rm -rf
                                            "${_vgrid_subproject}")
set_tests_properties(grid_subproject_clean PROPERTIES FIXTURES_SETUP
                                                      vgrid_subproject)
# Configures subproject/ into <dir> under _vgrid_subproject with the given -D
# args.
function(_vgrid_add_subproject_test name dir)
  add_test(
    NAME ${name}
    COMMAND
      "${CMAKE_COMMAND}" -S "${CMAKE_CURRENT_SOURCE_DIR}/subproject" -B
      "${_vgrid_subproject}/${dir}" -G "${CMAKE_GENERATOR}"
      "-DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}"
      "-DVGRID_SOURCE_DIR=${PROJECT_SOURCE_DIR}" ${_vgrid_fetched_sources}
      ${ARGN})
  set_tests_properties(${name} PROPERTIES FIXTURES_REQUIRED vgrid_subproject)
endfunction()
# Sets <test>'s PASS_REGULAR_EXPRESSION to <message>, a space in which matches
# however CMake wrapped the error's lines.
function(_vgrid_expect_message test message)
  string(REPLACE " " "[ \n]+" message "${message}")
  set_tests_properties(${test} PROPERTIES PASS_REGULAR_EXPRESSION "${message}")
endfunction()

_vgrid_add_subproject_test(grid_subproject_defaults defaults
                           "-DVGRID_REQUIRE_GRID=${PROJECT_VERSION}")
set_tests_properties(grid_subproject_defaults
                     PROPERTIES FIXTURES_SETUP vgrid_subproject_defaults)
math(EXPR _vgrid_next_minor "${PROJECT_VERSION_MINOR} + 1")
set(_vgrid_newer "${PROJECT_VERSION_MAJOR}.${_vgrid_next_minor}.0")
_vgrid_add_subproject_test(grid_subproject_old_grid old_grid
                           "-DVGRID_REQUIRE_GRID=${_vgrid_newer}")
_vgrid_expect_message(
  grid_subproject_old_grid
  "vgrid_subproject_consumer needs volumetric_kit_grid ${_vgrid_newer} or newer, but the grid at .+ is ${PROJECT_VERSION}: pin v${_vgrid_newer} or newer"
)
# The sibling compiles kernels that include grid's exported GLSL through the
# roots grid_base and grid_volume name, each holding exactly what its target
# exports (glsl_consumer.cmake).
add_test(NAME grid_subproject_glsl
         COMMAND "${CMAKE_COMMAND}" --build "${_vgrid_subproject}/defaults"
                 --target vgrid_glsl_consumer)
set_tests_properties(grid_subproject_glsl PROPERTIES FIXTURES_REQUIRED
                                                     vgrid_subproject_defaults)
# ... and compiles the public headers that reach the volume tier, which hold
# none of the hash table's internals (subproject/internal_names.cpp).
add_test(NAME grid_subproject_headers
         COMMAND "${CMAKE_COMMAND}" --build "${_vgrid_subproject}/defaults"
                 --target vgrid_public_headers)
set_tests_properties(grid_subproject_headers
                     PROPERTIES FIXTURES_REQUIRED vgrid_subproject_defaults)
_vgrid_add_subproject_test(
  grid_subproject_old_core old_core -DVGRID_CORE_FIRST=ON -DVKC_WITH_VULKAN=ON
  -DVGRID_CONSUMER_CORE_VERSION=0.0.1)
_vgrid_expect_message(
  grid_subproject_old_core
  "volumetric_kit_grid needs volumetric_kit_core ${VGRID_CORE_MIN_VERSION} or newer, but the core at .+ is 0.0.1: pin v${VGRID_CORE_MIN_VERSION} or newer"
)
_vgrid_add_subproject_test(grid_subproject_core_without_vulkan no_vulkan
                           -DVGRID_CORE_FIRST=ON -DVKC_WITH_VULKAN=OFF)
_vgrid_expect_message(
  grid_subproject_core_without_vulkan
  "volumetric_kit_grid needs volumetric_kit_core's vulkan tier, which the core at .+ was built without: set VKC_WITH_VULKAN ON before"
)

# Start the slowest tests first under `ctest --parallel`. Without a COST, CTest
# runs tests in the order they were added, and a fresh CI runner has no timing
# history, so a slow test added late finished last. Seconds under recon's
# sanitizers job, the longest in CI, in run 39717939263 (before the 2026-10-22
# test trims); only the order matters.
set_tests_properties(grid_codec_inter PROPERTIES COST 115)
set_tests_properties(grid_codec_decoder PROPERTIES COST 96)
set_tests_properties(grid_voxelize_mesh_integrate_exact PROPERTIES COST 56)
set_tests_properties(grid_eval_mesh_distance PROPERTIES COST 55)
set_tests_properties(grid_codec_device_frame PROPERTIES COST 52)
set_tests_properties(grid_codec_stream PROPERTIES COST 49)
set_tests_properties(grid_voxelize_mesh_integrate PROPERTIES COST 36)
set_tests_properties(grid_codec_encoder PROPERTIES COST 30)

# A test skips by exiting with the core's test::kSkipExitCode (gpu_test.hpp). A
# hang (a fence never signalled, a deadlocked thread) fails in 300 s rather than
# holding a shared runner to its job's limit; 300 is well above the largest
# COST.
get_property(
  _vgrid_tests
  DIRECTORY
  PROPERTY TESTS)
set_tests_properties(${_vgrid_tests} PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 300)
