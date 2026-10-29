// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// The voxel grid's device contract, which grid_volume exports for any
// kernel's #include (vkc_export_glsl, volume/CMakeLists.txt): the grid's
// shape, a listed block, a block slot's stamps, the in-block voxel index and
// the tick comparison. The structs mirror volume::VoxelGridParams
// (voxel_grid.hpp), volume::BlockIndex and volume::BlockStamp
// (hash_types.hpp), byte-identical under scalar block layout. The hash table's
// slot layout is not here: it stays internal (hash_common.glsl).

#ifndef VGRID_GRID_TYPES_GLSL
#define VGRID_GRID_TYPES_GLSL

struct VoxelGridParams {
  float voxel_size;
  int block_size;
  int voxels_per_block;
  float trunc_dist;
  int bucket_size;
  int num_buckets;
  int num_blocks;
  int max_chain;
};

struct BlockIndex {
  ivec3 coord;
  int ptr;
};

// One block slot's stamps, ticks of the map's clock.
struct BlockStamp {
  uint requested;
  uint weighted;
  uint changed;
};

// Whether tick a is later than tick b, modulo 2^32: volume::tick_after.
bool vgridTickAfter(uint a, uint b) { return int(a - b) > 0; }

// A voxel's index within its block, x fastest: its attribute slot is
// BlockIndex::ptr plus this.
int vgridLocalIndex(ivec3 lc, int bs) {
  return lc.x + bs * (lc.y + bs * lc.z);
}

// The voxel at in-block index `local`, in its block's frame: the inverse of
// vgridLocalIndex.
ivec3 vgridLocalBase(uint local, int bs) {
  return ivec3(int(local % uint(bs)), int((local / uint(bs)) % uint(bs)),
               int(local / uint(bs * bs)));
}

#endif  // VGRID_GRID_TYPES_GLSL
