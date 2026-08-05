// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// Shared block-allocation machinery for the voxel-hash allocate kernels
// (hash_allocate_coords / hash_allocate_depth / hash_allocate_points): the
// hash-table buffers they all touch, the lock-free/bucket-locked insert
// (allocate_block + helpers, ported from allocateBlock in hash_ops.metal), the
// per-reason failure tally, and the truncation-band dilation that depth/point
// allocation share. The per-kernel INPUT buffer (coords / depth / points) is
// declared at binding 4 by each kernel; everything common lives here.
//
// #include this AFTER hash_common.glsl: it supplies the struct layouts, the
// hash-table constants (kFreeEntry/...), the spin/heap retry caps, the coord
// transforms (worldToBlock / truncationBlocks), and the push-constant block.
//
// Concurrency: mutual exclusion is a per-bucket spin lock (atomicCompSwap on
// bucket_mutex); memoryBarrierBuffer() supplies the acquire/release ordering
// across vendors that hash_ops.metal got for free from Apple's coherent unified
// memory. Alloc and free run in SEPARATE dispatches (never concurrently), which
// the heap design relies on.

layout(set = 0, binding = 0, scalar) coherent buffer Entries {
  HashEntry entries[];
};
layout(set = 0, binding = 1) coherent buffer Heap { uint heap[]; };
layout(set = 0, binding = 2) coherent buffer HeapCounter { uint heap_counter; };
layout(set = 0, binding = 3) coherent buffer BucketMutex { int bucket_mutex[]; };
layout(set = 0, binding = 5) buffer FailCount { uint fail_count[]; };

// The fail-reason slots (kFailTotal / kFailLock / kFailChain / kFailHeap) come
// from hash_common.glsl -- the delete kernel reports through the same buffer,
// so the slot layout is shared rather than owned by this header. Every failure
// an allocate kernel reports is retryable, so none of them touch kFailTerminal.

const uint kHeapEmpty = 0xFFFFFFFFu;

// Sentinel for insert_block's preset pointer: draw a fresh block off the heap
// (normal allocation). A real block pointer is block_idx * voxels_per_block, so
// it is always >= 0 and never collides with this. A non-sentinel preset reuses
// that exact pointer instead of consuming the heap -- the rehash path preserves
// each block's index so its per-voxel attribute data (keyed by the pointer)
// survives a resize.
const int kNoPresetPtr = -1;

// Pop a free block index off the heap, or kHeapEmpty when exhausted.
uint consume_heap() {
  uint old = atomicAdd(heap_counter, 0u);  // atomic load
  for (int a = 0; a < kMaxHeapRetries; ++a) {
    if (old == 0u) {
      return kHeapEmpty;
    }
    uint prev = atomicCompSwap(heap_counter, old, old - 1u);
    if (prev == old) {
      return heap[old - 1u];
    }
    old = prev;
  }
  return kHeapEmpty;
}

bool try_lock_bucket(uint bucket, int max_retries) {
  for (int a = 0; a < max_retries; ++a) {
    if (atomicCompSwap(bucket_mutex[bucket], kFreeEntry, kLockEntry) ==
        kFreeEntry) {
      memoryBarrierBuffer();  // acquire: see the previous holder's writes
      return true;
    }
  }
  return false;
}

void unlock_bucket(uint bucket) {
  memoryBarrierBuffer();  // release: publish our writes before the unlock
  atomicExchange(bucket_mutex[bucket], kFreeEntry);
}

// Lock-free fast path. A stale read that MISSES only costs a redundant lock +
// recheck below (the under-lock path is authoritative). A stale read that
// false-MATCHES, though, returns "present" here WITHOUT that recheck and would
// silently drop the coord -- so before comparing pos we issue an acquire
// (memoryBarrierBuffer) that pairs with the insert's release barrier, so a
// just-published ptr is never observed alongside a stale pos (which, being the
// init sentinel, would false-match coord (0,0,0)).
bool block_exists(ivec3 coord) {
  uint bucket_size = uint(pc.grid.bucket_size);
  uint total_entries = uint(pc.grid.num_buckets) * bucket_size;
  uint bucket = computeHashPos(coord, pc.grid.num_buckets);
  uint bucket_start = bucket * bucket_size;

  for (uint i = 0u; i < bucket_size; ++i) {
    uint slot = bucket_start + i;
    if (entries[slot].ptr != kFreeEntry) {
      memoryBarrierBuffer();  // acquire: pos is current w.r.t. the ptr we saw
      if (entries[slot].pos == coord) {
        return true;
      }
    }
  }
  uint idx_last = (bucket + 1u) * bucket_size - 1u;
  uint idx = idx_last;
  for (int it = 0; it < pc.grid.max_chain; ++it) {
    if (entries[idx].ptr != kFreeEntry) {
      memoryBarrierBuffer();  // acquire (see above)
      if (entries[idx].pos == coord) {
        return true;
      }
    }
    int off = entries[idx].offset;
    if (off == kNoOffset) {
      break;
    }
    idx = (idx_last + uint(off)) % total_entries;
  }
  return false;
}

// Returns -1 on success, else the fail reason. Reporting the *reason* rather
// than a bool is what keeps the report honest: the only way this fails is an
// empty heap, and the only way allocate_in_overflow fails is an empty heap OR a
// table with no free entry -- two different answers to "should the caller grow
// the map", which the caller now acts on (AllocFailures::capacity_limited).
int allocate_in_primary(uint first_empty, ivec3 coord, int preset_ptr) {
  int voxel_block_ptr = preset_ptr;
  if (preset_ptr == kNoPresetPtr) {
    uint block_idx = consume_heap();
    if (block_idx == kHeapEmpty) {
      return kFailHeap;
    }
    voxel_block_ptr = int(block_idx * uint(pc.grid.voxels_per_block));
  }

  entries[first_empty].pos = coord;
  entries[first_empty].offset = kNoOffset;
  memoryBarrierBuffer();  // pos/offset visible before ptr becomes non-free
  atomicExchange(entries[first_empty].ptr, voxel_block_ptr);
  return -1;
}

// Returns -1 on success, else kFailHeap (no block left on the heap) or
// kFailTable (scanned every entry and found no free non-anchor slot).
int allocate_in_overflow(uint hash_bucket, uint bucket_start, ivec3 coord,
                         int preset_ptr) {
  uint bucket_size = uint(pc.grid.bucket_size);
  uint total_entries = uint(pc.grid.num_buckets) * bucket_size;
  uint idx_last = (hash_bucket + 1u) * bucket_size - 1u;
  uint target_idx = bucket_start + bucket_size;

  for (uint attempts = 0u; attempts < total_entries; ++attempts) {
    if (target_idx >= total_entries) {
      target_idx = 0u;
    }
    // Skip anchor slots (the last entry of each bucket owns its chain head).
    if ((target_idx + 1u) % bucket_size == 0u) {
      ++target_idx;
      continue;
    }

    uint target_bucket = target_idx / bucket_size;
    bool have_lock = (target_bucket == hash_bucket);
    if (!have_lock) {
      have_lock = try_lock_bucket(target_bucket, 1);
    }
    if (!have_lock) {
      ++target_idx;
      continue;
    }

    if (entries[target_idx].ptr == kFreeEntry) {
      int voxel_block_ptr = preset_ptr;
      if (preset_ptr == kNoPresetPtr) {
        uint block_idx = consume_heap();
        if (block_idx == kHeapEmpty) {
          if (target_bucket != hash_bucket) {
            unlock_bucket(target_bucket);
          }
          return kFailHeap;
        }
        voxel_block_ptr = int(block_idx * uint(pc.grid.voxels_per_block));
      }
      entries[target_idx].pos = coord;
      // Head-insert into the anchor's chain: adopt the old head, then repoint.
      entries[target_idx].offset = entries[idx_last].offset;
      memoryBarrierBuffer();
      atomicExchange(entries[target_idx].ptr, voxel_block_ptr);
      atomicExchange(entries[idx_last].offset,
                     int(target_idx) - int(idx_last));
      if (target_bucket != hash_bucket) {
        unlock_bucket(target_bucket);
      }
      return -1;
    }

    if (target_bucket != hash_bucket) {
      unlock_bucket(target_bucket);
    }
    ++target_idx;
  }
  // Every entry scanned and none free: the table is full. NOT kFailHeap, which
  // it used to report -- and which is provably impossible on the rehash path,
  // where preset_ptr is set and the heap is never consulted at all.
  return kFailTable;
}

// Insert `coord` if absent, giving its block `preset_ptr` (or kNoPresetPtr to
// draw a fresh block off the heap). Returns -1 on success (or already present),
// else the fail reason (kFailLock / kFailChain / kFailHeap). Shared by normal
// allocation (fresh heap pointer) and rehash (each block's pointer preserved).
int insert_block(ivec3 coord, int preset_ptr) {
  int last_fail = kFailLock;
  for (int attempt = 0; attempt < 5; ++attempt) {
    if (block_exists(coord)) {
      return -1;
    }

    uint hash_bucket = computeHashPos(coord, pc.grid.num_buckets);
    uint bucket_size = uint(pc.grid.bucket_size);
    uint bucket_start = hash_bucket * bucket_size;
    uint total_entries = uint(pc.grid.num_buckets) * bucket_size;

    if (!try_lock_bucket(hash_bucket, kMaxSpinRetries)) {
      last_fail = kFailLock;
      continue;
    }

    // Re-check the primary bucket under lock; remember the first empty slot.
    int first_empty = -1;
    bool found = false;
    for (uint i = 0u; i < bucket_size; ++i) {
      uint slot = bucket_start + i;
      if (entries[slot].ptr != kFreeEntry) {
        if (entries[slot].pos == coord) {
          found = true;
          break;
        }
      } else if (first_empty == -1) {
        first_empty = int(slot);
      }
    }
    if (found) {
      unlock_bucket(hash_bucket);
      return -1;
    }

    // Re-check the collision chain under lock.
    uint idx_last = (hash_bucket + 1u) * bucket_size - 1u;
    uint curr = idx_last;
    int chain_hops = 0;
    for (int it = 0; it < pc.grid.max_chain; ++it) {
      if (entries[curr].ptr != kFreeEntry && entries[curr].pos == coord) {
        found = true;
        break;
      }
      int off = entries[curr].offset;
      if (off == kNoOffset) {
        break;
      }
      ++chain_hops;
      curr = (idx_last + uint(off)) % total_entries;
    }
    if (found) {
      unlock_bucket(hash_bucket);
      return -1;
    }

    bool chain_at_limit = (chain_hops >= pc.grid.max_chain - 1);
    bool success = false;
    if (first_empty != -1) {
      int reason = allocate_in_primary(uint(first_empty), coord, preset_ptr);
      success = (reason < 0);
      if (!success) {
        last_fail = reason;
      }
    } else if (!chain_at_limit) {
      int reason =
          allocate_in_overflow(hash_bucket, bucket_start, coord, preset_ptr);
      success = (reason < 0);
      if (!success) {
        last_fail = reason;
      }
    } else {
      last_fail = kFailChain;
    }

    unlock_bucket(hash_bucket);
    if (success) {
      return -1;
    }
  }
  return last_fail;
}

// Allocate `coord` if absent, drawing a fresh block off the heap -- the name the
// allocate-from-coords / -depth / -points kernels call. A thin wrapper over
// insert_block's heap path (kNoPresetPtr); rehash calls insert_block directly
// with each block's preserved pointer.
int allocate_block(ivec3 coord) {
  return insert_block(coord, kNoPresetPtr);
}

// Record a per-block allocation outcome by reason: a no-op on success
// (fail < 0), else bump the total and the reason slot.
void report_alloc_fail(int fail) {
  if (fail >= 0) {
    atomicAdd(fail_count[kFailTotal], 1u);
    atomicAdd(fail_count[fail], 1u);
  }
}

// Allocate the solid (2*tb+1)^3 block cube centred on `center`, tb =
// truncationBlocks(grid) -- the truncation band a depth/point sample dilates
// into (mirrors allocateTruncationBand in hash_ops.metal). allocate_block is
// idempotent, so overlapping bands from neighbouring samples de-duplicate.
void allocate_truncation_band(ivec3 center) {
  int tb = truncationBlocks(pc.grid);
  for (int dz = -tb; dz <= tb; ++dz) {
    for (int dy = -tb; dy <= tb; ++dy) {
      for (int dx = -tb; dx <= tb; ++dx) {
        report_alloc_fail(allocate_block(center + ivec3(dx, dy, dz)));
      }
    }
  }
}
