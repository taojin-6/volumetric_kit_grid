// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// The free-block heap, a stack of block indices, shared by the allocate kernels
// (which only pop) and the delete kernel (which only push). Alloc and free run
// in SEPARATE dispatches, so within one the counter moves one way, one
// atomicAdd claims a slot, and a claim past either end is undone.
//
// #include this AFTER hash_common.glsl, which supplies the push-constant block.

layout(set = 0, binding = 1) coherent buffer Heap { uint heap[]; };
layout(set = 0, binding = 2) coherent buffer HeapCounter { uint heap_counter; };

const uint kHeapEmpty = 0xFFFFFFFFu;

// Whether a counter reading holds no block. A pop from an empty heap wraps the
// counter past num_blocks until it is undone, and that reads as empty too.
bool heap_empty(uint count) { return count - 1u >= uint(pc.grid.num_blocks); }

// Pop a free block index, or kHeapEmpty when the heap is empty.
uint consume_heap() {
  uint old = atomicAdd(heap_counter, 0xFFFFFFFFu);
  if (heap_empty(old)) {
    atomicAdd(heap_counter, 1u);
    return kHeapEmpty;
  }
  return heap[old - 1u];
}

// Push a freed block index, or return false when the heap is already full,
// which only a block freed twice can cause.
bool append_heap(uint block_idx) {
  uint old = atomicAdd(heap_counter, 1u);
  if (old >= uint(pc.grid.num_blocks)) {
    atomicAdd(heap_counter, 0xFFFFFFFFu);
    return false;
  }
  heap[old] = block_idx;
  return true;
}
