// Write a mesh's truncated distance field into a grid's tsdf/weight, one
// workgroup per block the band reaches, every voxel of an 8^3 block in one
// round. Each voxel takes the minimum over the triangles mesh_bin.comp filed
// under its block -- every triangle within trunc_dist of any of the block's
// voxels, since a bin holds each triangle within trunc_dist + half-diagonal of
// the block's centre -- so the minimum over the bin is the true minimum
// wherever the result is observed.
//
// Observed means within trunc_dist of the mesh, in both modes: that is the set
// the band test guarantees complete, and it is the set the allocation covers.
// An observed voxel gets weight 1 and its distance clamped to +-trunc_dist;
// every other voxel of the block gets tsdf 0, weight 0 -- a fresh block's
// zeros, which the mesher and the codec read as unobserved.
//
// With a full band (pc.masked), a block mesh_full_band.comp marked kFullBlock
// is written whole; any other block it kept writes only its apron -- the
// voxels one step (across a face, an edge or a corner) from a full neighbour
// whose direction bit its mask holds -- and its other voxels 0/0. Each voxel
// written is measured exactly as without the mask, over the same bin.
//
//   Signed -- +-distance, the sign from the closest triangle's face normal:
//             the rule the research codec shipped. No topology, so it trusts
//             the mesh to be closed and outward-wound, and a voxel off a sharp
//             edge can take the wrong side (see MeshSdfMode::Signed).
//   Shell  -- distance - pc.shell: a thin solid around every triangle.
//
// The minimum breaks ties on the triangle index. The fill pass wrote each bin
// in atomic-arrival order, and two triangles tie along every shared edge and at
// every shared vertex; taking "the first minimum" would sign those voxels by
// whichever face the atomics happened to file first.
//
// THE EARLY EXIT IS EXACT. Most of a bin cannot be a voxel's nearest triangle,
// so a voxel measures only the triangles no lower bound rules out, and still
// writes, bit for bit, what measuring its whole bin writes.
//
// The bounds, from a point x to every point of a triangle T:
//   support plane -- u.(x - c) + min over T's vertices v of u.(c - v), for a
//                    unit u and a point c: T lies behind that plane. Here c is
//                    the block's centre and u points to it from the closest
//                    point vgridClosestPointOnTriangle finds.
//   ball          -- |x - m| - R: T lies within R of m, R being its farthest
//                    vertex from m (the centroid here).
//   cylinder      -- the distance from x to the cylinder around m with axis a
//                    unit n (the face normal here), radius R, and half-height
//                    H, T's farthest vertex off the plane through m: T lies in
//                    it.
// Each holds for any c and m and any unit u and n, so it does not matter where
// rounding put them; what rounds is evaluating a bound (u and n come within a
// few ulps of unit length), and measuring.
//
// The margin. Let s be the largest |coordinate| of c and T's vertices, and
// band = trunc_dist + the block's half-diagonal, which bounds |x - c| for a
// voxel x and every distance the field records. The closest point lies within
// about 35 s*2^-24 of T (a vertex; a parameter in [0, 1] along an edge; or
// clamped barycentrics), so a measured distance falls short of the true one by
// at most that and a few band*2^-24; each bound evaluates within about
// 75 s*2^-24 + 30 band*2^-24 of its value (the cylinder's is the worst, as
// x - m reaches about 4 s). Every bound of T is taken less
// m_T = 2^-14 (s + band), about ten times both together, so a bound of T that
// is positive with its square above L puts T's measured d^2 above L,
// strictly, and its distance above sqrt(L) by more than the rounding of sqrt.
// The margin grows with s, so a triangle whose corners lie far beyond the
// block gets a margin to match the rounding they bring.
//
// The cylinder's bound is compared squared: dr^2 + dz^2 > L, not
// sqrt(dr^2 + dz^2)^2 > L. The two differ by the rounding of one sqrt and one
// multiply, a few units of float's roundoff in dr^2 + dz^2, so about 2^-23 of
// the bound itself, which is at most |x - m| + band, about 4 s + band: a few
// s*2^-24 and band*2^-24 more on the 75 s*2^-24 + 30 band*2^-24 above, still
// far inside m_T. So a triangle the squared compare rules out is still
// strictly farther than L, as the rule below requires.
//
// The rule. A voxel's limit L is its best d^2 so far, or trunc_dist^2 if that
// is smaller, and a triangle is ruled out only by a bound as above: then it
// can neither beat nor tie the best (whose d^2 is at least L), and where L is
// trunc_dist^2 it is not observed. The best only falls, so a bound taken
// against an earlier limit, or the largest of several voxels' limits, rules
// out no more. Each triangle measured is measured by the same routine on the
// same floats, and the minimum over (d^2, index) does not depend on the order
// -- so the voxel's minimum, and what it writes, are its whole bin's.
//
// The records. A triangle's constants come from its record
// (mesh_tri_records.comp), written once a call: (a, s_T), (b, R), (c, H),
// (n, 0) -- its vertices, the largest |coordinate| of any of them, R and H
// about m = (a + b + c) / 3, and n. The vertices are copies, and the
// constants feed only the sort's keys and the bounds, which hold for any m
// and any unit n; m_T is still taken per block, from s_T and the centre.
//
// The order. The workgroup sorts its bin into VGRID_BUCKETS buckets of the
// support-plane bound at c, less m_T (a counting sort in shared memory). A
// bucket's floor, bucket * step, is at most every entry's bound from its first
// position on, so from a voxel x each of them has a support-plane bound of at
// least that floor less |x - c|; less the block's own margin for rounding the
// floor and |x - c|, that is the stop: the voxel stops scanning at the first
// chunk it rules out. Each chunk of VGRID_CHUNK entries is staged once in
// shared memory with its bounds. Each culling group -- 32 invocations -- keeps
// the chunk's triangles that the support plane and the ball do not rule out
// anywhere in its voxels' box at the largest limit among them; each voxel then
// tests the support plane, which alone rules out most, and the cylinder at its
// own limit, and measures what is left. A bin past VGRID_MAX_SORTED sorts its
// first VGRID_MAX_SORTED entries; the rest follow in bin order, scanned by
// every voxel to the end.
//
// The layouts, one per build of this body (MeshIntegrator::create picks one by
// the device's facts):
//   VGRID_THREADS 512, VGRID_LANE_VOXELS 1 -- one invocation a voxel; a
//       culling group is a 4x4x2 brick of an 8^3 block.
//   VGRID_THREADS 256, VGRID_LANE_VOXELS 2 -- two voxels an invocation, a z
//       pair; a culling group is a 4x4x4 brick, and every bound it loads from
//       shared memory serves both voxels.
// Any block size other than 8 goes in row order (vgridLocalBase),
// VGRID_THREADS * VGRID_LANE_VOXELS voxels a round.
//
// VGRID_SUBGROUP_CULL 1: a culling group is one subgroup, which the host builds
// only where subgroups are 32 wide. The group's box, its limit and its kept
// entries are subgroup min / max / ballot instead of shared atomics and lists,
// so a chunk takes 2 barriers -- after its staging and after its voxels -- not
// 5, and the voxels walk the kept entries in index order. Which chunks are
// staged, which entries a group keeps and which checks a voxel makes are the
// shared build's; only the order of a group's kept entries differs, and the
// minimum does not depend on it.

#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_control_flow_attributes : require
#define VGRID_HASH_COMMON_NO_PUSH_CONSTANTS
#include "volumetric_kit/grid/volume/shaders/hash_common.glsl"
#include "volumetric_kit/grid/volume/shaders/triangle_common.glsl"

#ifndef VGRID_THREADS
#error "mesh_integrate_body.glsl: define VGRID_THREADS"
#endif
#ifndef VGRID_LANE_VOXELS
#error "mesh_integrate_body.glsl: define VGRID_LANE_VOXELS"
#endif
#ifndef VGRID_SUBGROUP_CULL
#define VGRID_SUBGROUP_CULL 0
#endif
#define VGRID_GROUP 32         // a culling group: 32 invocations
#define VGRID_GROUPS (VGRID_THREADS / VGRID_GROUP)
#define VGRID_CHUNK 64         // bin entries staged at a time
#define VGRID_BUCKETS 256      // of the sort key: 8 bits, 4 to a word
#define VGRID_MAX_SORTED 2048  // bin entries sorted: 16-bit indices, 2 to a word
// The shared arrays below, in 4-byte words: the sort's keys, order, starts and
// cursors; a chunk's triangles, 9 coordinates and 3 vec4s; and, without the
// subgroup cull, each group's box, limit, count and list and s_any, or with it
// the two s_any flags. MeshIntegrator::create checks the wide shared build's
// bytes against maxComputeSharedMemorySize (kWideSharedBytes); the others fit
// the 16,384 every Vulkan device offers.
#if VGRID_SUBGROUP_CULL
#extension GL_KHR_shader_subgroup_basic : require
#extension GL_KHR_shader_subgroup_vote : require
#extension GL_KHR_shader_subgroup_ballot : require
#extension GL_KHR_shader_subgroup_arithmetic : require
// 3,458 words, 13,832 bytes, at either layout.
#define VGRID_SHARED_WORDS                                      \
  (VGRID_MAX_SORTED / 4 + VGRID_MAX_SORTED / 2 + 2 * VGRID_BUCKETS +  \
   VGRID_CHUNK * (1 + 9 + 3 * 4) + 2)
#if VGRID_CHUNK != 2 * VGRID_GROUP || VGRID_SHARED_WORDS * 4 != 13832
#error "mesh_integrate_body.glsl: shared memory past its budget"
#endif
#else
// 4,033 words at 8 groups (16,132 bytes); 4,609 at 16 (18,436 bytes).
#define VGRID_SHARED_WORDS                                      \
  (VGRID_MAX_SORTED / 4 + VGRID_MAX_SORTED / 2 + 2 * VGRID_BUCKETS +  \
   VGRID_CHUNK * (1 + 9 + 3 * 4) + VGRID_GROUPS * (6 + 2 + VGRID_CHUNK) + 1)
#if VGRID_GROUPS * VGRID_GROUP != VGRID_THREADS || \
    (VGRID_THREADS == 256 && VGRID_SHARED_WORDS * 4 != 16132) || \
    (VGRID_THREADS == 512 && VGRID_SHARED_WORDS * 4 != 18436)
#error "mesh_integrate_body.glsl: shared memory past its budget"
#endif
#endif
#if VGRID_THREADS * VGRID_LANE_VOXELS != 512
#error "mesh_integrate_body.glsl: a round is an 8^3 block"
#endif

layout(local_size_x = VGRID_THREADS) in;

// One block to write: where its voxels live, and its bin.
// Scalar layout, 12 bytes, byte-identical to MeshBin in mesh_integrator.cpp.
struct MeshBin {
  int ptr;
  uint begin;
  uint count;
};

layout(set = 0, binding = 0, scalar) buffer Tsdf { float tsdf[]; };
layout(set = 0, binding = 1, scalar) buffer Weight { float weight[]; };
layout(set = 0, binding = 2, scalar) readonly buffer Vertices {
  vec3 vertices[];
};
layout(set = 0, binding = 3, scalar) readonly buffer Indices { uint indices[]; };
layout(set = 0, binding = 4, scalar) readonly buffer Bins { MeshBin bins[]; };
layout(set = 0, binding = 5, scalar) readonly buffer BinTris {
  uint bin_tris[];
};
// Each binned slot's block coordinate, as mesh_bin.comp's count pass stored
// it, so the host reads back only the counts.
layout(set = 0, binding = 6, scalar) readonly buffer SlotCoords {
  ivec3 slot_coord[];
};
// The map's block stamps: every block written is stamped changed, since the
// write overwrites it whole.
layout(set = 0, binding = 7, scalar) buffer Stamps { BlockStamp stamps[]; };
// Each slot's write mask (mesh_full_band.comp), read with pc.masked only.
layout(set = 0, binding = 9, scalar) readonly buffer SlotMask {
  uint slot_mask[];
};
// One record per triangle (mesh_tri_records.comp), 64 bytes.
struct TriRecord {
  vec4 a;  // vertex a, and s_T
  vec4 b;  // vertex b, and R
  vec4 c;  // vertex c, and H
  vec4 n;  // the unit normal
};
layout(set = 0, binding = 8, std430) readonly buffer TriRecords {
  TriRecord tri_rec[];
};

// The write is split across dispatches so that none measures more than
// MeshIntegrator::kMaxDispatchBinEntries bin entries; this one writes bins
// [first_bin, first_bin + num_bins), one workgroup each.
layout(push_constant, scalar) uniform PushConstants {
  VoxelGridParams grid;
  uint first_bin;
  uint num_bins;
  uint mode;    // kModeSigned or kModeShell (voxelize::MeshSdfMode)
  float shell;  // shell half-thickness, metres
  uint tick;    // the map's tick, which each block written is stamped with
  uint masked;  // 1: write the voxels the slot masks name (a full band)
} pc;

const uint kModeSigned = 0u;
const uint kModeShell = 1u;
const uint kNoTriangle = 0xFFFFFFFFu;
const uint kFullBlock = 0x80000000u;  // mesh_full_band.comp's

const uint kThreads = uint(VGRID_THREADS);
const uint kLaneVoxels = uint(VGRID_LANE_VOXELS);
const uint kRound = kThreads * kLaneVoxels;
const uint kGroup = uint(VGRID_GROUP);
const uint kChunk = uint(VGRID_CHUNK);
const uint kBuckets = uint(VGRID_BUCKETS);
const uint kMaxSorted = uint(VGRID_MAX_SORTED);
// 2^-14: the margin per metre of coordinate magnitude and of band (see the
// top), about 1,000 times float's unit roundoff.
const float kMargin = 1.0 / 16384.0;

// The sort: each entry's bucket (4 to a word), then the bin's indices in
// bucket order (2 to a word), and each bucket's first sorted position.
shared uint s_bucket[VGRID_MAX_SORTED / 4];
shared uint s_order[VGRID_MAX_SORTED / 2];
shared uint s_start[VGRID_BUCKETS];
shared uint s_cursor[VGRID_BUCKETS];
// One staged chunk: triangle, vertices, support plane (u, its bound at c less
// m_T), ball (centroid, R plus m_T), cylinder axis (normal, H plus m_T).
shared uint s_tri[VGRID_CHUNK];
shared float s_vtx[9 * VGRID_CHUNK];
shared vec4 s_plane[VGRID_CHUNK];
shared vec4 s_ball[VGRID_CHUNK];
shared vec4 s_axis[VGRID_CHUNK];
#if VGRID_SUBGROUP_CULL
// Whether any voxel scans a chunk, by chunk parity: chunk c's flag is set while
// chunk c - 1's voxels finish and cleared once chunk c is staged.
shared uint s_any[2];
#else
// Per culling group: its voxels' box, the largest limit among its scanning
// voxels (as float bits + 1; 0 for none), and the chunk entries it keeps.
shared int s_gbox[6 * VGRID_GROUPS];
shared uint s_glimit[VGRID_GROUPS];
shared uint s_kept[VGRID_GROUPS];
shared uint s_list[VGRID_GROUPS * VGRID_CHUNK];
shared uint s_any;
#endif

vec3 vertexOf(uint t, uint k) { return vertices[indices[3u * t + k]]; }

float sizeOf(vec3 v) { return max(abs(v.x), max(abs(v.y), abs(v.z))); }

// The support plane of triangle (a, b, cc) facing c: u, and its bound at c.
vec4 supportPlane(vec3 c, vec3 a, vec3 b, vec3 cc) {
  vec3 dir = c - vgridClosestPointOnTriangle(c, a, b, cc);
  float len2 = dot(dir, dir);
  if (!(len2 > 0.0)) {
    return vec4(0.0);
  }
  vec3 u = dir * inversesqrt(len2);
  return vec4(u, min(dot(u, c - a), min(dot(u, c - b), dot(u, c - cc))));
}

// The bucket of a bound lb: its floor, bucket * step, is at most lb (to the
// rounding the margin covers). Bucket 0, whose floor no voxel's stop can pass,
// takes every bound under one step, and NaN.
uint bucketOf(float lb, float step) {
  float k = floor(lb / step);
  return k >= 1.0 ? uint(min(k, float(kBuckets - 1u))) : 0u;
}

// The sorted bin's entry at position pos, and the bucket holding it.
uint sortedIndex(uint pos) {
  return (s_order[pos >> 1u] >> ((pos & 1u) << 4u)) & 0xFFFFu;
}
uint bucketAt(uint pos) {
  uint lo = 0u;  // s_start[0] == 0 <= pos
  uint hi = kBuckets;
  while (hi - lo > 1u) {
    uint mid = (lo + hi) >> 1u;
    if (s_start[mid] <= pos) {
      lo = mid;
    } else {
      hi = mid;
    }
  }
  return lo;
}

// Voxel k of invocation lid in the round starting at v0, and whether it is
// one of the block's.
ivec3 voxelOf(uint v0, uint lid, uint k, int bs, uint vpb, out bool live) {
  if (bs == 8) {
    live = true;
    uint w = lid & 31u;
    uint g = lid >> 5u;
#if VGRID_LANE_VOXELS == 2
    // A 4x4x4 brick per group; the lane's two voxels are a z pair.
    return ivec3(int((g & 1u) * 4u + (w & 3u)),
                 int(((g >> 1u) & 1u) * 4u + ((w >> 2u) & 3u)),
                 int((g >> 2u) * 4u + (w >> 4u) * 2u + k));
#else
    // A 4x4x2 brick per group.
    return ivec3(int((g & 1u) * 4u + (w & 3u)),
                 int(((g >> 1u) & 1u) * 4u + ((w >> 2u) & 3u)),
                 int((g >> 2u) * 2u + (w >> 4u)));
#endif
  }
  uint v = v0 + k * kThreads + lid;
  live = v < vpb;
  return vgridLocalBase(live ? v : 0u, bs);
}

// Whether local voxel lc lies one step from a full neighbour whose direction
// bit `mask` holds: bit (dx + 1) + 3 (dy + 1) + 9 (dz + 1) for the neighbour
// at offset (dx, dy, dz) (mesh_full_band.comp).
bool apronVoxel(ivec3 lc, int bs, uint mask) {
  for (int dz = -1; dz <= 1; ++dz) {
    if ((dz < 0 && lc.z != 0) || (dz > 0 && lc.z != bs - 1)) continue;
    for (int dy = -1; dy <= 1; ++dy) {
      if ((dy < 0 && lc.y != 0) || (dy > 0 && lc.y != bs - 1)) continue;
      for (int dx = -1; dx <= 1; ++dx) {
        if ((dx < 0 && lc.x != 0) || (dx > 0 && lc.x != bs - 1)) continue;
        uint bit = uint((dx + 1) + 3 * (dy + 1) + 9 * (dz + 1));
        if (bit != 13u && ((mask >> bit) & 1u) != 0u) return true;
      }
    }
  }
  return false;
}

// Measure triangle t, vertices (a, b, c), from p: the minimum over (d^2, t).
void measure(vec3 p, uint t, vec3 a, vec3 b, vec3 c, inout float best_d2,
             inout uint best_t, inout vec3 best_closest) {
  vec3 q = vgridClosestPointOnTriangle(p, a, b, c);
  vec3 d = p - q;
  float d2 = dot(d, d);
  if (d2 < best_d2 || (d2 == best_d2 && t < best_t)) {
    best_d2 = d2;
    best_t = t;
    best_closest = q;
  }
}

// Whether a bound rules a triangle out against a squared limit (strictly).
bool ruledOut(float lb, float limit) { return lb > 0.0 && lb * lb > limit; }

#if VGRID_SUBGROUP_CULL
// Which of the lane's voxels scan the chunk at `base` (in the unsorted tail,
// every wanted one), and its limit's bits + 1 over them: 0 when none does.
uint scanBits(uint base, uint nsorted, float step, float trunc2,
              bool wanted[VGRID_LANE_VOXELS], float reach[VGRID_LANE_VOXELS],
              float best_d2[VGRID_LANE_VOXELS],
              inout bool scanning[VGRID_LANE_VOXELS]) {
  bool tail = base >= nsorted;
  float floor_c = tail ? 0.0 : float(bucketAt(base)) * step;
  uint bits = 0u;
  [[unroll]] for (uint k = 0u; k < kLaneVoxels; ++k) {
    float limit = min(best_d2[k], trunc2);
    if (tail) {
      scanning[k] = wanted[k];
    } else if (scanning[k] && ruledOut(floor_c - reach[k], limit)) {
      scanning[k] = false;
    }
    if (scanning[k]) {
      bits = max(bits, floatBitsToUint(limit) + 1u);
    }
  }
  return bits;
}

// A lower bound on staged entry j's distance from any voxel of the group's
// box, as the shared build's cull.
float groupBound(uint j, vec3 gcentre, vec3 ghalf, vec3 goff) {
  vec4 plane = s_plane[j];
  vec4 ball = s_ball[j];
  float lb = plane.w + dot(plane.xyz, goff) - dot(abs(plane.xyz), ghalf);
  return max(lb, length(max(abs(ball.xyz - gcentre) - ghalf, vec3(0.0))) -
                     ball.w);
}
#endif

void main() {
  uint lid = gl_LocalInvocationID.x;
  MeshBin bin = bins[pc.first_bin + gl_WorkGroupID.x];
  uint vpb = uint(pc.grid.voxels_per_block);
  uint slot = uint(bin.ptr) / vpb;
  if (lid == 0u) {
    stamps[slot].changed = pc.tick;
  }
  uint wmask = pc.masked != 0u ? slot_mask[slot] : kFullBlock;
  bool whole = (wmask & kFullBlock) != 0u;

  int bs = pc.grid.block_size;
  ivec3 coord = slot_coord[slot];
  float span = float(bs - 1) * pc.grid.voxel_size;
  vec3 centre = vec3(coord * bs) * pc.grid.voxel_size + vec3(0.5 * span);
  // Every binned triangle comes within band of the centre, as does every
  // voxel of the block; the sort's buckets divide it.
  float band = pc.grid.trunc_dist + 0.8660254 * span;
  float step = band / float(kBuckets);
  float centre_size = sizeOf(centre);
  float trunc_dist = pc.grid.trunc_dist;
  float trunc2 = trunc_dist * trunc_dist;

  // Sort the first kMaxSorted entries; any past them, a tail, are measured in
  // bin order after them, without the stop.
  uint nsorted = min(bin.count, kMaxSorted);
  for (uint i = lid; i < kMaxSorted / 2u; i += kThreads) {
    s_order[i] = 0u;
    if (i < kMaxSorted / 4u) {
      s_bucket[i] = 0u;
    }
    if (i < kBuckets) {
      s_start[i] = 0u;
    }
  }
#if VGRID_SUBGROUP_CULL
  if (lid < 2u) {
    s_any[lid] = 0u;
  }
#endif
  barrier();
  for (uint i = lid; i < nsorted; i += kThreads) {
    uint t = bin_tris[bin.begin + i];
    vec4 ra = tri_rec[t].a;
    vec3 a = ra.xyz;
    vec3 b = tri_rec[t].b.xyz;
    vec3 c = tri_rec[t].c.xyz;
    float m = kMargin * (max(centre_size, ra.w) + band);
    uint key = bucketOf(supportPlane(centre, a, b, c).w - m, step);
    atomicAdd(s_start[key], 1u);
    atomicOr(s_bucket[i >> 2u], key << ((i & 3u) << 3u));
  }
  barrier();
  // Exclusive prefix sum of the bucket counts (Hillis-Steele).
  uint own = lid < kBuckets ? s_start[lid] : 0u;
  uint sum = own;
  for (uint off = 1u; off < kBuckets; off <<= 1u) {
    uint add = lid < kBuckets && lid >= off ? s_start[lid - off] : 0u;
    barrier();
    if (lid < kBuckets) {
      sum += add;
      s_start[lid] = sum;
    }
    barrier();
  }
  if (lid < kBuckets) {
    s_start[lid] = sum - own;
    s_cursor[lid] = sum - own;
  }
  barrier();
  for (uint i = lid; i < nsorted; i += kThreads) {
    uint key = (s_bucket[i >> 2u] >> ((i & 3u) << 3u)) & 0xFFu;
    uint pos = atomicAdd(s_cursor[key], 1u);
    atomicOr(s_order[pos >> 1u], i << ((pos & 1u) << 4u));
  }
  barrier();

#if VGRID_SUBGROUP_CULL
  // The group is the subgroup; its voxels follow from its id and the lane's.
  // TODO: request full 32-wide subgroups (REQUIRE_FULL_SUBGROUPS and a
  // required size of 32) once the core's KernelSetBuilder passes stage flags;
  // until then the host builds this only where subgroups are 32 wide.
  uint g = gl_SubgroupID;
  uint lane = gl_SubgroupInvocationID;
  uint vlid = g * kGroup + lane;
#else
  uint g = lid / kGroup;
  uint lane = lid % kGroup;
  uint vlid = lid;
#endif
  for (uint v0 = 0u; v0 < vpb; v0 += kRound) {
    bool live[VGRID_LANE_VOXELS];
    // The voxels this round measures: every live one of a whole block, the
    // apron's of another.
    bool wanted[VGRID_LANE_VOXELS];
    bool scanning[VGRID_LANE_VOXELS];
    uint local[VGRID_LANE_VOXELS];
    vec3 p[VGRID_LANE_VOXELS];
    float reach[VGRID_LANE_VOXELS];
    float best_d2[VGRID_LANE_VOXELS];
    uint best_t[VGRID_LANE_VOXELS];
    vec3 best_closest[VGRID_LANE_VOXELS];

#if VGRID_SUBGROUP_CULL
    ivec3 wlo = ivec3(0x7FFFFFFF);
    ivec3 whi = ivec3(-0x7FFFFFFF);
#else
    if (lane < 3u) {
      s_gbox[6u * g + lane] = 0x7FFFFFFF;
      s_gbox[6u * g + 3u + lane] = -0x7FFFFFFF;
    }
    barrier();
#endif
    [[unroll]] for (uint k = 0u; k < kLaneVoxels; ++k) {
      // Node-centred voxels (voxel * voxel_size), matching
      // volume/voxel_coords.hpp::voxel_to_world and tsdf_integrate.comp.
      ivec3 lc = voxelOf(v0, vlid, k, bs, vpb, live[k]);
      local[k] = uint(vgridLocalIndex(lc, bs));
      p[k] = vec3(coord * bs + lc) * pc.grid.voxel_size;
      wanted[k] = live[k] && (whole || apronVoxel(lc, bs, wmask));
      if (wanted[k]) {
#if VGRID_SUBGROUP_CULL
        wlo = min(wlo, lc);
        whi = max(whi, lc);
#else
        atomicMin(s_gbox[6u * g], lc.x);
        atomicMin(s_gbox[6u * g + 1u], lc.y);
        atomicMin(s_gbox[6u * g + 2u], lc.z);
        atomicMax(s_gbox[6u * g + 3u], lc.x);
        atomicMax(s_gbox[6u * g + 4u], lc.y);
        atomicMax(s_gbox[6u * g + 5u], lc.z);
#endif
      }
      best_d2[k] = 3.4e38;
      best_t[k] = kNoTriangle;
      best_closest[k] = p[k];
      reach[k] = length(p[k] - centre) + kMargin * (centre_size + band);
      scanning[k] = wanted[k];
    }
#if VGRID_SUBGROUP_CULL
    ivec3 glo = subgroupMin(wlo);
    ivec3 ghi = subgroupMax(whi);
    vec3 gcentre =
        (vec3(coord * bs) + 0.5 * vec3(glo + ghi)) * pc.grid.voxel_size;
    vec3 ghalf = 0.5 * vec3(ghi - glo) * pc.grid.voxel_size;
    vec3 goff = gcentre - centre;
    // lbits: this lane's limit bits + 1 for the chunk at base (0: it does not
    // scan it); glimit: the group's largest.
    uint lbits = 0u;
    if (bin.count > 0u) {
      lbits = scanBits(0u, nsorted, step, trunc2, wanted, reach, best_d2,
                       scanning);
      if (subgroupAny(lbits != 0u) && subgroupElect()) {
        atomicOr(s_any[0], 1u);
      }
    }
    uint glimit = subgroupMax(lbits);
    barrier();
    uint par = 0u;
    for (uint base = 0u; base < bin.count;) {
      if (s_any[par] == 0u) {
        // No voxel scans the rest of the sorted entries: on to the tail, if
        // there is one, where every wanted voxel scans every entry.
        if (base >= nsorted || nsorted == bin.count) {
          break;
        }
        base = nsorted;
        lbits = scanBits(base, nsorted, step, trunc2, wanted, reach, best_d2,
                         scanning);
        if (subgroupAny(lbits != 0u) && subgroupElect()) {
          atomicOr(s_any[par ^ 1u], 1u);
        }
        glimit = subgroupMax(lbits);
        par ^= 1u;
        barrier();
        continue;
      }
#else
    barrier();

    for (uint base = 0u; base < bin.count; base += kChunk) {
      if (lane == 0u) {
        s_glimit[g] = 0u;
        s_kept[g] = 0u;
      }
      if (lid == 0u) {
        s_any = 0u;
      }
      barrier();
      bool tail = base >= nsorted;
      float floor_c = tail ? 0.0 : float(bucketAt(base)) * step;
      uint glimit_bits = 0u;
      [[unroll]] for (uint k = 0u; k < kLaneVoxels; ++k) {
        float limit = min(best_d2[k], trunc2);
        if (tail) {
          scanning[k] = wanted[k];
        } else if (scanning[k] && ruledOut(floor_c - reach[k], limit)) {
          scanning[k] = false;
        }
        if (scanning[k]) {
          glimit_bits = max(glimit_bits, floatBitsToUint(limit) + 1u);
        }
      }
      if (glimit_bits != 0u) {
        atomicOr(s_any, 1u);
        atomicMax(s_glimit[g], glimit_bits);
      }
      barrier();
      if (s_any == 0u) {
        if (tail || nsorted == bin.count) {
          break;
        }
        // Every voxel is done with the sorted part: on to the tail.
        barrier();
        base = nsorted - kChunk;
        continue;
      }
#endif

      uint n = min(kChunk, bin.count - base);
      for (uint j = lid; j < n; j += kThreads) {
        uint pos = base + j;
        uint t = bin_tris[bin.begin + (pos < nsorted ? sortedIndex(pos) : pos)];
        vec4 ra = tri_rec[t].a;
        vec4 rb = tri_rec[t].b;
        vec4 rc = tri_rec[t].c;
        vec4 rn = tri_rec[t].n;
        vec3 a = ra.xyz;
        vec3 b = rb.xyz;
        vec3 c = rc.xyz;
        float m = kMargin * (max(centre_size, ra.w) + band);
        s_tri[j] = t;
        s_vtx[9u * j] = a.x;
        s_vtx[9u * j + 1u] = a.y;
        s_vtx[9u * j + 2u] = a.z;
        s_vtx[9u * j + 3u] = b.x;
        s_vtx[9u * j + 4u] = b.y;
        s_vtx[9u * j + 5u] = b.z;
        s_vtx[9u * j + 6u] = c.x;
        s_vtx[9u * j + 7u] = c.y;
        s_vtx[9u * j + 8u] = c.z;
        vec4 plane = supportPlane(centre, a, b, c);
        s_plane[j] = vec4(plane.xyz, plane.w - m);
        vec3 mid = (a + b + c) * (1.0 / 3.0);
        s_ball[j] = vec4(mid, rb.w + m);
        s_axis[j] = vec4(rn.xyz, rc.w + m);
      }
      barrier();
#if VGRID_SUBGROUP_CULL
      if (lid == 0u) {
        s_any[par] = 0u;
      }
      // The group's kept entries: bit i of keep_lo is entry i, of keep_hi
      // entry 32 + i.
      uint keep_lo = 0u;
      uint keep_hi = 0u;
      if (glimit != 0u) {
        float gl = uintBitsToFloat(glimit - 1u);
        keep_lo = subgroupBallot(
                      lane < n &&
                      !ruledOut(groupBound(lane, gcentre, ghalf, goff), gl))
                      .x;
        keep_hi = subgroupBallot(lane + kGroup < n &&
                                 !ruledOut(groupBound(lane + kGroup, gcentre,
                                                      ghalf, goff),
                                           gl))
                      .x;
      }
      if (lbits != 0u) {
        [[unroll]] for (uint half_k = 0u; half_k < 2u; ++half_k) {
          uint mask = half_k == 0u ? keep_lo : keep_hi;
          while (mask != 0u) {
            uint j = uint(findLSB(mask)) + half_k * kGroup;
            mask &= mask - 1u;
#else
      // The group keeps what some voxel of its box may need: the support
      // plane at the box's nearest corner along u, and the ball.
      uint glimit = s_glimit[g];
      if (glimit != 0u) {
        float gl = uintBitsToFloat(glimit - 1u);
        ivec3 glo = ivec3(s_gbox[6u * g], s_gbox[6u * g + 1u],
                          s_gbox[6u * g + 2u]);
        ivec3 ghi = ivec3(s_gbox[6u * g + 3u], s_gbox[6u * g + 4u],
                          s_gbox[6u * g + 5u]);
        vec3 gcentre =
            (vec3(coord * bs) + 0.5 * vec3(glo + ghi)) * pc.grid.voxel_size;
        vec3 ghalf = 0.5 * vec3(ghi - glo) * pc.grid.voxel_size;
        vec3 goff = gcentre - centre;
        for (uint j = lane; j < n; j += kGroup) {
          vec4 plane = s_plane[j];
          vec4 ball = s_ball[j];
          float lb = plane.w + dot(plane.xyz, goff) -
                     dot(abs(plane.xyz), ghalf);
          lb = max(lb, length(max(abs(ball.xyz - gcentre) - ghalf,
                                  vec3(0.0))) -
                           ball.w);
          if (!ruledOut(lb, gl)) {
            s_list[g * kChunk + atomicAdd(s_kept[g], 1u)] = j;
          }
        }
      }
      barrier();
      if (glimit_bits != 0u) {
        uint kept = s_kept[g];
        for (uint i = 0u; i < kept; ++i) {
          uint j = s_list[g * kChunk + i];
#endif
          vec4 plane = s_plane[j];
          // The support plane first: it alone rules out about three checks
          // in four, so the cylinder, and the loads it takes, wait for a
          // voxel the plane leaves in.
          bool cand[VGRID_LANE_VOXELS];
          bool any_cand = false;
          [[unroll]] for (uint k = 0u; k < kLaneVoxels; ++k) {
            cand[k] = scanning[k] &&
                      !ruledOut(plane.w + dot(plane.xyz, p[k] - centre),
                                min(best_d2[k], trunc2));
            any_cand = any_cand || cand[k];
          }
          if (any_cand) {
            vec4 ball = s_ball[j];
            vec4 axis = s_axis[j];
            [[unroll]] for (uint k = 0u; k < kLaneVoxels; ++k) {
              if (!cand[k]) {
                continue;
              }
              vec3 off = p[k] - ball.xyz;
              float h = dot(off, axis.xyz);
              float dr = max(length(off - h * axis.xyz) - ball.w, 0.0);
              float dz = max(abs(h) - axis.w, 0.0);
              if (dr * dr + dz * dz > min(best_d2[k], trunc2)) {
                continue;
              }
              measure(p[k], s_tri[j],
                      vec3(s_vtx[9u * j], s_vtx[9u * j + 1u],
                           s_vtx[9u * j + 2u]),
                      vec3(s_vtx[9u * j + 3u], s_vtx[9u * j + 4u],
                           s_vtx[9u * j + 5u]),
                      vec3(s_vtx[9u * j + 6u], s_vtx[9u * j + 7u],
                           s_vtx[9u * j + 8u]),
                      best_d2[k], best_t[k], best_closest[k]);
            }
          }
#if VGRID_SUBGROUP_CULL
          }
        }
      }
      // The next chunk's scanners, from this one's minima; its flag takes
      // the other parity, cleared once this chunk was staged.
      base += kChunk;
      if (base < bin.count) {
        lbits = scanBits(base, nsorted, step, trunc2, wanted, reach, best_d2,
                         scanning);
        if (subgroupAny(lbits != 0u) && subgroupElect()) {
          atomicOr(s_any[par ^ 1u], 1u);
        }
        glimit = subgroupMax(lbits);
      }
      par ^= 1u;
      barrier();
    }
    // Every invocation has read both flags (now clear) before the next round
    // sets one.
    barrier();
#else
        }
      }
      barrier();
    }
#endif

    [[unroll]] for (uint k = 0u; k < kLaneVoxels; ++k) {
      if (!live[k]) {
        continue;
      }
      uint idx = uint(bin.ptr) + local[k];
      if (!wanted[k]) {
        tsdf[idx] = 0.0;
        weight[idx] = 0.0;
        continue;
      }
      float dist = sqrt(best_d2[k]);
      bool observed = best_t[k] != kNoTriangle && dist <= trunc_dist;
      float sdf = 0.0;
      if (observed) {
        if (pc.mode == kModeSigned) {
          vec3 a = vertexOf(best_t[k], 0u);
          vec3 b = vertexOf(best_t[k], 1u);
          vec3 c = vertexOf(best_t[k], 2u);
          sdf = dot(p[k] - best_closest[k], cross(b - a, c - a)) >= 0.0
                    ? dist
                    : -dist;
        } else {
          sdf = dist - pc.shell;
        }
      }
      tsdf[idx] = observed ? clamp(sdf, -trunc_dist, trunc_dist) : 0.0;
      weight[idx] = observed ? 1.0 : 0.0;
    }
  }
}
