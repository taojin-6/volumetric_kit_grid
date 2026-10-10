// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// GPU test for codec::Decoder, and the codec end to end: a frame decodes to
// exactly the encoded grid's observed blocks and voxels, within the bound an
// orthonormal transform guarantees; the decoded grid meshes to the analytic
// sphere it came from; frame after frame into one grid leaves exactly the
// last frame; every refusal before the grid is touched leaves it untouched;
// a grid whose hash table cannot place the frame says so and recovers after a
// resize; read_frame_info; the stage rows; the moves. Links recon_mesh, which
// the codec tier itself may not. Skips with no device.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <utility>
#include <vector>

#include "codec_fixture.hpp"
#include "volumetric_kit/core/base/stage_metrics.hpp"
#include "volumetric_kit/core/vulkan/gpu_timer.hpp"
#include "volumetric_kit/recon/codec/decoder.hpp"
#include "volumetric_kit/recon/codec/encoder.hpp"
#include "volumetric_kit/recon/mesh/marching_cubes.hpp"
#include "volumetric_kit/recon/mesh/mesh.hpp"

#include "gpu_test.hpp"

namespace vkc = volumetric_kit::core;

namespace codec = volumetric_kit::recon::codec;
namespace mesh = volumetric_kit::recon::mesh;
using namespace codec_fixture;

namespace {

using Bytes = std::vector<std::uint8_t>;

const vkc::StageRow* find_row(const vkc::StageMetrics& m, const char* name) {
  for (const vkc::StageRow& row : m.rows()) {
    if (std::strcmp(row.name, name) == 0) {
      return &row;
    }
  }
  return nullptr;
}

vkc::Result<Bytes> encode_with(Gpu& gpu, vol::VoxelBlockGrid& grid,
                               const codec::EncoderConfig& config = {}) {
  VKC_ASSIGN(codec::Encoder enc,
             codec::Encoder::create(gpu.device, gpu.allocator, config));
  return enc.encode(grid);
}

// A grid built from a frame's header, as a player builds one.
vkc::Result<vol::VoxelBlockGrid> grid_for(Gpu& gpu, const Bytes& frame,
                                          std::int32_t num_buckets = 512) {
  VKC_ASSIGN(const codec::FrameInfo info,
             codec::read_frame_info(frame.data(), frame.size()));
  GridShape shape;
  shape.voxel_size = info.voxel_size;
  shape.trunc_dist = info.trunc_dist;
  shape.num_buckets = num_buckets;
  return make_grid(gpu, shape);
}

codec::EncoderConfig near_lossless() {
  codec::EncoderConfig c;
  c.params.coefficient_count = codec::kVoxelsPerBlock;
  c.params.quantization_scale = 0.002f;
  return c;
}

// The source's observed blocks and voxels, against the decoded grid's: the
// same blocks exactly, weight 1 exactly where the source observed and 0
// elsewhere, tsdf 0 where unobserved, and within `bound` metres where
// observed. Reports the largest error it saw.
int compare_decoded(const Snapshot& src, const Snapshot& dec, float bound,
                    float* max_err_out) {
  std::vector<vr::Vec3i> observed;
  std::vector<std::size_t> src_index;
  for (std::size_t i = 0; i < src.coords.size(); ++i) {
    for (std::uint32_t v = 0; v < kVpb; ++v) {
      if (src.weight[i * kVpb + v] >= vol::kObservedWeight) {
        observed.push_back(src.coords[i]);
        src_index.push_back(i);
        break;
      }
    }
  }
  CHECK(dec.coords == observed);
  float max_err = 0.0f;
  for (std::size_t d = 0; d < dec.coords.size(); ++d) {
    const std::size_t s = src_index[d];
    for (std::uint32_t v = 0; v < kVpb; ++v) {
      const bool obs = src.weight[s * kVpb + v] >= vol::kObservedWeight;
      const float w = dec.weight[d * kVpb + v];
      const float t = dec.tsdf[d * kVpb + v];
      if (obs) {
        CHECK(w == 1.0f);
        max_err = std::max(max_err, std::fabs(t - src.tsdf[s * kVpb + v]));
      } else {
        CHECK(w == 0.0f);
        CHECK(t == 0.0f);
      }
    }
  }
  CHECK(max_err <= bound);
  *max_err_out = max_err;
  return 0;
}

// Near-lossless (K = 512, steps 0.002): each coefficient is within half a
// step, so over a block the error's norm is at most sqrt(512) * step / 2 in
// units of trunc_dist -- a bound on any one voxel, however loose.
int round_trip_case(Gpu& gpu, codec::Decoder& dec) {
  const Sphere s{vr::Vec3f(0.01f, 0.0f, -0.01f), 0.1f};
  // Never observed, so the encoder drops both: one sorts after every sphere
  // block, the other before them all, so dropping it slides the whole
  // readback down one block.
  const std::vector<vr::Vec3i> far = {{30, 30, 30}, {-30, 2, -30}};
  vkc::Result<vol::VoxelBlockGrid> g = sphere_grid(gpu, s, {}, far);
  CHECK(g.ok());
  vol::VoxelBlockGrid src = std::move(g).value();
  vkc::Result<Snapshot> src_snap = snapshot(gpu, src);
  CHECK(src_snap.ok());

  vkc::Result<Bytes> frame = encode_with(gpu, src, near_lossless());
  CHECK(frame.ok());
  vkc::Result<vol::VoxelBlockGrid> d = grid_for(gpu, frame.value());
  CHECK(d.ok());
  vol::VoxelBlockGrid out = std::move(d).value();
  CHECK(dec.decode(frame.value().data(), frame.value().size(), out).ok());
  vkc::Result<Snapshot> out_snap = snapshot(gpu, out);
  CHECK(out_snap.ok());
  // Every block decoded is stamped changed, at a tick of the decode's own.
  {
    CHECK(out.map().tick() > 1);
    vkc::Result<std::vector<vol::BlockIndex>> active =
        out.map().compact_active_blocks();
    vkc::Result<std::vector<vol::BlockStamp>> st =
        out.map().read_block_stamps();
    CHECK(active.ok() && st.ok() && !active.value().empty());
    for (const vol::BlockIndex& b : active.value()) {
      CHECK(st.value()[static_cast<std::uint32_t>(b.ptr) / 512u].changed ==
            out.map().tick());
    }
    // The same frame again leaves every block as it was, and so stamps none.
    const std::uint32_t first = out.map().tick();
    CHECK(dec.decode(frame.value().data(), frame.value().size(), out).ok());
    CHECK(out.map().tick() != first);
    st = out.map().read_block_stamps();
    CHECK(st.ok());
    for (const vol::BlockIndex& b : active.value()) {
      CHECK(st.value()[static_cast<std::uint32_t>(b.ptr) / 512u].changed ==
            first);
    }
  }

  const float bound = std::sqrt(512.0f) * 0.002f / 2.0f * kTrunc + 1e-6f;
  float max_err = 0.0f;
  CHECK(compare_decoded(src_snap.value(), out_snap.value(), bound, &max_err) ==
        0);
  std::printf(
      "near-lossless: %zu blocks, %zu bytes, max voxel error %.3g m "
      "(bound %.3g)\n",
      out_snap.value().coords.size(), frame.value().size(), double(max_err),
      double(bound));
  return 0;
}

// Mesh a grid, and measure how far its vertices sit from the true sphere.
struct MeshFit {
  std::size_t triangles = 0;
  double max_off = 0.0;   // metres, worst vertex
  double mean_off = 0.0;  // metres
};

vkc::Result<MeshFit> fit(Gpu& gpu, vol::VoxelBlockGrid& grid, const Sphere& s) {
  VKC_ASSIGN(mesh::MarchingCubes mc,
             mesh::MarchingCubes::create(gpu.device, gpu.allocator));
  VKC_ASSIGN(const mesh::Mesh m, mc.extract_host(grid));
  MeshFit f;
  f.triangles = m.triangle_count();
  for (const mesh::Vertex& v : m.vertices) {
    const double off = std::fabs(double(s.sdf(v.position)));
    f.max_off = std::max(f.max_off, off);
    f.mean_off += off;
  }
  if (!m.vertices.empty()) {
    f.mean_off /= double(m.vertices.size());
  }
  return f;
}

// The promise the codec is for: the decoded grid meshes to the surface the
// source did. Judged against the analytic sphere, so a decoded mesh that
// matched a wrong source mesh would not pass.
int mesh_case(Gpu& gpu, codec::Decoder& dec) {
  const Sphere s{vr::Vec3f(0.0f, 0.02f, 0.0f), 0.12f};
  vkc::Result<vol::VoxelBlockGrid> g = sphere_grid(gpu, s);
  CHECK(g.ok());
  vol::VoxelBlockGrid src = std::move(g).value();
  vkc::Result<MeshFit> truth = fit(gpu, src, s);
  CHECK(truth.ok());
  CHECK(truth.value().triangles > 1000);

  // The defaults are tuned on room0, not on this sphere, and a smooth surface
  // is where they are weakest -- its energy is all in the low bands, which
  // the defaults' coarse AC step quantizes as coarsely as the high ones -- so
  // their bounds are the looser: 0.66 voxels worst and 0.070 mean on an M5
  // Max, against 0.27 and 0.060 at the prior engine's K = 32. The two mean
  // bounds and the defaults' worst-vertex bound are 1.2-1.35x what that GPU
  // measures (near-lossless: 0.0026 mean) -- room for another device's
  // rounding, and no more, so a regression shows.
  struct Setting {
    const char* name;
    codec::EncoderConfig config;
    double max_off_voxels;   // worst vertex, in voxels
    double mean_off_voxels;  // mean vertex, in voxels
    double tri_ratio;        // allowed triangle-count drift either way
  };
  const Setting settings[] = {
      {"defaults", {}, 0.8, 0.085, 0.02},
      {"near-lossless", near_lossless(), 0.1, 0.0035, 0.005}};
  std::printf("source mesh: %zu triangles, max %.3g / mean %.3g voxels off\n",
              truth.value().triangles, truth.value().max_off / kVoxel,
              truth.value().mean_off / kVoxel);
  for (const Setting& setting : settings) {
    vkc::Result<Bytes> frame = encode_with(gpu, src, setting.config);
    CHECK(frame.ok());
    vkc::Result<vol::VoxelBlockGrid> d = grid_for(gpu, frame.value());
    CHECK(d.ok());
    vol::VoxelBlockGrid out = std::move(d).value();
    CHECK(dec.decode(frame.value().data(), frame.value().size(), out).ok());
    vkc::Result<MeshFit> got = fit(gpu, out, s);
    CHECK(got.ok());
    vkc::Result<Snapshot> out_snap = snapshot(gpu, out);
    CHECK(out_snap.ok());
    const double ratio =
        double(got.value().triangles) / double(truth.value().triangles);
    std::printf(
        "%s: %zu bytes (%.1f B/block), %zu triangles (x%.4f), max "
        "%.3g / mean %.3g voxels off\n",
        setting.name, frame.value().size(),
        double(frame.value().size()) / double(out_snap.value().coords.size()),
        got.value().triangles, ratio, got.value().max_off / kVoxel,
        got.value().mean_off / kVoxel);
    CHECK(got.value().max_off <= setting.max_off_voxels * kVoxel);
    CHECK(got.value().mean_off <= setting.mean_off_voxels * kVoxel);
    CHECK(std::fabs(ratio - 1.0) <= setting.tri_ratio);
  }
  return 0;
}

// Frame after frame into one grid leaves exactly the last frame -- the same
// state, bit for bit, as decoding that frame into a fresh grid. The two
// spheres overlap, so the grid keeps some blocks, drops some and gains some.
int sequence_case(Gpu& gpu, codec::Decoder& dec) {
  const Sphere a{vr::Vec3f(0.0f), 0.08f};
  const Sphere b{vr::Vec3f(0.05f, 0.0f, 0.0f), 0.07f};
  vkc::Result<vol::VoxelBlockGrid> ga = sphere_grid(gpu, a);
  vkc::Result<vol::VoxelBlockGrid> gb = sphere_grid(gpu, b);
  CHECK(ga.ok() && gb.ok());
  vkc::Result<Bytes> fa = encode_with(gpu, ga.value());
  vkc::Result<Bytes> fb = encode_with(gpu, gb.value());
  CHECK(fa.ok() && fb.ok());

  vkc::Result<vol::VoxelBlockGrid> g = grid_for(gpu, fa.value());
  CHECK(g.ok());
  vol::VoxelBlockGrid player = std::move(g).value();
  CHECK(dec.decode(fa.value().data(), fa.value().size(), player).ok());
  vkc::Result<Snapshot> after_a = snapshot(gpu, player);
  CHECK(after_a.ok());
  vkc::Result<std::vector<vol::BlockIndex>> slots_a = active_sorted(player);
  CHECK(slots_a.ok());
  CHECK(dec.decode(fb.value().data(), fb.value().size(), player).ok());
  vkc::Result<Snapshot> after_b = snapshot(gpu, player);
  CHECK(after_b.ok());
  vkc::Result<std::vector<vol::BlockIndex>> slots_b = active_sorted(player);
  CHECK(slots_b.ok());
  // A block in both frames costs no allocation: it keeps its slot, where a
  // remove-and-reallocate would hand it whatever the LIFO heap had on top.
  for (const vol::BlockIndex& a_block : slots_a.value()) {
    for (const vol::BlockIndex& b_block : slots_b.value()) {
      if (a_block.coord == b_block.coord) {
        CHECK(a_block.ptr == b_block.ptr);
      }
    }
  }

  vkc::Result<vol::VoxelBlockGrid> f = grid_for(gpu, fb.value());
  CHECK(f.ok());
  vol::VoxelBlockGrid fresh = std::move(f).value();
  CHECK(dec.decode(fb.value().data(), fb.value().size(), fresh).ok());
  vkc::Result<Snapshot> only_b = snapshot(gpu, fresh);
  CHECK(only_b.ok());
  CHECK(after_b.value() == only_b.value());

  // Nothing to allocate -- the same frame again, then one holding a subset of
  // it -- takes the path that reuses the merge's slots instead of compacting
  // twice, and must land on the same state as a fresh grid.
  CHECK(dec.decode(fb.value().data(), fb.value().size(), player).ok());
  vkc::Result<Snapshot> again = snapshot(gpu, player);
  CHECK(again.ok() && again.value() == only_b.value());
  vkc::Result<std::vector<vol::BlockIndex>> slots_again = active_sorted(player);
  CHECK(slots_again.ok());
  CHECK(std::equal(slots_again.value().begin(), slots_again.value().end(),
                   slots_b.value().begin(), slots_b.value().end(),
                   [](const vol::BlockIndex& x, const vol::BlockIndex& y) {
                     return x.coord == y.coord && x.ptr == y.ptr;
                   }));
  std::vector<vol::BlockIndex> dropped(12);
  for (std::size_t i = 0; i < dropped.size(); ++i) {
    dropped[i].coord = after_b.value().coords[i * 5];
  }
  vkc::Result<std::uint32_t> removed = gb.value().remove(
      dropped.data(), static_cast<std::uint32_t>(dropped.size()));
  CHECK(removed.ok() && removed.value() == 0);
  vkc::Result<Bytes> fsub = encode_with(gpu, gb.value());
  CHECK(fsub.ok());
  CHECK(dec.decode(fsub.value().data(), fsub.value().size(), player).ok());
  vkc::Result<Snapshot> after_sub = snapshot(gpu, player);
  CHECK(after_sub.ok());
  CHECK(after_sub.value().coords.size() ==
        after_b.value().coords.size() - dropped.size());
  vkc::Result<vol::VoxelBlockGrid> fs = grid_for(gpu, fsub.value());
  CHECK(fs.ok());
  CHECK(dec.decode(fsub.value().data(), fsub.value().size(), fs.value()).ok());
  vkc::Result<Snapshot> only_sub = snapshot(gpu, fs.value());
  CHECK(only_sub.ok() && after_sub.value() == only_sub.value());

  // The premise: the grid did keep, drop and gain blocks between the two.
  std::size_t kept = 0;
  for (const vr::Vec3i& c : after_a.value().coords) {
    kept += std::binary_search(after_b.value().coords.begin(),
                               after_b.value().coords.end(), c, coord_less)
                ? 1
                : 0;
  }
  CHECK(kept > 0);
  CHECK(kept < after_a.value().coords.size());
  CHECK(kept < after_b.value().coords.size());

  // And a frame of no blocks empties the grid.
  vkc::Result<vol::VoxelBlockGrid> e = make_grid(gpu);
  CHECK(e.ok());
  vkc::Result<Bytes> empty = encode_with(gpu, e.value());
  CHECK(empty.ok());
  CHECK(dec.decode(empty.value().data(), empty.value().size(), player).ok());
  vkc::Result<Snapshot> none = snapshot(gpu, player);
  CHECK(none.ok() && none.value().coords.empty());
  return 0;
}

// Every refusal that comes before the grid changes leaves it exactly as it
// was: a corrupt frame, one of other geometry, one too big for the heap.
int untouched_case(Gpu& gpu, codec::Decoder& dec) {
  const Sphere a{vr::Vec3f(0.0f), 0.06f};
  const Sphere b{vr::Vec3f(0.03f, 0.0f, 0.0f), 0.09f};
  vkc::Result<vol::VoxelBlockGrid> ga = sphere_grid(gpu, a);
  vkc::Result<vol::VoxelBlockGrid> gb = sphere_grid(gpu, b);
  CHECK(ga.ok() && gb.ok());
  vkc::Result<Bytes> fa = encode_with(gpu, ga.value());
  vkc::Result<Bytes> fb = encode_with(gpu, gb.value());
  CHECK(fa.ok() && fb.ok());
  vkc::Result<vol::VoxelBlockGrid> g = grid_for(gpu, fa.value());
  CHECK(g.ok());
  vol::VoxelBlockGrid player = std::move(g).value();
  CHECK(dec.decode(fa.value().data(), fa.value().size(), player).ok());
  vkc::Result<Snapshot> before = snapshot(gpu, player);
  CHECK(before.ok());
  auto unchanged = [&]() {
    vkc::Result<Snapshot> now = snapshot(gpu, player);
    return now.ok() && now.value() == before.value();
  };

  // An invalid table entry beyond K is refused before the grid is touched.
  Bytes invalid_table = fb.value();
  const float zero_weight = 0.0f;
  std::memcpy(invalid_table.data() + 44 + 511 * sizeof(float), &zero_weight,
              sizeof(zero_weight));
  CHECK(!dec.decode(invalid_table.data(), invalid_table.size(), player).ok());
  CHECK(unchanged());

  // Truncated, and one byte flipped in the middle of the frame.
  CHECK(!dec.decode(fb.value().data(), fb.value().size() - 1, player).ok());
  CHECK(unchanged());
  Bytes flipped = fb.value();
  flipped[flipped.size() / 2] ^= 0xFF;
  if (!dec.decode(flipped.data(), flipped.size(), player).ok()) {
    CHECK(unchanged());  // (a flip in raw bits can decode; see codec_rans)
  } else {
    CHECK(dec.decode(fa.value().data(), fa.value().size(), player).ok());
  }
  CHECK(unchanged());

  // Another geometry: a frame from a grid of 4 mm voxels.
  GridShape other;
  other.voxel_size = 0.004f;
  vkc::Result<vol::VoxelBlockGrid> go = sphere_grid(gpu, b, other);
  CHECK(go.ok());
  vkc::Result<Bytes> fo = encode_with(gpu, go.value());
  CHECK(fo.ok());
  CHECK(!dec.decode(fo.value().data(), fo.value().size(), player).ok());
  CHECK(unchanged());

  // Too many blocks for the heap: a grid of 32 slots, holding a few blocks.
  GridShape tiny;
  tiny.num_buckets = 4;
  vkc::Result<vol::VoxelBlockGrid> gt = make_grid(gpu, tiny);
  CHECK(gt.ok());
  vol::VoxelBlockGrid small = std::move(gt).value();
  CHECK(allocate(small, {{0, 0, 0}, {1, 0, 0}}).ok());
  CHECK(write_sphere(gpu, small, Sphere{vr::Vec3f(0.02f, 0.02f, 0.02f), 0.02f})
            .ok());
  vkc::Result<Snapshot> small_before = snapshot(gpu, small);
  CHECK(small_before.ok());
  vkc::Result<codec::FrameInfo> info =
      codec::read_frame_info(fb.value().data(), fb.value().size());
  CHECK(info.ok() && info.value().block_count > 32);
  const vkc::Status s = dec.decode(fb.value().data(), fb.value().size(), small);
  CHECK(!s.ok());
  // Too small, not corrupt: the same answer as a table that cannot place the
  // blocks, so one recovery serves both.
  CHECK(s.domain() == vkc::Status::Code::OutOfMemory);
  vkc::Result<Snapshot> small_after = snapshot(gpu, small);
  CHECK(small_after.ok() && small_after.value() == small_before.value());
  // Another geometry with more blocks than the grid has slots is refused for
  // its geometry, read off the header, and not for its size: a player must
  // not be sent to grow a grid that could never take the frame.
  vkc::Result<codec::FrameInfo> other_info =
      codec::read_frame_info(fo.value().data(), fo.value().size());
  CHECK(other_info.ok() && other_info.value().block_count > 32);
  const vkc::Status so =
      dec.decode(fo.value().data(), fo.value().size(), small);
  CHECK(!so.ok() && so.domain() == vkc::Status::Code::InvalidArgument);
  std::int32_t buckets = tiny.num_buckets;
  while (buckets * tiny.bucket_size <
         std::int32_t(info.value().block_count) * 4) {
    buckets *= 2;
  }
  CHECK(small.resize(buckets).ok());
  CHECK(dec.decode(fb.value().data(), fb.value().size(), small).ok());
  vkc::Result<Snapshot> recovered = snapshot(gpu, small);
  CHECK(recovered.ok());
  CHECK(recovered.value().coords.size() == info.value().block_count);
  return 0;
}

// A hash table that cannot place the frame's blocks -- few buckets, short
// chains, as many blocks as it has slots -- is OutOfMemory, and a resize
// followed by the same decode recovers the frame exactly.
int out_of_memory_case(Gpu& gpu, codec::Decoder& dec) {
  // Sixteen blocks, every voxel observed, from a roomy source grid.
  vkc::Result<vol::VoxelBlockGrid> gs = make_grid(gpu);
  CHECK(gs.ok());
  vol::VoxelBlockGrid src = std::move(gs).value();
  std::vector<vr::Vec3i> coords;
  for (int i = 0; i < 16; ++i) {
    coords.push_back(vr::Vec3i(i % 4, i / 4, 0));
  }
  CHECK(allocate(src, coords).ok());
  CHECK(write_sphere(gpu, src, Sphere{vr::Vec3f(0.08f, 0.08f, 0.02f), 0.05f})
            .ok());
  vkc::Result<Snapshot> src_snap = snapshot(gpu, src);
  CHECK(src_snap.ok());
  vkc::Result<Bytes> frame = encode_with(gpu, src);
  CHECK(frame.ok());
  vkc::Result<codec::FrameInfo> info =
      codec::read_frame_info(frame.value().data(), frame.value().size());
  CHECK(info.ok());
  const std::uint32_t n = info.value().block_count;
  CHECK(n >= 8);

  // Exactly n slots -- so the heap check passes -- in two-slot buckets with
  // one-link chains, which n blocks hashed at random cannot all fit.
  GridShape cramped;
  cramped.bucket_size = 2;
  cramped.num_buckets = std::int32_t((n + 1) / 2);
  cramped.max_chain = 1;
  vkc::Result<vol::VoxelBlockGrid> gc = make_grid(gpu, cramped);
  CHECK(gc.ok());
  vol::VoxelBlockGrid tight = std::move(gc).value();
  const vkc::Status s =
      dec.decode(frame.value().data(), frame.value().size(), tight);
  CHECK(!s.ok());
  CHECK(s.domain() == vkc::Status::Code::OutOfMemory);

  // The documented recovery.
  CHECK(tight.resize(cramped.num_buckets * 8).ok());
  CHECK(dec.decode(frame.value().data(), frame.value().size(), tight).ok());
  vkc::Result<Snapshot> out = snapshot(gpu, tight);
  CHECK(out.ok());
  CHECK(out.value().coords.size() == n);
  return 0;
}

// A frame carries its kept weights itself: a reused decoder must not retain
// the previous frame's quantizer, even when its block coordinates are equal.
// @p config is @p dec's.
int quantization_sequence_case(Gpu& gpu, codec::Decoder& dec,
                               const codec::DecoderConfig& config = {}) {
  vkc::Result<vol::VoxelBlockGrid> source =
      sphere_grid(gpu, Sphere{vr::Vec3f(0.0f), 0.05f});
  CHECK(source.ok());
  codec::EncoderConfig a;
  a.params.coefficient_count = 35;
  a.params.quantization_scale = 0.03f;
  a.params.quantization_weights[1] = 4.0f;
  a.params.quantization_weights[8] = 2.0f;
  a.params.quantization_weights[64] = 0.5f;
  codec::EncoderConfig b = a;
  b.params.quantization_scale = 0.07f;
  std::swap(b.params.quantization_weights[1],
            b.params.quantization_weights[64]);
  vkc::Result<Bytes> fa = encode_with(gpu, source.value(), a);
  vkc::Result<Bytes> fb = encode_with(gpu, source.value(), b);
  CHECK(fa.ok() && fb.ok() && fa.value() != fb.value());
  vkc::Result<vol::VoxelBlockGrid> out = grid_for(gpu, fa.value());
  CHECK(out.ok());
  Snapshot first;
  for (const Bytes* frame : {&fa.value(), &fb.value(), &fa.value()}) {
    CHECK(dec.decode(frame->data(), frame->size(), out.value()).ok());
    vkc::Result<Snapshot> current = snapshot(gpu, out.value());
    CHECK(current.ok());
    if (first.coords.empty()) first = current.value();
    if (frame == &fa.value()) CHECK(first == current.value());
    if (frame == &fb.value()) CHECK(!(first == current.value()));

    // Its result agrees with a new decoder that has never seen another table.
    vkc::Result<codec::Decoder> fresh =
        codec::Decoder::create(gpu.device, gpu.allocator, config);
    vkc::Result<vol::VoxelBlockGrid> reference = grid_for(gpu, *frame);
    CHECK(fresh.ok() && reference.ok());
    CHECK(fresh.value()
              .decode(frame->data(), frame->size(), reference.value())
              .ok());
    vkc::Result<Snapshot> expected = snapshot(gpu, reference.value());
    CHECK(expected.ok() && expected.value() == current.value());
  }
  return 0;
}

int frame_info_case(Gpu& gpu) {
  const Sphere s{vr::Vec3f(0.0f), 0.05f};
  vkc::Result<vol::VoxelBlockGrid> g = sphere_grid(gpu, s);
  CHECK(g.ok());
  codec::EncoderConfig config;
  config.params.coefficient_count = 20;
  config.params.quantization_scale = 0.07f;
  for (std::size_t i = 0; i < config.params.quantization_weights.size(); ++i) {
    config.params.quantization_weights[i] = 0.5f + 0.125f * float(i % 31);
  }
  vkc::Result<Bytes> frame = encode_with(gpu, g.value(), config);
  CHECK(frame.ok());
  vkc::Result<codec::FrameInfo> info =
      codec::read_frame_info(frame.value().data(), frame.value().size());
  CHECK(info.ok());
  CHECK(info.value().voxel_size == kVoxel);
  CHECK(info.value().trunc_dist == kTrunc);
  CHECK(info.value().params.coefficient_count == 20);
  CHECK(info.value().params.quantization_scale ==
        config.params.quantization_scale);
  // The kept bases' weights travel (DC always is); one beyond K reads as 1.
  CHECK(info.value().params.quantization_weights[0] ==
        config.params.quantization_weights[0]);
  CHECK(info.value().params.quantization_weights[511] == 1.0f);
  CHECK(info.value().block_count > 0);
  // The header alone is enough; less than it is not.
  constexpr std::size_t header_bytes = 44 + 20 * sizeof(float);
  CHECK(codec::read_frame_info(frame.value().data(), header_bytes).ok());
  CHECK(!codec::read_frame_info(frame.value().data(), header_bytes - 1).ok());
  CHECK(!codec::read_frame_info(nullptr, 0).ok());
  Bytes bad = frame.value();
  bad[0] = 'X';
  CHECK(!codec::read_frame_info(bad.data(), bad.size()).ok());
  bad = frame.value();
  bad[4] = 2;  // version 2
  vkc::Result<codec::FrameInfo> v2 = codec::read_frame_info(bad.data(), 44);
  CHECK(!v2.ok() && v2.status().domain() == vkc::Status::Code::Unsupported);
  return 0;
}

int refusals_case(Gpu& gpu, codec::Decoder& dec) {
  // A frame with blocks in it, so a refusal that came after the grid started
  // to change would show.
  vkc::Result<vol::VoxelBlockGrid> g =
      sphere_grid(gpu, Sphere{vr::Vec3f(0.0f), 0.05f});
  CHECK(g.ok());
  vkc::Result<Bytes> frame = encode_with(gpu, g.value());
  CHECK(frame.ok());
  GridShape four;
  four.block_size = 4;
  vkc::Result<vol::VoxelBlockGrid> g4 = make_grid(gpu, four);
  CHECK(g4.ok());
  CHECK(
      !dec.decode(frame.value().data(), frame.value().size(), g4.value()).ok());
  GridShape no_weight;
  no_weight.weight = false;
  vkc::Result<vol::VoxelBlockGrid> gw = make_grid(gpu, no_weight);
  CHECK(gw.ok());
  CHECK(
      !dec.decode(frame.value().data(), frame.value().size(), gw.value()).ok());
  // Refused before a block was allocated.
  vkc::Result<std::vector<vol::BlockIndex>> none =
      gw.value().map().compact_active_blocks();
  CHECK(none.ok() && none.value().empty());
  // A third attribute, which a kept block would carry over stale.
  GridShape coloured;
  coloured.color = true;
  vkc::Result<vol::VoxelBlockGrid> gc = make_grid(gpu, coloured);
  CHECK(gc.ok());
  const vkc::Status sc =
      dec.decode(frame.value().data(), frame.value().size(), gc.value());
  CHECK(!sc.ok() && sc.domain() == vkc::Status::Code::InvalidArgument);
  none = gc.value().map().compact_active_blocks();
  CHECK(none.ok() && none.value().empty());
  vol::VoxelBlockGrid moved = std::move(g).value();
  vol::VoxelBlockGrid taken = std::move(moved);
  CHECK(!dec.decode(frame.value().data(), frame.value().size(), moved)
             .ok());  // NOLINT(bugprone-use-after-move)
  CHECK(dec.decode(frame.value().data(), frame.value().size(), taken).ok());
  return 0;
}

int metrics_case(Gpu& gpu, codec::Decoder& dec) {
  const Sphere s{vr::Vec3f(0.0f), 0.08f};
  vkc::Result<vol::VoxelBlockGrid> g = sphere_grid(gpu, s);
  CHECK(g.ok());
  vkc::Result<Bytes> frame = encode_with(gpu, g.value());
  CHECK(frame.ok());
  vkc::Result<vol::VoxelBlockGrid> d = grid_for(gpu, frame.value());
  CHECK(d.ok());
  vkc::StageMetrics m;
  CHECK(dec.decode(frame.value().data(), frame.value().size(), d.value(), &m)
            .ok());
  const vkc::StageRow* top = find_row(m, "codec decode");
  CHECK(top != nullptr);
  for (const char* sub :
       {"  ..rans decode", "  ..active set", "  ..apply", "  ..inverse"}) {
    const vkc::StageRow* row = find_row(m, sub);
    CHECK(row != nullptr);
    CHECK(row->cpu_ms <= top->cpu_ms);
  }
  CHECK(m.total_cpu_ms() == top->cpu_ms);
  // Encoding into the same metrics adds its own rows beside these rather
  // than summing into them (the one shared row is the map's compaction).
  vkc::StageMetrics both = m;
  vkc::Result<codec::Encoder> enc =
      codec::Encoder::create(gpu.device, gpu.allocator);
  CHECK(enc.ok());
  CHECK(enc.value().encode(g.value(), &both).ok());
  for (const char* sub : {"  ..rans decode", "  ..apply", "  ..inverse"}) {
    CHECK(find_row(both, sub)->cpu_ms == find_row(m, sub)->cpu_ms);
  }
  CHECK(find_row(both, "  ..forward") != nullptr);
  CHECK(find_row(both, "  ..rans encode") != nullptr);
  vkc::Result<vkc::GpuTimer> probe = vkc::GpuTimer::create(gpu.device);
  CHECK(probe.ok());
  if (probe.value().available()) {
    CHECK(top->has_gpu);
    CHECK(top->gpu_ms > 0.0 && top->gpu_ms < top->cpu_ms);
  }
  return 0;
}

int moves_case(Gpu& gpu, const codec::DecoderConfig& config = {}) {
  vkc::Result<codec::Decoder> a_r =
      codec::Decoder::create(gpu.device, gpu.allocator, config);
  CHECK(a_r.ok());
  codec::Decoder a = std::move(a_r).value();
  CHECK(a.valid());
  codec::Decoder b(std::move(a));
  CHECK(b.valid());
  CHECK(!a.valid());  // NOLINT(bugprone-use-after-move): asserting the source
  vkc::Result<codec::Decoder> c_r =
      codec::Decoder::create(gpu.device, gpu.allocator, config);
  CHECK(c_r.ok());
  codec::Decoder c = std::move(c_r).value();
  c = std::move(b);  // over a live decoder
  CHECK(c.valid());
  CHECK(!b.valid());  // NOLINT(bugprone-use-after-move)
  codec::Decoder* alias = &c;
  c = std::move(*alias);  // self-move, laundered past -Wself-move
  CHECK(c.valid());

  vkc::Result<vol::VoxelBlockGrid> g = make_grid(gpu);
  CHECK(g.ok());
  vkc::Result<Bytes> frame = encode_with(gpu, g.value());
  CHECK(frame.ok());
  CHECK(c.decode(frame.value().data(), frame.value().size(), g.value()).ok());
  CHECK(!a.decode(frame.value().data(), frame.value().size(), g.value()).ok());
  return 0;
}

}  // namespace

// Device decoding leaves a grid exactly as host decoding does, for the
// default segments and finer ones, which the automatic choice decodes on the
// device, past kMinDeviceDecodeSegments.
int device_matches_host_case(Gpu& gpu) {
  const Sphere s{vr::Vec3f(0.01f, -0.02f, 0.03f), 0.09f};
  vkc::Result<vol::VoxelBlockGrid> g = sphere_grid(gpu, s);
  CHECK(g.ok());
  for (std::uint32_t segment_size : {64u, 3u}) {
    codec::EncoderConfig ec;
    ec.params.coefficient_count = 37;
    ec.segment_size = segment_size;
    vkc::Result<Bytes> frame = encode_with(gpu, g.value(), ec);
    CHECK(frame.ok());
    vkc::Result<codec::FrameInfo> info =
        codec::read_frame_info(frame.value().data(), frame.value().size());
    CHECK(info.ok());
    if (segment_size == 3) {
      CHECK((info.value().block_count + 2) / 3 >=
            codec::kMinDeviceDecodeSegments);
    }
    std::vector<Snapshot> decoded;
    for (codec::EntropyCoding entropy :
         {codec::EntropyCoding::kHost, codec::EntropyCoding::kDevice,
          codec::EntropyCoding::kAuto}) {
      codec::DecoderConfig config;
      config.entropy = entropy;
      vkc::Result<codec::Decoder> dec =
          codec::Decoder::create(gpu.device, gpu.allocator, config);
      CHECK(dec.ok());
      vkc::Result<vol::VoxelBlockGrid> player = grid_for(gpu, frame.value());
      CHECK(player.ok());
      CHECK(dec.value()
                .decode(frame.value().data(), frame.value().size(),
                        player.value())
                .ok());
      vkc::Result<Snapshot> snap = snapshot(gpu, player.value());
      CHECK(snap.ok());
      CHECK(!snap.value().coords.empty());
      decoded.push_back(std::move(snap).value());
    }
    CHECK(decoded[0] == decoded[1]);
    CHECK(decoded[0] == decoded[2]);
  }
  return 0;
}

int gpu_main(vr_test::GpuContext& gpu) {
  vkc::Result<codec::Decoder> d =
      codec::Decoder::create(gpu.device, gpu.allocator);
  CHECK(d.ok());
  codec::Decoder dec = std::move(d).value();

  if (round_trip_case(gpu, dec) != 0) return 1;
  if (mesh_case(gpu, dec) != 0) return 1;
  if (sequence_case(gpu, dec) != 0) return 1;
  if (untouched_case(gpu, dec) != 0) return 1;
  if (out_of_memory_case(gpu, dec) != 0) return 1;
  if (quantization_sequence_case(gpu, dec) != 0) return 1;
  if (frame_info_case(gpu) != 0) return 1;
  if (refusals_case(gpu, dec) != 0) return 1;
  if (metrics_case(gpu, dec) != 0) return 1;
  if (device_matches_host_case(gpu) != 0) return 1;
  // The same contract with the decoding on the device.
  codec::DecoderConfig device_config;
  device_config.entropy = codec::EntropyCoding::kDevice;
  vkc::Result<codec::Decoder> dd =
      codec::Decoder::create(gpu.device, gpu.allocator, device_config);
  CHECK(dd.ok());
  codec::Decoder device_dec = std::move(dd).value();
  if (round_trip_case(gpu, device_dec) != 0) return 1;
  if (mesh_case(gpu, device_dec) != 0) return 1;
  if (sequence_case(gpu, device_dec) != 0) return 1;
  if (untouched_case(gpu, device_dec) != 0) return 1;
  if (out_of_memory_case(gpu, device_dec) != 0) return 1;
  if (quantization_sequence_case(gpu, device_dec, device_config) != 0) {
    return 1;
  }
  if (refusals_case(gpu, device_dec) != 0) return 1;
  if (metrics_case(gpu, device_dec) != 0) return 1;
  if (moves_case(gpu) != 0) return 1;
  // Decoders that hold a live device reader.
  if (moves_case(gpu, device_config) != 0) return 1;
  std::printf("codec Decoder: OK\n");
  return 0;
}

int main() { return vr_test::run_on_gpu(gpu_main); }
