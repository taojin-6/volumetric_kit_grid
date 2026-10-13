// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// GPU test for P-frame decoding: the reference store's apply against
// independent uint16 arithmetic, at every source kind and K, as its slots
// grow, and only a commit changing what it holds. Internal headers by
// include path; skips with no device.

#include <cstdint>
#include <cstdio>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "bitstream.hpp"
#include "codec_fixture.hpp"
#include "codec_frames.hpp"
#include "reference_store.hpp"
#include "volumetric_kit/core/vulkan/command_batch.hpp"

#include "gpu_test.hpp"

namespace vkc = volumetric_kit::core;
namespace codec = volumetric_kit::recon::codec;
namespace d = volumetric_kit::recon::codec::detail;
using namespace codec_fixture;
using codec_frames::append_payload;
using codec_frames::Lcg;
using codec_frames::Payload;
using codec_frames::payload_at;
using C = vkc::Status::Code;

namespace {

// --- The store. -------------------------------------------------------------

// Two int16 a word, each entry starting a word of its own, an odd K's pad
// half @p pad: the layout the device reader writes and the apply reads.
std::vector<std::uint32_t> pack(const d::DctBlocks& b, std::uint16_t pad = 0) {
  const std::uint32_t k = b.params.coefficient_count;
  const std::uint32_t words = (k + 1) / 2;
  const std::size_t n = b.masks.size() / codec::kMaskWordsPerBlock;
  std::vector<std::uint32_t> out(n * words, 0);
  for (std::size_t i = 0; i < n; ++i) {
    for (std::uint32_t j = 0; j < words * 2; ++j) {
      const std::uint16_t v =
          j < k ? static_cast<std::uint16_t>(b.coefficients[i * k + j]) : pad;
      out[i * words + j / 2] |= std::uint32_t(v) << (16 * (j & 1));
    }
  }
  return out;
}

// @p n entries of K coefficients drawn from all of int16, and masks of every
// kind.
d::DctBlocks random_entries(std::size_t n, std::uint32_t k, Lcg& rng) {
  d::DctBlocks b;
  b.params.coefficient_count = k;
  b.trunc_dist = kTrunc;
  for (std::size_t i = 0; i < n; ++i) {
    for (std::uint32_t w = 0; w < codec::kMaskWordsPerBlock; ++w) {
      b.masks.push_back(i % 3 == 0 ? ~0u : rng.next());
    }
    for (std::uint32_t j = 0; j < k; ++j) {
      b.coefficients.push_back(static_cast<std::int16_t>(rng.next()));
    }
  }
  return b;
}

// Strictly increasing coordinates, @p n of them.
std::vector<vr::Vec3i> row_of(std::size_t n) {
  std::vector<vr::Vec3i> out;
  for (std::size_t i = 0; i < n; ++i) out.push_back(vr::Vec3i(int(i), 0, 0));
  return out;
}

// The spare as the last apply left it, unpacked: each block's payload, and
// whether every odd K's pad half was written 0.
struct Applied {
  std::vector<Payload> blocks;
  std::vector<vr::Vec3i> coords;
  bool pads_zero = true;
};

vkc::Result<Applied> read_spare(Gpu& gpu, const d::ResidentBlocks& spare) {
  const std::uint32_t k = spare.coefficient_count;
  const std::uint32_t words = (k + 1) / 2;
  Applied out;
  if (spare.count == 0) return out;
  VKC_ASSIGN(const std::vector<std::uint32_t> masks,
             vr_test::read_back<std::uint32_t>(
                 gpu.device, gpu.allocator, *spare.masks,
                 std::size_t(spare.count) * codec::kMaskWordsPerBlock));
  VKC_ASSIGN(const std::vector<std::uint32_t> packed,
             vr_test::read_back<std::uint32_t>(
                 gpu.device, gpu.allocator, *spare.coefficients,
                 std::size_t(spare.count) * words));
  VKC_ASSIGN(const std::vector<vol::BlockIndex> list,
             vr_test::read_back<vol::BlockIndex>(gpu.device, gpu.allocator,
                                                 *spare.list, spare.count));
  for (std::size_t i = 0; i < spare.count; ++i) {
    Payload p;
    p.mask.assign(masks.begin() + std::ptrdiff_t(i * 16),
                  masks.begin() + std::ptrdiff_t(i * 16 + 16));
    for (std::uint32_t j = 0; j < k; ++j) {
      p.q.push_back(static_cast<std::int16_t>(packed[i * words + j / 2] >>
                                              (16 * (j & 1))));
    }
    if (k % 2 != 0 && (packed[i * words + words - 1] >> 16) != 0) {
      out.pads_zero = false;
    }
    out.blocks.push_back(std::move(p));
    out.coords.push_back(list[i].coord);
    if (list[i].ptr != 0) out.pads_zero = false;
  }
  return out;
}

// What each source kind makes, in plain uint16 arithmetic: keep adds the
// residual to the reference mod 2^16 and XORs the mask, skip is the
// reference, and the payload alone is itself.
Payload expected(const Payload* reference, const Payload* payload) {
  if (reference == nullptr) return *payload;
  if (payload == nullptr) return *reference;
  Payload t = *reference;
  for (std::size_t j = 0; j < t.q.size(); ++j) {
    t.q[j] = static_cast<std::int16_t>(std::uint16_t(
        std::uint16_t(reference->q[j]) + std::uint16_t(payload->q[j])));
  }
  for (std::size_t w = 0; w < t.mask.size(); ++w) {
    t.mask[w] = reference->mask[w] ^ payload->mask[w];
  }
  return t;
}

// An intra frame's apply, committed: what the store holds after decoding it.
int commit_intra(Gpu& gpu, d::ReferenceStore& store, const d::DctBlocks& r,
                 std::uint64_t hash) {
  const std::size_t m = r.masks.size() / codec::kMaskWordsPerBlock;
  const std::uint32_t k = r.params.coefficient_count;
  std::vector<d::ApplySource> sources(m);
  for (std::size_t i = 0; i < m; ++i) {
    sources[i] = {d::kNoSource, static_cast<std::uint32_t>(i)};
  }
  CHECK(store.reserve(std::uint32_t(m), k, std::uint32_t(m)).ok());
  vkc::CommandBatch batch(gpu.device, gpu.allocator);
  vkc::Result<d::ResidentBlocks> upload =
      store.record_upload(batch, r, sources);
  CHECK(upload.ok());
  vkc::Result<d::ResidentBlocks> spare =
      store.record_apply(batch, upload.value(), sources, row_of(m));
  CHECK(spare.ok());
  CHECK(batch.submit().ok());
  vkc::Result<Applied> got = read_spare(gpu, spare.value());
  CHECK(got.ok() && got.value().pads_zero);
  for (std::size_t i = 0; i < m; ++i) {
    CHECK(got.value().blocks[i] == payload_at(r, i));
  }
  d::FrameReference id;
  id.hash = hash;
  id.header.block_count = static_cast<std::uint32_t>(m);
  id.header.params.coefficient_count = k;
  store.commit(id, row_of(m));
  return 0;
}

// Every source kind against a held reference, its payload uploaded from the
// host or bound where the device left it, at K of 1, 21, 64 and 512, with
// residuals from all of int16 so many wrap.
int apply_case(Gpu& gpu) {
  vkc::Result<std::unique_ptr<d::ReferenceStore>> made =
      d::ReferenceStore::create(gpu.device, gpu.allocator);
  CHECK(made.ok());
  d::ReferenceStore& store = *made.value();
  CHECK(store.held() == nullptr && store.coords().empty());
  Lcg rng{17};
  std::uint64_t hash = 1;
  for (std::uint32_t k : {1u, 21u, 64u, codec::kVoxelsPerBlock}) {
    const std::size_t m = 9;
    const d::DctBlocks r = random_entries(m, k, rng);
    CHECK(commit_intra(gpu, store, r, hash) == 0);
    CHECK(store.held() != nullptr && store.held()->hash == hash);
    CHECK(store.coords() == row_of(m));

    // A P-frame's payload by stream position: M entries, then two new.
    const d::DctBlocks payload = random_entries(m + 2, k, rng);
    // keep, skip, zero, keep, new, skip, zero, keep, new: the reference's
    // entries 0, 1, 2, 4, 5, 6, 8 survive, 3 and 7 are gone.
    const std::vector<d::ApplySource> sources = {{0, 0},
                                                 {1, d::kNoSource},
                                                 {d::kNoSource, 2},
                                                 {4, 4},
                                                 {d::kNoSource, 9},
                                                 {5, d::kNoSource},
                                                 {d::kNoSource, 6},
                                                 {8, 8},
                                                 {d::kNoSource, 10}};
    const std::vector<vr::Vec3i> coords = row_of(sources.size());
    for (const bool device : {false, true}) {
      std::vector<d::ApplySource> s = sources;
      CHECK(store
                .reserve(std::uint32_t(s.size()), k,
                         device ? 0 : std::uint32_t(s.size()))
                .ok());
      vkc::CommandBatch batch(gpu.device, gpu.allocator);
      // The device reader's layout, its pads not 0, which the apply drops.
      vkc::Result<vkc::Buffer> masks = vr_test::upload_device_buffer(
          gpu.device, gpu.allocator, payload.masks.data(),
          payload.masks.size() * sizeof(std::uint32_t));
      const std::vector<std::uint32_t> packed = pack(payload, 0xBEEF);
      vkc::Result<vkc::Buffer> coefficients = vr_test::upload_device_buffer(
          gpu.device, gpu.allocator, packed.data(),
          packed.size() * sizeof(std::uint32_t));
      CHECK(masks.ok() && coefficients.ok());
      d::ResidentBlocks resident;
      resident.masks = &masks.value();
      resident.coefficients = &coefficients.value();
      resident.count = std::uint32_t(m + 2);
      resident.coefficient_count = k;
      if (!device) {
        vkc::Result<d::ResidentBlocks> up =
            store.record_upload(batch, payload, s);
        CHECK(up.ok());
        resident = up.value();
        // Only the seven entries with a payload went up.
        CHECK(resident.count == 7);
      }
      vkc::Result<d::ResidentBlocks> spare =
          store.record_apply(batch, resident, s, coords);
      CHECK(spare.ok());
      CHECK(batch.submit().ok());
      vkc::Result<Applied> got = read_spare(gpu, spare.value());
      CHECK(got.ok());
      CHECK(got.value().pads_zero);
      CHECK(got.value().coords == coords);
      for (std::size_t i = 0; i < sources.size(); ++i) {
        const d::ApplySource& src = sources[i];
        const Payload ref = src.reference != d::kNoSource
                                ? payload_at(r, src.reference)
                                : Payload{};
        const Payload pay = src.payload != d::kNoSource
                                ? payload_at(payload, src.payload)
                                : Payload{};
        const Payload want =
            expected(src.reference != d::kNoSource ? &ref : nullptr,
                     src.payload != d::kNoSource ? &pay : nullptr);
        if (!(got.value().blocks[i] == want)) {
          std::fprintf(stderr, "K %u, block %zu (%s payload)\n", k, i,
                       device ? "device" : "host");
          return 1;
        }
      }
      // Applied and submitted, but not committed: the reference is as it
      // was.
      CHECK(store.held()->hash == hash && store.coords() == row_of(m));
    }
    ++hash;
  }
  return 0;
}

// The slots grow under a held reference, whose words a growing spare never
// disturbs; and only commit changes what is held, whatever was recorded or
// refused before it.
int grow_case(Gpu& gpu) {
  vkc::Result<std::unique_ptr<d::ReferenceStore>> made =
      d::ReferenceStore::create(gpu.device, gpu.allocator);
  CHECK(made.ok());
  d::ReferenceStore& store = *made.value();
  Lcg rng{23};
  const std::uint32_t k = 64;
  d::DctBlocks held = random_entries(3, k, rng);
  CHECK(commit_intra(gpu, store, held, 100) == 0);
  for (const std::size_t n : {std::size_t(50), std::size_t(700)}) {
    // Every block a skip of a held one, then the rest new: a frame of n
    // blocks, the spare grown to hold it.
    const std::size_t m = store.coords().size();
    const d::DctBlocks fresh = random_entries(n - m, k, rng);
    std::vector<d::ApplySource> sources;
    for (std::size_t e = 0; e < m; ++e) {
      sources.push_back({std::uint32_t(e), d::kNoSource});
    }
    for (std::size_t q = 0; q < n - m; ++q) {
      sources.push_back({d::kNoSource, std::uint32_t(q)});
    }
    CHECK(store.reserve(std::uint32_t(n), k, std::uint32_t(n - m)).ok());
    vkc::CommandBatch batch(gpu.device, gpu.allocator);
    vkc::Result<d::ResidentBlocks> up =
        store.record_upload(batch, fresh, sources);
    CHECK(up.ok());
    vkc::Result<d::ResidentBlocks> spare =
        store.record_apply(batch, up.value(), sources, row_of(n));
    CHECK(spare.ok());
    CHECK(batch.submit().ok());
    vkc::Result<Applied> got = read_spare(gpu, spare.value());
    CHECK(got.ok());
    d::DctBlocks next;
    next.params.coefficient_count = k;
    for (std::size_t i = 0; i < n; ++i) {
      const Payload want =
          i < m ? payload_at(held, i) : payload_at(fresh, i - m);
      CHECK(got.value().blocks[i] == want);
      append_payload(next, want);
    }
    // Refused, and recorded but never committed: nothing changes.
    const d::FrameReference before = *store.held();
    vkc::CommandBatch refused(gpu.device, gpu.allocator);
    CHECK(store.record_apply(refused, up.value(), sources, row_of(n - 1))
              .status()
              .domain() == C::InvalidArgument);
    CHECK(store.held()->hash == before.hash && store.coords() == row_of(m));
    d::FrameReference id;
    id.hash = before.hash + 1;
    id.header.block_count = std::uint32_t(n);
    id.header.params.coefficient_count = k;
    store.commit(id, row_of(n));
    CHECK(store.held()->hash == id.hash && store.coords() == row_of(n));
    held = std::move(next);
  }
  return 0;
}

// What the apply refuses before it records anything.
int apply_refusals_case(Gpu& gpu) {
  vkc::Result<std::unique_ptr<d::ReferenceStore>> made =
      d::ReferenceStore::create(gpu.device, gpu.allocator);
  CHECK(made.ok());
  d::ReferenceStore& store = *made.value();
  Lcg rng{29};
  CHECK(commit_intra(gpu, store, random_entries(4, 8, rng), 7) == 0);
  const d::DctBlocks payload = random_entries(4, 8, rng);
  CHECK(store.reserve(4, 8, 4).ok());
  // The payload uploaded as given, then applied at @p k over @p blocks.
  auto refuses = [&](std::vector<d::ApplySource> sources, std::uint32_t k,
                     std::size_t blocks, const char* says) {
    vkc::CommandBatch batch(gpu.device, gpu.allocator);
    vkc::Result<d::ResidentBlocks> r =
        store.record_upload(batch, payload, sources);
    if (r.ok()) {
      d::ResidentBlocks up = r.value();
      up.coefficient_count = k;
      r = store.record_apply(batch, up, sources, row_of(blocks));
    }
    return !r.ok() && r.status().domain() == C::InvalidArgument &&
           r.status().message().find(says) != std::string::npos;
  };
  const std::uint32_t none = d::kNoSource;
  CHECK(refuses({{0, none}, {4, none}}, 8, 2, "reads reference entry 4"));
  CHECK(refuses({{0, 0}}, 9, 1, "at K 9"));
  CHECK(refuses({{none, 4}}, 8, 1, "names entry 4 of 4"));
  CHECK(refuses(std::vector<d::ApplySource>(100, {0, none}), 8, 100,
                "more than were reserved"));
  CHECK(refuses({{0, none}}, 8, 2, "1 sources for 2 blocks"));
  CHECK(store.reserve(1u << 31, 8, 0).domain() == C::InvalidArgument);
  CHECK(store.held()->hash == 7 && store.coords() == row_of(4));
  return 0;
}

}  // namespace

int gpu_main(vr_test::GpuContext& gpu) {
  if (apply_case(gpu) != 0) return 1;
  if (grow_case(gpu) != 0) return 1;
  if (apply_refusals_case(gpu) != 0) return 1;
  std::printf("codec reference store: OK\n");
  return 0;
}

int main() { return vr_test::run_on_gpu(gpu_main); }
