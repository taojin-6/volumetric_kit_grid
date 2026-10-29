// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file frame_hash.hpp
/// @brief The hash a P-frame names its reference frame by (@ref
///        FrameReference).
///
/// Internal (under src/, never installed). XXH64 with seed 0, written from
/// the published algorithm (no dependency), over a frame's exact bytes: what
/// `Encoder::encode` returned and what `Decoder::decode` was given. It reads
/// bytes with shifts, so it needs no unaligned load and gives the same value on
/// any host.

#include <cstddef>
#include <cstdint>

#include "volumetric_kit/grid/codec/export.hpp"

namespace volumetric_kit::grid::codec::detail {

/// @return XXH64, seed 0, of the @p size bytes at @p data (which may be null
///         when @p size is 0).
VGRID_CODEC_API std::uint64_t frame_hash(const std::uint8_t* data,
                                         std::size_t size);

/// @return @ref frame_hash of @p head then @p payload back to back, without
///         joining them: the encoder's frame, whose payload stays in its
///         slot. Either may be null at size 0.
VGRID_CODEC_API std::uint64_t frame_hash(const std::uint8_t* head,
                                         std::size_t head_size,
                                         const std::uint8_t* payload,
                                         std::size_t payload_size);

}  // namespace volumetric_kit::grid::codec::detail
