// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file stream/audio.hpp
/// @brief An Audio track's params (@ref TrackKind::Audio): how its samples
///        are coded, how many channels a frame holds and how many frames a
///        second.
///
/// The params are 8 bytes, little-endian; a reader ignores any after them,
/// room for a later codec's fields:
///
/// @code
///   offset  size  field
///        0     2  codec (AudioCodec)
///        2     2  channels (1..64)
///        4     4  sample_rate: frames a second (>= 1)
/// @endcode

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>

#include "volumetric_kit/core/base/result.hpp"
#include "volumetric_kit/grid/stream/export.hpp"

namespace volumetric_kit::grid::stream {

/// @brief How an Audio track's samples are coded.
enum class AudioCodec : std::uint16_t {
  /// Signed 16-bit samples, little-endian, interleaved by channel: a
  /// record's payload is whole frames of `channels` samples.
  PcmS16 = 1,
};

/// @brief The most channels an Audio track's frame holds.
inline constexpr std::uint16_t kMaxAudioChannels = 64;

/// @brief The length of @ref encode_audio_params's bytes.
inline constexpr std::size_t kAudioParamsBytes = 8;

/// @brief What an Audio track's params say.
struct AudioParams {
  AudioCodec codec = AudioCodec::PcmS16;  ///< How its samples are coded.
  std::uint16_t channels = 1;             ///< Samples a frame: 1 to 64.
  std::uint32_t sample_rate = 48000;      ///< Frames a second; at least 1.
};

/// @brief @p params as an Audio track's params (the layout above).
/// @return @ref kAudioParamsBytes bytes, @p params as given: what
///         @ref decode_audio_params refuses is written all the same, and
///         refused when read.
VGRID_STREAM_API std::string encode_audio_params(
    const AudioParams& params) noexcept;

/// @brief Read an Audio track's params.
/// @param bytes  The track's @ref Track::params; bytes past the first
///               @ref kAudioParamsBytes are ignored.
/// @return The params; `Status::Code::InvalidArgument` for fewer than
///         @ref kAudioParamsBytes bytes, 0 or more than
///         @ref kMaxAudioChannels channels, or a sample rate of 0;
///         `Status::Code::Unsupported` for a codec this version does not
///         know, whatever the other fields hold, so a reader skips the
///         track; `Status::Code::OutOfMemory` when a diagnostic cannot be
///         allocated.
VGRID_STREAM_API core::Result<AudioParams> decode_audio_params(
    std::string_view bytes);

}  // namespace volumetric_kit::grid::stream
