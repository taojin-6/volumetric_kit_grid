// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include "volumetric_kit/grid/stream/audio.hpp"

#include <string>

#include "chunk.hpp"
#include "volumetric_kit/core/base/exception_status.hpp"

namespace volumetric_kit::grid::stream {
namespace {

constexpr const char* kDecode = "decode_audio_params";

core::Status invalid(const std::string& why) {
  return core::Status::invalid_argument(std::string(kDecode) + ": " + why);
}

}  // namespace

// Eight bytes fit every standard library's small-string buffer, so the
// string allocates nothing and cannot throw.
std::string encode_audio_params(const AudioParams& params) noexcept {
  std::uint8_t bytes[kAudioParamsBytes];
  detail::store_le16(bytes, static_cast<std::uint16_t>(params.codec));
  detail::store_le16(bytes + 2, params.channels);
  detail::store_le32(bytes + 4, params.sample_rate);
  return std::string(reinterpret_cast<const char*>(bytes), sizeof bytes);
}

core::Result<AudioParams> decode_audio_params(std::string_view bytes) try {
  if (bytes.size() < kAudioParamsBytes) {
    return invalid(std::to_string(bytes.size()) + " bytes, under " +
                   std::to_string(kAudioParamsBytes));
  }
  const auto* p = reinterpret_cast<const std::uint8_t*>(bytes.data());
  AudioParams params;
  params.codec = static_cast<AudioCodec>(detail::load_le16(p));
  params.channels = detail::load_le16(p + 2);
  params.sample_rate = detail::load_le32(p + 4);
  // The codec first: a later codec may define its other fields anew.
  if (params.codec != AudioCodec::PcmS16) {
    return core::Status::unsupported(
        std::string(kDecode) + ": codec " +
        std::to_string(static_cast<unsigned>(params.codec)));
  }
  if (params.channels == 0 || params.channels > kMaxAudioChannels) {
    return invalid(std::to_string(params.channels) + " channels, not 1 to " +
                   std::to_string(kMaxAudioChannels));
  }
  if (params.sample_rate == 0) return invalid("a sample rate of 0");
  return params;
} catch (...) {
  return core::exception_status(kDecode, core::Status::Code::InvalidArgument);
}

}  // namespace volumetric_kit::grid::stream
