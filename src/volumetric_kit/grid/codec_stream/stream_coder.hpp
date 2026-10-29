// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

// What a CodecStream holds and does, its calls each CodecStream's of the same
// name, the frame in halves included. Internal: CodecStream owns one and
// forwards to it, and only the tier's sources include this header.

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "volumetric_kit/core/base/result.hpp"
#include "volumetric_kit/core/base/stage_metrics.hpp"
#include "volumetric_kit/core/vulkan/command_batch.hpp"
#include "volumetric_kit/recon/codec/decoder.hpp"
#include "volumetric_kit/recon/codec/encoder.hpp"
#include "volumetric_kit/recon/codec_stream/codec_stream.hpp"
#include "volumetric_kit/recon/networking/stream_server.hpp"
#include "volumetric_kit/recon/stream/chunks.hpp"
#include "volumetric_kit/recon/stream/session.hpp"
#include "volumetric_kit/recon/stream/stream_writer.hpp"
#include "volumetric_kit/recon/volume/voxel_block_grid.hpp"

namespace volumetric_kit::recon::codec_stream::detail {

class StreamCoder {
 public:
  // As CodecStream::create.
  static core::Result<std::unique_ptr<StreamCoder>> create(
      core::Device& device, core::Allocator& allocator,
      const codec::EncoderConfig& config, const stream::Session& session,
      const CodecOutputs& outputs);

  StreamCoder(const StreamCoder&) = delete;
  StreamCoder& operator=(const StreamCoder&) = delete;
  StreamCoder(StreamCoder&&) = delete;
  StreamCoder& operator=(StreamCoder&&) = delete;
  // As ~CodecStream.
  ~StreamCoder();

  // As CodecStream::code.
  core::Result<SentFrame> code(volume::VoxelBlockGrid& grid, std::uint64_t set,
                               std::uint64_t pts_ns);

  // As CodecStream::record_front, ::front_ran, ::redo and ::finish_frame.
  core::Result<codec::PendingEncode> record_front(core::CommandBatch& front,
                                                  volume::VoxelBlockGrid& grid);
  void front_ran(const std::vector<core::StageRow>& device_rows,
                 core::StageMetrics* rest);
  core::Result<codec::PendingEncode> redo(const codec::PendingEncode& pending,
                                          volume::VoxelBlockGrid& grid,
                                          core::StageMetrics* rows);
  core::Result<SentFrame> finish_frame(const codec::PendingEncode& pending,
                                       std::uint64_t set, std::uint64_t pts_ns,
                                       core::StageMetrics* rows,
                                       const std::function<void()>& sent);

  // As CodecStream::finish, ::player, ::server, ::last_pts_ns and ::coded.
  core::Status finish();
  volume::VoxelBlockGrid* player() noexcept {
    return player_ ? &*player_ : nullptr;
  }
  const networking::StreamServer* server() const noexcept {
    return server_.get();
  }
  std::uint64_t last_pts_ns() const noexcept { return last_pts_ns_; }
  std::uint64_t coded() const noexcept { return coded_; }

 private:
  StreamCoder(core::Device& device, core::Allocator& allocator,
              codec::Encoder enc, std::optional<codec::Decoder> dec,
              std::string path,
              std::unique_ptr<networking::StreamServer> server,
              std::unique_ptr<stream::StreamWriter> writer);

  // Make the next frame an I-frame (Encoder::force_intra) when a viewer
  // joined or fell back since the last frame read the server's count: it
  // waits for a key. The request stands until a frame is returned, so a frame
  // refused or recorded again still makes it; joins between two frames make
  // one.
  void key_on_request() noexcept;

  // What code() and finish_frame() do once the encoder returns frame `set`'s
  // `bytes` -- its head, then its payload where the encoder left it, under
  // one CRC: write and serve them, run `sent`, and decode them into the
  // player if the stream decodes, from the decoder's input buffer they are
  // copied into. `encoded` is the encode's stage rows and `tail` whether its
  // payload took a tail. The source's block capacity, `max_blocks`, bounds
  // its frames' blocks: the player's ceiling. `rows`, when given, takes the
  // write's and the serve's rows.
  core::Result<SentFrame> play(const stream::RecordPayload& bytes,
                               std::uint64_t set, std::uint64_t pts_ns,
                               core::StageMetrics encoded, bool tail,
                               std::uint32_t max_blocks,
                               core::StageMetrics* rows,
                               const std::function<void()>& sent);

  core::Device* device_;
  core::Allocator* allocator_;
  codec::Encoder encoder_;
  std::optional<codec::Decoder> decoder_;  // with CodecOutputs::decode
  std::string path_;                       // the file's, or empty
  // Null when not serving, or once finished.
  std::unique_ptr<networking::StreamServer> server_;
  std::unique_ptr<stream::StreamWriter> writer_;  // null without a file
  // The server's key_wanted() when a frame last read it (key_on_request).
  std::uint64_t key_wanted_ = 0;
  std::uint64_t coded_ = 0;     // frames encoded and written: the next seq
  std::uint64_t last_set_ = 0;  // the last of them, once coded_ > 0
  std::uint64_t last_pts_ns_ = 0;
  bool finished_ = false;
  std::optional<volume::VoxelBlockGrid> player_;
  // The frame in flight between record() and finish_frame(): its front
  // half's rows, and its source's block capacity.
  core::StageMetrics front_rows_;
  std::uint32_t max_blocks_ = 0;
};

}  // namespace volumetric_kit::recon::codec_stream::detail
