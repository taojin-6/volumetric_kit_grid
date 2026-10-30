// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include "volumetric_kit/grid/codec_stream/codec_stream.hpp"

#include <cstring>
#include <filesystem>
#include <string>
#include <system_error>
#include <utility>

#include "stream_coder.hpp"
#include "volumetric_kit/core/base/log.hpp"
#include "volumetric_kit/grid/base/log.hpp"
#include "volumetric_kit/grid/codec/decode_grid.hpp"
#include "volumetric_kit/grid/stream/audio.hpp"

namespace volumetric_kit::grid::codec_stream {

namespace detail {
namespace {

// Whether `rows` holds a row named `name`.
bool has_row(const core::StageMetrics& rows, const char* name) {
  for (const core::StageRow& row : rows.rows()) {
    if (std::strcmp(row.name, name) == 0) return true;
  }
  return false;
}

// Close a stream that holds no record and remove its file at `path`, so a
// rerun is refused for its real cause, not for this empty stream.
void take_back(std::unique_ptr<stream::StreamWriter>& writer,
               const std::string& path) {
  if (!writer) return;
  writer.reset();
  std::error_code ignored;
  std::filesystem::remove(path, ignored);
}

// A frame `set` at `pts_ns` that does not follow the last one coded.
core::Status out_of_order(std::uint64_t set, std::uint64_t pts_ns,
                          std::uint64_t last_set, std::uint64_t last_pts_ns) {
  return core::Status::invalid_argument(
      "CodecStream: frame " + std::to_string(set) + " at " +
      std::to_string(pts_ns) + " ns does not follow frame " +
      std::to_string(last_set) + " at " + std::to_string(last_pts_ns) + " ns");
}

// OK for a session of a Geometry track 0 and Audio tracks after it, each
// with params a reader plays and on track 0's clock, as the mux compares
// their pts; InvalidArgument naming the first track that is not so.
core::Status check_session(const stream::Session& session) {
  if (session.tracks.empty()) {
    return core::Status::invalid_argument(
        "CodecStream: the session has no track; track 0 must be Geometry");
  }
  const stream::Track& frames = session.tracks[0];
  if (frames.kind != stream::TrackKind::Geometry) {
    return core::Status::invalid_argument(
        "CodecStream: track 0 is of kind " +
        std::to_string(static_cast<unsigned>(frames.kind)) +
        "; it must be Geometry");
  }
  for (std::size_t i = 1; i < session.tracks.size(); ++i) {
    const stream::Track& track = session.tracks[i];
    const std::string name = "CodecStream: track " + std::to_string(i);
    if (track.kind != stream::TrackKind::Audio) {
      return core::Status::invalid_argument(
          name + " is of kind " +
          std::to_string(static_cast<unsigned>(track.kind)) +
          "; a track after the Geometry track 0 must be Audio");
    }
    const core::Result<stream::AudioParams> params =
        stream::decode_audio_params(track.params);
    if (!params.ok()) {
      return core::Status::invalid_argument(name + ": " +
                                            params.status().message());
    }
    if (track.clock != frames.clock) {
      return core::Status::invalid_argument(
          name + " is on clock " + std::to_string(track.clock) +
          ", track 0 on clock " + std::to_string(frames.clock) +
          ": their pts are compared");
    }
  }
  return {};
}

}  // namespace

core::Result<std::unique_ptr<StreamCoder>> StreamCoder::create(
    core::Device& device, core::Allocator& allocator,
    const codec::EncoderConfig& config, const stream::Session& session,
    const CodecOutputs& outputs) {
  VKC_TRY(check_session(session));
  std::unique_ptr<networking::StreamServer> server;
  // A session with an Audio track is refused here, as Unsupported: the
  // server serves one track. TODO: serve it once StreamServer serves more.
  if (outputs.serve) {
    VKC_ASSIGN(server,
               networking::StreamServer::start(session, *outputs.serve));
  }
  std::unique_ptr<stream::StreamWriter> writer;
  if (!outputs.stream.empty()) {
    VKC_ASSIGN(writer, stream::StreamWriter::create(outputs.stream, session));
  }
  // A codec error takes back the file made above.
  core::Result<codec::Encoder> enc =
      codec::Encoder::create(device, allocator, config);
  if (!enc.ok()) {
    take_back(writer, outputs.stream);
    return enc.status();
  }
  std::optional<codec::Decoder> dec;
  if (outputs.decode) {
    core::Result<codec::Decoder> made =
        codec::Decoder::create(device, allocator);
    if (!made.ok()) {
      take_back(writer, outputs.stream);
      return made.status();
    }
    dec.emplace(std::move(made).value());
  }
  return std::unique_ptr<StreamCoder>(new StreamCoder(
      device, allocator, std::move(enc).value(), std::move(dec), outputs.stream,
      std::move(server), std::move(writer), session.tracks.size()));
}

StreamCoder::StreamCoder(core::Device& device, core::Allocator& allocator,
                         codec::Encoder enc, std::optional<codec::Decoder> dec,
                         std::string path,
                         std::unique_ptr<networking::StreamServer> server,
                         std::unique_ptr<stream::StreamWriter> writer,
                         std::size_t tracks)
    : device_(&device),
      allocator_(&allocator),
      encoder_(std::move(enc)),
      decoder_(std::move(dec)),
      path_(std::move(path)),
      server_(std::move(server)),
      writer_(std::move(writer)),
      mux_(tracks) {}

StreamCoder::~StreamCoder() {
  if (!finished_ && coded_ == 0) take_back(writer_, path_);
}

core::Result<SentFrame> StreamCoder::code(volume::VoxelBlockGrid& grid,
                                          std::uint64_t set,
                                          std::uint64_t pts_ns) {
  // Refused here, not by the writer once the encoder holds the frame.
  if (finished_) {
    return core::Status::invalid_argument("CodecStream: finished");
  }
  if (coded_ > 0 && (set <= last_set_ || pts_ns < last_pts_ns_)) {
    return out_of_order(set, pts_ns, last_set_, last_pts_ns_);
  }
  key_on_request();
  core::StageMetrics encoded;
  VKC_ASSIGN(const std::vector<std::uint8_t> bytes,
             encoder_.encode(grid, &encoded));
  const bool tail = has_row(encoded, "  ..tail");
  core::Result<SentFrame> played =
      play(stream::RecordPayload::of(bytes.data(), bytes.size()), set, pts_ns,
           std::move(encoded), tail,
           static_cast<std::uint32_t>(grid.grid().num_blocks), nullptr, {});
  if (!played.ok()) encoder_.force_intra();
  return played;
}

core::Result<codec::PendingEncode> StreamCoder::record_front(
    core::CommandBatch& front, volume::VoxelBlockGrid& grid) {
  if (finished_) {
    return core::Status::invalid_argument("CodecStream: finished");
  }
  key_on_request();
  front_rows_ = core::StageMetrics{};
  max_blocks_ = static_cast<std::uint32_t>(grid.grid().num_blocks);
  return encoder_.record_front(front, grid, &front_rows_);
}

void StreamCoder::front_ran(const std::vector<core::StageRow>& device_rows,
                            core::StageMetrics* rest) {
  std::vector<core::StageRow> encode;
  std::vector<core::StageRow> others;
  for (const core::StageRow& row : device_rows) {
    (has_row(front_rows_, row.name) ? encode : others).push_back(row);
  }
  front_rows_.merge_device(encode);
  if (rest != nullptr) rest->merge_device(others);
}

core::Result<codec::PendingEncode> StreamCoder::redo(
    const codec::PendingEncode& pending, volume::VoxelBlockGrid& grid,
    core::StageMetrics* rows) {
  encoder_.abandon(pending);
  const core::StageMetrics first = std::move(front_rows_);
  core::CommandBatch front(*device_, *allocator_);
  VKC_ASSIGN(const codec::PendingEncode again, record_front(front, grid));
  const core::Status ran = [&]() -> core::Status {
    VKC_ASSIGN(core::PendingBatch done, front.submit_async({}, {}));
    {
      const core::WaitScope wait(rows, "wait: redo");
      VKC_TRY(done.wait());
    }
    front_ran(done.device_rows(), nullptr);
    return {};
  }();
  if (!ran.ok()) {
    encoder_.abandon(again);
    return ran;
  }
  core::StageMetrics both = first;
  both.merge(front_rows_);
  front_rows_ = std::move(both);
  return again;
}

core::Result<SentFrame> StreamCoder::finish_frame(
    const codec::PendingEncode& pending, std::uint64_t set,
    std::uint64_t pts_ns, core::StageMetrics* rows,
    const std::function<void()>& sent) {
  if (finished_) {
    encoder_.abandon(pending);
    return core::Status::invalid_argument("CodecStream: finished");
  }
  if (coded_ > 0 && (set <= last_set_ || pts_ns < last_pts_ns_)) {
    encoder_.abandon(pending);
    return out_of_order(set, pts_ns, last_set_, last_pts_ns_);
  }
  core::StageMetrics back_rows;
  core::CommandBatch back(*device_, *allocator_);
  const core::Status coded = [&]() -> core::Status {
    VKC_TRY(encoder_.record_back(back, pending, &back_rows));
    VKC_ASSIGN(core::PendingBatch done, back.submit_async({}, {}));
    {
      // The output: waited on at once.
      const core::WaitScope wait(rows, "wait: frame");
      VKC_TRY(done.wait());
    }
    back_rows.merge_device(done.device_rows());
    return {};
  }();
  if (!coded.ok()) {
    encoder_.abandon(pending);
    return coded;
  }
  VKC_ASSIGN(const codec::EncodedFrame encoded,
             encoder_.finish(pending, &back_rows));
  core::StageMetrics frame_rows = front_rows_;
  frame_rows.merge(back_rows);
  if (rows != nullptr) rows->merge(frame_rows);
  const bool tail = has_row(back_rows, "  ..tail");
  const stream::RecordPayload payload = [&] {
    // The record's one CRC, for the file and the viewers alike.
    const core::StageScope crc(rows, "crc");
    return stream::RecordPayload::of(encoded.head.data(), encoded.head.size(),
                                     encoded.payload, encoded.payload_size);
  }();
  core::Result<SentFrame> played =
      play(payload, set, pts_ns, std::move(frame_rows), tail, max_blocks_, rows,
           sent);
  if (!played.ok()) encoder_.force_intra();
  return played;
}

core::Status StreamCoder::append(const stream::RecordInfo& info,
                                 const std::uint8_t* data, std::size_t size) {
  if (finished_) {
    return core::Status::invalid_argument("CodecStream: finished");
  }
  return mux_.append(info, data, size, sink());
}

core::Status StreamCoder::finish() {
  if (finished_) {
    return core::Status::invalid_argument("CodecStream: finished");
  }
  finished_ = true;
  const core::Status flushed = mux_.flush(sink());
  server_.reset();
  const core::Status footer = writer_ ? writer_->finish() : core::Status{};
  return flushed.ok() ? footer : flushed;
}

core::Status StreamCoder::send(const stream::RecordInfo& info,
                               const std::uint8_t* data, std::size_t size) {
  if (writer_) VKC_TRY(writer_->append(info, data, size));
  if (server_) {
    const core::Status served = server_->publish(info, data, size);
    if (!served.ok()) {
      log_message(core::LogLevel::Warning,
                  "CodecStream: track " + std::to_string(info.track) + " seq " +
                      std::to_string(info.seq) +
                      " not served: " + served.message());
    }
  }
  return {};
}

void StreamCoder::key_on_request() noexcept {
  if (!server_) return;
  const std::uint64_t wanted = server_->key_wanted();
  if (wanted == key_wanted_) return;
  key_wanted_ = wanted;
  encoder_.force_intra();
}

core::Result<SentFrame> StreamCoder::play(
    const stream::RecordPayload& bytes, std::uint64_t set, std::uint64_t pts_ns,
    core::StageMetrics encoded, bool tail, std::uint32_t max_blocks,
    core::StageMetrics* rows, const std::function<void()>& sent) {
  VKC_ASSIGN(const codec::FrameInfo info,
             codec::read_frame_info(bytes.first(), bytes.first_size()));
  stream::RecordInfo record;
  // A reader starts, or seeks, at a key: a frame that decodes alone.
  record.key = info.type == codec::FrameType::kIntra;
  record.pts_ns = pts_ns;
  record.seq = coded_;
  record.set = set;
  if (writer_) {
    const core::StageScope write(rows, "write");
    VKC_TRY(writer_->append(record, bytes));
  }
  // After the file, so no viewer gets a frame the file lacks. Returning a
  // refusal would leave coded_ behind the file, whose writer would then
  // refuse the next seq.
  if (server_) {
    const core::StageScope publish(rows, "publish");
    const core::Status served = server_->publish(record, bytes);
    if (!served.ok()) {
      log_message(core::LogLevel::Warning,
                  "CodecStream: frame " + std::to_string(set) +
                      " not served: " + served.message());
    }
  }
  ++coded_;
  last_set_ = set;
  last_pts_ns_ = pts_ns;
  if (sent) sent();
  VKC_TRY(mux_.frame(pts_ns, sink()));
  SentFrame frame;
  frame.type = info.type;
  frame.bytes = bytes.size();
  frame.blocks = info.block_count;
  frame.tail = tail;
  frame.encode = std::move(encoded);
  if (decoder_) {
    // The frame's one copy on the way to the decoder: into its input buffer.
    VKC_ASSIGN(const core::HostSlot input, decoder_->claim_input(bytes.size()));
    if (bytes.first_size() != 0) {
      std::memcpy(input.data(), bytes.first(), bytes.first_size());
    }
    if (bytes.second_size() != 0) {
      std::memcpy(input.data() + bytes.first_size(), bytes.second(),
                  bytes.second_size());
    }
    if (!player_) {
      VKC_ASSIGN(volume::VoxelBlockGrid grid,
                 codec::make_decode_grid(*device_, *allocator_, input.data(),
                                         bytes.size(), max_blocks));
      player_.emplace(std::move(grid));
    }
    VKC_TRY(codec::decode_growing(*decoder_, input, bytes.size(), *player_,
                                  max_blocks, &frame.decode));
  }
  return frame;
}

}  // namespace detail

core::Result<CodecStream> CodecStream::create(
    core::Device& device, core::Allocator& allocator,
    const codec::EncoderConfig& config, const stream::Session& session,
    CodecOutputs outputs) {
  VKC_ASSIGN(
      std::unique_ptr<detail::StreamCoder> coder,
      detail::StreamCoder::create(device, allocator, config, session, outputs));
  return CodecStream(std::move(coder));
}

CodecStream::CodecStream(std::unique_ptr<detail::StreamCoder> coder) noexcept
    : coder_(std::move(coder)) {}

CodecStream::CodecStream(CodecStream&& other) noexcept = default;

CodecStream::~CodecStream() = default;

core::Result<SentFrame> CodecStream::code(volume::VoxelBlockGrid& grid,
                                          std::uint64_t set,
                                          std::uint64_t pts_ns) {
  if (!coder_) {
    return core::Status::invalid_argument("CodecStream: moved from");
  }
  return coder_->code(grid, set, pts_ns);
}

core::Result<codec::PendingEncode> CodecStream::record_front(
    core::CommandBatch& front, volume::VoxelBlockGrid& grid) {
  if (!coder_) {
    return core::Status::invalid_argument("CodecStream: moved from");
  }
  return coder_->record_front(front, grid);
}

void CodecStream::front_ran(const std::vector<core::StageRow>& device_rows,
                            core::StageMetrics* rest) {
  if (coder_) coder_->front_ran(device_rows, rest);
}

core::Result<codec::PendingEncode> CodecStream::redo(
    const codec::PendingEncode& pending, volume::VoxelBlockGrid& grid,
    core::StageMetrics* rows) {
  if (!coder_) {
    return core::Status::invalid_argument("CodecStream: moved from");
  }
  return coder_->redo(pending, grid, rows);
}

core::Result<SentFrame> CodecStream::finish_frame(
    const codec::PendingEncode& pending, std::uint64_t set,
    std::uint64_t pts_ns, core::StageMetrics* rows,
    const std::function<void()>& sent) {
  if (!coder_) {
    return core::Status::invalid_argument("CodecStream: moved from");
  }
  return coder_->finish_frame(pending, set, pts_ns, rows, sent);
}

core::Status CodecStream::finish() {
  if (!coder_) {
    return core::Status::invalid_argument("CodecStream: moved from");
  }
  return coder_->finish();
}

volume::VoxelBlockGrid* CodecStream::player() noexcept {
  return coder_ ? coder_->player() : nullptr;
}

const networking::StreamServer* CodecStream::server() const noexcept {
  return coder_ ? coder_->server() : nullptr;
}

std::uint64_t CodecStream::last_pts_ns() const noexcept {
  return coder_ ? coder_->last_pts_ns() : 0;
}

std::uint64_t CodecStream::coded() const noexcept {
  return coder_ ? coder_->coded() : 0;
}

core::Status CodecStream::append(const stream::RecordInfo& info,
                                 const std::uint8_t* data, std::size_t size) {
  if (!coder_) {
    return core::Status::invalid_argument("CodecStream: moved from");
  }
  return coder_->append(info, data, size);
}

AppendedRecords CodecStream::appended(std::uint16_t track) const noexcept {
  return coder_ ? coder_->appended(track) : AppendedRecords{};
}

}  // namespace volumetric_kit::grid::codec_stream
