// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file codec_stream/codec_stream.hpp
/// @brief A codec stream: each frame encoded, written to a `.vrst` file,
///        served to live viewers, and decoded back as a viewer would.

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "volumetric_kit/core/base/result.hpp"
#include "volumetric_kit/core/base/stage_metrics.hpp"
#include "volumetric_kit/recon/codec/codec_params.hpp"
#include "volumetric_kit/recon/codec/encoder.hpp"
#include "volumetric_kit/recon/codec_stream/export.hpp"
#include "volumetric_kit/recon/core/fwd.hpp"
#include "volumetric_kit/recon/networking/stream_server.hpp"
#include "volumetric_kit/recon/stream/session.hpp"
#include "volumetric_kit/recon/volume/voxel_block_grid.hpp"

namespace volumetric_kit::recon::codec_stream {

namespace detail {
class StreamCoder;
}  // namespace detail

/// @brief Where a @ref CodecStream's frames go, each as it is coded.
struct CodecOutputs {
  /// A new `.vrst` file (stream/session.hpp) of every frame the encoder
  /// returns, a record each, an I-frame's a key; empty writes none. Nothing
  /// may be at the path, and its directory must exist.
  std::string stream;
  /// Where to serve the same records live, as they are recorded
  /// (@ref networking::StreamServer, on the stream's session); none serves
  /// nothing. A viewer's request for a key is met at the next frame, so it
  /// may serve without `replay_gop`. Defaulted, so one-field braces stay
  /// quiet under -Wmissing-field-initializers.
  std::optional<networking::ServerOptions> serve = std::nullopt;
  /// Decode each frame back into @ref CodecStream::player, as a viewer
  /// does, for @ref SentFrame::decode. Off, no decoder or player grid is
  /// made.
  bool decode = true;
};

/// @brief A frame as it went out, from @ref CodecStream::code or
///        @ref CodecStream::finish_frame.
struct SentFrame {
  /// An I-frame, or a P-frame on the frame coded before.
  codec::FrameType type = codec::FrameType::kIntra;
  std::size_t bytes = 0;     ///< Its record's payload, the compressed frame.
  std::uint32_t blocks = 0;  ///< The blocks it carries.
  /// Its payload outgrew what its batch copied and took a tail
  /// (`"  ..tail"`, @ref codec::Encoder::finish).
  bool tail = false;
  /// The encode's stage rows; a frame coded in halves, both halves'.
  core::StageMetrics encode;
  /// The decode's stage rows; empty without @ref CodecOutputs::decode.
  core::StageMetrics decode;
};

/// @brief Codes grids as frames (@ref codec::Encoder): each written to a
///        `.vrst` file and served live as @ref CodecOutputs says, then
///        decoded back into a player grid, as a viewer decodes it.
///
/// The file and the viewers get one session of one Geometry track, each
/// frame a record of it. The player grid is built from the first frame's
/// header and grows as later frames need (codec/decode_grid.hpp). The
/// encoder codes its next P-frame against every frame it returns, so any
/// failure after an encode makes the next frame an I-frame
/// (@ref codec::Encoder::force_intra): a frame the file or the player lacks
/// is never a reference. A viewer joining or falling back makes the next
/// frame an I-frame too (@ref networking::StreamServer::key_wanted), as it
/// waits for a key: the request stands until a frame is returned, and joins
/// between two frames make one key.
///
/// A frame is coded whole (@ref code), or in halves around a caller's
/// batch, as a sender codes a set's frame in the batch that fuses the set:
/// @ref record_front into the batch, @ref front_ran once it has resolved,
/// @ref redo if the grid changed after the front half, then
/// @ref finish_frame, which codes the back half in a batch of its own and
/// writes, serves and decodes the frame as @ref code does. The halves make
/// the frames @ref code makes of the same grids. One frame is in flight: a
/// later front half or @ref code supersedes it, and the encoder then refuses
/// its pending frame.
///
/// @code
/// VKC_ASSIGN(codec_stream::CodecStream stream,
///            codec_stream::CodecStream::create(device, allocator, config,
///                                              session, outputs));
/// for (std::uint64_t set = 0; set < sets; ++set) {
///   fuse(grid, set);
///   VKC_ASSIGN(const codec_stream::SentFrame sent,
///              stream.code(grid, set, pts_ns(set)));
/// }
/// VKC_TRY(stream.finish());
/// @endcode
///
/// In halves, after the caller's own work on the grid in its batch `front`:
/// @code
/// VKC_ASSIGN(const codec::PendingEncode pending,
///            stream.record_front(front, grid));
/// VKC_ASSIGN(core::PendingBatch done, front.submit_async({}, {}));
/// VKC_TRY(done.wait());
/// stream.front_ran(done.device_rows(), &rows);
/// VKC_ASSIGN(const codec_stream::SentFrame sent,
///            stream.finish_frame(pending, set, pts_ns(set), &rows));
/// @endcode
///
/// @warning The `Device` and `Allocator` passed to @ref create must outlive
///          the stream. Not thread-safe: one call at a time.
class VR_CODEC_STREAM_API CodecStream {
 public:
  /// @brief Make the outputs, then the encoder and, with
  ///        @ref CodecOutputs::decode, the decoder.
  ///
  /// The server first, so a port in use touches no file, then the file,
  /// whose rerun onto an existing one is the common refusal; the codec's
  /// kernels last, so a refused output costs none.
  /// @param device     The compute device (must outlive the stream).
  /// @param allocator  The allocator its buffers come from (must outlive the
  ///                   stream).
  /// @param config     The encoder's settings.
  /// @param session    The file's and the viewers' session, written as given:
  ///                   one Geometry track, whose records the frames are.
  ///                   With no id or start time, a rerun writes the same
  ///                   bytes.
  /// @param outputs    Where the frames go.
  /// @return The stream; `Status::Code::InvalidArgument` for a session that
  ///         is not one Geometry track; what
  ///         @ref networking::StreamServer::start returns
  ///         (`Status::Code::IoError` for a port in use); what
  ///         @ref stream::StreamWriter::create returns
  ///         (`Status::Code::InvalidArgument` when something is at the path,
  ///         `Status::Code::NotFound` when its directory is missing); or the
  ///         encoder's or the decoder's `create` error, the new file removed
  ///         again.
  static core::Result<CodecStream> create(core::Device& device,
                                          core::Allocator& allocator,
                                          const codec::EncoderConfig& config,
                                          const stream::Session& session,
                                          CodecOutputs outputs = {});

  CodecStream(CodecStream&& other) noexcept;
  CodecStream& operator=(CodecStream&&) = delete;
  /// Closes the viewers and the file. A file never finished that holds no
  /// record, as after a run that failed before its first frame, is removed,
  /// so a rerun onto it is not refused; one that holds records stays, read as
  /// a killed writer's.
  ~CodecStream();

  /// @brief Encode @p grid as one frame, write it, serve it, then decode it
  ///        back.
  ///
  /// Written as soon as it is encoded, so a frame the decoder refuses is
  /// kept; served once written, so no viewer gets a frame the file lacks. A
  /// frame the server refuses is logged at Warning and fails nothing: the
  /// viewers lose the rest of its GOP, the file nothing.
  /// @param grid    The grid to code.
  /// @param set     Its record's set: above every frame's coded before.
  /// @param pts_ns  Its record's pts: no earlier than the frames' coded
  ///                before.
  /// @return The frame; `Status::Code::InvalidArgument`, before anything is
  ///         encoded, for a @p set or @p pts_ns out of order or a finished or
  ///         moved-from stream; the encode's error; what
  ///         @ref stream::StreamWriter::append returns; or the decode's
  ///         error, the frame written and counted by @ref coded.
  core::Result<SentFrame> code(volume::VoxelBlockGrid& grid, std::uint64_t set,
                               std::uint64_t pts_ns);

  /// @brief Record @p grid's frame's front half into @p front, after what it
  ///        holds (@ref codec::Encoder::record_front), its rows the frame's
  ///        own.
  ///
  /// The front half decides the frame's type, so a viewer's request for a
  /// key is read here, as @ref code reads it; it stands until a frame is
  /// returned, so a front half recorded again (@ref redo) keeps it.
  /// @param front  Records the half; the caller submits it, and calls
  ///               @ref front_ran once it has resolved.
  /// @param grid   The grid to code; nothing but @p front changes it until
  ///               @ref finish_frame, or @ref redo records the half again.
  /// @return The pending frame; `Status::Code::InvalidArgument` for a
  ///         finished or moved-from stream; or the encoder's refusal, with
  ///         nothing recorded.
  /// @pre The batches of the stream's last frame have resolved.
  core::Result<codec::PendingEncode> record_front(core::CommandBatch& front,
                                                  volume::VoxelBlockGrid& grid);

  /// @brief Once the batch @ref record_front recorded into has resolved:
  ///        its device rows into the frame's rows where the front half
  ///        placed that row (@ref codec::Encoder::record_front places its
  ///        stage's and each breakdown's as it opens), the rest into
  ///        @p rest. Does nothing on a moved-from stream.
  /// @param device_rows  The batch's `PendingBatch::device_rows`.
  /// @param rest         Optional: takes the rows of the batch's other work.
  void front_ran(const std::vector<core::StageRow>& device_rows,
                 core::StageMetrics* rest = nullptr);

  /// @brief The front half again, in a batch of its own and waited on: what
  ///        a caller does when the grid changed after @p pending's front
  ///        half was recorded, such as a fusion done again once its batch
  ///        ran, so the front half coded the grid without it (rare).
  ///
  /// The frame's rows keep the first front half's, which the device ran too,
  /// and add this one's.
  /// @param pending  From @ref record_front, its batch resolved and
  ///                 @ref front_ran called; dropped.
  /// @param grid     The grid, as it now is.
  /// @param rows     Optional: takes the wait (`"wait: redo"`).
  /// @return The new pending frame; `Status::Code::InvalidArgument` for a
  ///         finished or moved-from stream; or the encoder's or the batch's
  ///         failure.
  core::Result<codec::PendingEncode> redo(const codec::PendingEncode& pending,
                                          volume::VoxelBlockGrid& grid,
                                          core::StageMetrics* rows = nullptr);

  /// @brief Once @p pending's front batch has resolved (@ref front_ran):
  ///        code its back half in a batch of its own and wait on it
  ///        (`"wait: frame"`), finish the frame, then write, serve and
  ///        decode it as @ref code does.
  ///
  /// The frame's head and payload are written and served as they lie, under
  /// one CRC (`"crc"`), and decoded from the decoder's input buffer they are
  /// copied into.
  /// @param pending  From @ref record_front or @ref redo.
  /// @param set      As @ref code.
  /// @param pts_ns   As @ref code.
  /// @param rows     Optional: the frame's rows, `"crc"`, `"write"`,
  ///                 `"publish"` and the wait.
  /// @param sent     Optional: runs once the frame is written and served,
  ///                 before the decode.
  /// @return As @ref code; a @p set or @p pts_ns out of order, or a
  ///         finished or moved-from stream, refused with @p pending
  ///         dropped.
  core::Result<SentFrame> finish_frame(const codec::PendingEncode& pending,
                                       std::uint64_t set, std::uint64_t pts_ns,
                                       core::StageMetrics* rows = nullptr,
                                       const std::function<void()>& sent = {});

  /// @brief Close the viewers, which see the stream end, and finish the file
  ///        with its footer index (@ref stream::StreamWriter::finish);
  ///        without either, only mark the stream finished. One attempt:
  ///        however it ends, every later call, this one included, is refused,
  ///        so a file whose footer failed stays unfinished, read as a killed
  ///        writer's.
  /// @return OK; `Status::Code::InvalidArgument` once finished or on a
  ///         moved-from stream; or the writer's error.
  core::Status finish();

  /// @return The grid the frames decode into, holding the last decoded; null
  ///         before the first frame, without @ref CodecOutputs::decode, and
  ///         on a moved-from stream.
  volume::VoxelBlockGrid* player() noexcept;

  /// @return The server the frames are served on; null when not serving,
  ///         once finished, and on a moved-from stream.
  const networking::StreamServer* server() const noexcept;

  /// @return The pts of the last frame written, below which @ref code and
  ///         @ref finish_frame refuse one; 0 before the first.
  std::uint64_t last_pts_ns() const noexcept;

  /// @return The frames encoded and written so far, one the decoder refused
  ///         among them; the destructor removes an unfinished stream's file
  ///         only while this is 0.
  std::uint64_t coded() const noexcept;

  /// @return Whether this owns its coder; false when moved-from.
  bool valid() const noexcept { return coder_ != nullptr; }

 private:
  explicit CodecStream(std::unique_ptr<detail::StreamCoder> coder) noexcept;

  std::unique_ptr<detail::StreamCoder> coder_;
};

}  // namespace volumetric_kit::recon::codec_stream
