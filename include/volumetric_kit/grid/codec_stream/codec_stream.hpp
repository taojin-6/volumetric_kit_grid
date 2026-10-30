// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file codec_stream/codec_stream.hpp
/// @brief A codec stream: each frame encoded, written to a `.vrst` file,
///        served to live viewers, and decoded back as a viewer would; the
///        records of its Audio tracks muxed in by pts.

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "volumetric_kit/core/base/result.hpp"
#include "volumetric_kit/core/base/stage_metrics.hpp"
#include "volumetric_kit/core/vulkan/fwd.hpp"
#include "volumetric_kit/grid/codec/codec_params.hpp"
#include "volumetric_kit/grid/codec/encoder.hpp"
#include "volumetric_kit/grid/codec_stream/export.hpp"
#include "volumetric_kit/grid/networking/stream_server.hpp"
#include "volumetric_kit/grid/stream/session.hpp"
#include "volumetric_kit/grid/volume/voxel_block_grid.hpp"

namespace volumetric_kit::grid::codec_stream {

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

/// @brief How far past the last frame's pts @ref CodecStream::append holds
///        a record, and how far below the newest it holds one before the
///        first frame: 1 s.
inline constexpr std::uint64_t kHoldLimitNs = 1'000'000'000;

/// @brief What became of the records @ref CodecStream::append took on one
///        track.
struct AppendedRecords {
  std::uint64_t held = 0;  ///< Held now, for a frame at or past their pts.
  /// Dropped before the first frame: below its pts, or, before it, more
  /// than @ref kHoldLimitNs below the newest held.
  std::uint64_t dropped = 0;
  /// Gone out: written and served as @ref CodecOutputs says (counted with
  /// neither too).
  std::uint64_t written = 0;
};

/// @brief Codes grids as frames (@ref codec::Encoder): each written to a
///        `.vrst` file and served live as @ref CodecOutputs says, then
///        decoded back into a player grid, as a viewer decodes it.
///
/// The file and the viewers get one session: track 0 a Geometry track, each
/// frame a record of it, and any Audio tracks after it, whose records
/// @ref append muxes in by pts. The player grid is built from the first frame's
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
/// An Audio track's records (track 1 here) as its source hands them over,
/// between frames, each a key on the frames' clock:
/// @code
/// stream::RecordInfo info;
/// info.track = 1;
/// info.key = true;
/// info.pts_ns = chunk.pts_ns;
/// info.seq = chunk.sequence;
/// VKC_TRY(stream.append(info, chunk.bytes.data(), chunk.bytes.size()));
/// @endcode
///
/// @warning The `Device` and `Allocator` passed to @ref create must outlive
///          the stream. Not thread-safe: one call at a time.
class VGRID_CODEC_STREAM_API CodecStream {
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
  ///                   track 0 a Geometry track, whose records the frames
  ///                   are, then any Audio tracks (@ref append), each with
  ///                   params `stream::decode_audio_params` takes and on
  ///                   track 0's clock, as @ref append compares their pts.
  ///                   With no id or start time, a rerun writes the same
  ///                   bytes.
  /// @param outputs    Where the frames go.
  /// @return The stream; `Status::Code::InvalidArgument`, naming the track,
  ///         for a session that is not so; what
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
  /// Closes the viewers at once, unlike @ref finish, and the file. A file
  /// never finished that holds no record, as after a run that failed before
  /// its first frame, is removed, so a rerun onto it is not refused; one that
  /// holds records stays, read as a killed writer's.
  ~CodecStream();

  /// @brief Encode @p grid as one frame, write it, serve it, then decode it
  ///        back.
  ///
  /// Written as soon as it is encoded, so a frame the decoder refuses is
  /// kept; served once written, so no viewer gets a frame the file lacks;
  /// then the records @ref append held for it go out, before the decode. A
  /// frame the server refuses is logged at Warning and fails nothing: the
  /// viewers lose the rest of its GOP, the file nothing.
  /// @param grid    The grid to code.
  /// @param set     Its record's set: above every frame's coded before.
  /// @param pts_ns  Its record's pts: no earlier than the frames' coded
  ///                before.
  /// @return The frame; `Status::Code::InvalidArgument`, before anything is
  ///         encoded, for a @p set or @p pts_ns out of order or a finished or
  ///         moved-from stream; the encode's error; what
  ///         @ref stream::StreamWriter::append returns for the frame; or,
  ///         the frame written, served and counted by @ref coded, the
  ///         writer's error on a record the frame releases (@ref append), or
  ///         the decode's error.
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
  ///         finished or moved-from stream; the encoder's refusal, with
  ///         nothing recorded; or, after the first commands, a buffer
  ///         failure, so the caller drops @p front
  ///         (@ref codec::Encoder::record_front).
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
  ///                 before the records it releases and the decode, so it
  ///                 has run whenever the frame was counted.
  /// @return As @ref code; a @p set or @p pts_ns out of order, or a
  ///         finished stream, refused with @p pending dropped;
  ///         `Status::Code::InvalidArgument` on a moved-from stream.
  core::Result<SentFrame> finish_frame(const codec::PendingEncode& pending,
                                       std::uint64_t set, std::uint64_t pts_ns,
                                       core::StageMetrics* rows = nullptr,
                                       const std::function<void()>& sent = {});

  /// @brief Write and serve one record of an Audio track, muxed with the
  ///        frames by pts: it goes out right after the first frame whose
  ///        pts is at or past its own, so a reader in file order meets it
  ///        no earlier than the frame it plays with.
  ///
  /// - Held until such a frame is written and served; the records a frame
  ///   releases follow it in pts order across tracks (the lower track first
  ///   at equal pts), each track's in its order.
  /// - A record at or below the last frame's pts goes out at once.
  /// - Before the first frame every record is held; the first frame drops
  ///   those below its pts, and a record below it later is dropped at once
  ///   (@ref AppendedRecords::dropped). Before the first frame, a record
  ///   more than @ref kHoldLimitNs below the newest held is dropped too.
  /// - At most @ref kHoldLimitNs of pts past the last frame is held: when
  ///   a record would be held further past it (the geometry stalls),
  ///   everything held goes out, in pts order, and so does each record
  ///   after it until a frame catches up. The bound is in pts, so it bounds
  ///   memory only while a track's pts advance with its samples, as an
  ///   audio sensor's do.
  /// - @ref finish writes what is held, or drops it before any frame.
  ///
  /// Held records reach the file later, so the record rules are checked
  /// here, against the track's last record taken (dropped ones too). A
  /// record held is copied; @p data need not outlive the call.
  /// @param info  Its track (an Audio track of the session), its key (set:
  ///              every Audio record is one), its pts on the track's clock,
  ///              no lower than the track's last, its seq above the track's
  ///              last, and its set as written.
  /// @param data  The payload; may be null when @p size is 0.
  /// @param size  The payload's bytes.
  /// @return OK, the record held, dropped or written;
  ///         `Status::Code::InvalidArgument`, with nothing changed, for
  ///         track 0, a track the session lacks, a record that is not a
  ///         key, a seq or pts that goes back on its track, a null @p data
  ///         with a nonzero @p size, or a finished or moved-from stream;
  ///         `Status::Code::OutOfMemory`, with nothing changed, when it
  ///         cannot be held; or what @ref stream::StreamWriter::append
  ///         returns for a record it writes (a record the server refuses is
  ///         logged at Warning, as a frame is).
  core::Status append(const stream::RecordInfo& info, const std::uint8_t* data,
                      std::size_t size);

  /// @brief Finish the file with its footer index
  ///        (@ref stream::StreamWriter::finish), then the viewers' stream
  ///        (@ref networking::StreamServer::finish), each viewer getting
  ///        every record served before its end unless still behind after
  ///        @ref networking::kFinishLimit; without either, only mark the
  ///        stream finished. One attempt: however it ends, every later call,
  ///        this one included, is refused, so a file whose footer failed
  ///        stays unfinished, read as a killed writer's. First, the records
  ///        @ref append holds are written and served, in pts order, or
  ///        dropped before any frame.
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

  /// @return What became of the records @ref append took on @p track; all
  ///         zero for track 0, a track the session lacks, and on a
  ///         moved-from stream.
  AppendedRecords appended(std::uint16_t track) const noexcept;

  /// @return Whether this owns its coder; false when moved-from.
  bool valid() const noexcept { return coder_ != nullptr; }

 private:
  explicit CodecStream(std::unique_ptr<detail::StreamCoder> coder) noexcept;

  std::unique_ptr<detail::StreamCoder> coder_;
};

}  // namespace volumetric_kit::grid::codec_stream
