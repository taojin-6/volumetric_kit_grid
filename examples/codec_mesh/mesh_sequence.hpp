// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file examples/codec_mesh/mesh_sequence.hpp
/// @brief codec_mesh's frames: the meshes `source.path` numbers, placed in
///        metres by one transform fitted to the whole range, each converted
///        into a cleared grid and coded as one record (@ref code_frame).

#include <cstdint>
#include <string>
#include <vector>

#include "codec_stream.hpp"
#include "config.hpp"
#include "volumetric_kit/core/base/result.hpp"
#include "volumetric_kit/core/base/stage_metrics.hpp"
#include "volumetric_kit/recon/io/mesh_normalization.hpp"
#include "volumetric_kit/recon/io/triangle_mesh.hpp"
#include "volumetric_kit/recon/tsdf/mesh_integrator.hpp"
#include "volumetric_kit/recon/volume/voxel_block_grid.hpp"

namespace vr_example {

/// @brief Frame @p number's path in @p pattern.
///
/// The pattern is the part of printf that ffmpeg's image sequences read: at
/// most one `%d`, or `%0Nd` (1 <= N <= 9) zero-padded to at least N digits,
/// in any path component, and `%%` for a literal `%`. `%5d` is refused,
/// since printf pads it with spaces and ffmpeg with zeros.
/// @return The path; for a pattern with no number, the pattern with each
///         `%%` made `%`; InvalidArgument naming `source.path` for any other
///         `%` or a second number.
vkc::Result<std::string> numbered_path(const std::string& pattern,
                                       std::uint32_t number);

/// @brief One frame, loaded and placed in metres.
struct MeshFrame {
  std::uint32_t number = 0;       ///< Its number: its record's set.
  std::string path;               ///< Its file.
  vr::io::TriangleMesh mesh;      ///< Its geometry, in metres.
  vr::io::MeshTopology topology;  ///< The audit of those coordinates.
};

/// @brief The frames a run codes -- `run.frames` of them, numbered from
///        `run.first` -- and the one transform that places every one.
class MeshSequence {
 public:
  /// @brief Check the pattern and every frame's file, then make the first
  ///        pass and fit the transform.
  ///
  /// The first pass loads each frame, measures it along
  /// `source.up_vector`'s axes, audits it on its own coordinates and drops
  /// it, so memory does not grow with the range. The audit's counts do not
  /// change under the transform, so unless `codec.inspect_only`, a frame
  /// signed mode cannot use is refused here (@ref admit), before any GPU
  /// work. One transform is fitted to the union of the frames' bounds:
  /// `io::fit_transform` to `source.height_m`, or `io::scale_transform` by
  /// `source.metres_per_unit`.
  /// @return The sequence; before any load, InvalidArgument for a malformed
  ///         pattern, for `run.first` above 0 or `run.frames` above 1 with no
  ///         number in `source.path` (naming the key), or for a frame's file
  ///         that is not a regular file, and NotFound for a missing one; or,
  ///         naming the frame, a load, audit, @ref admit or fit error.
  static vkc::Result<MeshSequence> open(const config::Config& settings);

  std::uint32_t first() const noexcept { return first_; }  ///< First number.
  std::uint32_t count() const noexcept {                   ///< How many frames.
    return std::uint32_t(paths_.size());
  }
  std::uint32_t last() const noexcept {  ///< Last number.
    return first_ + count() - 1;
  }
  /// @return Frame @p number's file. @pre first() <= number <= last().
  const std::string& path(std::uint32_t number) const;
  /// @return The transform that places every frame.
  const vr::io::MeshTransform& transform() const noexcept { return transform_; }

  /// @brief Print the range, its bounds and the transform.
  void print() const;

  /// @brief The second pass's step: load frame @p number, apply the
  ///        transform, and audit the coordinates it gave, the audit that
  ///        decides (rounding can make a triangle degenerate).
  /// @return The frame, or the load, transform or audit error naming it.
  vkc::Result<MeshFrame> load(std::uint32_t number) const;

  /// @brief Signed mode's rule: the closest-face sign needs
  ///        `MeshTopology::supports_signed(source.allow_holes)`. Shell mode
  ///        takes any mesh.
  /// @return OK, or InvalidArgument naming the frame, its file and its
  ///         defects, and `source.allow_holes` when holes are its only one.
  vkc::Status admit(std::uint32_t number,
                    const vr::io::MeshTopology& topology) const;

 private:
  MeshSequence() = default;

  std::string pattern_;  // source.path
  std::uint32_t first_ = 0;
  std::vector<std::string> paths_;  // frame first_ + i's at i
  bool signed_ = true;
  bool allow_holes_ = false;
  bool by_height_ = true;       // fitted to source.height_m, else scaled
  vr::io::MeshBounds input_;    // the union in input coordinates
  vr::io::MeshBounds upright_;  // the union along the up axes
  vr::io::MeshTransform transform_;
};

/// @return @p topology's defects, each a count and what it counts, joined
///         by ", "; empty when signed mode can take it without holes.
std::string topology_defects(const vr::io::MeshTopology& topology);

/// @brief Print a frame's size and its full topology audit.
void print_frame(const MeshFrame& frame);

/// @brief `codec.inspect_only`'s second pass, on the host alone: every frame
///        loaded, placed and audited as a run that codes would, a line for
///        each with a defect and whether signed mode takes it, then a count.
/// @return The last frame, or the first error.
vkc::Result<MeshFrame> inspect_frames(const MeshSequence& sequence);

/// @brief Convert @p mesh into @p grid alone: clear the grid, so no block of
///        an earlier frame stays, allocate the mesh's band, growing the grid
///        for capacity and retrying lost bucket-lock races, then integrate.
/// @return The integrator's stats, or the first error.
vkc::Result<vr::tsdf::MeshIntegrateStats> convert_frame(
    vr::volume::VoxelBlockGrid& grid, vr::tsdf::MeshIntegrator& integrator,
    const vr::io::TriangleMesh& mesh, const vr::tsdf::MeshSdfParams& params,
    vkc::StageMetrics* metrics);

/// @brief What @ref code_frame did with one frame.
struct CodedFrame {
  MeshFrame frame;                     ///< The frame as converted.
  vr::tsdf::MeshIntegrateStats stats;  ///< Its conversion.
};

/// @brief One frame of a run: @ref MeshSequence::load, @ref
///        MeshSequence::admit, @ref convert_frame into @p grid, then one
///        record of @p stream, its set the frame's number and its pts that
///        number at @ref kNominalFps.
/// @return The frame, or the first error, naming the frame.
vkc::Result<CodedFrame> code_frame(const MeshSequence& sequence,
                                   std::uint32_t number,
                                   vr::volume::VoxelBlockGrid& grid,
                                   vr::tsdf::MeshIntegrator& integrator,
                                   const vr::tsdf::MeshSdfParams& params,
                                   CodecStream& stream,
                                   vkc::StageMetrics* conversion);

}  // namespace vr_example
