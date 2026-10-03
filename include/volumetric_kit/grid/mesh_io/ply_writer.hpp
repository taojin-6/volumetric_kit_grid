// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file io/ply_writer.hpp
/// @brief Export a host mesh as binary little-endian PLY.

#include <string>

#include "volumetric_kit/recon/core/result.hpp"
#include "volumetric_kit/recon/io/export.hpp"
#include "volumetric_kit/recon/mesh/mesh.hpp"

namespace volumetric_kit::recon::io {

/// @brief Serialize a mesh to a binary little-endian PLY file.
///
/// Writes float `x y z nx ny nz`, then uchar `red green blue`, per vertex.
/// Linear BT.709/D65 RGB is clamped to [0,1] (NaN writes 0) and encoded with
/// the exact sRGB curve. Faces contain a uchar count (3) and three signed
/// 32-bit indices. Alpha, tangent, and UV attributes are omitted. Empty meshes
/// are supported.
///
/// Positions and normals must be finite; indices must form complete
/// triangles, reference existing vertices, and fit int32. Invalid input is
/// rejected before opening the output. A write failure can leave a partial
/// file.
/// @param path Output path without embedded NUL characters; replaced on
/// success.
/// @param mesh Mesh to serialize; geometry and winding are preserved.
/// @return OK after closing the file, or non-OK Status for invalid input,
///         failure to allocate the export buffers, an unsupported host byte
///         order, or write failure.
VR_IO_API Status write_ply(const std::string& path, const mesh::Mesh& mesh);

}  // namespace volumetric_kit::recon::io
