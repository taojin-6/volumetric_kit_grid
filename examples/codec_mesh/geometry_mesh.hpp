// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

// codec_mesh's input as the host mesh the evaluator and the PLY writer take.

#include "volumetric_kit/grid/mesh/mesh.hpp"
#include "volumetric_kit/grid/mesh_io/triangle_mesh.hpp"

namespace vgrid_example {

namespace vgrid = volumetric_kit::grid;

/// Convert positions to the evaluator/exporter's host mesh container, with
/// area-weighted vertex normals.
vgrid::mesh::Mesh geometry_mesh(const vgrid::mesh_io::TriangleMesh& geometry);

}  // namespace vgrid_example
