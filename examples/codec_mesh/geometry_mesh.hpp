// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

// codec_mesh's input as the host mesh the evaluator and the PLY writer take.

#include "volumetric_kit/recon/io/triangle_mesh.hpp"
#include "volumetric_kit/recon/mesh/mesh.hpp"

namespace vr_example {

namespace vr = volumetric_kit::recon;

/// Convert positions to the evaluator/exporter's host mesh container, with
/// area-weighted vertex normals.
vr::mesh::Mesh geometry_mesh(const vr::io::TriangleMesh& geometry);

}  // namespace vr_example
