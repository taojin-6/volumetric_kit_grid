// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// The single translation unit that compiles tinyply's implementation. Kept
// apart from ply_writer.cpp (which uses only the declarations) so this
// third-party code can be built with warnings disabled -- it does not survive
// the tier's -Werror surface. CMake attaches `-w` to this file alone.

#define TINYPLY_IMPLEMENTATION
#include "tinyply_backend.hpp"
