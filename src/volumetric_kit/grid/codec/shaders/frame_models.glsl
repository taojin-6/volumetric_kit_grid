// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// The v3 frame's symbol models -- one frequency table each, in this order in
// TABLES -- with each model's alphabet and its first entry in the per-symbol
// arrays (the device's counts and tables), and the symbols a block's mask is
// coded with. The one definition: the kernels include it as GLSL, and
// bitstream.cpp as C++ (see frame_grammar.glsl), so it is written in what the
// two languages share.

#ifndef VR_FRAME_MODELS_GLSL
#define VR_FRAME_MODELS_GLSL

// The coordinate models are split by what the sort already guarantees about
// the delta: after a block in the same (y, z) row the x step is at least 1
// (kDxRun codes step - 1, so a run is a stream of zeros), within the same z
// slice the y step is non-negative (kDySame), and after a step in z the other
// two are free to go either way.
//
// A partial mask is coded a z plane at a time, and a plane a line (one byte,
// eight voxels along x) at a time, each against the one before it: a fused
// band's mask edge is a smooth surface, so most planes and lines repeat their
// neighbour or are all one value. On room0 that is 2.2 bytes a block where
// the 64 raw bytes through one table were 8.3 (the 2026-10-01 entry).
const uint kDz = 0u;         // unsigned: z step
const uint kDySame = 1u;     // unsigned: y step, z unchanged
const uint kDyFree = 2u;     // signed:   y step after a z step
const uint kDxRun = 3u;      // unsigned: x step - 1, y and z unchanged
const uint kDxFree = 4u;     // signed:   x step after a y or z step
const uint kMaskClass = 5u;  // 0 all observed, 1 none observed, 2 partial
const uint kPlane = 6u;      // a partial mask's plane: same, empty, full, lines
const uint kLine = 7u;       // then its lines, three models by predictor
const uint kByte = kLine + 3u;       // then a line's byte, by the same three
const uint kFirstCoef = kByte + 3u;  // then one model per coefficient j < K

// A class is a bit length: 0 for zero, else 1 + floor(log2 |v|). Coordinate
// deltas reach 2^32 - 1 (INT32_MIN to INT32_MAX), so 33 classes; quantized
// coefficients are within +-32767 < 2^15, so 16.
const uint kCoordClasses = 33u;
const uint kMaskClasses = 3u;
const uint kRunClasses = 4u;  // a plane's or a line's symbols, below
const uint kByteSymbols = 256u;
const uint kCoefClasses = 16u;

// A mask class.
const uint kMaskFull = 0u;
const uint kMaskEmpty = 1u;
const uint kMaskPartial = 2u;
// A plane or a line: the same as its predictor, all unobserved, all observed,
// or neither -- a plane then coded line by line, a line by its byte. The
// writer tries them in this order, so a value two of them name is written as
// the first.
const uint kSame = 0u;
const uint kAllEmpty = 1u;
const uint kAllFull = 2u;
const uint kOther = 3u;

// A block's observed mask: 512 voxels, 32 a word, line l = y + 8 z (eight
// voxels along x) its byte l % 4 of word l / 4.
const uint kMaskWords = 16u;
const uint kLinesPerPlane = 8u;  // y
const uint kPlanes = 8u;         // z

uint model_alphabet(uint model) {
  if (model < kMaskClass) return kCoordClasses;
  if (model < kPlane) return kMaskClasses;
  if (model < kByte) return kRunClasses;
  if (model < kFirstCoef) return kByteSymbols;
  return kCoefClasses;
}

// The first entry of each run of models with one alphabet.
const uint kMaskClassBase = kCoordClasses * kMaskClass;
const uint kPlaneBase = kMaskClassBase + kMaskClasses;
const uint kByteBase = kPlaneBase + kRunClasses * (kByte - kPlane);
const uint kFirstCoefBase = kByteBase + kByteSymbols * (kFirstCoef - kByte);

// The sum of every earlier model's alphabet; for one past the last model, the
// whole array's size.
uint model_base(uint model) {
  if (model < kMaskClass) return kCoordClasses * model;
  if (model < kPlane) return kMaskClassBase;
  if (model < kByte) return kPlaneBase + kRunClasses * (model - kPlane);
  if (model < kFirstCoef) return kByteBase + kByteSymbols * (model - kByte);
  return kFirstCoefBase + kCoefClasses * (model - kFirstCoef);
}

#endif  // VR_FRAME_MODELS_GLSL
