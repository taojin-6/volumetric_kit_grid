# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Tao Jin

# codec_mesh on a device, in the order it works: the stream is made after the
# GPU setup and before any frame is converted or written, so a rerun onto an
# existing stream is refused with nothing converted, no PLY or CSV written and
# the file left as it was. Then a two-frame run at a period of 2 codes frame 1
# as an I-frame and frame 2 as a P-frame, and a single mesh has no bitrate.
# Skips with no Vulkan device unless VKC_REQUIRE_VULKAN_DEVICE is set.

file(REMOVE_RECURSE "${WORK_DIR}")
file(MAKE_DIRECTORY "${WORK_DIR}/out")
# A small closed tetrahedron wound outward, then the same with its apex higher:
# triangles well inside the comparison's reach.
foreach(frame_apex "1;0.06" "2;0.08")
  list(GET frame_apex 0 frame)
  list(GET frame_apex 1 apex)
  file(
    WRITE "${WORK_DIR}/tet_0${frame}.obj"
    "v 0 0 0\nv 0.06 0 0\nv 0 0 0.06\nv 0.02 ${apex} 0.02\nf 1 2 3\nf 1 4 2\nf 1 3 4\nf 2 4 3\n"
  )
endforeach()
file(
  WRITE "${WORK_DIR}/run.json"
  "{\"version\":1,\"operation\":\"codec\",\"source\":{\"kind\":\"mesh\",\"path\":\"tet_%02d.obj\",\"metres_per_unit\":1,\"up_axis\":\"y\"},\"run\":{\"first\":1,\"frames\":2},\"fusion\":{\"voxel_size_m\":0.02},\"codec\":{\"intra_period\":2,\"update_size\":16},\"output\":{\"prefix\":\"out/run\",\"stream\":\"out/run.vrst\",\"csv\":\"out/run.csv\"}}\n"
)

file(WRITE "${WORK_DIR}/out/run.vrst" "earlier")
execute_process(
  COMMAND "${PROGRAM}" --config "${WORK_DIR}/run.json"
  RESULT_VARIABLE result
  OUTPUT_VARIABLE output
  ERROR_VARIABLE error)
if(NOT error MATCHES "run.vrst")
  if(error MATCHES "Vulkan|vkCreate|physical device|VK_ERROR"
     AND "$ENV{VKC_REQUIRE_VULKAN_DEVICE}" STREQUAL "")
    message(STATUS "skipping: no Vulkan device (${error})")
    return()
  endif()
  message(FATAL_ERROR "The existing stream was not refused: ${result}: "
                      "${output}${error}")
endif()
file(READ "${WORK_DIR}/out/run.vrst" kept)
file(GLOB written "${WORK_DIR}/out/run_*" "${WORK_DIR}/out/run.csv")
if(NOT result EQUAL 1
   OR output MATCHES "mesh -> TSDF"
   OR written
   OR NOT kept STREQUAL "earlier")
  message(FATAL_ERROR "The refusal came after work: ${result}: ${written}: "
                      "${output}${error}")
endif()

file(REMOVE "${WORK_DIR}/out/run.vrst")
execute_process(
  COMMAND "${PROGRAM}" --config "${WORK_DIR}/run.json"
  RESULT_VARIABLE result
  OUTPUT_VARIABLE output
  ERROR_VARIABLE error)
if(NOT result EQUAL 0 OR NOT output MATCHES "mesh: frames 1..2 of .*2 frame")
  message(FATAL_ERROR "The run failed: ${result}: ${output}${error}")
endif()
# The scenario's period and update size reach the encoder: frame 1 an I-frame,
# frame 2 a P-frame on it, and the report a GOP's bitrate.
if(NOT output MATCHES "update segments of 16 entries; an I-frame every 2 frames"
   OR NOT output MATCHES "GOP of 2: [0-9.]+ Mbit/s"
   OR NOT output MATCHES "1 I-frames: .*1 P-frames: ")
  message(FATAL_ERROR "The period did not reach the encoder: ${output}")
endif()
if(NOT output MATCHES "Mbit/s at 30.0 coded frames/s")
  message(FATAL_ERROR "A sequence has a bitrate at 30 fps: ${output}")
endif()
file(STRINGS "${WORK_DIR}/out/run.csv" rows)
list(LENGTH rows count)
list(GET rows 1 first)
list(GET rows 2 second)
if(NOT count EQUAL 3
   OR NOT first MATCHES "^1,I,"
   OR NOT second MATCHES "^2,P,")
  message(FATAL_ERROR "Expected I-frame 1 and P-frame 2 in the CSV: ${rows}")
endif()
# The Geometry track's params hold the transform: "VRMT", version 1.
file(READ "${WORK_DIR}/out/run.vrst" stream HEX)
if(NOT stream MATCHES "56524d5401000000")
  message(FATAL_ERROR "The stream does not hold the transform")
endif()
foreach(ply input source decoded)
  if(NOT EXISTS "${WORK_DIR}/out/run_${ply}.ply")
    message(FATAL_ERROR "No ${ply} PLY: ${output}")
  endif()
endforeach()
# A single mesh is no measured sequence: it has no bitrate.
file(
  WRITE "${WORK_DIR}/one.json"
  "{\"version\":1,\"operation\":\"codec\",\"source\":{\"kind\":\"mesh\",\"path\":\"tet_01.obj\",\"metres_per_unit\":1,\"up_axis\":\"y\"},\"fusion\":{\"voxel_size_m\":0.02}}\n"
)
execute_process(
  COMMAND "${PROGRAM}" --config "${WORK_DIR}/one.json"
  RESULT_VARIABLE result
  OUTPUT_VARIABLE output
  ERROR_VARIABLE error)
if(NOT result EQUAL 0
   OR NOT output MATCHES "stream: 1 frames coded"
   OR output MATCHES "Mbit/s")
  message(FATAL_ERROR "The single mesh: ${result}: ${output}${error}")
endif()
file(REMOVE_RECURSE "${WORK_DIR}")
