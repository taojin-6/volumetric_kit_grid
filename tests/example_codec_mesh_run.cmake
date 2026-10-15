# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Tao Jin

# codec_mesh on a device, in the order it works: the stream is made after the
# GPU setup and before any frame is converted or written, so a rerun onto an
# existing stream is refused with nothing converted, no PLY or CSV written and
# the file left as it was. Then a two-frame run codes frames 1 and 2. Skips with
# no Vulkan device unless VKC_REQUIRE_VULKAN_DEVICE is set.

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
  "{\"version\":1,\"operation\":\"codec\",\"source\":{\"kind\":\"mesh\",\"path\":\"tet_%02d.obj\",\"metres_per_unit\":1,\"up_axis\":\"y\"},\"run\":{\"first\":1,\"frames\":2},\"fusion\":{\"voxel_size_m\":0.02},\"output\":{\"prefix\":\"out/run\",\"stream\":\"out/run.vrst\",\"csv\":\"out/run.csv\"}}\n"
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
file(STRINGS "${WORK_DIR}/out/run.csv" rows)
list(LENGTH rows count)
list(GET rows 1 first)
list(GET rows 2 second)
if(NOT count EQUAL 3
   OR NOT first MATCHES "^1,"
   OR NOT second MATCHES "^2,")
  message(FATAL_ERROR "Expected frames 1 and 2 in the CSV: ${rows}")
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
file(REMOVE_RECURSE "${WORK_DIR}")
