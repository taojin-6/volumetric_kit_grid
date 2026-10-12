# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Tao Jin

execute_process(
  COMMAND "${PROGRAM}" --help
  RESULT_VARIABLE result
  OUTPUT_VARIABLE output
  ERROR_VARIABLE error)
if(NOT result EQUAL 0
   OR NOT output MATCHES "--config"
   OR error)
  message(FATAL_ERROR "Help failed: ${result}: ${output}${error}")
endif()

# No legacy option is silently accepted; startup returns before GPU/camera I/O.
execute_process(
  COMMAND "${PROGRAM}" --voxel 0.02
  RESULT_VARIABLE result
  OUTPUT_VARIABLE output
  ERROR_VARIABLE error)
if(NOT result EQUAL 2 OR NOT error MATCHES "--config")
  message(
    FATAL_ERROR "Legacy input was not refused: ${result}: ${output}${error}")
endif()

file(MAKE_DIRECTORY "${WORK_DIR}")
file(WRITE "${WORK_DIR}/bad.json" "{\n")
execute_process(
  COMMAND "${PROGRAM}" --config "${WORK_DIR}/bad.json"
  RESULT_VARIABLE result
  OUTPUT_VARIABLE output
  ERROR_VARIABLE error)
if(NOT result EQUAL 2 OR error STREQUAL "")
  message(
    FATAL_ERROR
      "Invalid configuration reached setup: ${result}: ${output}${error}")
endif()
file(REMOVE "${WORK_DIR}/bad.json")

# A valid mesh configuration reaches the real runtime adapter and resolves its
# asset relative to the config even when launched from another directory.
get_filename_component(program_name "${PROGRAM}" NAME)
if(program_name STREQUAL "codec_mesh")
  file(WRITE "${WORK_DIR}/shape.obj" "v 0 0 0\nv 1 0 0\nv 0 2 0\nf 1 2 3\n")
  file(
    WRITE "${WORK_DIR}/inspect.json"
    "{\"version\":1,\"operation\":\"codec\",\"source\":{\"kind\":\"mesh\",\"path\":\"shape.obj\",\"height_m\":1.7,\"up_axis\":\"y\"},\"codec\":{\"inspect_only\":true}}\n"
  )
  file(MAKE_DIRECTORY "${WORK_DIR}/elsewhere")
  execute_process(
    COMMAND "${PROGRAM}" --config "${WORK_DIR}/inspect.json"
    WORKING_DIRECTORY "${WORK_DIR}/elsewhere"
    RESULT_VARIABLE result
    OUTPUT_VARIABLE output
    ERROR_VARIABLE error)
  if(NOT result EQUAL 0 OR NOT output MATCHES "normalized bounds")
    message(
      FATAL_ERROR
        "Mesh configuration did not reach normalization: ${result}: ${output}${error}"
    )
  endif()
  file(REMOVE "${WORK_DIR}/shape.obj" "${WORK_DIR}/inspect.json")
endif()
