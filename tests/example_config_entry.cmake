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

  # A numbered range is inspected on the host, every frame; a missing frame, a
  # malformed pattern and a range on a plain path are refused before any work.
  file(WRITE "${WORK_DIR}/shape_01.obj" "v 0 0 0\nv 1 0 0\nv 0 2 0\nf 1 2 3\n")
  file(WRITE "${WORK_DIR}/shape_02.obj" "v 0 0 0\nv 1 0 0\nv 0 3 0\nf 1 2 3\n")
  set(mesh_source
      "{\"version\":1,\"operation\":\"codec\",\"codec\":{\"inspect_only\":true},\"source\":{\"kind\":\"mesh\",\"height_m\":1.7,\"up_axis\":\"y\",\"path\":"
  )
  foreach(
    case
    "range|\"shape_%02d.obj\"},\"run\":{\"first\":1,\"frames\":2}}|0|frame 1: 3 boundary edges.*frame 2: 3 boundary edges.*inspection: 2 frame"
    "missing|\"shape_%02d.obj\"},\"run\":{\"first\":1,\"frames\":3}}|1|shape_03.obj"
    "pattern|\"shape_%s.obj\"}}|1|source.path"
    "plain|\"shape_01.obj\"},\"run\":{\"frames\":2}}|1|run.frames")
    string(REPLACE "|" ";" fields "${case}")
    list(GET fields 0 name)
    list(GET fields 1 rest)
    list(GET fields 2 expected)
    list(GET fields 3 pattern)
    file(WRITE "${WORK_DIR}/${name}.json" "${mesh_source}${rest}\n")
    execute_process(
      COMMAND "${PROGRAM}" --config "${WORK_DIR}/${name}.json"
      WORKING_DIRECTORY "${WORK_DIR}/elsewhere"
      RESULT_VARIABLE result
      OUTPUT_VARIABLE output
      ERROR_VARIABLE error)
    if(NOT result EQUAL expected OR NOT "${output}${error}" MATCHES
                                    "${pattern}")
      message(FATAL_ERROR "Mesh range ${name}: ${result}: ${output}${error}")
    endif()
    file(REMOVE "${WORK_DIR}/${name}.json")
  endforeach()
  file(REMOVE "${WORK_DIR}/shape_01.obj" "${WORK_DIR}/shape_02.obj")
endif()
