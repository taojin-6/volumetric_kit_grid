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

  # Frames: a closed tetrahedron, one missing a face (open, twice), a lone
  # triangle signed mode refuses, a directory where a frame should be, an empty
  # file, vertices with no face, and a flat triangle.
  set(tetra "v 0 0 0\nv 1 0 0\nv 0 0 1\nv 0.2 1 0.2\n")
  file(WRITE "${WORK_DIR}/t_01.obj"
       "${tetra}f 1 2 3\nf 1 4 2\nf 1 3 4\nf 2 4 3\n")
  file(WRITE "${WORK_DIR}/t_02.obj"
       "${tetra}f 1 2 3\nf 1 4 2\nf 1 3 4\nf 2 4 3\n")
  file(WRITE "${WORK_DIR}/d_01.obj"
       "${tetra}f 1 2 3\nf 1 4 2\nf 1 3 4\nf 2 4 3\n")
  file(MAKE_DIRECTORY "${WORK_DIR}/d_02.obj")
  file(WRITE "${WORK_DIR}/open_01.obj" "${tetra}f 1 4 2\nf 1 3 4\nf 2 4 3\n")
  file(WRITE "${WORK_DIR}/open_02.obj" "v 0 0 0\nv 1 0 0\nv 0 2 0\nf 1 2 3\n")
  file(WRITE "${WORK_DIR}/open_03.obj" "${tetra}f 1 4 2\nf 1 3 4\nf 2 4 3\n")
  file(WRITE "${WORK_DIR}/empty.obj" "")
  file(WRITE "${WORK_DIR}/vertices.obj" "v 0 0 0\nv 1 0 0\n")
  file(WRITE "${WORK_DIR}/flat.obj" "v 0 0 0\nv 1 0 0\nv 0 0 1\nf 1 2 3\n")

  # A literal % in the scenario directory must survive config::load. The take0
  # decoy is closed; the intended mesh is open, so checking the inspection
  # counts also catches silently selecting the wrong file.
  file(MAKE_DIRECTORY "${WORK_DIR}/take0")
  configure_file("${WORK_DIR}/t_01.obj" "${WORK_DIR}/take0/shape.obj" COPYONLY)
  foreach(directory "50%off" "take%d")
    file(MAKE_DIRECTORY "${WORK_DIR}/${directory}")
    configure_file("${WORK_DIR}/open_01.obj"
                   "${WORK_DIR}/${directory}/shape.obj" COPYONLY)
    file(
      WRITE "${WORK_DIR}/${directory}/inspect.json"
      "{\"version\":1,\"operation\":\"codec\",\"source\":{\"kind\":\"mesh\",\"path\":\"shape.obj\",\"metres_per_unit\":1,\"up_axis\":\"y\"},\"codec\":{\"inspect_only\":true}}\n"
    )
    execute_process(
      COMMAND "${PROGRAM}" --config "${WORK_DIR}/${directory}/inspect.json"
      WORKING_DIRECTORY "${WORK_DIR}/elsewhere"
      RESULT_VARIABLE result
      OUTPUT_VARIABLE output
      ERROR_VARIABLE error)
    if(NOT result EQUAL 0 OR NOT output MATCHES
                             "inspection: 1 frame.s.; 1 open")
      message(
        SEND_ERROR
          "Mesh scenario under ${directory}: ${result}: ${output}${error}")
    endif()
    file(REMOVE_RECURSE "${WORK_DIR}/${directory}")
  endforeach()
  file(REMOVE_RECURSE "${WORK_DIR}/take0")

  # A numbered range is inspected on the host, every frame, and counted; a
  # missing frame, a malformed pattern and a range on a plain path are refused.
  set(mesh_source
      "{\"version\":1,\"operation\":\"codec\",\"codec\":{\"inspect_only\":true},\"source\":{\"kind\":\"mesh\",\"height_m\":1.7,\"up_axis\":\"y\",\"path\":"
  )
  foreach(
    case
    "range|\"open_%02d.obj\"},\"run\":{\"first\":1,\"frames\":3}}|0|frame 1: 3 boundary edges. signed mode needs source.allow_holes.*frame 2: 3 boundary edges, 1 components with nonpositive volume. signed mode refuses it.*frame 3: 3 boundary edges. signed mode needs.*inspection: 3 frame.s.. 2 open, which signed mode takes with source.allow_holes. 1 it refuses"
    "missing|\"t_%02d.obj\"},\"run\":{\"first\":1,\"frames\":3}}|1|t_03.obj"
    "pattern|\"t_%s.obj\"}}|1|source.path"
    "plain|\"t_01.obj\"},\"run\":{\"frames\":2}}|1|run.frames")
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

  # A run that would code is refused before any GPU work: with no Vulkan driver
  # to find, each case must fail with its own reason, never Vulkan's. The band,
  # a case's fourth field or 0.15 m, is the least a true SDF takes at 1 cm: 15
  # voxels, and the shell's half-thickness on top in shell mode.
  set(coding_band
      "{\"version\":1,\"operation\":\"codec\",\"fusion\":{\"truncation_m\":")
  set(coding_source "},\"source\":{\"kind\":\"mesh\",\"up_axis\":\"y\",")
  foreach(
    case
    "pattern|\"height_m\":1.7,\"path\":\"t_%s.obj\"}}|source.path"
    "missing|\"height_m\":1.7,\"path\":\"t_%02d.obj\"},\"run\":{\"first\":1,\"frames\":3}}|missing file.*t_03.obj"
    "directory|\"height_m\":1.7,\"path\":\"d_%02d.obj\"},\"run\":{\"first\":1,\"frames\":2}}|not a regular file.*d_02.obj"
    "plain_first|\"height_m\":1.7,\"path\":\"t_01.obj\"},\"run\":{\"first\":1}}|run.first"
    "open|\"height_m\":1.7,\"path\":\"open_%02d.obj\"},\"run\":{\"first\":1}}|source.allow_holes"
    "shell|\"height_m\":1.7,\"sdf_mode\":\"shell\",\"shell_voxels\":16,\"path\":\"t_01.obj\"}}|shell_voxels"
    "codec_step|\"height_m\":1.7,\"path\":\"t_01.obj\"},\"codec\":{\"quantization_scale\":1e-30}}|CodecParams"
    "update_size|\"height_m\":1.7,\"path\":\"t_01.obj\"},\"codec\":{\"intra_period\":2,\"update_size\":1025}}|update_size 1025"
    "empty|\"height_m\":1.7,\"path\":\"empty.obj\"}}|empty.obj"
    "vertices|\"height_m\":1.7,\"path\":\"vertices.obj\"}}|vertices.obj"
    "flat|\"height_m\":1.7,\"sdf_mode\":\"shell\",\"path\":\"flat.obj\"}}|zero extent along up|0.2"
    "overflow|\"metres_per_unit\":1e39,\"path\":\"t_%02d.obj\"},\"run\":{\"first\":1,\"frames\":2}}|past float range"
    "shell_band|\"height_m\":1.7,\"sdf_mode\":\"shell\",\"path\":\"t_01.obj\"}}|at least 16.5 voxels .0.165 m., the shell's half-thickness included"
    "band|\"height_m\":1.7,\"path\":\"t_01.obj\"}}|fusion.truncation_m 0.14 m is 14 voxels. a mesh codes as a true SDF in at least 15 voxels .0.15 m.|0.14"
  )
    string(REPLACE "|" ";" fields "${case}")
    list(GET fields 0 name)
    list(GET fields 1 rest)
    list(GET fields 2 pattern)
    set(band 0.15)
    list(LENGTH fields count)
    if(count EQUAL 4)
      list(GET fields 3 band)
    endif()
    file(WRITE "${WORK_DIR}/${name}.json"
         "${coding_band}${band}${coding_source}${rest}\n")
    execute_process(
      COMMAND
        "${CMAKE_COMMAND}" -E env VK_DRIVER_FILES=/nonexistent
        VK_ICD_FILENAMES=/nonexistent "${PROGRAM}" --config
        "${WORK_DIR}/${name}.json"
      RESULT_VARIABLE result
      OUTPUT_VARIABLE output
      ERROR_VARIABLE error)
    if(NOT result EQUAL 1
       OR NOT error MATCHES "${pattern}"
       OR error MATCHES "vkCreate|Vulkan|physical device")
      message(
        FATAL_ERROR "Mesh run ${name} reached the GPU or was not refused: "
                    "${result}: ${output}${error}")
    endif()
    file(REMOVE "${WORK_DIR}/${name}.json")
  endforeach()
  file(REMOVE_RECURSE "${WORK_DIR}/d_02.obj")
  file(
    REMOVE
    "${WORK_DIR}/t_01.obj"
    "${WORK_DIR}/t_02.obj"
    "${WORK_DIR}/d_01.obj"
    "${WORK_DIR}/open_01.obj"
    "${WORK_DIR}/open_02.obj"
    "${WORK_DIR}/open_03.obj"
    "${WORK_DIR}/empty.obj"
    "${WORK_DIR}/vertices.obj"
    "${WORK_DIR}/flat.obj")
endif()
