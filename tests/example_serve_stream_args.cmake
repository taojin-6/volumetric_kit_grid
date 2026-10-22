# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Tao Jin

# serve_stream's arguments (examples/serve_stream): help exits 0; bad ones exit
# 2 with the usage; a recording that is missing exits 1, naming it, before
# anything listens.

execute_process(
  COMMAND "${PROGRAM}" --help
  RESULT_VARIABLE result
  OUTPUT_VARIABLE output
  ERROR_VARIABLE error)
if(NOT result EQUAL 0
   OR NOT output MATCHES "usage: serve_stream"
   OR error)
  message(FATAL_ERROR "Help failed: ${result}: ${output}${error}")
endif()

foreach(
  case
  "|no recording"
  "--port|--port needs"
  "a.vrst;--port|--port needs"
  "a.vrst;--port;65536|65536 is not a port"
  "a.vrst;--port;-1|-1 is not a port"
  "a.vrst;--port;x|x is not a port"
  "a.vrst;--port;4x|4x is not a port"
  "a.vrst;b.vrst|one recording"
  "a.vrst;--host;127.0.0.1|unknown option --host")
  string(FIND "${case}" "|" bar)
  string(SUBSTRING "${case}" 0 ${bar} args)
  math(EXPR bar "${bar} + 1")
  string(SUBSTRING "${case}" ${bar} -1 pattern)
  execute_process(
    COMMAND "${PROGRAM}" ${args}
    RESULT_VARIABLE result
    OUTPUT_VARIABLE output
    ERROR_VARIABLE error)
  if(NOT result EQUAL 2
     OR NOT error MATCHES "${pattern}"
     OR NOT error MATCHES "usage: serve_stream")
    message(SEND_ERROR "Arguments '${args}': ${result}: ${output}${error}")
  endif()
endforeach()

file(MAKE_DIRECTORY "${WORK_DIR}")
execute_process(
  COMMAND "${PROGRAM}" "${WORK_DIR}/missing.vrst" --port 0
  RESULT_VARIABLE result
  OUTPUT_VARIABLE output
  ERROR_VARIABLE error)
if(NOT result EQUAL 1
   OR NOT error MATCHES "missing.vrst"
   OR output MATCHES "serving")
  message(FATAL_ERROR "A missing file: ${result}: ${output}${error}")
endif()
