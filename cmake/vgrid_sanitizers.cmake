# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Tao Jin

# Sanitizer flags, applied GLOBALLY (compile + link) when VGRID_SANITIZE is set.
#
# Unlike vgrid_warnings (per-target PRIVATE), sanitizers are applied globally on
# purpose: the flags must reach the link line, and ASan's ABI is contagious --
# mixing a sanitized library with an un-sanitized executable is broken.
# Including this before add_subdirectory(third_party) also instruments any
# vendored static deps, which otherwise raise alloc/dealloc-mismatch false
# positives when linked into a sanitized test binary. Off by default, so normal
# builds are unaffected. A project that adds grid applies its own sanitizers
# before it does, and they reach grid's directories too.
#
# Example: cmake -B build -DCMAKE_BUILD_TYPE=Debug
# -DVGRID_SANITIZE="address;undefined"
if(VGRID_SANITIZE)
  # address;undefined -> address,undefined (the -fsanitize= argument form).
  list(JOIN VGRID_SANITIZE "," _vgrid_sanitize_list)
  set(_vgrid_sanitize_flags -fsanitize=${_vgrid_sanitize_list}
                            -fno-omit-frame-pointer -fno-sanitize-recover=all)

  add_compile_options(${_vgrid_sanitize_flags})
  add_link_options(${_vgrid_sanitize_flags})

  message(STATUS "Sanitizers enabled (VGRID_SANITIZE): ${VGRID_SANITIZE}")
endif()
