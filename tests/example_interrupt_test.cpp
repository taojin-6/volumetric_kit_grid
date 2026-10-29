// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// The live examples' Ctrl+C (examples/common/interrupt.hpp): once caught, a
// raised SIGINT sets the flag and restores SIGINT's default handler, so a
// second one would end the process, and leaves SIGTERM caught. One SIGINT is
// raised, and only once the handler is seen installed. Host only.

#include <signal.h>

#include <csignal>
#include <cstdio>

#include "interrupt.hpp"
#include "volumetric_kit/core/testing/test_check.hpp"

namespace {

// `signal`'s handler now, left as it is.
void (*handler_of(int signal))(int) {
  struct sigaction now{};
  if (::sigaction(signal, nullptr, &now) != 0) return SIG_ERR;
  return now.sa_handler;
}

}  // namespace

int main() {
  CHECK(!vgrid_example::interrupted());
  vgrid_example::catch_interrupts();
  CHECK(handler_of(SIGINT) != SIG_DFL && handler_of(SIGINT) != SIG_ERR);
  CHECK(handler_of(SIGTERM) != SIG_DFL && handler_of(SIGTERM) != SIG_ERR);
  CHECK(!vgrid_example::interrupted());

  CHECK(std::raise(SIGINT) == 0);
  CHECK(vgrid_example::interrupted());
  CHECK(handler_of(SIGINT) == SIG_DFL);
  CHECK(handler_of(SIGTERM) != SIG_DFL);
  std::puts("example_interrupt: OK");
  return 0;
}
