// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// serve_stream: serve a recorded .vrst live, paced by its pts and looping
// until Ctrl+C: a sender with no camera, for a player to connect to.
//
//   serve_stream <in.vrst> [--port N]
//
// It serves every interface on port 47801, or N (0 takes a free one, which it
// prints), and prints each viewer joining and leaving. No scenario file: its
// whole input is a file and a port.

#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <optional>
#include <string>
#include <vector>

#include "info_log.hpp"
#include "stream_replay.hpp"
#include "volumetric_kit/core/base/result.hpp"
#include "volumetric_kit/grid/networking/stream_server.hpp"
#include "volumetric_kit/grid/stream/stream_reader.hpp"

namespace vkc = volumetric_kit::core;
namespace networking = volumetric_kit::grid::networking;
namespace stream = volumetric_kit::grid::stream;

namespace {

constexpr const char* kUsage =
    "usage: serve_stream <in.vrst> [--port N]\n"
    "  Serves the recording live on every interface, paced by its pts and\n"
    "  looping until Ctrl+C. --port N: the port (default 47801; 0 takes a\n"
    "  free one).";

struct Args {
  std::string path;
  std::uint16_t port = networking::kDefaultPort;
  bool help = false;
};

// The arguments, or why they are refused.
std::optional<Args> parse(int argc, char** argv, std::string* why) {
  Args args;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--help" || arg == "-h") {
      args.help = true;
    } else if (arg == "--port") {
      if (i + 1 == argc) {
        *why = "--port needs a number";
        return std::nullopt;
      }
      const char* text = argv[++i];
      char* end = nullptr;
      errno = 0;
      const unsigned long port = std::strtoul(text, &end, 10);
      if (*text < '0' || *text > '9' || *end != '\0' || errno != 0 ||
          port > 65535) {
        *why = std::string("--port ") + text + " is not a port (0 to 65535)";
        return std::nullopt;
      }
      args.port = static_cast<std::uint16_t>(port);
    } else if (arg.size() > 1 && arg[0] == '-') {
      *why = "unknown option " + arg;
      return std::nullopt;
    } else if (!args.path.empty()) {
      *why = "one recording, not " + args.path + " and " + arg;
      return std::nullopt;
    } else {
      args.path = arg;
    }
  }
  if (args.path.empty() && !args.help) {
    *why = "no recording named";
    return std::nullopt;
  }
  return args;
}

// Serve until stopped; returns only a failure.
vkc::Status serve(const Args& args) {
  VKC_ASSIGN(const auto reader, stream::StreamReader::open(args.path));
  VKC_ASSIGN(const vgrid_example::ReplayLoop loop,
             vgrid_example::ReplayLoop::create(*reader));
  networking::ServerOptions options;
  options.port = args.port;
  // A recording cannot make a key: a joiner takes the GOP since the newest.
  options.replay_gop = true;
  VKC_ASSIGN(const auto server,
             networking::StreamServer::start(reader->session(), options));
  std::printf("serving %s on port %u: %zu records a pass\n", args.path.c_str(),
              unsigned{server->port()}, loop.records().size());
  std::fflush(stdout);
  const vgrid_example::ReplayLoop::Clock::time_point start =
      vgrid_example::ReplayLoop::Clock::now();
  const auto publish = [&](const stream::RecordInfo& info,
                           const std::vector<std::uint8_t>& payload) {
    return server->publish(info, payload.data(), payload.size());
  };
  for (std::uint64_t pass = 0;; ++pass) {
    VKC_TRY(loop.send_pass(*reader, start, pass, publish));
  }
}

}  // namespace

int main(int argc, char** argv) {
  std::string why;
  const std::optional<Args> args = parse(argc, argv, &why);
  if (!args) {
    std::fprintf(stderr, "serve_stream: %s\n%s\n", why.c_str(), kUsage);
    return 2;
  }
  if (args->help) {
    std::printf("%s\n", kUsage);
    return 0;
  }
  vgrid_example::print_info_log();  // each viewer joining and leaving
  const vkc::Status failed = serve(*args);
  std::fprintf(stderr, "serve_stream: %s\n", failed.message().c_str());
  return 1;
}
