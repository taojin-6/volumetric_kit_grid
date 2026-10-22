// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// The senders' log (examples/common/info_log.hpp), over a temporary file: Info,
// Warning and Error each one line, `[<source> <level>] <message>`, the message
// by its length; Debug dropped; the same once installed as the log handler;
// and 4 threads' lines, each whole. Host only.

#include <cstdio>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include "info_log.hpp"
#include "test_check.hpp"

namespace vkc = volumetric_kit::core;

namespace {

// What `file` holds, from its start.
std::string contents(std::FILE* file) {
  std::fflush(file);
  std::rewind(file);
  std::string out;
  char buffer[4096];
  std::size_t n = 0;
  while ((n = std::fread(buffer, 1, sizeof buffer, file)) > 0) {
    out.append(buffer, n);
  }
  return out;
}

}  // namespace

int main() {
  std::FILE* file = std::tmpfile();
  CHECK(file != nullptr);
  const vkc::LogHandler handler = vr_example::info_log_handler(file);
  const std::string_view joined = "viewer 127.0.0.1:5 joined, 1 of 4, unread";
  handler(vkc::LogLevel::Debug, "vr", "a debug line");
  handler(vkc::LogLevel::Info, "vr", joined.substr(0, 33));
  handler(vkc::LogLevel::Warning, "core", "a warning");
  handler(vkc::LogLevel::Error, "gfx", "");
  CHECK(contents(file) ==
        "[vr info] viewer 127.0.0.1:5 joined, 1 of 4\n"
        "[core warning] a warning\n"
        "[gfx error] \n");
  std::fclose(file);

  // Installed, it serves log_message.
  file = std::tmpfile();
  CHECK(file != nullptr);
  vkc::set_log_handler(vr_example::info_log_handler(file));
  vkc::log_message(vkc::LogLevel::Debug, "vr", "dropped");
  vkc::log_message(vkc::LogLevel::Info, "vr", "viewer 127.0.0.1:5 left");
  vkc::set_log_handler({});
  CHECK(contents(file) == "[vr info] viewer 127.0.0.1:5 left\n");
  std::fclose(file);

  // Lines from 4 threads at once, each whole.
  constexpr int kThreads = 4;
  constexpr int kLines = 1000;
  const std::string padding(200, 'x');
  file = std::tmpfile();
  CHECK(file != nullptr);
  const vkc::LogHandler shared = vr_example::info_log_handler(file);
  std::vector<std::thread> threads;
  for (int t = 0; t < kThreads; ++t) {
    threads.emplace_back([&, t] {
      const std::string source = "t" + std::to_string(t);
      for (int i = 0; i < kLines; ++i) {
        shared(vkc::LogLevel::Info, source, padding + std::to_string(i));
      }
    });
  }
  for (std::thread& thread : threads) thread.join();
  const std::string all = contents(file);
  std::fclose(file);
  std::vector<int> next(kThreads, 0);
  std::size_t at = 0;
  for (int line = 0; line < kThreads * kLines; ++line) {
    const std::size_t end = all.find('\n', at);
    CHECK(end != std::string::npos);
    const std::string text = all.substr(at, end - at);
    at = end + 1;
    // "[t<k> info] <padding><i>", each thread's i in order.
    CHECK(text.size() > 4 && text[0] == '[' && text[1] == 't');
    const int t = text[2] - '0';
    CHECK(t >= 0 && t < kThreads);
    CHECK(text == "[t" + std::to_string(t) + " info] " + padding +
                      std::to_string(next[t]));
    ++next[t];
  }
  CHECK(at == all.size());
  std::puts("example_info_log: OK");
  return 0;
}
