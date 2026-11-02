// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include "volumetric_kit/grid/networking/dial.hpp"

#include <arpa/inet.h>
#include <poll.h>

#include <string_view>
#include <utility>

#include "socket.hpp"
#include "volumetric_kit/core/base/exception_status.hpp"

namespace volumetric_kit::grid::networking {

core::Result<core::UniqueFd> dial(const std::string& host, std::uint16_t port,
                                  std::chrono::milliseconds limit) {
  constexpr std::string_view kApi = "dial";
  try {
    VKC_ASSIGN(const detail::Address address,
               detail::one_host(kApi, host, port));
    if (port == 0) {
      return core::Status::invalid_argument(std::string(kApi) +
                                            ": port 0 is not a port to dial");
    }
    if (limit <= std::chrono::milliseconds{0}) {
      return core::Status::invalid_argument(
          std::string(kApi) + ": a limit of " + std::to_string(limit.count()) +
          " ms; it must be positive");
    }
    return detail::connect_within(
        std::string(kApi) + ": " + detail::to_string(address), address, limit);
  } catch (...) {
    return core::exception_status(kApi, core::Status::Code::IoError);
  }
}

Listener::Listener(core::UniqueFd fd, std::uint16_t port,
                   std::string name) noexcept
    : fd_(std::move(fd)), port_(port), name_(std::move(name)) {}

Listener::~Listener() = default;

core::Result<std::unique_ptr<Listener>> Listener::create(
    const std::string& host, std::uint16_t port) {
  constexpr std::string_view kApi = "Listener::create";
  try {
    VKC_ASSIGN(detail::Address address, detail::one_host(kApi, host, port));
    VKC_ASSIGN(core::UniqueFd fd,
               detail::listen_on(kApi, address, detail::kReceiveBuffer));
    VKC_ASSIGN(const std::uint16_t bound, detail::bound_port(kApi, fd.get()));
    address.in.sin_port = htons(bound);
    std::string name = "Listener: " + detail::to_string(address);
    return std::unique_ptr<Listener>(
        new Listener(std::move(fd), bound, std::move(name)));
  } catch (...) {
    return core::exception_status(kApi, core::Status::Code::IoError);
  }
}

core::Result<core::UniqueFd> Listener::accept(std::chrono::milliseconds wait) {
  try {
    VKC_ASSIGN(const short ready,
               detail::wait_for(name_, fd_.get(), POLLIN, wait));
    if (ready == 0) return core::UniqueFd();
    detail::Address peer;
    return detail::accept_one(name_, fd_.get(), &peer);
  } catch (...) {
    return core::exception_status(name_, core::Status::Code::IoError);
  }
}

std::uint16_t Listener::port() const noexcept { return port_; }

}  // namespace volumetric_kit::grid::networking
