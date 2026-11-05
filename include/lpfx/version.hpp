#ifndef LPFX_VERSION_HPP
#define LPFX_VERSION_HPP

#include <cstdint>

#include <lpfx/config.hpp>

namespace lpfx {

inline constexpr std::uint32_t version_major = LPFX_VERSION_MAJOR;
inline constexpr std::uint32_t version_minor = LPFX_VERSION_MINOR;
inline constexpr std::uint32_t version_patch = LPFX_VERSION_PATCH;

}  // namespace lpfx

#endif
