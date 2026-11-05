#ifndef LPFX_DISTRIBUTED_DESCRIPTOR_FAMILY_HPP
#define LPFX_DISTRIBUTED_DESCRIPTOR_FAMILY_HPP

#include <cstdint>

namespace lpfx::distributed {

enum class operation_family : std::uint8_t {
  collective = 0,
  field = 1,
  particle = 2,
  message = 3,
};

} // namespace lpfx::distributed

#endif
