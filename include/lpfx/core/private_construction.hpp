#ifndef LPFX_CORE_PRIVATE_CONSTRUCTION_HPP
#define LPFX_CORE_PRIVATE_CONSTRUCTION_HPP

#include <utility>

#include <lpfx/core/attributes.hpp>

namespace lpfx::detail {

// Prepared types and device views friend this zero-state gate when
// construction must remain restricted to validated factories.
struct PrivateConstructionAccess final {
  LPFX_EXEC_CHECK_DISABLE
  template <class T, class... Argument>
  [[nodiscard]] LPFX_HOST_DEVICE static constexpr T
  create(Argument &&...argument) noexcept(
      noexcept(T{std::forward<Argument>(argument)...})) {
    return T{std::forward<Argument>(argument)...};
  }
};

} // namespace lpfx::detail

#endif
