#ifndef LPFX_DATA_FIELD_WIRE_TRAITS_HPP
#define LPFX_DATA_FIELD_WIRE_TRAITS_HPP

#include <cstdint>

namespace lpfx::field {

// Stable protocol identities for the built-in fixed-width scalar
// representations.  A custom representation opts in once by specializing
// wire_traits<T>; operation sites never repeat an integer wire ID.
template <class T> struct wire_traits;

template <> struct wire_traits<float> final {
  static constexpr std::uint64_t identity = UINT64_C(0x4c50465853000001);
};
template <> struct wire_traits<double> final {
  static constexpr std::uint64_t identity = UINT64_C(0x4c50465853000002);
};
template <> struct wire_traits<std::int8_t> final {
  static constexpr std::uint64_t identity = UINT64_C(0x4c50465853000003);
};
template <> struct wire_traits<std::uint8_t> final {
  static constexpr std::uint64_t identity = UINT64_C(0x4c50465853000004);
};
template <> struct wire_traits<std::int16_t> final {
  static constexpr std::uint64_t identity = UINT64_C(0x4c50465853000005);
};
template <> struct wire_traits<std::uint16_t> final {
  static constexpr std::uint64_t identity = UINT64_C(0x4c50465853000006);
};
template <> struct wire_traits<std::int32_t> final {
  static constexpr std::uint64_t identity = UINT64_C(0x4c50465853000007);
};
template <> struct wire_traits<std::uint32_t> final {
  static constexpr std::uint64_t identity = UINT64_C(0x4c50465853000008);
};
template <> struct wire_traits<std::int64_t> final {
  static constexpr std::uint64_t identity = UINT64_C(0x4c50465853000009);
};
template <> struct wire_traits<std::uint64_t> final {
  static constexpr std::uint64_t identity = UINT64_C(0x4c5046585300000a);
};

template <class T>
concept wire_scalar = requires {
  requires wire_traits<T>::identity != 0;
};

} // namespace lpfx::field

#endif
