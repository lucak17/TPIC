#ifndef LPFX_API_FIELD_DESCRIPTORS_HPP
#define LPFX_API_FIELD_DESCRIPTORS_HPP

#include <concepts>
#include <cstddef>
#include <cstdint>
#include <tuple>
#include <type_traits>
#include <utility>

#include <lpfx/core/checked_arithmetic.hpp>
#include <lpfx/core/error.hpp>
#include <lpfx/core/meta.hpp>
#include <lpfx/data/field/component.hpp>
#include <lpfx/domain/cartesian/halo.hpp>

namespace lpfx {

class cells final {
public:
  constexpr cells() noexcept = default;

  template <detail::checked_integer X, detail::checked_integer Y,
            detail::checked_integer Z>
  constexpr cells(const X x, const Y y, const Z z) noexcept {
    if (!detail::checked_integral_cast(x, x_) ||
        !detail::checked_integral_cast(y, y_) ||
        !detail::checked_integral_cast(z, z_)) {
      error_ = ErrorCode::overflow;
    } else if (x_ == 0 || y_ == 0 || z_ == 0) {
      error_ = ErrorCode::invalid_argument;
    } else {
      error_ = ErrorCode::success;
    }
  }

  [[nodiscard]] constexpr bool valid() const noexcept {
    return error_ == ErrorCode::success;
  }
  [[nodiscard]] constexpr ErrorCode error() const noexcept { return error_; }
  [[nodiscard]] constexpr std::uint64_t x() const noexcept { return x_; }
  [[nodiscard]] constexpr std::uint64_t y() const noexcept { return y_; }
  [[nodiscard]] constexpr std::uint64_t z() const noexcept { return z_; }

private:
  std::uint64_t x_{};
  std::uint64_t y_{};
  std::uint64_t z_{};
  ErrorCode error_{ErrorCode::invalid_argument};
};

namespace field {

using lpfx::cells;

// A value keyed by one component.
template <lpfx::detail::FieldComponentType Component, class T>
struct component_entry final {
  using component_type = Component;
  using value_type = T;
  T value;
};

namespace detail {

template <class T> inline constexpr bool is_component_entry = false;

template <class Component, class T>
inline constexpr bool is_component_entry<component_entry<Component, T>> = true;

template <class T>
concept ComponentEntry = is_component_entry<std::remove_cv_t<T>>;

// Entries with distinct components, looked up by component type.
template <ComponentEntry... Entries>
  requires(sizeof...(Entries) > 0) &&
          meta::all_distinct<typename Entries::component_type...>
class component_map {
public:
  constexpr explicit component_map(Entries... entries) noexcept(
      (std::is_nothrow_move_constructible_v<Entries> && ...))
      : entries_(std::move(entries)...) {}

  template <class Component>
    requires meta::contains<Component, typename Entries::component_type...>
  [[nodiscard]] constexpr const auto &get() const noexcept {
    return std::get<meta::index_of<Component,
                                   typename Entries::component_type...>>(
               entries_)
        .value;
  }

  template <class Predicate>
  [[nodiscard]] constexpr bool all_of(const Predicate predicate) const noexcept {
    return std::apply(
        [&](const Entries &...entries) noexcept {
          return (predicate(entries.value) && ...);
        },
        entries_);
  }

private:
  std::tuple<Entries...> entries_;
};

// True when the entries name exactly the schema's components.
template <class... Entries, class... Components>
[[nodiscard]] consteval bool
exact_entries(const component_map<Entries...> *,
              const schema<Components...> *) noexcept {
  return sizeof...(Entries) == sizeof...(Components) &&
         (meta::contains<typename Entries::component_type, Components...> &&
          ...);
}

[[nodiscard]] consteval bool exact_entries(const void *, const void *) noexcept {
  return false;
}

} // namespace detail

namespace halos {

class value final {
public:
  constexpr value() noexcept = default;

  // A scalar halo applies to both sides of every axis. Validation is kept in
  // the descriptor so factories can report errors without exceptions.
  template <lpfx::detail::checked_integer Depth>
  constexpr value(const Depth depth) noexcept {
    std::uint64_t converted{};
    if (!lpfx::detail::checked_integral_cast(depth, converted)) {
      error_ = ErrorCode::overflow;
      return;
    }
    x_minus_ = x_plus_ = y_minus_ = y_plus_ = z_minus_ = z_plus_ = converted;
  }

  [[nodiscard]] constexpr bool valid() const noexcept {
    return error_ == ErrorCode::success;
  }
  [[nodiscard]] constexpr ErrorCode error() const noexcept { return error_; }

  template <lpfx::detail::CartesianUnsignedWord Count>
  [[nodiscard]] constexpr ErrorCode
  try_make(HaloDepth3<Count> &output) const noexcept {
    if (!valid()) {
      return error_;
    }
    return try_make_halo_depth3(x_minus_, x_plus_, y_minus_, y_plus_, z_minus_,
                                z_plus_, output);
  }

private:
  constexpr value(const std::uint64_t x_minus, const std::uint64_t x_plus,
                  const std::uint64_t y_minus, const std::uint64_t y_plus,
                  const std::uint64_t z_minus, const std::uint64_t z_plus,
                  const ErrorCode error) noexcept
      : x_minus_(x_minus), x_plus_(x_plus), y_minus_(y_minus), y_plus_(y_plus),
        z_minus_(z_minus), z_plus_(z_plus), error_(error) {}

  template <lpfx::detail::checked_integer XMinus,
            lpfx::detail::checked_integer XPlus,
            lpfx::detail::checked_integer YMinus,
            lpfx::detail::checked_integer YPlus,
            lpfx::detail::checked_integer ZMinus,
            lpfx::detail::checked_integer ZPlus>
  friend constexpr value asymmetric(XMinus, XPlus, YMinus, YPlus, ZMinus,
                                    ZPlus) noexcept;

  std::uint64_t x_minus_{};
  std::uint64_t x_plus_{};
  std::uint64_t y_minus_{};
  std::uint64_t y_plus_{};
  std::uint64_t z_minus_{};
  std::uint64_t z_plus_{};
  ErrorCode error_{ErrorCode::success};
};

template <lpfx::detail::checked_integer XMinus,
          lpfx::detail::checked_integer XPlus,
          lpfx::detail::checked_integer YMinus,
          lpfx::detail::checked_integer YPlus,
          lpfx::detail::checked_integer ZMinus,
          lpfx::detail::checked_integer ZPlus>
[[nodiscard]] constexpr value asymmetric(const XMinus x_minus,
                                         const XPlus x_plus,
                                         const YMinus y_minus,
                                         const YPlus y_plus,
                                         const ZMinus z_minus,
                                         const ZPlus z_plus) noexcept {
  std::uint64_t prepared_x_minus{};
  std::uint64_t prepared_x_plus{};
  std::uint64_t prepared_y_minus{};
  std::uint64_t prepared_y_plus{};
  std::uint64_t prepared_z_minus{};
  std::uint64_t prepared_z_plus{};
  const bool valid =
      lpfx::detail::checked_integral_cast(x_minus, prepared_x_minus) &&
      lpfx::detail::checked_integral_cast(x_plus, prepared_x_plus) &&
      lpfx::detail::checked_integral_cast(y_minus, prepared_y_minus) &&
      lpfx::detail::checked_integral_cast(y_plus, prepared_y_plus) &&
      lpfx::detail::checked_integral_cast(z_minus, prepared_z_minus) &&
      lpfx::detail::checked_integral_cast(z_plus, prepared_z_plus);
  return value{prepared_x_minus, prepared_x_plus, prepared_y_minus,
               prepared_y_plus,  prepared_z_minus, prepared_z_plus,
               valid ? ErrorCode::success : ErrorCode::overflow};
}

template <lpfx::detail::checked_integer Depth>
[[nodiscard]] constexpr value uniform(const Depth depth) noexcept {
  return asymmetric(depth, depth, depth, depth, depth, depth);
}

template <lpfx::detail::checked_integer X, lpfx::detail::checked_integer Y,
          lpfx::detail::checked_integer Z>
[[nodiscard]] constexpr value symmetric(const X x, const Y y,
                                        const Z z) noexcept {
  return asymmetric(x, x, y, y, z, z);
}

template <field::detail::Component Component>
[[nodiscard]] constexpr component_entry<Component, value>
for_component(const value halo) noexcept {
  return {halo};
}

template <field::detail::ComponentEntry... Entries>
  requires(std::same_as<typename Entries::value_type, value> && ...)
[[nodiscard]] constexpr auto per_component(const Entries... entries) noexcept {
  return field::detail::component_map<Entries...>{entries...};
}

} // namespace halos

template <lpfx::detail::FieldComponentType Component, class T>
[[nodiscard]] constexpr auto component_value(T &&value) noexcept(
    std::is_nothrow_constructible_v<std::remove_cvref_t<T>, T &&>) {
  return component_entry<Component, std::remove_cvref_t<T>>{
      std::forward<T>(value)};
}

// fill{value} sets every component; fill{component_value<C>(v)...} names one
// value per component.
template <class... Values> class fill;

template <class T>
  requires(!detail::ComponentEntry<T>)
class fill<T> final {
public:
  using value_type = T;

  constexpr explicit fill(T value) noexcept(
      std::is_nothrow_move_constructible_v<T>)
      : value_(std::move(value)) {}

  [[nodiscard]] constexpr const T &value() const noexcept { return value_; }

private:
  T value_;
};

template <detail::ComponentEntry... Entries>
  requires(sizeof...(Entries) > 0) &&
          meta::all_distinct<typename Entries::component_type...>
class fill<Entries...> final : public detail::component_map<Entries...> {
public:
  using detail::component_map<Entries...>::component_map;
};

template <class... Values>
fill(Values &&...) -> fill<std::remove_cvref_t<Values>...>;

// Common creation settings. Omitted halos have depth zero; component
// locations still determine the sample extents.
struct options final {
  lpfx::cells cells{};
  halos::value halo{};
};

namespace detail {

template <class T> inline constexpr bool is_halo_spec = false;

template <> inline constexpr bool is_halo_spec<halos::value> = true;

template <class... Entries>
inline constexpr bool is_halo_spec<component_map<Entries...>> =
    (std::same_as<typename Entries::value_type, halos::value> && ...);

template <class T>
concept HaloSpec = is_halo_spec<std::remove_cv_t<T>>;

template <class T> inline constexpr bool is_fill = false;

template <class... Values> inline constexpr bool is_fill<fill<Values...>> = true;

template <class T>
concept FillSpec = is_fill<std::remove_cv_t<T>>;

template <class T> inline constexpr bool is_scalar_fill = false;

template <class T>
  requires(!ComponentEntry<T>)
inline constexpr bool is_scalar_fill<fill<T>> = true;

// A halo or fill spec that supplies exactly the schema's components.
template <class Spec, class Schema>
concept ExactFor =
    std::same_as<Spec, halos::value> || is_scalar_fill<Spec> ||
    exact_entries(static_cast<const Spec *>(nullptr),
                  static_cast<const Schema *>(nullptr));

template <Component Component>
[[nodiscard]] constexpr halos::value get(const halos::value common) noexcept {
  return common;
}

template <Component Component, class T>
  requires(!ComponentEntry<T>)
[[nodiscard]] constexpr const T &get(const fill<T> &values) noexcept {
  return values.value();
}

template <Component Component, class... Entries>
[[nodiscard]] constexpr const auto &
get(const component_map<Entries...> &values) noexcept {
  return values.template get<Component>();
}

} // namespace detail

template <detail::HaloSpec HaloSpec> class shape final {
public:
  using halo_spec_type = HaloSpec;

  constexpr shape(const lpfx::cells cells, HaloSpec halos) noexcept
      : cells_(cells), halos_(std::move(halos)) {}

  [[nodiscard]] constexpr bool valid() const noexcept {
    if constexpr (std::same_as<HaloSpec, halos::value>) {
      return cells_.valid() && halos_.valid();
    } else {
      return cells_.valid() &&
             halos_.all_of([](const halos::value &halo) noexcept {
               return halo.valid();
             });
    }
  }
  [[nodiscard]] constexpr lpfx::cells cells() const noexcept { return cells_; }
  [[nodiscard]] constexpr const HaloSpec &halos() const noexcept {
    return halos_;
  }

private:
  lpfx::cells cells_{};
  HaloSpec halos_;
};

template <detail::HaloSpec HaloSpec>
shape(lpfx::cells, HaloSpec) -> shape<HaloSpec>;

namespace detail {

[[nodiscard]] constexpr shape<halos::value>
as_shape(const options &settings) noexcept {
  return {settings.cells, settings.halo};
}

template <class HaloSpec>
[[nodiscard]] constexpr const shape<HaloSpec> &
as_shape(const shape<HaloSpec> &descriptor) noexcept {
  return descriptor;
}

template <FillSpec Initial>
[[nodiscard]] constexpr const Initial &as_fill(const Initial &initial) noexcept {
  return initial;
}

template <class Initial>
  requires(!FillSpec<Initial>) && (!ComponentEntry<Initial>) &&
          std::is_nothrow_copy_constructible_v<Initial> &&
          std::is_nothrow_move_constructible_v<Initial>
[[nodiscard]] constexpr auto as_fill(const Initial &initial) noexcept {
  return fill{initial};
}

template <Schema Schema> struct SchemaValues;

template <Component... Components>
struct SchemaValues<schema<Components...>> final {
  template <class... Values>
    requires(sizeof...(Values) == sizeof...(Components)) &&
            (std::is_nothrow_constructible_v<typename Components::value_type,
                                             Values &&> &&
             ...)
  [[nodiscard]] static constexpr auto make(Values &&...values) noexcept {
    return fill{component_value<Components>(
        typename Components::value_type(std::forward<Values>(values)))...};
  }

  [[nodiscard]] static constexpr auto make() noexcept
    requires(std::is_nothrow_default_constructible_v<
                 typename Components::value_type> &&
             ...)
  {
    return fill{
        component_value<Components>(typename Components::value_type{})...};
  }
};

} // namespace detail

// One value per component in schema order. An empty argument list
// value-initializes every component.
template <detail::Schema Schema, class... Values>
  requires requires(Values &&...values) {
    detail::SchemaValues<Schema>::make(std::forward<Values>(values)...);
  }
[[nodiscard]] constexpr auto values(Values &&...values) noexcept {
  return detail::SchemaValues<Schema>::make(std::forward<Values>(values)...);
}

} // namespace field

} // namespace lpfx

#endif
