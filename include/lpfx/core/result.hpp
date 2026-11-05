#ifndef LPFX_CORE_RESULT_HPP
#define LPFX_CORE_RESULT_HPP

#include <memory>
#include <type_traits>
#include <utility>

#include <lpfx/core/error.hpp>

namespace lpfx {

namespace detail {

template<class T>
concept result_value =
    requires { sizeof(T); } && std::is_object_v<T> && !std::is_array_v<T> &&
    std::is_same_v<T, std::remove_cv_t<T>> &&
    std::is_nothrow_move_constructible_v<T> &&
    std::is_nothrow_destructible_v<T>;

struct ResultValueTag final {};
struct ResultErrorTag final {};

template<class T>
union ResultStorage {
  constexpr ResultStorage() noexcept : inactive{} {}

  constexpr ~ResultStorage()
    requires std::is_trivially_destructible_v<T>
  = default;

  constexpr ~ResultStorage() noexcept
    requires (!std::is_trivially_destructible_v<T>)
  {}

  unsigned char inactive;
  T value;
  ErrorCode error;
};

}  // namespace detail

// Result owns either T or one byte of error state.  T is constrained to
// no-throw lifetime operations so switching the active union member cannot
// leave a partially formed result when exceptions are disabled.
template<detail::result_value T>
class [[nodiscard]] Result final {
public:
  [[nodiscard]] static constexpr Result from_value(T&& value) noexcept {
    return Result{detail::ResultValueTag{}, std::move(value)};
  }

  [[nodiscard]] static constexpr Result from_value(const T& value) noexcept
    requires std::is_nothrow_copy_constructible_v<T>
  {
    return Result{detail::ResultValueTag{}, value};
  }

  [[nodiscard]] static constexpr Result from_error(
      const ErrorCode error) noexcept {
    return Result{detail::ResultErrorTag{}, normalize_error(error)};
  }

  constexpr Result(Result&& other) noexcept
      : storage_{}, has_value_(other.has_value_) {
    construct_from_move(other);
  }

  constexpr Result(const Result& other) noexcept
    requires std::is_nothrow_copy_constructible_v<T>
      : storage_{}, has_value_(other.has_value_) {
    construct_from_copy(other);
  }

  constexpr Result& operator=(Result&& other) noexcept {
    if (this != &other) {
      destroy_active();
      has_value_ = other.has_value_;
      construct_from_move(other);
    }
    return *this;
  }

  constexpr Result& operator=(const Result& other) noexcept
    requires std::is_nothrow_copy_constructible_v<T>
  {
    if (this != &other) {
      destroy_active();
      has_value_ = other.has_value_;
      construct_from_copy(other);
    }
    return *this;
  }

  constexpr ~Result()
    requires std::is_trivially_destructible_v<T>
  = default;

  constexpr ~Result() noexcept
    requires (!std::is_trivially_destructible_v<T>)
  {
    destroy_active();
  }

  [[nodiscard]] constexpr bool has_value() const noexcept {
    return has_value_;
  }

  // Hard precondition: has_value() is true.
  [[nodiscard]] constexpr T& value() & noexcept {
    return storage_.value;
  }

  // Hard precondition: has_value() is true.
  [[nodiscard]] constexpr const T& value() const& noexcept {
    return storage_.value;
  }

  // Hard precondition: has_value() is true.
  [[nodiscard]] constexpr T&& value() && noexcept {
    return std::move(storage_.value);
  }

  // Hard precondition: has_value() is false.
  [[nodiscard]] constexpr ErrorCode error() const noexcept {
    return storage_.error;
  }

private:
  constexpr Result(detail::ResultValueTag, T&& value) noexcept
      : storage_{}, has_value_(true) {
    std::construct_at(std::addressof(storage_.value), std::move(value));
  }

  constexpr Result(detail::ResultValueTag, const T& value) noexcept
    requires std::is_nothrow_copy_constructible_v<T>
      : storage_{}, has_value_(true) {
    std::construct_at(std::addressof(storage_.value), value);
  }

  constexpr Result(
      detail::ResultErrorTag, const ErrorCode error) noexcept
      : storage_{}, has_value_(false) {
    std::construct_at(std::addressof(storage_.error), error);
  }

  [[nodiscard]] static constexpr ErrorCode normalize_error(
      const ErrorCode error) noexcept {
    // A failed result cannot carry success.  Turn that caller mistake into the
    // existing invalid-state category instead of adding a fallible factory.
    return error == ErrorCode::success ? ErrorCode::invalid_state : error;
  }

  constexpr void destroy_active() noexcept {
    if (has_value_) {
      std::destroy_at(std::addressof(storage_.value));
    }
  }

  constexpr void construct_from_move(Result& other) noexcept {
    if (has_value_) {
      std::construct_at(
          std::addressof(storage_.value), std::move(other.storage_.value));
    } else {
      std::construct_at(
          std::addressof(storage_.error), other.storage_.error);
    }
  }

  constexpr void construct_from_copy(const Result& other) noexcept
    requires std::is_nothrow_copy_constructible_v<T>
  {
    if (has_value_) {
      std::construct_at(std::addressof(storage_.value), other.storage_.value);
    } else {
      std::construct_at(
          std::addressof(storage_.error), other.storage_.error);
    }
  }

  detail::ResultStorage<T> storage_;
  bool has_value_;
};

namespace detail {

template <class T>
[[nodiscard]] constexpr ErrorCode try_error(const Result<T>& result) noexcept {
  return result.has_value() ? ErrorCode::success : result.error();
}

}  // namespace detail

}  // namespace lpfx

#endif
