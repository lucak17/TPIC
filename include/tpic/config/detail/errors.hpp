#ifndef TPIC_CONFIG_DETAIL_ERRORS_HPP
#define TPIC_CONFIG_DETAIL_ERRORS_HPP

#include <stdexcept>
#include <string>
#include <string_view>

namespace tpic::config::detail {

// The preparation errors of one config topic (input, domain, particles,
// physics, output): standard exceptions whose text is "TPIC <topic>: <message>".
// Each source of src/config holds one constant of this type.
struct Errors {
  std::string_view topic;

  [[nodiscard]] std::string text(std::string_view message) const {
    return "TPIC " + std::string(topic) + ": " + std::string(message);
  }
  [[noreturn]] void invalid(std::string_view message) const {
    throw std::invalid_argument(text(message));
  }
  [[noreturn]] void overflow(std::string_view message) const {
    throw std::overflow_error(text(message));
  }
  void require(bool condition, std::string_view message) const {
    if (!condition) invalid(message);
  }
};

} // namespace tpic::config::detail

#endif // TPIC_CONFIG_DETAIL_ERRORS_HPP
