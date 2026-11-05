#ifndef LPFX_API_DISTRIBUTED_COMMUNICATION_HPP
#define LPFX_API_DISTRIBUTED_COMMUNICATION_HPP

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <tuple>
#include <type_traits>
#include <utility>

#include <lpfx/api/distributed/requests.hpp>
#include <lpfx/core/meta.hpp>
#include <lpfx/distributed/communication.hpp>
#include <lpfx/distributed/detail/integer_codec.hpp>

namespace lpfx::distributed {

struct PreparationLimits final {
  std::size_t registration = std::numeric_limits<std::size_t>::max();
  std::size_t operations = std::numeric_limits<std::size_t>::max();
};

namespace detail {

template <class T> constexpr auto request_tuple(T &&request) noexcept {
  return std::tuple<std::remove_cvref_t<T>>{std::forward<T>(request)};
}
template <class... T>
constexpr auto request_tuple(std::tuple<T...> requests) noexcept {
  return requests;
}

// Requests in key-identity order, so equal sets give equal types.
template <class Tuple, std::size_t... I>
[[nodiscard]] constexpr auto
canonical_requests(Tuple requests, std::index_sequence<I...>) noexcept {
  constexpr auto order = [] {
    constexpr std::array ids{std::tuple_element_t<I, Tuple>::key_type::identity...};
    std::array<std::size_t, sizeof...(I)> output{I...};
    std::sort(output.begin(), output.end(),
              [&](const std::size_t left, const std::size_t right) {
                return ids[left] < ids[right];
              });
    return output;
  }();
  return std::tuple{std::move(std::get<order[I]>(requests))...};
}

template <class... Requests> class ObjectCommunication final {
  using requests_type = std::tuple<Requests...>;
  using core_type = Communication<typename Requests::binding_type...>;
  static constexpr std::size_t count = sizeof...(Requests);

  // Agreed before any per-request collective, so ranks with different
  // catalogs never enter mismatched plan preparation.
  static constexpr std::uint64_t catalog_identity = [] {
    std::array<std::uint64_t, 1 + 5 * count> words{count};
    std::size_t offset = 1;
    const auto append = [&]<class Request>() {
      using Lowering =
          CommunicationEntryLowering<typename Request::binding_type>;
      words[offset++] = Request::key_type::identity;
      words[offset++] = static_cast<std::uint64_t>(Request::binding_type::family);
      words[offset++] = static_cast<std::uint64_t>(Lowering::lowering_identity);
      words[offset++] = Request::preparation_shape;
      words[offset++] = Request::collective_preparation;
    };
    (append.template operator()<Requests>(), ...);
    return lpfx::detail::hash_words(words);
  }();

  // Local plans need one agreement at the end of a contiguous run. A run
  // stops on its first local failure; no rank enters a collective request
  // until that failure has been agreed.
  template <class Setup, class Plans>
  static ErrorCode prepare_plans(const Setup &setup, requests_type &requests,
                                 Plans &plans) noexcept {
    ErrorCode local = ErrorCode::success;
    return meta::for_each_until_error<count>([&](const auto index) noexcept {
      constexpr std::size_t I = decltype(index)::value;
      using Request = std::tuple_element_t<I, requests_type>;
      if (local == ErrorCode::success) {
        local = std::get<I>(requests).prepare_plan(setup, std::get<I>(plans));
      }
      if constexpr (I + 1 == count || Request::collective_preparation ||
                    std::tuple_element_t<std::min(I + 1, count - 1),
                                         requests_type>::collective_preparation) {
        return ObjectPreparation<Setup>::agree_error(setup, local);
      } else {
        return ErrorCode::success;
      }
    });
  }

  template <class Setup, class Plans>
  ErrorCode reserve_requests(const Setup &setup, requests_type &requests,
                             Plans &plans) noexcept {
    return meta::for_each_until_error<count>([&](const auto index) noexcept {
      constexpr std::size_t I = decltype(index)::value;
      using Request = std::tuple_element_t<I, requests_type>;
      auto &request = std::get<I>(requests);
      auto &plan = std::get<I>(plans);
      auto &program =
          core_.template program<typename Request::definition_type>();
      if constexpr (requires {
                      request.reserve(program, operation_budget_,
                                      core_.session_, setup, plan);
                    }) {
        return request.reserve(program, operation_budget_, core_.session_,
                               setup, plan);
      } else {
        return ErrorCode::success;
      }
    });
  }

  [[nodiscard]] static bool try_seal(HostMemoryBudget &budget,
                                     const std::size_t limit) noexcept {
    return budget.sealed() ? budget.limit() == limit
                           : budget.try_set_limit(limit) && budget.seal();
  }

public:
  ObjectCommunication() noexcept = default;
  LPFX_NONMOVABLE(ObjectCommunication);

  template <field::fixed_string Name, std::uint64_t... Indices>
  [[nodiscard]] decltype(auto) get() noexcept {
    using Key = operation_key<Name, Indices...>;
    static_assert(meta::contains<Key, typename Requests::key_type...>,
                  "Unknown communication key");
    using Request = std::tuple_element_t<
        meta::index_of<Key, typename Requests::key_type...>, requests_type>;
    return core_.template get<typename Request::definition_type>();
  }
  template <field::fixed_string... Names>
  [[nodiscard]] auto select() noexcept {
    return std::tie(get<Names>()...);
  }
  [[nodiscard]] bool is_valid() const noexcept { return core_.is_valid(); }
  [[nodiscard]] bool empty() const noexcept { return core_.empty(); }
  [[nodiscard]] Status close() noexcept { return core_.close(); }

  template <class Setup>
  [[nodiscard]] ErrorCode
  prepare(const Setup &setup, requests_type &&requests,
          const std::size_t registration_limit =
              std::numeric_limits<std::size_t>::max(),
          const std::size_t operation_limit =
              std::numeric_limits<std::size_t>::max()) noexcept {
    LPFX_TRY(ObjectPreparation<Setup>::agree_words(
        setup, std::array<std::uint64_t, 1>{catalog_identity}));
    // A recoverable rollback may be retried with the same sealed budget
    // policy. A live or closed core cannot be re-prepared.
    const bool budgets = try_seal(registration_budget_, registration_limit) &&
                         try_seal(operation_budget_, operation_limit);
    LPFX_TRY(ObjectPreparation<Setup>::agree_error(
        setup, budgets && core_.can_prepare() ? ErrorCode::success
                                              : ErrorCode::invalid_state));
    std::tuple<typename Requests::plan_type...> plans{};
    LPFX_TRY(prepare_plans(setup, requests, plans));
    return [&]<std::size_t... I>(std::index_sequence<I...>) {
      return core_.prepare(
          setup, registration_budget_, operation_budget_,
          [&]() noexcept { return reserve_requests(setup, requests, plans); },
          std::get<I>(requests).binding(std::get<I>(plans))...);
    }(std::make_index_sequence<count>{});
  }

private:
  HostMemoryBudget registration_budget_, operation_budget_;
  core_type core_;
};

template <class Tuple> struct CommunicationFor;
template <class... Requests> struct CommunicationFor<std::tuple<Requests...>> {
  using type = ObjectCommunication<Requests...>;
};

} // namespace detail

template <class... Requests>
[[nodiscard]] constexpr auto operations(Requests &&...requests) noexcept {
  auto tuple =
      std::tuple_cat(detail::request_tuple(std::forward<Requests>(requests))...);
  return detail::canonical_requests(
      std::move(tuple),
      std::make_index_sequence<std::tuple_size_v<decltype(tuple)>>{});
}

template <class Specification>
using communication_for = typename detail::CommunicationFor<
    std::remove_cvref_t<Specification>>::type;

template <class Specification>
[[nodiscard]] auto make_communication(const Specification &) noexcept
    -> communication_for<Specification> {
  return {};
}

template <class Communication, class Setup, class Specification>
[[nodiscard]] ErrorCode prepare(Communication &communication,
                                const Setup &setup,
                                Specification &&specification) noexcept {
  return communication.prepare(setup,
                               std::forward<Specification>(specification));
}

template <class Communication, class Setup, class Specification>
[[nodiscard]] ErrorCode prepare(Communication &communication,
                                const Setup &setup,
                                Specification &&specification,
                                const PreparationLimits limits) noexcept {
  return communication.prepare(setup,
                               std::forward<Specification>(specification),
                               limits.registration, limits.operations);
}

} // namespace lpfx::distributed
#endif
