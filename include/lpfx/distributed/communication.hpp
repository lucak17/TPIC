#ifndef LPFX_DISTRIBUTED_COMMUNICATION_HPP
#define LPFX_DISTRIBUTED_COMMUNICATION_HPP

#include <algorithm>
#include <array>
#include <climits>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <tuple>
#include <type_traits>
#include <utility>

#include <lpfx/core/attributes.hpp>
#include <lpfx/core/error.hpp>
#include <lpfx/core/meta.hpp>
#include <lpfx/distributed/composition/entry.hpp>
#include <lpfx/distributed/detail/integer_codec.hpp>
#include <lpfx/distributed/operation/prepared_operation.hpp>
#include <lpfx/memory/host/budget.hpp>
#include <lpfx/memory/space.hpp>

namespace lpfx::distributed {

class CommunicationSetup;

namespace detail {

template <class... Requests> class ObjectCommunication;

template <class Binding>
using lowering_program_t =
    typename CommunicationEntryLowering<Binding>::program_type;

template <class Space> struct SpaceDeviceBackend final {
  using type = void;
};
template <class Backend> struct SpaceDeviceBackend<DeviceSpace<Backend>> final {
  using type = Backend;
};

template <class... Types> struct FirstNonVoid final {
  using type = void;
};
template <class First, class... Rest> struct FirstNonVoid<First, Rest...> final {
  using type = std::conditional_t<std::is_void_v<First>,
                                  typename FirstNonVoid<Rest...>::type, First>;
};

// Compile-time catalog of a communication's bindings: canonical (channel
// identity) order, tag bases, native resources and the identity every rank
// agrees before registration.
template <class... Bindings> struct CommunicationCatalog final {
  static constexpr std::size_t count = sizeof...(Bindings);
  static constexpr std::array<std::uint64_t, count> channels{
      Bindings::definition_type::channel_identity...};

  // order[k] is the binding index at canonical position k.
  static constexpr std::array<std::size_t, count> order = [] {
    std::array<std::size_t, count> output{};
    for (std::size_t index = 0; index != count; ++index) {
      output[index] = index;
    }
    std::sort(output.begin(), output.end(),
              [](const std::size_t left, const std::size_t right) {
                return channels[left] < channels[right];
              });
    return output;
  }();

  static_assert(count != 0, "a communication needs at least one binding");
  static_assert(
      [] {
        for (std::size_t index = 0; index != count; ++index) {
          if (channels[order[index]] == 0 ||
              (index != 0 &&
               channels[order[index]] == channels[order[index - 1]])) {
            return false;
          }
        }
        return true;
      }(),
      "channel identities must be nonzero and unique");

  static constexpr std::array<std::size_t, count> tag_extents{
      CommunicationEntryLowering<Bindings>::tag_extent...};
  static constexpr std::size_t tag_extent = [] {
    std::size_t total = 0;
    for (const std::size_t extent : tag_extents) {
      total += extent;
    }
    return total;
  }();
  static_assert(tag_extent <= std::size_t{INT_MAX} + 1,
                "communication tags exceed the int tag range");

  // Bases are assigned in canonical order; a binding without tags gets 0.
  static constexpr std::array<int, count> tag_base = [] {
    std::array<int, count> output{};
    std::size_t next = 0;
    for (const std::size_t index : order) {
      if (tag_extents[index] != 0) {
        output[index] = static_cast<int>(next);
      }
      next += tag_extents[index];
    }
    return output;
  }();

  using device_backend_type = typename FirstNonVoid<
      typename SpaceDeviceBackend<typename Bindings::space_type>::type...>::type;
  static_assert(
      ((std::is_void_v<
            typename SpaceDeviceBackend<typename Bindings::space_type>::type> ||
        std::same_as<
            typename SpaceDeviceBackend<typename Bindings::space_type>::type,
            device_backend_type>) &&
       ...),
      "all device bindings must use one device backend");

  using xccl_native_type = typename FirstNonVoid<
      typename CommunicationEntryLowering<Bindings>::xccl_native_type...>::type;
  static_assert(
      ((std::is_void_v<
            typename CommunicationEntryLowering<Bindings>::xccl_native_type> ||
        std::same_as<
            typename CommunicationEntryLowering<Bindings>::xccl_native_type,
            xccl_native_type>) &&
       ...),
      "a communication uses at most one XCCL native provider");

  static constexpr bool mpi_payload =
      (CommunicationEntryLowering<Bindings>::mpi_payload || ...);

  // Hash of the canonical entry records: channel, family, lowering, device
  // runtime and tag extent.
  static constexpr std::uint64_t identity = [] {
    constexpr std::array<std::array<std::uint64_t, 5>, count> records{
        std::array<std::uint64_t, 5>{
            Bindings::definition_type::channel_identity,
            static_cast<std::uint64_t>(Bindings::family),
            static_cast<std::uint64_t>(
                CommunicationEntryLowering<Bindings>::lowering_identity),
            CommunicationDeviceRuntimeIdentity<typename SpaceDeviceBackend<
                typename Bindings::space_type>::type>::value,
            CommunicationEntryLowering<Bindings>::tag_extent}...};
    std::array<std::uint64_t, 2 + 5 * count> words{UINT64_C(0x4c50465843303036),
                                                   count};
    for (std::size_t position = 0; position != count; ++position) {
      for (std::size_t word = 0; word != 5; ++word) {
        words[2 + 5 * position + word] = records[order[position]][word];
      }
    }
    return lpfx::detail::hash_words(words);
  }();

  template <class Definition>
    requires meta::contains<Definition, typename Bindings::definition_type...>
  static constexpr std::size_t index =
      meta::index_of<Definition, typename Bindings::definition_type...>;
};

} // namespace detail

// Prepared operations of one set of bindings under one native session. Every
// rank prepares, uses and closes the same set in the same order; get<>() is a
// compile-time lookup by definition type.
template <class... Bindings> class Communication final {
  using catalog = detail::CommunicationCatalog<Bindings...>;
  template <std::size_t I>
  using binding_t = std::tuple_element_t<I, std::tuple<Bindings...>>;
  template <std::size_t I>
  using program_t = detail::lowering_program_t<binding_t<I>>;
  template <std::size_t I>
  using lowering_t = detail::CommunicationEntryLowering<binding_t<I>>;

public:
  using session_type =
      detail::CommunicationSession<catalog::mpi_payload,
                                   typename catalog::xccl_native_type>;
  static constexpr std::size_t operation_count = catalog::count;

  constexpr Communication() noexcept = default;
  LPFX_NONMOVABLE(Communication);

  ~Communication() noexcept {
    // A destructor cannot invent the matching collective close on peers.
    if (phase_ == Phase::prepared) {
      std::abort();
    }
  }

  [[nodiscard]] bool empty() const noexcept {
    bool result = phase_ != Phase::prepared && session_.empty();
    meta::for_each_index<operation_count>([&](const auto index) noexcept {
      result = result && std::get<decltype(index)::value>(operations_).empty();
    });
    return result;
  }

  [[nodiscard]] bool is_valid() const noexcept {
    bool result = phase_ == Phase::prepared && session_.is_valid();
    meta::for_each_index<operation_count>([&](const auto index) noexcept {
      result =
          result && std::get<decltype(index)::value>(operations_).is_valid();
    });
    return result;
  }

  template <class Definition> [[nodiscard]] auto &get() noexcept {
    return std::get<catalog::template index<Definition>>(operations_);
  }
  template <class Definition> [[nodiscard]] const auto &get() const noexcept {
    return std::get<catalog::template index<Definition>>(operations_);
  }

  template <class Definition, class Capacity>
  [[nodiscard]] ErrorCode
  reserve_particles(const CommunicationSetup &setup,
                    HostMemoryBudget &operation_budget,
                    const Capacity &requested) noexcept {
    return reserve_particles<Definition>(setup, operation_budget, requested,
                                         requested.maximum_final());
  }

  template <class Definition, class Capacity>
  [[nodiscard]] ErrorCode reserve_particles(
      const CommunicationSetup &setup, HostMemoryBudget &operation_budget,
      const Capacity &requested, const std::uint64_t arrival_records) noexcept {
    if (!is_valid()) {
      return ErrorCode::invalid_state;
    }
    LPFX_TRY(session_.agree(quiescent() && get<Definition>().reusable()
                                ? ErrorCode::success
                                : ErrorCode::invalid_state));
    return program<Definition>().reserve(operation_budget, session_, setup,
                                         requested, arrival_records);
  }

  [[nodiscard]] Status close() noexcept {
    if (phase_ != Phase::prepared) {
      return empty() ? Status::success()
                     : Status::failure(ErrorCode::invalid_state);
    }
    if (!is_valid()) {
      return Status::failure(ErrorCode::invalid_state);
    }
    const ErrorCode agreed = session_.agree(
        quiescent() ? ErrorCode::success : ErrorCode::invalid_state);
    if (agreed != ErrorCode::success) {
      return Status::failure(agreed);
    }
    meta::reverse_for_each<operation_count>([&](const auto index) noexcept {
      std::get<decltype(index)::value>(operations_).release_for_close();
    });
    session_.close();
    phase_ = Phase::closed;
    return Status::success();
  }

private:
  enum class Phase : std::uint8_t { empty = 0, prepared = 1, closed = 2 };

  template <class... Requests> friend class detail::ObjectCommunication;
  template <class... Other>
  friend ErrorCode try_prepare_communication(Communication<Other...> &,
                                             const CommunicationSetup &,
                                             HostMemoryBudget &,
                                             HostMemoryBudget &,
                                             Other &&...) noexcept;

  [[nodiscard]] bool can_prepare() const noexcept {
    return phase_ == Phase::empty && empty();
  }

  [[nodiscard]] bool quiescent() const noexcept {
    bool result = true;
    meta::for_each_index<operation_count>([&](const auto index) noexcept {
      result = result && std::get<decltype(index)::value>(operations_)
                             .quiescent_for_close();
    });
    return result;
  }

  template <class Definition> [[nodiscard]] auto &program() noexcept {
    return get<Definition>().program_;
  }

  // Validation, registration and session activation, then one agreed
  // preparation per operation in binding order. before_publication runs after
  // every operation is prepared; any failure rolls the whole set back.
  // The setup is the concrete type: a caller-side wrapper (for example one
  // counting agreements) converts to it here.
  template <class BeforePublication>
  [[nodiscard]] ErrorCode
  prepare(const CommunicationSetup &setup, HostMemoryBudget &registration_budget,
          HostMemoryBudget &operation_budget,
          BeforePublication &&before_publication,
          Bindings &&...bindings) noexcept {
    if (!can_prepare()) {
      return ErrorCode::invalid_state;
    }
    const std::tuple<Bindings *...> inputs{&bindings...};
    detail::CommunicationRequirements requirements{};
    ErrorCode local = registration_budget.sealed() && operation_budget.sealed()
                          ? ErrorCode::success
                          : ErrorCode::invalid_state;
    if (local == ErrorCode::success) {
      local = meta::for_each_until_error<operation_count>(
          [&](const auto index) noexcept {
            constexpr std::size_t I = decltype(index)::value;
            const auto &binding = *std::get<I>(inputs);
            LPFX_TRY(program_t<I>::try_validate(setup, binding));
            if (program_t<I>::needs_native_work(binding)) {
              requirements.mpi_payload =
                  requirements.mpi_payload || lowering_t<I>::mpi_payload;
              requirements.xccl =
                  requirements.xccl ||
                  !std::is_void_v<typename lowering_t<I>::xccl_native_type>;
            }
            return ErrorCode::success;
          });
    }
    LPFX_TRY(session_.try_open(
        setup, registration_budget, local, catalog::identity,
        catalog::tag_extent, requirements,
        registration_plans(inputs, std::make_index_sequence<operation_count>{})));

    std::size_t prepared = 0;
    ErrorCode error = meta::for_each_until_error<operation_count>(
        [&](const auto index) noexcept {
          constexpr std::size_t I = decltype(index)::value;
          auto &operation = std::get<I>(operations_);
          const ErrorCode agreed =
              session_.agree(operation.program_.try_prepare(
                  operation_budget, session_, setup,
                  std::move(*std::get<I>(inputs)), catalog::tag_base[I]));
          if (agreed != ErrorCode::success) {
            operation.reset_unpublished();
            return agreed;
          }
          if (!operation.ready_to_publish()) {
            std::abort();
          }
          ++prepared;
          return ErrorCode::success;
        });
    if (error == ErrorCode::success) {
      error = before_publication();
    }
    if (error != ErrorCode::success) {
      meta::reverse_for_each<operation_count>([&](const auto index) noexcept {
        constexpr std::size_t I = decltype(index)::value;
        if (I < prepared) {
          std::get<I>(operations_).reset_unpublished();
        }
      });
      session_.rollback();
      return error;
    }
    meta::for_each_index<operation_count>([&](const auto index) noexcept {
      std::get<decltype(index)::value>(operations_).publish();
    });
    phase_ = Phase::prepared;
    return ErrorCode::success;
  }

  // Borrowed plans in canonical channel order.
  template <std::size_t... Position>
  [[nodiscard]] static auto
  registration_plans(const std::tuple<Bindings *...> &inputs,
                     std::index_sequence<Position...>) noexcept {
    return std::tuple{&std::get<catalog::order[Position]>(inputs)->plan()...};
  }

  session_type session_{};
  std::tuple<PreparedOperation<detail::lowering_program_t<Bindings>>...>
      operations_{};
  Phase phase_ = Phase::empty;
};

// Bindings are consumed; every rank passes the same binding types.
template <class... Bindings>
[[nodiscard]] ErrorCode try_prepare_communication(
    Communication<Bindings...> &output, const CommunicationSetup &setup,
    HostMemoryBudget &registration_budget, HostMemoryBudget &operation_budget,
    Bindings &&...bindings) noexcept {
  return output.prepare(
      setup, registration_budget, operation_budget,
      []() noexcept { return ErrorCode::success; }, std::move(bindings)...);
}

} // namespace lpfx::distributed

#endif
