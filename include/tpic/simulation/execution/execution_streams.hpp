#ifndef TPIC_SIMULATION_EXECUTION_EXECUTION_STREAMS_HPP
#define TPIC_SIMULATION_EXECUTION_EXECUTION_STREAMS_HPP

#include <lpfx/backend/error_reporting.hpp>
#include <lpfx/backend/handles.hpp>
#include <lpfx/execution/device/context.hpp>

#include <tpic/config/limits.hpp>
#include <tpic/simulation/execution/execution_layout.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <string_view>
#include <utility>

namespace tpic::simulation {

// The application owns role registration; LPFx owns native stream admission.
// Construct and fill on setup's device before publishing to workers; seal()
// then forbids further registration. Borrowed owners must outlive this table,
// and the caller must retire all work before destroying owned streams.
template <class Backend,
          std::size_t Capacity = ExecutionLayout::streams(config::max_real_species, true, true)>
class ExecutionStreams {
  using Traits = lpfx::device::BackendTraits<Backend>;
  using Stream = lpfx::device::DeviceStream<Backend>;

public:
  using Executor = lpfx::device::Executor<Backend>;
  static constexpr bool high_priority_supported = lpfx::device::DeviceStreamPriorityBackend<Backend>;

  struct Registration {
    Executor executor{Traits::null_stream};
    // Labels are static role names owned by the application, not per-job text.
    std::string_view role{};
    bool owned{};
    bool high_priority{};
  };

  explicit ExecutionStreams(std::int32_t device_ordinal) : device_ordinal_(device_ordinal) {
    lpfx::require(device_ordinal >= 0, "TPIC execution: invalid device ordinal");
  }
  ExecutionStreams(const ExecutionStreams&) = delete;
  ExecutionStreams& operator=(const ExecutionStreams&) = delete;

  // Borrow only application-owned streams whose creation device is known to
  // be device_ordinal. Newer runtimes independently verify stream affinity;
  // older supported runtimes verify the current device under that provenance
  // contract. Both always inspect flags and reject every special default.
  [[nodiscard]] static Executor validate_borrowed(Executor executor, std::int32_t device_ordinal) {
    const auto error = lpfx::device::try_validate_explicit_executor(executor, device_ordinal);
    lpfx::require(error != lpfx::ErrorCode::backend_failure,
                  "TPIC execution: native flags/affinity inspection failed");
    lpfx::check_error(
        error,
        "TPIC execution: executor requires an explicit nonblocking stream on the selected device");
    return executor;
  }

  [[nodiscard]] Executor admit(Executor executor, std::string_view role) {
    lpfx::require(!sealed_, "TPIC execution: registry is already sealed");
    lpfx::require(!role.empty(), "TPIC execution: role name is required");
    (void)validate_borrowed(executor, device_ordinal_);
    for (std::size_t index = 0; index < size_; ++index)
      if (entries_[index].executor.stream() == executor.stream()) return executor;
    lpfx::require(size_ < Capacity, "TPIC execution: registry capacity exhausted");
    entries_[size_++] = {executor, role, false, false};
    return executor;
  }

  [[nodiscard]] Executor create(std::string_view role, bool high_priority = false) {
    lpfx::require(!sealed_, "TPIC execution: registry is already sealed");
    lpfx::require(!role.empty(), "TPIC execution: role name is required");
    lpfx::require(size_ < Capacity && owned_size_ < Capacity,
                  "TPIC execution: registry capacity exhausted");
    Stream stream;
    const auto error = [&] {
      if constexpr (high_priority_supported) {
        if (high_priority)
          return Stream::template create<lpfx::device::StreamPriority::high>(stream);
      }
      return Stream::create(stream);
    }();
    lpfx::device::check_device_error<Backend>(error, "TPIC execution: device stream creation failed");
    const auto executor = validate_borrowed(stream.executor(), device_ordinal_);
    for (std::size_t index = 0; index < size_; ++index)
      lpfx::require(entries_[index].executor.stream() != executor.stream(),
                    "TPIC execution: owned role stream aliases an existing stream");
    owned_[owned_size_++] = std::move(stream);
    entries_[size_++] = {executor, role, true,
                        high_priority && high_priority_supported};
    return executor;
  }

  void seal() noexcept { sealed_ = true; }
  [[nodiscard]] std::size_t size() const noexcept { return size_; }
  [[nodiscard]] const Registration& registration(std::size_t index) const {
    lpfx::require(index < size_, "TPIC execution: invalid registration index");
    return entries_[index];
  }

private:
  std::int32_t device_ordinal_;
  std::array<Stream, Capacity> owned_{};
  std::array<Registration, Capacity> entries_{};
  std::size_t size_{};
  std::size_t owned_size_{};
  bool sealed_{};
};

// The borrowed control executor and the executors created for the run, in role
// order. Construction order is creation order: the snapshot-only roles alias the
// diagnostics and control streams while snapshot output is inactive; sort
// aliases local when unsorted.
// Ns is the compiled species capacity: only the first species entries of local
// and sort own a stream, the others stay unbound.
template <class Backend, std::size_t Ns> struct StreamExecutors {
  using Executor = lpfx::device::Executor<Backend>;

  StreamExecutors(ExecutionStreams<Backend>& streams, Executor control, bool snapshot_output,
                  bool sorted, std::size_t species)
      : StreamExecutors(streams, control, snapshot_output, sorted, species,
                        std::make_index_sequence<Ns>{}) {}
  StreamExecutors(const StreamExecutors&) = delete;
  StreamExecutors& operator=(const StreamExecutors&) = delete;

  Executor control, current, mass, diagnostics, diagnostics_helper, solver, solver_interior,
      solver_mass, route, move, field_update, output_capture, output_transfer;
  std::array<Executor, Ns> local, sort;

private:
  template <std::size_t... Index>
  StreamExecutors(ExecutionStreams<Backend>& streams, Executor control, bool snapshot_output,
                  bool sorted, std::size_t species, std::index_sequence<Index...>)
      : control(control), current(streams.create("current")),
        mass(streams.create("mass")),
        diagnostics(streams.create("diagnostics")),
        diagnostics_helper(streams.create("diagnostics_helper")),
        solver(streams.create("solver")),
        solver_interior(streams.create("solver_interior")),
        solver_mass(streams.create("solver_mass")),
        route(streams.create("route", true)), move(streams.create("move")),
        field_update(streams.create("field_update")),
        output_capture(snapshot_output ? streams.create("output_capture")
                                       : diagnostics),
        output_transfer(snapshot_output ? streams.create("output_transfer")
                                        : control),
        local{(Index < species ? streams.create("local") : unbound())...},
        sort{(!sorted || Index >= species ? local[Index] : streams.create("sort"))...} {}
  static Executor unbound() noexcept {
    return Executor{lpfx::device::BackendTraits<Backend>::null_stream};
  }
};

} // namespace tpic::simulation

#endif // TPIC_SIMULATION_EXECUTION_EXECUTION_STREAMS_HPP
