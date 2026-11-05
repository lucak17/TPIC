#ifndef TPIC_ECSIM_MAXWELL_MAXWELL_DEVICE_HPP
#define TPIC_ECSIM_MAXWELL_MAXWELL_DEVICE_HPP

#include <lpfx/backend/cuda_hip/kernel.hpp>
#include <lpfx/backend/error_reporting.hpp>
#include <lpfx/profiling.hpp>

#include <tpic/config/precision.hpp>
#include <tpic/ecsim/maxwell/maxwell.hpp>

#include <tpic/ecsim/fields/cell_to_node_device.hpp>
#include <tpic/ecsim/fields/field_boundary_device.hpp>
#include <tpic/ecsim/fields/field_vector_device.hpp>
#include <tpic/ecsim/mass_matrix/mass_matrix_apply_device.hpp>
#include <tpic/ecsim/maxwell/field_update_device.hpp>
#include <tpic/ecsim/maxwell/maxwell_apply_policy.hpp>
#include <tpic/ecsim/fields/node_cell_curl_device.hpp>
#include <tpic/ecsim/fields/node_smoothing_device.hpp>
#include <tpic/data/fields/field_storage.hpp>
#include <tpic/communication/fields/field_exchange.hpp>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <numbers>

namespace tpic::ecsim {

// Device owners of the solver field schemas.
template <class Precision, class Backend>
using SolverNodeOwner = data::DeviceFieldOwner<typename SolverFieldSchema<Precision>::Node, Backend>;
template <class Precision, class Backend>
using SolverCellOwner = data::DeviceFieldOwner<typename SolverFieldSchema<Precision>::Cell, Backend>;

// All owners and their bound communication plans are persistent and address
// stable. This operator implements periodic and closed-wall ECSIM, including
// symmetric smoothing on periodic and closed-wall domains.
// Keeping next B-cell/B-node separate makes the particle-facing old epoch
// explicit even after the field update has been enqueued.
template <class Precision, class Backend> class MaxwellOperator {
  using Schema = SolverFieldSchema<Precision>;
  using Fields = data::FieldSchema<Precision>;
  using Executor = lpfx::device::Executor<Backend>;
  using Traits = lpfx::device::BackendTraits<Backend>;
  using Event = lpfx::device::DeviceEvent<Backend>;

public:
  // Compute types the right-hand side, the field update and the Arnoldi
  // applies; Residual types the apply that evaluates the solver residual.
  using Compute = typename Precision::Compute::Electric;
  using Residual = typename Precision::Compute::SolverReduction;
  using MagneticCompute = typename Precision::Compute::Magnetic;

  SolverNodeOwner<Precision, Backend> theta, rhs, input, image, mass_image, curl_node, smooth_a,
      smooth_b;
  SolverCellOwner<Precision, Backend> curl_cell;
  data::MagneticCellOwner<Precision, Backend> next_cell;
  data::MagneticNodeOwner<Precision, Backend> next_node;

  // Helper executors must share the selected device. Application ownership
  // keeps all streams alive until the final main-stream reader is complete.
  // Aliased executors retain the same dependency graph with FIFO execution.
  MaxwellOperator(MaxwellParameters prepared, Executor executor,
                  Executor mass_executor, Executor interior_executor)
      : theta(data::make_zero_field<typename Schema::Node>(prepared.domain, executor)),
        rhs(data::make_zero_field<typename Schema::Node>(prepared.domain, executor)),
        input(data::make_zero_field<typename Schema::Node>(prepared.domain, executor)),
        image(data::make_zero_field<typename Schema::Node>(prepared.domain, executor)),
        mass_image(data::make_zero_field<typename Schema::Node>(prepared.domain, executor)),
        curl_node(data::make_zero_field<typename Schema::Node>(prepared.domain, executor)),
        smooth_a(data::make_zero_field<typename Schema::Node>(prepared.domain, executor)),
        smooth_b(data::make_zero_field<typename Schema::Node>(prepared.domain, executor)),
        curl_cell(data::make_zero_field<typename Schema::Cell>(prepared.domain, executor)),
        next_cell(data::make_zero_field<typename Fields::BCell>(prepared.domain, executor)),
        next_node(data::make_zero_field<typename Fields::BNode>(prepared.domain, executor)),
        periodic_(std::all_of(prepared.physics.field_boundary.begin(), prepared.physics.field_boundary.end(),
            [](auto face) { return face == config::FieldBoundary::periodic; })),
        plans_(prepare_field_operator_plans(prepared.domain, prepared.physics)),
        executor_(executor),
        mass_executor_(mass_executor), interior_executor_(interior_executor),
        constants_(make_constants<Compute>(prepared)),
        residual_constants_(make_constants<Residual>(prepared)),
        smoothing_enabled_(prepared.physics.smoothing_enabled),
        smooth_iterations_(prepared.smooth_iterations),
        smooth_operator_(prepared.smooth_operator), theta_value_(config::convert<Compute>(prepared.theta)),
        faraday_(config::convert<MagneticCompute>(-prepared.c * prepared.dt)) {
    lpfx::check_error(try_make_node_cell_spacing(prepared.domain.spacing, magnetic_spacing_),
                      "TPIC Maxwell: invalid spacing");
    lpfx::require(std::isfinite(prepared.domain.inverse_cell_volume) &&
                      prepared.domain.inverse_cell_volume > 0 &&
                      std::isfinite(constants_.inverse_volume) && constants_.inverse_volume > 0,
                  "TPIC Maxwell: invalid inverse cell volume");
    lpfx::require(mass_executor_.bound(), "TPIC Maxwell: invalid mass helper executor");
    lpfx::require(interior_executor_.bound(), "TPIC Maxwell: invalid interior helper executor");
    if (!smoothing_enabled_) {
      lpfx::device::create_event(input_halo_ready_event_,
                           "TPIC Maxwell: mass dependency event creation failed");
      lpfx::device::create_event(mass_ready_event_, "TPIC Maxwell: mass dependency event creation failed");
    }
    // Even an empty core records completion, and aliased executors use the same
    // compiled schedule.
    lpfx::device::create_event(input_local_ready_event_, "TPIC Maxwell: curl dependency event creation failed");
    lpfx::device::create_event(curl_cell_ready_event_, "TPIC Maxwell: curl dependency event creation failed");
    lpfx::device::create_event(curl_interior_ready_event_,
                         "TPIC Maxwell: curl dependency event creation failed");
  }

  MaxwellOperator(const MaxwellOperator&) = delete;
  MaxwellOperator& operator=(const MaxwellOperator&) = delete;

  // Trial vectors are projected before halo packing, so the exchange sends
  // projected values. All ranks call the exchange even when only boundary ranks project.
  template <class Owner, class Exchange>
  void complete_electric_refresh(Owner& field, Exchange& exchange) {
    project(field);
    communication::complete_field_exchange(exchange,
        "TPIC Maxwell: halo exchange rejected", "TPIC Maxwell: halo exchange failed");
    lpfx::check_error(enqueue_wall_ghosts(field, plans_.boundary, executor_),
                      "TPIC Maxwell: electric ghost closure failed");
    project(field);
  }

  template <class Owner> void close_magnetic_ghosts(Owner& field) {
    lpfx::check_error(enqueue_wall_ghosts(field, plans_.boundary, executor_),
                      "TPIC Maxwell: magnetic/curl ghost closure failed");
  }
  [[nodiscard]] const FieldBoundaryPlans& boundary_plans() const noexcept { return plans_.boundary; }

  // This J-independent prefix can run while current and mass deposition are
  // still in flight. It touches only old-field interiors, B ghosts, curl_node
  // and theta, and performs no communication. Preserve electric and curl_node
  // until complete_rhs; apply and the field update may not reuse the scratch yet.
  template <class E, class B>
  void enqueue_rhs_fields(const E& electric, B& magnetic) {
    // Startup deliberately retains analytic B ghosts and the corresponding old
    // B-node particle epoch. Close B-cell here for the first RHS; subsequent
    // steps already copied the completed next-cell halo into this stable owner.
    close_magnetic_ghosts(magnetic);
    lpfx::check_error(enqueue_node_cell_curl<Precision>(magnetic, curl_node, plans_.nodes,
                                                        constants_.spacing, executor_),
                      "TPIC Maxwell: RHS curl failed");
    lpfx::check_error(enqueue_field_vector_copy(theta, electric, false, nullptr, executor_),
                      "TPIC Maxwell: theta initial copy failed");
    project(theta);
  }

  // Call after enqueue_rhs_fields and current assembly on this executor.
  // With smoothing, MPI blocks for each halo refresh; NCCL/RCCL return at enqueue.
  // R1: E + delta*curl(B) - current_factor*response stays one expression with
  // its validation; an early partial sum would add a rounding point.
  template <class E, class J, class SmoothAExchange, class SmoothBExchange>
  void complete_rhs(const E& electric, const J& current,
      SmoothAExchange& exchange_a, SmoothBExchange& exchange_b, unsigned* invalid) {
    const auto combine = [&](const auto& response) {
      lpfx::check_error(enqueue_field_vector_linear_combination<Precision>(rhs, invalid, executor_,
          scaled(Compute(1), electric), scaled(constants_.delta, curl_node),
          scaled(-constants_.current_factor, response)),
          "TPIC Maxwell: right-hand side combination failed");
    };
    // Raw Jhat has Neumann physical ghosts. A solver-precision copy must acquire
    // E-like closure before S(Jhat). Use a rank-uniform condition to keep the
    // scratch exchange schedule identical on interior and boundary ranks.
    if (smoothing_enabled_)
      combine(complete_smoothing(current, exchange_a, exchange_b, periodic_, false, invalid));
    else
      combine(current);
    // Solve in the closed-wall constrained subspace: project the E^n guess as
    // well as the RHS. Arnoldi then retains exact zeros on constrained rows,
    // and final theta closure is idempotent on the true-residual-checked solve.
    project(rhs);
  }

  // Schedule must match the immutable smoothing configuration. Startup chooses
  // one typed solve entry before GMRES; no schedule dispatch occurs per matvec.
  // Input exchange must preserve every interior node (including shared copies).
  // Arithmetic is the type the apply computes in: Compute for the Arnoldi
  // applies, Residual for the apply whose image forms the solver residual.
  // MPI blocks for the input and curl halos and any smoothing refreshes;
  // NCCL/RCCL return at enqueue. host_wait_mass_ghosts(executor) runs before
  // the first kernel that reads the ghost records of mass and orders that
  // kernel's executor after them.
  template <MaxwellApplySchedule Schedule, class Arithmetic = Compute, class InputExchange,
            class CurlExchange, class SmoothAExchange, class SmoothBExchange, class HostWaitMassGhosts>
  void complete_apply(const data::CompactMassMatrix<Precision, Backend>& mass,
      InputExchange& input_exchange, CurlExchange& curl_exchange,
      SmoothAExchange& exchange_a, SmoothBExchange& exchange_b, unsigned* invalid,
      const HostWaitMassGhosts& host_wait_mass_ghosts) {
    static_assert(Schedule == MaxwellApplySchedule::full ||
                  Schedule == MaxwellApplySchedule::split ||
                  Schedule == MaxwellApplySchedule::smoothed);
    LPFX_PROFILE_SCOPE("TPIC::MaxwellApply");
    communication::FieldExchangeEpoch input_epoch(input_exchange,
        "TPIC Maxwell: input halo rejected", "TPIC Maxwell: input halo failed");
    communication::FieldExchangeEpoch curl_epoch(curl_exchange,
        "TPIC Maxwell: curl halo rejected", "TPIC Maxwell: curl halo failed");
    try {
      if constexpr (Schedule == MaxwellApplySchedule::smoothed)
        complete_apply_smoothed<Arithmetic>(mass, input_epoch, curl_epoch, exchange_a, exchange_b, invalid,
                                            host_wait_mass_ghosts);
      else
        complete_apply_unsmoothed<Schedule, Arithmetic>(mass, input_epoch, curl_epoch, invalid,
                                                        host_wait_mass_ghosts);
    } catch (...) {
      // A launch can fail after begin accepted a borrowed communication epoch.
      // Retire it before draining streams or allowing any owner to unwind.
      curl_epoch.complete_or_abort();
      input_epoch.complete_or_abort();
      host_sync_apply_streams<Schedule>();
      throw;
    }
  }

  // Call after the solve. A smoothed particle field is a scratch owner that
  // remains valid until the next smoothing call; finish all species' gathers
  // on this stream before that reuse. Current/next magnetic epochs are separate.
  // With smoothing, MPI blocks for each halo refresh; NCCL/RCCL return at enqueue.
  template <class ThetaExchange, class SmoothAExchange, class SmoothBExchange>
  const SolverNodeOwner<Precision, Backend>& complete_particle_field(ThetaExchange& theta_exchange,
      SmoothAExchange& exchange_a, SmoothBExchange& exchange_b, unsigned* invalid) {
    // Only S(theta) reads the ghosts of theta: unsmoothed, the particle field
    // needs the wall projection alone.
    if (!smoothing_enabled_) {
      project(theta);
      return theta;
    }
    complete_electric_refresh(theta, theta_exchange);
    return complete_smoothing(theta, exchange_a, exchange_b, true, false, invalid);
  }

  // Stream: executor, onto which the caller has ordered the main producer
  // event and the release of the old E source read. Writes E and next_cell
  // (enqueue_next_fields); the current B-node stays untouched, so the movers
  // still gather the old epoch.
  void enqueue_next_e_and_cell(data::DeviceFieldImage<Precision, Backend>& fields,
      unsigned* invalid, Executor executor) {
    enqueue_next_fields<Precision>(fields.electric, fields.magnetic_cell,
        theta, next_cell, magnetic_spacing_, theta_value_, faraday_, invalid, executor);
  }

  // Stream: executor, after the next_cell halo has concluded on it. Closes
  // next_cell's wall ghosts and interpolates it to next_node.
  void enqueue_next_cell_closure_and_node_interpolation(unsigned* invalid, Executor executor) {
    lpfx::check_error(enqueue_wall_ghosts(next_cell, plans_.boundary, executor),
                      "TPIC field update: next magnetic ghost closure failed");
    lpfx::check_error(
        enqueue_magnetic_cell_to_node<Precision>(next_cell, next_node, invalid, executor),
        "TPIC field update: next magnetic C2N failed");
  }

  // R1: every moved_event is recorded and its wait submitted on executor
  // before this call. Deposition may read next B concurrently; the commit only
  // reads it. Copies keep the current owners' addresses and registrations.
  // The copies carry no flag: the cell interiors passed the Faraday check, the
  // cell ghosts and the node interiors the cell-to-node check of the same step.
  // The node copy takes the interiors, the only nodes a gather reads.
  void enqueue_commit_to_current(data::DeviceFieldImage<Precision, Backend>& fields,
      data::MagneticNodeOwner<Precision, Backend>& current_node, Executor executor) {
    lpfx::check_error(enqueue_field_vector_copy(fields.magnetic_cell, next_cell, true,
                                                nullptr, executor),
                      "TPIC field update: magnetic cell commit failed");
    lpfx::check_error(enqueue_field_vector_copy(current_node, next_node, false, nullptr,
                                                executor),
                      "TPIC field update: magnetic node commit failed");
    lpfx::check_error(enqueue_constrained_rows(fields.electric, detail::NoRowInput{},
                          plans_.boundary.constrained_rows, executor),
                      "TPIC field update: committed electric boundary projection failed");
  }

private:
  template <class Arithmetic, class Epoch> void begin_apply_input(Epoch& epoch) {
    project(input);
    lpfx::device::record_event(input_local_ready_event_, executor_,
                         "TPIC Maxwell: local input recording failed");
    epoch.begin();
    lpfx::device::wait_event(input_local_ready_event_, interior_executor_,
                       "TPIC Maxwell: local input dependency failed");
    // All N2C corners are local. Record its completion before any later helper
    // work so curl packing never waits for the interior C2N or mass kernels.
    lpfx::check_error(
        enqueue_node_cell_curl<Precision, StencilRegion::full, Arithmetic>(
            input, curl_cell, plans_.cells, constants<Arithmetic>().spacing, interior_executor_),
        "TPIC Maxwell: asynchronous N2C curl failed");
    lpfx::device::record_event(curl_cell_ready_event_, interior_executor_,
                         "TPIC Maxwell: N2C completion recording failed");
  }

  // Stream: executor_. Also begins the curl halo: before the host waits for an
  // input halo still in flight (MPI), so both share the network; after the input
  // ghost closure and `closed` when it completed at begin (NCCL/RCCL), so M never
  // waits for curl communication. R1: `closed` is recorded before any host wait
  // for curl communication, so a later wait on M sees this apply's record.
  template <class InputEpoch, class CurlEpoch>
  void complete_input_halo(InputEpoch& input_epoch, CurlEpoch& curl_epoch, Event* closed) {
    const auto begin_curl = [&] {
      lpfx::device::wait_event(curl_cell_ready_event_, executor_,
                               "TPIC Maxwell: N2C completion dependency failed");
      curl_epoch.begin();
    };
    const bool in_flight = !input_epoch.progress();
    if (in_flight) {
      begin_curl();
      while (!input_epoch.progress()) static_cast<void>(curl_epoch.progress());
    }
    input_epoch.complete();
    lpfx::check_error(enqueue_wall_ghosts(input, plans_.boundary, executor_),
                      "TPIC Maxwell: electric ghost closure failed");
    // The ghost copy and physical closure preserve all interior nodes. The initial
    // projection is still valid; repeating it would race with early M reads.
    // The final theta projection and the smoothing refreshes are separate.
    if (closed)
      lpfx::device::record_event(*closed, executor_, "TPIC Maxwell: input halo readiness recording failed");
    if (!in_flight) begin_curl();
  }

  // Stream: interior_executor_, after N2C has written curl_cell on it. Writes
  // the strict interior of curl_node; the caller records curl_interior_ready_event_.
  template <class Arithmetic> void enqueue_curl_interior() {
    LPFX_PROFILE_SCOPE("TPIC::MaxwellApply::CurlInterior");
    lpfx::check_error(enqueue_node_cell_curl<Precision, StencilRegion::strict_interior, Arithmetic>(
                          curl_cell, curl_node, plans_.nodes, constants<Arithmetic>().spacing,
                          interior_executor_),
                      "TPIC Maxwell: interior C2N curl failed");
  }

  // Stream: executor_, after the curl_cell exchange has finished. Closes the
  // curl_cell wall ghosts, then writes the shell of curl_node.
  template <class Arithmetic> void enqueue_curl_shell() {
    LPFX_PROFILE_SCOPE("TPIC::MaxwellApply::CurlShell");
    close_magnetic_ghosts(curl_cell);
    lpfx::check_error(enqueue_node_cell_curl<Precision, StencilRegion::shell, Arithmetic>(
                          curl_cell, curl_node, plans_.nodes, constants<Arithmetic>().spacing, executor_),
                      "TPIC Maxwell: shell C2N curl failed");
  }

  // Stream: executor, on which source is ready as Region requires
  // (stencil_region.hpp). Writes Region of mass_image and flags *invalid; the
  // caller records mass completion next.
  template <StencilRegion Region, class Arithmetic, class Source>
  void enqueue_mass_response(const data::CompactMassMatrix<Precision, Backend>& mass,
      const Source& source, unsigned* invalid, Executor executor) {
    [[maybe_unused]] constexpr const char* name = Region == StencilRegion::full ? "TPIC::MaxwellApply::MassFull"
        : Region == StencilRegion::strict_interior ? "TPIC::MaxwellApply::MassInterior"
                                                   : "TPIC::MaxwellApply::MassShell";
    LPFX_PROFILE_SCOPE(name);
    lpfx::check_error(enqueue_mass_matrix_apply<Precision, Region, Arithmetic>(
                          mass, source, mass_image, plans_.nodes, plans_.boundary.node_ghosts.faces,
                          constants<Arithmetic>().inverse_volume, invalid, executor),
                      "TPIC Maxwell: mass application failed");
  }

  template <MaxwellApplySchedule Schedule, class Arithmetic, class InputEpoch, class CurlEpoch,
            class HostWaitMassGhosts>
  void complete_apply_unsmoothed(const data::CompactMassMatrix<Precision, Backend>& mass,
      InputEpoch& input_epoch, CurlEpoch& curl_epoch, unsigned* invalid,
      const HostWaitMassGhosts& host_wait_mass_ghosts) {
    {
      LPFX_PROFILE_SCOPE("TPIC::MaxwellApply::InputHalo");
      begin_apply_input<Arithmetic>(input_epoch);
      // R1: N2C then strict C2N in FIFO order, so both curls hide the first
      // exchange; this precedes M even when the two helper executors alias.
      enqueue_curl_interior<Arithmetic>();
      lpfx::device::record_event(curl_interior_ready_event_, interior_executor_,
                                 "TPIC Maxwell: interior curl completion recording failed");
      if constexpr (Schedule == MaxwellApplySchedule::split) {
        lpfx::device::wait_event(input_local_ready_event_, mass_executor_,
                           "TPIC Maxwell: local mass input dependency failed");
        enqueue_mass_response<StencilRegion::strict_interior, Arithmetic>(mass, input, invalid, mass_executor_);
      }
      complete_input_halo(input_epoch, curl_epoch, &input_halo_ready_event_);
    }
    {
      LPFX_PROFILE_SCOPE("TPIC::MaxwellApply::CurlHalo");
      lpfx::device::wait_event(input_halo_ready_event_, mass_executor_,
                         "TPIC Maxwell: mass halo dependency failed");
      host_wait_mass_ghosts(mass_executor_);
      if constexpr (Schedule == MaxwellApplySchedule::split)
        enqueue_mass_response<StencilRegion::shell, Arithmetic>(mass, input, invalid, mass_executor_);
      else
        enqueue_mass_response<StencilRegion::full, Arithmetic>(mass, input, invalid, mass_executor_);
      lpfx::device::record_event(mass_ready_event_, mass_executor_,
                           "TPIC Maxwell: mass completion recording failed");
      curl_epoch.complete();
    }
    enqueue_curl_shell<Arithmetic>();
    lpfx::device::wait_event(mass_ready_event_, executor_, "TPIC Maxwell: mass completion dependency failed");
    combine_image<Arithmetic>(mass_image);
  }

  template <class Arithmetic, class InputEpoch, class CurlEpoch, class SmoothAExchange,
            class SmoothBExchange, class HostWaitMassGhosts>
  void complete_apply_smoothed(const data::CompactMassMatrix<Precision, Backend>& mass,
      InputEpoch& input_epoch, CurlEpoch& curl_epoch,
      SmoothAExchange& exchange_a, SmoothBExchange& exchange_b, unsigned* invalid,
      const HostWaitMassGhosts& host_wait_mass_ghosts) {
    {
      LPFX_PROFILE_SCOPE("TPIC::MaxwellApply::InputHalo");
      begin_apply_input<Arithmetic>(input_epoch);
      complete_input_halo(input_epoch, curl_epoch, nullptr);
    }
    // Smoothed schedule: the interior C2N hides only the rest of the curl halo;
    // smoothing and the full mass apply run on executor_ with the two stable
    // scratch owners.
    enqueue_curl_interior<Arithmetic>();
    lpfx::device::record_event(curl_interior_ready_event_, interior_executor_,
                               "TPIC Maxwell: interior curl completion recording failed");
    {
      LPFX_PROFILE_SCOPE("TPIC::MaxwellApply::CurlHalo");
      curl_epoch.complete();
    }
    enqueue_curl_shell<Arithmetic>();
    const auto& mass_input =
        complete_smoothing<Arithmetic>(input, exchange_a, exchange_b, true, true, invalid);
    host_wait_mass_ghosts(executor_);
    enqueue_mass_response<StencilRegion::full, Arithmetic>(mass, mass_input, invalid, executor_);
    const auto& response =
        complete_smoothing<Arithmetic>(mass_image, exchange_a, exchange_b, false, false, invalid);
    combine_image<Arithmetic>(response);
  }

  template <class Arithmetic> void combine_image(const SolverNodeOwner<Precision, Backend>& response) {
    const auto& k = constants<Arithmetic>();
    // This first whole-curl consumer also joins all helper readers before the
    // next input unpack, matrix/validation epoch, RHS or Faraday scratch reuse.
    lpfx::device::wait_event(curl_interior_ready_event_, executor_,
                       "TPIC Maxwell: interior curl completion dependency failed");
    lpfx::check_error(enqueue_field_vector_linear_combination<Precision, Arithmetic>(image, nullptr, executor_,
        scaled(Arithmetic(1), input), scaled(k.delta * k.delta, curl_node),
        scaled(k.current_factor, response)),
        "TPIC Maxwell: operator image combination failed");
    lpfx::check_error(enqueue_constrained_rows(image, input,
                          plans_.boundary.constrained_rows, executor_),
                      "TPIC Maxwell: operator boundary rows failed");
  }

  template <MaxwellApplySchedule Schedule> void host_sync_apply_streams() noexcept {
    const Executor used[] = {executor_, interior_executor_, mass_executor_};
    constexpr unsigned count = Schedule == MaxwellApplySchedule::smoothed ? 2 : 3;
    for (unsigned i = 0; i < count; ++i) {
      bool duplicate = false;
      for (unsigned j = 0; j < i; ++j) duplicate |= used[i].stream() == used[j].stream();
      if (!duplicate && !Traits::succeeded(Traits::synchronize_stream(used[i].stream())))
        std::abort();
    }
  }

  template <class Owner> void project(Owner& field) {
    lpfx::check_error(enqueue_constrained_rows(field, detail::NoRowInput{},
        plans_.boundary.constrained_rows, executor_),
        "TPIC Maxwell: electric boundary projection failed");
  }

  // Source is external to the two work owners. Exchange bindings always keep
  // their original owner addresses; only these local pointers alternate.
  // Full direction refreshes also cover Binomial27's edge/corner samples.
  template <class Arithmetic = Compute, class Source, class SmoothAExchange, class SmoothBExchange>
  const SolverNodeOwner<Precision, Backend>& complete_smoothing(const Source& source, SmoothAExchange& exchange_a,
      SmoothBExchange& exchange_b, bool source_halos_valid, bool final_halos,
      unsigned* invalid) {
    auto refresh = [this](SolverNodeOwner<Precision, Backend>* owner, auto& smooth_a_exchange,
                          auto& smooth_b_exchange) {
      if (owner == &smooth_a) complete_electric_refresh(*owner, smooth_a_exchange);
      else complete_electric_refresh(*owner, smooth_b_exchange);
    };
    auto pass = [&](const auto& from, SolverNodeOwner<Precision, Backend>& to) {
      lpfx::check_error(enqueue_node_smoothing<Precision, Arithmetic>(
                            from, to, smooth_operator_, constants<Arithmetic>().smooth_alpha, invalid, executor_),
                        "TPIC Maxwell: node smoothing failed");
    };
    SolverNodeOwner<Precision, Backend>* current = &smooth_a;
    SolverNodeOwner<Precision, Backend>* next = &smooth_b;
    int completed = 0;
    if (source_halos_valid) {
      pass(source, *current);
      completed = 1;
    } else {
      lpfx::check_error(enqueue_field_vector_copy(*current, source, false, nullptr, executor_),
                        "TPIC Maxwell: smoothing source copy failed");
    }
    for (; completed < smooth_iterations_; ++completed) {
      refresh(current, exchange_a, exchange_b);
      pass(*current, *next);
      std::swap(current, next);
    }
    if (final_halos) refresh(current, exchange_a, exchange_b);
    else project(*current); // Projection only, no exchange.
    return *current;
  }

  const bool periodic_;
  const FieldOperatorPlans plans_;
  Executor executor_;
  Executor mass_executor_;
  Executor interior_executor_;
  Event input_halo_ready_event_, mass_ready_event_;
  Event input_local_ready_event_, curl_cell_ready_event_, curl_interior_ready_event_;
  // The constants of the right-hand side and of one operator apply in the
  // arithmetic type C.
  template <class C> struct Constants {
    NodeCellSpacing<C> spacing{};
    C inverse_volume{}, delta{}, current_factor{}, smooth_alpha{};
  };
  template <class C> [[nodiscard]] static Constants<C> make_constants(const MaxwellParameters& prepared) {
    Constants<C> result;
    lpfx::check_error(try_make_node_cell_spacing(prepared.domain.spacing, result.spacing),
                      "TPIC Maxwell: invalid spacing");
    result.inverse_volume = config::convert<C>(prepared.domain.inverse_cell_volume);
    result.delta = config::convert<C>(prepared.physics.c_theta_dt);
    result.current_factor = 4 * std::numbers::pi_v<C> * config::convert<C>(prepared.physics.theta_dt);
    result.smooth_alpha = config::convert<C>(prepared.smooth_alpha);
    return result;
  }
  template <class C> [[nodiscard]] const Constants<C>& constants() const noexcept {
    if constexpr (std::is_same_v<C, Compute>) return constants_;
    else return residual_constants_;
  }
  const Constants<Compute> constants_;
  const Constants<Residual> residual_constants_;
  NodeCellSpacing<MagneticCompute> magnetic_spacing_{};
  bool smoothing_enabled_;
  int smooth_iterations_;
  config::EcsimSmoothingOperator smooth_operator_;
  const Compute theta_value_;
  const MagneticCompute faraday_;
};

} // namespace tpic::ecsim

#endif // TPIC_ECSIM_MAXWELL_MAXWELL_DEVICE_HPP
