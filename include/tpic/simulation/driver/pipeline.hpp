#ifndef TPIC_SIMULATION_DRIVER_PIPELINE_HPP
#define TPIC_SIMULATION_DRIVER_PIPELINE_HPP

#include <lpfx/backend/error_reporting.hpp>
#include <lpfx/core/attributes.hpp>
#include <lpfx/core/error_reporting.hpp>
#include <lpfx/execution/device/context.hpp>
#include <lpfx/execution/host/channel.hpp>
#include <lpfx/execution/host/fence.hpp>
#include <lpfx/execution/host/worker.hpp>
#include <lpfx/profiling.hpp>

#include <tpic/communication/mass_halo/mass_halo_plan.hpp>
#include <tpic/data/output/gauss_sample.hpp>
#include <tpic/data/output/scalar_sample.hpp>
#include <tpic/simulation/execution/deposition_ledger.hpp>
#include <tpic/simulation/execution/execution_layout.hpp>
#include <tpic/simulation/execution/execution_streams.hpp>
#include <tpic/simulation/planning/run_plan.hpp>
#include <tpic/simulation/reporting/run_reporter.hpp>
#include <tpic/simulation/reporting/startup_report.hpp>
#include <tpic/simulation/timing/stage_timing.hpp>

#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <memory>
#include <optional>
#include <span>
#include <string_view>
#include <thread>
#include <utility>

namespace tpic::simulation {

// The runtime of one run. Physics owns the device owners, actions and distributed primitives and decides nothing
// about order. Each cycle admits a solve (field_solve, diagnostics_output read state n) and an advance (the other
// workers write state n + 1). Every worker is one host thread and owns what it produces, so every wait names its
// producer. A *_event is a device event waited on a stream; every other wait is a host fence (*_fence) or channel of
// the epoch's generation. Host-compilable: MPI, io and device code are reached through Physics only.
//   main (control): waits diagnostics_output.diagnosed_fence, .piece_ready_fence of a snapshot that publishes,
//     .done_fence, field_solve.done_fence, field_advance.committed_fence, local_species[s].owner_ready_fence,
//     field_advance.committed_event, local_species[s].owner_ready_event, on field_update and move
//     diagnostics_output.sources_read_event of an epoch that reads the state and its capture source events of one that
//     captures, the done_fence of every advance worker during the next solve (bootstrap: deposition.done_fence),
//     and with no cycle deposition.assembled_fence[q] + assembled_event[q].
//     Produces state_ready_event_, publication_allowed_fence_; at startup field_solve.deposition_ready_event and
//     the bootstrap deposition.inputs.
//     Every worker signals its done_fence after each epoch.
//   field_solve (solver): waits state_ready_event_, deposition.assembled_fence[current] + assembled_event[current],
//     deposition.mass_interior_fence + mass_interior_event, and before its first read of mass ghost records
//     deposition.assembled_fence[mass] + assembled_event[mass]. Produces particle_field, solve_validation,
//     particle_field_ready_event, deposition_ready_event, result, done_fence.
//   diagnostics_output (diagnostics, diagnostics_helper, output_capture): waits state_ready_event_,
//     field_solve.done_fence when it reduces scalars, works on a Gauss sample or captures, on output_capture
//     sources_read_event before packing a Gauss residual, publication_allowed_fence_ of a snapshot that
//     publishes. Produces sources_read_event when it reads the state,
//     sample, gauss_sample, diagnosed_fence, piece_ready_fence, done_fence; feeds the output and checkpoint
//     writer threads.
//   mover (move): waits field_advance.cell_halo_submitted_fence, field_solve.particle_field_ready_event. Produces
//     per species moved_event[s] then moved_fence[s], then all_moved_fence.
//   field_advance (field_update): waits field_solve.particle_field_ready_event + field_solve.solve_validation,
//     mover.all_moved_fence, mover.moved_event. Produces cell_halo_submitted_fence, next_magnetic_event +
//     next_magnetic_fence, committed_event + committed_fence.
//   routing (route): waits field_advance.cell_halo_submitted_fence, local_species[s].migration_prepared_fence.
//     Produces routed_fence[s], then all_routed_fence.
//   local_species[s] (local[s], sort[s]): waits mover.moved_fence[s] + mover.validation[s], mover.moved_event[s],
//     routing.routed_fence[s]; in partial order deposition.resident_done_event[s][q] announced on its
//     resident_readers inbox. Produces migration_prepared_fence, residents, resident_release_done_event,
//     owner_ready_event, owner_ready_fence, deposition.inputs.
//   deposition (current, mass): waits field_solve.deposition_ready_event, its inputs,
//     field_advance.next_magnetic_fence + next_magnetic_event, local_species[s].owner_ready_event,
//     routing.all_routed_fence. Produces resident_done_event[s][q] and local_species[s].resident_readers,
//     mass_interior_event + mass_interior_fence, assembled_event[q] + assembled_fence[q], done_fence.
template <class Physics> class Pipeline {
  using Backend = typename Physics::Backend;
  using Executor = lpfx::device::Executor<Backend>;
  using Event = lpfx::device::DeviceEvent<Backend>;
  using Timing = StageTiming<Backend>;
  using Validation = typename Physics::Validation;
  using Clock = std::chrono::steady_clock;
  static constexpr std::size_t species_capacity = Physics::Ns;
  static constexpr std::size_t moment_quantities = ExecutionLayout::moment_quantities;
  static constexpr bool Partial = Physics::partial;
  static constexpr auto success = lpfx::HostChannelStatus::success;

  // ---- workers: one host thread each, running its program for every submitted epoch
  struct Worker {
    Worker(const char* worker_name, bool advancing) : name(worker_name), advances(advancing) {}
    void submit(const Epoch& epoch) {
      lpfx::require(thread.submit(Epoch{epoch}), "TPIC pipeline: worker admission failed");
    }

    const char* name;
    bool advances; // timing label: the state advanced from (bootstrap: the first state), else the state read
    unsigned species = 0;
    lpfx::HostFence done_fence;
    std::unique_ptr<Timing> timing;
    lpfx::HostWorker<Epoch> thread; // started by Pipeline::start
  };
  using Program = void (Pipeline::*)(Worker&, const Epoch&);
  struct MoverWorker : Worker {
    MoverWorker() : Worker("mover", true) {}
    std::array<lpfx::HostFence, species_capacity> moved_fence;
    lpfx::HostFence all_moved_fence;
    std::array<Event, species_capacity> moved_event;
    std::array<std::unique_ptr<Validation>, species_capacity> validation;
  };
  struct FieldAdvanceWorker : Worker {
    FieldAdvanceWorker() : Worker("field_advance", true) {}
    lpfx::HostFence cell_halo_submitted_fence, next_magnetic_fence, committed_fence;
    Event next_magnetic_event, committed_event;
    Validation advance_validation;
  };
  struct RoutingWorker : Worker {
    RoutingWorker() : Worker("routing", true) {}
    std::array<lpfx::HostFence, species_capacity> routed_fence;
    lpfx::HostFence all_routed_fence;
  };
  // The resident members are used in partial order only.
  struct LocalSpeciesWorker : Worker {
    LocalSpeciesWorker() : Worker("local_species", true) {}
    lpfx::HostFence migration_prepared_fence, owner_ready_fence;
    Event owner_ready_event, resident_release_done_event;
    lpfx::HostChannel<MomentQuantity, moment_quantities> resident_readers;
    std::optional<typename Physics::Residents> residents; // handed to deposition until its last resident reader
    std::uint32_t resident_prefix{};
  };
  struct DepositionWorker : Worker {
    explicit DepositionWorker(unsigned species_count) : Worker("deposition", true), ledger(species_count) {}
    std::array<lpfx::HostFence, moment_quantities> assembled_fence;
    std::array<Event, moment_quantities> assembled_event;
    lpfx::HostFence mass_interior_fence;
    Event mass_interior_event;
    std::array<std::array<Event, moment_quantities>, Partial ? species_capacity : 0> resident_done_event;
    lpfx::HostChannel<DepositionInput, 2 * species_capacity> inputs;
    DepositionLedger<species_capacity, Partial> ledger;
    std::array<Validation, moment_quantities> moment_validation;
  };
  struct FieldSolveWorker : Worker {
    FieldSolveWorker() : Worker("field_solve", false) {}
    Event particle_field_ready_event, deposition_ready_event;
    // Flags a non-finite right-hand side, mass response or smoothed particle field.
    Validation solve_validation;
    const typename Physics::SolverField* particle_field{};
    typename Physics::Solved result{};
  };
  struct DiagnosticsOutputWorker : Worker {
    DiagnosticsOutputWorker() : Worker("diagnostics_output", false) {}
    lpfx::HostFence diagnosed_fence, piece_ready_fence;
    Event helper_sources_read_event, sources_read_event;
    Validation output_validation, gauss_validation;
    data::ScalarDiagnosticSample sample{};
    data::GaussSample gauss_sample{};
    bool scalar_written{};
    typename Physics::Capture capture;    // the host capture that crosses the output writer thread
    typename Physics::Capture checkpoint; // the one that crosses the checkpoint writer thread
    double initial_energy{};              // of the run's first state: the dE baseline every checkpoint carries
  };

public:
  Pipeline(Physics& physics, ExecutionStreams<Backend>& stream_registry,
           const StreamExecutors<Backend, species_capacity>& streams)
      : physics_(physics), stream_registry_(stream_registry), streams_(streams),
        species_count_(physics.species_count), first_cycle_(physics.plan.first_cycle()),
        cycle_count_(physics.input.parameters.ncycles), deposition_(species_count_),
        main_timing_(physics.plan.debug.profile_stages, species_count_, first_cycle_),
        reporter_(physics.rank == 0, first_cycle_) {
    for (auto* event : {&state_ready_event_, &field_solve_.particle_field_ready_event,
                        &field_solve_.deposition_ready_event, &diagnostics_output_.helper_sources_read_event,
                        &diagnostics_output_.sources_read_event, &field_advance_.next_magnetic_event,
                        &field_advance_.committed_event, &deposition_.assembled_event[0],
                        &deposition_.assembled_event[1], &deposition_.mass_interior_event})
      create(*event);
    // The advance workers in submission order, then the two solve workers.
    workers_ = {&mover_, &field_advance_, &routing_, &deposition_};
    workers_[species_count_ + 4] = &field_solve_;
    workers_[species_count_ + 5] = &diagnostics_output_;
    for (unsigned species = 0; species < species_count_; ++species) {
      local_species_[species].species = species;
      workers_[4 + species] = &local_species_[species];
      create(mover_.moved_event[species]);
      create(local_species_[species].owner_ready_event);
      if constexpr (Partial) {
        create(local_species_[species].resident_release_done_event);
        for (auto& event : deposition_.resident_done_event[species]) create(event);
      }
      mover_.validation[species] = std::make_unique<Validation>();
    }
    for (Worker* worker : all_workers())
      worker->timing = std::make_unique<Timing>(physics.plan.debug.profile_stages, species_count_, first_cycle_);
    lpfx::require(stream_registry.size() == ExecutionLayout::streams(species_count_,
                      physics.input.particles.order != config::ParticleOrder::unsorted, physics.output.enabled()),
                  "TPIC pipeline: simulation stream layout disagrees with resource admission");
    stream_registry.seal();
    if (physics.plan.restart)
      diagnostics_output_.initial_energy = physics.plan.restart->checkpoint.header.initial_energy;
    physics.output.configure(output_writer_);
    physics.output.configure_checkpoints(checkpoint_writer_);
  }
  // The worker threads borrow each other's members and the writer: all are joined before any is destroyed.
  ~Pipeline() { host_join_workers(); }

  [[nodiscard]] RunResult execute() {
    LPFX_PROFILE_SCOPE("TPIC::Execute");
    try {
      physics_.prepare();
      initialize();
      physics_.host_blocking_report_memory();
      // Started after the memory gather: their thread setups do not overlap it. The diagnostics-output worker
      // writes the scalar file and feeds the writer threads.
      start(field_solve_, &Pipeline::run_field_solve);
      physics_.diagnostics.open_scalars(physics_.plan.restart);
      const auto writer_setup = [this]() noexcept { setup_thread(); };
      output_writer_.start(writer_setup);
      checkpoint_writer_.start(writer_setup);
      start(diagnostics_output_, &Pipeline::run_diagnostics_output);
      if (advances(physics_.plan.mode)) run_cycles();
      close();
      return run_result_;
    } catch (const std::exception& error) {
      physics_.abort(error.what());
    } catch (...) {
      physics_.abort("TPIC pipeline: unknown simulation failure");
    }
  }

private:
  // ---- task distribution
  // The workers one advance epoch is submitted to, in submission order.
  std::span<Worker* const> advance_workers() const { return {workers_.data(), species_count_ + 4}; }
  // Every worker: the advance workers, then the two solve workers.
  std::span<Worker* const> all_workers() const { return {workers_.data(), species_count_ + 6}; }
  void start(Worker& worker, Program program) {
    worker.thread.start([this, &worker, program](Epoch&& epoch) noexcept { run(worker, program, epoch); },
                        [this]() noexcept { setup_thread(); });
  }
  // Every worker and writer thread calls MPI or the device runtime on the rank's device.
  void setup_thread() noexcept {
    try {
      lpfx::device::check_device_error<Backend>(
          lpfx::device::select_device<Backend>(physics_.properties.device_ordinal),
          "TPIC pipeline: worker device selection failed");
    } catch (const std::exception& error) {
      physics_.abort(error.what());
    }
  }
  void run(Worker& worker, Program program, const Epoch& epoch) noexcept {
    try {
      const bool timed = !epoch.final_state;
      if (timed)
        worker.timing->begin_cycle(worker.advances && !epoch.bootstrap ? epoch.cycle - 1 : epoch.cycle,
                                   epoch.captures());
      (this->*program)(worker, epoch);
      if (timed) worker.timing->close_cycle();
      host_signal(worker.done_fence, epoch.generation());
    } catch (const std::exception& error) {
      fail(worker, epoch, error.what());
    } catch (...) {
      fail(worker, epoch, "unknown worker failure");
    }
  }
  // Accepted MPI and native work cannot be cancelled by closing host
  // dependencies: a worker failure aborts the run.
  [[noreturn]] void fail(const Worker& worker, const Epoch& epoch, const char* reason) noexcept {
    char species[32] = "", message[1400];
    if (std::string_view(worker.name) == "local_species")
      std::snprintf(species, sizeof(species), " species %u", worker.species);
    std::snprintf(message, sizeof(message), "TPIC pipeline: worker %s%s cycle %d: %.1200s", worker.name, species,
                  epoch.cycle, reason);
    physics_.abort(message);
  }

  // ---- helpers
  static LPFX_ALWAYS_INLINE void yield_every_64(unsigned& polls) noexcept {
    if ((++polls & 63U) == 0) std::this_thread::yield();
  }
  template <class Poll> static LPFX_ALWAYS_INLINE void host_poll_until(Poll&& poll) {
    unsigned polls = 0;
    while (!poll()) yield_every_64(polls);
  }
  // Timing collection never synchronizes streams: only the profiling stop
  // markers can still be pending.
  static LPFX_ALWAYS_INLINE void host_poll_timing(Timing& timing) {
    while (!timing.progress_collect_cycle()) std::this_thread::yield();
  }
  static LPFX_ALWAYS_INLINE void host_signal(lpfx::HostFence& fence, std::uint64_t generation) {
    lpfx::require(fence.signal(generation), "TPIC pipeline: host fence signalled twice");
  }
  static LPFX_ALWAYS_INLINE void host_wait(const lpfx::HostFence& fence, std::uint64_t generation) {
    lpfx::require(fence.wait(generation), "TPIC pipeline: dependency closed before completion");
  }
  static LPFX_ALWAYS_INLINE void create(Event& event) {
    lpfx::device::create_event(event, "TPIC pipeline: event creation failed");
  }
  static LPFX_ALWAYS_INLINE void device_record(Event& event, Executor executor) {
    lpfx::device::record_event(event, executor, "TPIC pipeline: event recording failed");
  }
  static LPFX_ALWAYS_INLINE void device_wait(const Event& event, Executor executor) {
    lpfx::device::wait_event(event, executor, "TPIC pipeline: event dependency failed");
  }
  static LPFX_ALWAYS_INLINE void host_sync_device_stream(Executor executor) {
    lpfx::device::check_device_error<Backend>(
        lpfx::device::BackendTraits<Backend>::synchronize_stream(executor.stream()),
        "TPIC pipeline: stream completion failed");
  }
  // At most two inputs per species exist per generation: no producer can overfill the
  // channel even if the deposition worker has not started yet.
  void offer_deposition(std::uint64_t generation, unsigned species, DepositionPhase phase, ParticleRange range) {
    lpfx::require(deposition_.inputs.try_send(DepositionInput{generation, species, phase, range}) == success,
                  "TPIC pipeline: descriptor capacity/protocol violated");
  }
  void count_samples(const Epoch& epoch) {
    if (diagnostics_output_.scalar_written) ++run_result_.diagnostic_samples;
    if (epoch.gauss_finish) ++run_result_.gauss_samples;
    if (epoch.fields) ++run_result_.field_samples;
    if (epoch.particles) ++run_result_.particle_samples;
    if (epoch.checkpoint) ++run_result_.checkpoints;
  }
  // The writer has written its pending snapshot (one that publishes nothing)
  // and returns it to its slot; a write failure is rethrown here.
  static void host_join(typename Physics::Writer& writer, typename Physics::Capture& slot) {
    writer.host_wait_piece();
    slot = writer.complete_snapshot();
  }

  // ---- main thread
  // Blocking startup on main, then the bootstrap advance.
  void initialize() {
    LPFX_PROFILE_SCOPE("TPIC::Initialize");
    {
      LPFX_PROFILE_SCOPE("TPIC::Initialize::FieldsAndParticles");
      physics_.field_advance->complete_initial_halos(physics_.plan.restart.has_value());
      // Finish C2N before releasing this cold startup result to its readers or
      // allowing callers to release its owners.
      host_sync_device_stream(streams_.control);
      for (unsigned species = 0; species < species_count_; ++species) physics_.local.host_blocking_center(species);
      for (unsigned species = 0; species < species_count_; ++species) migrate_initial_species(species);
      if (physics_.local.full_order())
        for (unsigned species = 0; species < species_count_; ++species) physics_.local.host_wait_sorted(species);
    }
    physics_.local.host_blocking_check_population();
    start(mover_, &Pipeline::run_mover);
    start(field_advance_, &Pipeline::run_field_advance);
    start(routing_, &Pipeline::run_routing);
    start(deposition_, physics_.diagnose ? &Pipeline::run_deposition<true> : &Pipeline::run_deposition<false>);
    for (unsigned species = 0; species < species_count_; ++species)
      start(local_species_[species], &Pipeline::run_local_species);
    bootstrap_start_ = Clock::now();
    start_bootstrap();
    bootstrap_seconds_ = std::chrono::duration<double>(Clock::now() - bootstrap_start_).count();
    if (!advances(physics_.plan.mode) || cycle_count_ <= first_cycle_)
      for (unsigned quantity = 0; quantity < moment_quantities; ++quantity) {
        host_wait(deposition_.assembled_fence[quantity], advance_epoch_.generation());
        device_wait(deposition_.assembled_event[quantity], streams_.control);
      }
    // Initializer and its uploads have retired; this event marks the initial state as ready.
    device_record(state_ready_event_, streams_.control);
  }

  void migrate_initial_species(unsigned species) {
    const Executor control_stream = streams_.control, local_stream = streams_.local[species],
                   sort_stream = streams_.sort[species];
    if (physics_.local.bypassed()) {
      Validation& validation = *mover_.validation[species];
      validation.clear(control_stream);
      physics_.local.apply_initial_boundaries(species, validation.data(), control_stream);
      validation.capture(control_stream);
      validation.host_wait_check("TPIC pipeline: invalid or nonresident particle after local startup boundaries");
      if (physics_.local.full_order()) {
        validation.device_wait(sort_stream);
        LPFX_PROFILE_SCOPE("TPIC::Initialize::ParticleSort");
        physics_.local.sort(species, sort_stream);
      }
      return;
    }
    physics_.local.classify_initial(species, control_stream);
    physics_.local.host_wait_retired(species, local_stream);
    {
      LPFX_PROFILE_SCOPE("TPIC::Initialize::ParticleMigration");
      physics_.local.begin(species, local_stream);
      bool resident_sort_launched = false;
      host_poll_until([&] {
        const bool done = physics_.local.progress_prepare(species);
        if constexpr (Partial)
          if (!resident_sort_launched && physics_.local.resident_access_ready(species)) {
            auto range = physics_.local.acquire_resident_access(species);
            device_wait(*range.ready, sort_stream);
            physics_.local.sort_resident(species, std::move(range), sort_stream);
            resident_sort_launched = true;
          }
        return done;
      });
      physics_.local.host_wait_resident(species);
      if constexpr (Partial) {
        lpfx::require(resident_sort_launched, "TPIC pipeline: startup resident sort was not launched");
        physics_.local.host_wait_sorted(species);
      }
      physics_.local.host_wait_route(species);
      if constexpr (Partial) {
        physics_.local.release_growth_scratch(species);
        const auto& readers_closed = physics_.local.close_empty_readers(species, local_stream);
        host_poll_until([&] { return physics_.local.progress_release_access(species, readers_closed); });
      }
      host_poll_until([&] { return physics_.local.host_wait_finalize(species); });
      lpfx::require(physics_.local.progress_conclude(species),
                    "TPIC local species: particle migration conclusion failed");
      physics_.local.refresh(species);
    }
    physics_.local.finish_initial(species);
    if (physics_.local.full_order()) {
      LPFX_PROFILE_SCOPE("TPIC::Initialize::ParticleSort");
      physics_.local.sort(species, sort_stream);
    }
  }

  void start_bootstrap() {
    LPFX_PROFILE_SCOPE("TPIC::Bootstrap");
    advance_epoch_ = Epoch{.cycle = first_cycle_, .bootstrap = true, .moments = true};
    advance_active_ = true;
    const auto generation = advance_epoch_.generation();
    // Initial owners and B0 are checked and committed on main at startup.
    device_record(field_solve_.deposition_ready_event, streams_.control);
    deposition_.submit(advance_epoch_);
    for (unsigned species = 0; species < species_count_; ++species) {
      const auto range = physics_.local.completed(species);
      if constexpr (Partial) {
        offer_deposition(generation, species, DepositionPhase::resident, {0, range.first, range.first});
        offer_deposition(generation, species, DepositionPhase::tail, range);
      } else
        offer_deposition(generation, species, DepositionPhase::whole, range);
    }
  }

  void run_cycles() {
    LPFX_PROFILE_SCOPE("TPIC::Run");
    reporter_.begin();
    const auto loop_start = bootstrap_start_;
    Clock::time_point steady_start{};
    for (int cycle = first_cycle_; cycle < cycle_count_; ++cycle) {
      LPFX_PROFILE_SCOPE("TPIC::Cycle");
      // The first state is a restart's source or the initial state: no checkpoint.
      // A fresh run's first state reduces its scalars for the initial energy.
      const bool energy_reported = physics_.input.output.should_report_energy(cycle);
      const Epoch solve_epoch{.cycle = cycle, .solve = true,
                        .scalars = energy_reported || physics_.input.output.should_write_diagnostics(cycle) ||
                                   (cycle == first_cycle_ && !physics_.plan.restart),
                        .gauss_begin = physics_.input.output.should_sample_gauss(cycle, cycle_count_),
                        .gauss_finish = solve_epoch_.gauss_begin,
                        .fields = physics_.output.fields_due(cycle),
                        .particles = physics_.output.particles_due(cycle),
                        .checkpoint = cycle != first_cycle_ && physics_.output.checkpoint_due(cycle)};
      main_timing_.begin_cycle(cycle, solve_epoch.captures());
      const auto cycle_start = Clock::now();
      {
        LPFX_PROFILE_SCOPE("TPIC::CycleTotal");
        auto cycle_scope = main_timing_.scope(TimingRegion::CycleTotal, streams_.control);
        start_solve(solve_epoch);
        // While the solve runs: rank 0 prints, and the previous advance, whose
        // assembly the solve waits for, is joined and its worker timing printed.
        reporter_.cycle(cycle, double(cycle) * physics_.input.parameters.dt);
        host_wait_advance_done();
        host_wait_solved();
        count_samples(solve_epoch);
        // R1: the host fence signal proves these events were recorded for this
        // generation. Every reader wait is enqueued before the first E overwrite of
        // the field advance and the mover's first particle write; disk completion is
        // independent. Scalars and Gauss samples read the state on the diagnostics lanes.
        if (solve_epoch.reads_state()) device_wait(diagnostics_output_.sources_read_event, streams_.field_update);
        if (solve_epoch.captures()) physics_.output.device_wait_sources(streams_.field_update);
        if (solve_epoch.reads_state()) device_wait(diagnostics_output_.sources_read_event, streams_.move);
        if (solve_epoch.captures()) physics_.output.device_wait_sources(streams_.move);
        start_advance(cycle + 1, cycle + 1 < cycle_count_, solve_epoch);
        if (energy_reported) reporter_.energy(diagnostics_output_.sample, diagnostics_output_.initial_energy);
        if (solve_epoch.gauss_finish) reporter_.gauss(diagnostics_output_.gauss_sample);
        host_wait_advanced();
        device_record(state_ready_event_, streams_.control);
        host_blocking_finish_solve(solve_epoch);
        for (Timing* timing : {field_solve_.timing.get(), diagnostics_output_.timing.get()}) {
          host_poll_timing(*timing);
          main_timing_.merge_cycle(*timing);
        }
        if (cycle == first_cycle_) host_blocking_timing_boundary(true);
      }
      main_timing_.close_cycle();
      host_poll_timing(main_timing_);
      if (physics_.rank == 0) {
        const double elapsed_ms = std::chrono::duration<double, std::milli>(Clock::now() - cycle_start).count();
        reporter_.finish_cycle(cycle, solve_epoch.captures(), field_solve_.result.iterations,
                               double(field_solve_.result.relative_residual),
                               elapsed_ms);
        main_timing_.print_cycle();
        std::fflush(stdout);
      }
      run_result_.steps_completed = cycle + 1 - first_cycle_;
      if (cycle == first_cycle_ && cycle_count_ > first_cycle_ + 1) {
        // Start steady-state timing after root has reported warm-up and
        // all ranks are ready; the steady-state interval covers the later cycles.
        host_blocking_timing_boundary(false);
        LPFX_PROFILE_MARK("TPIC::SteadyStateBegin");
        // CrayPat records the steady-state interval: on the main thread the switch covers every worker.
        LPFX_PROFILE_RECORD(true);
        steady_start = Clock::now();
      }
    }
    const int steps = run_result_.steps_completed;
    if (steps > 0) host_blocking_timing_boundary(true);
    // After the closing boundary, so that the recorded interval is the reported steady state.
    LPFX_PROFILE_RECORD(false);
    const auto loop_end = Clock::now();
    reporter_.loop(physics_.host_blocking_max_over_ranks({std::chrono::duration<double>(loop_end - loop_start).count(),
                                      steps > 1 ? std::chrono::duration<double>(loop_end - steady_start).count()
                                                : 0.0}),
                   steps);
    if (physics_.rank == 0) {
      main_timing_.print_summary();
      for (const Worker* worker : advance_workers()) {
        if (!worker->timing->enabled()) continue;
        std::printf("\nWorker timing role %s species %u\n", worker->name, worker->species);
        worker->timing->print_summary();
      }
      reporter_.bootstrap(bootstrap_seconds_);
    }
    // The final solve skips the field solve: the last advance produces no moments.
    const Epoch final_epoch{.cycle = cycle_count_, .scalars = true, .gauss_finish = solve_epoch_.gauss_begin,
                     .final_state = true,
                     .fields = physics_.output.fields_enabled(),
                     .particles = physics_.output.particles_enabled(),
                     .checkpoint = cycle_count_ != first_cycle_ &&
                                   physics_.output.checkpoint_due(cycle_count_, true)};
    start_solve(final_epoch);
    host_wait_solved();
    reporter_.final_state(diagnostics_output_.sample, diagnostics_output_.initial_energy);
    if (final_epoch.gauss_finish) reporter_.gauss(diagnostics_output_.gauss_sample);
    count_samples(final_epoch);
    // After the drain this joins no advance; with no cycles a final state that publishes joins the bootstrap.
    host_blocking_finish_solve(final_epoch);
    physics_.local.host_blocking_check_population();
  }

  void start_solve(const Epoch& epoch) {
    solve_epoch_ = epoch;
    if (epoch.solve) field_solve_.submit(epoch);
    diagnostics_output_.submit(epoch);
  }
  void host_wait_solved() {
    const auto generation = solve_epoch_.generation();
    host_wait(diagnostics_output_.diagnosed_fence, generation);
    if (solve_epoch_.solve) host_wait(field_solve_.done_fence, generation);
  }
  // The caller has joined the previous advance.
  void start_advance(int destination, bool moments, const Epoch& solve_epoch) {
    advance_epoch_ = Epoch{.cycle = destination, .moments = moments, .fields = solve_epoch.fields,
                           .particles = solve_epoch.particles, .checkpoint = solve_epoch.checkpoint};
    advance_active_ = true;
    // Every worker owns a running program and separate handoffs; no program
    // waits for a job queued behind itself.
    for (Worker* worker : advance_workers()) worker->submit(advance_epoch_);
  }
  void host_wait_advanced() {
    LPFX_PROFILE_SCOPE("TPIC::WaitAdvanced");
    const auto generation = advance_epoch_.generation();
    host_wait(field_advance_.committed_fence, generation);
    for (unsigned species = 0; species < species_count_; ++species)
      host_wait(local_species_[species].owner_ready_fence, generation);
    device_wait(field_advance_.committed_event, streams_.control);
    for (unsigned species = 0; species < species_count_; ++species)
      device_wait(local_species_[species].owner_ready_event, streams_.control);
  }
  // The admission join: every worker of the advance (the bootstrap: deposition only)
  // has signalled its done_fence, and its timing is collected and printed.
  void host_wait_advance_done() {
    if (!advance_active_) return;
    const auto generation = advance_epoch_.generation();
    const auto join = [&](Worker& worker) {
      host_wait(worker.done_fence, generation);
      if (!worker.timing->enabled()) return;
      host_poll_timing(*worker.timing);
      if (physics_.rank == 0) {
        std::printf("  Worker timing epoch %llu destination %d role %s species %u%s\n",
                    static_cast<unsigned long long>(generation), advance_epoch_.cycle, worker.name, worker.species,
                    advance_epoch_.bootstrap ? " bootstrap" : "");
        worker.timing->print_cycle();
      }
    };
    if (advance_epoch_.bootstrap) join(deposition_);
    else
      for (Worker* worker : advance_workers()) join(*worker);
    advance_active_ = false;
  }
  void host_blocking_finish_solve(const Epoch& epoch) {
    const auto generation = epoch.generation();
    if (physics_.output.publishes(epoch)) {
      host_wait(diagnostics_output_.piece_ready_fence, generation);
      // R1: control-domain agreement cannot overlap the next advance's native
      // interval. Deposition, field advance and routing are independent of the
      // detached writer at this point.
      host_wait_advance_done();
      physics_.host_blocking_barrier();
      host_signal(publication_allowed_fence_, generation);
    }
    host_wait(diagnostics_output_.done_fence, generation);
  }
  // A small, fixed number of fences defines comparable whole-loop intervals.
  // Detailed scopes never introduce per-stage fences or change LPFx lifecycle.
  void host_blocking_timing_boundary(bool drain_first) {
    if (drain_first) host_sync_device_streams();
    if (physics_.ranks > 1) physics_.host_blocking_barrier();
  }
  void host_sync_device_streams() {
    LPFX_PROFILE_SCOPE("TPIC::DrainExecutors");
    host_wait_advance_done();
    for (std::size_t stream = 0; stream < stream_registry_.size(); ++stream)
      host_sync_device_stream(stream_registry_.registration(stream).executor);
  }
  void close() {
    LPFX_PROFILE_SCOPE("TPIC::Close");
    host_sync_device_streams();
    for (unsigned species = 0; species < species_count_; ++species)
      physics_.local.host_wait_retired(species, streams_.local[species]);
    host_join_workers();
    physics_.transport.close();
  }
  void host_join_workers() noexcept {
    for (Worker* worker : all_workers()) worker->thread.close();
    for (Worker* worker : all_workers()) worker->thread.join();
  }

  // ---- worker programs
  void run_mover(Worker&, const Epoch& epoch) {
    LPFX_PROFILE_SCOPE("TPIC::Mover");
    const Executor move_stream = streams_.move;
    const auto generation = epoch.generation();
    // R1: the movers are enqueued after the field advance has submitted its first
    // cell-halo operation, which keeps that submission ahead of the mover launches.
    host_wait(field_advance_.cell_halo_submitted_fence, generation);
    device_wait(field_solve_.particle_field_ready_event, move_stream);
    for (unsigned species = 0; species < species_count_; ++species) {
      // No deposit wait: particle_field_ready_event follows the assembled waits, hence every deposit.
      {
        LPFX_PROFILE_SCOPE("TPIC::FinalizeAndMove");
        auto scope = mover_.timing->scope(TimingRegion::Mover, move_stream, species);
        Validation& validation = *mover_.validation[species];
        validation.clear(move_stream);
        physics_.mover.move(species, *field_solve_.particle_field, validation.data(), move_stream);
        validation.capture(move_stream);
        device_record(mover_.moved_event[species], move_stream);
      }
      host_signal(mover_.moved_fence[species], generation);
    }
    host_signal(mover_.all_moved_fence, generation);
  }

  void run_field_advance(Worker&, const Epoch& epoch) {
    LPFX_PROFILE_SCOPE("TPIC::FieldAdvance");
    const Executor field_update_stream = streams_.field_update;
    const auto generation = epoch.generation();
    device_wait(field_solve_.particle_field_ready_event, field_update_stream);
    {
      auto scope = field_advance_.timing->scope(TimingRegion::FieldAdvance, field_update_stream);
      field_advance_.advance_validation.clear(field_update_stream);
      physics_.field_advance->advance(field_advance_.advance_validation.data());
      bool cell_halo_started = false;
      host_poll_until([&] {
        const auto cell_progress = physics_.field_advance->progress_cell();
        if (!cell_halo_started && cell_progress.first_submitted) {
          host_signal(field_advance_.cell_halo_submitted_fence, generation);
          cell_halo_started = true;
        }
        return cell_progress.complete;
      });
      physics_.field_advance->close_cell(field_advance_.advance_validation.data());
      field_advance_.advance_validation.capture(field_update_stream);
    }
    device_record(field_advance_.next_magnetic_event, field_update_stream);
    host_signal(field_advance_.next_magnetic_fence, generation);
    // R1: every moved_event of this generation is recorded before the commit waits
    // on it.
    host_wait(mover_.all_moved_fence, generation);
    // R1: the commit makes the solved and the next fields current, so both flags
    // are read first.
    host_poll_until([&] {
      return field_solve_.solve_validation.ready(
          "TPIC pipeline: invalid Maxwell RHS, mass response or particle field");
    });
    host_poll_until([&] {
      return field_advance_.advance_validation.ready("TPIC field advance: invalid next magnetic fields");
    });
    {
      auto scope = field_advance_.timing->scope(TimingRegion::FieldCommit, field_update_stream);
      // R1: the commit overwrites the fields the movers read. Every moved_event is
      // recorded on the one move stream in species order, so FIFO makes the last
      // species' record imply all earlier ones.
      device_wait(mover_.moved_event[species_count_ - 1], field_update_stream);
      physics_.field_advance->commit();
    }
    device_record(field_advance_.committed_event, field_update_stream);
    LPFX_PROFILE_MARK("TPIC::FieldCommitted");
    host_signal(field_advance_.committed_fence, generation);
  }

  void run_routing(Worker&, const Epoch& epoch) {
    LPFX_PROFILE_SCOPE("TPIC::Routing");
    const auto generation = epoch.generation();
    // R1: the field advance submits its only native work, the next-cell halo,
    // before every migration route, and the species are routed in species
    // order on every rank.
    host_wait(field_advance_.cell_halo_submitted_fence, generation);
    for (unsigned species = 0; species < species_count_; ++species) {
      host_wait(local_species_[species].migration_prepared_fence, generation);
      if (!physics_.local.bypassed()) {
        auto scope = routing_.timing->scope(TimingRegion::ParticleMigration, streams_.route, species);
        host_poll_until([&] { return physics_.local.progress_route(species); });
      }
      host_signal(routing_.routed_fence[species], generation);
    }
    LPFX_PROFILE_MARK("TPIC::TransportReleased");
    host_signal(routing_.all_routed_fence, generation);
  }

  void run_local_species(Worker& species_worker, const Epoch& epoch) {
    LPFX_PROFILE_SCOPE("TPIC::LocalSpecies");
    const unsigned species = species_worker.species;
    LocalSpeciesWorker& worker = local_species_[species];
    const Executor local_stream = streams_.local[species], sort_stream = streams_.sort[species];
    const auto generation = epoch.generation();
    Timing& timing = *worker.timing;
    const bool migration_bypassed = physics_.local.bypassed();
    host_wait(mover_.moved_fence[species], generation);
    host_poll_until([&] { return mover_.validation[species]->ready("TPIC mover: invalid moved particle"); });
    physics_.local.retire_after(species, mover_.moved_event[species], local_stream);
    // R1: compaction and the local resources inherit every deposition reader through
    // moved_event: the mover waited particle_field_ready_event, recorded after both assembled events.
    device_wait(mover_.moved_event[species], local_stream);
    host_poll_until([&] { return physics_.local.progress_retirement(species); });
    ProgressScope<Backend> migration_scope, sort_scope;
    bool resident_sort_launched = false;
    if (!migration_bypassed) {
      LPFX_PROFILE_SCOPE("TPIC::MigrationPrepare");
      migration_scope.begin(timing, TimingRegion::ParticleMigration, local_stream, species);
      physics_.local.begin(species, local_stream);
      host_poll_until([&] {
        const bool done = physics_.local.progress_prepare(species);
        if constexpr (Partial)
          if (!resident_sort_launched && physics_.local.resident_access_ready(species)) {
            auto range = physics_.local.acquire_resident_access(species);
            device_wait(*range.ready, sort_stream);
            sort_scope.begin(timing, TimingRegion::ParticleSort, sort_stream, species);
            physics_.local.sort_resident(species, std::move(range), sort_stream);
            resident_sort_launched = true;
          }
        return done;
      });
    }
    host_signal(worker.migration_prepared_fence, generation);
    if (!migration_bypassed) {
      LPFX_PROFILE_SCOPE("TPIC::ResidentReorder");
      host_poll_until([&] {
        const bool compacted = physics_.local.progress_compact(species);
        if constexpr (Partial) {
          lpfx::require(resident_sort_launched, "TPIC local species: missing resident sort launch");
          const bool sorted = physics_.local.progress_sort(species);
          if (sorted && compacted) sort_scope.close();
          return sorted && compacted;
        } else
          return compacted;
      });
    }
    if constexpr (Partial) {
      if (epoch.moments) {
        const auto& residents = worker.residents.emplace(physics_.local.resident_read(species));
        worker.resident_prefix = residents.prefix_size();
        offer_deposition(generation, species, DepositionPhase::resident,
                         {0, worker.resident_prefix, worker.resident_prefix});
      }
      unsigned registered_readers = 0;
      bool readers_sealed = false;
      host_poll_until([&] {
        MomentQuantity quantity;
        while (worker.resident_readers.try_receive(quantity) == success) {
          device_wait(deposition_.resident_done_event[species][unsigned(quantity)], local_stream);
          registered_readers |= 1U << unsigned(quantity);
        }
        // The checked sort is complete. This marker closes every registered
        // reader wait, including an empty reader set.
        if (!readers_sealed && (!epoch.moments || registered_readers == 3U)) {
          device_record(worker.resident_release_done_event, local_stream);
          readers_sealed = true;
        }
        return readers_sealed && routing_.routed_fence[species].ready(generation);
      });
    } else
      host_wait(routing_.routed_fence[species], generation);
    if (!migration_bypassed) {
      LPFX_PROFILE_SCOPE("TPIC::MigrationFinalize");
      if constexpr (Partial) {
        host_poll_until(
            [&] { return physics_.local.progress_release_access(species, worker.resident_release_done_event); });
        physics_.local.release_growth_scratch(species);
      }
      host_poll_until([&] { return physics_.local.progress_finalize(species); });
      host_poll_until([&] { return physics_.local.progress_conclude(species); });
      migration_scope.close();
    }
    {
      LPFX_PROFILE_SCOPE("TPIC::ParticleSort");
      if (physics_.local.full_order()) {
        auto scope = timing.scope(TimingRegion::ParticleSort, sort_stream, species);
        physics_.local.sort(species, sort_stream);
        host_poll_until([&] { return physics_.local.progress_sort(species); });
      }
    }
    const ParticleRange range = physics_.local.completed(species);
    if constexpr (Partial)
      if (epoch.moments)
        lpfx::require(range.first == worker.resident_prefix,
                      "TPIC local species: resident prefix changed during append");
    device_record(worker.owner_ready_event, physics_.local.full_order() ? sort_stream : local_stream);
    if (epoch.moments)
      offer_deposition(generation, species, Partial ? DepositionPhase::tail : DepositionPhase::whole, range);
    LPFX_PROFILE_MARK("TPIC::OwnerReady");
    host_signal(worker.owner_ready_fence, generation);
    if (!migration_bypassed) physics_.local.refresh(species);
  }

  template <bool Diagnose> void run_deposition(Worker&, const Epoch& epoch) {
    LPFX_PROFILE_SCOPE("TPIC::Deposition");
    const auto generation = epoch.generation();
    deposition_.ledger.begin(generation, epoch.moments);
    if (!epoch.moments) return;
    Timing& timing = *deposition_.timing;
    for (unsigned quantity = 0; quantity < moment_quantities; ++quantity) {
      device_wait(field_solve_.deposition_ready_event, moment_stream(quantity));
      LPFX_PROFILE_SCOPE("TPIC::ZeroBuffers");
      auto scope = timing.scope(TimingRegion::ClearMoments, moment_stream(quantity), quantity);
      deposition_.moment_validation[quantity].clear(moment_stream(quantity));
      physics_.deposition->template clear<Diagnose>(MomentQuantity(quantity), epoch.cycle, epoch.bootstrap,
                                                    moment_stream(quantity));
    }
    std::uint64_t ready_pass = 0;
    unsigned next_quantity = 0, idle_polls = 0, assembled_quantities = 0;
    bool assembling = false;
    std::array<bool, moment_quantities> moment_captured{}, moment_checked{};
    ProgressScope<Backend> assembly_scope;
    while (assembled_quantities != moment_quantities) {
      bool changed = false;
      ++ready_pass;
      DepositionInput input;
      while (deposition_.inputs.try_receive(input) == success) {
        deposition_.ledger.offer(input, ready_pass);
        changed = true;
      }
      if (epoch.bootstrap || field_advance_.next_magnetic_fence.ready(generation)) {
        for (unsigned attempt = 0; attempt < moment_quantities; ++attempt) {
          const unsigned quantity = next_quantity;
          next_quantity ^= 1U;
          const auto job = deposition_.ledger.next(MomentQuantity(quantity));
          if (!job) continue;
          deposit_job<Diagnose>(job->input, quantity, epoch);
          const bool last_host_reader = deposition_.ledger.submit(*job);
          if constexpr (Partial) {
            if (!epoch.bootstrap && job->input.phase == DepositionPhase::resident) {
              LocalSpeciesWorker& owner = local_species_[job->input.species];
              if (last_host_reader) owner.residents.reset(); // No capability use after the last reader.
              lpfx::require(owner.resident_readers.try_send(MomentQuantity(quantity)) == success,
                            "TPIC pipeline: resident reader inbox capacity violated");
            }
          }
          changed = true;
          break;
        }
      }
      for (unsigned quantity = 0; quantity < moment_quantities; ++quantity) {
        if (deposition_.ledger.all_submitted(MomentQuantity(quantity)) && !moment_captured[quantity]) {
          deposition_.moment_validation[quantity].capture(moment_stream(quantity));
          moment_captured[quantity] = changed = true;
        }
        if (moment_captured[quantity] && !moment_checked[quantity] &&
            deposition_.moment_validation[quantity].ready("TPIC deposition: invalid particle moment contribution"))
          moment_checked[quantity] = changed = true;
      }
      // R1: service every available resident release before native progress.
      // Assembly starts after routing has released the transport and runs current
      // then mass, whatever order the species complete in.
      if ((epoch.bootstrap || routing_.all_routed_fence.ready(generation)) &&
          assembled_quantities < moment_quantities) {
        const auto assembly_quantity = MomentQuantity(assembled_quantities);
        if (!assembling && moment_captured[assembled_quantities] &&
            deposition_.ledger.all_submitted(MomentQuantity::current) &&
            deposition_.ledger.all_submitted(MomentQuantity::mass)) {
          assembly_scope.begin(timing,
                               assembled_quantities ? TimingRegion::MassAssembly : TimingRegion::CurrentAssembly,
                               moment_stream(assembled_quantities));
          physics_.deposition->begin_assembly(assembly_quantity);
          assembling = changed = true;
        }
        // R1: the pack follows the flag copy on its stream; progress, which posts the
        // MPI sends and publishes the assembled moments, follows the flag read.
        const auto progress = assembling && moment_checked[assembled_quantities]
            ? physics_.deposition->progress_assembly(assembly_quantity) : communication::HaloProgress::pending;
        // R1: the fence follows the ghost phase's native call, so on NCCL/RCCL
        // every rank issues the solve's operations after both mass phases.
        if (progress == communication::HaloProgress::interior) {
          device_record(deposition_.mass_interior_event, streams_.mass);
          host_signal(deposition_.mass_interior_fence, generation);
          changed = true;
        } else if (progress == communication::HaloProgress::complete) {
          assembly_scope.close();
          device_record(deposition_.assembled_event[assembled_quantities], moment_stream(assembled_quantities));
          LPFX_PROFILE_MARK(assembled_quantities ? "TPIC::MassAssembled" : "TPIC::CurrentAssembled");
          host_signal(deposition_.assembled_fence[assembled_quantities], generation);
          ++assembled_quantities;
          assembling = false;
          changed = true;
        }
      }
      if (changed) idle_polls = 0;
      else yield_every_64(idle_polls);
    }
  }
  template <bool Diagnose> void deposit_job(const DepositionInput& job, unsigned quantity, const Epoch& epoch) {
    const Executor stream = moment_stream(quantity);
    const unsigned species = job.species;
    if (!epoch.bootstrap) device_wait(field_advance_.next_magnetic_event, stream);
    LPFX_PROFILE_SCOPE(quantity ? "TPIC::MassDeposition" : "TPIC::CurrentDeposition");
    const auto phase = job.phase == DepositionPhase::resident ? TimingPhase::resident
        : job.phase == DepositionPhase::tail ? TimingPhase::tail : TimingPhase::full;
    auto scope = deposition_.timing->scope(quantity ? TimingRegion::MassDeposition : TimingRegion::CurrentDeposition,
                                           stream, species, phase);
    if constexpr (Partial) {
      if (!epoch.bootstrap && job.phase == DepositionPhase::resident) {
        physics_.deposition->template deposit_resident<Diagnose>(
            MomentQuantity(quantity), job, *local_species_[species].residents, epoch.cycle,
            deposition_.moment_validation[quantity].data(), stream);
        device_record(deposition_.resident_done_event[species][quantity], stream);
        return;
      }
    }
    if (!epoch.bootstrap) device_wait(local_species_[species].owner_ready_event, stream);
    physics_.deposition->template deposit<Diagnose>(MomentQuantity(quantity), job, epoch.bootstrap, epoch.cycle,
                                                    deposition_.moment_validation[quantity].data(), stream);
  }
  Executor moment_stream(unsigned quantity) const { return quantity ? streams_.mass : streams_.current; }

  void run_field_solve(Worker&, const Epoch& epoch) {
    LPFX_PROFILE_SCOPE_COLOR("TPIC::FieldSolve", 0xFF3B82F6);
    const Executor solver_stream = streams_.solver;
    const auto generation = epoch.generation();
    Timing& timing = *field_solve_.timing;
    {
      auto field_solve_scope = timing.scope(TimingRegion::FieldSolve, solver_stream);
      device_wait(state_ready_event_, solver_stream);
      {
        auto scope = timing.scope(TimingRegion::RhsFields, solver_stream);
        field_solve_.solve_validation.clear(solver_stream);
        physics_.field_solve.rhs_fields();
      }
      // Solve n reads the moments of the advance that wrote state n: same g.
      host_wait(deposition_.assembled_fence[unsigned(MomentQuantity::current)], generation);
      device_wait(deposition_.assembled_event[unsigned(MomentQuantity::current)], solver_stream);
      // Unsmoothed RHS is local and can finish during the deposition's mass interval.
      if (!physics_.field_solve.smoothed()) {
        auto scope = timing.scope(TimingRegion::FinishRhs, solver_stream);
        physics_.field_solve.complete_rhs(field_solve_.solve_validation.data());
      }
      // The solve starts on the interior records of M.
      host_wait(deposition_.mass_interior_fence, generation);
      device_wait(deposition_.mass_interior_event, solver_stream);
      if (physics_.field_solve.smoothed()) {
        auto scope = timing.scope(TimingRegion::FinishRhs, solver_stream);
        physics_.field_solve.complete_rhs(field_solve_.solve_validation.data());
      }
      {
        auto scope = timing.scope(TimingRegion::SolveMaxwell, solver_stream);
        field_solve_.result = physics_.field_solve.host_blocking_solve(field_solve_.solve_validation.data(),
            [&](const Executor executor) {
              host_wait(deposition_.assembled_fence[unsigned(MomentQuantity::mass)], generation);
              device_wait(deposition_.assembled_event[unsigned(MomentQuantity::mass)], executor);
            });
        lpfx::require(field_solve_.result.converged,
                      "TPIC pipeline: GMRES did not converge; timestep was not committed");
      }
      {
        auto scope = timing.scope(TimingRegion::ParticleField, solver_stream);
        const auto& electric = physics_.field_solve.complete_particle_field(field_solve_.solve_validation.data());
        // R1: captured before the particle-field record, so the field advance reads it
        // without a wait once its own work has run.
        field_solve_.solve_validation.capture(solver_stream);
        field_solve_.particle_field = &electric;
        device_record(field_solve_.particle_field_ready_event, solver_stream);
        device_record(field_solve_.deposition_ready_event, solver_stream);
      }
    }
    LPFX_PROFILE_MARK("TPIC::FieldSolveNativeDone");
  }

  void run_diagnostics_output(Worker&, const Epoch& epoch) {
    LPFX_PROFILE_SCOPE_COLOR("TPIC::DiagnosticsOutput", 0xFFF59E0B);
    const Executor diagnostics_stream = streams_.diagnostics, helper_stream = streams_.diagnostics_helper,
                   capture_stream = streams_.output_capture;
    const auto generation = epoch.generation();
    Timing& timing = *diagnostics_output_.timing;
    for (const auto stream : {diagnostics_stream, helper_stream, capture_stream})
      device_wait(state_ready_event_, stream);
    const bool gauss = epoch.gauss_begin || epoch.gauss_finish;
    if (epoch.scalars) {
      {
        auto scope = timing.scope(TimingRegion::FieldDiagnostics, diagnostics_stream);
        physics_.diagnostics.field_energy();
      }
      for (unsigned species = 0; species < species_count_; ++species) {
        auto scope = timing.scope(TimingRegion::ParticleDiagnostics, physics_.diagnostics.executor(species), species);
        physics_.diagnostics.species_scalars(species);
      }
      device_record(diagnostics_output_.helper_sources_read_event, helper_stream);
      device_wait(diagnostics_output_.helper_sources_read_event, diagnostics_stream);
    }
    if (gauss) {
      auto scope = timing.scope(TimingRegion::GaussDiagnostics, diagnostics_stream);
      if (epoch.gauss_begin) diagnostics_output_.gauss_validation.clear(diagnostics_stream);
      physics_.diagnostics.read_gauss_state(epoch.gauss_begin, epoch.gauss_finish,
                                            diagnostics_output_.gauss_validation.data());
      if (epoch.gauss_begin) diagnostics_output_.gauss_validation.capture(diagnostics_stream);
    }
    if (epoch.reads_state()) {
      // Both lanes have recorded their final source kernels. No collective or
      // readback is needed to let physics stop borrowing the committed state.
      device_record(diagnostics_output_.sources_read_event, diagnostics_stream);
      LPFX_PROFILE_MARK("TPIC::DiagnosticSourcesRead");
    }
    unsigned first_moment_species = species_capacity;
    bool capture_pending = false;
    if (epoch.captures()) {
      LPFX_PROFILE_SCOPE("TPIC::OutputFields");
      // R1: a particle capture rewrites the images that a checkpoint still being
      // written borrows.
      if ((epoch.particles || epoch.checkpoint) && checkpoint_writer_.pending()) {
        auto scope = timing.scope(TimingRegion::Checkpoint);
        host_join(checkpoint_writer_, diagnostics_output_.checkpoint);
      }
      auto scope = timing.scope(TimingRegion::OutputFields);
      // R1: the capture packs the residual of a Gauss sample finishing here.
      if (epoch.fields && epoch.gauss_finish && physics_.input.output.output_gauss)
        device_wait(diagnostics_output_.sources_read_event, capture_stream);
      physics_.output.begin_capture(epoch);
      capture_pending = true;
      if (epoch.fields) {
        first_moment_species = physics_.output.first_moment_species();
        if (first_moment_species < species_capacity) {
          diagnostics_output_.output_validation.clear(capture_stream);
          physics_.output.deposit(first_moment_species, diagnostics_output_.output_validation.data());
          diagnostics_output_.output_validation.capture(capture_stream);
        } else
          physics_.output.detach();
      }
    }
    if (epoch.checkpoint) {
      auto scope = timing.scope(TimingRegion::Checkpoint);
      physics_.output.host_blocking_capture_checkpoint(diagnostics_output_.checkpoint,
                                                       diagnostics_output_.initial_energy);
    }
    const auto submit_capture = [&] {
      LPFX_PROFILE_SCOPE("TPIC::OutputCapturePublish");
      auto scope = timing.scope(TimingRegion::OutputCapturePublish);
      if (epoch.snapshot()) {
        // R1: the pending snapshot holds this capture's buffers; it is joined
        // only now, after the capture's device work.
        if (output_writer_.pending()) host_join(output_writer_, diagnostics_output_.capture);
        physics_.output.copy_into(diagnostics_output_.capture);
        output_writer_.submit(std::move(diagnostics_output_.capture));
      }
      if (epoch.checkpoint) {
        physics_.output.lend_checkpoint_particles(diagnostics_output_.checkpoint);
        checkpoint_writer_.submit(std::move(diagnostics_output_.checkpoint));
      }
      capture_pending = false;
    };
    if (epoch.solve && (epoch.reads_state() || epoch.captures())) {
      unsigned idle_polls = 0;
      // R1: the field solve concludes every native operation before finalize enters
      // any communication; until then only local progress runs. An epoch without
      // scalars, a Gauss sample or captures enters none.
      while (!field_solve_.done_fence.ready(generation)) {
        if (capture_pending && first_moment_species == species_capacity && physics_.output.transfer_ready())
          submit_capture();
        yield_every_64(idle_polls);
      }
    }
    if (epoch.scalars) {
      auto scope = timing.scope(TimingRegion::DiagnosticsCollect, diagnostics_stream);
      diagnostics_output_.sample = physics_.diagnostics.host_blocking_collect(epoch.cycle, epoch.final_state);
      if (epoch.cycle == first_cycle_ && !physics_.plan.restart)
        diagnostics_output_.initial_energy = data::total_energy(diagnostics_output_.sample);
    }
    diagnostics_output_.scalar_written = epoch.scalars && physics_.diagnostics.scalars_due(diagnostics_output_.sample);
    if (diagnostics_output_.scalar_written && physics_.rank == 0) {
      auto scope = timing.scope(TimingRegion::OutputDiagnostics);
      physics_.diagnostics.write_scalars(diagnostics_output_.sample, diagnostics_output_.initial_energy);
    }
    if (gauss) {
      // R1: the finished sample reads its charge slot before a new one rewrites it.
      auto scope = timing.scope(TimingRegion::GaussCollect);
      if (epoch.gauss_finish)
        diagnostics_output_.gauss_sample = physics_.diagnostics.host_blocking_collect_gauss(epoch.cycle - 1);
      if (epoch.gauss_begin) {
        diagnostics_output_.gauss_validation.host_wait_check("TPIC pipeline: invalid Gauss charge deposition");
        physics_.diagnostics.complete_gauss_charge();
      }
    }
    if (first_moment_species < species_capacity) {
      for (unsigned species = 0; species < species_count_; ++species) {
        if (!physics_.output.moments_selected(species)) continue;
        {
          LPFX_PROFILE_SCOPE("TPIC::OutputMoments");
          auto scope = timing.scope(TimingRegion::OutputMoments, capture_stream, species);
          if (species != first_moment_species) {
            diagnostics_output_.output_validation.clear(capture_stream);
            physics_.output.deposit(species, diagnostics_output_.output_validation.data());
            diagnostics_output_.output_validation.capture(capture_stream);
          }
          diagnostics_output_.output_validation.host_wait_check("TPIC pipeline: invalid output moment deposition");
          physics_.output.complete_assembly();
        }
        physics_.output.pack(species);
      }
      physics_.output.detach();
    }
    LPFX_PROFILE_MARK("TPIC::DiagnosticsOutputNativeDone");
    // R1: the physics readers are released before the piece waits for the disk
    // worker or the driver's barrier.
    host_signal(diagnostics_output_.diagnosed_fence, generation);
    if (physics_.output.publishes(epoch)) {
      {
        LPFX_PROFILE_SCOPE("TPIC::OutputWriterWait");
        auto scope = timing.scope(TimingRegion::OutputWriterWait);
        if (capture_pending) {
          physics_.output.host_wait_transfer();
          submit_capture();
        }
        output_writer_.host_wait_piece();
      }
      host_signal(diagnostics_output_.piece_ready_fence, generation);
      // R1: the root publishes only after the driver's barrier has agreed that
      // every rank wrote its piece.
      host_wait(publication_allowed_fence_, generation);
      diagnostics_output_.capture = output_writer_.complete_snapshot();
    } else if (capture_pending) {
      // R1: a snapshot that publishes nothing is written while the next cycles run.
      auto scope = timing.scope(epoch.snapshot() ? TimingRegion::OutputWriterWait : TimingRegion::Checkpoint);
      physics_.output.host_wait_transfer();
      submit_capture();
    }
    if (epoch.final_state) {
      // The last checkpoint and the last snapshot that publishes nothing.
      if (checkpoint_writer_.pending()) host_join(checkpoint_writer_, diagnostics_output_.checkpoint);
      if (output_writer_.pending()) host_join(output_writer_, diagnostics_output_.capture);
      run_result_.diagnostics_path = physics_.diagnostics.close_scalars();
      run_result_.gauss_path = physics_.diagnostics.close_gauss();
      const bool root = physics_.rank == 0;
      run_result_.fields_path = root && physics_.output.fields_enabled() ? output_writer_.fields_path() : "";
      run_result_.particles_path =
          root && physics_.output.particles_enabled() ? output_writer_.particles_path() : "";
      // The checkpoint engine names the directory of its last checkpoint.
      run_result_.checkpoint_path =
          root && physics_.output.checkpoints_enabled() ? checkpoint_writer_.fields_path() : "";
    }
  }

  // ---- borrowed
  Physics& physics_;
  ExecutionStreams<Backend>& stream_registry_;                 // the drain synchronizes every registered stream
  const StreamExecutors<Backend, species_capacity>& streams_; // the stream table
  const unsigned species_count_;
  const int first_cycle_; // the state the run starts from: 0, or a restart's checkpoint
  const int cycle_count_;
  // ---- main: the epochs (main is their only reader) and what main signals and records
  Epoch advance_epoch_{}, solve_epoch_{};
  bool advance_active_{};
  Event state_ready_event_;
  lpfx::HostFence publication_allowed_fence_;
  // ---- the workers, each owning what it produces; device events exist for the species of the run only
  MoverWorker mover_;
  FieldAdvanceWorker field_advance_;
  RoutingWorker routing_;
  DepositionWorker deposition_;
  std::array<LocalSpeciesWorker, species_capacity> local_species_;
  FieldSolveWorker field_solve_;
  DiagnosticsOutputWorker diagnostics_output_;
  std::array<Worker*, ExecutionLayout::workers(species_capacity)> workers_{}; // advance workers, then solve workers
  static_assert(ExecutionLayout::host_fences(species_capacity) ==
                9 + 2 + ExecutionLayout::workers(species_capacity) + 4 * species_capacity);
  // ---- main's timing, report and result
  Timing main_timing_;
  RunReporter reporter_;
  RunResult run_result_{};
  Clock::time_point bootstrap_start_{};
  double bootstrap_seconds_{};
  // ---- the writer threads: destroyed before the captures of diagnostics_output they return
  typename Physics::Writer output_writer_, checkpoint_writer_;
};

} // namespace tpic::simulation

#endif // TPIC_SIMULATION_DRIVER_PIPELINE_HPP
