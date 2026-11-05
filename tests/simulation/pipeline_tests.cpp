#include <lpfx/backend/traits.hpp>

#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <map>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <sys/wait.h>
#include <unistd.h>

// The real Pipeline runs on host threads against a logging backend and a fake
// physics owner: every call, event record and stream wait lands in one log per
// runtime, and the scenarios check that log against the R1 order and the
// generation-less event reuse chains.
namespace pipeline_test {
enum class Kind { call, record, wait };
struct Entry {
  Kind kind;
  unsigned what, arg, records; // call: (Call, species or quantity); event: (event id, stream)
  std::thread::id thread;
};
// The indices of the entries that match the key's kind, what and arg.
inline std::vector<std::size_t> where(const std::vector<Entry>& log, const Entry& key) {
  std::vector<std::size_t> indices;
  for (std::size_t i = 0; i < log.size(); ++i)
    if (log[i].kind == key.kind && log[i].what == key.what && log[i].arg == key.arg) indices.push_back(i);
  return indices;
}
struct Log;
struct Event {
  unsigned id;
  Log* log;
  unsigned records{};
};
struct Log {
  std::mutex mutex;
  std::vector<Entry> entries;
  void add(Kind kind, unsigned what, unsigned arg, unsigned records = 0) {
    std::lock_guard lock(mutex);
    entries.push_back({kind, what, arg, records, std::this_thread::get_id()});
  }
  // A wait before any record has nothing to wait on: the backend reports a failure.
  bool event(Kind kind, Event& event, int stream) {
    std::lock_guard lock(mutex);
    if (kind == Kind::record) ++event.records;
    entries.push_back({kind, event.id, unsigned(stream), event.records, std::this_thread::get_id()});
    return event.records != 0;
  }
};
// Events take their ids in creation order on the creating thread.
inline thread_local Log* creating_log{};
inline thread_local unsigned next_event{};
inline std::atomic<int> next_stream{10};
inline std::atomic<unsigned> selections{};
inline std::atomic<bool> fail_selection{};
} // namespace pipeline_test

namespace lpfx::device {
struct HostBackend {};
template <> struct BackendTraits<HostBackend> {
  using stream_type = int;
  using event_type = pipeline_test::Event*;
  using error_type = int;
  static constexpr int null_stream = 0, success_error = 0;
  static constexpr event_type null_event = nullptr;
  static bool succeeded(int error) noexcept { return error == 0; }
  static const char* native_error_name(int) noexcept { return nullptr; }
  static const char* native_error_string(int) noexcept { return nullptr; }
  static int create_stream(int& stream) noexcept { stream = ++pipeline_test::next_stream; return 0; }
  static int create_stream(int& stream, int) noexcept { return create_stream(stream); }
  static int query_stream_priority_range(int& least, int& greatest) noexcept { least = 0; greatest = -1; return 0; }
  static int destroy_stream(int) noexcept { return 0; }
  static int synchronize_stream(int) noexcept { return 0; }
  static bool is_default_stream(int stream) noexcept { return stream == 0; }
  static int query_stream_nonblocking(int, bool& nonblocking) noexcept { nonblocking = true; return 0; }
  static int query_stream_device_ordinal(int, std::int32_t& ordinal) noexcept { ordinal = 0; return 0; }
  static bool runtime_unavailable(int) noexcept { return false; }
  static int query_device_count(int& count) noexcept { count = 1; return 0; }
  static int select_device(int) noexcept {
    ++pipeline_test::selections;
    return pipeline_test::fail_selection ? 1 : 0;
  }
  static int query_current_device_memory(DeviceMemoryInfo&) noexcept { return 1; }
  static int query_current_device_properties(DeviceProperties&) noexcept { return 1; }
  static int query_current_device_identity(DeviceIdentity&, std::int32_t&) noexcept { return 1; }
  static int query_device_pointer_ordinal(const void*, std::int32_t&) noexcept { return 1; }
  static int create_event(event_type& event) noexcept {
    event = new pipeline_test::Event{pipeline_test::next_event++, pipeline_test::creating_log};
    return 0;
  }
  static int create_timing_event(event_type& event) noexcept { return create_event(event); }
  static int destroy_event(event_type event) noexcept { delete event; return 0; }
  static int record_event(event_type event, int stream) noexcept {
    event->log->event(pipeline_test::Kind::record, *event, stream);
    return 0;
  }
  static int stream_wait_event(int stream, event_type event) noexcept {
    return event->log->event(pipeline_test::Kind::wait, *event, stream) ? 0 : 1;
  }
  static int query_event(event_type) noexcept { return 0; }
  static int synchronize_event(event_type) noexcept { return 0; }
  static int event_elapsed_time(float& milliseconds, event_type, event_type) noexcept {
    milliseconds = 0;
    return 0;
  }
  static bool not_ready(int) noexcept { return false; }
};
} // namespace lpfx::device

#include <tpic/simulation/driver/pipeline.hpp>

namespace {
using namespace pipeline_test;
namespace vs = tpic::simulation;
using namespace std::chrono_literals;
using B = lpfx::device::HostBackend;
using Exec = lpfx::device::Executor<B>;
using DeviceEvent = lpfx::device::DeviceEvent<B>;
using Order = tpic::config::ParticleOrder;
using vs::MomentQuantity;
constexpr std::size_t Ns = 3;
using X = vs::StreamExecutors<B, Ns>;
constexpr int fake_abort_status = 86;

void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

// The calls before mpi_end carry communication: two ranks must issue them in
// one order whatever their local readiness.
enum class Call : unsigned {
  advance, progress_cell, progress_route, begin_assembly, progress_assembly,
  host_blocking_solve, complete_particle_field, host_blocking_collect, host_blocking_collect_gauss,
  complete_gauss_charge, complete_assembly, host_blocking_barrier, host_blocking_max_over_ranks, host_blocking_check_population,
  mpi_end,
  prepare = mpi_end, host_blocking_report_memory, close_transport, complete_initial_halos, close_cell, commit,
  host_blocking_center, apply_initial_boundaries, sort, classify_initial, host_wait_retired, begin,
  progress_prepare, acquire, sort_resident, host_wait_resident, host_wait_sorted, host_wait_route,
  release_growth_scratch, close_empty_readers, progress_release_access, host_wait_finalize, refresh, finish_initial,
  retire_after, progress_retirement, progress_compact, progress_sort, resident_read, release_residents,
  progress_finalize, progress_conclude, move, clear, deposit_resident, deposit, rhs_fields,
  complete_rhs, open_scalars, field_energy, species_scalars, write_scalars, close_scalars, read_gauss_state,
  close_gauss,
  device_wait_sources, begin_capture, output_deposit, detach, copy_into, pack, host_wait_transfer, submit,
  host_wait_piece, complete_snapshot
};
struct Delay {
  Call what;
  unsigned arg;
  std::chrono::microseconds time;
};
struct Config {
  Order order = Order::partial;
  bool snapshots{};
  int ncycles{};
  bool advance = true;
  unsigned species = 2;
  int rank = 1;
  std::vector<Delay> delays;
  int fail_piece = -1;
  bool fail_solve{}, fail_collect{};
  // A snapshot that publishes nothing is joined by the next capture; with
  // alternate, only the snapshots of odd cycles publish.
  bool publishes = true, alternate{};
  int energy_cycle = 1;
  int gauss_cycle = 0;
};

struct Solved {
  int iterations{};
  float relative_residual{};
  bool converged{};
};
using SolverField = int;
using Capture = int;
// Always open: numerical validity is not under test.
struct Validation {
  void clear(Exec) {}
  int data() { return 0; }
  void capture(Exec) {}
  void host_wait_check(const char*) {}
  bool ready(const char*) { return true; }
  void device_wait(Exec) {}
};
// The host capability over resident particles; its destruction is the release.
struct Residents {
  Log* log;
  unsigned species;
  std::uint32_t prefix;
  Residents(Log* l, unsigned s, std::uint32_t p) : log(l), species(s), prefix(p) {}
  Residents(Residents&& other) noexcept
      : log(std::exchange(other.log, nullptr)), species(other.species), prefix(other.prefix) {}
  ~Residents() {
    if (log) log->add(Kind::call, unsigned(Call::release_residents), species);
  }
  std::uint32_t prefix_size() const { return prefix; }
};
struct ResidentAccess {
  const DeviceEvent* ready;
};
// Every action of the physics owner: it logs each call after its delay.
struct Fake {
  // Thread-free: the piece completes in the call. Before the piece of a cycle
  // that an advance follows, it waits for that advance's commit, which only a
  // reader release published ahead of the piece wait allows.
  struct Writer {
    Fake* fake{};
    int pieces{};
    bool held{};
    template <class Setup> void start(Setup setup) { setup(); }
    bool pending() const { return held; }
    void submit(Capture&&) {
      require(!held, "submit while a snapshot is pending");
      fake->call(Call::submit);
      held = true;
    }
    void host_wait_piece() {
      fake->call(Call::host_wait_piece);
      require(pieces != fake->c.fail_piece, "disk failure");
      const auto deadline = std::chrono::steady_clock::now() + 5s;
      const auto commits = [&] {
        std::lock_guard lock(fake->log.mutex);
        return where(fake->log.entries, {Kind::call, unsigned(Call::commit), 0, 0, {}}).size();
      };
      if (pieces < fake->c.ncycles)
        while (commits() <= std::size_t(pieces)) {
          require(std::chrono::steady_clock::now() < deadline, "the piece wait held the physics readers");
          std::this_thread::yield();
        }
    }
    Capture complete_snapshot() {
      fake->call(Call::complete_snapshot);
      ++pieces;
      held = false;
      return {};
    }
    std::string fields_path() const { return "fields"; }
    std::string particles_path() const { return "particles"; }
  };
  Log& log;
  const Config& c;
  const X& x;
  std::array<DeviceEvent, Ns> resident_ready;
  DeviceEvent closed;
  SolverField field{};
  Fake(Log& l, const Config& config, const X& executors) : log(l), c(config), x(executors) {
    for (unsigned s = 0; s < c.species; ++s) lpfx::device::create_event(resident_ready[s], "fake event");
    lpfx::device::create_event(closed, "fake event");
  }
  void call(Call what, unsigned arg = 0) {
    for (const auto& delay : c.delays)
      if (delay.what == what && delay.arg == arg) std::this_thread::sleep_for(delay.time);
    log.add(Kind::call, unsigned(what), arg);
  }
  bool polled(Call what, unsigned arg = 0) {
    call(what, arg);
    return true;
  }
  vs::ParticleRange completed(unsigned s) const {
    return c.order == Order::partial ? vs::ParticleRange{3 + s, 2, 5 + s} : vs::ParticleRange{0, 5 + s, 5 + s};
  }
  // field advance
  void complete_initial_halos(bool) { call(Call::complete_initial_halos); }
  void advance(int) { call(Call::advance); }
  vs::FieldAdvanceProgress progress_cell() { call(Call::progress_cell); return {true, true}; }
  void close_cell(int) { call(Call::close_cell); }
  void commit() { call(Call::commit); }
  // mover
  void move(unsigned s, const SolverField&, int, Exec) { call(Call::move, s); }
  // local species and routing
  void host_blocking_center(unsigned s) { call(Call::host_blocking_center, s); }
  bool full_order() const { return c.order == Order::full; }
  bool bypassed() const { return false; }
  void host_blocking_check_population() { call(Call::host_blocking_check_population); }
  void apply_initial_boundaries(unsigned s, int, Exec) { call(Call::apply_initial_boundaries, s); }
  void sort(unsigned s, Exec) { call(Call::sort, s); }
  void classify_initial(unsigned s, Exec) { call(Call::classify_initial, s); }
  void host_wait_retired(unsigned s, Exec) { call(Call::host_wait_retired, s); }
  void begin(unsigned s, Exec) { call(Call::begin, s); }
  bool progress_prepare(unsigned s) { return polled(Call::progress_prepare, s); }
  bool resident_access_ready(unsigned) const { return true; }
  ResidentAccess acquire_resident_access(unsigned s) {
    call(Call::acquire, s);
    lpfx::device::record_event(resident_ready[s], x.local[s], "fake record");
    return {&resident_ready[s]};
  }
  void sort_resident(unsigned s, ResidentAccess, Exec) { call(Call::sort_resident, s); }
  void host_wait_resident(unsigned s) { call(Call::host_wait_resident, s); }
  void host_wait_sorted(unsigned s) { call(Call::host_wait_sorted, s); }
  void host_wait_route(unsigned s) { call(Call::host_wait_route, s); }
  void release_growth_scratch(unsigned s) { call(Call::release_growth_scratch, s); }
  const DeviceEvent& close_empty_readers(unsigned s, Exec) { call(Call::close_empty_readers, s); return closed; }
  bool progress_release_access(unsigned s, const DeviceEvent&) { return polled(Call::progress_release_access, s); }
  bool host_wait_finalize(unsigned s) { return polled(Call::host_wait_finalize, s); }
  void refresh(unsigned s) { call(Call::refresh, s); }
  void finish_initial(unsigned s) { call(Call::finish_initial, s); }
  void retire_after(unsigned s, const DeviceEvent&, Exec) { call(Call::retire_after, s); }
  bool progress_retirement(unsigned s) { return polled(Call::progress_retirement, s); }
  bool progress_compact(unsigned s) { return polled(Call::progress_compact, s); }
  bool progress_sort(unsigned s) { return polled(Call::progress_sort, s); }
  Residents resident_read(unsigned s) { call(Call::resident_read, s); return {&log, s, completed(s).first}; }
  bool progress_route(unsigned s) { return polled(Call::progress_route, s); }
  bool progress_finalize(unsigned s) { return polled(Call::progress_finalize, s); }
  bool progress_conclude(unsigned s) { return polled(Call::progress_conclude, s); }
  // deposition
  template <bool Diagnose> void clear(MomentQuantity q, int, bool, Exec) { call(Call::clear, unsigned(q)); }
  void begin_assembly(MomentQuantity q) {
    call(Call::begin_assembly, unsigned(q));
    mass_ghosts_pending = q == MomentQuantity::mass;
  }
  // The mass reaches its interior, whose call issues the ghost phase, before
  // it completes; the current completes at once.
  tpic::communication::HaloProgress progress_assembly(MomentQuantity q) {
    using tpic::communication::HaloProgress;
    if (q == MomentQuantity::mass && !mass_ghosts_pending) return HaloProgress::complete;
    call(Call::progress_assembly, unsigned(q));
    mass_ghosts_pending = false;
    return q == MomentQuantity::mass ? HaloProgress::interior : HaloProgress::complete;
  }
  bool mass_ghosts_pending{};
  template <bool Diagnose>
  void deposit_resident(MomentQuantity, const vs::DepositionInput& job, const Residents&, int, int, Exec) {
    call(Call::deposit_resident, job.species);
  }
  template <bool Diagnose> void deposit(MomentQuantity, const vs::DepositionInput& job, bool, int, int, Exec) {
    call(Call::deposit, job.species);
  }
  // field solve
  void rhs_fields() { call(Call::rhs_fields); }
  bool smoothed() const { return c.snapshots; }
  void complete_rhs(int) { call(Call::complete_rhs); }
  // The first apply orders its ghost-reading mass kernel after the mass ghosts.
  template <class HostWaitMassGhosts> Solved host_blocking_solve(int, const HostWaitMassGhosts& host_wait_mass_ghosts) {
    host_wait_mass_ghosts(x.solver_mass);
    call(Call::host_blocking_solve);
    return {3, 1.e-9f, !c.fail_solve};
  }
  const SolverField& complete_particle_field(int) { call(Call::complete_particle_field); return field; }
  // diagnostics
  void open_scalars(const std::optional<tpic::data::PreparedRestart>&) { call(Call::open_scalars); }
  void field_energy() { call(Call::field_energy); }
  Exec executor(unsigned) const { return x.diagnostics; }
  void species_scalars(unsigned s) { call(Call::species_scalars, s); }
  tpic::data::ScalarDiagnosticSample host_blocking_collect(int, bool) {
    call(Call::host_blocking_collect);
    require(!c.fail_collect, "diagnostics failure");
    return {};
  }
  bool scalars_due(const tpic::data::ScalarDiagnosticSample&) const { return true; }
  void write_scalars(const tpic::data::ScalarDiagnosticSample&, double) { call(Call::write_scalars); }
  std::string close_scalars() { call(Call::close_scalars); return "scalars"; }
  void read_gauss_state(bool, bool, int) { call(Call::read_gauss_state); }
  void complete_gauss_charge() { call(Call::complete_gauss_charge); }
  tpic::data::GaussSample host_blocking_collect_gauss(int cycle) {
    call(Call::host_blocking_collect_gauss);
    return {cycle, 0, {}};
  }
  std::string close_gauss() { call(Call::close_gauss); return {}; }
  // output
  bool enabled() const { return c.snapshots && c.advance; }
  void configure(Writer& writer) { writer.fake = this; }
  void configure_checkpoints(Writer&) {}
  bool fields_due(int) const { return c.snapshots; }
  bool particles_due(int) const { return false; }
  bool checkpoint_due(int, bool = false) const { return false; }
  bool fields_enabled() const { return c.snapshots; }
  bool publishes(const vs::Epoch& epoch) const {
    return epoch.snapshot() && (c.publishes || (c.alternate && epoch.cycle % 2 == 1));
  }
  bool particles_enabled() const { return false; }
  bool checkpoints_enabled() const { return false; }
  void device_wait_sources(Exec executor) { call(Call::device_wait_sources, unsigned(executor.stream())); }
  void begin_capture(const vs::Epoch&) { call(Call::begin_capture); }
  void host_blocking_capture_checkpoint(Capture&, double) {}
  void lend_checkpoint_particles(Capture&) {}
  unsigned first_moment_species() const { return 0; }
  void deposit(unsigned s, int) { call(Call::output_deposit, s); }
  void detach() { call(Call::detach); }
  void copy_into(Capture&) { call(Call::copy_into); }
  bool transfer_ready() const { return true; }
  bool moments_selected(unsigned) const { return true; }
  void complete_assembly() { call(Call::complete_assembly); }
  void pack(unsigned s) { call(Call::pack, s); }
  void host_wait_transfer() { call(Call::host_wait_transfer); }
  void close() { call(Call::close_transport); }
};

template <bool Partial> struct FakeSimulation {
  using Backend = B;
  using Validation = ::Validation;
  using Writer = Fake::Writer;
  using Capture = ::Capture;
  using Solved = ::Solved;
  using SolverField = ::SolverField;
  using Residents = ::Residents;
  static constexpr std::size_t Ns = ::Ns;
  static constexpr bool partial = Partial;

  FakeSimulation(Log& log, const Config& c, const X& x)
      : input{{c.ncycles, .1}, {c.order}, {.diagnostic_energy_cycle = c.energy_cycle, .gauss_cycle = c.gauss_cycle}},
        plan{{}, c.advance ? vs::RunMode::advance : vs::RunMode::initialize_only},
        rank(c.rank), species_count(c.species), diagnose(c.snapshots), actions(log, c, x) {}

  void prepare() { actions.call(Call::prepare); }
  void host_blocking_barrier() { actions.call(Call::host_blocking_barrier); }
  std::array<double, 2> host_blocking_max_over_ranks(std::array<double, 2> seconds) {
    actions.call(Call::host_blocking_max_over_ranks);
    return seconds;
  }
  void host_blocking_report_memory() { actions.call(Call::host_blocking_report_memory); }
  [[noreturn]] void abort(const char* message) noexcept {
    std::fprintf(stderr, "fake abort: %s\n", message);
    std::_Exit(fake_abort_status);
  }

  struct {
    struct { int ncycles; double dt; } parameters;
    struct { Order order; } particles;
    tpic::config::PreparedOutput output;
  } input;
  struct {
    vs::RuntimeDebugOptions debug;
    vs::RunMode mode;
    std::optional<tpic::data::PreparedRestart> restart{};
    int first_cycle() const { return 0; }
  } plan;
  lpfx::device::DeviceProperties properties{};
  int rank, ranks = 1;
  unsigned species_count;
  bool diagnose;
  Fake actions;
  Fake &mover = actions, &local = actions, &field_solve = actions, &diagnostics = actions, &output = actions,
       &transport = actions;
  Fake *field_advance = &actions, *deposition = &actions;
};

// Runs one Pipeline to completion, then checks it while its streams live.
template <bool Partial, class Check> void run(const Config& c, Log& log, Check&& check) {
  creating_log = &log;
  next_event = 1000;
  vs::ExecutionStreams<B> streams(0);
  int control = 0;
  (void)lpfx::device::BackendTraits<B>::create_stream(control);
  const X x(streams, streams.admit(Exec{control}, "control"), c.snapshots && c.advance, c.order != Order::unsorted,
            c.species);
  FakeSimulation<Partial> p(log, c, x);
  next_event = 0;
  const auto result = vs::Pipeline<FakeSimulation<Partial>>(p, streams, x).execute();
  check(result, x, next_event);
}

Entry call(Call what, unsigned arg = 0) { return {Kind::call, unsigned(what), arg, 0, {}}; }
Entry recorded(unsigned event, Exec executor) { return {Kind::record, event, unsigned(executor.stream()), 0, {}}; }
Entry waited(unsigned event, Exec executor) { return {Kind::wait, event, unsigned(executor.stream()), 0, {}}; }
// The (k + shift)-th occurrence of a precedes the k-th occurrence of b.
void order(const std::vector<Entry>& log, const Entry& a, const Entry& b, int shift, const char* message) {
  const auto first = where(log, a), second = where(log, b);
  for (std::size_t k = 0; k < second.size(); ++k) {
    const long j = long(k) + shift;
    if (j >= 0 && j < long(first.size())) require(first[std::size_t(j)] < second[k], message);
  }
}

// The Pipeline's event ids follow its creation order: ten shared events, then
// per species moved_event, owner_ready_event and in partial order resident_release_done_event
// and the two resident_done_event. The fake's resident-access events are 1000 + s.
enum SharedEvent : unsigned {
  state_ready, particle_field_ready, deposition_ready, helper_read, sources_read, new_b, commit_done,
  current_assembled, mass_assembled, mass_interior, shared_events
};
unsigned moved_event(bool partial, unsigned s) { return shared_events + s * (partial ? 5 : 2); }
unsigned owner_ready_event(bool partial, unsigned s) { return moved_event(partial, s) + 1; }

template <bool Partial> void verify(const Config& c, const std::vector<Entry>& log, const vs::RunResult& result,
                                    const X& x, unsigned events) {
  const unsigned S = c.species;
  const int N = c.advance ? c.ncycles : 0;
  const std::size_t moments = N ? std::size_t(N) : 1, solves = c.advance ? std::size_t(N) + 1 : 0;
  const bool snapshot = c.snapshots && c.advance, full = c.order == Order::full;
  const auto n = [&](const Entry& key) { return where(log, key).size(); };
  // Scalars: the energy reports, the first state and the final state. A Gauss
  // sample begins at a multiple of gauss_cycle below N and finishes at the next
  // solve; the cycles that read the state release it before their advance.
  std::vector<std::size_t> scalar_cycles, read_cycles;
  const auto begins = [&](int cycle) { return c.gauss_cycle > 0 && cycle < N && cycle % c.gauss_cycle == 0; };
  std::size_t gauss_begins = 0, gauss_finishes = 0, gauss_reads = 0;
  for (int cycle = 0; cycle <= N; ++cycle) {
    const bool begin = begins(cycle), finish = cycle > 0 && begins(cycle - 1);
    gauss_begins += begin;
    gauss_finishes += finish;
    gauss_reads += begin || finish;
    if (cycle == N) break;
    const bool scalar = cycle == 0 || (c.energy_cycle > 0 && cycle % c.energy_cycle == 0);
    if (scalar) scalar_cycles.push_back(std::size_t(cycle));
    if (scalar || begin || finish) read_cycles.push_back(std::size_t(cycle));
  }
  const std::size_t scalars = (solves != 0) + scalar_cycles.size();
  // S + 6 workers and the two writer threads select the device.
  require(events == moved_event(Partial, S) && selections == S + 8,
          "a thread missed the device setup, or workers or events exist for species outside the run");
  require(result.steps_completed == N && result.diagnostic_samples == scalars &&
              result.gauss_samples == gauss_finishes && result.field_samples == (snapshot ? solves : 0),
          "run result disagrees with the admitted cycles");
  require(n(call(Call::read_gauss_state)) == gauss_reads && n(call(Call::complete_gauss_charge)) == gauss_begins &&
              n(call(Call::host_blocking_collect_gauss)) == gauss_finishes,
          "a Gauss step ran a wrong number of times");
  // R1: a sample finishing in a solve reads the charge slot before the one beginning there rewrites it.
  order(log, call(Call::host_blocking_collect_gauss), call(Call::complete_gauss_charge), -1,
        "a Gauss sample began before the previous one finished");
  // Bootstrap, N advances of which the last one has no moments, N solves and
  // the final solve without field solve.
  for (const unsigned q : {0U, 1U})
    require(n(call(Call::clear, q)) == moments && n(call(Call::begin_assembly, q)) == moments &&
                n(call(Call::progress_assembly, q)) == moments,
            "moments cleared or assembled outside the moment advances");
  require(n(call(Call::advance)) == std::size_t(N) && n(call(Call::commit)) == std::size_t(N) &&
              n(call(Call::host_blocking_solve)) == std::size_t(N) && n(call(Call::host_blocking_collect)) == scalars &&
              n(call(Call::field_energy)) == scalars &&
              n(call(Call::host_wait_piece)) == (snapshot ? solves : 0) &&
              n(call(Call::host_blocking_barrier)) ==
                  (!snapshot ? 0 : c.publishes ? solves : c.alternate ? solves / 2 : 0) &&
              n(call(Call::host_blocking_check_population)) == (c.advance ? 2U : 1U) &&
              n(call(Call::close_transport)) == 1,
          "a step ran a wrong number of times");
  require(n(waited(current_assembled, x.control)) == (N == 0) && n(waited(mass_assembled, x.control)) == (N == 0),
          "only a run without advances joins the bootstrap moments on control");
  // R1: each solve starts on the interior records of M, recorded before the
  // assembled ones; its first apply waits the assembled ghosts.
  require(n(recorded(mass_interior, x.mass)) == moments && n(waited(mass_interior, x.solver)) == std::size_t(N) &&
              n(waited(mass_assembled, x.solver_mass)) == std::size_t(N),
          "the solve waited the mass interior or ghosts on another stream or count");
  order(log, recorded(mass_interior, x.mass), recorded(mass_assembled, x.mass), 0, "mass ghosts before its interior");
  order(log, recorded(mass_interior, x.mass), waited(mass_interior, x.solver), 0, "solve before the mass interior");
  order(log, waited(mass_interior, x.solver), waited(mass_assembled, x.solver_mass), 0,
        "first apply waited the mass ghosts before the solve waited the interior");
  order(log, recorded(mass_assembled, x.mass), waited(mass_assembled, x.solver_mass), 0,
        "first apply read the mass ghosts before their record");
  require(!full || x.sort[0].stream() != x.local[0].stream(), "full order shares the sort and local streams");
  for (unsigned s = 0; s < S; ++s) {
    require(n(call(Call::move, s)) == std::size_t(N), "species moved outside the advances");
    // R1: movers follow the first B submission; the commit follows every moved_event record.
    order(log, call(Call::progress_cell), call(Call::move, s), 0, "mover before the first B submission");
    order(log, recorded(moved_event(Partial, s), x.move), call(Call::commit), 0, "commit before a moved_event record");
    // Only the last species' moved_event is waited by the commit; local[s] waits its own.
    const std::size_t commit_waits = s + 1 == S ? std::size_t(N) : 0;
    std::size_t waits = 0;
    for (int stream = 0; stream <= next_stream; ++stream) waits += n(waited(moved_event(Partial, s), Exec{stream}));
    require(n(waited(moved_event(Partial, s), x.local[s])) == std::size_t(N) &&
                n(waited(moved_event(Partial, s), x.field_update)) == commit_waits && waits == N + commit_waits,
            "moved_event waited on another stream");
    // R1: routes follow the B submission in species order; assembly follows every route.
    order(log, call(Call::progress_cell), call(Call::progress_route, s), 0, "route before the B submission");
    if (s)
      order(log, call(Call::progress_route, s - 1), call(Call::progress_route, s), 0, "routes out of species order");
    order(log, call(Call::progress_route, s), call(Call::begin_assembly), -1, "assembly before the transport release");
    if constexpr (Partial) {
      require(n(call(Call::resident_read, s)) == moments - 1 && n(call(Call::release_residents, s)) == moments - 1,
              "resident capabilities not released once per moment advance");
      order(log, call(Call::release_residents, s), call(Call::begin_assembly), -1,
            "assembly progress before a resident release");
    }
    require(n(recorded(owner_ready_event(Partial, s), full ? x.sort[s] : x.local[s])) == std::size_t(N),
            "owner_ready_event not recorded on the ordering stream");
    // R1: the snapshot barrier follows the join of the advance that the solve admitted.
    order(log, call(Call::refresh, s), call(Call::host_blocking_barrier), 1, "barrier before the local species joined");
  }
  order(log, call(Call::begin_assembly, 0), call(Call::progress_assembly, 0), 0, "current assembly order");
  order(log, call(Call::progress_assembly, 0), call(Call::begin_assembly, 1), 0, "mass assembly before current");
  order(log, call(Call::complete_particle_field), call(Call::host_blocking_collect), 0,
        "collect before the field solve concluded");
  order(log, call(Call::host_wait_piece), call(Call::host_blocking_barrier), 0, "barrier before the piece");
  order(log, call(Call::commit), call(Call::host_blocking_barrier), 0, "barrier before the field advance joined");
  order(log, call(Call::progress_assembly, 1), call(Call::host_blocking_barrier), N ? 1 : 0,
        "barrier before the deposition joined");
  // With alternate publication the k-th completion may belong to an unpublished snapshot.
  if (c.publishes)
    order(log, call(Call::host_blocking_barrier), call(Call::complete_snapshot), 0, "publication before the barrier");
  // R1: a snapshot that publishes nothing is written while its cycle's advance commits.
  if (!c.publishes && !c.alternate)
    order(log, call(Call::commit), call(Call::host_wait_piece), 0, "an unpublished snapshot was joined in its cycle");
  // The source releases of the epochs with scalars are enqueued before the
  // first field or particle write of the following advance.
  const auto advances = where(log, call(Call::advance)), moves = where(log, call(Call::move));
  for (const Exec executor : {x.field_update, x.move}) {
    const auto releases = where(log, waited(sources_read, executor));
    require(releases.size() == read_cycles.size(), "a source release is missing");
    for (std::size_t j = 0; j < releases.size(); ++j)
      require(releases[j] < advances[read_cycles[j]] && releases[j] < moves[read_cycles[j]],
              "field advance or mover before the source release");
    order(log, call(Call::device_wait_sources, unsigned(executor.stream())), call(Call::advance), 0,
          "field advance before the snapshot source release");
  }
  // The j-th wait of an event on a stream sees the record of its own
  // generation, with m waits per generation on that stream.
  std::map<std::pair<unsigned, unsigned>, unsigned> seen;
  for (const auto& e : log) {
    if (e.kind != Kind::wait) continue;
    const unsigned m = e.what == new_b ? S * (Partial ? 2 : 1)
        : e.what == state_ready && e.arg == unsigned(x.diagnostics.stream()) && !snapshot ? 2 : 1;
    require(e.records == seen[{e.what, e.arg}]++ / m + 1, "a wait saw another generation's record");
  }
}

template <bool Partial>
void order_matrix(Order order, bool snapshots, int ncycles, bool advance = true, bool publishes = true,
                  bool alternate = false, int energy_cycle = 1, int gauss_cycle = 0) {
  Config c{.order = order, .snapshots = snapshots, .ncycles = ncycles, .advance = advance,
           .species = ncycles == 1 && !snapshots ? 1U : 2U,
           .delays = {{Call::progress_compact, 0, 300us}, {Call::progress_cell, 0, 300us},
                      {Call::progress_route, 0, 300us}, {Call::progress_assembly, 1, 300us}},
           .publishes = publishes, .alternate = alternate, .energy_cycle = energy_cycle,
           .gauss_cycle = gauss_cycle};
  // Longer than the progress_cell delay: only the mover.all_moved_fence wait orders the commit.
  c.delays.push_back({Call::move, c.species - 1, 1ms});
  Log log;
  selections = 0;
  run<Partial>(c, log, [&](const vs::RunResult& result, const X& x, unsigned events) {
    verify<Partial>(c, log.entries, result, x, events);
  });
}

// Two ranks reach their local work in opposite orders; their
// communication calls must still be issued in one sequence.
void opposite_readiness() {
  std::array<Log, 2> logs;
  const auto rank = [&](unsigned i) {
    Config c{.snapshots = true, .ncycles = 3, .species = 3, .rank = 0,
             .delays = {{Call::progress_compact, i, 2ms}, {Call::deposit_resident, i, 1ms},
                        {Call::deposit, 2 - i, 1ms}}};
    run<true>(c, logs[i], [](const vs::RunResult&, const X&, unsigned) {});
  };
  std::thread peer(rank, 1U);
  rank(0);
  peer.join();
  std::array<std::vector<std::pair<unsigned, unsigned>>, 2> sequence;
  for (unsigned i = 0; i < 2; ++i)
    for (const auto& e : logs[i].entries)
      if (e.kind == Kind::call && e.what < unsigned(Call::mpi_end)) sequence[i].push_back({e.what, e.arg});
  require(!sequence[0].empty() && sequence[0] == sequence[1], "local readiness reordered communication");
}

// A failure after admission ends in the physics owner's abort: the scenario
// runs in a child process, which must exit with the fake abort status.
template <class Scenario> void aborts(Scenario&& scenario, const char* message) {
  std::fflush(nullptr);
  const pid_t child = fork();
  require(child >= 0, "abort expectation could not fork");
  if (child == 0) {
    try { scenario(); } catch (...) {}
    std::_Exit(0);
  }
  int status = 0;
  require(waitpid(child, &status, 0) == child, "abort expectation lost its child");
  require(WIFEXITED(status) && WEXITSTATUS(status) == fake_abort_status, message);
}
void failures() {
  const auto fails = [](Config c, const char* message, bool setup = false) {
    aborts([&] {
      fail_selection = setup;
      Log log;
      run<true>(c, log, [](const vs::RunResult&, const X&, unsigned) {});
    }, message);
  };
  const Config base{.snapshots = true, .ncycles = 1};
  auto c = base;
  c.fail_solve = true;
  fails(c, "field-solve failure did not abort the run");
  c = base;
  c.fail_collect = true;
  fails(c, "diagnostics failure did not abort the run");
  for (const bool publishes : {true, false})
    for (const int piece : {0, 1}) {
      c = base;
      c.fail_piece = piece;
      c.publishes = publishes;
      fails(c, "disk failure did not abort the run");
    }
  fails(base, "worker setup failure did not abort the run", true);
}
} // namespace

int main() {
  try {
    for (const bool snapshots : {false, true})
      for (const int ncycles : {0, 1, 3}) {
        order_matrix<false>(Order::unsorted, snapshots, ncycles);
        order_matrix<false>(Order::full, snapshots, ncycles);
        order_matrix<true>(Order::partial, snapshots, ncycles);
      }
    order_matrix<false>(Order::full, false, 2, false);
    order_matrix<true>(Order::partial, false, 2, false);
    for (const int ncycles : {1, 3}) {
      order_matrix<false>(Order::full, true, ncycles, true, false);
      order_matrix<true>(Order::partial, true, ncycles, true, false);
      order_matrix<true>(Order::partial, true, ncycles, true, false, true);
    }
    order_matrix<true>(Order::partial, false, 3, true, true, false, 0);
    order_matrix<true>(Order::partial, true, 3, true, true, false, 2);
    // Gauss samples every cycle, every third cycle (one finishing in the final
    // state) and on a run with scalars only at its ends.
    order_matrix<true>(Order::partial, false, 4, true, true, false, 1, 1);
    order_matrix<false>(Order::full, true, 4, true, true, false, 1, 3);
    order_matrix<true>(Order::partial, false, 5, true, true, false, 0, 2);
    opposite_readiness();
    failures();
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
  std::cout << "TPIC pipeline tests passed\n";
}
