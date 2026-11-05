#include <tpic/simulation/reporting/startup_report.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <iomanip>
#include <iostream>
#include <ostream>
#include <sstream>
#include <string>
#include <variant>

namespace tpic::simulation {
namespace {
constexpr int report_width = 72;
constexpr double gib = 1024.0 * 1024.0 * 1024.0;

void section(std::ostream& out, std::string_view title) {
  out << '\n' << title << '\n' << std::string(report_width, '-') << '\n';
}

std::ostream& label(std::ostream& out, std::string_view text) {
  return out << std::left << std::setw(30) << text << " : ";
}

std::string cadence(int cycles) {
  return cycles > 0 ? "every " + std::to_string(cycles) + " cycle(s)" : "disabled";
}

std::string_view particle_positions(const config::SimulationParameters& p) {
  return p.quiet_start ? "subcell centres (QuietStart)" : "random in each subcell";
}

// "<requested mode>; effective: <schedule>"
std::string mass_apply_value(const PreparedInput& input) {
  return std::string(config::mass_apply_mode_name(input.mass_apply)) + "; effective: " +
         std::string(ecsim::maxwell_apply_schedule_name(input.maxwell_schedule));
}

// The largest |B| of the analytic initial field, uniform background included,
// over a lattice of at most 33 points per axis that spans the domain.
initialization::FieldVector largest_initial_magnetic(const PreparedInput& input) {
  const auto& p = input.parameters;
  const std::array<double, 3> length{p.grid.lx, p.grid.ly, p.grid.lz};
  const std::array<int, 3> intervals{
      std::min(p.grid.nxc, 32), std::min(p.grid.nyc, 32), std::min(p.grid.nzc, 32)};
  initialization::FieldVector largest{};
  double largest_squared = 0.0;
  for (int k = 0; k <= intervals[2]; ++k)
    for (int j = 0; j <= intervals[1]; ++j)
      for (int i = 0; i <= intervals[0]; ++i) {
        const auto b = initialization::initial_magnetic(p, input.initial_case,
            length[0] * i / intervals[0], length[1] * j / intervals[1], length[2] * k / intervals[2]);
        const double squared = b[0] * b[0] + b[1] * b[1] + b[2] * b[2];
        if (squared > largest_squared) {
          largest = b;
          largest_squared = squared;
        }
      }
  return largest;
}

// The time step and the cell against the scales of each species. The scales
// come from the deck values (rhoINIT, thermal speeds, drifts) and the largest
// sampled initial |B|; rhoINIT is 4*pi times the charge density, so
// omega_p^2 = |qom|*rhoINIT and Omega_c = |qom|*B/c. The gyroradius uses the
// thermal speed per direction perpendicular to that field. The numbers are
// reported only: admission sets no limit on any of them.
void resolution(std::ostream& out, const PreparedInput& input) {
  const auto& p = input.parameters;
  const auto& cell = input.domain.spacing;
  // A length along each axis as the number of cells of that axis it spans.
  const auto in_cells = [&](double x, double y, double z) {
    std::ostringstream text;
    text << x / cell[0] << " x " << y / cell[1] << " x " << z / cell[2] << " cells";
    return text.str();
  };
  // One length: its value, then the cells it spans.
  const auto resolved = [&](double length) {
    std::ostringstream text;
    text << length << " = " << in_cells(length, length, length);
    return text.str();
  };
  section(out, "Numerical resolution");
  out << "  Deck values of each species. Lengths are also given in cells, as\n"
      << "  length / cell size along x, y, z: the cells that resolve the length.\n";
  const double light_step = p.c * p.dt;
  label(out, "Light step c*dt") << resolved(light_step) << '\n';
  // A single-cell axis is periodic and carries no variation.
  double inverse_cell_squared = 0.0;
  const std::array<int, 3> cells{p.grid.nxc, p.grid.nyc, p.grid.nzc};
  for (std::size_t axis = 0; axis != 3; ++axis)
    if (cells[axis] > 1) inverse_cell_squared += 1.0 / (cell[axis] * cell[axis]);
  label(out, "c*dt*sqrt(sum 1/cell^2)") << light_step * std::sqrt(inverse_cell_squared)
      << " (axes with more than one cell)\n";
  const auto b = largest_initial_magnetic(input);
  const double b_squared = b[0] * b[0] + b[1] * b[1] + b[2] * b[2];
  const double b_magnitude = std::sqrt(b_squared);
  label(out, "Largest initial |B|") << b_magnitude << " (analytic field on a lattice, background included)\n";
  for (std::size_t species = 0; species != p.species.size(); ++species) {
    const auto& sp = p.species[species];
    const double omega_p = std::sqrt(std::abs(sp.q_over_m) * sp.rho_init);
    const double omega_c = std::abs(sp.q_over_m) * b_magnitude / p.c;
    out << "  Species " << species << ": omega_p*dt = " << omega_p * p.dt
        << ", Omega_c*dt = " << omega_c * p.dt << '\n';
    if (omega_p > 0.0) out << "    Skin depth c/omega_p         = " << resolved(p.c / omega_p) << '\n';
    if (sp.uth == 0.0 && sp.vth == 0.0 && sp.wth == 0.0) {
      out << "    Thermal speeds are zero in the deck: no Debye length, gyroradius or thermal step\n";
    } else {
      if (omega_p > 0.0)
        out << "    Debye length vth/omega_p     = "
            << in_cells(sp.uth / omega_p, sp.vth / omega_p, sp.wth / omega_p) << '\n';
      if (omega_c > 0.0) {
        const double perpendicular = std::sqrt(0.5 * (sp.uth * sp.uth * (1.0 - b[0] * b[0] / b_squared) +
                                                      sp.vth * sp.vth * (1.0 - b[1] * b[1] / b_squared) +
                                                      sp.wth * sp.wth * (1.0 - b[2] * b[2] / b_squared)));
        out << "    Gyroradius vth_perp/Omega_c  = " << resolved(perpendicular / omega_c) << '\n';
      }
      out << "    Thermal step vth*dt          = "
          << in_cells(sp.uth * p.dt, sp.vth * p.dt, sp.wth * p.dt) << '\n';
    }
    out << "    Drift step |v0|*dt           = "
        << in_cells(std::abs(sp.u0) * p.dt, std::abs(sp.v0) * p.dt, std::abs(sp.w0) * p.dt) << '\n';
  }
}

} // namespace

void print_input_check(const PreparedInput& input, int rank) {
  const auto &p = input.parameters;
  std::cout << std::setprecision(17)
      << "TPIC host input check: " << (input.host_admitted() ? "passed" : "unsupported requests") << '\n'
      << "Case: " << p.case_name
      << "; particle recipe: "
      << std::visit([](const auto& selected) { return selected.particle_label(); }, input.initial_case)
      << "; particle positions: " << particle_positions(p) << '\n'
      << "Precision: " << config::precision_name(input.precision) << "; particle order plan: " << config::particle_order_name(input.particles.order) << "; species: " << p.ns << '\n'
      << "Precision roles: " << config::precision_metadata(input.precision).roles << '\n'
      << "Mass matrix kernel: " << config::mass_deposition_kernel_name(input.mass_matrix) << '\n'
      << "Mass apply: " << mass_apply_value(input) << '\n'
      << "Rank: " << rank << " / " << input.communicator_size << "; local cells: "
      << input.domain.local_cells[0] << 'x' << input.domain.local_cells[1] << 'x' << input.domain.local_cells[2] << '\n'
      << "Particles: local=" << input.particles.local_count << ", global=" << input.particles.global_count << '\n';
  for (std::size_t s = 0; s < input.particles.species.size(); ++s) {
    const auto &plan = input.particles.species[s];
    std::cout << "  species " << s << ": count=" << plan.local_count
        << ", capacity=" << plan.initial_capacity
        << ", migration capacity=" << plan.initial_migration_capacity
        << ", IDs=[" << plan.first_id << ',' << plan.past_last_id << ")\n";
  }
  const auto bytes = report_particle_bytes(input);
  std::cout << "Particle/sort/classification device bytes (lower bound): "
      << bytes.device_bytes_lower_bound << '\n'
      << "Logical particle record bytes: " << bytes.particle_record_bytes << '\n'
      << "Excluded: alignment, sort scan scratch, fields, mass matrix, GMRES, transport, output and growth peaks.\n"
      << "EM faces [Xright Xleft Yright Yleft Zright Zleft]:";
  for (const auto boundary : input.physics.field_boundary)
    std::cout << ' ' << config::field_boundary_rule(boundary).name;
  std::cout << "\nParticle faces:";
  for (const auto boundary : input.physics.particle_boundary)
    std::cout << ' ' << config::particle_boundary_rule(boundary).name;
  std::cout << "\nSmoothing: " << (input.physics.smoothing_enabled ? "active" : "inactive")
      << "; GMRES restart=" << input.physics.gmres_restart
      << ", max iterations=" << input.physics.gmres_max_iterations
      << ", tolerance=" << p.gmres_tol << '\n'
      << "Output plan: field engine=" << config::output_engine_name(input.output.field_engine)
      << ", particle engine=" << config::output_engine_name(input.output.particle_engine)
      << ", fields=" << (input.output.fields_enabled() ? "requested" : "disabled")
      << ", particles=" << (input.output.particles_enabled() ? "requested" : "disabled")
      << ", pressure=" << (input.output.pressure_active() ? "selected" : "off")
      << ", existing files=" << (input.output.overwrite_output ? "overwrite emitted files" : "refuse") << '\n';
  if (const auto* wave = std::get_if<initialization::WhistlerCase>(&input.initial_case))
    std::cout << "Whistler: k=" << wave->wavenumber << ", omega=" << wave->omega
        << ", noise modes=" << wave->noise_mode_max << '\n';
  if (const auto* turbulence = std::get_if<initialization::WhistlerTurbulenceCase>(&input.initial_case)) {
    const auto [slowest, fastest] = std::ranges::minmax(turbulence->modes.phase_speed);
    std::cout << "WhistlerTurbulence: " << initialization::WhistlerTurbulenceModes::count
        << " modes, amplitude=" << turbulence->modes.amplitude << ", omega/(c*k) from " << slowest << " to "
        << fastest << '\n';
  }
  // Format privately: the 17 digits of this report are not those of the section.
  std::ostringstream resolution_text;
  resolution(resolution_text, input);
  std::cout << resolution_text.str();
  for (const auto &note : input.physics.notes) std::cout << "Note: " << note << '\n';
  for (const auto &warning : input.warnings) std::cout << "Warning: " << warning << '\n';
  for (const auto &request : input.unsupported_requests) std::cout << "Unsupported: " << request << '\n';
  std::cout << "Input checking performs no simulation startup or ECSIM timestep.\n"
      << "GPU builds provide --initialize with full/partial/unsorted ordering; device memory is estimated, not admitted.\n"
      << "No output/restart directories were inspected or modified.\n";
}

void print_configuration(const PreparedInput& input, std::string_view path, RunMode mode,
                         std::string_view device_backend, std::string_view payload_backend,
                         const std::optional<data::PreparedRestart>& restart) {
  const auto& p = input.parameters;
  const bool advance = advances(mode);
  // Format privately so precision/alignment never leak into timestep reports.
  std::ostringstream out;
  out << '\n' << std::string(report_width, '=') << '\n'
      << "                         TPIC ECSIM Simulation\n"
      << std::string(report_width, '=') << '\n';
  section(out, "Simulation setup");
  label(out, "Input file") << path << '\n';
  label(out, "Simulation / case") << (p.sim_name.empty() ? "(unnamed)" : p.sim_name)
                                   << " / " << p.case_name << '\n';
  label(out, "Execution") << (advance ? "time integration" : "initialization only") << '\n';
  label(out, "Backend") << "LPFx / " << device_backend << " / " << payload_backend << " (device payload)\n";
  label(out, "Host / control transport") << "LPFx / MPI\n";
  label(out, "MPI processes / grid") << input.communicator_size << " / " << p.topology.xlen << " x "
      << p.topology.ylen << " x " << p.topology.zlen << '\n';
  label(out, "Grid cells (x, y, z)") << p.grid.nxc << " x " << p.grid.nyc << " x " << p.grid.nzc << '\n';
  label(out, "Domain size (x, y, z)") << p.grid.lx << " x " << p.grid.ly << " x " << p.grid.lz << '\n';
  label(out, "Cell size (x, y, z)") << input.domain.spacing[0] << " x "
      << input.domain.spacing[1] << " x " << input.domain.spacing[2] << '\n';
  label(out, "Time step / theta / cycles") << p.dt << " / " << p.th << " / " << p.ncycles << '\n';
  label(out, "Final time / speed of light") << input.physics.final_time << " / " << p.c << '\n';

  section(out, "Precision and particles");
  label(out, "Precision") << config::precision_name(input.precision) << '\n';
  label(out, "Precision roles") << config::precision_metadata(input.precision).roles << '\n';
  label(out, "Mass matrix kernel") << config::mass_deposition_kernel_name(input.mass_matrix) << '\n';
  label(out, "Mass apply") << mass_apply_value(input) << '\n';
  if (input.mass_matrix == config::MassDepositionKernel::matrix_core)
    label(out, "Matrix-core cell mass sums") << "FP64; narrow to mass storage at cell flush; scalar arrival tail\n";
  label(out, "Output moment accumulation") << "FP64\n";
  label(out, "Particle cell / ID") << "int32 / uint64\n";
  label(out, "Particle order") << config::particle_order_name(input.particles.order) << '\n';
  label(out, "Species / initial particles") << p.ns << " / " << input.particles.global_count << '\n';
  label(out, "Initial particle positions") << particle_positions(p) << '\n';
  for (std::size_t species = 0; species != input.particles.species.size(); ++species) {
    const auto& sp = p.species[species];
    out << "  Species " << species << ": " << input.particles.species[species].global_count
        << " particles, " << input.particles.species[species].particles_per_cell << " per cell\n"
        << "    q/m = " << sp.q_over_m << ", rho_init = " << sp.rho_init
        << ", centered velocity = " << (sp.center_initial_velocity ? "yes" : "no") << '\n'
        << "    Thermal velocity = (" << sp.uth << ", " << sp.vth << ", " << sp.wth << ")"
        << "; drift = (" << sp.u0 << ", " << sp.v0 << ", " << sp.w0 << ")\n";
  }

  section(out, "Boundaries and solver");
  for (std::size_t axis = 0; axis != 3; ++axis) {
    const std::string axis_name(1, "XYZ"[axis]);
    label(out, "Fields " + axis_name + " (lower / upper)")
        << config::field_boundary_rule(input.physics.field_boundary[2 * axis + 1]).name << " / "
        << config::field_boundary_rule(input.physics.field_boundary[2 * axis]).name << '\n';
    label(out, "Particles " + axis_name + " (lower / upper)")
        << config::particle_boundary_rule(input.physics.particle_boundary[2 * axis + 1]).name << " / "
        << config::particle_boundary_rule(input.physics.particle_boundary[2 * axis]).name << '\n';
  }
  label(out, "Smoothing");
  if (input.physics.smoothing_enabled)
    out << "energy conserving, "
        << (p.ecsim_smoothing_operator == config::EcsimSmoothingOperator::Face7 ? "Face7" : "Binomial27")
        << ", " << p.smooth_niter << " passes, factor " << p.smooth << '\n';
  else out << "disabled\n";
  label(out, "GMRES tolerance") << p.gmres_tol << '\n';
  label(out, "GMRES restart / max iterations") << input.physics.gmres_restart
      << " / " << input.physics.gmres_max_iterations << '\n';

  resolution(out, input);

  section(out, "Output");
  label(out, "Output directory") << input.output.save_dir << '\n';
  label(out, "Existing files") << (input.output.overwrite_output
      ? "replace files emitted by this run; retain other files"
      : "protected (use --overwrite-output to replace)") << '\n';
  label(out, "Scalar diagnostics (CSV)") << cadence(input.output.diagnostics_cycle) << '\n';
  label(out, "Energy diagnostics (console)") << (input.output.diagnostic_energy_cycle > 0
      ? cadence(input.output.diagnostic_energy_cycle) + " and the final state" : "the final state") << '\n';
  label(out, "Gauss residual (CSV, log)") << (input.output.output_gauss
      ? "before each field snapshot" + (input.output.gauss_cycle > 0 ? " and " + cadence(input.output.gauss_cycle)
                                                                       : std::string())
      : cadence(input.output.gauss_cycle)) << '\n';
  label(out, "Field snapshot engine") << config::output_engine_name(input.output.field_engine) << '\n';
  label(out, "Particle snapshot engine") << config::output_engine_name(input.output.particle_engine) << '\n';
  label(out, "Field / moment snapshots") << (input.output.fields_enabled()
      ? cadence(input.output.field_cycle) : "disabled") << '\n';
  label(out, "Particle snapshots") << cadence(input.output.particles_cycle) << '\n';
  label(out, "Checkpoints") << (input.output.checkpoints_enabled()
      ? cadence(input.output.checkpoint_cycle) + (input.output.checkpoint_final ? " and the final state" : "") +
            ", " + std::string(config::output_engine_name(input.output.checkpoint_engine)) + ", in " +
            input.output.checkpoint_dir
      : "disabled") << '\n';
  if (restart) {
    const auto& header = restart->checkpoint.header;
    label(out, "Restart from") << restart->checkpoint.directory << " (cycle " << header.cycle << ", process grid "
        << header.processes[0] << " x " << header.processes[1] << " x " << header.processes[2] << ")\n";
  }
  if (!advance) out << "  Initialization-only execution writes no output files.\n";
  if (!input.physics.notes.empty() || !input.warnings.empty()) {
    section(out, "Notes and warnings");
    for (const auto& note : input.physics.notes) out << "  Note: " << note << '\n';
    for (const auto& warning : input.warnings) out << "  Warning: " << warning << '\n';
  }
  std::cout << out.str() << std::flush;
}

void print_memory_plan(const std::vector<RankMemoryReport>& reports) {
  const int ranks = static_cast<int>(reports.size());
  std::ostringstream out;
  section(out, "Memory plan (GiB; informational, never an allocation limit)");
  out << "  Device amounts below are per rank; host RAM is listed separately.\n"
      << "  Transport/runtime are allowances, not measured allocations.\n"
      << std::fixed << std::setprecision(3);
  for (int peer = 0; peer != ranks; ++peer) {
    const auto& r = reports[peer];
    const auto& e = r.estimate;
    out << "  Rank " << peer << " | local rank " << r.local_rank << " | GPU " << r.device << '\n';
    if (!r.estimate_valid) {
      out << "    Estimate unavailable; proceeding with actual allocations.\n";
      continue;
    }
    label(out, "    Startup plan + allowances") << e.estimated_device_bytes / gib << '\n';
    label(out, "    Device owners") << e.persistent_device_bytes / gib << '\n';
    label(out, "      Particles / sort") << e.particle_device_bytes / gib << " / " << e.sort_device_bytes / gib << '\n';
    label(out, "      Fields / mass / solver") << e.field_device_bytes / gib << " / " << e.mass_device_bytes / gib
                                                   << " / " << e.solver_device_bytes / gib << '\n';
    label(out, "      Output staging") << e.output_staging_device_bytes / gib << '\n';
    label(out, "      Other device owners") <<
        (e.persistent_device_bytes - e.particle_device_bytes - e.sort_device_bytes - e.field_device_bytes -
         e.mass_device_bytes - e.solver_device_bytes - e.output_staging_device_bytes) / gib << '\n';
    label(out, "    Transport allowance") << e.transport_device_bytes / gib
        << " (reserved arrivals " << e.migration_reserved_device_bytes / gib << ")\n";
    label(out, "    Runtime allowance") << e.runtime_reserve_bytes / gib << '\n';
    label(out, "    Hypothetical growth extra") << e.growth_reserve_bytes / gib
        << " (not allocated; excluded above)\n";
    label(out, "    Host pinned startup allowance") << e.startup_pinned_host_estimate_bytes / gib << '\n';
    label(out, "    Host output pinned / heap") << e.output_pinned_host_bytes / gib << " / "
        << e.output_owned_host_bytes / gib << " (initial population; particle images allocated on capture)\n";
    label(out, "    Runtime host / timing bytes") << e.runtime_host_bytes << " / "
        << e.timing_host_bytes << '\n';
    label(out, "    CPU workers / streams") << e.worker_count << " / " << e.stream_count << '\n';
    label(out, "    Validators / dependency events") << e.validation_count << " / "
        << e.dependency_event_count << '\n';
    label(out, "    Worker profiling events") << e.timing_event_count << '\n';
  }
  // Ordinals are process-local; UUIDs identify ranks sharing the physical GPU.
  std::vector<bool> shown(ranks, false);
  for (int peer = 0; peer != ranks; ++peer) {
    if (shown[peer]) continue;
    const auto& r = reports[peer];
    double planned = 0;
    bool complete = true;
    auto free = r.free_bytes;
    out << "  Shared GPU ranks:";
    for (int other = peer; other != ranks; ++other) {
      const auto& s = reports[other];
      if (other != peer && (!(r.device_uuid[0] || r.device_uuid[1]) ||
                            r.device_uuid[0] != s.device_uuid[0] || r.device_uuid[1] != s.device_uuid[1]))
        continue;
      shown[other] = true;
      out << ' ' << other;
      planned += s.estimate.estimated_device_bytes / gib;
      complete = complete && s.estimate_valid;
      if (s.memory_valid) free = std::min(free, s.free_bytes);
    }
    out << '\n';
    if (r.memory_valid)
      label(out, "    Measured free / total") << free / gib << " / " << r.total_bytes / gib << '\n';
    if (complete)
      label(out, "    Combined startup plan") << planned << '\n';
  }
  out << std::string(report_width, '=') << '\n';
  std::cout << out.str() << std::flush;
}

void print_initialized_memory(const std::vector<RankMemoryReport>& reports) {
  std::ostringstream out;
  section(out, "GPU memory after initialization (GiB; measured, shared device)");
  out << std::fixed << std::setprecision(3);
  for (std::size_t peer = 0; peer != reports.size(); ++peer) {
    const auto& r = reports[peer];
    out << "  Rank " << peer << " | GPU " << r.device;
    if (r.memory_valid) out << " | used / free / total: " << (r.total_bytes - r.free_bytes) / gib
                            << " / " << r.free_bytes / gib << " / " << r.total_bytes / gib;
    else out << " | memory query unavailable";
    out << '\n';
  }
  out << "  Usage includes all ranks, other processes and driver allocations.\n";
  std::cout << out.str() << std::flush;
}

void print_error(std::string_view message, int rank) noexcept {
  const std::string_view prefix = message.starts_with("TPIC ") ? "" : "TPIC: ";
  if (rank >= 0)
    std::fprintf(stderr, "%.*s%.*s (rank %d)\n", int(prefix.size()), prefix.data(), int(message.size()),
                 message.data(), rank);
  else
    std::fprintf(stderr, "%.*s%.*s\n", int(prefix.size()), prefix.data(), int(message.size()), message.data());
}

void print_completion(const RunResult& result, const PreparedInput& input, RunMode mode) {
  std::ostringstream out;
  section(out, advances(mode) ? "Simulation complete" : "Initialization complete");
  label(out, "Timesteps completed") << result.steps_completed << '\n';
  label(out, "Species / final particles") << input.parameters.ns << " / " << input.particles.global_count << '\n';
  for (std::size_t species = 0; species != input.particles.species.size(); ++species)
    label(out, "  Species " + std::to_string(species)) << input.particles.species[species].global_count << " particles\n";
  if (result.diagnostic_samples != 0) {
    label(out, "Diagnostics rows") << result.diagnostic_samples << '\n';
    label(out, "Diagnostics file") << result.diagnostics_path << '\n';
  }
  if (result.gauss_samples != 0) {
    label(out, "Gauss-law rows") << result.gauss_samples << '\n';
    label(out, "Gauss-law file") << result.gauss_path << '\n';
  }
  if (result.field_samples != 0) {
    label(out, "Field snapshots") << result.field_samples << '\n';
    label(out, "Last snapshot") << result.fields_path << '\n';
  }
  if (result.particle_samples != 0) {
    label(out, "Particle snapshots") << result.particle_samples << '\n';
    label(out, "Last particle snapshot") << result.particles_path << '\n';
  }
  if (result.checkpoints != 0) {
    label(out, "Checkpoints") << result.checkpoints << '\n';
    label(out, "Last checkpoint") << result.checkpoint_path << '\n';
  }
  if (result.diagnostic_samples == 0 && result.gauss_samples == 0 && result.field_samples == 0 &&
      result.particle_samples == 0 && result.checkpoints == 0)
    out << "No output files written.\n";
  out << std::string(report_width, '=') << '\n';
  std::cout << out.str() << std::flush;
}

} // namespace tpic::simulation
