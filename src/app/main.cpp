#include <lpfx/profiling.hpp>

#include <tpic/simulation/planning/run_plan.hpp>
#include <tpic/simulation/reporting/startup_report.hpp>
#include <tpic/config/precision.hpp>
#if TPIC_HAS_DEVICE_STARTUP
#include <tpic/simulation/driver/run_distributed.hpp>
#endif

#include <charconv>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>

namespace {

constexpr std::string_view usage =
    "Usage: tpic --check-input INPUT [--rank N] [--order full|partial|unsorted] [--overwrite-output]\n"
    "       tpic --initialize INPUT [--order full|partial|unsorted] [--overwrite-output] [--restart]  (CUDA/HIP build)\n"
    "       tpic --run INPUT [--order full|partial|unsorted] [--overwrite-output] [--restart]         (CUDA/HIP build)\n"
    "       tpic --help\n"
    "All commands accept --mass-matrix scalar|matrix-core (default scalar).\n"
    "All commands accept --mass-apply full|split (default full); smoothing uses full-domain mass.\n"
    "matrix-core requires full/partial ordering, double mass compute and CUDA SM80+ or AMD FP64 MFMA (wave64).\n"
    "All commands accept --precision LEVEL, which selects the same level for storage and compute:\n"
    "double, single-position, single-position-velocity, single-particles-moments-fields-double-mass\n"
    "or single-particles-moments-fields. The double-mass level computes the mass matrix in FP64\n"
    "and stores it in FP32 (storage single-particles-moments-fields).\n"
    "--storage-precision and --compute-precision select the two levels separately; unmatched\n"
    "pairs need a build with TPIC_BUILD_ALL_PRECISION_PAIRS=ON or a TPIC_PRECISION_PAIRS list.\n"
    "Rank defaults to 0 (the largest partition); size comes from XLEN*YLEN*ZLEN.\n"
    "Exit codes: 0 host checks passed, 1 invalid input/command, 2 unsupported requests.\n"
    "--check-input checks the selected ECSIM precision pair without GPU/MPI initialization or output writes.\n"
    "--initialize completes distributed GPU startup and exits before timestepping.\n"
    "--order defaults to partial for every command.\n"
    "--run advances ncycles: full, partial or unsorted, c=1, energy every DiagnosticEnergyCycle cycles.\n"
    "DiagnosticsOutputCycle controls CSV cadence independently of console energy.\n"
    "DiagnosticGaussCycle starts a Gauss-law residual sample (gauss.csv and log) every so many cycles.\n"
    "WriteMethod selects pvtk (default), adios2 or hdf5 for asynchronous fields and particles.\n"
    "FieldWriteMethod and ParticlesWriteMethod independently override WriteMethod.\n"
    "ADIOS2 and HDF5 require their build options; ParticlesOutputCycle controls real particles.\n"
    "RestartOutputCycle writes checkpoints to RestartDirName with RestartWriteMethod adios2 or hdf5;\n"
    "--restart continues from the newest complete one, on any rank count, up to the absolute ncycles.\n"
    "--overwrite-output replaces files emitted by a fresh --run and sets aside earlier checkpoints;\n"
    "default refuses existing files. It does not clear the folder. A restart replaces its outputs.\n"
    "EM faces: periodic, PEC or PMC, each with optional smoothing.\n"
    "Particle faces: periodic or mirror. Open EM faces are not implemented.\n";

int parse_rank(std::string_view text) {
  int rank = 0;
  const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), rank);
  if (error != std::errc{} || end != text.data() + text.size() || rank < 0)
    throw std::invalid_argument("TPIC options: --rank requires a nonnegative signed integer");
  return rank;
}

} // namespace

int main(int argc, char **argv) {
  // CrayPat records only the steady-state cycles, which the pipeline's run_cycles switches on and off.
  LPFX_PROFILE_RECORD(false);
  try {
    tpic::simulation::InputOptions options;
    bool check = false, initialize = false, run = false, rank_given = false;
    std::string_view path;
    for (int i = 1; i < argc; ++i) {
      const std::string_view arg = argv[i];
      if (arg == "--help") {
        std::cout << usage << "Runtime default precision (storage/compute): "
                  << tpic::config::precision_name(tpic::config::PrecisionSelection{}) << '\n';
        return 0;
      }
      if (arg == "--check-input") { check = true; continue; }
      if (arg == "--initialize") { initialize = true; continue; }
      if (arg == "--run") { run = true; continue; }
      if (arg == "--overwrite-output") { options.overwrite_output = true; continue; }
      if (arg == "--restart") { options.restart = true; continue; }
      if (arg == "--rank" || arg == "--order" || arg == "--mass-matrix" ||
          arg == "--mass-apply" || arg == "--precision" || arg == "--storage-precision" ||
          arg == "--compute-precision") {
        if (++i == argc) throw std::invalid_argument("TPIC options: " + std::string(arg) + " requires a value");
        if (arg == "--rank") { options.rank = parse_rank(argv[i]); rank_given = true; }
        else if (arg == "--precision") options.precision = tpic::config::parse_precision(argv[i]);
        else if (arg == "--storage-precision")
          options.precision.storage = tpic::config::parse_storage_precision(argv[i]);
        else if (arg == "--compute-precision")
          options.precision.compute = tpic::config::parse_compute_precision(argv[i]);
        else if (arg == "--mass-apply") options.mass_apply = tpic::config::parse_mass_apply_mode(argv[i]);
        else if (arg == "--mass-matrix")
          options.mass_matrix = tpic::config::parse_mass_deposition_kernel(argv[i]);
        else {
          const std::string_view order = argv[i];
          if (order == "full") options.order = tpic::config::ParticleOrder::full;
          else if (order == "partial") options.order = tpic::config::ParticleOrder::partial;
          else if (order == "unsorted") options.order = tpic::config::ParticleOrder::unsorted;
          else throw std::invalid_argument("TPIC options: --order must be full, partial or unsorted");
        }
        continue;
      }
      if (arg.starts_with('-')) throw std::invalid_argument("TPIC options: unknown option: " + std::string(arg));
      if (!path.empty()) throw std::invalid_argument("TPIC options: expected exactly one input file");
      path = arg;
    }
    if (int(check) + int(initialize) + int(run) != 1 || path.empty())
      throw std::invalid_argument("TPIC options: give exactly one of --check-input, --initialize or --run and "
                                  "one input file\n" + std::string(usage));
    if (initialize || run) {
      if (rank_given) throw std::invalid_argument("TPIC options: --rank is only valid with --check-input; MPI supplies startup ranks");
#if TPIC_HAS_DEVICE_STARTUP
      return tpic::simulation::run_distributed(argc, argv, path, options,
          run ? tpic::simulation::RunMode::advance : tpic::simulation::RunMode::initialize_only);
#else
      throw std::invalid_argument("TPIC options: --initialize and --run require a TPIC CUDA/HIP build");
#endif
    }
    if (options.restart) throw std::invalid_argument("TPIC options: --restart is only valid with --run or --initialize");
    auto input = tpic::simulation::prepare_input(tpic::config::load_simulation_parameters(path), options);
    for (auto &request : tpic::simulation::unavailable_output_backends(input))
      input.unsupported_requests.push_back(std::move(request));
    tpic::simulation::print_input_check(input, options.rank);
    return input.host_admitted() ? 0 : 2;
  } catch (const std::exception &error) {
    tpic::simulation::print_error(error.what());
    return 1;
  }
}
