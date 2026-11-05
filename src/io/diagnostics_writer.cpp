#include <tpic/io/diagnostics_writer.hpp>
#include <tpic/io/detail/output_file.hpp>
#include <tpic/config/precision.hpp>

#include <charconv>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>

namespace tpic::io {
namespace {

class CsvBuffer {
public:
  explicit CsvBuffer(std::array<char, 8192> &buffer)
      : begin_(buffer.data()), end_(begin_ + buffer.size()), next_(begin_) {}

  void append(std::string_view value) {
    if (value.size() > static_cast<std::size_t>(end_ - next_))
      throw std::length_error("TPIC diagnostics: scalar CSV record exceeds its fixed buffer");
    std::memcpy(next_, value.data(), value.size());
    next_ += value.size();
  }

  void append(int value) { checked(std::to_chars(next_, end_, value)); }
  void append(double value) {
    checked(std::to_chars(next_, end_, value, std::chars_format::general,
                          std::numeric_limits<double>::max_digits10));
  }

  [[nodiscard]] std::size_t size() const noexcept {
    return static_cast<std::size_t>(next_ - begin_);
  }

private:
  void checked(std::to_chars_result result) {
    if (result.ec != std::errc{})
      throw std::length_error("TPIC diagnostics: scalar CSV value exceeds its fixed buffer");
    next_ = result.ptr;
  }

  char *begin_;
  char *end_;
  char *next_;
};

std::string diagnostics_header(std::size_t species_count) {
  std::string text =
      "# TPIC scalar diagnostics v1\n"
      "# Values are globally reduced in double precision with no visualization rescaling.\n"
      "# electric_energy = cell_volume/(8*pi) * sum_unique_nodes(dual_volume_weight * |E_node|^2).\n"
      "# Node ownership excludes upper shared/periodic duplicates; each physical boundary axis contributes a factor 1/2.\n"
      "# magnetic_energy = cell_volume/(8*pi) * sum_owned_cells(|B_cell|^2).\n"
      "# Species kinetic_energy = sum((q/qom)*|v|^2/2); charge = sum(q); momentum = sum((q/qom)*v).\n"
      "# state=pre_solve identifies the state before the cycle's solve/push; state=final identifies the completed run state.\n"
      "# time is physical simulation time at the recorded state; total_energy_change = total_energy - the total energy of the run's first state, kept across restarts (absolute units).\n"
      "cycle,time,state,electric_energy,magnetic_energy,kinetic_energy,total_energy,total_energy_change,charge,momentum_x,momentum_y,momentum_z";
  constexpr std::array<std::string_view, data::species_scalar_count> names{
      "kinetic_energy", "charge", "momentum_x", "momentum_y", "momentum_z"};
  for (std::size_t species = 0; species < species_count; ++species)
    for (const auto name : names)
      text.append(",").append(name).append("_").append(std::to_string(species));
  return text.append("\n");
}

constexpr std::string_view gauss_header =
    "# TPIC Gauss-law residual v1\n"
    "# A sample begins at cycle n and finishes at n + 1: per cell r = (divC(E^n) + divC(E^{n+1}))/2 - c, where divC is\n"
    "# the solver's node-to-cell divergence and c = 4*pi times the eight-node mean of the charge density of the particles\n"
    "# at x^{n+1/2} (dual-volume scaled, zero on PEC wall nodes, smoothed like the solver's current). time = (n + 1/2)*dt.\n"
    "# Root mean squares over the global cells: residual_rms of r; drift_rms of r minus the first finished sample of the\n"
    "# run (a restart starts a new baseline); wall_residual_rms of r over the cells touching a physical wall (0 without\n"
    "# walls); charge_rms of c.\n"
    "cycle,time,residual_rms,drift_rms,wall_residual_rms,charge_rms\n";

} // namespace

namespace detail {

void CycleCsvFile::open(const config::PreparedOutput &output, std::string_view name, config::PrecisionMetadata precision,
                        std::string_view header, std::optional<int> restart_cycle) {
  if (file_.is_open() || !path_.empty())
    throw std::logic_error("TPIC diagnostics: scalar CSV writer can only be opened once");
  if (output.save_dir.empty())
    throw std::invalid_argument("TPIC diagnostics: scalar CSV output directory is empty");
  path_ = (std::filesystem::path(output.save_dir) / name).string();
  if (!restart_cycle) {
    file_.open(path_, output.overwrite_output, "diagnostics output");
    write_header(precision, header);
    return;
  }
  const int cycle = *restart_cycle;
  // The kept bytes end before the first row whose cycle is at least cycle, or
  // before an unterminated last line.
  std::uintmax_t kept = 0;
  bool found_header = false;
  if (std::ifstream csv{path_, std::ios::binary}) {
    for (std::string line; std::getline(csv, line) && !csv.eof();) {
      if (line.starts_with("cycle,")) found_header = true;
      else if (!line.empty() && line.front() != '#') {
        int row_cycle = 0;
        double row_time = 0;
        const auto comma = line.find(',');
        const auto parsed = std::from_chars(line.data(), line.data() + line.size(), row_cycle);
        if (parsed.ec != std::errc{} || comma == std::string::npos ||
            std::from_chars(line.data() + comma + 1, line.data() + line.size(), row_time).ec != std::errc{})
          throw std::runtime_error("TPIC diagnostics: scalar CSV resume: unreadable row in " + path_);
        if (row_cycle >= cycle) break;
        last_cycle_ = row_cycle;
        last_time_ = row_time;
      }
      kept += line.size() + 1;
    }
  }
  file_.open_after(path_, found_header ? kept : 0, "diagnostics output");
  if (!found_header) {
    last_cycle_ = -1;
    write_header(precision, header);
  }
  CsvBuffer csv(buffer);
  csv.append("# restart at cycle ");
  csv.append(cycle);
  csv.append(": precision_profile = ");
  csv.append(precision.name);
  csv.append(", precision_roles = ");
  csv.append(precision.roles);
  csv.append("\n");
  file_.write(buffer.data(), csv.size());
  file_.flush();
}

void CycleCsvFile::write_header(config::PrecisionMetadata precision, std::string_view header) {
  CsvBuffer csv(buffer);
  csv.append("# precision_profile = ");
  csv.append(precision.name);
  csv.append("\n# precision_roles = ");
  csv.append(precision.roles);
  csv.append("\n");
  csv.append(header);
  file_.write(buffer.data(), csv.size());
  file_.flush();
}

void CycleCsvFile::write(std::size_t size, int cycle, double time) {
  if (cycle < 0 || cycle <= last_cycle_ || !std::isfinite(time) || time < 0.0 ||
      (last_cycle_ >= 0 && time <= last_time_))
    throw std::invalid_argument("TPIC diagnostics: scalar CSV cycles and times must increase");
  file_.write(buffer.data(), size);
  file_.flush();
  last_cycle_ = cycle;
  last_time_ = time;
}

void CycleCsvFile::close() {
  if (file_.is_open()) file_.close();
}

} // namespace detail

void DiagnosticsWriter::open(const config::PreparedOutput &output, std::size_t species_count,
                             config::PrecisionMetadata precision, std::optional<int> restart_cycle) {
  csv_.open(output, "diagnostics.csv", precision, diagnostics_header(species_count), restart_cycle);
  species_count_ = species_count;
}

void DiagnosticsWriter::write(const data::ScalarDiagnosticSample &sample, double initial_energy) {
  if (sample.species_count != species_count_)
    throw std::invalid_argument("TPIC diagnostics: sample has a different species count");
  if (final_written_)
    throw std::invalid_argument("TPIC diagnostics: rows end with one final row");

  using Index = data::SpeciesScalarIndex;
  std::array<double, data::species_scalar_count> totals{};
  const std::size_t sample_values = data::scalar_count(species_count_);
  for (std::size_t species = 0; species < species_count_; ++species)
    for (std::size_t component = 0; component < totals.size(); ++component)
      totals[component] += sample.values[data::field_scalar_count +
          data::species_scalar_count * species + component];
  const double total_energy = sample.values[0] + sample.values[1] + totals[Index::kinetic_energy];
  const double energy_change = total_energy - initial_energy;
  for (const double value : totals)
    if (!std::isfinite(value))
      throw std::overflow_error("TPIC diagnostics: species total is nonfinite");
  if (!std::isfinite(total_energy) || !std::isfinite(energy_change))
    throw std::overflow_error("TPIC diagnostics: total energy or change is nonfinite");

  CsvBuffer csv(csv_.buffer);
  csv.append(sample.cycle);
  csv.append(",");
  csv.append(sample.time);
  csv.append(sample.final_state ? ",final" : ",pre_solve");
  const auto column = [&csv](double value) {
    csv.append(",");
    csv.append(value);
  };
  column(sample.values[0]);
  column(sample.values[1]);
  column(totals[Index::kinetic_energy]);
  column(total_energy);
  column(energy_change);
  for (std::size_t component = Index::charge; component < totals.size(); ++component)
    column(totals[component]);
  for (std::size_t index = data::field_scalar_count; index < sample_values; ++index)
    column(sample.values[index]);
  csv.append("\n");
  csv_.write(csv.size(), sample.cycle, sample.time);
  final_written_ = sample.final_state;
}

void GaussWriter::open(const config::PreparedOutput &output, config::PrecisionMetadata precision,
                       std::optional<int> restart_cycle) {
  csv_.open(output, "gauss.csv", precision, gauss_header, restart_cycle);
}

void GaussWriter::write(const data::GaussSample &sample) {
  CsvBuffer csv(csv_.buffer);
  csv.append(sample.cycle);
  csv.append(",");
  csv.append(sample.time);
  for (const double value : sample.rms) {
    csv.append(",");
    csv.append(value);
  }
  csv.append("\n");
  csv_.write(csv.size(), sample.cycle, sample.time);
}

} // namespace tpic::io
