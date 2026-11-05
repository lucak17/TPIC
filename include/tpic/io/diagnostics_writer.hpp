#ifndef TPIC_IO_DIAGNOSTICS_WRITER_HPP
#define TPIC_IO_DIAGNOSTICS_WRITER_HPP

#include <array>
#include <cstddef>
#include <optional>
#include <string>
#include <string_view>

#include <tpic/config/precision.hpp>
#include <tpic/config/output.hpp>
#include <tpic/io/detail/output_file.hpp>
#include <tpic/data/output/gauss_sample.hpp>
#include <tpic/data/output/scalar_sample.hpp>

namespace tpic::io {
namespace detail {

// A rank-zero CSV file in save_dir whose rows start with cycle,time and are
// flushed whole: the precision header, the restart truncation and the
// increasing cycles and times shared by the scalar files.
class CycleCsvFile {
public:
  // Checks the directory, then starts the file with the precision lines and
  // header, its comment lines and column line. A restart from restart_cycle
  // keeps the rows before it instead, or starts the file when it has no column
  // line, and marks the restart in a comment line.
  void open(const config::PreparedOutput &output, std::string_view name, config::PrecisionMetadata precision,
            std::string_view header, std::optional<int> restart_cycle);
  // Writes the first size bytes of buffer as the row of cycle at time; cycles
  // and times increase from row to row.
  void write(std::size_t size, int cycle, double time);
  void close();
  [[nodiscard]] const std::string &path() const noexcept { return path_; }

  std::array<char, 8192> buffer{};

private:
  void write_header(config::PrecisionMetadata precision, std::string_view header);

  OutputFile file_;
  std::string path_;
  int last_cycle_ = -1;
  double last_time_ = 0.0;
};

} // namespace detail

// Rank-zero scalar output only. Construction performs no filesystem work.
// Explicit close reports flush/close failures before MPI finalization; the
// destructor only releases the file during exceptional teardown.
class DiagnosticsWriter {
public:
  // species_count is the admitted real-species count (config::require_species_count);
  // restart_cycle as in detail::CycleCsvFile::open.
  void open(const config::PreparedOutput &output, std::size_t species_count, config::PrecisionMetadata precision,
            std::optional<int> restart_cycle = {});
  // initial_energy is the total energy of the run's first state.
  void write(const data::ScalarDiagnosticSample &sample, double initial_energy);
  void close() { csv_.close(); }
  [[nodiscard]] const std::string &path() const noexcept { return csv_.path(); }

private:
  detail::CycleCsvFile csv_;
  std::size_t species_count_ = 0;
  bool final_written_ = false;
};

// Rank-zero gauss.csv (DiagnosticGaussCycle): one row per finished sample, the
// cycle it began at and the time of its residual. Same lifecycle as above.
class GaussWriter {
public:
  void open(const config::PreparedOutput &output, config::PrecisionMetadata precision, std::optional<int> restart_cycle = {});
  void write(const data::GaussSample &sample);
  void close() { csv_.close(); }
  [[nodiscard]] const std::string &path() const noexcept { return csv_.path(); }

private:
  detail::CycleCsvFile csv_;
};

} // namespace tpic::io

#endif // TPIC_IO_DIAGNOSTICS_WRITER_HPP
