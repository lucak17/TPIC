#ifndef TPIC_TESTS_SUPPORT_GOLDEN_DATA_HPP
#define TPIC_TESTS_SUPPORT_GOLDEN_DATA_HPP

// Golden tables are line-oriented text files in tests/data/golden/ holding
// values TPIC itself produced. A golden test builds the same lines and
// compares them with the table; started with --print it writes its lines to
// standard output instead, which regenerates a table after a reviewed,
// intended behaviour change. Lines starting with '#' are comments.

#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace tpic::test::golden {

using Lines = std::vector<std::string>;
using LineEqual = std::function<bool(const std::string& actual, const std::string& expected)>;

struct Options {
  std::filesystem::path source_root;
  bool print = false;
};

// Command line of every golden test: SOURCE_ROOT [--print].
[[nodiscard]] inline Options parse_options(int argc, char** argv) {
  if (argc < 2 || argc > 3 || (argc == 3 && std::string_view(argv[2]) != "--print"))
    throw std::invalid_argument("usage: test SOURCE_ROOT [--print]");
  return {argv[1], argc == 3};
}

// Exact text of a double: C99 hexadecimal floating point.
[[nodiscard]] inline std::string hex(double value) {
  char buffer[64];
  std::snprintf(buffer, sizeof buffer, "%a", value);
  return buffer;
}

[[nodiscard]] inline std::string quoted(std::string_view text) {
  std::string result = "\"";
  for (const char c : text) {
    if (c == '\\' || c == '"') {
      result += '\\';
      result += c;
    } else if (c == '\n') {
      result += "\\n";
    } else if (c == '\r') {
      result += "\\r";
    } else if (c == '\t') {
      result += "\\t";
    } else {
      result += c;
    }
  }
  return result += '"';
}

[[nodiscard]] inline std::vector<std::string> split(const std::string& line, char separator) {
  std::vector<std::string> fields;
  std::size_t begin = 0;
  for (std::size_t end; (end = line.find(separator, begin)) != std::string::npos; begin = end + 1)
    fields.push_back(line.substr(begin, end - begin));
  fields.push_back(line.substr(begin));
  return fields;
}

// The decks of the source tree under input/ and performance/scaling/, sorted by
// relative path. A scaling campaign leaves copies of its decks in a results_*
// folder of performance/scaling/: that is run output, and no deck of the tables.
[[nodiscard]] inline std::vector<std::filesystem::path> deck_files(const std::filesystem::path& root) {
  std::vector<std::filesystem::path> decks;
  for (const auto* folder : {"input", "performance/scaling"}) {
    for (std::filesystem::recursive_directory_iterator entry(root / folder), end; entry != end; ++entry) {
      if (entry->is_directory() && entry->path().filename().string().starts_with("results_"))
        entry.disable_recursion_pending();
      else if (entry->is_regular_file() && entry->path().extension() == ".inp")
        decks.push_back(entry->path().lexically_relative(root));
    }
  }
  std::sort(decks.begin(), decks.end());
  if (decks.empty()) throw std::runtime_error("no input decks found under input/ and performance/scaling/");
  return decks;
}

[[nodiscard]] inline Lines read_table(const std::filesystem::path& path) {
  std::ifstream input(path);
  if (!input) throw std::runtime_error("missing golden table " + path.string());
  Lines lines;
  for (std::string line; std::getline(input, line);)
    if (!line.starts_with('#')) lines.push_back(line);
  return lines;
}

// Compare with tests/data/golden/<table>, or print the lines with --print.
inline void check(const Options& options, std::string_view table, const Lines& lines,
                  const LineEqual& equal = {}) {
  if (options.print) {
    for (const auto& line : lines) std::cout << line << '\n';
    return;
  }
  const auto expected = read_table(options.source_root / "tests/data/golden" / table);
  const std::string name(table);
  const auto count = std::min(lines.size(), expected.size());
  for (std::size_t index = 0; index < count; ++index) {
    if (equal ? !equal(lines[index], expected[index]) : lines[index] != expected[index])
      throw std::runtime_error(name + ": entry " + std::to_string(index + 1) +
          " differs\n  golden: " + expected[index] + "\n  actual: " + lines[index]);
  }
  if (lines.size() != expected.size())
    throw std::runtime_error(name + ": " + std::to_string(lines.size()) +
        " entries, golden table has " + std::to_string(expected.size()));
  std::cout << name << ": " << lines.size() << " golden entries matched\n";
}

} // namespace tpic::test::golden

#endif // TPIC_TESTS_SUPPORT_GOLDEN_DATA_HPP
