#ifndef TPIC_IO_DETAIL_VTK_OUTPUT_HPP
#define TPIC_IO_DETAIL_VTK_OUTPUT_HPP

#include <tpic/config/precision.hpp>

#include <bit>
#include <cstdint>
#include <iomanip>
#include <limits>
#include <locale>
#include <ostream>
#include <sstream>
#include <string_view>

namespace tpic::io::detail {

inline constexpr std::string_view vtk_byte_order =
    std::endian::native == std::endian::little ? "LittleEndian" : "BigEndian";

inline std::ostringstream vtk_xml_stream() {
  std::ostringstream xml;
  xml.imbue(std::locale::classic());
  xml << std::setprecision(std::numeric_limits<double>::max_digits10);
  return xml;
}

inline void vtk_field_data(std::ostream &xml, int cycle, double time, bool final_state, config::PrecisionMetadata precision) {
  xml << "    <FieldData>\n"
      << "      <DataArray type=\"Float64\" Name=\"TimeValue\" NumberOfTuples=\"1\" format=\"ascii\">"
      << time << "</DataArray>\n"
      << "      <DataArray type=\"Int32\" Name=\"cycle\" NumberOfTuples=\"1\" format=\"ascii\">"
      << cycle << "</DataArray>\n"
      << "      <DataArray type=\"UInt8\" Name=\"final_state\" NumberOfTuples=\"1\" format=\"ascii\">"
      << (final_state ? 1 : 0) << "</DataArray>\n";
  // VTK String arrays use ASCII byte values followed by a zero terminator.
  const auto string_array = [&xml](std::string_view name, std::string_view value) {
    xml << "      <Array type=\"String\" Name=\"" << name
        << "\" NumberOfTuples=\"1\" format=\"ascii\">";
    for (const unsigned char character : value) xml << static_cast<unsigned>(character) << ' ';
    xml << "0</Array>\n";
  };
  string_array("state", final_state ? "final" : "pre_solve");
  string_array("precision_profile", precision.name);
  string_array("precision_roles", precision.roles);
  xml << "    </FieldData>\n";
}

// XML prolog and VTKFile element; the caller continues with the dataset element.
inline void vtk_file_open(std::ostream &xml, std::string_view type) {
  xml << "<?xml version=\"1.0\"?>\n"
      << "<VTKFile type=\"" << type << "\" version=\"1.0\" byte_order=\"" << vtk_byte_order
      << "\" header_type=\"UInt64\">\n";
}

// Appended-data array descriptor of a piece.
inline void vtk_data_array(std::ostream &xml, std::string_view type, std::string_view name,
                           int components, std::uint64_t offset) {
  xml << "        <DataArray type=\"" << type << "\" Name=\"" << name
      << "\" NumberOfComponents=\"" << components
      << "\" format=\"appended\" offset=\"" << offset << "\"/>\n";
}

// Array descriptor of a parallel manifest.
inline void vtk_parallel_array(std::ostream &xml, std::string_view type, std::string_view name,
                               int components) {
  xml << "      <PDataArray type=\"" << type << "\" Name=\"" << name
      << "\" NumberOfComponents=\"" << components << "\"/>\n";
}

// Raw appended payload: the opening ends with the '_' marker, each array is a
// UInt64 byte count followed by its bytes, and the closing ends the file.
inline constexpr std::string_view vtk_appended_data_open = "  <AppendedData encoding=\"raw\">\n_";
inline constexpr std::string_view vtk_appended_data_close = "\n  </AppendedData>\n</VTKFile>\n";

} // namespace tpic::io::detail

#endif // TPIC_IO_DETAIL_VTK_OUTPUT_HPP
