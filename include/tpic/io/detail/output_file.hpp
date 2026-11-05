#ifndef TPIC_IO_DETAIL_OUTPUT_FILE_HPP
#define TPIC_IO_DETAIL_OUTPUT_FILE_HPP

#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <type_traits>
#include <utility>

#include <fcntl.h>
#include <unistd.h>

namespace tpic::io::detail {

inline std::string output_padded(std::uint64_t value) {
  auto digits = std::to_string(value);
  return std::string(digits.size() < 6 ? 6 - digits.size() : 0, '0') + digits;
}

inline std::string output_stem(std::string_view channel, std::uint64_t cycle) {
  return std::string(channel) + "_t" + output_padded(cycle);
}

// Extensions include their leading dot; padding expands for large runs.
inline std::string output_piece_name(std::string_view channel, std::uint64_t cycle,
                                     std::uint64_t rank, std::string_view extension) {
  return output_stem(channel, cycle) + "_r" + output_padded(rank) + std::string(extension);
}

// Filesystem policy shared by streaming writers. Format, cadence and overwrite
// authorization remain application concerns.
[[noreturn]] inline void output_error(std::string_view context, const char* operation,
                                      const std::string& path, int error) {
  throw std::system_error(error == 0 ? EIO : error, std::generic_category(),
      "TPIC " + std::string(context) + ": " + operation + " " + path +
      (error == EEXIST ? "; use --overwrite-output or a fresh output directory" : ""));
}

// A checked streaming sink. Explicit close reports delayed write errors;
// destruction closes an interrupted write without masking its exception.
class OutputFile {
public:
  OutputFile() = default;
  OutputFile(std::string path, bool overwrite, std::string context) {
    open(std::move(path), overwrite, std::move(context));
  }
  ~OutputFile() noexcept { if (file_ != nullptr) std::fclose(file_); }
  OutputFile(const OutputFile&) = delete;
  OutputFile& operator=(const OutputFile&) = delete;

  void open(std::string path, bool overwrite, std::string context) {
    open(std::move(path), std::move(context), overwrite ? O_TRUNC : O_EXCL);
  }
  // Keeps the first length bytes of the file, created when missing, and
  // appends after them.
  void open_after(std::string path, std::uintmax_t length, std::string context) {
    open(std::move(path), std::move(context), O_APPEND);
    if (::ftruncate(::fileno(file_), static_cast<off_t>(length)) != 0)
      output_error(context_, "truncate", path_, errno);
  }
  [[nodiscard]] bool is_open() const noexcept { return file_ != nullptr; }
  void write(const void* data, std::size_t bytes) {
    if (file_ == nullptr) throw std::logic_error("TPIC output: file is not open");
    if (bytes != 0 && std::fwrite(data, 1, bytes, file_) != bytes)
      output_error(context_, "write", path_, errno);
  }
  void write(std::string_view text) { write(text.data(), text.size()); }
  template <class T> void value(const T& value) {
    static_assert(std::is_trivially_copyable_v<T>);
    write(&value, sizeof(value));
  }
  void flush() {
    if (file_ == nullptr) throw std::logic_error("TPIC output: file is not open");
    if (std::fflush(file_) != 0) output_error(context_, "flush", path_, errno);
  }
  void close() {
    if (file_ == nullptr) throw std::logic_error("TPIC output: file is not open");
    auto* file = std::exchange(file_, nullptr);
    // fclose flushes buffered writes and reports flush/close failures together.
    if (std::fclose(file) != 0) output_error(context_, "close", path_, errno);
  }
private:
  void open(std::string path, std::string context, int flags) {
    if (file_ != nullptr) throw std::logic_error("TPIC output: file is already open");
    path_ = std::move(path);
    context_ = std::move(context);
    // Creation checks existence atomically; no mode follows a final symlink.
    const int fd = ::open(path_.c_str(), O_WRONLY | O_CREAT | O_CLOEXEC | O_NOFOLLOW | flags, 0666);
    if (fd < 0) output_error(context_, "open", path_, errno);
    file_ = ::fdopen(fd, "wb");
    if (file_ == nullptr) {
      const int error = errno;
      ::close(fd);
      output_error(context_, "stream open", path_, error);
    }
  }

  std::string path_;
  std::string context_;
  std::FILE* file_ = nullptr;
};

// Creates path, writes the complete text and closes it with checked errors.
inline void write_text_file(std::string path, bool overwrite, std::string context,
                            std::string_view text) {
  OutputFile file(std::move(path), overwrite, std::move(context));
  file.write(text);
  file.close();
}

} // namespace tpic::io::detail

#endif // TPIC_IO_DETAIL_OUTPUT_FILE_HPP
