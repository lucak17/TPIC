#ifndef TPIC_DATA_MASS_MATRIX_COMPACT_MASS_MATRIX_DEVICE_HPP
#define TPIC_DATA_MASS_MATRIX_COMPACT_MASS_MATRIX_DEVICE_HPP

#include <lpfx/backend/cuda_hip/kernel.hpp>
#include <lpfx/backend/error_reporting.hpp>
#include <lpfx/memory/native_allocation.hpp>
#include <tpic/data/mass_matrix/mass_matrix_layout.hpp>

#include <cstddef>
#include <type_traits>

namespace tpic::data {

// One persistent coefficient allocation without the volume factor. Construct
// on the selected device, clear before every deposition, and keep alive until
// every deposition, mass-halo and apply reader has finished. Halo validity is
// established by the mass-halo exchange, not by this owner.
template <class Precision, class Backend> class CompactMassMatrix {
  using Allocation = lpfx::device::DeviceAllocation<Backend>;

public:
  using value_type = typename Precision::Storage::Mass;

  // Throws what prepare_mass_matrix_layout throws for an invalid domain. The
  // node count fits int, so the byte count cannot overflow std::size_t.
  explicit CompactMassMatrix(const config::PreparedDomain& domain)
      : layout_(prepare_mass_matrix_layout(domain)) {
    static_assert(std::is_same_v<value_type, float> || std::is_same_v<value_type, double>);
    static_assert(sizeof(std::size_t) >= 8, "int nodes x 126 entries x 8 bytes needs 64-bit sizes");
    lpfx::device::check_device_error<Backend>(
        Allocation::create(layout_.storage_size() * sizeof(value_type), storage_),
        "TPIC mass matrix: device allocation failed");
  }

  [[nodiscard]] bool initialized() const noexcept { return !storage_.empty(); }
  [[nodiscard]] MassMatrixLayout layout() const noexcept { return layout_; }
  [[nodiscard]] std::size_t storage_bytes() const noexcept { return storage_.capacity_bytes(); }
  [[nodiscard]] value_type* data() noexcept {
    return reinterpret_cast<value_type*>(storage_.raw_base());
  }
  [[nodiscard]] const value_type* data() const noexcept {
    return reinterpret_cast<const value_type*>(storage_.raw_base());
  }

private:
  MassMatrixLayout layout_{};
  Allocation storage_;
};

} // namespace tpic::data

#endif // TPIC_DATA_MASS_MATRIX_COMPACT_MASS_MATRIX_DEVICE_HPP
