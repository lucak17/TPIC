#ifndef LPFX_ALGORITHM_PARTICLE_DEVICE_SORT_HPP
#define LPFX_ALGORITHM_PARTICLE_DEVICE_SORT_HPP

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
#include <limits>
#include <optional>
#include <type_traits>
#include <utility>

#include <lpfx/algorithm/particle/cell_bins.hpp>
#include <lpfx/algorithm/particle/device/launch.hpp>
#include <lpfx/algorithm/particle/key.hpp>
#include <lpfx/algorithm/scan/device/workspace.hpp>
#include <lpfx/backend/handles.hpp>
#include <lpfx/backend/launch.hpp>
#include <lpfx/backend/operation_registration.hpp>
#include <lpfx/backend/properties.hpp>
#include <lpfx/core/checked_arithmetic.hpp>
#include <lpfx/core/error.hpp>
#include <lpfx/core/private_construction.hpp>
#include <lpfx/core/result.hpp>
#include <lpfx/data/particle/device/storage.hpp>
#include <lpfx/data/particle/device/view.hpp>
#include <lpfx/execution/device/completion.hpp>
#include <lpfx/memory/host/budget.hpp>
#include <lpfx/memory/native_allocation.hpp>

namespace lpfx {

// Device workspace of one cell sort, in byte order: the summary, bin_count
// counts and bin_count + 1 offsets (the published image, read back as one
// prefix), then the bin scan scratch. The permutation uses offsets[1,
// bin_count] as its bin cursors. The capacity-dependent permutation is a
// separate allocation.
template <detail::DeviceScanLaunchPolicyType LaunchPolicy =
              DefaultDeviceScanLaunchPolicy>
class PreparedDeviceParticleSortWorkspace final {
public:
  using launch_policy = LaunchPolicy;
  using scan_workspace_type =
      PreparedDeviceScanWorkspace<std::uint32_t, LaunchPolicy>;

  constexpr PreparedDeviceParticleSortWorkspace() noexcept = default;

  [[nodiscard]] constexpr bool valid() const noexcept { return valid_; }
  [[nodiscard]] constexpr std::uint32_t particle_capacity() const noexcept {
    return particle_capacity_;
  }
  [[nodiscard]] constexpr std::uint32_t bin_count() const noexcept {
    return bin_count_;
  }
  [[nodiscard]] constexpr std::int32_t device_ordinal() const noexcept {
    return device_ordinal_;
  }
  [[nodiscard]] constexpr const scan_workspace_type &
  scan_workspace() const noexcept {
    return scan_workspace_;
  }
  [[nodiscard]] constexpr std::size_t counts_byte_offset() const noexcept {
    return sizeof(DeviceParticleSortSummary);
  }
  [[nodiscard]] constexpr std::size_t counts_bytes() const noexcept {
    return static_cast<std::size_t>(bin_count_) * sizeof(std::uint32_t);
  }
  [[nodiscard]] constexpr std::size_t offsets_byte_offset() const noexcept {
    return counts_byte_offset() + counts_bytes();
  }
  [[nodiscard]] constexpr std::size_t offsets_bytes() const noexcept {
    return counts_bytes() + sizeof(std::uint32_t);
  }
  // Summary, counts and offsets.
  [[nodiscard]] constexpr std::size_t publication_bytes() const noexcept {
    return offsets_byte_offset() + offsets_bytes();
  }
  [[nodiscard]] constexpr std::size_t
  scan_scratch_byte_offset() const noexcept {
    return publication_bytes();
  }
  // Capacity-independent workspace bytes.
  [[nodiscard]] constexpr std::size_t fixed_workspace_bytes() const noexcept {
    return scan_scratch_byte_offset() + scan_workspace_.total_bytes();
  }
  [[nodiscard]] constexpr std::size_t permutation_bytes() const noexcept {
    return static_cast<std::size_t>(particle_capacity_) * sizeof(std::uint32_t);
  }
  [[nodiscard]] constexpr std::size_t total_bytes() const noexcept {
    return fixed_workspace_bytes() + permutation_bytes();
  }

  [[nodiscard]] friend constexpr bool
  operator==(const PreparedDeviceParticleSortWorkspace &,
             const PreparedDeviceParticleSortWorkspace &) noexcept = default;

private:
  constexpr PreparedDeviceParticleSortWorkspace(
      const std::uint32_t particle_capacity, const std::uint32_t bin_count,
      const scan_workspace_type scan_workspace,
      const std::int32_t device_ordinal) noexcept
      : scan_workspace_(scan_workspace), particle_capacity_(particle_capacity),
        bin_count_(bin_count), device_ordinal_(device_ordinal), valid_(true) {}

  scan_workspace_type scan_workspace_{};
  std::uint32_t particle_capacity_{};
  std::uint32_t bin_count_{};
  std::int32_t device_ordinal_{};
  bool valid_{};

  friend struct detail::PrivateConstructionAccess;
};

template <detail::DeviceScanLaunchPolicyType LaunchPolicy =
              DefaultDeviceScanLaunchPolicy,
          detail::checked_integer ParticleCapacity,
          detail::checked_integer BinCount>
[[nodiscard]] constexpr ErrorCode try_prepare_device_particle_sort_workspace(
    const ParticleCapacity requested_particle_capacity,
    const BinCount requested_bin_count,
    const device::DeviceProperties &properties,
    PreparedDeviceParticleSortWorkspace<LaunchPolicy> &output) noexcept {
  using Prepared = PreparedDeviceParticleSortWorkspace<LaunchPolicy>;
  if constexpr (std::is_signed_v<ParticleCapacity>) {
    if (requested_particle_capacity < ParticleCapacity{0}) {
      return ErrorCode::invalid_argument;
    }
  }
  if constexpr (std::is_signed_v<BinCount>) {
    if (requested_bin_count <= BinCount{0}) {
      return ErrorCode::invalid_argument;
    }
  } else if (requested_bin_count == BinCount{0}) {
    return ErrorCode::invalid_argument;
  }

  std::uint32_t particle_capacity{};
  std::uint32_t bin_count{};
  if (!detail::checked_integral_cast(requested_particle_capacity,
                                     particle_capacity) ||
      !detail::checked_integral_cast(requested_bin_count, bin_count)) {
    return ErrorCode::overflow;
  }

  constexpr std::uint32_t block_threads = LaunchPolicy::block_threads;
  if (properties.maximum_threads_per_block < block_threads ||
      properties.maximum_grid_size_x <
          detail::device_block_count(particle_capacity, block_threads) ||
      properties.maximum_grid_size_x <
          detail::device_block_count(bin_count, block_threads)) {
    return ErrorCode::invalid_argument;
  }

  typename Prepared::scan_workspace_type scan_workspace{};
  LPFX_TRY((try_prepare_device_scan_workspace<std::uint32_t, LaunchPolicy>(
      bin_count, properties, scan_workspace)));

  // Summary, counts and offsets: 2 * bin_count + 2 words.
  std::size_t fixed_bytes{};
  std::size_t permutation_bytes{};
  std::size_t total_bytes{};
  if (!detail::checked_multiply(static_cast<std::size_t>(bin_count),
                                std::size_t{2}, fixed_bytes) ||
      !detail::checked_add(fixed_bytes, std::size_t{2}, fixed_bytes) ||
      !detail::checked_byte_count<std::uint32_t>(fixed_bytes, fixed_bytes) ||
      !detail::checked_add(fixed_bytes, scan_workspace.total_bytes(),
                           fixed_bytes) ||
      !detail::checked_byte_count<std::uint32_t>(particle_capacity,
                                                 permutation_bytes) ||
      !detail::checked_add(fixed_bytes, permutation_bytes, total_bytes)) {
    return ErrorCode::overflow;
  }

  output = detail::PrivateConstructionAccess::create<Prepared>(
      particle_capacity, bin_count, scan_workspace, properties.device_ordinal);
  return ErrorCode::success;
}

// Epoch capability issued by migration once the outgoing image is independent
// and resident compaction has been enqueued. It lets the sanctioned sorter
// reorder [0, count) of the stamped owner while the issuing lease is open.
template <detail::ParticleSchemaType Schema, device::DeviceEventBackend Backend>
struct ResidentReorderRange final {
  using schema_type = Schema;
  using backend_type = Backend;

  DeviceParticleOwnerStamp owner{};
  std::uint32_t count = 0;
  std::uint64_t generation = 0;
  const std::uint64_t *epoch_source = nullptr;
  const bool *lease_active = nullptr;
  // Recorded once the reorderable records are final.
  const device::DeviceEvent<Backend> *ready = nullptr;
  // Upstream device failure word; a nonzero value suppresses the sort.
  const std::uint32_t *upstream_failure = nullptr;

  [[nodiscard]] bool valid() const noexcept {
    return ready != nullptr && !ready->empty() && epoch_source != nullptr &&
           lease_active != nullptr && *lease_active &&
           *epoch_source == generation && generation != 0 &&
           count <= owner.capacity();
  }
};

namespace detail {

// Sorter state shared with its receipts. Receipts of older epochs, and every
// receipt once the epoch is retired, are invalid.
template <ParticleSchemaType Schema, device::DeviceEventBackend Backend>
struct DeviceSortControl final {
  DeviceCellBinsView bins{};
  DeviceParticleSlotMapping<Schema> mapping{};
  device::DeviceEvent<Backend> producer{};
  Particles<Schema, DeviceSpace<Backend>> *owner = nullptr;
  const std::uint32_t *failure = nullptr;
  std::uint64_t epoch = 0;
  // Owner epoch the live mapping belongs to.
  std::uint64_t source_generation = 0;
  bool live = false;
};

} // namespace detail

template <detail::ParticleSchemaType Schema, device::DeviceEventBackend Backend>
class ResidentParticleRead;

// Borrowed, single-control-thread receipt. Every consumer must be ordered after
// the producer and test failure before reading bins or the pending mapping.
// The owner remains unpublished until a checked completion boundary.
template <detail::ParticleSchemaType Schema, device::DeviceEventBackend Backend>
class DeviceSortSubmission final {
public:
  using control_type = detail::DeviceSortControl<Schema, Backend>;

  DeviceSortSubmission() noexcept = default;
  DeviceSortSubmission(DeviceSortSubmission &&other) noexcept
      : control_(std::exchange(other.control_, nullptr)), epoch_(other.epoch_),
        migration_generation_(other.migration_generation_),
        prefix_(other.prefix_) {}
  DeviceSortSubmission &operator=(DeviceSortSubmission &&other) noexcept {
    control_ = std::exchange(other.control_, nullptr);
    epoch_ = other.epoch_;
    migration_generation_ = other.migration_generation_;
    prefix_ = other.prefix_;
    return *this;
  }

  [[nodiscard]] bool valid() const noexcept {
    return control_ != nullptr && control_->owner != nullptr &&
           control_->live && control_->epoch == epoch_ &&
           control_->mapping.owner.matches(*control_->owner) &&
           DeviceParticleOwnerStamp::capture(*control_->owner).generation() ==
               control_->source_generation;
  }
  [[nodiscard]] DeviceCellBinsView bins() const noexcept {
    require_valid();
    return control_->bins;
  }
  [[nodiscard]] auto pending_particle_view() const noexcept {
    require_valid();
    return detail::DeviceParticlesAccess::pending_view<Schema, Backend>(
        control_->mapping, prefix_);
  }
  [[nodiscard]] auto resident_particle_view() const noexcept {
    return BasicParticleView<Schema, DeviceSpace<Backend>, true>{
        pending_particle_view()};
  }
  [[nodiscard]] std::uint32_t prefix_size() const noexcept { return prefix_; }
  [[nodiscard]] std::uint64_t generation() const noexcept { return epoch_; }
  [[nodiscard]] std::uint64_t migration_generation() const noexcept {
    return migration_generation_;
  }
  [[nodiscard]] const std::uint32_t *failure_word() const noexcept {
    require_valid();
    return control_->failure;
  }
  [[nodiscard]] bool issued_by(const control_type &control) const noexcept {
    return control_ == &control;
  }
  [[nodiscard]] auto
  wait_on(const device::Executor<Backend> executor) const noexcept {
    // Invalid borrowed output is a programming error, before any native call.
    if (!valid() || !executor.bound())
      std::abort();
    return control_->producer.wait_on(executor);
  }

private:
  DeviceSortSubmission(const control_type &control, const std::uint64_t epoch,
                       const std::uint64_t migration_generation,
                       const std::uint32_t prefix) noexcept
      : control_(&control), epoch_(epoch),
        migration_generation_(migration_generation), prefix_(prefix) {}

  void require_valid() const noexcept {
    if (!valid())
      std::abort();
  }

  const control_type *control_{};
  std::uint64_t epoch_{};
  std::uint64_t migration_generation_{};
  std::uint32_t prefix_{};

  friend class ResidentParticleRead<Schema, Backend>;
  friend struct detail::PrivateConstructionAccess;
};

// Read-only snapshot issued after the sorter has checked successful completion.
// It borrows the receipt and owner, which must outlive it. Copying this
// small capability preserves its epoch and storage incarnation: later growth,
// receipt retirement or another sort invalidates every older snapshot.
template <detail::ParticleSchemaType Schema, device::DeviceEventBackend Backend>
class ResidentParticleRead final {
public:
  ResidentParticleRead() noexcept = default;

  [[nodiscard]] bool valid() const noexcept {
    return receipt_ != nullptr && receipt_->valid() &&
           receipt_->epoch_ == epoch_ &&
           receipt_->control_->mapping.owner.allocation_generation() ==
               storage_incarnation_;
  }
  [[nodiscard]] auto resident_particle_view() const noexcept {
    require_valid();
    return BasicParticleView<Schema, DeviceSpace<Backend>, true>{
        detail::DeviceParticlesAccess::pending_view<Schema, Backend>(
            receipt_->control_->mapping, receipt_->prefix_)};
  }
  [[nodiscard]] DeviceCellBinsView bins() const noexcept {
    require_valid();
    return receipt_->control_->bins;
  }
  [[nodiscard]] std::uint32_t prefix_size() const noexcept {
    require_valid();
    return receipt_->prefix_;
  }
  [[nodiscard]] const std::uint32_t *failure_word() const noexcept {
    require_valid();
    return receipt_->control_->failure;
  }
  [[nodiscard]] std::uint64_t generation() const noexcept { return epoch_; }
  [[nodiscard]] auto
  wait_on(const device::Executor<Backend> executor) const noexcept {
    if (!valid() || !executor.bound())
      std::abort();
    return receipt_->control_->producer.wait_on(executor);
  }

private:
  explicit ResidentParticleRead(
      const DeviceSortSubmission<Schema, Backend> &receipt) noexcept
      : receipt_(&receipt), epoch_(receipt.epoch_),
        storage_incarnation_(
            receipt.control_ == nullptr
                ? 0
                : receipt.control_->mapping.owner.allocation_generation()) {}

  void require_valid() const noexcept {
    if (!valid())
      std::abort();
  }

  const DeviceSortSubmission<Schema, Backend> *receipt_{};
  std::uint64_t epoch_{};
  std::uint64_t storage_incarnation_{};

  friend struct detail::PrivateConstructionAccess;
};

// Direct-cell counting sort of one device owner that rotates the owner's
// independently allocated particle slots. The histogram and atomic scatter
// are unstable within each bin. enqueue() returns a borrowed receipt; the
// owner stays unpublished until a checked completion (inspect() followed by
// conclude()/wait(), or complete()) commits the rotation, and the sorter is
// reusable after retire_after(). The owner must stay alive at its address;
// capacity growth goes through release_growth_scratch() and
// rebind_after_growth(). Invalid keys publish nothing and keep the owner.
template <detail::ParticleSchemaType Schema,
          detail::ParticleKeyAccessor KeyAccessor,
          detail::DeviceParticleSortBackend Backend,
          detail::DeviceScanLaunchPolicyType LaunchPolicy =
              DefaultDeviceScanLaunchPolicy>
  requires detail::ParticleDirectCellKeyFor<Schema, KeyAccessor>
class DeviceCellSort final {
public:
  using schema_type = Schema;
  using backend_type = Backend;
  using particle_owner = Particles<Schema, DeviceSpace<Backend>>;
  using submission_type = DeviceSortSubmission<Schema, Backend>;
  using prepared_type = PreparedDeviceParticleSortWorkspace<LaunchPolicy>;
  using traits_type = device::BackendTraits<Backend>;
  using executor_type = device::Executor<Backend>;
  using key_type = detail::particle_key_t<Schema, KeyAccessor>;
  using host_bins_type = CellBins<std::uint32_t, std::uint32_t>;

  DeviceCellSort() noexcept = default;
  DeviceCellSort(DeviceCellSort &&other) noexcept
      : scratch_(std::move(other.scratch_)),
        permutation_(std::move(other.permutation_)),
        reuse_(std::move(other.reuse_)),
        inspection_(std::move(other.inspection_)),
        control_(std::move(other.control_)), prepared_(other.prepared_),
        owner_(other.owner_), prepared_stream_(other.prepared_stream_),
        inspection_bytes_(other.inspection_bytes_) {
    // A borrowed receipt refers to the source's control state.
    if (control_.live || other.retirement_pending_ || other.growth_released_)
      std::abort();
    other.control_.owner = nullptr;
  }
  DeviceCellSort &operator=(DeviceCellSort &&) = delete;
  ~DeviceCellSort() noexcept {
    if (control_.live || retirement_pending_)
      std::abort();
  }

  [[nodiscard]] bool valid() const noexcept { return control_.owner != nullptr; }
  [[nodiscard]] bool reusable() const noexcept {
    return owner_matches() && !control_.live && !growth_released_;
  }
  [[nodiscard]] particle_owner *bound_owner() const noexcept {
    return control_.owner;
  }
  [[nodiscard]] std::uint32_t capacity() const noexcept {
    return prepared_.particle_capacity();
  }
  [[nodiscard]] const void *allocation_identity() const noexcept {
    return owner_.allocation();
  }
  [[nodiscard]] const void *scratch_identity() const noexcept {
    return scratch_.raw_base();
  }
  [[nodiscard]] std::size_t scratch_capacity_bytes() const noexcept {
    return scratch_.capacity_bytes();
  }

  // Setup-only device/stream affinity validation. Enqueue rechecks only the
  // stable host identity/capacity operands, never queries native device state.
  [[nodiscard]] ErrorCode validate_binding(particle_owner &owner,
                                           const executor_type executor) noexcept {
    if (control_.live)
      return ErrorCode::invalid_state;
    if (!valid() || &owner != control_.owner || !owner_matches() ||
        !executor.bound())
      return ErrorCode::invalid_argument;
    if constexpr (device::DeviceDiscoveryBackend<Backend>) {
      std::int32_t ordinal{};
      if (!traits_type::succeeded(traits_type::query_stream_device_ordinal(
              executor.stream(), ordinal)))
        return ErrorCode::backend_failure;
      if (ordinal != prepared_.device_ordinal())
        return ErrorCode::invalid_argument;
      for (const void *pointer :
           {static_cast<const void *>(owner_.allocation()),
            static_cast<const void *>(scratch_.raw_base()),
            static_cast<const void *>(permutation_.raw_base())}) {
        if (pointer == nullptr)
          continue;
        if (!traits_type::succeeded(
                traits_type::query_device_pointer_ordinal(pointer, ordinal)))
          return ErrorCode::backend_failure;
        if (ordinal != prepared_.device_ordinal())
          return ErrorCode::invalid_argument;
      }
    }
    prepared_stream_ = executor.stream();
    return ErrorCode::success;
  }

  [[nodiscard]] Result<submission_type>
  enqueue(const executor_type executor) noexcept {
    return enqueue_count<false>(
        control_.owner == nullptr ? 0 : control_.owner->size(), executor,
        nullptr, nullptr, 0);
  }
  [[nodiscard]] Result<submission_type>
  enqueue(ResidentReorderRange<Schema, Backend> &&range,
          const executor_type executor) noexcept {
    return enqueue_resident<true>(range, executor);
  }
  // Caller has explicitly placed *range.ready on this executor.
  [[nodiscard]] Result<submission_type>
  enqueue_ready(ResidentReorderRange<Schema, Backend> &&range,
                const executor_type executor) noexcept {
    return enqueue_resident<false>(range, executor);
  }

  // Retirement joins readers on the sort producer stream. Every independent
  // particle writer must inherit that dependency before its first access.
  void order_reuse(const executor_type executor) const noexcept {
    if (retirement_pending_ && executor.stream() != producer_stream_)
      device::abort_on_failure<Backend>(reuse_.wait_on(executor));
  }

  [[nodiscard]] std::optional<ResidentParticleRead<Schema, Backend>>
  acquire_resident_read(const submission_type &receipt) const noexcept {
    if (!matches(receipt) || !checked_ || !successful_ || growth_released_ ||
        receipt.migration_generation() == 0)
      return std::nullopt;
    return detail::PrivateConstructionAccess::create<
        ResidentParticleRead<Schema, Backend>>(receipt);
  }

  // Admits and allocates the pinned failure-word readback of inspect().
  [[nodiscard]] ErrorCode
  prepare_status_inspection(HostMemoryBudget &budget) noexcept {
    return prepare_inspection(budget, sizeof(std::uint32_t));
  }
  // As prepare_status_inspection(), but inspect() reads back the whole bins
  // image so that publish_host_bins() can copy it to host bins.
  [[nodiscard]] ErrorCode
  prepare_bins_inspection(HostMemoryBudget &budget) noexcept {
    return prepare_inspection(budget, prepared_.publication_bytes());
  }

  [[nodiscard]] ErrorCode inspect(const submission_type &receipt,
                                  const executor_type executor) noexcept {
    if (!matches(receipt) || inspection_bytes_ == 0 || inspection_pending_ ||
        checked_ || !executor.bound())
      return ErrorCode::invalid_state;
    if (executor.stream() != producer_stream_)
      device::abort_on_failure<Backend>(control_.producer.wait_on(executor));
    inspection_.enqueue(scratch_.raw_base(), inspection_bytes_, executor);
    inspection_pending_ = true;
    return ErrorCode::success;
  }
  [[nodiscard]] Status conclude(const submission_type &receipt) noexcept {
    return finish_inspection<false>(receipt);
  }
  [[nodiscard]] Status wait(const submission_type &receipt) noexcept {
    return finish_inspection<true>(receipt);
  }

  // Host-bins adapter: once conclude()/wait() of receipt succeeded with a
  // bins inspection, copies the read-back image into bins and publishes a
  // new completed generation.
  [[nodiscard]] ErrorCode publish_host_bins(const submission_type &receipt,
                                            host_bins_type &bins) noexcept {
    if (!matches(receipt) || !successful_ ||
        inspected_epoch_ != control_.epoch ||
        inspection_bytes_ != prepared_.publication_bytes())
      return ErrorCode::invalid_state;
    if (!bins.valid() || bins.bin_count() != prepared_.bin_count())
      return ErrorCode::invalid_argument;
    if (!bins.can_publish())
      return ErrorCode::overflow;
    const std::byte *const image = inspection_.data();
    std::memcpy(detail::CellBinsAccess::counts(bins),
                image + prepared_.counts_byte_offset(), prepared_.counts_bytes());
    std::memcpy(detail::CellBinsAccess::offsets(bins),
                image + prepared_.offsets_byte_offset(),
                prepared_.offsets_bytes());
    detail::CellBinsAccess::publish(bins);
    return ErrorCode::success;
  }

  // Precondition: an external completion/error boundary has finished all
  // producer accesses and obtained this attempt's stable failure word.
  [[nodiscard]] Status complete(const submission_type &receipt,
                                const std::uint32_t checked_failure) noexcept {
    if (!matches(receipt) || checked_ || inspection_pending_)
      return Status::failure(ErrorCode::invalid_state);
    if (checked_failure == 0) {
      if (detail::DeviceParticlesAccess::commit_reorder(
              *control_.owner, control_.mapping) != ErrorCode::success)
        std::abort();
      control_.source_generation = control_.mapping.generation();
    }
    checked_ = true;
    successful_ = checked_failure == 0;
    if (checked_failure == 0)
      return Status::success();
    if (checked_failure ==
        static_cast<std::uint32_t>(
            detail::DeviceParticleSortFailureWord::invalid_key))
      return Status::failure(ErrorCode::invalid_argument);
    // Upstream failures in accepted migration are terminal, never checked
    // success.
    return Status::failure(ErrorCode::backend_failure);
  }

  [[nodiscard]] ErrorCode
  retire_after(submission_type &&receipt,
               const device::DeviceEvent<Backend> &last_consumer) noexcept {
    if (!matches(receipt) || !checked_ || inspection_pending_ ||
        last_consumer.empty())
      return ErrorCode::invalid_state;
    const executor_type executor{producer_stream_};
    device::abort_on_failure<Backend>(last_consumer.wait_on(executor));
    device::abort_on_failure<Backend>(reuse_.record(executor));
    receipt = submission_type{};
    control_.live = false;
    checked_ = false;
    retirement_pending_ = true;
    return ErrorCode::success;
  }
  [[nodiscard]] bool quiescent_for_close() noexcept {
    if (control_.live)
      return false;
    if (retirement_pending_) {
      const Status retired = reuse_.template advance<false>();
      if (retired.is_pending())
        return false;
      if (retired.is_failure())
        std::abort();
      retirement_pending_ = false;
    }
    return true;
  }
  [[nodiscard]] Status wait_for_retirement() noexcept {
    if (control_.live)
      return Status::failure(ErrorCode::invalid_state);
    if (retirement_pending_) {
      if (reuse_.template advance<true>().is_failure())
        std::abort();
      retirement_pending_ = false;
    }
    return Status::success();
  }

  // The caller has joined every particle reader before opening this exclusive
  // growth interval. Bins and status keep their allocation and borrowed identity;
  // the dead capacity-dependent permutation is released before main growth.
  [[nodiscard]] ErrorCode release_growth_scratch() noexcept {
    if (growth_released_)
      return ErrorCode::success;
    if (!owner_matches() || inspection_pending_ ||
        (control_.live && (!checked_ || !successful_)) ||
        (!control_.live && !quiescent_for_close()))
      return ErrorCode::invalid_state;
    permutation_ = {};
    growth_released_ = true;
    return ErrorCode::success;
  }

  // Retryable after allocation failure: main storage may already be grown,
  // but no new sort is admitted until its permutation has been recreated.
  [[nodiscard]] ErrorCode rebind_after_growth(
      particle_owner &owner, const device::DeviceProperties &properties,
      submission_type *const receipt = nullptr) noexcept {
    if (!growth_released_ || &owner != control_.owner || !owner.valid() ||
        owner.capacity() < capacity() ||
        properties.device_ordinal != prepared_.device_ordinal() ||
        inspection_pending_ || retirement_pending_ ||
        (control_.live &&
         (!checked_ || !successful_ || receipt == nullptr ||
          !receipt->issued_by(control_) ||
          receipt->generation() != control_.epoch)) ||
        (!control_.live && receipt != nullptr))
      return ErrorCode::invalid_state;
    prepared_type prepared;
    LPFX_TRY(try_prepare_device_particle_sort_workspace<LaunchPolicy>(
        owner.capacity(), prepared_.bin_count(), properties, prepared));
    if (prepared.fixed_workspace_bytes() != prepared_.fixed_workspace_bytes())
      return ErrorCode::invalid_state;
    if (!traits_type::succeeded(device::DeviceAllocation<Backend>::create(
            prepared.permutation_bytes(), permutation_)))
      return ErrorCode::allocation_failed;
    prepared_ = prepared;
    owner_ = DeviceParticleOwnerStamp::capture(owner);
    control_.mapping = detail::DeviceParticlesAccess::current_mapping(owner);
    control_.source_generation = owner_.generation();
    if (control_.live) {
      // In-place growth completes each feature copy before publication.
      // Rerecording does not require a bin copy or new event allocation.
      device::abort_on_failure<Backend>(
          control_.producer.record(executor_type{prepared_stream_}));
      producer_stream_ = prepared_stream_;
    }
    growth_released_ = false;
    return ErrorCode::success;
  }

  // Deduced: Result<DeviceCellSort> needs the complete class.
  [[nodiscard]] static auto create(const prepared_type &prepared,
                                   const DirectCellBinLayout<key_type> layout,
                                   particle_owner &particles) noexcept {
    using Output = Result<DeviceCellSort>;
    if (!prepared.valid() || !layout.valid() ||
        layout.bin_count() != prepared.bin_count() || !particles.valid() ||
        particles.capacity() != prepared.particle_capacity())
      return Output::from_error(ErrorCode::invalid_argument);
    DeviceCellSort result;
    if (!traits_type::succeeded(device::DeviceAllocation<Backend>::create(
            prepared.fixed_workspace_bytes(), result.scratch_)) ||
        !traits_type::succeeded(device::DeviceAllocation<Backend>::create(
            prepared.permutation_bytes(), result.permutation_)))
      return Output::from_error(ErrorCode::allocation_failed);
    if (!traits_type::succeeded(
            device::DeviceEvent<Backend>::create(result.control_.producer)) ||
        !traits_type::succeeded(
            device::DeviceEvent<Backend>::create(result.reuse_)))
      return Output::from_error(ErrorCode::backend_failure);
    std::byte *const base = result.scratch_.raw_base();
    result.control_.bins = DeviceCellBinsView{
        reinterpret_cast<const std::uint32_t *>(
            base + prepared.counts_byte_offset()),
        reinterpret_cast<const std::uint32_t *>(
            base + prepared.offsets_byte_offset()),
        prepared.bin_count()};
    result.control_.failure =
        &reinterpret_cast<DeviceParticleSortSummary *>(base)->failure;
    result.control_.owner = &particles;
    result.prepared_ = prepared;
    result.owner_ = DeviceParticleOwnerStamp::capture(particles);
    return Output::from_value(std::move(result));
  }

private:
  [[nodiscard]] bool owner_matches() const noexcept {
    return control_.owner != nullptr && owner_.matches(*control_.owner);
  }
  [[nodiscard]] bool matches(const submission_type &receipt) const noexcept {
    return receipt.valid() && receipt.issued_by(control_);
  }

  [[nodiscard]] ErrorCode prepare_inspection(HostMemoryBudget &budget,
                                             const std::size_t bytes) noexcept {
    if (!valid() || control_.live || retirement_pending_)
      return ErrorCode::invalid_state;
    if (inspection_bytes_ == bytes)
      return ErrorCode::success;
    auto made = detail::PinnedReadback<Backend>::create(&budget, bytes);
    LPFX_TRY(made);
    inspection_ = std::move(made).value();
    inspection_bytes_ = bytes;
    return ErrorCode::success;
  }

  template <bool Wait>
  [[nodiscard]] Status
  finish_inspection(const submission_type &receipt) noexcept {
    if (!matches(receipt) || !inspection_pending_)
      return Status::failure(ErrorCode::invalid_state);
    const Status completed = inspection_.template advance<Wait>();
    if (completed.is_pending())
      return completed;
    if (completed.is_failure())
      std::abort();
    inspection_pending_ = false;
    inspected_epoch_ = control_.epoch;
    std::uint32_t failure{};
    std::memcpy(&failure, inspection_.data(), sizeof(failure));
    return complete(receipt, failure);
  }

  template <bool OrderReady>
  [[nodiscard]] Result<submission_type>
  enqueue_resident(ResidentReorderRange<Schema, Backend> &range,
                   const executor_type executor) noexcept {
    if (!range.valid() || range.owner.allocation() != owner_.allocation() ||
        range.owner.allocation_generation() != owner_.allocation_generation() ||
        range.owner.capacity() != owner_.capacity())
      return Result<submission_type>::from_error(ErrorCode::invalid_argument);
    auto result = enqueue_count<true>(range.count, executor,
                                      OrderReady ? range.ready : nullptr,
                                      range.upstream_failure, range.generation);
    if (result.has_value())
      range = {};
    return result;
  }

  template <bool Leased>
  [[nodiscard]] Result<submission_type>
  enqueue_count(const std::uint32_t count, const executor_type executor,
                const device::DeviceEvent<Backend> *const upstream,
                const std::uint32_t *const upstream_failure,
                const std::uint64_t migration_generation) noexcept {
    using Output = Result<submission_type>;
    if (control_.live || growth_released_ || !owner_matches())
      return Output::from_error(ErrorCode::invalid_state);
    if (count > capacity() || !executor.bound() ||
        executor.stream() != prepared_stream_)
      return Output::from_error(ErrorCode::invalid_argument);
    if (control_.epoch == std::numeric_limits<std::uint64_t>::max())
      return Output::from_error(ErrorCode::overflow);
    detail::DeviceParticleSlotMapping<Schema> mapping{};
    LPFX_TRY_RESULT(detail::DeviceParticlesAccess::plan_reorder(
                        *control_.owner, count, mapping),
                    Output);
    // All recoverable admission checks precede the first native dependency.
    // Rerecording is safe: retirement submitted every previous reader wait.
    order_reuse(executor);
    if (upstream != nullptr)
      device::abort_on_failure<Backend>(upstream->wait_on(executor));
    std::byte *const base = scratch_.raw_base();
    bool accepted = false;
    const auto error =
        detail::DeviceParticleSortLaunch<Backend>::template enqueue<
            KeyAccessor, Schema, LaunchPolicy, Leased>(
            count, prepared_.bin_count(),
            detail::DeviceParticlesAccess::const_prefix_view(*control_.owner,
                                                             count),
            detail::DeviceParticlesAccess::pending_view<Schema, Backend>(
                mapping, capacity()),
            reinterpret_cast<std::uint32_t *>(base +
                                              prepared_.counts_byte_offset()),
            reinterpret_cast<std::uint32_t *>(base +
                                              prepared_.offsets_byte_offset()),
            reinterpret_cast<std::uint32_t *>(permutation_.raw_base()),
            reinterpret_cast<DeviceParticleSortSummary *>(base),
            upstream_failure, base + prepared_.scan_scratch_byte_offset(),
            prepared_.scan_workspace(), mapping.forward, executor.stream(),
            accepted);
    LPFX_TRY_RESULT(device::finish_device_enqueue<Backend>(error, accepted),
                    Output);
    device::abort_on_failure<Backend>(control_.producer.record(executor));
    control_.mapping = mapping;
    control_.source_generation = mapping.owner.generation();
    producer_stream_ = executor.stream();
    ++control_.epoch;
    control_.live = true;
    retirement_pending_ = false;
    checked_ = false;
    successful_ = false;
    return Output::from_value(
        detail::PrivateConstructionAccess::create<submission_type>(
            std::as_const(control_), control_.epoch, migration_generation,
            count));
  }

  device::DeviceAllocation<Backend> scratch_{};
  device::DeviceAllocation<Backend> permutation_{};
  device::DeviceEvent<Backend> reuse_{};
  detail::PinnedReadback<Backend> inspection_{};
  detail::DeviceSortControl<Schema, Backend> control_{};
  prepared_type prepared_{};
  DeviceParticleOwnerStamp owner_{};
  typename traits_type::stream_type prepared_stream_ = traits_type::null_stream;
  typename traits_type::stream_type producer_stream_ = traits_type::null_stream;
  std::uint64_t inspected_epoch_{};
  std::size_t inspection_bytes_{};
  bool checked_{};
  bool successful_{};
  bool inspection_pending_{};
  bool retirement_pending_{};
  bool growth_released_{};
};

template <detail::ParticleKeyAccessor Key, detail::ParticleSchemaType Schema,
          detail::DeviceParticleSortBackend Backend,
          detail::DeviceScanLaunchPolicyType Policy =
              DefaultDeviceScanLaunchPolicy>
  requires detail::ParticleDirectCellKeyFor<Schema, Key>
[[nodiscard]] Result<DeviceCellSort<Schema, Key, Backend, Policy>>
make_device_cell_sort(
    const PreparedDeviceParticleSortWorkspace<Policy> &prepared,
    const DirectCellBinLayout<detail::particle_key_t<Schema, Key>> layout,
    Particles<Schema, DeviceSpace<Backend>> &particles,
    const device::Executor<Backend> executor = device::Executor<Backend>{
        device::BackendTraits<Backend>::null_stream}) noexcept {
  auto result = DeviceCellSort<Schema, Key, Backend, Policy>::create(
      prepared, layout, particles);
  if (result.has_value() && executor.bound()) {
    LPFX_TRY_RESULT(result.value().validate_binding(particles, executor),
                    Result<DeviceCellSort<Schema, Key, Backend, Policy>>);
  }
  return result;
}

} // namespace lpfx

#if LPFX_NATIVE_DEVICE_OPERATIONS
#include <lpfx/algorithm/particle/device/cuda_hip/sort.hpp>
#endif

#endif
