#ifndef TPIC_ECSIM_DEPOSITION_CURRENT_ASSEMBLY_DEVICE_HPP
#define TPIC_ECSIM_DEPOSITION_CURRENT_ASSEMBLY_DEVICE_HPP

#include <lpfx/backend/cuda_hip/kernel.hpp>
#include <lpfx/core/error_reporting.hpp>
#include <tpic/ecsim/deposition/moment_field_device.hpp>


namespace tpic::ecsim {

// Assembly of Jhat over two borrowed halo operations: shared-node addition,
// dual-volume scale, then with smoothing the ghost copy and Neumann closure.
// The caller records the publication event of the deposition generation once
// progress() returns true, after every local closure kernel, the no-neighbour
// case included.
template <class Current, class Backend, class AddPort, class CopyPort>
class CurrentAssembly {
  using Executor = lpfx::device::Executor<Backend>;
  enum class Phase { idle, add, copy, complete };
public:
  CurrentAssembly(Current& current, const config::PreparedDomain& domain,
      bool smoothing, AddPort& add, CopyPort& copy, Executor executor)
      : current_(current), plan_(prepare_moment_field_plan(domain)), smoothing_(smoothing),
        add_(add), copy_(copy), executor_(executor) {}

  void begin() {
    lpfx::require(phase_ == Phase::idle || phase_ == Phase::complete,
                  "TPIC current assembly: prior generation active");
    phase_ = Phase::add;
    begin_exchange(add_);
  }

  [[nodiscard]] bool progress() {
    lpfx::require(phase_ != Phase::idle, "TPIC current assembly: no accepted generation");
    if (phase_ == Phase::complete) return true;
    if (phase_ == Phase::add) {
      if (!progress_exchange(add_)) return false;
      lpfx::check_error(ecsim::enqueue_scale_moment_dual_volume(current_, plan_, executor_),
                        "TPIC current assembly: physical-volume correction failed");
      if (!smoothing_) {
        phase_ = Phase::complete;
        return true;
      }
      phase_ = Phase::copy;
      begin_exchange(copy_);
      return false;
    }
    if (!progress_exchange(copy_)) return false;
    lpfx::check_error(ecsim::enqueue_fill_moment_neumann_ghost(current_, plan_, executor_),
                      "TPIC current assembly: Neumann ghost closure failed");
    phase_ = Phase::complete;
    return true;
  }

private:
  template <class Port> void begin_exchange(Port& port) {
    const auto status = port.begin();
    lpfx::check_error(status, "TPIC current assembly: exchange rejected");
    active_ = status.requires_conclusion();
    progressed_ = !active_;
  }
  template <class Port> bool progress_exchange(Port& port) {
    if (!active_) return true;
    if (!progressed_) {
      const auto status = port.progress();
      progressed_ = lpfx::check_progress(status, "TPIC current assembly: exchange progress failed");
      if (!progressed_) return false;
    }
    const auto status = port.conclude();
    if (!lpfx::check_progress(status, "TPIC current assembly: exchange conclusion failed"))
      return false;
    active_ = false;
    return true;
  }

  Current& current_;
  const MomentFieldPlan plan_;
  const bool smoothing_;
  AddPort& add_;
  CopyPort& copy_;
  Executor executor_;
  Phase phase_ = Phase::idle;
  bool active_{}, progressed_{};
};

} // namespace tpic::ecsim

#endif // TPIC_ECSIM_DEPOSITION_CURRENT_ASSEMBLY_DEVICE_HPP
