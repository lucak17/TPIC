#ifndef TPIC_COMMUNICATION_FIELDS_FIELD_EXCHANGE_HPP
#define TPIC_COMMUNICATION_FIELDS_FIELD_EXCHANGE_HPP

#include <lpfx/core/error_reporting.hpp>
#include <lpfx/profiling.hpp>

#include <cstdlib>
#include <stdexcept>

namespace tpic::communication {

// An explicit accepted-field lifetime for schedules that enqueue independent
// kernels between begin and complete. The exchange and its bound owners must
// outlive this object. The submitting thread owns both progress and retirement;
// destructors never communicate. Even a no-neighbour field begin must conclude.
template <class Exchange> class FieldExchangeEpoch {
public:
  FieldExchangeEpoch(Exchange& exchange, const char* rejected, const char* failed) noexcept
      : exchange_(exchange), rejected_(rejected), failed_(failed) {}
  FieldExchangeEpoch(const FieldExchangeEpoch&) = delete;
  FieldExchangeEpoch& operator=(const FieldExchangeEpoch&) = delete;

  ~FieldExchangeEpoch() noexcept {
    // Assert the lifetime contract in release builds too; an accepted field
    // cannot safely be forgotten or implicitly canceled during unwinding.
    if (state_ != State::idle && state_ != State::concluded) std::abort();
  }

  void begin() {
    if (state_ != State::idle && state_ != State::concluded)
      throw std::logic_error("TPIC field epoch: begin requires a retired exchange");
    state_ = State::idle;
    if (!exchange_.begin().requires_conclusion()) throw std::runtime_error(rejected_);
    state_ = State::accepted;
  }

  // Polls once without waiting for the network; true once complete() has no
  // wait left. MPI posts its sends from a poll after the pack has finished;
  // NCCL/RCCL completed at begin.
  [[nodiscard]] bool progress() {
    if (state_ != State::accepted && state_ != State::wait_complete)
      throw std::logic_error("TPIC field epoch: progress requires an accepted exchange");
    if (state_ == State::accepted) {
      const auto status = exchange_.progress();
      if (status.is_pending()) return false;
      if (!status.is_success() || status.requires_conclusion()) fail();
      state_ = State::wait_complete;
    }
    return true;
  }

  // MPI blocks until the halos have arrived and enqueues their unpack;
  // NCCL/RCCL return at enqueue.
  void complete() {
    if (state_ != State::accepted && state_ != State::wait_complete)
      throw std::logic_error("TPIC field epoch: complete requires an accepted exchange");
    if (state_ == State::accepted) {
      const auto status = exchange_.wait();
      if (!status.is_success() || status.requires_conclusion()) fail();
      state_ = State::wait_complete;
    }
    // A pending conclusion retains ownership and does not repeat the wait.
    for (;;) {
      const auto status = exchange_.conclude();
      if (status.is_pending()) continue;
      if (!status.is_success() || status.requires_conclusion()) fail();
      state_ = State::concluded;
      return;
    }
  }

  // Call explicitly from the enclosing schedule's catch, before draining its
  // device streams or releasing borrowed owners. A failed wait/conclusion is
  // fail-stop: retrying or concluding live failed work is not recovery.
  void complete_or_abort() noexcept {
    if (state_ == State::idle || state_ == State::concluded) return;
    if (state_ == State::failed) std::abort();
    try {
      complete();
    } catch (...) {
      std::abort();
    }
  }

private:
  enum class State { idle, accepted, wait_complete, concluded, failed };

  [[noreturn]] void fail() {
    state_ = State::failed;
    throw std::runtime_error(failed_);
  }

  Exchange& exchange_;
  const char* rejected_;
  const char* failed_;
  State state_{State::idle};
};

// Begins, waits for and concludes one field exchange on the calling thread,
// throwing with `rejected` or `failed`. MPI blocks until the halos have
// arrived and enqueues their unpack; NCCL/RCCL return at enqueue; an exchange
// without a neighbour has no halo work. A host reader of the field completes
// the stream itself. Not for schedules that overlap work with the exchange:
// those use FieldExchangeEpoch.
template <class Exchange>
void complete_field_exchange(Exchange& exchange, const char* rejected, const char* failed) {
  LPFX_PROFILE_SCOPE("TPIC::FieldHaloExchange");
  const auto accepted = exchange.begin();
  lpfx::check_acceptance(accepted, rejected);
  // R1: wait, then conclude; a failed wait throws before live work is concluded.
  lpfx::check_error(exchange.wait(), failed);
  lpfx::check_error(exchange.conclude(), failed);
}

} // namespace tpic::communication

#endif // TPIC_COMMUNICATION_FIELDS_FIELD_EXCHANGE_HPP
