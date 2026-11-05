#include <lpfx/core/error.hpp>
#include <tpic/communication/fields/field_exchange.hpp>

#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

#if defined(__unix__)
#include <csignal>
#include <sys/resource.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

namespace {

void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

struct Exchange {
  lpfx::Status accepted = lpfx::Status::accepted();
  lpfx::Status waited = lpfx::Status::success();
  lpfx::Status concluded = lpfx::Status::success();
  unsigned begins{}, waits{}, concludes{}, pending_conclusions{};
  bool forbid_wait{}, forbid_conclude{};

  lpfx::Status begin() noexcept { ++begins; return accepted; }
  lpfx::Status wait() noexcept {
    if (forbid_wait) std::_Exit(9);
    ++waits;
    return waited;
  }
  lpfx::Status conclude() noexcept {
    if (forbid_conclude) std::_Exit(10);
    ++concludes;
    if (pending_conclusions) {
      --pending_conclusions;
      return lpfx::Status::pending();
    }
    return concluded;
  }
};

using Epoch = tpic::communication::FieldExchangeEpoch<Exchange>;
static_assert(!std::is_copy_constructible_v<Epoch> && !std::is_move_constructible_v<Epoch>);
static_assert(noexcept(std::declval<Epoch&>().complete_or_abort()));

void healthy_lifetimes() {
  Exchange unused;
  { Epoch input(unused, "rejected", "failed"), curl(unused, "rejected", "failed"); }
  require(unused.begins + unused.waits + unused.concludes == 0,
          "idle epoch destructor communicated");

  Exchange exchange;
  Epoch epoch(exchange, "rejected", "failed");
  for (unsigned n = 1; n <= 3; ++n) {
    epoch.begin();
    require(exchange.begins == n && exchange.waits == n - 1,
            "begin eagerly waited for communication");
    epoch.complete();
    epoch.complete_or_abort();
    require(exchange.waits == n && exchange.concludes == n,
            "successful or no-neighbor epoch did not conclude exactly once");
  }

  Exchange pending;
  pending.pending_conclusions = 2;
  Epoch pending_epoch(pending, "rejected", "failed");
  pending_epoch.begin();
  pending_epoch.complete();
  require(pending.waits == 1 && pending.concludes == 3,
          "pending conclusion repeated wait or lost its obligation");
}

void submission_exception() {
  Exchange input, unused_curl;
  Epoch input_epoch(input, "rejected", "failed"), curl_epoch(unused_curl, "rejected", "failed");
  bool helper_submitted = false;
  try {
    input_epoch.begin();
    helper_submitted = true;
    throw std::runtime_error("independent helper submission failed");
  } catch (const std::runtime_error&) {
    input_epoch.complete_or_abort();
    curl_epoch.complete_or_abort();
  }
  require(helper_submitted && input.begins == 1 && input.waits == 1 && input.concludes == 1,
          "accepted exchange was not retired after a helper failure");
  require(unused_curl.begins + unused_curl.waits + unused_curl.concludes == 0,
          "cleanup started an unused exchange");
}

void rejected_begin() {
  for (const auto rejected : {lpfx::Status::failure(lpfx::ErrorCode::invalid_state),
                              lpfx::Status::success(), lpfx::Status::pending()}) {
    Exchange exchange;
    exchange.accepted = rejected;
    Epoch epoch(exchange, "begin rejected", "failed");
    bool caught = false;
    try { epoch.begin(); }
    catch (const std::runtime_error& error) { caught = std::string(error.what()) == "begin rejected"; }
    epoch.complete_or_abort();
    require(caught && exchange.begins == 1 && exchange.waits == 0 && exchange.concludes == 0,
            "rejected begin created a conclusion obligation");
  }
  Exchange exchange;
  Epoch epoch(exchange, "rejected", "failed");
  epoch.begin();
  bool caught = false;
  try { epoch.begin(); }
  catch (const std::logic_error&) { caught = true; }
  epoch.complete_or_abort();
  require(caught && exchange.begins == 1 && exchange.waits == 1 && exchange.concludes == 1,
          "double begin changed the accepted operation");
}

#if defined(__unix__)
template <class Function> void aborts(Function function) {
  const auto child = fork();
  require(child >= 0, "cannot fork lifecycle failure check");
  if (child == 0) {
    const rlimit no_core{0, 0};
    static_cast<void>(setrlimit(RLIMIT_CORE, &no_core));
    function();
    std::_Exit(0);
  }
  int status = 0;
  require(waitpid(child, &status, 0) == child, "cannot wait for lifecycle failure check");
  require(WIFSIGNALED(status) && WTERMSIG(status) == SIGABRT,
          "failed lifecycle did not abort or called forbidden communication");
}

void failed_lifetimes() {
  for (bool fail_wait : {false, true}) {
    aborts([=] {
      Exchange exchange;
      if (fail_wait) {
        exchange.waited = lpfx::Status::failure(lpfx::ErrorCode::backend_failure);
        exchange.forbid_conclude = true;
      } else {
        exchange.concluded = lpfx::Status::failure(lpfx::ErrorCode::backend_failure);
      }
      Epoch epoch(exchange, "rejected", "failed epoch");
      epoch.begin();
      bool caught = false;
      try { epoch.complete(); }
      catch (const std::runtime_error& error) { caught = std::string(error.what()) == "failed epoch"; }
      if (!caught || exchange.waits != 1 || exchange.concludes != unsigned(!fail_wait))
        std::_Exit(11);
      exchange.forbid_wait = exchange.forbid_conclude = true;
      epoch.complete_or_abort(); // Failed work must not be retried or concluded.
    });
  }
  aborts([] {
    Exchange exchange;
    exchange.forbid_wait = exchange.forbid_conclude = true;
    { Epoch epoch(exchange, "rejected", "failed"); epoch.begin(); }
  }); // Destructor checks the lifetime contract, but must never initiate communication.
  aborts([] {
    Exchange exchange;
    exchange.waited = lpfx::Status::failure(lpfx::ErrorCode::backend_failure);
    exchange.forbid_conclude = true;
    Epoch epoch(exchange, "rejected", "failed");
    epoch.begin();
    epoch.complete_or_abort(); // Retirement itself can fail and must be fail-stop.
  });
}
#endif

} // namespace

int main() {
  try {
    healthy_lifetimes();
    submission_exception();
    rejected_begin();
#if defined(__unix__)
    failed_lifetimes();
#endif
    std::cout << "TPIC field exchange tests passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
