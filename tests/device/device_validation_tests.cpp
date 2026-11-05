#include <device/mock_backend.hpp>
#include <tpic/device/device_validation.hpp>

#include <iostream>
#include <stdexcept>
#include <string_view>

namespace {
using Backend = lpfx::device::MockBackend;
using Traits = lpfx::device::BackendTraits<Backend>;
using Validation = tpic::device::DeviceValidation<Backend>;
const lpfx::device::Executor<Backend> executor{lpfx::device::MockStream{1}};

void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

template <class Function> void require_message(Function&& function, const char* message) {
  try {
    function();
  } catch (const std::runtime_error& error) {
    require(std::string_view(error.what()) == message, "validation message was lost");
    return;
  }
  throw std::runtime_error("expected validation failure");
}

void application_policy() {
  Traits::reset();
  Validation validation;
  validation.clear(executor);
  *validation.data() = 7;
  validation.capture(executor);
  require_message([&] { validation.host_wait_check("species mover failed"); }, "species mover failed");
  validation.clear(executor);
  validation.capture(executor);
  validation.host_wait_check("previous producer contaminated this epoch");

  validation.clear(executor);
  *validation.data() = 2;
  validation.capture(executor);
  require(!validation.ready("nonzero flag accepted"), "pending capture reported ready");
  Traits::event_ready = true;
  require_message([&] { (void)validation.ready("nonzero flag accepted"); }, "nonzero flag accepted");
  validation.clear(executor);
  validation.capture(executor);
  Traits::event_ready = true;
  require(validation.ready("zero flag rejected"), "zero flag not ready");
}

void backend_error_messages() {
  Traits::reset();
  Traits::fail_next_allocation = true;
  require_message([] { Validation failed; },
                  "TPIC device validation: storage preparation failed: lpfx=allocation_failed(4)");
  Validation validation;
  require_message([&] { validation.capture(executor); },
                  "TPIC device validation: capture failed: lpfx=invalid_state(5)");
  validation.clear(executor);
  validation.capture(executor);
  Traits::fail_next_event_query = true;
  require_message([&] { (void)validation.ready("unexpected numerical failure"); },
                  "TPIC device validation: completion query failed: lpfx=backend_failure(6) status=failure");
  Traits::fail_next_event_synchronization = true;
  require_message([&] { validation.host_wait_check("numerical failure"); },
                  "TPIC device validation: completion failed: lpfx=backend_failure(6) status=failure");
  validation.host_wait_check("unexpected numerical failure");
}
} // namespace

int main() {
  try {
    application_policy();
    backend_error_messages();
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
