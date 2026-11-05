#ifndef LPFX_DISTRIBUTED_FIELD_SUBMISSION_HPP
#define LPFX_DISTRIBUTED_FIELD_SUBMISSION_HPP

#include <cstdint>

namespace lpfx::distributed {

// A host-side milestone for the accepted field generation. Submission does
// not establish transport/GPU completion or completion of field application.
// No native work describes the whole prepared exchange, not an empty batch.
enum class FieldSubmission : std::uint8_t {
  pending,
  submitted,
  no_native_work,
};

} // namespace lpfx::distributed

#endif
