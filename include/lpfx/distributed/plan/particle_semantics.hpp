#ifndef LPFX_DISTRIBUTED_PLAN_PARTICLE_SEMANTICS_HPP
#define LPFX_DISTRIBUTED_PLAN_PARTICLE_SEMANTICS_HPP

#include <concepts>
#include <type_traits>
#include <utility>

#include <lpfx/core/attributes.hpp>

namespace lpfx::distributed {

enum class ParticleClassificationPipeline { positions, user_flags };

namespace detail {

struct NoInitialParticleClassifier final {};
template <class Semantics>
inline constexpr ParticleClassificationPipeline particle_classification_pipeline_v = [] {
  if constexpr (requires { Semantics::classification_pipeline; })
    return Semantics::classification_pipeline;
  else
    return ParticleClassificationPipeline::positions;
}();

template <class T>
concept ParticleSemanticValue = std::is_object_v<T> && (!std::is_array_v<T>) &&
                                std::same_as<T, std::remove_cv_t<T>> &&
                                std::is_nothrow_move_constructible_v<T> &&
                                std::is_nothrow_destructible_v<T>;

} // namespace detail

// Concrete application policies, so host loops and device kernels see the
// exact classifier and codec-context types. Pointer members inside a policy
// are borrowed and must outlive the prepared Communication.
template <detail::ParticleSemanticValue InitialOwner,
          detail::ParticleSemanticValue ReceivedOwner,
          detail::ParticleSemanticValue CodecContext>
class ParticleSemantics final {
public:
  static constexpr auto classification_pipeline =
      std::same_as<InitialOwner, detail::NoInitialParticleClassifier>
          ? ParticleClassificationPipeline::user_flags
          : ParticleClassificationPipeline::positions;
  using initial_owner_type = InitialOwner;
  using received_owner_type = ReceivedOwner;
  using codec_context_type = CodecContext;

  constexpr ParticleSemantics(InitialOwner initial_owner,
                              ReceivedOwner received_owner,
                              CodecContext codec_context) noexcept
      : initial_owner_(std::move(initial_owner)),
        received_owner_(std::move(received_owner)),
        codec_context_(std::move(codec_context)) {}

  [[nodiscard]] LPFX_HOST_DEVICE constexpr const InitialOwner &
  initial_owner() const noexcept {
    return initial_owner_;
  }

  [[nodiscard]] LPFX_HOST_DEVICE constexpr const ReceivedOwner &
  received_owner() const noexcept {
    return received_owner_;
  }

  [[nodiscard]] LPFX_HOST_DEVICE constexpr const CodecContext &
  codec_context() const noexcept {
    return codec_context_;
  }

  // The device projection. A bundle needing a distinct device representation
  // provides the same const accessors and its own device_view().
  [[nodiscard]] constexpr ParticleSemantics device_view() const noexcept
    requires(std::is_trivially_copyable_v<InitialOwner> &&
             std::is_trivially_copyable_v<ReceivedOwner> &&
             std::is_trivially_copyable_v<CodecContext> &&
             std::is_nothrow_copy_constructible_v<InitialOwner> &&
             std::is_nothrow_copy_constructible_v<ReceivedOwner> &&
             std::is_nothrow_copy_constructible_v<CodecContext>)
  {
    return *this;
  }

private:
  [[no_unique_address]] InitialOwner initial_owner_;
  [[no_unique_address]] ReceivedOwner received_owner_;
  [[no_unique_address]] CodecContext codec_context_;
};

namespace detail {

// A nothrow-movable semantic bundle; only device bindings need device_view().
template <class T>
concept ParticleSemanticsType =
    ParticleSemanticValue<T> && requires(const T &value) {
      typename T::initial_owner_type;
      typename T::received_owner_type;
      typename T::codec_context_type;
      {
        value.initial_owner()
      } noexcept -> std::same_as<const typename T::initial_owner_type &>;
      {
        value.received_owner()
      } noexcept -> std::same_as<const typename T::received_owner_type &>;
      {
        value.codec_context()
      } noexcept -> std::same_as<const typename T::codec_context_type &>;
    };

} // namespace detail

template <class InitialOwner, class ReceivedOwner, class CodecContext>
  requires detail::ParticleSemanticValue<std::decay_t<InitialOwner>> &&
           detail::ParticleSemanticValue<std::decay_t<ReceivedOwner>> &&
           detail::ParticleSemanticValue<std::decay_t<CodecContext>> &&
           std::is_nothrow_constructible_v<std::decay_t<InitialOwner>,
                                           InitialOwner &&> &&
           std::is_nothrow_constructible_v<std::decay_t<ReceivedOwner>,
                                           ReceivedOwner &&> &&
           std::is_nothrow_constructible_v<std::decay_t<CodecContext>,
                                           CodecContext &&>
[[nodiscard]] constexpr auto
particle_semantics(InitialOwner &&initial_owner, ReceivedOwner &&received_owner,
                   CodecContext &&codec_context) noexcept {
  using Semantics =
      ParticleSemantics<std::decay_t<InitialOwner>, std::decay_t<ReceivedOwner>,
                        std::decay_t<CodecContext>>;
  return Semantics{std::forward<InitialOwner>(initial_owner),
                   std::forward<ReceivedOwner>(received_owner),
                   std::forward<CodecContext>(codec_context)};
}

// For user-produced flags: migration needs only the received-record
// classifier and the codec context.
template <class ReceivedOwner, class CodecContext>
[[nodiscard]] constexpr auto particle_semantics(
    ReceivedOwner &&received_owner, CodecContext &&codec_context) noexcept {
  return particle_semantics(detail::NoInitialParticleClassifier{},
      std::forward<ReceivedOwner>(received_owner),
      std::forward<CodecContext>(codec_context));
}

} // namespace lpfx::distributed

#endif
