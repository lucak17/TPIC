#ifndef TPIC_INITIALIZATION_PARTICLES_PARTICLE_SUPPORT_HPP
#define TPIC_INITIALIZATION_PARTICLES_PARTICLE_SUPPORT_HPP

#include <tpic/config/domain.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace tpic::initialization {

// Where a species loads: the subcells whose centre lies in the ellipsoid
// sum_a ((x_a - center_a)/radius_a)^2 <= 1 over the axes with radius_a > 0,
// or every subcell when no radius is positive.
struct ParticleSupport {
  double center[3]{}, radius[3]{};
  [[nodiscard]] bool bounded() const noexcept { return radius[0] > 0.0 || radius[1] > 0.0 || radius[2] > 0.0; }
  bool operator==(const ParticleSupport&) const = default;
};

// Calls visit(first, points) for every row of subcells (y and z fixed) holding
// points of the support, in increasing order: the rows of this rank's
// subcells, or of the whole domain's, with first the x-fastest index of the
// row's first point in that box. Only the host decides which subcells a
// support holds, from their global indices, so the points of the ranks add up
// to the whole domain's. A row cuts the ellipsoid in one segment: a
// square-root estimate of its ends, moved subcell by subcell to the exact
// predicate.
template <class Visit>
void visit_support_rows(const ParticleSupport& support, const config::PreparedDomain& domain,
                        const int (&ppc)[3], bool whole_domain, Visit&& visit) {
  std::int64_t first[3]{}, extent[3]{};
  double step[3]{};
  for (std::size_t axis = 0; axis < 3; ++axis) {
    first[axis] = whole_domain ? 0 : std::int64_t{domain.global_offset[axis]} * ppc[axis];
    extent[axis] = std::int64_t{whole_domain ? domain.global_cells[axis] : domain.local_cells[axis]} * ppc[axis];
    step[axis] = domain.spacing[axis] / ppc[axis];
  }
  // The squared scaled distance along axis at a global subcell; 0 on an unbounded axis.
  const auto term = [&](std::size_t axis, std::int64_t index) {
    if (!(support.radius[axis] > 0.0)) return 0.0;
    const double d = ((static_cast<double>(index) + 0.5) * step[axis] - support.center[axis]) / support.radius[axis];
    return d * d;
  };
  const auto clamped = [](double index, std::int64_t low, std::int64_t high) {
    return static_cast<std::int64_t>(std::clamp(index, static_cast<double>(low), static_cast<double>(high)));
  };
  const std::int64_t lowest = first[0], highest = first[0] + extent[0] - 1;
  for (std::int64_t z = 0; z < extent[2]; ++z) {
    const double tz = term(2, first[2] + z);
    for (std::int64_t y = 0; y < extent[1]; ++y) {
      const double ty = term(1, first[1] + y);
      if (ty + tz > 1.0) continue;
      const auto inside = [&](std::int64_t x) { return (term(0, x) + ty) + tz <= 1.0; };
      std::int64_t lower = lowest, upper = highest;
      if (support.radius[0] > 0.0) {
        const double half = support.radius[0] * std::sqrt(1.0 - (ty + tz));
        lower = clamped(std::ceil((support.center[0] - half) / step[0] - 0.5), lowest, highest + 1);
        upper = clamped(std::floor((support.center[0] + half) / step[0] - 0.5), lowest - 1, highest);
        while (lower > lowest && inside(lower - 1)) --lower;
        while (lower <= upper && !inside(lower)) ++lower;
        while (upper < highest && inside(upper + 1)) ++upper;
        while (upper >= lower && !inside(upper)) --upper;
        if (lower > upper) continue;
      }
      visit(lower - lowest + extent[0] * (y + extent[1] * z), upper - lower + 1);
    }
  }
}

} // namespace tpic::initialization

#endif // TPIC_INITIALIZATION_PARTICLES_PARTICLE_SUPPORT_HPP
