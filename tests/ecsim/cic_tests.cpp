#include <lpfx/algorithm/coupling/trilinear.hpp>
#include <tpic/data/particles/particle_fractions.hpp>
#include <tpic/ecsim/particles/cic.hpp>

#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace {

void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

template <class Compute, class Position>
void check_weights(Position rx, Position ry, Position rz) {
  const Compute x = static_cast<Compute>(rx), y = static_cast<Compute>(ry),
                z = static_cast<Compute>(rz);
  const auto actual = tpic::ecsim::make_cic_weights<Compute>(rx, ry, rz);
  const auto lpfx_stencil = lpfx::make_trilinear_stencil<lpfx::Node,
      tpic::config::PreparedDomain::IndexPolicy>(x, y, z);
  const std::array<Compute, 8> expected{
      lpfx_stencil.template weight<0>(), lpfx_stencil.template weight<1>(),
      lpfx_stencil.template weight<2>(), lpfx_stencil.template weight<3>(),
      lpfx_stencil.template weight<4>(), lpfx_stencil.template weight<5>(),
      lpfx_stencil.template weight<6>(), lpfx_stencil.template weight<7>()};
  const auto polynomial = [](Compute i, Compute j, Compute k) {
    return Compute(1) + Compute(2)*i - Compute(3)*j + Compute(4)*k +
        Compute(2)*i*j - Compute(4)*j*k + Compute(8)*i*k + Compute(16)*i*j*k;
  };
  Compute sum = 0, interpolated = 0;
  for (int corner = 0; corner < 8; ++corner) {
    const Compute weight = actual.value[corner];
    require(weight == expected[corner], "canonical CIC differs from LPFx corner weights");
    // The sorted mass kernels rebuild each corner weight; it must equal the
    // stored weight bit for bit.
    require(tpic::ecsim::cic_corner_weight(corner, x, y, z) == weight,
            "per-corner CIC weight differs from make_cic_weights");
    require(std::isfinite(weight) && weight >= Compute(0) && weight <= Compute(1),
            "canonical CIC weight is not finite and bounded");
    sum += weight;
    const auto node = tpic::ecsim::cic_corner({}, corner);
    interpolated += weight * polynomial(Compute(node.i), Compute(node.j), Compute(node.k));
  }
  const Compute tolerance = Compute(64) * std::numeric_limits<Compute>::epsilon();
  require(std::abs(sum - Compute(1)) <= tolerance, "canonical CIC partition of unity failed");
  require(std::abs(interpolated - polynomial(x, y, z)) <= tolerance * Compute(32),
          "canonical CIC trilinear reproduction failed");
}

template <class Position> void check_unit_cell() {
  using tpic::data::in_unit_cell;
  const Position below_one = std::nextafter(Position(1), Position(0));
  require(in_unit_cell(Position(0), -Position(0), below_one), "canonical fractions rejected");
  require(!in_unit_cell(Position(1), Position(0), Position(0)) &&
          !in_unit_cell(Position(0), -std::numeric_limits<Position>::denorm_min(), Position(0)) &&
          !in_unit_cell(Position(0), Position(0), std::numeric_limits<Position>::quiet_NaN()),
          "noncanonical fraction accepted");
}

template <class Compute, class Position> void run() {
  check_unit_cell<Position>();
  // Include exact corner/edge coordinates, negative zero, subnormals, and the
  // largest canonical fraction. Mixed positions are promoted before weighting.
  const std::array<Position, 9> fractions{
      Position(0), -Position(0), std::numeric_limits<Position>::denorm_min(),
      std::numeric_limits<Position>::min(), Position(0.125), Position(0.5),
      Position(0.75), Position(0.987654), std::nextafter(Position(1), Position(0))};
  for (const Position x : fractions)
    for (const Position y : fractions)
      for (const Position z : fractions) check_weights<Compute>(x, y, z);
  for (unsigned sample = 0; sample < 4096; ++sample)
    check_weights<Compute>(Position((sample % 997) / 997.0),
        Position(((sample * 127) % 991) / 991.0),
        Position(((sample * 313) % 983) / 983.0));
}

} // namespace

int main() {
  try {
    run<double, double>();
    run<double, float>();
    run<float, float>();
    std::cout << "TPIC canonical CIC equivalence, bounds and polynomial checks passed\n";
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
