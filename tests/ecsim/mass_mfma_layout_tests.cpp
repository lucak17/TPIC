#include <tpic/ecsim/mass_matrix/mass_matrix_mfma_layout.hpp>

#include <iostream>
#include <stdexcept>

namespace {
void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}
}

int main() {
  using namespace tpic::ecsim;
  using Map = MassMfma4Layout;
  try {
    for (const auto arch : {"gfx90a", "gfx940", "gfx941", "gfx942", "gfx950",
                            "gfx90a:xnack-", "gfx942:sramecc+:xnack-"}) {
      require(mass_mfma4_supported_architecture(arch, 64), "supported MFMA architecture rejected");
      require(!mass_mfma4_supported_architecture(arch, 32), "MFMA admitted wave32");
    }
    for (const auto arch : {"", "gfx908", "gfx900", "gfx1100", "gfx1200", "gfx90a0", "gfx942suffix"})
      require(!mass_mfma4_supported_architecture(arch, 64), "unsupported MFMA architecture admitted");

    // Emulate the native four-block instruction's lane layout independently
    // of TPIC's helpers. Distinct signed coefficients, corners, and particles
    // detect quadrant swaps, a wrong B shuffle, and component transposition.
    constexpr auto stencils = make_mass_lane_stencil_map<Map>();
    for (unsigned count : {0U, 1U, 2U, 3U, 4U, 5U, 31U, 32U, 33U, 63U, 64U,
                           65U, 127U, 128U, 129U, 257U}) {
      double actual[64][9]{};
      double expected[8][8][9]{};
      auto weight = [](unsigned p, int corner) { return double(1 + (p * 7 + corner * 3) % 19); };
      auto alpha = [](unsigned p, int c) { return double(int((p * 3 + c * 11) % 23) - 11); };
      for (unsigned p = 0; p < count; ++p)
        for (int a = 0; a < 8; ++a)
          for (int b = 0; b < 8; ++b)
            for (int c = 0; c < 9; ++c)
              expected[a][b][c] += (weight(p, a) * alpha(p, c)) * weight(p, b);
      for (unsigned chunk = 0; chunk < count; chunk += 64) {
        for (unsigned sub = 0; sub < 64 && chunk + sub < count; sub += 4) {
          double weights[64]{};
          for (unsigned lane = 0; lane < 64; ++lane) {
            const unsigned p = chunk + sub + Map::particle(lane);
            if (p < count) weights[lane] = weight(p, Map::input_corner(lane));
          }
          for (int c = 0; c < 9; ++c) {
            double a[64]{}, b[64]{};
            for (unsigned lane = 0; lane < 64; ++lane) {
              const unsigned p = chunk + sub + Map::particle(lane);
              a[lane] = weights[lane] * (p < count ? alpha(p, c) : 0.0);
              b[lane] = weights[Map::partner_source(lane)];
            }
            // ISA: each block q consumes A[row,k], B[k,col] from
            // lanes 16*k+4*q+row/col and emits row at lane/16.
            for (unsigned lane = 0; lane < 64; ++lane) {
              const unsigned q = (lane / 4) % 4, row = lane / 16, col = lane % 4;
              for (unsigned k = 0; k < 4; ++k)
                actual[lane][c] += a[16*k + 4*q + row] * b[16*k + 4*q + col];
            }
          }
        }
      }
      bool seen[8][8]{};
      unsigned destinations = 0;
      for (unsigned lane = 0; lane < 64; ++lane) {
        const int a = Map::output_a(lane), b = Map::output_b(lane, 0);
        require(!seen[a][b], "MFMA output pair repeated");
        seen[a][b] = true;
        require(stencils.value[lane][0] == mass_pair_stencil(a, b), "MFMA stencil mismatch");
        destinations += stencils.value[lane][0] >= 0;
        for (int c = 0; c < 9; ++c)
          require(actual[lane][c] == expected[a][b][c], "MFMA quadrant product mismatch");
      }
      require(destinations == 36, "MFMA canonical destination count mismatch");
    }
    std::cout << "TPIC MFMA4 lane and architecture tests passed\n";
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
