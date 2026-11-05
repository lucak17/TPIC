#include <lpfx/backend/launch.hpp>
#include <tpic/config/domain.hpp>
#include <tpic/config/simulation_parameters.hpp>
#include <tpic/communication/mass_halo/mass_halo_plan.hpp>
#include <tpic/config/precision.hpp>
#include <tpic/ecsim/mass_matrix/mass_matrix_stencil.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {

namespace vc = tpic::communication;
using Precision = tpic::config::Precision<tpic::config::StorageDouble, tpic::config::ComputeDouble>;
using Scalar = typename Precision::Storage::Mass;
using Plan = vc::MassHaloPlan<Precision>;
constexpr std::size_t tensor = tpic::data::mass_tensor_entries;
using Range = std::pair<std::size_t, std::size_t>; // offset and size in bytes

void require(const bool condition, const char *const message) {
  if (!condition) throw std::runtime_error(message);
}

// The ranges cover [begin, end) without gap or overlap.
[[nodiscard]] bool tiles(std::vector<Range> ranges, std::size_t begin, const std::size_t end) {
  std::sort(ranges.begin(), ranges.end());
  for (const auto &range : ranges) {
    if (range.first != begin) return false;
    begin += range.second;
  }
  return begin == end;
}

[[nodiscard]] Range bytes(const vc::MassHaloSide &side) {
  return {side.offset * sizeof(Scalar), side.entries * sizeof(Scalar)};
}

[[nodiscard]] tpic::config::PreparedDomain domain_3d(const int rank, const std::array<int, 3> ranks,
                                                     const std::array<bool, 3> periodic,
                                                     const std::array<int, 3> cells) {
  tpic::config::TopologyConfig topology{};
  topology.xlen = ranks[0];
  topology.ylen = ranks[1];
  topology.zlen = ranks[2];
  topology.periodic_x = periodic[0];
  topology.periodic_y = periodic[1];
  topology.periodic_z = periodic[2];
  const tpic::config::GridConfig box{double(cells[0]), double(cells[1]), double(cells[2]),
                                     cells[0], cells[1], cells[2]};
  return tpic::config::prepare_domain(box, topology, rank, ranks[0] * ranks[1] * ranks[2]);
}

// A side carries the nodes of its region times the nine entries of each
// stencil of its mask, listed in ascending order.
void check_side(const vc::MassHaloDirection &direction, const vc::MassHaloSide &side, const std::uint16_t mask) {
  require(side.stencil_mask == mask, "mass side stencils differ from the phase rule");
  std::size_t kept = 0;
  for (int stencil = 0; stencil != tpic::data::mass_stencil_count; ++stencil)
    if (mask >> stencil & 1U)
      require(side.stencils[kept++] == stencil, "mass side stencil list is not its mask in order");
  const std::size_t nodes = std::size_t(direction.lengths[0]) * std::size_t(direction.lengths[1]) *
                            std::size_t(direction.lengths[2]);
  require(side.entries == nodes * kept * tensor, "mass side entries are not its region's kept records");
}

// Rule, grids and images of one phase: the sides tile their kernel grids in
// ordinal order; remote sends and receives tile the message plan's images,
// and the self snapshots follow the remote sends.
template <bool Additive> void check_phase(const Plan &plan) {
  const auto &phase = plan.template phase<Additive>();
  const auto directions = phase.directions();
  std::vector<Range> remote_sends, self_sends, receives;
  std::uint32_t pack = 0, unpack = 0;
  for (std::size_t ordinal = 0; ordinal != directions.size(); ++ordinal) {
    const auto &direction = directions[ordinal];
    const auto vector = lpfx::canonical_cartesian_neighbor_directions[direction.canonical_index];
    check_side(direction, direction.send, vc::mass_halo_stencil_mask(Additive, vector));
    check_side(direction, direction.receive,
               vc::mass_halo_stencil_mask(Additive, lpfx::opposite_cartesian_direction(vector)));
    require(direction.send.first_block == pack &&
                direction.send.block_count == vc::MassHaloLaunch::blocks(direction.send.entries) &&
                direction.receive.first_block == unpack &&
                direction.receive.block_count == vc::MassHaloLaunch::blocks(direction.receive.entries),
            "mass halo side blocks are not the prefix of their block counts");
    pack += direction.send.block_count;
    unpack += direction.receive.block_count;
    if (ordinal != 0)
      require(direction.canonical_index > directions[ordinal - 1].canonical_index,
              "mass numerical directions are not in canonical order");
    const auto send = bytes(direction.send);
    if (direction.self) {
      const auto opposite = std::find_if(directions.begin(), directions.end(), [&](const auto &candidate) {
        return candidate.canonical_index == lpfx::opposite_cartesian_neighbor_index(direction.canonical_index);
      });
      require(opposite != directions.end() && opposite->self && direction.receive.offset == opposite->send.offset &&
                  direction.receive.entries == opposite->send.entries,
              "self direction does not receive its opposite snapshot");
      self_sends.push_back(send);
    } else {
      if (send.second) remote_sends.push_back(send);
      if (direction.receive.entries) receives.push_back(bytes(direction.receive));
    }
  }
  require(pack == phase.pack_blocks() && unpack == phase.unpack_blocks(),
          "mass halo grids are not the sums of the side block counts");
  const auto &messages = phase.message_plan();
  require(tiles(remote_sends, 0, messages.outgoing_bytes()) &&
              tiles(self_sends, messages.outgoing_bytes(), phase.outgoing_storage_bytes()) &&
              tiles(receives, 0, messages.incoming_bytes()),
          "mass halo images are not tiled by their sides");
}

void check_plan(const Plan &plan) {
  check_phase<true>(plan);
  check_phase<false>(plan);
  const auto add = plan.template phase<true>().directions(), ghost = plan.template phase<false>().directions();
  require(std::equal(add.begin(), add.end(), ghost.begin(), ghost.end(),
                     [](const auto &left, const auto &right) {
                       return left.canonical_index == right.canonical_index && left.self == right.self;
                     }),
          "the two mass phases disagree on their directions");
}

// The additive masks hold exactly the stencils a rank deposits at a shared
// node, from the deposition's corner pairs of its own cells; the ghost masks
// exactly those the apply reads from a ghost record as a backward neighbour.
void stencil_rules() {
  constexpr int n = 3;
  for (std::uint32_t canonical = 0; canonical != vc::mass_halo_neighbour_count; ++canonical) {
    const auto delta = lpfx::canonical_cartesian_neighbor_directions[canonical];
    const int toward[]{delta.x, delta.y, delta.z};
    int shared[3], ghost[3];
    for (int axis = 0; axis != 3; ++axis) {
      shared[axis] = toward[axis] > 0 ? n - 1 : toward[axis] < 0 ? 0 : 1;
      ghost[axis] = toward[axis] > 0 ? n : toward[axis] < 0 ? -1 : 1;
    }
    std::uint16_t deposited = 0, read = 0;
    for (int k = 0; k + 1 < n; ++k)
      for (int j = 0; j + 1 < n; ++j)
        for (int i = 0; i + 1 < n; ++i)
          for (int a = 0; a != tpic::ecsim::cic_corner_count; ++a)
            for (int b = 0; b != tpic::ecsim::cic_corner_count; ++b) {
              const int stencil = tpic::ecsim::mass_pair_stencil(a, b);
              const auto corner = tpic::ecsim::cic_corner({}, a);
              if (stencil >= 0 && i + corner.i == shared[0] && j + corner.j == shared[1] &&
                  k + corner.k == shared[2])
                deposited = std::uint16_t(deposited | 1U << stencil);
            }
    for (int k = 0; k != n; ++k)
      for (int j = 0; j != n; ++j)
        for (int i = 0; i != n; ++i)
          for (int stencil = 1; stencil != tpic::data::mass_stencil_count; ++stencil) {
            const auto offset = tpic::data::mass_stencil_offset(stencil);
            if (i - offset.i == ghost[0] && j - offset.j == ghost[1] && k - offset.k == ghost[2])
              read = std::uint16_t(read | 1U << stencil);
          }
    require(deposited == vc::mass_halo_stencil_mask(true, delta),
            "additive mass stencils differ from the deposited pairs");
    // The ghost on the delta side is filled by the neighbour's opposite send.
    require(read == vc::mass_halo_stencil_mask(false, lpfx::opposite_cartesian_direction(delta)),
            "ghost mass stencils differ from the apply's ghost reads");
  }
}

void one_rank() {
  const auto physical = Plan::prepare(domain_3d(0, {1, 1, 1}, {false, false, false}, {3, 3, 2}));
  check_plan(physical);
  require(physical.device_bytes() == 0, "physical one-rank mass plan contains halo work");
  const auto periodic = Plan::prepare(domain_3d(0, {1, 1, 1}, {true, true, true}, {3, 3, 2}));
  check_plan(periodic);
  const auto directions = periodic.template phase<true>().directions();
  require(directions.size() == vc::mass_halo_neighbour_count &&
              std::all_of(directions.begin(), directions.end(), [](const auto &d) { return d.self; }) &&
              periodic.incoming_bytes() == 0 &&
              periodic.template phase<true>().message_plan().peer_messages().empty() &&
              periodic.template phase<false>().message_plan().peer_messages().empty(),
          "periodic one-rank mass plan has remote messages");
}

// Every remote side matches the opposite direction of its peer: the peer
// sends what this rank receives and receives what it sends. A phase sends one
// contiguous message to each peer and receives one from it, tiled exactly by
// the packed sides of the directions to that peer.
template <bool Additive>
void check_peers(const std::vector<tpic::config::PreparedDomain> &domains, const std::vector<Plan> &plans) {
  for (std::size_t rank = 0; rank != plans.size(); ++rank) {
    const auto &topology = domains[rank].field_neighborhood.topology();
    const auto directions = plans[rank].template phase<Additive>().directions();
    const auto peer_of = [&](const auto &direction) {
      return std::size_t(topology.neighbor_by_index(direction.canonical_index).peer_rank());
    };
    for (const auto &direction : directions) {
      if (direction.self) continue;
      const auto remote = plans[peer_of(direction)].template phase<Additive>().directions();
      const auto reciprocal = std::find_if(remote.begin(), remote.end(), [&](const auto &candidate) {
        return candidate.canonical_index == lpfx::opposite_cartesian_neighbor_index(direction.canonical_index);
      });
      require(reciprocal != remote.end() && reciprocal->send.entries == direction.receive.entries &&
                  reciprocal->receive.entries == direction.send.entries,
              "mass sides are not reciprocal between neighbours");
    }
    for (const auto &peer : plans[rank].template phase<Additive>().message_plan().peer_messages()) {
      std::vector<Range> sends, receives;
      for (const auto &direction : directions)
        if (!direction.self && peer_of(direction) == peer.peer) {
          if (direction.send.entries) sends.push_back(bytes(direction.send));
          if (direction.receive.entries) receives.push_back(bytes(direction.receive));
        }
      require(tiles(sends, peer.outgoing.offset_bytes, peer.outgoing.offset_bytes + peer.outgoing.size_bytes) &&
                  tiles(receives, peer.incoming.offset_bytes, peer.incoming.offset_bytes + peer.incoming.size_bytes),
              "a peer message is not the contiguous packed sides of its directions");
      const auto remote = plans[peer.peer].template phase<Additive>().message_plan().peer_messages();
      const auto reciprocal = std::find_if(remote.begin(), remote.end(),
          [&](const auto &candidate) { return candidate.peer == rank; });
      require(reciprocal != remote.end() &&
                  reciprocal->outgoing.size_bytes == peer.incoming.size_bytes &&
                  reciprocal->incoming.size_bytes == peer.outgoing.size_bytes,
              "mass peer envelopes are not reciprocal");
    }
  }
}

// The 2x1x1 fixture coalesces its 18 remote directions into one peer; the
// uneven 3x1x1 one has one direction per peer.
void x_decompositions() {
  for (const int count : {2, 3}) {
    std::vector<tpic::config::PreparedDomain> domains;
    std::vector<Plan> plans;
    for (int rank = 0; rank != count; ++rank) {
      domains.push_back(domain_3d(rank, {count, 1, 1}, {true, count == 2, count == 2}, {2 * count + 1, 3, 2}));
      plans.push_back(Plan::prepare(domains.back()));
      check_plan(plans.back());
    }
    check_peers<true>(domains, plans);
    check_peers<false>(domains, plans);
    for (std::uint32_t rank = 0; rank != plans.size(); ++rank) {
      const auto check = [&](const auto &phase) {
        const auto directions = phase.directions();
        const auto peers = phase.message_plan().peer_messages();
        const auto remote = std::count_if(directions.begin(), directions.end(), [](const auto &d) { return !d.self; });
        require(count == 2 ? peers.size() == 1 && remote == 18
                           : peers.size() == 2 && remote == 2 && directions[0].canonical_index == 12 &&
                                 directions[1].canonical_index == 13,
                "x-decomposed mass plan lost its peer topology");
      };
      check(plans[rank].template phase<true>());
      check(plans[rank].template phase<false>());
    }
    // Seven x cells cannot divide evenly by 3.
    require(count == 2 || plans[0].layout().nx != plans[2].layout().nx,
            "three-rank mass fixture is not an uneven x partition");
  }
}

// Every direction kind, on every rank: 2x2x2, 3x2x2 and 1x3x4 periodic, walls
// and a mix, uneven splits; the union over the ranks covers all 26 directions.
void three_dimensional() {
  struct Case {
    std::array<int, 3> ranks;
    std::array<bool, 3> periodic;
    std::array<int, 3> cells;
  };
  for (const Case c : {Case{{2, 2, 2}, {true, true, true}, {6, 4, 4}},
                       Case{{2, 2, 2}, {false, false, false}, {5, 4, 5}},
                       Case{{3, 2, 2}, {true, true, true}, {7, 5, 4}},
                       Case{{3, 2, 2}, {true, false, true}, {7, 4, 5}},
                       Case{{1, 3, 4}, {true, true, true}, {3, 7, 9}}}) {
    std::vector<tpic::config::PreparedDomain> domains;
    std::vector<Plan> plans;
    std::array<bool, vc::mass_halo_neighbour_count> covered{};
    const int count = c.ranks[0] * c.ranks[1] * c.ranks[2];
    for (int rank = 0; rank != count; ++rank) {
      domains.push_back(domain_3d(rank, c.ranks, c.periodic, c.cells));
      plans.push_back(Plan::prepare(domains.back()));
      check_plan(plans.back());
      for (const auto &direction : plans.back().template phase<true>().directions())
        covered[direction.canonical_index] = true;
    }
    require(std::all_of(covered.begin(), covered.end(), [](bool seen) { return seen; }),
            "three-dimensional mass fixture misses a direction");
    check_peers<true>(domains, plans);
    check_peers<false>(domains, plans);
  }
}

void maximum_grid_stride_boundary() {
  // The additive -x send keeps 13 of the 14 stencils.
  constexpr std::uint64_t nodes = 6058;
  constexpr std::uint64_t expected_entries = 13 * tensor * nodes * nodes;
  constexpr std::uint64_t maximum_grid_stride = 128ULL * 65535ULL;
  static_assert(expected_entries <=
                std::numeric_limits<std::uint32_t>::max());
  static_assert(expected_entries >
                std::numeric_limits<std::uint32_t>::max() -
                    maximum_grid_stride);

  const auto plan = Plan::prepare(domain_3d(0, {1, 1, 1}, {true, false, false}, {1, 6057, 6057}));
  const auto &phase = plan.template phase<true>();
  require(phase.directions().size() == 2 &&
              phase.pack_blocks() == 2 * lpfx::device::kernel_grid_block_limit &&
              phase.unpack_blocks() == 2 * lpfx::device::kernel_grid_block_limit,
          "near-uint32 mass direction was rejected or truncated");
  const auto &minus = phase.directions()[0], &plus = phase.directions()[1];
  require(minus.self && plus.self && minus.send.entries == expected_entries &&
              plus.receive.entries == expected_entries &&
              minus.send.block_count == lpfx::device::kernel_grid_block_limit &&
              plus.send.first_block == lpfx::device::kernel_grid_block_limit,
          "near-uint32 mass direction metadata changed");
}

} // namespace

int main() {
  try {
    stencil_rules();
    one_rank();
    x_decompositions();
    three_dimensional();
    maximum_grid_stride_boundary();
    std::cout << "TPIC compact mass halo plans passed\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
