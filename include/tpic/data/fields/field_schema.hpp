#ifndef TPIC_DATA_FIELDS_FIELD_SCHEMA_HPP
#define TPIC_DATA_FIELDS_FIELD_SCHEMA_HPP

#include <lpfx/data/field/component.hpp>

namespace tpic::data {

// All E components are nodal; Bx, By and Bz are cell centred.
template <class Precision> struct FieldSchema {
  using Ex = lpfx::field::component<"ex", typename Precision::Storage::Electric,
                                   lpfx::field::node>;
  using Ey = lpfx::field::component<"ey", typename Precision::Storage::Electric,
                                   lpfx::field::node>;
  using Ez = lpfx::field::component<"ez", typename Precision::Storage::Electric,
                                   lpfx::field::node>;
  using Bx = lpfx::field::component<"bx", typename Precision::Storage::Magnetic,
                                   lpfx::field::cell>;
  using By = lpfx::field::component<"by", typename Precision::Storage::Magnetic,
                                   lpfx::field::cell>;
  using Bz = lpfx::field::component<"bz", typename Precision::Storage::Magnetic,
                                   lpfx::field::cell>;
  using E = lpfx::field::schema<Ex, Ey, Ez>;
  using BCell = lpfx::field::schema<Bx, By, Bz>;

  using Bnx = lpfx::field::component<"bnx", typename Precision::Storage::Magnetic,
                                    lpfx::field::node>;
  using Bny = lpfx::field::component<"bny", typename Precision::Storage::Magnetic,
                                    lpfx::field::node>;
  using Bnz = lpfx::field::component<"bnz", typename Precision::Storage::Magnetic,
                                    lpfx::field::node>;
  using BNode = lpfx::field::schema<Bnx, Bny, Bnz>;

  using Jx = lpfx::field::component<"jx", typename Precision::Storage::Current,
                                   lpfx::field::node>;
  using Jy = lpfx::field::component<"jy", typename Precision::Storage::Current,
                                   lpfx::field::node>;
  using Jz = lpfx::field::component<"jz", typename Precision::Storage::Current,
                                   lpfx::field::node>;
  using Current = lpfx::field::schema<Jx, Jy, Jz>;
};

} // namespace tpic::data

#endif // TPIC_DATA_FIELDS_FIELD_SCHEMA_HPP
