#ifdef ATOM_CLASS
// clang-format off
AtomStyle(lpm,AtomVecLpm);
// clang-format on
#else

#ifndef LMP_ATOM_VEC_LPM_H
#define LMP_ATOM_VEC_LPM_H

#include "atom_vec.h"

namespace LAMMPS_NS {

class AtomVecLpm : public AtomVec {
 public:
  AtomVecLpm(class LAMMPS *);

  void grow_pointers() override;
  void create_atom_post(int) override;
  void data_atom_post(int) override;
  int property_atom(const std::string &) override;
  void pack_property_atom(int, double *, int, int) override;

 private:
  double *rmass, *s0;
  double **x0;
};

}    // namespace LAMMPS_NS

#endif
#endif
