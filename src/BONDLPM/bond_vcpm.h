/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifdef BOND_CLASS

BondStyle(vcpm,BondVcpm)

#else

#ifndef LMP_BOND_VCPM_H
#define LMP_BOND_VCPM_H

#include "bond.h"

namespace LAMMPS_NS {

class BondVcpm : public Bond {
 public:
  BondVcpm(class LAMMPS *);
  virtual ~BondVcpm();
  virtual void compute(int, int);
  virtual void coeff(int, char **);
  double equilibrium_distance(int);
  void write_restart(FILE *);
  virtual void read_restart(FILE *);
  void write_data(FILE *);
  double single(int, double, int, int, double &);
  virtual void *extract(const char *, int &);

 protected:
   int num_ind;
  double lattice_a, lattice_c, ca_ratio, m1, m2, m3, m4, m5, m6, m7, m8, m9, m10, m11, m12, m13;
  double *Cijkl, *Knkl, *k,*r0, *t;

  virtual void allocate();
};

}

#endif
#endif

/* ERROR/WARNING messages:

E: Incorrect args for bond coefficients

Self-explanatory.  Check the input script or data file.

*/
