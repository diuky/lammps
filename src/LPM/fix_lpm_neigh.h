/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifdef FIX_CLASS
// clang-format off
FixStyle(LPM_NEIGH,FixLpmNeigh);
// clang-format on
#else

#ifndef LMP_FIX_LPM_NEIGH_H
#define LMP_FIX_LPM_NEIGH_H

#include "fix.h"

namespace LAMMPS_NS {

class FixLpmNeigh : public Fix {
  friend class PairLpm;
  friend class PairLpmLE;
  friend class PairLpmLEOMP;

 public:
  FixLpmNeigh(class LAMMPS *, int, char **);
  ~FixLpmNeigh() override;
  int setmask() override;
  void init() override;
  void init_list(int, class NeighList *) override;
  void setup(int) override;
  void min_setup(int) override;

  double memory_usage() override;
  void grow_arrays(int) override;
  void copy_arrays(int, int, int) override;
  int pack_exchange(int, double *) override;
  int unpack_exchange(int, double *) override;
  void write_restart(FILE *) override;
  void restart(char *) override;
  int pack_restart(int, double *) override;
  void unpack_restart(int, int) override;
  int size_restart(int) override;
  int maxsize_restart() override;
  //int pack_forward_comm(int, int *, double *, int, int *) override;
  //void unpack_forward_comm(int, int, double *) override;

 protected:
  int first;                            // flag for first time initialization
  int maxnneigh_1;                        // # of max num of neighbor for specified type of atom
  int maxnneigh;                         // # of max num of neighbor for each atom
  int *nneighbor1;                       // # of neighbors for 1st type atom
  int *nneighbor2;                       // # of neighbors for 2nd type atom
  int *nneighbor3;                       // # of neighbors for 3rd type atom

  tagint **fullneigh1;                     // full list of nearest neighs for 1st type, stored as global IDs
  tagint **fullneigh2;                     // full list of nearest neighs for 2nd type, stored as global IDs
  tagint **fullneigh3;                     // full list of nearest neighs for 3rd type, stored as global IDs
  double **r0_1;                       // initial distance to nearest neighbor of 1st type
  double **r0_2;                       // initial distance to nearest neighbor of 2nd type
  double **r0_3;                       // initial distance to nearest neighbor of 3rd type
  double **r1_1;                       // instanteneous distance to nearest neighbor of 1st type
  double **r1_2;                       // instanteneous distance to nearest neighbor of 2nd type
  double **r1_3;                       // instanteneous distance to nearest neighbor of 3rd type
  //double *sum_dr1;                     //nonlocal sum term for 1st nearest neighbor
  //double *sum_dr2;                     //nonlocal sum term for 2nd nearest neighbor
  //double *sum_dr3;                     //nonlocal sum term for 3rd nearest neighbor

  class NeighList *list;
};

}    // namespace LAMMPS_NS

#endif
#endif
