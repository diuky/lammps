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

#ifndef LMP_PAIR_LPM_H
#define LMP_PAIR_LPM_H

#include "pair.h"
#include <cmath>

namespace LAMMPS_NS {

class PairLpm : public Pair {
 public:
  PairLpm(class LAMMPS *);
  ~PairLpm() override;

  int pack_forward_comm(int, int *, double *, int, int *) override;
  void unpack_forward_comm(int, int, double *) override;

  //int pack_reverse_comm(int, int, double *) override;
  //void unpack_reverse_comm(int, int *, double *) override;

  void compute_nonlocal(int, int);
  void *extract(const char *, int &) override;
  void init_style() override;
  void settings(int, char **) override;
  void ev_tally(int, int, int, int, double, double, double, double, double, double) override;

  double memory_usage() override;
  int num_ind;
  double lattice_a, lattice_c, ca_ratio, m1, m2, m3, m4, m5, m6, m7, m8, m9, m10, m11, m12, m13;
  double *Cijkl, *Knkl, *sum_dr1, *sum_dr2, *sum_dr3;
  double **cut;


 protected:
  class FixLpmNeigh *fix_lpm_neigh;


  int nmax;

 protected:
  void allocate();
};

}    // namespace LAMMPS_NS

#endif
