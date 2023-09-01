/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#include "pair_lpm.h"

#include "atom.h"
#include "domain.h"
#include "error.h"
#include "fix_lpm_neigh.h"
#include "lattice.h"
#include "memory.h"
#include "modify.h"
#include "neighbor.h"

#include <cstring>

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

PairLpm::PairLpm(LAMMPS *_lmp) :
    Pair(_lmp), fix_lpm_neigh(nullptr), Cijkl(nullptr), Knkl(nullptr),sum_dr1(nullptr),sum_dr2(nullptr),sum_dr3(nullptr), cut(nullptr)
{
  fprintf(screen,"Constructor starts\n");
  for (int i = 0; i < 6; i++) virial[i] = 0.0;
  no_virial_fdotr_compute = 1;
  single_enable = 0;
  nmax = -1;
  fprintf(screen,"Constructor ends\n");
}

/* ---------------------------------------------------------------------- */

PairLpm::~PairLpm()
{
  if (fix_lpm_neigh) modify->delete_fix(fix_lpm_neigh->id);
  if (allocated) {

    memory->destroy(Cijkl);
    memory->destroy(Knkl);
    memory->destroy(cut);
    memory->destroy(cutsq);
    memory->destroy(setflag);
    memory->destroy(sum_dr1);
    memory->destroy(sum_dr2);
    memory->destroy(sum_dr3);
  }
}

/* ----------------------------------------------------------------------
   allocate all arrays
------------------------------------------------------------------------- */

void PairLpm::allocate()
{
  fprintf(screen,"Memory allocation starts\n");
  allocated = 1;
  int n = num_ind;
  int nlocal = atom->nlocal;
  int ntype = atom->ntypes + 1;
  fprintf(screen, "num_ind: %ld\n", n);
  memory->create(setflag, n, n, "pair:setflag");
  for (int i = 0; i < ntype; i++)
    for (int j = 0; j < ntype; j++) setflag[i][j] = 0;

  memory->create(Cijkl, n, "pair:Cijkl");
  memory->create(Knkl, n, "pair:Knkl");
  memory->create(cutsq, ntype, ntype, "pair:cutsq");
  memory->create(cut, ntype, ntype, "pair:cut");
  memory->create(sum_dr1, nlocal, "pair:sum_dr1");
  memory->create(sum_dr2, nlocal, "pair:sum_dr2");
  memory->create(sum_dr3, nlocal, "pair:sum_dr3");
  fprintf(screen,"Memory allocation ends\n");
}

/* ----------------------------------------------------------------------
   memory usage of local arrays
------------------------------------------------------------------------- */

double PairLpm::memory_usage()
{
  fprintf(screen,"Assign memeory starts\n");
  int nlocal = atom->nlocal;
  int ntype = atom->ntypes + 1;
  double bytes = 2.0 * (double) num_ind * sizeof(double);
  bytes += 2.0 * (double) ntype*ntype*sizeof(double);
  bytes += 3.0 * (double) nlocal * sizeof(double);
  return bytes;
  fprintf(screen,"Assign memeory ends\n");
}

/* ----------------------------------------------------------------------
   global settings
------------------------------------------------------------------------- */

void PairLpm::settings(int narg, char ** /*arg*/)
{
  if (narg) error->all(FLERR, "Illegal pair_style command");
}

/* ----------------------------------------------------------------------
   init common to all lpm pair styles
------------------------------------------------------------------------- */

void PairLpm::init_style()
{
  // error checks
  if (!atom->lpm_flag) error->all(FLERR, "Pair style lpm requires atom style lpm");
  if (atom->map_style == Atom::MAP_NONE)
    error->all(FLERR, "Pair lpm requires an atom map, see atom_modify");

  if (domain->lattice == nullptr) error->all(FLERR, "Pair lpm requires a lattice be defined");

    if (!fix_lpm_neigh)
      fix_lpm_neigh = dynamic_cast<FixLpmNeigh *>(modify->add_fix("LPM_NEIGH all LPM_NEIGH"));

  neighbor->add_request(this);
}

void PairLpm::compute_nonlocal(int ifrom, int ito)
{
  int i, j, jj, jnum, itype, jtype;
  double xtmp, ytmp, ztmp, delx, dely, delz;
  double xtmp0, ytmp0, ztmp0, delx0, dely0, delz0;
  double rsq, r, dr, rsq0;
  double delta;

  double **x = atom->x;
  double **x0 = atom->x0;
  int *nneighbor1 = fix_lpm_neigh->nneighbor1;
  int *nneighbor2 = fix_lpm_neigh->nneighbor2;
  int *nneighbor3 = fix_lpm_neigh->nneighbor3;
  tagint **fullneigh1 = fix_lpm_neigh->fullneigh1;
  tagint **fullneigh2 = fix_lpm_neigh->fullneigh2;
  tagint **fullneigh3 = fix_lpm_neigh->fullneigh3;

  for (i = ifrom; i < ito; i++){
    xtmp = x[i][0];
    ytmp = x[i][1];
    ztmp = x[i][2];
    xtmp0 = x0[i][0];
    ytmp0 = x0[i][1];
    ztmp0 = x0[i][2];
    //fprintf(screen,"no problem pt1\n");
    sum_dr1[i] = 0.0;
    sum_dr2[i] = 0.0;
    sum_dr3[i] = 0.0;
    //fprintf(screen,"no problem pt2\n");
    for (jj = 0; jj < nneighbor1[i]; jj++){
      if (fullneigh1[i][jj] == 0) continue;
      j = atom->map(fullneigh1[i][jj]);
      if (j < 0) continue;
      delx = xtmp - x[j][0];
      dely = ytmp - x[j][1];
      delz = ztmp - x[j][2];
      delx0 = xtmp0 - x0[j][0];
      dely0 = ytmp0 - x0[j][1];
      delz0 = ztmp0 - x0[j][2];
      rsq = sqrt(delx*delx+dely*dely+delz*delz);
      rsq0 = sqrt(delx0*delx0+dely0*dely0+delz0*delz0);
      dr = rsq - rsq0;
      sum_dr1[i] += dr;
    }
    for (jj = 0; jj < nneighbor2[i]; jj++){
      if (fullneigh2[i][jj] == 0) continue;
      j = atom->map(fullneigh2[i][jj]);
      if (j < 0) continue;
      delx = xtmp - x[j][0];
      dely = ytmp - x[j][1];
      delz = ztmp - x[j][2];
      delx0 = xtmp0 - x0[j][0];
      dely0 = ytmp0 - x0[j][1];
      delz0 = ztmp0 - x0[j][2];
      rsq = sqrt(delx*delx+dely*dely+delz*delz);
      rsq0 = sqrt(delx0*delx0+dely0*dely0+delz0*delz0);
      dr = rsq - rsq0;
      sum_dr2[i] += dr;
    }
    for (jj = 0; jj < nneighbor3[i]; jj++){
      if (fullneigh3[i][jj] == 0) continue;
      j = atom->map(fullneigh3[i][jj]);
      if (j < 0) continue;
      j = fullneigh3[i][jj];
      delx = xtmp - x[j][0];
      dely = ytmp - x[j][1];
      delz = ztmp - x[j][2];
      delx0 = xtmp0 - x0[j][0];
      dely0 = ytmp0 - x0[j][1];
      delz0 = ztmp0 - x0[j][2];
      rsq = sqrt(delx*delx+dely*dely+delz*delz);
      rsq0 = sqrt(delx0*delx0+dely0*dely0+delz0*delz0);
      dr = rsq - rsq0;
      sum_dr3[i] += dr;
    }
  }
}

int PairLpm::pack_forward_comm(int n, int *list, double *buf, int /*pbc_flag*/, int * /*pbc*/){
  int i, j, m;

  m = 0;
  for (i = 0; i < n; i++) {
    j = list[i];
    buf[m] = sum_dr1[j];
    m++;
    buf[m] = sum_dr2[j];
    m++;
    buf[m] = sum_dr3[j];
    m++;
  }
  return m;
}

/* ---------------------------------------------------------------------- */

void PairLpm::unpack_forward_comm(int n, int first, double *buf)
{
  int i, m, last;

  m = 0;
  last = first + n;
  for (i = first; i < last; i++) {
     sum_dr1[i] = buf[m];
     m++;
     sum_dr2[i] = buf[m];
     m++;
     sum_dr3[i] = buf[m];
     m++;
   }
}

//int PairLpm::pack_reverse_comm(int n, int first, double *buf)
//{
  //int i,m,last;

  //m = 0;
  //last = first + n;
  //for (i = first; i < last; i++){
    //buf[m] = sum_dr1[i];
    //m++;
    //buf[m] = sum_dr2[i];
    //m++;
    //buf[m] = sum_dr3[i];
    //m++;
  //}
  //return m;
//}

/* ---------------------------------------------------------------------- */

//void PairLpm::unpack_reverse_comm(int n, int *list, double *buf)
//{
  //int i,j,m;

  //m = 0;
  //for (i = 0; i < n; i++) {
    //j = list[i];
    //sum_dr1[j] += buf[m];
    //m++;
    //sum_dr2[j] += buf[m];
    //m++;
    //sum_dr3[j] += buf[m];
    //m++;
  //}
//}




void *PairLpm::extract(const char *name, int &dim)
{
  dim = 1;
  if (strcmp(name, "sum_dr1") == 0) return (void *) sum_dr1;
  if (strcmp(name, "sum_dr2") == 0) return (void *) sum_dr2;
  if (strcmp(name, "sum_dr3") == 0) return (void *) sum_dr3;
  return nullptr;
}


void PairLpm::ev_tally(int i, int j, int nlocal, int newton_pair,
                    double evdwl, double ecoul, double fpair,
                    double delx, double dely, double delz)
                    {
                      double evdwlhalf,ecoulhalf,epairhalf,v[6];

                      if (eflag_either) {
                        if (eflag_global) {
                          if (newton_pair) {
                            eng_vdwl += evdwl;
                            eng_coul += ecoul;
                          } else {
                            evdwlhalf = 0.5*evdwl;
                            ecoulhalf = 0.5*ecoul;
                            if (i < nlocal) {
                              eng_vdwl += evdwlhalf;
                              eng_coul += ecoulhalf;
                            }
                            if (j < nlocal) {
                              eng_vdwl += evdwlhalf;
                              eng_coul += ecoulhalf;
                            }
                          }
                        }
                        if (eflag_atom) {
                          epairhalf = (evdwl + ecoul);
                          if (newton_pair || i < nlocal) eatom[i] += epairhalf;
                          //if (newton_pair || j < nlocal) eatom[j] += epairhalf;
                        }
                      }

                      if (vflag_either) {
                        v[0] = delx*delx*fpair;
                        v[1] = dely*dely*fpair;
                        v[2] = delz*delz*fpair;
                        v[3] = delx*dely*fpair;
                        v[4] = delx*delz*fpair;
                        v[5] = dely*delz*fpair;

                        if (vflag_global) {
                          if (newton_pair) {
                            virial[0] += v[0];
                            virial[1] += v[1];
                            virial[2] += v[2];
                            virial[3] += v[3];
                            virial[4] += v[4];
                            virial[5] += v[5];
                          } else {
                            if (i < nlocal) {
                              virial[0] += v[0];
                              virial[1] += v[1];
                              virial[2] += v[2];
                              virial[3] += v[3];
                              virial[4] += v[4];
                              virial[5] += v[5];
                            }
                            if (j < nlocal) {
                              virial[0] += v[0];
                              virial[1] += v[1];
                              virial[2] += v[2];
                              virial[3] += v[3];
                              virial[4] += v[4];
                              virial[5] += v[5];
                            }
                          }
                        }

                        if (vflag_atom) {
                          if (newton_pair || i < nlocal) {
                            vatom[i][0] += v[0];
                            vatom[i][1] += v[1];
                            vatom[i][2] += v[2];
                            vatom[i][3] += v[3];
                            vatom[i][4] += v[4];
                            vatom[i][5] += v[5];
                          }
                          if (newton_pair || j < nlocal) {
                            vatom[j][0] += v[0];
                            vatom[j][1] += v[1];
                            vatom[j][2] += v[2];
                            vatom[j][3] += v[3];
                            vatom[j][4] += v[4];
                            vatom[j][5] += v[5];
                          }
                        }
                      }
                    }
