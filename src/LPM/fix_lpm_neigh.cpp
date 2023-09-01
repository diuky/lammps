// clang-format off
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

/* ----------------------------------------------------------------------
   Contributing authors: Mike Parks (SNL), Ezwanur Rahman, J.T. Foster (UTSA)
------------------------------------------------------------------------- */

#include "fix_lpm_neigh.h"

#include "pair_lpm_le.h"
#include "atom.h"
#include "domain.h"
#include "force.h"
#include "comm.h"
#include "neighbor.h"
#include "neigh_list.h"
#include "neigh_request.h"
#include "pair.h"
#include "lattice.h"
#include "memory.h"
#include "error.h"

#include <cmath>
#include <cstring>

using namespace LAMMPS_NS;
using namespace FixConst;

/* ---------------------------------------------------------------------- */

FixLpmNeigh::FixLpmNeigh(LAMMPS *lmp,int narg, char **arg) :
  Fix(lmp, narg, arg)
{
  restart_global = 1;
  restart_peratom = 1;
  first = 1;

  // perform initial allocation of atom-based arrays
  // register with atom class
  // set maxnneigh = 1 as placeholder

  maxnneigh_1 = 6;
  maxnneigh = 18;
  nneighbor1 = nullptr;
  nneighbor2 = nullptr;
  nneighbor3 = nullptr;
  fullneigh1 = nullptr;
  fullneigh2 = nullptr;
  fullneigh3 = nullptr;

  r0_1 = nullptr;
  r0_2 = nullptr;
  r0_3 = nullptr;

  //sum_dr1 = nullptr;
  //sum_dr2 = nullptr;
  //sum_dr3 = nullptr;

  grow_arrays(atom->nmax);
  //memset(sum_dr1, 0, atom->nmax*sizeof(double));
  //memset(sum_dr2, 0, atom->nmax*sizeof(double));
  //memset(sum_dr3, 0, atom->nmax*sizeof(double));
  atom->add_callback(Atom::GROW);
  atom->add_callback(Atom::RESTART);

  // initialize num of neighbors to 0 so atom migration is OK the 1st time

  int nlocal = atom->nlocal;
  for (int i = 0; i < nlocal; i++){
    nneighbor1[i] = 0;
    nneighbor2[i] = 0;
    nneighbor3[i] = 0;
  }

  // set comm sizes needed by this fix

  comm_forward = 1;
}

/* ---------------------------------------------------------------------- */

FixLpmNeigh::~FixLpmNeigh()
{
  // unregister this fix so atom class doesn't invoke it any more

  atom->delete_callback(id,Atom::GROW);
  atom->delete_callback(id,Atom::RESTART);

  // delete locally stored arrays

  memory->destroy(nneighbor1);
  memory->destroy(nneighbor2);
  memory->destroy(nneighbor3);
  memory->destroy(fullneigh1);
  memory->destroy(fullneigh2);
  memory->destroy(fullneigh3);
  memory->destroy(r0_1);
  memory->destroy(r0_2);
  memory->destroy(r0_3);
  //memory->destroy(sum_dr1);
  //memory->destroy(sum_dr2);
  //memory->destroy(sum_dr3);
}

/* ---------------------------------------------------------------------- */

int FixLpmNeigh::setmask()
{
  int mask = 0;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixLpmNeigh::init()
{
  if (!first) return;

  // need a full neighbor list once

  neighbor->add_request(this,NeighConst::REQ_FULL|NeighConst::REQ_OCCASIONAL);
  int nlocal = atom->nlocal;
}

/* ---------------------------------------------------------------------- */

void FixLpmNeigh::init_list(int /*id*/, NeighList *ptr)
{
  list = ptr;
}

/* ----------------------------------------------------------------------
   For minimization: setup as with dynamics
------------------------------------------------------------------------- */

void FixLpmNeigh::min_setup(int vflag)
{
  setup(vflag);
}

/* ----------------------------------------------------------------------
   create initial list of neighboring particles via call to neighbor->build()
   must be done in setup (not init) since fix init comes before neigh init
------------------------------------------------------------------------- */

void FixLpmNeigh::setup(int /*vflag*/)
{
  fprintf(screen,"Neighbor setup starts");
  int i,j,ii,jj,inum,jnum;
  double xtmp,ytmp,ztmp,delx,dely,delz,rsq;
  int *ilist,*jlist,*numneigh;
  int **firstneigh;

  double **x0 = atom->x0;
  tagint *tag = atom->tag;
  int nlocal = atom->nlocal;

 // only build list of bonds on very first run

  if (!first) return;
  first = 0;

  // build full neighbor list, will copy or build as necessary

  neighbor->build_one(list);

  inum = list->inum;
  ilist = list->ilist;
  numneigh = list->numneigh;
  firstneigh = list->firstneigh;

  Pair *anypair = force->pair_match("lpm",0);
  PairLpmLE *pairle = nullptr;
  pairle = static_cast<PairLpmLE*>(anypair);

  // scan neighbor list to set maxnneigh
  double lattice_a = pairle->lattice_a;
  double ca_ratio = pairle->ca_ratio;
  double ideal_ca_ratio = 1.63299;
  double lattice_c = lattice_a*ca_ratio;
  double first_neighbors_dist_in = lattice_a;
  double first_neighbor_dist_out = sqrt(pow(lattice_a,2)/3.+pow(lattice_c,2)/4.);
  double lattice_radius = 0.5*MIN(first_neighbors_dist_in, first_neighbor_dist_out);
  double second_neighbor_dist = sqrt(4./3.*pow(lattice_a,2)+pow(lattice_c,2)/4.);
  //fprintf(screen,"lattice_a = %f\n",lattice_a);
  //fprintf(screen,"ca_ratio = %f\n",ca_ratio);
  //fprintf(screen,"lattice_c = %f\n",lattice_c);
  //fprintf(screen,"first_neighbors_dist_in = %f\n",first_neighbors_dist_in);
  //fprintf(screen,"first_neighbor_dist_out = %f\n",first_neighbor_dist_out);

  int max_neigh = 6;
  int maxall;
  MPI_Allreduce(&max_neigh, &maxall, 1, MPI_INT, MPI_MAX, world);
  max_neigh = maxall;
  MPI_Allreduce(&max_neigh, &maxall, 1, MPI_INT, MPI_MAX, world);
  max_neigh = maxall;
  MPI_Allreduce(&max_neigh, &maxall, 1, MPI_INT, MPI_MAX, world);
  max_neigh = maxall;

  for (ii = 0; ii < inum; ii++) {
    i = ilist[ii];
    xtmp = x0[i][0];
    ytmp = x0[i][1];
    ztmp = x0[i][2];
    jlist = firstneigh[i];
    jnum = numneigh[i];
    //if (i == 0) fprintf(screen,"full neighbor list of %d\n",i);
    //sum_dr1[i] = 0;
    //sum_dr2[i] = 0;
    //sum_dr3[i] = 0;

    for (jj = 0; jj < jnum; jj++) {
      double tol = 1e-8;
      j = jlist[jj];
      j &= NEIGHMASK;

      delx = xtmp - x0[j][0];
      dely = ytmp - x0[j][1];
      delz = ztmp - x0[j][2];
      rsq = sqrt(delx*delx + dely*dely + delz*delz);
      if (ca_ratio > ideal_ca_ratio){
        if (abs(rsq - MIN(first_neighbors_dist_in, first_neighbor_dist_out)) <= tol) {
          fullneigh1[i][nneighbor1[i]] = j;
          r0_1[i][nneighbor1[i]] = rsq;
          nneighbor1[i]++;
        }else if (abs(rsq - MAX(first_neighbors_dist_in, first_neighbor_dist_out)) <= tol){
          fullneigh2[i][nneighbor2[i]] = j;
          r0_2[i][nneighbor2[i]] = rsq;
          nneighbor2[i]++;
        }else if (abs(rsq - second_neighbor_dist) <= tol){
          fullneigh3[i][nneighbor3[i]] = j;
          r0_3[i][nneighbor3[i]] = rsq;
          nneighbor3[i]++;
        }
      }else{
        if (abs(rsq - MAX(first_neighbors_dist_in, first_neighbor_dist_out)) <= tol){
          //if (nneighbor1[i] == 0) fprintf(screen,"neighbor list1 of %d\n",i);
          fullneigh1[i][nneighbor1[i]] = j;
          //fprintf(screen,"%d\n",fullneigh1[i][nneighbor1[i]]);
          r0_1[i][nneighbor1[i]] = rsq;
          nneighbor1[i]++;
        }else if (abs(rsq - MIN(first_neighbors_dist_in, first_neighbor_dist_out)) <= tol){
          //if (nneighbor2[i] == 0) fprintf(screen,"neighbor list2 of %d\n",i);
          fullneigh2[i][nneighbor2[i]] = j;
          //fprintf(screen,"%d\n",fullneigh2[i][nneighbor2[i]]);
          r0_2[i][nneighbor2[i]] = rsq;
          nneighbor2[i]++;
        }else if (abs(rsq - second_neighbor_dist) <= tol){
          //if (nneighbor3[i] == 0) fprintf(screen,"neighbor list3 of %d\n",i);
          fullneigh3[i][nneighbor3[i]] = j;
          //fprintf(screen,"%d\n",fullneigh3[i][nneighbor3[i]]);
          r0_3[i][nneighbor3[i]] = rsq;
          nneighbor3[i]++;
        }
      }
    }
  }

  // communicate neighbor lists to ghosts
  // comm->forward_comm(this);

  // bond statistics

  int n = 0;
  for (i = 0; i < nlocal; i++) n += (nneighbor1[i]+nneighbor2[i]+nneighbor3[i]);
  int nall;
  MPI_Allreduce(&n,&nall,1,MPI_INT,MPI_SUM,world);

  if (comm->me == 0) {
    if (screen) {
      fprintf(screen,"LPM bonds:\n");
      fprintf(screen,"  total # of bonds = %d\n",nall);
      fprintf(screen,"  bonds/atom = %g\n",(double)nall/atom->natoms);
    }
    if (logfile) {
      fprintf(logfile,"LPM bonds:\n");
      fprintf(logfile,"  total # of bonds = %d\n",nall);
      fprintf(logfile,"  bonds/atom = %g\n",(double)nall/atom->natoms);
    }
  }
}

/* ----------------------------------------------------------------------
   memory usage of local atom-based arrays
------------------------------------------------------------------------- */

double FixLpmNeigh::memory_usage()
{
  int nmax = atom->nmax;
  int bytes = (double)nmax * sizeof(int);
  bytes += 3.0 * (double)nmax * sizeof(int);
  bytes += (double)nmax*maxnneigh * sizeof(tagint);
  bytes += (double)nmax*maxnneigh * sizeof(double);
  //bytes += 3.0 * (double)nmax*sizeof(double);
  return bytes;
}

/* ----------------------------------------------------------------------
   allocate local atom-based arrays
------------------------------------------------------------------------- */

void FixLpmNeigh::grow_arrays(int nmax)
{
   memory->grow(nneighbor1,nmax,"lpm_neigh:nneighbor1");
   memory->grow(nneighbor2,nmax,"lpm_neigh:nneighbor2");
   memory->grow(nneighbor3,nmax,"lpm_neigh:nneighbor3");
   memory->grow(fullneigh1,nmax,maxnneigh_1,"lpm_neigh:fullneigh1");
   memory->grow(fullneigh2,nmax,maxnneigh_1,"lpm_neigh:fullneigh2");
   memory->grow(fullneigh3,nmax,maxnneigh_1,"lpm_neigh:fullneigh3");
   memory->grow(r0_1,nmax,maxnneigh_1,"lpm_neigh:r0_1");
   memory->grow(r0_2,nmax,maxnneigh_1,"lpm_neigh:r0_2");
   memory->grow(r0_3,nmax,maxnneigh_1,"lpm_neigh:r0_3");
   //memory->grow(sum_dr1,nmax,"lpm_neigh:sum_dr1");
   //memory->grow(sum_dr2,nmax,"lpm_neigh:sum_dr2");
   //memory->grow(sum_dr3,nmax,"lpm_neigh:sum_dr3");

}

/* ----------------------------------------------------------------------
   copy values within local atom-based arrays
------------------------------------------------------------------------- */

void FixLpmNeigh::copy_arrays(int i, int j, int /*delflag*/)
{
  nneighbor1[j] = nneighbor1[i];
  nneighbor2[j] = nneighbor2[i];
  nneighbor3[j] = nneighbor3[i];
  //sum_dr1[j] = sum_dr1[i];
  //sum_dr2[j] = sum_dr2[i];
  //sum_dr3[j] = sum_dr3[i];
  for (int m = 0; m < nneighbor1[j]; m++) {
    fullneigh1[j][m] = fullneigh1[i][m];
    r0_1[j][m] = r0_1[i][m];
  }
  for (int m = 0; m < nneighbor2[j]; m++) {
    fullneigh2[j][m] = fullneigh2[i][m];
    r0_2[j][m] = r0_2[i][m];
  }
  for (int m = 0; m < nneighbor3[j]; m++) {
    fullneigh3[j][m] = fullneigh3[i][m];
    r0_3[j][m] = r0_3[i][m];
  }
}

/* ----------------------------------------------------------------------
   pack values in local atom-based arrays for exchange with another proc
------------------------------------------------------------------------- */

int FixLpmNeigh::pack_exchange(int i, double *buf)
{
  // compact list by eliminating fullneigh = 0 entries
  // set buf[0] after compaction
  //fprintf(screen,"pack exchange starts");
  int m = 3;
  int m1 = 0, m2 = 0, m3 = 0;
  for (int n = 0; n < nneighbor1[i]; n++) {
    if (fullneigh1[i][n] == 0) continue;
    buf[m] = fullneigh1[i][n];
    m++;
    buf[m] = r0_1[i][n];
    m++;
    m1++;
  }
  for (int n = 0; n < nneighbor2[i]; n++) {
    if (fullneigh2[i][n] == 0) continue;
    buf[m] = fullneigh2[i][n];
    m++;
    buf[m] = r0_2[i][n];
    m++;
    m2++;
  }
  for (int n = 0; n < nneighbor3[i]; n++) {
    if (fullneigh3[i][n] == 0) continue;
    buf[m] = fullneigh3[i][n];
    m++;
    buf[m] = r0_3[i][n];
    m++;
    m3++;
  }
  //buf[m] = sum_dr1[i];
  //m++;
  //buf[m] = sum_dr2[i];
  //m++;
  //buf[m] = sum_dr3[i];
  //m++;
  buf[0] = m1;
  buf[1] = m2;
  buf[2] = m3;
  return m;
}

/* ----------------------------------------------------------------------
   unpack values in local atom-based arrays from exchange with another proc
------------------------------------------------------------------------- */

int FixLpmNeigh::unpack_exchange(int nlocal, double *buf)
{
  //fprintf(screen,"unpack exchange starts");
  int m = 0;
  nneighbor1[nlocal] = static_cast<int> (buf[m]);
  m++;
  nneighbor2[nlocal] = static_cast<int> (buf[m]);
  m++;
  nneighbor3[nlocal] = static_cast<int> (buf[m]);
  m++;
  int n_tag;
  for (int n = 0; n < nneighbor1[nlocal]; n++) {
    fullneigh1[nlocal][n] = static_cast<tagint> (buf[m]);
    m++;
    r0_1[nlocal][n] = buf[m];
    m++;
  }
  for (int n = 0; n < nneighbor2[nlocal]; n++) {
    fullneigh2[nlocal][n] = static_cast<tagint> (buf[m]);
    m++;
    r0_2[nlocal][n] = buf[m];
    m++;
  }
  for (int n = 0; n < nneighbor3[nlocal]; n++) {
    fullneigh3[nlocal][n] = static_cast<tagint> (buf[m]);
    m++;
    r0_3[nlocal][n] = buf[m];
    m++;
  }
  //sum_dr1[nlocal] = buf[m];
  //m++;
  //sum_dr2[nlocal] = buf[m];
  //m++;
  //sum_dr3[nlocal] = buf[m];
  //m++;
  return m;
}



//int FixLpmNeigh::pack_forward_comm(int n, int *list, double *buf,
  //                                  int /*pbc_flag*/, int * /*pbc*/)
//{
  //int i,j,m;
  //m = 0;
  //for (i = 0; i < n; i++) {
  //  j = list[i];
  //  buf[m] = sum_dr1[j];
  //  m++;
  //  buf[m] = sum_dr2[j];
  //  m++;
  //  buf[m] = sum_dr3[j];
  //  m++;
  //}
  //return m;
//}

/* ---------------------------------------------------------------------- */

//void FixLpmNeigh::unpack_forward_comm(int n, int first, double *buf)
//{
  //int i,m,last;

  //m = 0;
  //last = first + n;
  //for (i = first; i < last; i++){
    //sum_dr1[i] = buf[m];
    //m++;
    //sum_dr2[i] = buf[m];
    //m++;
    //sum_dr3[i] = buf[m];
    //m++;
  //}

//}

/* ----------------------------------------------------------------------
   pack entire state of Fix into one write
------------------------------------------------------------------------- */

void FixLpmNeigh::write_restart(FILE *fp)
{
  int n = 0;
  double list[3];
  list[n++] = first;
  list[n++] = maxnneigh_1;
  list[n++] = maxnneigh;

  if (comm->me == 0) {
    int size = n * sizeof(double);
    fwrite(&size,sizeof(int),1,fp);
    fwrite(list,sizeof(double),n,fp);
  }
}

/* ----------------------------------------------------------------------
   use state info from restart file to restart the Fix
------------------------------------------------------------------------- */

void FixLpmNeigh::restart(char *buf)
{
  int n = 0;
  auto list = (double *) buf;

  first = static_cast<int> (list[n++]);
  maxnneigh_1 = static_cast<int> (list[n++]);
  maxnneigh = static_cast<int> (list[n++]);

  // grow 2D arrays now, cannot change size of 2nd array index later

  grow_arrays(atom->nmax);
}

/* ----------------------------------------------------------------------
   pack values in local atom-based arrays for restart file
------------------------------------------------------------------------- */

int FixLpmNeigh::pack_restart(int i, double *buf)
{
  int m = 0;
  // pack buf[0] this way b/c other fixes unpack it
  buf[m++] = 2*nneighbor1[i] + 4;
  buf[m++] = nneighbor1[i];
  buf[m++] = 2*nneighbor2[i] + 4;
  buf[m++] = nneighbor2[i];
  buf[m++] = 2*nneighbor3[i] + 4;
  buf[m++] = nneighbor3[i];
  for (int n = 0; n < nneighbor1[i]; n++) {
    buf[m++] = fullneigh1[i][n];
    buf[m++] = r0_1[i][n];
  }
  for (int n = 0; n < nneighbor2[i]; n++) {
    buf[m++] = fullneigh2[i][n];
    buf[m++] = r0_2[i][n];
  }
  for (int n = 0; n < nneighbor3[i]; n++) {
    buf[m++] = fullneigh3[i][n];
    buf[m++] = r0_3[i][n];
  }
  return m;
}

/* ----------------------------------------------------------------------
   unpack values from atom->extra array to restart the fix
------------------------------------------------------------------------- */

void FixLpmNeigh::unpack_restart(int nlocal, int nth)
{

  double **extra = atom->extra;

  // skip to Nth set of extra values
  // unpack the Nth first values this way b/c other fixes pack them

  int m = 0;
  for (int i = 0; i < nth; i++) m += static_cast<int> (extra[nlocal][m]);
  m++;

  nneighbor1[nlocal] = static_cast<int> (extra[nlocal][m++]);
  nneighbor2[nlocal] = static_cast<int> (extra[nlocal][m++]);
  nneighbor3[nlocal] = static_cast<int> (extra[nlocal][m++]);
  for (int n = 0; n < nneighbor1[nlocal]; n++) {
    fullneigh1[nlocal][n] = static_cast<tagint> (extra[nlocal][m++]);
    r0_1[nlocal][n] = extra[nlocal][m++];
  }
  for (int n = 0; n < nneighbor2[nlocal]; n++) {
    fullneigh2[nlocal][n] = static_cast<tagint> (extra[nlocal][m++]);
    r0_2[nlocal][n] = extra[nlocal][m++];
  }
  for (int n = 0; n < nneighbor3[nlocal]; n++) {
    fullneigh3[nlocal][n] = static_cast<tagint> (extra[nlocal][m++]);
    r0_3[nlocal][n] = extra[nlocal][m++];
  }
}

/* ----------------------------------------------------------------------
   maxsize of any atom's restart data
------------------------------------------------------------------------- */

int FixLpmNeigh::maxsize_restart()
{
  return 2*maxnneigh + 4;
}

/* ----------------------------------------------------------------------
   size of atom nlocal's restart data
------------------------------------------------------------------------- */

int FixLpmNeigh::size_restart(int nlocal)
{
  return 2*(nneighbor1[nlocal]+nneighbor2[nlocal]+nneighbor3[nlocal] ) + 4;
}
