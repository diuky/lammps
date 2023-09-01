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
   Contributing author: Mike Parks (SNL)
------------------------------------------------------------------------- */

#include "pair_lpm_le.h"

#include "atom.h"
#include "comm.h"
#include "domain.h"
#include "comm.h"
#include "error.h"
#include "fix_lpm_neigh.h"
#include "force.h"
#include "lattice.h"
#include "neighbor.h"
#include "neigh_list.h"
#include "neigh_request.h"
#include "math_const.h"
#include "memory.h"
#include "pair.h"

#include <cmath>

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

PairLpmLE::PairLpmLE(LAMMPS *_lmp) : PairLpm(_lmp)
{
  single_enable = 1;
}

void PairLpmLE::coeff(int narg, char **arg)
{
  if (narg != 10) error->all(FLERR,"Incorrect args for pair coefficients");

  int ilo,ihi,jlo,jhi;
  utils::bounds(FLERR,arg[0],1,atom->ntypes,ilo,ihi,error);
  utils::bounds(FLERR,arg[1],1,atom->ntypes,jlo,jhi,error);

  num_ind = utils::numeric(FLERR,arg[2],false,lmp);
  lattice_a = utils::numeric(FLERR,arg[3],false,lmp);
  ca_ratio = utils::numeric(FLERR,arg[4],false,lmp);
  if (!allocated) allocate();
  for (int i = 0; i < num_ind; i++){
    Cijkl[i] = utils::numeric(FLERR,arg[i+5],false,lmp); //setting up the independent material stiffness constants
  }
lattice_c = lattice_a * ca_ratio;

for (int i = ilo; i <= ihi; i++) {
  for (int j = MAX(jlo,i); j <= jhi; j++) {
    cut[i][j] = sqrt(4./3.*pow(lattice_a,2)+pow(lattice_c,2)/4.);
    setflag[i][j] = 1;
  }
}
m1 = -4.*sqrt(3.)*(40.*pow(lattice_a,4)+42.*pow(lattice_a,2)*pow(lattice_c,2)+9.*pow(lattice_c,4))/(9.*lattice_c*(32.*pow(lattice_a,2)+15.*pow(lattice_c,2)));
m2 = 4.*sqrt(3.)*pow(lattice_a,2)*(4.*pow(lattice_a,2)+3.*pow(lattice_c,2))/(27.*pow(lattice_c,3));
m3 = sqrt(3.)*(64.*pow(lattice_a,4)+60.*pow(lattice_a,2)*pow(lattice_c,2)+9.*pow(lattice_c,4))/(6.*lattice_c*(32.*pow(lattice_a,2)+15.*pow(lattice_c,2)));
m4 = -16.*sqrt(3.)*(10.*pow(lattice_a,4)+3.*pow(lattice_a,2)*pow(lattice_c,2))/(9.*lattice_c*(32.*pow(lattice_a,2)+15.*pow(lattice_c,2)));
m5 = 16.*sqrt(3.)*pow(lattice_a,4)/(27.*pow(lattice_c,3));
m6 = -2.*sqrt(3.)*(80.*pow(lattice_a,4)+51*pow(lattice_a,2)*pow(lattice_c,2))/(9*lattice_c*(32.*pow(lattice_a,2)+15.*pow(lattice_c,2)));
m7 = sqrt(3.)*(160.*pow(lattice_a,4)+78.*pow(lattice_a,2)*pow(lattice_c,2)+9.*pow(lattice_c,4))/(9.*lattice_c*(32.*pow(lattice_a,2)+15.*pow(lattice_c,2)));
m8 = -sqrt(3.)*pow(lattice_a,2)*(16.*pow(lattice_a,2)+3.*pow(lattice_c,2))/(27.*pow(lattice_c,3));
m9 = sqrt(3.)*(64.*pow(lattice_a,4)+60.*pow(lattice_a,2)*pow(lattice_c,2)+9.*pow(lattice_c,4))/(6.*lattice_c*(32.*pow(lattice_a,2)+15.*pow(lattice_c,2)));
m10 = -2.*sqrt(3.)*(80.*pow(lattice_a,4)+51.*pow(lattice_a,2)*pow(lattice_c,2))/(27.*lattice_c*(32.*pow(lattice_a,2)+15.*pow(lattice_c,2)));
m11 = 2.*sqrt(3.)*(80.*pow(lattice_a,4)+51.*pow(lattice_a,2)*pow(lattice_c,2))/(27.*lattice_c*(32.*pow(lattice_a,2)+15.*pow(lattice_c,2)));
m12 = sqrt(3.)*(4.*pow(lattice_a,2)+3.*pow(lattice_c,2))*(16.*pow(lattice_a,2)+3.*pow(lattice_c,2))/(18.*lattice_c*(32.*pow(lattice_a,2)+15.*pow(lattice_c,2)));
m13 = -sqrt(3.)*(4.*pow(lattice_a,2)+3.*pow(lattice_c,2))*(16.*pow(lattice_a,2)+3.*pow(lattice_c,2))/(18.*lattice_c*(32.*pow(lattice_a,2)+15.*pow(lattice_c,2)));

double mapping_c[num_ind][num_ind] = {
 {sqrt(3.)*lattice_c/3., -sqrt(3.)*lattice_c/3., m4, m5, m6},
 {0.0, 0.0, m1, m2, m3},
 {0.0, 0.0, m7, m8, m9},
 {-sqrt(3.)*lattice_c/18., sqrt(3.)*lattice_c/6., m10, 0.0, m11},
 {0.0, 0.0, m12, 0.0, m13}
};

// Calculate bond coefficients
for (int ii=0; ii < num_ind; ii++){
  Knkl[ii] = 0;
  for (int jj = 0; jj < num_ind; jj++){
    Knkl[ii] += mapping_c[ii][jj]*Cijkl[jj];
  }
  // fprintf(screen, "Value for KN: %f\n", Knkl[ii]);
}
}

/* ---------------------------------------------------------------------- */

void PairLpmLE::compute(int eflag, int vflag)
{
  int i,j,ii,jj,inum;
  double xtmp,ytmp,ztmp,delx,dely,delz, rsq, rsq0, dr, rk, tk, fpair, epair = 0.0;
  double xtmp0,ytmp0,ztmp0,delx0,dely0,delz0;
  int *ilist,*jlist1,*jlist2,*jlist3,*numneigh,**firstneigh;

  double **f = atom->f;
  double **x = atom->x;
  int nlocal = atom->nlocal;

  double **x0 = atom->x0;
  tagint **fullneigh1 = fix_lpm_neigh->fullneigh1;
  tagint **fullneigh2 = fix_lpm_neigh->fullneigh2;
  tagint **fullneigh3 = fix_lpm_neigh->fullneigh3;
  int *nneighbor1 = fix_lpm_neigh->nneighbor1;
  int *nneighbor2 = fix_lpm_neigh->nneighbor2;
  int *nneighbor3 = fix_lpm_neigh->nneighbor3;
  //double *sum_dr1 = fix_lpm_neigh->sum_dr1;
  //double *sum_dr2 = fix_lpm_neigh->sum_dr2;
  //double *sum_dr3 = fix_lpm_neigh->sum_dr3;

  ev_init(eflag, vflag);

  // forces for each atoms

  int newton_pair = force->newton_pair;

  inum = list->inum;
  ilist = list->ilist;
 // compute nonlocal term for each particle
  compute_nonlocal(0,nlocal);

  // communicate nonlocal term of each particles
//if (newton_pair)  comm->reverse_comm(this);
  comm->forward_comm(this);

  // loop over neighbors of my atoms
  // need minimg() for x0 difference since not ghosted

  for (ii = 0; ii < inum; ii++) {
    i = ilist[ii];
    xtmp = x[i][0];
    ytmp = x[i][1];
    ztmp = x[i][2];
    xtmp0 = x0[i][0];
    ytmp0 = x0[i][1];
    ztmp0 = x0[i][2];
    jlist1 = fullneigh1[i];
    jlist2 = fullneigh2[i];
    jlist3 = fullneigh3[i];

  /*  for (jj = 0; jj < nneighbor1[i]; jj++) {
      j = jlist1[jj];
      delx = xtmp - x[j][0];
      dely = ytmp - x[j][1];
      delz = ztmp - x[j][2];
      delx0 = xtmp0 - x0[j][0];
      dely0 = ytmp0 - x0[j][1];
      delz0 = ztmp0 - x0[j][2];
      rsq = sqrt(delx*delx+dely*dely+delz*delz);
      // rsq0 = sqrt(delx0*delx0+dely0*dely0+delz0*delz0);
      rsq0 = lattice_a;
      dr = rsq -rsq0;
      sum_dr1 += dr;
      //if (rsq == 0){
      //  fprintf(screen, "the couples with same coord: %d  and  %d\n", i,j);
      //}
    } */

  for (jj = 0; jj < nneighbor1[i]; jj++) {
      j = jlist1[jj];
      delx = xtmp - x[j][0];
      dely = ytmp - x[j][1];
      delz = ztmp - x[j][2];
      delx0 = xtmp0 - x0[j][0];
      dely0 = ytmp0 - x0[j][1];
      delz0 = ztmp0 - x0[j][2];
      rsq = sqrt(delx*delx+dely*dely+delz*delz);
      //rsq0 = sqrt(delx0*delx0+dely0*dely0+delz0*delz0);
      rsq0 = lattice_a;
      dr = rsq - rsq0;
      rk = 0.5*Knkl[0]*dr;
      tk = 0.25*Knkl[3]*sum_dr1[i];
      if (rsq > 0.0) fpair = -(rk+tk)/rsq;
      else fpair = 0.0;
      if (newton_pair || i < nlocal){
        f[i][0] += delx*fpair;
        f[i][1] += dely*fpair;
        f[i][2] += delz*fpair;
      }
      if (newton_pair || j < nlocal){
        f[j][0] -= delx*fpair;
        f[j][1] -= dely*fpair;
        f[j][2] -= delz*fpair;
      }

      if (eflag)  {
        epair = 0.5*(rk+tk)*dr;
      }
      if (evflag) ev_tally(i,j,nlocal,newton_pair,epair,0.0,fpair,delx,dely,delz);
    }

  /*  for (jj = 0; jj < nneighbor2[i]; jj++) {
      j = jlist2[jj];
      delx = xtmp - x[j][0];
      dely = ytmp - x[j][1];
      delz = ztmp - x[j][2];
      delx0 = xtmp0 - x0[j][0];
      dely0 = ytmp0 - x0[j][1];
      delz0 = ztmp0 - x0[j][2];
      rsq = sqrt(delx*delx+dely*dely+delz*delz);
      //rsq0 = sqrt(delx0*delx0+dely0*dely0+delz0*delz0);
      rsq0 = sqrt(pow(lattice_a,2)/3.+pow(lattice_c,2)/4.);
      dr = rsq -rsq0;
      sum_dr2 += dr;
      //if (rsq == 0){
      //  fprintf(screen, "the couples with same coord: %d  and  %d\n", i,j);
      //}
    }*/

  for (jj = 0; jj < nneighbor2[i]; jj++) {
      j = jlist2[jj];
      delx = xtmp - x[j][0];
      dely = ytmp - x[j][1];
      delz = ztmp - x[j][2];
      delx0 = xtmp0 - x0[j][0];
      dely0 = ytmp0 - x0[j][1];
      delz0 = ztmp0 - x0[j][2];
      rsq = sqrt(delx*delx+dely*dely+delz*delz);
      //rsq0 = sqrt(delx0*delx0+dely0*dely0+delz0*delz0);
      rsq0 = sqrt(pow(lattice_a,2)/3.+pow(lattice_c,2)/4.);
      dr = rsq - rsq0;
      rk = 0.5*Knkl[1]*dr;
      tk = 0.25*Knkl[4]*sum_dr2[i];
      if (rsq > 0.0) fpair = -(rk+tk)/rsq;
      else fpair = 0.0;
      if (newton_pair || i < nlocal){
        f[i][0] += delx*fpair;
        f[i][1] += dely*fpair;
        f[i][2] += delz*fpair;
      }
      if (newton_pair || j < nlocal){
        f[j][0] -= delx*fpair;
        f[j][1] -= dely*fpair;
        f[j][2] -= delz*fpair;
      }

      if (eflag)  {
        epair = 0.5*(rk+tk)*dr;
      }
      if (evflag) ev_tally(i,j,nlocal,newton_pair,epair,0.0,fpair,delx,dely,delz);
    }

    /* for (jj = 0; jj < nneighbor3[i]; jj++) {
      j = jlist3[jj];
      delx = xtmp - x[j][0];
      dely = ytmp - x[j][1];
      delz = ztmp - x[j][2];
      delx0 = xtmp0 - x0[j][0];
      dely0 = ytmp0 - x0[j][1];
      delz0 = ztmp0 - x0[j][2];
      rsq = sqrt(delx*delx+dely*dely+delz*delz);
      //rsq0 = sqrt(delx0*delx0+dely0*dely0+delz0*delz0);
      rsq0 = sqrt(4./3.*pow(lattice_a,2)+pow(lattice_c,2)/4.);
      dr = rsq -rsq0;
      sum_dr3 += dr;
      //if (rsq == 0){
        //fprintf(screen, "the couples with same coord: %d  and  %d\n", i,j);
      //}
    }*/

  for (jj = 0; jj < nneighbor3[i]; jj++) {
      j = jlist3[jj];
      delx = xtmp - x[j][0];
      dely = ytmp - x[j][1];
      delz = ztmp - x[j][2];
      delx0 = xtmp0 - x0[j][0];
      dely0 = ytmp0 - x0[j][1];
      delz0 = ztmp0 - x0[j][2];
      rsq = sqrt(delx*delx+dely*dely+delz*delz);
      //rsq0 = sqrt(delx0*delx0+dely0*dely0+delz0*delz0);
      rsq0 = sqrt(4./3.*pow(lattice_a,2)+pow(lattice_c,2)/4.);
      dr = rsq - rsq0;
      rk = 0.5*Knkl[2]*dr;
      tk = 0.25*Knkl[4]*sum_dr3[i];
      if (rsq > 0.0) fpair = -(rk+tk)/rsq;
      else fpair = 0.0;
      if (newton_pair || i < nlocal){
        f[i][0] += delx*fpair;
        f[i][1] += dely*fpair;
        f[i][2] += delz*fpair;
      }
      if (newton_pair || j < nlocal){
        f[j][0] -= delx*fpair;
        f[j][1] -= dely*fpair;
        f[j][2] -= delz*fpair;
      }

      if (eflag)  {
        epair = 0.5*(rk+tk)*dr;
      }
      if (evflag) ev_tally(i,j,nlocal,newton_pair,epair,0.0,fpair,delx,dely,delz);
    }
    }

    //for(ii = 0; ii < inum; ii++){
      //i = ilist[ii];
      //fprintf(screen,"xforces of particle %d\n",i);
      //fprintf(screen,"%lf\n", atom->f[i][0]);
    //}
}

double PairLpmLE::init_one(int i, int j)
{
  if (setflag[i][j] == 0) error->all(FLERR,"All pair coeffs are not set");
  cut[j][i] = cut[i][j];

  return cut[i][j];
}


/* ----------------------------------------------------------------------
  proc 0 writes to restart file
------------------------------------------------------------------------- */

void PairLpmLE::write_restart(FILE *fp)
{
  int i;
  for (i = 1; i <= num_ind; i++){
        fwrite(&Cijkl[i],sizeof(double),1,fp);
    }
}

/* ----------------------------------------------------------------------
  proc 0 reads from restart file, bcasts
------------------------------------------------------------------------- */

void PairLpmLE::read_restart(FILE *fp)
{
  allocate();

  int i;
  int me = comm->me;
  for (i = 1; i <= num_ind; i++){
        if (me == 0) {
          utils::sfread(FLERR,&Cijkl[i],sizeof(double),1,fp,nullptr,error);
        }
        MPI_Bcast(&Cijkl[i],1,MPI_DOUBLE,0,world);
      }
}

double PairLpmLE::single(int i, int j, int itype, int jtype, double rsq,
                           double /*factor_coul*/, double /*factor_lj*/,
                           double &fforce)
{
  double **x0 = atom->x0;
  double r0 = sqrt((x0[i][0]-x0[j][0])*(x0[i][0]-x0[j][0])+(x0[i][1]-x0[j][1])*(x0[i][1]-x0[j][1])+(x0[i][2]-x0[j][2])*(x0[i][2]-x0[j][2]));
  double r = sqrt(rsq);
  double dr = r - r0;
  int *nneighbor1 = fix_lpm_neigh->nneighbor1;
  int *nneighbor2 = fix_lpm_neigh->nneighbor2;
  int *nneighbor3 = fix_lpm_neigh->nneighbor3;
  tagint **fullneigh1 = fix_lpm_neigh->fullneigh1;
  tagint **fullneigh2 = fix_lpm_neigh->fullneigh2;
  tagint **fullneigh3 = fix_lpm_neigh->fullneigh3;
  int nlocal = atom->nlocal;
  int *jlist1, *jlist2, *jlist3;
  double energy = 0;
  fforce = 0.0;
  if (r > 0){
    jlist1 = fullneigh1[i];
    jlist2 = fullneigh2[i];
    jlist3 = fullneigh3[i];
    for(int jj = 0; jj < nneighbor1[i]; jj++){
      if(j == jlist1[jj]) {energy += (Knkl[0]*dr)*dr;
         fforce = -(Knkl[0]*dr)/r;
      }
    }

    for(int jj = 0; jj < nneighbor2[i]; jj++){
      if(j == jlist2[jj]) {energy += (Knkl[1]*dr)*dr;
      fforce = -(Knkl[1]*dr)/r;
    }
    }


    for(int jj = 0; jj < nneighbor3[i]; jj++){
      if(j == jlist3[jj]) {energy += (Knkl[2]*dr)*dr;
       fforce = -(Knkl[2]*dr)/r;
      }
    }
}
  return energy;
}
