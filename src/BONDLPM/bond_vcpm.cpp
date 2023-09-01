/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://lammps.sandia.gov/, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#include "bond_vcpm.h"

#include <cmath>
#include <cstring>
#include "atom.h"
#include "neighbor.h"
#include "comm.h"
#include "force.h"
#include "memory.h"
#include "error.h"


using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

BondVcpm::BondVcpm(LAMMPS *lmp) : Bond(lmp)
{
  reinitflag = 1;
}

/* ---------------------------------------------------------------------- */

BondVcpm::~BondVcpm()
{
  if (allocated && !copymode) {
    memory->destroy(setflag);
    memory->destroy(k);
    memory->destroy(r0);
    memory->destroy(t);

}
}

/* ---------------------------------------------------------------------- */

void BondVcpm::compute(int eflag, int vflag)
{
  int i1,i2,n,type;
  double ebond,fbond;
  double rk,rsq,tk, delx,dely,delz,dr;

  ebond = 0.0;
  ev_init(eflag,vflag);

  double **x = atom->x;
  double **f = atom->f;
  int **bondlist = neighbor->bondlist;
  int nbondlist = neighbor->nbondlist;
  //fprintf(screen,"num of bond types: %d\n",nbondlist);
  int nlocal = atom->nlocal;
  int total = atom->nlocal +  atom->nghost;
  int newton_bond = force->newton_bond;
  double bodystrain[total]={0},r;

  for (n = 0; n < nbondlist; n++) {
    i1 = bondlist[n][0];
    i2 = bondlist[n][1];
    type = bondlist[n][2];
    //fprintf(screen,"bond list: %d, %d, %d\n",i1, i2, type);

    delx = x[i1][0] - x[i2][0];
    dely = x[i1][1] - x[i2][1];
    delz = x[i1][2] - x[i2][2];

    rsq = delx*delx + dely*dely + delz*delz;
    r = sqrt(rsq);

    dr = (r - r0[type])/2;
    bodystrain[i1] += dr/2;
    bodystrain[i2] += dr/2;
}

  for (n = 0; n < nbondlist; n++) {

    i1 = bondlist[n][0];
    i2 = bondlist[n][1];
    type = bondlist[n][2];

    delx = x[i1][0] - x[i2][0];
    dely = x[i1][1] - x[i2][1];
    delz = x[i1][2] - x[i2][2];

    rsq = delx*delx + dely*dely + delz*delz;
    r = sqrt(rsq);

    dr = (r - r0[type])/2;

    rk = k[type] * dr;
    tk = t[type] * (bodystrain[i1]+bodystrain[i2])/2;

    // force & energy

    if (r > 0.0) fbond = -(rk+tk);
    else fbond = 0.0;

    if (eflag) ebond = 0.5*(rk*dr+tk*dr);

    // apply force to each of 2 atoms

    if (newton_bond || i1 < nlocal) {
      f[i1][0] += delx*fbond/r;
      f[i1][1] += dely*fbond/r;
      f[i1][2] += delz*fbond/r;
    }

    if (newton_bond || i2 < nlocal) {
      f[i2][0] -= delx*fbond/r;
      f[i2][1] -= dely*fbond/r;
      f[i2][2] -= delz*fbond/r;
    }

    if (evflag) ev_tally(i1,i2,nlocal,newton_bond,ebond,fbond,delx,dely,delz);
  }

  //for(int ii = 0; ii < nlocal; ii++){
    //fprintf(screen,"xforces of particle %d\n",ii);
    //fprintf(screen,"%lf\n", atom->f[ii][0]);
  //}

}

/* ---------------------------------------------------------------------- */

void BondVcpm::allocate()
{
  allocated = 1;
  int n = atom->nbondtypes;

  memory->create(k,n+1,"bond:k");
  memory->create(t,n+1,"bond:t");
  memory->create(r0,n+1,"bond:r0");

  memory->create(setflag,n+1,"bond:setflag");
  for (int i = 1; i <= n; i++) setflag[i] = 0;
}

/* ----------------------------------------------------------------------
   set coeffs for one or more types
------------------------------------------------------------------------- */

void BondVcpm::coeff(int narg, char **arg)
{
  if (narg != 4) error->all(FLERR,"Incorrect args for bond coefficients");
  if (!allocated) allocate();

  int ilo,ihi;
  utils::bounds(FLERR,arg[0],1,atom->nbondtypes,ilo,ihi,error);

  double k_one = utils::numeric(FLERR,arg[1],false,lmp);
  double t_one = utils::numeric(FLERR,arg[2],false,lmp);
  double r0_one = utils::numeric(FLERR,arg[3],false,lmp);

  int count = 0;
  for (int i = ilo; i <= ihi; i++) {
    k[i] = k_one;
    t[i] = t_one;
    r0[i] = r0_one;
    setflag[i] = 1;
    count++;
  }

  if (count == 0) error->all(FLERR,"Incorrect args for bond coefficients");
}

/* ----------------------------------------------------------------------
   return an equilbrium bond length
------------------------------------------------------------------------- */

double BondVcpm::equilibrium_distance(int i)
{
  return r0[i];
}

/* ----------------------------------------------------------------------
   proc 0 writes out coeffs to restart file
------------------------------------------------------------------------- */

void BondVcpm::write_restart(FILE *fp)
{
  fwrite(&k[1],sizeof(double),atom->nbondtypes,fp);
  fwrite(&t[1],sizeof(double),atom->nbondtypes,fp);
  fwrite(&r0[1],sizeof(double),atom->nbondtypes,fp);
}

/* ----------------------------------------------------------------------
   proc 0 reads coeffs from restart file, bcasts them
------------------------------------------------------------------------- */

void BondVcpm::read_restart(FILE *fp)
{
  allocate();

  if (comm->me == 0) {
    utils::sfread(FLERR,&k[1],sizeof(double),atom->nbondtypes,fp,nullptr,error);
    utils::sfread(FLERR,&t[1],sizeof(double),atom->nbondtypes,fp,nullptr,error);
    utils::sfread(FLERR,&r0[1],sizeof(double),atom->nbondtypes,fp,nullptr,error);
  }
  MPI_Bcast(&k[1],atom->nbondtypes,MPI_DOUBLE,0,world);
  MPI_Bcast(&t[1],atom->nbondtypes,MPI_DOUBLE,0,world);
  MPI_Bcast(&r0[1],atom->nbondtypes,MPI_DOUBLE,0,world);

  for (int i = 1; i <= atom->nbondtypes; i++) setflag[i] = 1;
}

/* ----------------------------------------------------------------------
   proc 0 writes to data file
------------------------------------------------------------------------- */

void BondVcpm::write_data(FILE *fp)
{
  for (int i = 1; i <= atom->nbondtypes; i++)
    fprintf(fp,"%d %g %g %g\n",i,k[i],t[i],r0[i]);
}

/* ---------------------------------------------------------------------- */

double BondVcpm::single(int type, double rsq, int /*i*/, int /*j*/,
                            double &fforce)
{
  double r = sqrt(rsq);
  double dr = r - r0[type];
  double rk = k[type] * dr;
  fforce = 0;
  if (r > 0.0) fforce = -rk/r;
  return rk*dr;
}

/* ----------------------------------------------------------------------
    Return ptr to internal members upon request.
------------------------------------------------------------------------ */
void *BondVcpm::extract(const char *str, int &dim)
{
  dim = 1;
  if (strcmp(str,"kappa")==0) return (void*) k;
  if (strcmp(str,"r0")==0) return (void*) r0;
  return nullptr;
}
