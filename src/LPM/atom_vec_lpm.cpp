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

#include "atom_vec_lpm.h"

#include "atom.h"
#include "citeme.h"
#include "error.h"

#include <cfloat>

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

AtomVecLpm::AtomVecLpm(LAMMPS *lmp) : AtomVec(lmp)
{

  molecular = Atom::ATOMIC;

  atom->rmass_flag = 1;
  atom->lpm_flag = 1;

  // strings with peratom variables to include in each AtomVec method
  // strings cannot contain fields in corresponding AtomVec default strings
  // order of fields in a string does not matter
  // except: fields_data_atom & fields_data_vel must match data file

  fields_grow = {"rmass", "s0", "x0"};
  fields_copy = {"rmass", "s0", "x0"};
  fields_comm = {"s0"};
  fields_comm_vel = {"s0"};
  fields_border = {"rmass", "s0", "x0"};
  fields_border_vel = {"rmass", "s0", "x0"};
  fields_exchange = {"rmass", "s0", "x0"};
  fields_restart = {"rmass", "s0", "x0"};
  fields_create = {"rmass", "s0", "x0"};
  fields_data_atom = {"id", "type", "rmass", "x"};
  fields_data_vel = {"id", "v"};

  setup_fields();
}

/* ----------------------------------------------------------------------
   set local copies of all grow ptrs used by this class, except defaults
   needed in replicate when 2 atom classes exist and it calls pack_restart()
------------------------------------------------------------------------- */

void AtomVecLpm::grow_pointers()
{
  rmass = atom->rmass;
  s0 = atom->s0;
  x0 = atom->x0;
}

/* ----------------------------------------------------------------------
   initialize non-zero atom quantities
------------------------------------------------------------------------- */

void AtomVecLpm::create_atom_post(int ilocal)
{
  rmass[ilocal] = 1.0;
  s0[ilocal] = DBL_MAX;
  x0[ilocal][0] = x[ilocal][0];
  x0[ilocal][1] = x[ilocal][1];
  x0[ilocal][2] = x[ilocal][2];
}

/* ----------------------------------------------------------------------
   modify what AtomVec::data_atom() just unpacked
   or initialize other atom quantities
------------------------------------------------------------------------- */

void AtomVecLpm::data_atom_post(int ilocal)
{
  s0[ilocal] = DBL_MAX;
  x0[ilocal][0] = x[ilocal][0];
  x0[ilocal][1] = x[ilocal][1];
  x0[ilocal][2] = x[ilocal][2];

  if (rmass[ilocal] <= 0.0) error->one(FLERR, "Invalid mass in Atoms section of data file");
}

/* ----------------------------------------------------------------------
   assign an index to named atom property and return index
   return -1 if name is unknown to this atom style
------------------------------------------------------------------------- */

int AtomVecLpm::property_atom(const std::string &name)
{
  if (name == "s0") return 1;
  return -1;
}

/* ----------------------------------------------------------------------
   pack per-atom data into buf for ComputePropertyAtom
   index maps to data specific to this atom style
------------------------------------------------------------------------- */

void AtomVecLpm::pack_property_atom(int index, double *buf, int nvalues, int groupbit)
{
  int *mask = atom->mask;
  int nlocal = atom->nlocal;
  int n = 0;

  if (index == 0) {
    for (int i = 0; i < nlocal; i++) {
        buf[n] = 0.0;
      n += nvalues;
    }
  } else if (index == 1) {
    for (int i = 0; i < nlocal; i++) {
        buf[n] = s0[i];
      n += nvalues;
    }
  }
}
