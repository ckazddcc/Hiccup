import sys
model_path = '<MACE_MODEL_PATH>'
dimension = 0
opt_flag = True
opt_method = "BFGS"
ediffg = 0.1
nsw = 50
constrain_z = 0
gpu = sys.argv[1]
# end of snippet

import time
start = time.time()
import os
os.environ["CUDA_VISIBLE_DEVICES"] = gpu
import glob
from ase.constraints import ExpCellFilter
from ase.io import read, write
from ase.optimize import FIRE
from ase.optimize import BFGS
from ase.optimize.sciopt import SciPyFminCG
from ase.constraints import FixAtoms
from mace.calculators import MACECalculator
import shutil


cwd = os.getcwd()
model_name = os.path.basename(model_path)
_mace_model = os.path.join(cwd, model_name)
if not os.path.exists(_mace_model):
    shutil.copy(model_path, _mace_model)
calculator = MACECalculator(model_path=_mace_model, device='cuda:0', default_dtype='float64')
atoms = read("POSCAR")
atoms.calc = calculator


def bulk_relax(atoms):
    # Performs a variable-cell relaxation of the structure
    converged = False
    niter = 0
    while not converged and niter < 10:
        ecf = ExpCellFilter(atoms)
        dyn = FIRE(ecf, maxmove=0.2, logfile="bulk_opt.log", trajectory="bulk_opt_%d.traj" % niter)
        dyn.run(fmax=ediffg, steps=nsw)

        converged = dyn.converged()
        niter += 1

    dyn = FIRE(atoms, maxmove=0.2, logfile="bulk_opt.log", trajectory="bulk_opt.traj")
    dyn.run(fmax=ediffg, steps=nsw)

    e = atoms.get_potential_energy()
    f = atoms.get_forces()
    s = atoms.get_stress()
    return atoms, e, f, s


if opt_flag:
    if dimension == 3:
        try:
            atoms, energy, forces, stress = bulk_relax(atoms)
        except:
            energy = atoms.get_potential_energy()
    else:
        if dimension == 2:
            # Fix atoms below the average z position
            positions_z = atoms.get_scaled_positions()[2]
            cell = atoms.get_cell()[2][2]
            fix_z = (constrain_z + 1) / cell
            fix_indexs = [atom.index for atom in atoms if atom.scaled_position[2] < fix_z]
            c = FixAtoms(indices=[atom.index for atom in atoms if atom.index in fix_indexs])
            atoms.set_constraint(c)
        try:
            if opt_method == "FIRE":
                dyn = FIRE(atoms, maxmove=0.2, logfile="fire_opt.log", trajectory="fire_opt.traj")
            if opt_method == "BFGS":
                dyn = BFGS(atoms, logfile=None, trajectory=None)
            if opt_method == "CG":
                dyn = SciPyFminCG(atoms, logfile="cg.log", trajectory="cg.traj")
            dyn.run(fmax=ediffg, steps=nsw)
        except:
            pass
        energy = atoms.get_potential_energy()
        forces = atoms.get_forces()
else:
    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()

open('all_energy.txt', 'a').write("{}\n".format(energy))
open('energy.txt', 'w').write("{}".format(energy))
# write('POSCAR', atoms)

cost = time.time() - start
atoms_clean = atoms.copy()
atoms_clean.set_constraint()
write('CONTCAR', atoms_clean)
open('cost.txt', 'a').write("{}\n".format(cost))

