model_path = "/home/cchen/slab/hiccup/dp/nn2/000/frozen_model.pb"
dimension = 2
opt_flag = False
opt_method = "BFGS"
ediffg = 0.2
nsw = 200
# end of snippet

import os
import glob
from ase.constraints import ExpCellFilter
from ase.io import read, write
from ase.optimize import FIRE
from deepmd.calculator import DP
from ase.optimize import BFGS
from ase.optimize.sciopt import SciPyFminCG
from ase.constraints import FixAtoms


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


cwd = os.getcwd()
calc = DP(model=model_path)

atoms = read("POSCAR")
atoms.calc = calc
if opt_flag:
    if dimension == 3:
        try:
            atoms, energy, forces, stress = bulk_relax(atoms, calc)
        except:
            energy = atoms.get_potential_energy()
    else:
        if dimension == 2:
            # 固定z轴平均值以下原子
            positions_z = atoms.get_scaled_positions()[2]
            mean_z = positions_z.mean()
            fix_indexs = [atom.index for atom in atoms if atom.position[2] < mean_z]
            c = FixAtoms(indices=[atom.index for atom in atoms if atom.index in fix_indexs])
            atoms.set_constraint(c)
        try:
            if opt_method == "FIRE":
                dyn = FIRE(atoms, maxmove=0.2, logfile="fire_opt.log", trajectory="fire_opt.traj")
            if opt_method == "BFGS":
                dyn = BFGS(atoms, logfile="bfgs.log", trajectory="bfgs_%d.traj" % (len(glob("*.traj"))))
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
write('CONTCAR', atoms)
