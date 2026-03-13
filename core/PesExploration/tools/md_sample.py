import os
import multiprocessing as mp
from dataclasses import dataclass
from typing import Optional, Dict, Any, List
from itertools import cycle
import numpy as np
from pathlib import Path
from deepmd.calculator import DP

from ase.io.trajectory import Trajectory
from ase import units
from ase.io import write
from ase.calculators.lammpslib import LAMMPSlib
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary
from ase.md.langevin import Langevin



def _set_env_for_dp_md(
    cpu_only: bool = True,
    dp_intra: int = 4,
    dp_inter: int = 2,
    blas_threads: int = 1,
    gpu_id: int = 0,
):
    """
    Use explicit assignment instead of setdefault to make behavior deterministic
    across multiple runs / spawned workers.
    """
    print(f"Setting env for DP MD with gpu_id {gpu_id}")
    # deepmd_plugin_dir = "/home/cchen/apps/deepmd-kit/lib/deepmd_lmp"
    old = os.environ.get("LAMMPS_PLUGIN_PATH", "")
    print(old)
    # if old:
    #     # avoid duplicates
    #     paths = old.split(":")
    #     if deepmd_plugin_dir not in paths:
    #         os.environ["LAMMPS_PLUGIN_PATH"] = deepmd_plugin_dir + ":" + old
    # else:
    #     os.environ["LAMMPS_PLUGIN_PATH"] = deepmd_plugin_dir

    os.environ["DP_INTRA_OP_PARALLELISM_THREADS"] = str(dp_intra)
    os.environ["DP_INTER_OP_PARALLELISM_THREADS"] = str(dp_inter)

    os.environ["OPENBLAS_NUM_THREADS"] = str(blas_threads)
    os.environ["OMP_NUM_THREADS"] = str(blas_threads)
    os.environ["MKL_NUM_THREADS"] = str(blas_threads)
    os.environ["NUMEXPR_NUM_THREADS"] = str(blas_threads)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    if cpu_only:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


@dataclass
class SeedMDConfig:
    seed_id: int
    temperature_K: float = 300.0
    timestep_fs: float = 1.0
    nsteps: int = 50000
    friction_1_per_fs: float = 0.01
    dump_interval: int = 100
    rng_seed: Optional[int] = None


def run_one_seed_md(
    atoms,
    type_map: List[str],
    dp_model_path: str,
    cfg: SeedMDConfig,
    base_workdir: str,
    cpu_only_inference: bool = True,
    gpu_id: int = 0,
    ) -> Dict[str, Any]:
    """
    One process = one atoms seed = one LAMMPS instance.
    """
    _set_env_for_dp_md(cpu_only=cpu_only_inference, dp_intra=4, dp_inter=2, blas_threads=1, gpu_id=gpu_id)

    workdir = os.path.join(base_workdir, f"seed_{cfg.seed_id:03d}")
    os.makedirs(workdir, exist_ok=True)

    atoms = atoms.copy()

    lammps_header = [
        "units metal",
        "atom_style atomic",
        "boundary p p p",
        "atom_modify map array sort 0 0",
    ]

    lmpcmds = [
        # box 定义之后再设置这些是允许的
        "neighbor 2.0 bin",
        "neigh_modify every 1 delay 0 check yes",
        f"pair_style deepmd {dp_model_path}",
        "pair_coeff * * " + " ".join(type_map)
    ]

    calc = LAMMPSlib(
        lmpcmds=lmpcmds,
        lammps_header=lammps_header,
        log_file=os.path.join(workdir, "lammps.log"),
        keep_alive=True,
    )

    atoms.calc = calc

    # Initialize velocities
    if cfg.rng_seed is not None:
        rng = np.random.default_rng(int(cfg.rng_seed))
        MaxwellBoltzmannDistribution(atoms, temperature_K=cfg.temperature_K, rng=rng)
    else:
        MaxwellBoltzmannDistribution(atoms, temperature_K=cfg.temperature_K)

    Stationary(atoms)  # remove COM momentum

    # MD
    dt = cfg.timestep_fs * units.fs
    dyn = Langevin(
        atoms,
        dt,
        temperature_K=cfg.temperature_K,
        friction=cfg.friction_1_per_fs,  # ensure it's in 1/fs
    )

    traj_path = os.path.join(workdir, "md.traj")
    traj = Trajectory(traj_path, "w")

    # write initial snapshot
    traj.write(atoms.copy())

    # write snapshots during MD (copy to avoid reference issues)
    dyn.attach(lambda: traj.write(atoms.copy()), interval=cfg.dump_interval)

    dyn.run(cfg.nsteps)

    write(os.path.join(workdir, "final.cif"), atoms)
    traj.close()

    return {
        "seed_id": cfg.seed_id,
        "workdir": workdir,
        "temperature_K": cfg.temperature_K,
        "nsteps": cfg.nsteps,
        "natoms": len(atoms),
    }


def run_md_parallel(
    atoms_list: List,
    dp_model_path: str,
    type_map: List[str],
    base_workdir: str,
    nproc: int = 8,
    nsteps: int = 50000,
    timestep_fs: float = 1.0,
    dump_interval: int = 100,
    cpu_only_inference: bool = True,
    base_rng_seed: int = 12345,
    temperature_K: float = 300.0,
    gpu_ids: List[int] = None,
):
    os.makedirs(base_workdir, exist_ok=True)

    cfgs = []
    for i in range(len(atoms_list)):
        cfgs.append(
            SeedMDConfig(
                seed_id=i,
                temperature_K=temperature_K,
                timestep_fs=timestep_fs,
                nsteps=nsteps,
                dump_interval=dump_interval,
                rng_seed=base_rng_seed + i,
            )
        )

    ctx = mp.get_context("spawn")
    gpu_cycle = cycle(gpu_ids)
    args = [(atoms_list[i], type_map, dp_model_path, cfgs[i], base_workdir, cpu_only_inference, next(gpu_cycle)) for i in range(len(atoms_list))]

    with ctx.Pool(processes=nproc, maxtasksperchild=1) as pool:
        results = pool.starmap(run_one_seed_md, args)
    return results

def gather_md_traj(md_dir, model_path, gpu_id):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    md_traj_db_path = os.path.join(md_dir, "md_traj.db")
    md_traj_db = connect(md_traj_db_path)
    calc = DP(model=model_path)
    _root = Path(md_dir)
    seed_dirs = [root for root in _root.iterdir() if root.is_dir()]
    for seed_dir in seed_dirs:
        traj_path = os.path.join(seed_dir, "md.traj")
        traj = Trajectory(traj_path)
        for atoms in traj:
            atoms.calc = calc
            energy = atoms.get_potential_energy()
            forces = atoms.get_forces()
            md_traj_db.write(atoms, data={'fitness': energy, 'forces': forces}, key_value_pairs={})
    print( f"Total {len(md_traj_db)} atoms in {md_traj_db_path}")
    return md_traj_db_path



if __name__ == "__main__":
    from ase.build import bulk
    from ase.db import connect
    from ase.constraints import FixAtoms

    atoms_list = []
    db = connect("/home/cchen/CuY/hiccup2/s4_3/md/md3.db")
    for row in db.select():
        atoms0 = row.toatoms()
        # fix = [atom.index for atom in atoms0 if atom.position[2] < 2.5]
        # atoms0.set_constraint(FixAtoms(indices=fix))
        atoms_list.append(atoms0)

    dp_model = "/home/cchen/CuY/hiccup2/workdir/dp/nn8/002/frozen_model.pb"
    results = run_md_parallel(
                atoms_list=atoms_list,
                dp_model_path=dp_model,
                base_workdir="/home/cchen/CuY/hiccup3/s4_3/md/800",
                nproc=3,            # one process per atoms seed, up to you
                nsteps=20000,       # 20 ps at dt=1fs
                timestep_fs=0.5,
                dump_interval=2,
                temperature_K=800.0,
                cpu_only_inference=False,
                gpu_ids=[4,5]
            )

    print("Done:", results)

