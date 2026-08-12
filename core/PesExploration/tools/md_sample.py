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
    """Set environment variables for DP-based MD runs.

    Configures thread counts, GPU visibility, and logging level to ensure
    deterministic behavior across spawned worker processes.

    Args:
        cpu_only: if True, disable GPU (CPU-only inference).
        dp_intra: DP_INTRA_OP_PARALLELISM_THREADS.
        dp_inter: DP_INTER_OP_PARALLELISM_THREADS.
        blas_threads: thread count for OpenBLAS, OMP, MKL, NumExpr.
        gpu_id: GPU device index to expose.
    """
    print(f"Setting env for DP MD with gpu_id {gpu_id}")
    # deepmd_plugin_dir = "<YOUR_DEEPMD_PLUGIN_DIR>"
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
    """Configuration for a single MD run from a seed structure.

    Attributes:
        seed_id: unique identifier for the seed.
        temperature_K: target temperature in Kelvin.
        timestep_fs: MD timestep in femtoseconds.
        nsteps: total number of MD steps.
        friction_1_per_fs: Langevin friction coefficient in 1/fs.
        dump_interval: trajectory write interval (steps).
        rng_seed: optional random seed for velocity initialization.
    """
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
    """Run a Langevin MD simulation for a single seed structure.

    Each call spawns its own LAMMPS instance with a DP potential, initializes
    velocities at the target temperature, and writes a trajectory file.

    Args:
        atoms: ASE Atoms object (seed structure).
        type_map: list of element symbols matching the DP model.
        dp_model_path: path to the frozen DP model.
        cfg: SeedMDConfig with MD parameters.
        base_workdir: parent directory for this seed's output.
        cpu_only_inference: if True, run DP inference on CPU.
        gpu_id: GPU device index.

    Returns:
        Dict with seed_id, workdir, temperature, nsteps, and natoms.
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
        # These settings are allowed after box definition
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
    """Run MD simulations for multiple seed structures in parallel.

    Each seed gets its own process with a deterministic RNG seed. GPUs are
    assigned round-robin from *gpu_ids*.

    Args:
        atoms_list: list of ASE Atoms objects (seed structures).
        dp_model_path: path to the frozen DP model.
        type_map: list of element symbols matching the DP model.
        base_workdir: parent directory for all seed outputs.
        nproc: number of parallel processes.
        nsteps: total MD steps per seed.
        timestep_fs: MD timestep in femtoseconds.
        dump_interval: trajectory write interval (steps).
        cpu_only_inference: if True, run DP inference on CPU.
        base_rng_seed: base random seed (incremented per seed).
        temperature_K: target temperature in Kelvin.
        gpu_ids: list of GPU device IDs (cycled across seeds).

    Returns:
        List of result dicts from run_one_seed_md.
    """
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
    """Collect MD trajectories into a database with energies and forces.

    Reads all trajectory files under *md_dir*, recomputes energies and forces
    with the given DP model, and writes them to a database.

    Args:
        md_dir: directory containing seed_* subdirectories with md.traj files.
        model_path: path to the frozen DP model.
        gpu_id: GPU device index for inference.

    Returns:
        Path to the gathered trajectory database.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    md_traj_db_path = "<YOUR_MD_TRAJ_DB_PATH>"
    #md_traj_db_path = os.path.join(md_dir, "md_traj.db")
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

    # atoms_list = []
    # db = connect("<YOUR_TO_MD_DB_PATH>")
    # for row in db.select():
    #     atoms0 = row.toatoms()
    #     fix = [atom.index for atom in atoms0 if atom.position[2] < 2.5]
    #     atoms0.set_constraint(FixAtoms(indices=fix))
    #     atoms_list.append(atoms0)
    #
    dp_model = "<YOUR_DP_MODEL_PATH>"
    # results = run_md_parallel(
    #             atoms_list=atoms_list,
    #             dp_model_path=dp_model,
    #             type_map=['O', 'Cl', 'Cu'],
    #             base_workdir="<YOUR_MD_WORKDIR>",
    #             nproc=3,
    # one process per atoms seed, up to you
    #             nsteps=10000,       # 20 ps at dt=1fs
    #             timestep_fs=0.5,
    #             dump_interval=2,
    #             temperature_K=500.0,
    #             cpu_only_inference=False,
    #             gpu_ids=[5,6,0]
    #         )
    #
    # print("Done:", results)
    gather_md_traj(md_dir="<YOUR_MD_DIR>", model_path=dp_model, gpu_id=0)

