"""
MACE-based structure optimizer.

Optimizes random seed structures using a MACE foundation model. Energies and
forces are recorded in the output database for downstream use.
"""
import os
import logging
import shutil

cwd = os.getcwd()
logging.basicConfig(filename=os.path.join(cwd, 'warnings.log'),
                    level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logging.captureWarnings(True)
from ase.db import connect
from concurrent.futures import ProcessPoolExecutor
from mace.calculators import MACECalculator
from pathlib import Path
import time
from ase.optimize import BFGS
import torch
import gc
import multiprocessing as mp


def split_list(data, n):
    """Split a list into *n* roughly equal-sized sublists."""
    # Compute the size of each sublist
    avg_len = len(data) // n
    remainder = len(data) % n
    result = []
    start = 0
    for i in range(n):
        # Compute the length of each sublist, accounting for remainder
        end = start + avg_len + (1 if i < remainder else 0)
        result.append(data[start:end])
        start = end
    return result


def optimizer(model_path, seeds_db_path, seeds_ids, gpu_i):
    """Optimize a subset of seed structures on a single GPU.

    Args:
        model_path: path to the MACE model file.
        seeds_db_path: path to the database of seed structures.
        seeds_ids: list of row IDs to optimize.
        gpu_i: GPU device index for this process.

    Returns:
        List of [atoms, data, key_value_pairs] for each optimized structure.
    """
    db = connect(seeds_db_path)
    calculator = MACECalculator(model_path=model_path, device=f'cuda:{gpu_i}', default_dtype='float32')
    results = []
    for id in seeds_ids:
        row = db.get(id=id)
        atoms = row.toatoms()
        atoms.calc = calculator
        dyn = BFGS(atoms, logfile=None, trajectory=None)
        dyn.run(fmax=0.1, steps=100)
        energy = atoms.get_potential_energy()
        forces = atoms.get_forces()
        data = row.data
        data["energy"] = energy
        data["forces"] = forces
        kvp = row.key_value_pairs
        results.append([atoms, data, kvp])
    del calculator
    torch.cuda.empty_cache()
    gc.collect()
    return results


def seeds_optimizer(seeds_db_path, gpus):
    """Optimize all seed structures in parallel across multiple GPUs.

    Args:
        seeds_db_path: path to the database of seed structures.
        gpus: list of GPU device IDs.

    Returns:
        Path to the optimized structures database.
    """
    print("Start seeds optimizer...")
    mp.set_start_method("spawn", force=True)
    db = connect(seeds_db_path)
    ids = [row.id for row in db.select()]
    split_ids = split_list(ids, len(gpus))
    script_directory = Path(__file__).parent
    model_path = os.path.join(script_directory, "mace-mpa-0-medium-float32.model")
    model_path_i = []
    cwd = os.getcwd()
    for i, gpu in enumerate(gpus):
        model_i = os.path.join(cwd, f"mace-mpa-0-medium-float32_{gpu}.model")
        if not os.path.exists(model_i):
            shutil.copy(model_path, model_i)
            model_path_i.append(model_i)
    visible_devices = ",".join(str(g) for g in gpus)
    os.environ["CUDA_VISIBLE_DEVICES"] = visible_devices

    with mp.Pool(processes=len(gpus)) as pool:
        results = pool.starmap(optimizer,
                               [(model_path_i[i], seeds_db_path, split_ids[i], i) for i in range(len(gpus))])

    # Collect optimization results
    new_db_path = seeds_db_path.replace('.db', '_opt.db')
    if os.path.exists(new_db_path):
        os.remove(new_db_path)
    new_db = connect(new_db_path)
    for result in results:
        for atoms, data, kvp in result:
            new_db.write(atoms, data=data, key_value_pairs=kvp)
            print(data["energy"])

    # Remove temporary model files
    for i in model_path_i:
        if os.path.exists(i):
            os.remove(i)
    print("Seeds optimizer finished.")
    return new_db_path


if __name__ == "__main__":
    seeds_db_path = "<YOUR_SEEDS_DB_PATH>"
    gpus = [1, 2, 3]
    start = time.time()
    seeds_optimizer(seeds_db_path, gpus)
    end = time.time()
    print("Time cost: ", end - start)
    if os.path.exists(os.path.join(cwd, 'warnings.log')):
        os.remove(os.path.join(cwd, 'warnings.log'))
