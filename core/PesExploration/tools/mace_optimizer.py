"""
mace优化器，用于优化随机种子结构。
生成的随机种子结构.db文件统一进行优化，得到优化后的随机种子结构，同时在data中记录Energy和Force信息。
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
    # 计算每个子列表的大小
    avg_len = len(data) // n
    remainder = len(data) % n
    result = []
    start = 0
    for i in range(n):
        # 计算每个子列表的长度，考虑余数
        end = start + avg_len + (1 if i < remainder else 0)
        result.append(data[start:end])
        start = end
    return result


def optimizer(model_path, seeds_db_path, seeds_ids, gpu_i):
    db = connect(seeds_db_path)
    calculator = MACECalculator(model_path=model_path, device=f'cuda:{gpu_i}', default_dtype='float64')
    results = []
    for id in seeds_ids:
        row = db.get(id=id)
        atoms = row.toatoms()
        atoms.calc = calculator
        dyn = BFGS(atoms, logfile=None, trajectory=None)
        dyn.run(fmax=0.1, steps=50)
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
    print("Start seeds optimizer...")
    mp.set_start_method("spawn", force=True)
    db = connect(seeds_db_path)
    ids = [row.id for row in db.select()]
    split_ids = split_list(ids, len(gpus))
    script_directory = Path(__file__).parent
    model_path = os.path.join(script_directory, "mace-mpa-0-medium.model")
    model_path_i = []
    cwd = os.getcwd()
    for i, gpu in enumerate(gpus):
        model_i = os.path.join(cwd, f"mace-mpa-0-medium_{gpu}.model")
        if not os.path.exists(model_i):
            shutil.copy(model_path, model_i)
            model_path_i.append(model_i)
    visible_devices = ",".join(str(g) for g in gpus)
    os.environ["CUDA_VISIBLE_DEVICES"] = visible_devices

    with mp.Pool(processes=len(gpus)) as pool:
        results = pool.starmap(optimizer,
                               [(model_path_i[i], seeds_db_path, split_ids[i], i) for i in range(len(gpus))])

    # 收集优化结果
    new_db_path = seeds_db_path.replace('.db', '_opt.db')
    if os.path.exists(new_db_path):
        os.remove(new_db_path)
    new_db = connect(new_db_path)
    for result in results:
        for atoms, data, kvp in result:
            new_db.write(atoms, data=data, key_value_pairs=kvp)
            print(data["energy"])

    # 删除临时模型文件
    for i in model_path_i:
        if os.path.exists(i):
            os.remove(i)
    print("Seeds optimizer finished.")
    return new_db_path


if __name__ == "__main__":
    seeds_db_path = "/home/cchen/CuY/test/Cu5Y/seeds.db"
    gpus = [1, 2, 3]
    start = time.time()
    seeds_optimizer(seeds_db_path, gpus)
    end = time.time()
    print("Time cost: ", end - start)
    if os.path.exists(os.path.join(cwd, 'warnings.log')):
        os.remove(os.path.join(cwd, 'warnings.log'))
