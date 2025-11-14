"""
给定数据库文件，评估最优dp模型和mace模型的性能
1. 根据排序能力选择模型
"""

import os
import logging

cwd = os.getcwd()
logging.basicConfig(filename=os.path.join(cwd, 'warnings.log'),
                    level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logging.captureWarnings(True)
import shutil
from ase.db import connect
from pathlib import Path
from deepmd.calculator import DP
from mace.calculators import MACECalculator
import random
import numpy as np
import time


def calculator_select(workdir, db_path, dp_model_path, gpu_id):
    # 设置GPU
    if os.path.exists(workdir):
        shutil.rmtree(workdir)
    os.makedirs(workdir)
    # 读取数据库
    db = connect(db_path)
    new_db = connect(os.path.join(workdir, "selected.db"))
    if db.count() <= 1000:
        selected_ids = [i for i in range(1, db.count() + 1)]
    else:
        selected_ids = random.sample(range(1, db.count() + 1), 1000)
    for i in selected_ids:
        row = db.get(id=i)
        new_db.write(atoms=row.toatoms(), data=row.data, key_value_pairs=row.key_value_pairs)

    # 获取dp和dft能量
    formula_dict = {}
    dp_calculator = DP(model=dp_model_path)
    dp_energy = []
    dft_energy = []
    for row in new_db.select():
        dft_energy.append([row.id, row.data["energy"]])
        atoms = row.toatoms()
        formula = atoms.get_chemical_formula()
        if formula not in formula_dict:
            formula_dict[formula] = [row.id]
        else:
            formula_dict[formula].append(row.id)
        atoms.set_calculator(dp_calculator)
        dp_energy.append([row.id, atoms.get_potential_energy()])

    # 获取mace能量
    mace_energy = []
    script_directory = Path(__file__).parent
    model_path = os.path.join(script_directory, "mace-mpa-0-medium.model")
    mace_calculator = MACECalculator(model_path=model_path, device=f"cuda:{gpu_id}", default_dtype='float64')

    for row in new_db.select():
        atoms = row.toatoms()
        atoms.calc = mace_calculator
        energy = atoms.get_potential_energy()
        mace_energy.append([row.id, energy])

    # 数据转换为字典
    dft_info = {k: v for k, v in dft_energy}
    dp_info = {k: v for k, v in dp_energy}
    mace_info = {k: v for k, v in mace_energy}

    # 按照原子数分组
    groups = []
    for f, ids in formula_dict.items():
        if len(ids) > 1:
            groups.append(ids)
    results = []
    for group in groups:
        group = group[:256]
        true_energy = [[id, dft_info[id]] for id in group]
        true_energy.sort(key=lambda x: x[1])
        group = [i[0] for i in true_energy]
        dft_energy = [[i + 1, dft_info[id]] for i, id in enumerate(group)]
        dp_energy = [[i + 1, dp_info[id]] for i, id in enumerate(group)]
        mace_energy = [[i + 1, mace_info[id]] for i, id in enumerate(group)]
        dft_energy.sort(key=lambda x: x[1])
        dp_energy.sort(key=lambda x: x[1])
        mace_energy.sort(key=lambda x: x[1])
        dft_ranks = np.array([i[0] for i in dft_energy])
        dp_ranks = np.array([i[0] for i in dp_energy])
        mace_ranks = np.array([int(i[0]) for i in mace_energy])
        dp_dis = np.linalg.norm(dft_ranks - dp_ranks)
        mace_dis = np.linalg.norm(dft_ranks - mace_ranks)
        print("DP dist: ", dp_dis, "MACE dist: ", mace_dis)
        if dp_dis < mace_dis:
            results.append(-1)
        elif dp_dis == mace_dis:
            results.append(0)
        else:
            results.append(1)
    print(results)
    # 删除临时文件夹
    shutil.rmtree(workdir)
    # 返回结果
    if sum(results) < 0:
        return "DP"
    else:
        return "MACE"


if __name__ == "__main__":
    start = time.time()
    s = calculator_select(workdir="/home/cchen/cluster/hiccup/pes/ga/tmp",  # 临时文件夹
                          db_path="/home/cchen/cluster/ga0_sp.db",  # 数据库文件
                          dp_model_path="/home/cchen/test/iter2.pb",
                          gpu_id=5)
    print(s)
    if os.path.exists(os.path.join(cwd, 'warnings.log')):
        os.remove(os.path.join(cwd, 'warnings.log'))
    print("Time: ", time.time() - start)
