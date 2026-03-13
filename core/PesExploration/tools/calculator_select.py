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

def energy_rmse(y_pred, y_true):
    y_pred = np.asarray(y_pred, dtype=float)
    y_true = np.asarray(y_true, dtype=float)

    if y_pred.shape != y_true.shape:
        raise ValueError(f"Shape mismatch: {y_pred.shape} vs {y_true.shape}")

    return np.sqrt(np.mean((y_pred - y_true) ** 2))

def forces_rmse(pred_list, true_list):
    """
    pred_list, true_list:
        list of arrays, each array shape (n_atoms_i, 3)
    """

    total_sq_error = 0.0
    total_count = 0

    for f_pred, f_true in zip(pred_list, true_list):
        f_pred = np.asarray(f_pred, dtype=float)
        f_true = np.asarray(f_true, dtype=float)

        if f_pred.shape != f_true.shape:
            raise ValueError(f"Shape mismatch: {f_pred.shape} vs {f_true.shape}")

        diff2 = (f_pred - f_true) ** 2
        total_sq_error += diff2.sum()
        total_count += diff2.size  # n_atoms * 3

    return np.sqrt(total_sq_error / total_count)

def calculator_select(workdir, db_path, dp_model_path, gpu_id):
    # 设置GPU
    if os.path.exists(workdir):
        shutil.rmtree(workdir)
    os.makedirs(workdir)
    # 读取数据库
    db = connect(db_path)
    new_db = connect(os.path.join(workdir, "selected.db"))
    if db.count() <= 5000:
        selected_ids = [i for i in range(1, db.count() + 1)]
    else:
        selected_ids = random.sample(range(1, db.count() + 1), 5000)
    for i in selected_ids:
        row = db.get(id=i)
        new_db.write(atoms=row.toatoms(), data=row.data, key_value_pairs=row.key_value_pairs)

    # 获取dp和dft能量
    formula_dict = {}
    dp_calculator = DP(model=dp_model_path)
    dp_forces = []
    dft_forces = []
    for row in new_db.select():
        dft_forces.append(row.data["forces"])
        atoms = row.toatoms()
        formula = atoms.get_chemical_formula()
        if formula not in formula_dict:
            formula_dict[formula] = [row.id]
        else:
            formula_dict[formula].append(row.id)
        atoms.set_calculator(dp_calculator)
        dp_forces.append(atoms.get_forces())

    # 获取mace能量
    mace_forces = []
    script_directory = Path(__file__).parent
    model_path = os.path.join(script_directory, "mace-mpa-0-medium-float32.model")
    mace_calculator = MACECalculator(model_path=model_path, device=f"cuda:{gpu_id}", default_dtype='float32')

    for row in new_db.select():
        atoms = row.toatoms()
        atoms.calc = mace_calculator
        mace_forces.append(atoms.get_forces())

    # 计算rmse
    dp_rmse_forces = forces_rmse(dp_forces, dft_forces)
    mace_rmse_forces = forces_rmse(mace_forces, dft_forces)
    print(f"dp_rmse_forces: {dp_rmse_forces}, mace_rmse_forces: {mace_rmse_forces}")

    # 删除临时文件夹
    shutil.rmtree(workdir)
    # 返回结果
    if dp_rmse_forces - mace_rmse_forces <= 0:
        return "DP"
    else:
        return "MACE"


if __name__ == "__main__":
    start = time.time()
    s = calculator_select(workdir="/home/cchen/CuY/hiccup2/workdir/dp/nn1/tmp",  # 临时文件夹
                          db_path="/home/cchen/CuY/hiccup2/workdir/dp/nn1/merged.db",  # 数据库文件
                          dp_model_path="/home/cchen/CuY/hiccup2/workdir/dp/nn1/000/frozen_model.pb",
                          gpu_id=0)
    print(s)
    if os.path.exists(os.path.join(cwd, 'warnings.log')):
        os.remove(os.path.join(cwd, 'warnings.log'))
    print("Time: ", time.time() - start)
