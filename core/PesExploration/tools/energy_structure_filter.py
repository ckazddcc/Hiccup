import logging
import os
cwd = os.getcwd()
logging.basicConfig(filename=os.path.join(cwd, 'warnings.log'),
                    level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logging.captureWarnings(True)
from deepmd.calculator import DP
from mace.calculators import MACECalculator
from scipy.spatial.distance import euclidean
from dscribe.descriptors import MBTR
import numpy as np
import math
from ase.db import connect
from ase.io import write
import warnings
import random


def fp_mbtr(atoms):
    mbtr_k2 = MBTR(
        species=list(set(atoms.get_atomic_numbers())),
        geometry={"function": "distance"},
        grid={"min": 0, "max": 5, "n": 100, "sigma": 0.1},
        weighting={"function": "inverse_square", "r_cut": 4.0, "threshold": 1e-3},
        periodic=True,
        normalization="l2")
    with warnings.catch_warnings():
        warnings.filterwarnings(action="ignore", message=".*invalid value encountered in true_divide.*",
                                category=RuntimeWarning)
        k2 = mbtr_k2.create(atoms)
        k2[np.isnan(k2)] = 0
    return k2


def split_db(db_path, fold_name):
    """
    将db文件按照化学式分类，输出多个db文件
    :param db: 待分类的db文件
    :param fold_name: 输出文件夹
    """
    if not os.path.exists(fold_name):
        os.mkdir(fold_name)
    N2db = {}
    db = connect(db_path)
    for row in db.select():
        atoms = row.toatoms()
        n = atoms.get_chemical_formula()
        if N2db.get(n, None) is None:
            N2db[n] = connect('{}/{}.db'.format(fold_name, n))
            N2db[n].write(atoms, data=row.data, key_value_pairs=row.key_value_pairs)
        else:
            N2db[n].write(atoms, data=row.data, key_value_pairs=row.key_value_pairs)
    return


def energy_structure_filter(db_path,
                            best_model_path,
                            max_filter_ratio=0.8,
                            max_filter_num=100000,
                            similarity_threshold=0.95,
                            output_mode="delete"  # "delete" or "split"
                            ):
    """
    Filter structures based on energy and similarity.
    """
    gathered_db = connect(db_path)
    gathered1 = os.path.join(os.path.dirname(db_path), f"{os.path.basename(db_path)[:-3]}_1.db")
    gathered2 = os.path.join(os.path.dirname(db_path), f"{os.path.basename(db_path)[:-3]}_2.db")
    if os.path.exists(gathered1):
        os.remove(gathered1)
    if os.path.exists(gathered2):
        os.remove(gathered2)
    gathered_db_1 = connect(gathered1)
    gathered_db_2 = connect(gathered2)
    print(f"Total number of structures: {gathered_db.count()}")
    tmp_dir = os.path.join(os.path.dirname(db_path), "split")
    if not os.path.exists(tmp_dir):
        os.mkdir(tmp_dir)
    split_db(db_path, tmp_dir)

    # 计算能量和指纹
    if best_model_path.endswith(".pb") or best_model_path.endswith("checkpoint"):
        calculator = DP(model=best_model_path)
    elif best_model_path.endswith(".model"):
        calculator = MACECalculator(model_path=best_model_path, device='cuda:0', default_dtype='float64')
    filenames = os.listdir(tmp_dir)
    for filename in filenames:
        energy_info = {}
        fp_dict = {}
        db_path_i = os.path.join(tmp_dir, filename)
        db_i = connect(db_path_i)
        # if db_i.count() > 1:
        _similarity_threshold = similarity_threshold
        for row in db_i.select():
            atoms = row.toatoms()
            atoms.set_calculator(calculator)
            energy = atoms.get_potential_energy()
            energy_info[row.id] = energy
            fp = fp_mbtr(atoms)
            fp_dict[row.id] = fp
        # 根据相似度对结构进行初筛，能量非常不稳定的10%的结构
        sorted_energy_info = sorted(energy_info.items(), key=lambda x: x[1])
        remove_num = math.floor(len(sorted_energy_info) * 0.1)
        remove_ids = [i[0] for i in sorted_energy_info[-remove_num:]]
        # 筛选结构数量
        x = math.floor(max_filter_num * (db_i.count() / gathered_db.count()))
        filter_num = min(x, math.floor(db_i.count() * max_filter_ratio))
        filter_num = max(filter_num, 1)
        print("filter_num: ", filter_num)
        all_ids = [i[0] for i in sorted_energy_info if i[0] not in remove_ids]
        selected_ids = all_ids
        count = 0
        while len(selected_ids) >= filter_num and count <= 5:
            for i in range(len(all_ids) - 1):
                if all_ids[i] in remove_ids:
                    continue
                for j in range(i + 1, len(all_ids)):
                    fp1 = fp_dict[all_ids[i]]
                    fp2 = fp_dict[all_ids[j]]
                    similarity = 1 - euclidean(fp1, fp2)
                    if similarity > _similarity_threshold:
                        remove_ids.append(all_ids[i])
                        break
            selected_ids = [i for i in all_ids if i not in remove_ids]
            if len(selected_ids) == 0:
                break
            print(f"Similarity threshold: {_similarity_threshold}")
            print(f"Selected number of structures: {len(selected_ids)}")
            _similarity_threshold -= 0.01
            count += 1
        if len(selected_ids) > filter_num:
            selected_ids = selected_ids[:filter_num]

        new_db_path_1 = os.path.join(tmp_dir, f"{os.path.basename(db_path_i)[:-3]}_1.db")
        new_db_path_2 = os.path.join(tmp_dir, f"{os.path.basename(db_path_i)[:-3]}_2.db")
        if os.path.exists(new_db_path_1):
            os.remove(new_db_path_1)
        if os.path.exists(new_db_path_2):
            os.remove(new_db_path_2)
        new_db_1 = connect(new_db_path_1)
        new_db_2 = connect(new_db_path_2)
        for row in db_i.select():
            atoms = row.toatoms()
            data = row.data
            kvp = row.key_value_pairs
            if row.id in selected_ids:
                new_db_1.write(atoms, data=data, key_value_pairs=kvp)
            else:
                new_db_2.write(atoms, data=data, key_value_pairs=kvp)

    for filename in os.listdir(tmp_dir):
        if filename.endswith("_1.db"):
            db_path1 = os.path.join(tmp_dir, filename)
            db1 = connect(db_path1)
            for row in db1.select():
                atoms = row.toatoms()
                data = row.data
                kvp = row.key_value_pairs
                gathered_db_1.write(atoms, data=data, key_value_pairs=kvp)
        elif filename.endswith("_2.db"):
            db_path2 = os.path.join(tmp_dir, filename)
            db2 = connect(db_path2)
            for row in db2.select():
                atoms = row.toatoms()
                data = row.data
                kvp = row.key_value_pairs
                gathered_db_2.write(atoms, data=data, key_value_pairs=kvp)

    # 保证1和2的数量都不为0
    if gathered_db_1.count() == 0:
        for row in gathered_db.select():
            atoms = row.toatoms()
            data = row.data
            kvp = row.key_value_pairs
            gathered_db_1.write(atoms, data=data, key_value_pairs=kvp)
            break
    if gathered_db_2.count() == 0:
        for row in gathered_db.select():
            atoms = row.toatoms()
            data = row.data
            kvp = row.key_value_pairs
            gathered_db_2.write(atoms, data=data, key_value_pairs=kvp)
            break

    # 删除临时文件夹
    count1 = gathered_db_1.count()
    count2 = gathered_db_2.count()
    os.system(f"rm -r {tmp_dir}")
    if output_mode == "delete":
        os.remove(gathered2)
        print(db_path)
        os.remove(db_path)
        os.rename(gathered1, db_path)
    print(f"Total number of structures after filtering: {count1}/{count1 + count2}")
    return


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    cs = [
          "/home/cchen/CuY/CuM/test/gathered.db",
    ]
          # "/home/cchen/CuY/gcga/r5_0.20/O2Cu72Y8.db"]
    # alls_db = connect("/home/cchen/CuY/test/ga/alls.db")
    for c in cs:
        energy_structure_filter(db_path=c,
                            best_model_path="/home/cchen/CuY/gcga/frozen_model.pb",
                            max_filter_ratio=0.80,
                            max_filter_num=5000,
                            similarity_threshold=0.98,
                            output_mode="split")
        # db = connect(f"/home/cchen/CuY/test/ga/{c}/gathered_1.db")
        # print(f"{c} Total number of structures: {db.count()}")
        # for row in db.select():
        #     atoms = row.toatoms()
        #     data = row.data
        #     kvp = row.key_value_pairs
        #     alls_db.write(atoms, data=data, key_value_pairs=kvp)

    if os.path.exists(os.path.join(cwd, 'warnings.log')):
        os.remove(os.path.join(cwd, 'warnings.log'))
