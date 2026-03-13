import time
import math
import glob
import os
from deepmd.calculator import DP
from ase.db import connect
import numpy as np
from ase.constraints import FixAtoms
from ase.neighborlist import NeighborList, natural_cutoffs
from ase import Atoms
from PesExploration.tools.process_layers import process_layers
# from .tools.process_layers import process_layers
import random


class NNDeviation:
    def __init__(self,
                 model_dir,
                 ga_db_path,
                 force_err_lower=0.1,
                 force_err_upper=0.2,
                 type="slab",
                 lcs_radius=5.0,
                 lcs_layers_num=3,
                 vaccum_thickness=[10, 10, 10]
                 ):
        self.model_dir = model_dir
        self.db = ga_db_path
        self.workdir = os.path.dirname(self.db)
        self.force_err_lower = force_err_lower
        self.force_err_upper = force_err_upper
        self.candidates_path = os.path.join(self.workdir, "candidates.db")
        self.failed_path = os.path.join(self.workdir, "failed.db")
        self.accurate_path = os.path.join(self.workdir, "accurate.db")
        self.dev_results = {}

        self.type = type
        if self.type == "slab":
            self.lcs_layers_num = lcs_layers_num
            self.vaccum_thickness = [0, 0, vaccum_thickness[2]]
        else:
            self.lcs_radius = lcs_radius
            self.vaccum_thickness = vaccum_thickness

    def get_calculators(self):
        calculators = []
        pbs = os.path.join(self.model_dir, "*/frozen_model.pb")
        mdoel_paths = glob.glob(pbs)
        for pb in mdoel_paths:
            calc = DP(model=pb)
            calculators.append(calc)
        return calculators

    def get_max_deviation(self, id, nns_force_dict):
        forces = nns_force_dict[id]
        mean_force = sum(forces) / len(forces)
        num = len(forces[0])
        f_std = 0.0
        for f in forces:
            diff_f = f - mean_force
            f_std += (diff_f ** 2).sum(axis=1)

        # 统计所有原子受力的标准差
        f_std = (f_std / num) ** 0.5
        max_force_dev = f_std.max()
        f_std = f_std.tolist()
        wrong_atoms = [i for i in range(len(f_std)) if self.force_err_lower < f_std[i]]
        wrong_list = [f_std[i] for i in range(len(f_std)) if self.force_err_lower < f_std[i]]
        return max_force_dev, wrong_atoms, wrong_list

    def get_deviation(self):
        """
        判断wrong_atoms, 并将数据分为accurate, candidates, failed
        """
        if os.path.exists(self.candidates_path):
            os.remove(self.candidates_path)
        if os.path.exists(self.failed_path):
            os.remove(self.failed_path)
        if os.path.exists(self.accurate_path):
            os.remove(self.accurate_path)
        db = connect(self.db)
        accurate_ids = []
        candidates_ids = []
        failed_ids = []

        tmp_data = {}
        tmp_failed_data = []
        nns_force_dict = {}
        calculators = self.get_calculators()
        for i, calc in enumerate(calculators):
            for row in db.select():
                atoms = row.toatoms()
                atoms.set_calculator(calc)
                force = atoms.get_forces(apply_constraint=False)
                nns_force_dict[row.id] = nns_force_dict.get(row.id, [])
                nns_force_dict[row.id].append(force)

        for row in db.select():
            max_force_dev, wrong_atoms, wrong_list = self.get_max_deviation(row.id, nns_force_dict)
            print(f"ID: {row.id}, Max Force Deviation: {max_force_dev}")
            tmp_data[row.id] = {"dev": max_force_dev, "wrong_atoms": wrong_atoms, "wrong_list": wrong_list}
            if max_force_dev < self.force_err_lower:
                accurate_ids.append(row.id)
            elif self.force_err_lower <= max_force_dev <= self.force_err_upper:
                candidates_ids.append(row.id)
            else:
                tmp_failed_data.append((row.id, max_force_dev))
                failed_ids.append(row.id)

        total = db.count()
        devs_data = {"Accurate": [len(accurate_ids), round(len(accurate_ids) / total, 2)],
                     "Candidate": [len(candidates_ids), round(len(candidates_ids) / total, 2)],
                     "Failed": [len(failed_ids), round(len(failed_ids) / total, 2)]}
        print(devs_data)
        self.dev_results = devs_data
        # 前期NN模型训练不足，导致合格数据不足，需要补充数据
        min_num = min(math.ceil(total * 0.5), 1)
        if len(candidates_ids) < min_num:
            supply_num = min_num - len(candidates_ids)
            # 补充failed数据
            if supply_num >= len(failed_ids):
                candidates_ids.extend(failed_ids)
            else:
                tmp_failed_data = sorted(tmp_failed_data, key=lambda x: x[1])
                candidates_ids.extend([i[0] for i in tmp_failed_data[:supply_num]])

        if len(candidates_ids) < min_num:
            random_select = random.sample(accurate_ids, min_num - len(candidates_ids))
            for r in random_select:
                candidates_ids.append(r)
                accurate_ids.remove(r)

        accurate_db = connect(self.accurate_path)
        candidate_db = connect(self.candidates_path)
        failed_db = connect(self.failed_path)

        for row in db.select():
            if row.id in accurate_ids:
                accurate_db.write(row.toatoms(), data=tmp_data[row.id], key_value_pairs=row.key_value_pairs)
            elif row.id in candidates_ids:
                candidate_db.write(row.toatoms(), data=tmp_data[row.id], key_value_pairs=row.key_value_pairs)
            else:
                failed_db.write(row.toatoms(), data=tmp_data[row.id], key_value_pairs=row.key_value_pairs)
        print("accurate:", len(accurate_ids), "candidates:", len(candidates_ids), "failed:", len(failed_ids))
        print("The deviation data has been successfully saved !")
        return

    @staticmethod
    def adjust_vacuum_layer(atoms, vacuum_thickness):
        atoms.center()
        positions = atoms.get_positions()
        cell = atoms.get_cell()
        for i, vacm in enumerate(vacuum_thickness):
            if vacm:
                all_coords = [p[i] for p in positions]
                cell[i][i] = max(all_coords) - min(all_coords) + vacm
        atoms.set_cell(cell)
        atoms.center()
        return atoms

    @staticmethod
    def remove_atoms(atoms, remove_index):
        if len(remove_index) > 0:
            atoms.center()
            positions = atoms.get_positions()
            cell = atoms.get_cell()
            all_coords = [positions[r][2] for r in remove_index]
            remove_thickness = max(all_coords) - min(all_coords)
            cell[2][2] = cell[2][2] - remove_thickness
            atoms.set_cell(cell)
        new_atoms = atoms[[atom.index for atom in atoms if atom.index not in remove_index]]
        atoms.center()
        return new_atoms

    @staticmethod
    def get_coordination_num_list(atoms):
        coordination_num_list = []
        nc = np.array(natural_cutoffs(atoms))
        nl = NeighborList(nc, bothways=True)
        nl.update(atoms)
        for i in range(len(atoms)):
            indices = [j for j in list(set(nl.get_neighbors(i)[0])) if j != i]
            coordination_num_list.append(len(indices))
        return coordination_num_list

    def lcs_layer(self, atoms, wrong_atoms, layer_num=3):
        """
        LCS for slab
        """
        cluster_dicts = process_layers(atoms, layer_num=layer_num, substrate_path=None)
        wrong_atoms_z_pos = [(w, atoms[w].position[2]) for w in wrong_atoms]
        wrong_atoms_z_pos = sorted(wrong_atoms_z_pos, key=lambda x: x[1])
        lowest_wrong_atom = wrong_atoms_z_pos[0][0]
        lowest_wrong_cluster = -1
        for key, value in cluster_dicts.items():
            if lowest_wrong_atom in value:
                lowest_wrong_cluster = key
                break

        remove_indexs = []
        if lowest_wrong_cluster <= 1:
            fix_indexs = cluster_dicts[0]
        else:
            fix_indexs = cluster_dicts[lowest_wrong_cluster - 1]
            for idx in range(lowest_wrong_cluster - 1):
                remove_indexs.extend(cluster_dicts[idx])

        f = FixAtoms(indices=[atom.index for atom in atoms if atom.index in fix_indexs])
        atoms.set_constraint(f)
        atoms = self.remove_atoms(atoms, remove_indexs)
        return atoms

    def lcs_cluster(self, atoms, wrong_atoms, radius=5.0):
        """
        LCS for large NP
        """
        atoms_list = []
        cn_list = self.get_coordination_num_list(atoms)
        wrong_ids = wrong_atoms.copy()
        while wrong_ids:
            for w in wrong_ids:
                distances = atoms.get_distances(w, indices=list(range(len(atoms))), mic=True)
                selected_index = [i for i, dis in enumerate(distances) if dis <= radius]
                selected_index.sort()
                ats = Atoms(atoms[selected_index])
                # 固定边界原子，判断方法优化
                new_cn_list = self.get_coordination_num_list(ats)
                bondary_atoms = [i for i, id in enumerate(selected_index) if (new_cn_list[i] - cn_list[id]) < 1]
                f = FixAtoms(indices=[atom.index for atom in atoms if atom.index in bondary_atoms])
                ats.set_constraint(f)
                ats = self.adjust_vacuum_layer(ats, vacuum_thickness=self.vaccum_thickness)
                atoms_list.append(ats)
                for i in selected_index:
                    if i in wrong_ids:
                        wrong_ids.remove(i)
        return atoms_list

    def lcs_process(self, db_path):
        db = connect(db_path)
        new_db_name = db_path[:-3] + "_lcs.db"
        new_db = connect(new_db_name)
        for row in db.select():
            atoms = row.toatoms()
            data = row.data
            wrong_atoms = data["wrong_atoms"]
            if self.type == "slab":
                try:
                    atoms = self.lcs_layer(atoms, wrong_atoms, self.lcs_layers_num)
                    atoms = self.adjust_vacuum_layer(atoms, self.vaccum_thickness)
                except:
                    print(f"ID: {row.id} failed to process")
                new_db.write(atoms, data=data)
            else:
                try:
                    atomss = self.lcs_cluster(atoms, wrong_atoms, self.lcs_radius)
                except:
                    atomss = [atoms]
                    print(f"ID: {row.id} failed to process")
                for ats in atomss:
                    new_db.write(ats, data=data)
        os.remove(db_path)
        os.rename(new_db_name, db_path)
        print("The LCS data has been successfully saved !")
        new_db = connect(db_path)
        return new_db.count()


if __name__ == '__main__':
    start = time.time()
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    nn_dev = NNDeviation(model_dir="/home/cchen/CuY/hiccup2/workdir/dp/nn7",
                         ga_db_path="/home/cchen/CuY/hiccup2/workdir/pes/ga/ga7/alls.db",
                         force_err_lower=0.05,
                         force_err_upper=0.2,
                         type="slab",
                         lcs_layers_num=3,
                         lcs_radius=5.0,
                         vaccum_thickness=[10, 10, 10]
                         )
    nn_dev.get_deviation()
    # nn_dev.lcs_process("/home/cchen/slab/hiccup/pes/ga/ga1/candidates.db")
    print("Time:", time.time() - start)
    # nn_dev.lcs_process("/home/ubuntu/PycharmProjects/Train_NN/tmp/workdir/pes/ga/3/candidates.db")
    # 可视化lcs结构
    # t = connect("/home/ubuntu/PycharmProjects/Train_NN/tmp/workdir/pes/ga/3/candidates_lcs.db")
    # for row in t.select():
    #     a = row.toatoms()
    #     write(f"/home/ubuntu/PycharmProjects/Train_NN/tmp/workdir/pes/ga/3/0/{row.id}.cif", a)
