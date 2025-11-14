import json
import os
import logging

cwd = os.getcwd()
logging.basicConfig(filename=os.path.join(cwd, 'warnings.log'),
                    level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logging.captureWarnings(True)
from ase.io import read, write
from ase.db import connect
import numpy as np
import random
import shutil
import math
import re
import time
from collections import Counter
from datetime import datetime, timedelta
from ase.data import atomic_numbers
from ase.build.tools import sort
from pathlib import Path
import subprocess
from ase.constraints import FixAtoms


class UspexSystem:
    def __init__(self,
                 elements,
                 target_composition,
                 dimension,
                 workdir,  # pes/ga/1
                 uspex_templates_dir,
                 generation_num,
                 pop_size,
                 ini_pop_size,
                 model_path=None,
                 calculator="MACE",
                 substrate_path=None,
                 random_seeds_path=None,
                 init_seeds_path=None,
                 multi_substrates=False,
                 opt_method="BFGS",
                 ediffg=0.2,
                 nsw=200,
                 gpu=0
                 ):
        """
        读取随机种子数据库
        产生Uspex运行输入文件、脚本
        """
        self.work_dir = workdir
        if not os.path.exists(workdir):
            os.makedirs(workdir)
        # 按照元素顺序更新元素和组成列表
        elem_dict = {elem: target_composition[i] for i, elem in enumerate(elements)}
        elem_sort = sorted(elem_dict.items(), key=lambda x: atomic_numbers[x[0]])
        elements = [ele for ele, _ in elem_sort]
        target_composition = [elem_dict[ele] for ele, _ in elem_sort]

        self.elements = elements
        self.target_composition = target_composition
        self.dimension = dimension
        chemformula = "".join(f"{ele}{num}" for ele, num in zip(elements, target_composition))
        unique_mark = f"{self.dimension}_{chemformula}"
        # 如果并行任务中设计多种基底，需要区分基底
        if multi_substrates:
            substrate_index = os.path.basename(substrate_path).split("_")[-1]
            unique_mark += f"_{substrate_index}"
        self.unique_mark = unique_mark
        # 创建compos工作目录
        self.compos_work_dir = os.path.join(workdir, f"{unique_mark}")
        self.template_dir = uspex_templates_dir
        self.substrate_path = substrate_path
        self.model_path = model_path
        self.gpu = gpu
        # GA设置
        self.generation_num = generation_num
        self.pop_size = pop_size
        self.ini_pop_size = ini_pop_size
        self.calculator = calculator
        self.opt_method = opt_method
        self.ediffg = ediffg
        self.nsw = nsw
        # 种子结构设置
        self.random_seeds_path = random_seeds_path
        self.init_seeds_path = init_seeds_path
        self.random_seeds_db = connect(random_seeds_path)
        if random_seeds_path:
            self.select_json = os.path.join(os.path.dirname(random_seeds_path), "selected.json")
            if os.path.exists(self.select_json):
                with open(self.select_json, "r") as f:
                    try:
                        self.selected_dict = json.load(f)
                    except json.JSONDecodeError:
                        self.selected_dict = {}
            else:
                self.selected_dict = {}
        else:
            self.selected_dict = {}

    @staticmethod
    def write_poscars(atoms_list, output_file):
        """
        将Atoms对象转化为POSCARS格式
        :param atoms_list: 选定的Atoms对象列表
        :param output_file: 输出文件POSCARS路径
        """
        gathered_poscars = []
        for i, atoms in enumerate(atoms_list):
            scaled_positions = atoms.get_scaled_positions()
            for j, pos in enumerate(scaled_positions):
                for k in range(3):
                    if pos[k] < 0:
                        pos[k] += 1
                    elif pos[k] > 1:
                        pos[k] -= 1
            atoms.set_scaled_positions(scaled_positions)
            # 按照元素序号排序
            atoms = sort(atoms, tags=atoms.get_atomic_numbers())
            tmp_file = os.path.join(os.path.dirname(output_file), f"POSCAR_{i}")
            # 清除原子的固定信息
            atoms_clean = atoms.copy()
            atoms_clean.set_constraint()
            # 通过ase的write函数转化为POSCAR格式并在第一行标记EA
            write(tmp_file, atoms_clean, direct=True, vasp5=True)
            with open(tmp_file, "r") as f:
                content = f.readlines()
            content[0] = f"EA{i + 1}\n"
            gathered_poscars.extend(content)
            os.remove(tmp_file)

        with open(output_file, "w") as f:
            f.writelines(gathered_poscars)

    def random_seeds_selection(self, num):
        """
        从随机种子中选取最稳定的结构作为初始种子，同时记录所选择的结构uid
        """
        target_compos = Counter(dict(zip(self.elements, self.target_composition)))
        info = []
        uidss = []
        for row in self.random_seeds_db.select():
            atoms = row.toatoms()
            compos = Counter(atoms.get_chemical_symbols())
            if compos == target_compos:
                uid = row.id
                # uid = row.get("uid", row.id)
                uidss.append(uid)
                try:
                    energy = row.data["energy"]
                    forces = row.data["forces"]
                    force_magnitudes = np.linalg.norm(forces, axis=1)
                    max_force_magnitude = np.max(force_magnitudes)
                    selected_times = self.selected_dict.get(uid, 0)
                    if max_force_magnitude < 0.1:
                        info.append([uid, energy, selected_times])
                except Exception as e:
                    continue

        if 0 < len(info) <= num:
            selected_uids = [i[0] for i in info]
        elif len(info) == 0:
            if num >= len(uidss):
                selected_uids = uidss
            else:
                selected_uids = random.sample(uidss, int(num))
        else:
            # selected_times 筛选
            info.sort(key=lambda x: x[2])
            _num = min(math.ceil(num * 2), len(info))
            selected_times = info[:_num]
            # energy 筛选
            selected_times = sorted(selected_times, key=lambda x: x[1])
            selected_times_energy = selected_times[:int(num)]
            selected_uids = [uid for uid, _, _ in selected_times_energy]

        self.selected_dict.update({uid: self.selected_dict.get(uid, 0) + 1 for uid in selected_uids})
        with open(self.select_json, "w") as f:
            json.dump(self.selected_dict, f)
        # selected_atoms = [self.random_seeds_db.get_atoms(uid=uid) for uid in selected_uids]
        selected_atoms = [self.random_seeds_db.get_atoms(uid) for uid in selected_uids]
        print(f"Selected {len(selected_atoms)} atoms from random seeds.")
        return selected_atoms

    def generate_input_txt(self, bash_path):
        """
        生成输入文件INPUT.txt
        适用于USPEX10.5及之前版本 新版本输入文件使用input.uspex
        """
        self.bash_path = str(bash_path) + f" {self.gpu}"
        atomtype_str = "  ".join(self.elements)
        tmp = {}
        if self.dimension == 2:
            sub = read(self.substrate_path)
            sub_composition = Counter(sub.get_chemical_symbols())
            for i, n in enumerate(self.target_composition):
                tmp[self.elements[i]] = n
            numspecies = []
            for ele in self.elements:
                numspecies.append(tmp[ele] - sub_composition.get(ele, 0))
            numspecies_str = "  ".join(map(str, numspecies))
        else:
            numspecies_str = "  ".join(map(str, self.target_composition))

        input_template = os.path.join(self.template_dir, f"TEMP_INPUT_{self.dimension}.txt")
        with open(input_template) as f:
            content = f.read()
        input_dir = os.path.join(self.compos_work_dir, "INPUT.txt")
        if self.dimension == 3:
            with open(input_dir, 'w') as f:
                f.write(content.format(numspecies_str, atomtype_str, self.pop_size, self.ini_pop_size,
                                       self.generation_num, self.bash_path))
        else:
            vacuum_size = 10
            with open(input_dir, 'w') as f:
                f.write(content.format(numspecies_str, atomtype_str, vacuum_size, self.pop_size,
                                       self.ini_pop_size, self.generation_num, self.bash_path))

    def renumber_EA(self, good_poscars_path):
        """
        对EA重新编号为123...
        """
        with open(good_poscars_path, 'r') as f:
            good_poscars = f.readlines()
        config_num = []
        for i in range(len(good_poscars)):
            line = good_poscars[i].split()
            if "EA" in line[0]:
                config_num.append(i)
        for i in range(len(config_num)):
            line = good_poscars[config_num[i]].split()
            line[0] = 'EA%d' % (i + 1)
            good_poscars[config_num[i]] = line[0] + '  ' + ' '.join(line[1:-1]) + '    ' + line[-1] + '\n'

        # 返回包含EA行在文件中第几行
        config_indices = [i for i, line in enumerate(good_poscars) if line.startswith("EA")]
        # 把EA重新编号123...
        for idx, line_idx in enumerate(config_indices, start=1):
            line = good_poscars[line_idx].split()
            line[0] = f'EA{idx}'
            good_poscars[line_idx] = f"{line[0]}  {'  '.join(line[1:-1])}    {line[-1]}\n"
        return good_poscars

    def create_uspex_input(self):
        """
        生成 USPEX 的输入文件和所需资源
        1.根据元素和组成，生成一个唯一的标识符 unique_mark
        2.如果有之前的计算结果直接调用最新的
        2.根据initial_seed_paths判断是使用指定的初始种子，还是进行random_seeds_selection
        3.产生各种uspex运行文件
        """
        if os.path.exists(self.compos_work_dir):
            shutil.rmtree(self.compos_work_dir)
        os.makedirs(self.compos_work_dir)
        seeds_dir = os.path.join(self.compos_work_dir, 'Seeds')
        specific_dir = os.path.join(self.compos_work_dir, 'Specific')
        if not os.path.exists(seeds_dir):
            os.makedirs(seeds_dir)
        if not os.path.exists(specific_dir):
            os.makedirs(specific_dir)

        # 如果是 2D
        if self.dimension == 2:
            shutil.copy(self.substrate_path, os.path.join(self.compos_work_dir, "POSCAR_SUBSTRATE"))

        # 生成POSCARS1：如果有指定的初始种子，直接使用，否则从随机种子中选取最稳定的结构作为初始种子
        selected_atoms = []
        if not self.init_seeds_path:
            self.init_seeds_path = self.random_seeds_path
            selected_atoms = self.random_seeds_selection(self.ini_pop_size * 0.1)
            POSCARS_1 = os.path.join(seeds_dir, 'POSCARS_1')
            self.write_poscars(selected_atoms, POSCARS_1)

        elif "POSCARS" in os.path.basename(self.init_seeds_path):
            # 重新编号
            good_poscars = self.renumber_EA(self.init_seeds_path)
            # 写入更新后的内容到 POSCARS_1 文件
            seeds_file = os.path.join(self.compos_work_dir, 'Seeds', 'POSCARS_1')
            with open(seeds_file, 'w') as file:
                file.writelines(good_poscars)

        elif os.path.basename(self.init_seeds_path).endswith(".db"):
            init_seeds_db = connect(self.init_seeds_path)
            target_compos = Counter(dict(zip(self.elements, self.target_composition)))
            for row in init_seeds_db.select():
                atoms = row.toatoms()
                compos = Counter(atoms.get_chemical_symbols())
                if compos == target_compos:
                    selected_atoms.append(row.toatoms())
            if len(selected_atoms) > self.ini_pop_size * 0.2:
                num = int(math.ceil(self.ini_pop_size * 0.1))
                selected_atoms = random.sample(selected_atoms, num)
            POSCARS_1 = os.path.join(seeds_dir, 'POSCARS_1')
            self.write_poscars(selected_atoms, POSCARS_1)

        # 产生POSCAR_2 3 4
        for i in range(2, self.generation_num + 1):
            seeds_num = math.ceil(0.1 * self.pop_size)
            if self.random_seeds_path:
                selected_atoms = self.random_seeds_selection(seeds_num)
            else:
                selected_atoms = []
            POSCARS_i = os.path.join(seeds_dir, f'POSCARS_{i}')
            self.write_poscars(selected_atoms, POSCARS_i)

        # 从目标目录复制POTCAR到Specific
        potcar_1 = os.path.join(self.template_dir, "POTCAR_1")
        if os.path.exists(potcar_1):
            shutil.copy(os.path.join(self.template_dir, "POTCAR_1"), os.path.join(specific_dir, "POTCAR_1"))
        else:
            for elem in self.elements:
                potcar_path = os.path.join(self.template_dir, f"POTCAR_{elem}")
                shutil.copy(potcar_path, os.path.join(specific_dir, f"POTCAR_{elem}"))
        shutil.copy(os.path.join(self.template_dir, "INCAR_1"), os.path.join(specific_dir, "INCAR_1"))

        # 产生INPUT.txt
        calc_tag = "mace" if self.calculator == "MACE" else "dp"
        bash_path = os.path.join(self.template_dir, f"run_{calc_tag}.sh")
        self.generate_input_txt(bash_path)

        # 修改run_dp/mace.sh中的.py文件路径
        nn_inf_path = os.path.join(self.template_dir, f"{calc_tag}_opt.py")
        with open(bash_path, "r") as f:
            content = f.readlines()
            for i, line in enumerate(content):
                if "cp" in line:
                    content[i] = f"  cp {nn_inf_path} .\n"
        with open(bash_path, "w") as f:
            f.writelines(content)

        # 修改dp/mace_opt.py中的路径
        constraint_z = 0
        if self.dimension == 2:
            input_dir = os.path.join(self.compos_work_dir, "INPUT.txt")
            with open(input_dir, "r") as f:
                content_input = f.readlines()
                for i, line in enumerate(content_input):
                    if "thicknessB" in line:
                        constraint_z = float(line.split(" ")[0])

        with open(nn_inf_path, "r") as f:
            content = f.readlines()
            for i, line in enumerate(content):
                # 修改dp_opt.py中的路径
                if calc_tag == "dp":
                    if "model_path =" in line and "#" not in line:
                        content[i] = f"model_path = \"{self.model_path}\"\n"
                # 修改mace_opt.py中的路径
                elif calc_tag == "mace":
                    if "model_path =" in line and "#" not in line:
                        script_directory = Path(__file__).parent
                        model_path = os.path.join(script_directory, f"tools/mace-mpa-0-medium.model")
                        content[i] = f"model_path = \"{model_path}\"\n"

                if "dimension =" in line and "#" not in line:
                    content[i] = f"dimension = {self.dimension}\n"
                if "constrain_z =" in line and "#" not in line:
                    content[i] = f"constrain_z = {constraint_z}\n"
                if "end of snippet" in line:
                    break

        with open(os.path.join(self.template_dir, f"{calc_tag}_opt.py"), "w") as f:
            f.writelines(content)
        print("USPEX input and related files have been created successfully.")

    def kill_uspex(self):
        name = self.compos_work_dir
        dir_name = os.path.dirname(self.compos_work_dir)
        new_name = os.path.join(dir_name, os.path.basename(name + "_"))
        os.rename(name, new_name)
        time.sleep(300)
        if os.path.exists(os.path.join(new_name, "CalcFold1")):
            backup_input = os.path.join(new_name, "BACKUP_INPUT.txt")
            shutil.copy(os.path.join(new_name, "INPUT.txt"), backup_input)
            os.remove(os.path.join(new_name, "INPUT.txt"))
            os.remove(os.path.join(new_name, "still_running"))
            for i in range(10):
                try:
                    shutil.rmtree(os.path.join(new_name, "CalcFold1"))
                    shutil.rmtree(os.path.join(new_name, "CalcFoldTemp"))
                    break
                except:
                    pass
        os.rename(new_name, name)

    def uspex_monitor(self):
        """
        监控USPEX的运行状态
        """
        STATE = "RUNNING"
        done_flag = os.path.join(self.compos_work_dir, "USPEX_IS_DONE")
        fail_falg = os.path.join(self.compos_work_dir, "USPEX_IS_FAILED")
        if os.path.exists(done_flag):
            STATE = "DONE"
            return STATE
        elif os.path.exists(fail_falg):
            STATE = "FAILED"
            return STATE
        else:
            output_file_path = os.path.join(self.compos_work_dir, f"uspex.log")
            if not os.path.exists(output_file_path):
                time.sleep(60)
            # 获取文件的最后修改时间
            last_modified_time = os.path.getmtime(output_file_path)
            # 将最后修改时间转换为 datetime 对象
            last_modified_datetime = datetime.fromtimestamp(last_modified_time)
            # 获取当前时间
            current_time = datetime.now()
            # 检查文件是否超过10分钟没有更新
            time_difference = current_time - last_modified_datetime
            if time_difference > timedelta(minutes=30):
                with open(fail_falg, "w") as f:
                    f.write("USPEX IS FAILED")
                STATE = "FAILED"
                self.kill_uspex()
                return STATE
            else:
                return STATE


# 对individuals文件进行处理，1）去重，写入gathered.db 2）根据能量和力的阈值进行筛选，写入filtered.db
def pick_individuals(individuals_path):
    """
    从.individuals文件中提取出去重后的结构
    """
    ids = []
    fits = []
    with open(individuals_path) as fp:
        # 跳过前两行
        fp.readline()
        fp.readline()
        while True:
            line = fp.readline()
            if not line:
                break
            ss = [_ for _ in re.split(r'\s+', line) if _ not in ['', '[', ']']]
            if len(ss) <= 9:
                continue
            idx = int(ss[1])
            volume = float(ss[-10])
            density = float(ss[-9])
            try:
                fit = float(ss[-8])
            except:
                fit = 0
            ids.append(idx)
            fits.append(str(volume) + "_" + str(density) + "_" + str(fit))
    # 去重
    unique_fits = set(fits)
    # 返回unique_fits的索引在原its中的位置
    unique_ids = [ids[fits.index(u)] for u in unique_fits]
    # 将字符串转化为整数
    unique_ids = [int(ii) - 1 for ii in unique_ids]
    return unique_ids


def write_to_db(work_dir, constraint_z=0):
    """
    将简单去重后的结构写入gathered.db文件
    """
    subdirectories = []
    for root, dirs, files in os.walk(work_dir):
        # 获取子目录名，只返回当前目录的子目录，避免递归深度遍历
        subdirectories.extend(dirs)
        break
    subdirectories = [int(i[7:]) for i in subdirectories if i.startswith("results")]
    if not subdirectories:
        print("No results directory found.")
        return False
    results_id = max(subdirectories)
    results_path = os.path.join(work_dir, f"results{results_id}")
    alls_db_path = os.path.join(work_dir, "gathered.db")
    # 若gathered.db文件存在则删除，否则生成一个空的gathered.db文件
    if os.path.exists(alls_db_path):
        os.remove(alls_db_path)
    else:
        alls_db = connect(alls_db_path)
        alls_db.count()
    gather_poscars_path = os.path.join(results_path, "gatheredPOSCARS")

    if os.path.exists(gather_poscars_path):
        with open(gather_poscars_path) as poscars:
            contents = poscars.readlines()
        num_poscar = sum(1 for line in contents if 'EA' in line)
        length_poscar = len(contents) // num_poscar
        unique_ids = pick_individuals(os.path.join(results_path, "Individuals"))
        all_db = connect(alls_db_path)
        for i in unique_ids:
            try:
                poscar_dir = os.path.join(results_path, "POSCAR")
                poscar_content = contents[i * length_poscar:(i + 1) * length_poscar]
                with open(poscar_dir, 'w') as poscar_file:
                    poscar_file.writelines(poscar_content)
                atoms = read(poscar_dir)
                # 添加原子的固定信息
                if constraint_z > 0:
                    fix_indexs = [atom.index for atom in atoms if atom.position[2] < constraint_z]
                    c = FixAtoms(indices=[atom.index for atom in atoms if atom.index in fix_indexs])
                    atoms.set_constraint(c)
                all_db.write(atoms)
            except:
                print(f"{i} Error")
                continue
        print(f"{work_dir}/gathered.db has been created successfully.")
    return alls_db_path


if __name__ == '__main__':
    # write_to_db("/home/cchen/CuY/round1/Cu5Y/ga/2_O3Cu52Y10", 6.5)
    merged_db_path = "/home/cchen/CuY/round1/Cu5Y/ga/2_O3Cu52Y10/gathered.db"

    if os.path.exists(os.path.join(cwd, 'warnings.log')):
        os.remove(os.path.join(cwd, 'warnings.log'))
