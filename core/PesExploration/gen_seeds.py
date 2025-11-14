import uuid
from ase import Atom
from ase.db import connect
from ase.data import covalent_radii, atomic_numbers
from pyxtal import pyxtal
import random
import time
import concurrent.futures
from tqdm import tqdm
import multiprocessing


class GenSeeds:
    def __init__(self,
                 elements,
                 target_composition,
                 dimension,
                 seeds_db,
                 seeds_num,
                 vacuum_layer_thickness=10):
        self.elements, self.target_composition = self.update_composition(elements, target_composition)
        # 根据原子序数重排
        self.dimension = dimension
        self.seeds_db = connect(seeds_db)
        self.seeds_num = seeds_num
        self.vacuum_layer_thickness = vacuum_layer_thickness

    @staticmethod
    # 可以用矩阵改进算法
    def update_composition(elements, target_composition):
        """
        根据原子序数更新目标组分的元素顺序
        """
        ele_c = {}
        for i, ele in enumerate(elements):
            tmp = []
            for c in target_composition:
                tmp.append(c[i])
            ele_c[ele] = tmp
        new_elements = sorted(elements, key=lambda x: Atom(x).number)
        new_target_composition = []
        for i in range(len(target_composition)):
            tmp = []
            for ele in new_elements:
                tmp.append(ele_c[ele][i])
            new_target_composition.append(tmp)
        return new_elements, new_target_composition

    def add_vacuum_layer(self, atoms):
        thickness = self.vacuum_layer_thickness

        # cluster
        if self.dimension == 0:
            vacuum_layer_thickness = [thickness, thickness, thickness]
        # bulk
        else:
            vacuum_layer_thickness = [0.0, 0.0, thickness]

        atoms.center()
        positions = atoms.get_positions()

        all_directions = ["x", "y", "z"]
        cell = [0.0 for i in all_directions]

        for i, d in enumerate(all_directions):
            all_coords = [p[i] for p in positions]
            cell[i] = max(all_coords) - min(all_coords) + vacuum_layer_thickness[i]

        atoms.set_cell(cell)
        atoms.center()
        return atoms

    @staticmethod
    def seed_filter(atoms):
        """
        判断传入的Atoms对象键长是否合理
        """
        IS_VALID = True
        symbols = atoms.get_chemical_symbols()
        n = len(atoms)
        distances = atoms.get_all_distances(mic=True)
        for i in range(n):
            dis_i = distances[i, :]
            dis_i_sort = sorted([(i, dis) for i, dis in enumerate(dis_i)], key=lambda x: x[1])
            for j_dis in dis_i_sort[1:]:
                j, dis = j_dis
                r1 = covalent_radii[atomic_numbers[symbols[i]]]
                r2 = covalent_radii[atomic_numbers[symbols[j]]]
                if dis < (r1 + r2) * 0.3:
                    IS_VALID = False
                    break
            if not IS_VALID:
                break
        return IS_VALID  # 返回优化后的结构

    def gen_random_seed(self, target_composition):
        """
        生成单个随机种子结构
        """
        seeds_atoms = []
        chemical_formula = "".join([f"{ele}{num}" for ele, num in zip(self.elements, target_composition)])
        while len(seeds_atoms) < 1:
            # Bulk
            if self.dimension == 3:
                try:
                    syms = range(1, 231)
                    c1 = pyxtal()
                    c1.from_random(dim=3,
                                   group=random.choice(syms),
                                   species=self.elements,
                                   numIons=target_composition)
                    c1 = self.add_vacuum_layer(c1.to_ase())
                    valid = self.seed_filter(c1)
                    if valid:
                        seeds_atoms.append(c1)
                        uid = str(uuid.uuid4())[:16]
                        self.seeds_db.write(c1, data={"formula": chemical_formula}, key_value_pairs={"uid": uid})
                except:
                    pass
            # Cluster
            elif self.dimension == 0:
                try:
                    syms = range(1, 57)
                    c1 = pyxtal()
                    c1.from_random(dim=0,
                                   group=random.choice(syms),
                                   species=self.elements,
                                   numIons=target_composition)
                    c1 = self.add_vacuum_layer(c1.to_ase())
                    valid = self.seed_filter(c1)
                    if valid:
                        seeds_atoms.append(c1)
                        uid = str(uuid.uuid4())[:16]
                        self.seeds_db.write(c1, data={"formula": chemical_formula}, key_value_pairs={"uid": uid})
                except:
                    pass

        return

    def run_with_timeout(self, target_composition):
        timeout = 60
        retries = 5
        for attempt in range(retries):
            # 创建一个进程来运行函数A
            process = multiprocessing.Process(target=self.gen_random_seed, args=(target_composition,))
            process.start()
            # 等待指定的时间（timeout）
            process.join(timeout)
            # 如果函数A还没结束，则停止该进程并重试
            if process.is_alive():
                process.terminate()  # 终止进程
                process.join()  # 确保进程终止
            else:
                return True
            time.sleep(5)
        return False

    def gen_seeds(self):
        start = time.time()
        futures = []
        seeds_num = int(self.seeds_num)
        with concurrent.futures.ProcessPoolExecutor(max_workers=48) as executor:
            for i in range(seeds_num):
                for c in self.target_composition:
                    futures.append(executor.submit(self.run_with_timeout, c))
            success_count = 0
            for future in tqdm(concurrent.futures.as_completed(futures),
                               total=len(futures),
                               disable=False,
                               desc='Running···',
                               colour='green',
                               ncols=90):
                if future.result() is True:
                    success_count += 1
                    pass
                else:
                    pass
        print(f"Gen Random Seeds Success count: {success_count}")
        time_cost = time.time() - start
        print(f"Time cost: {time_cost:.2f}s")


if __name__ == '__main__':
    from ase.io import read
    from tools.mace_optimizer import seeds_optimizer

    dimension = 2
    substrate_pwd = "/home/cchen/CuY/gcga/POSCAR_SUBSTRATE"
    seeds_db_path = "/home/cchen/CuY/gcga/r7/seeds.db"
    gpu_ids = [1, 2, 4, 5, 6, 7]

    test = GenSeeds(elements=["O", "Cu", "Y"],
                    target_composition=[[1, 70, 10], [2, 70, 10], [3, 70, 10], [4, 70, 10],
                                        [5, 70, 10], [6, 70, 10], [7, 70, 10], [0, 70, 10]],
                    dimension=3,
                    seeds_db=seeds_db_path,
                    seeds_num=500)
    test.gen_seeds()

    if dimension == 2:
        sub = read(substrate_pwd)
        cell = sub.get_cell()
        cell_x = cell[0][0] * 1.4
        cell_y = cell[1][1] * 1.4
        cell_z = 19
        for row in connect(seeds_db_path).select():
            atoms = row.toatoms()
            seeds_cell = atoms.get_cell()
            if seeds_cell[0][0] > cell_x or seeds_cell[1][1] > cell_y or seeds_cell[2][2] > cell_z:
                connect(seeds_db_path).delete([row.id])
    print(f"Seeds DB: {connect(seeds_db_path).count()}")

    # 提交随机种子结构优化任务
    Rand_seeds_opt = seeds_optimizer(seeds_db_path=seeds_db_path, gpus=gpu_ids)
