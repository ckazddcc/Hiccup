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
    """Generate random crystal seed structures using pyxtal.

    Supports 3D bulk and 0D cluster generation. Structures are validated by
    bond-length checks and optionally filtered by cell size for 2D systems.
    """

    def __init__(self,
                 elements,
                 target_composition,
                 dimension,
                 seeds_db,
                 seeds_num,
                 vacuum_layer_thickness=10):
        self.elements, self.target_composition = self.update_composition(elements, target_composition)
        self.dimension = dimension
        self.seeds_db = seeds_db
        self.seeds_num = seeds_num
        self.vacuum_layer_thickness = vacuum_layer_thickness

    @staticmethod
    def update_composition(elements, target_composition):
        """Reorder elements by atomic number and align compositions accordingly.

        Args:
            elements: list of element symbols.
            target_composition: list of composition lists, each parallel to *elements*.

        Returns:
            Tuple of (sorted_elements, reordered_target_composition).
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
        """Add vacuum padding around a structure based on dimensionality.

        Clusters get vacuum in all three directions; slabs/bulk get vacuum
        only along z.

        Args:
            atoms: ASE Atoms object.

        Returns:
            Atoms object with adjusted cell and centered positions.
        """
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
        """Check whether bond lengths in a structure are physically reasonable.

        Returns False if any pair of atoms is closer than 30% of the sum of
        their covalent radii.
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
        return IS_VALID  # Return validity flag

    def gen_random_seed(self, target_composition):
        """Generate a single random seed structure for the given composition.

        Uses pyxtal to generate a random crystal (3D) or cluster (0D), adds a
        vacuum layer, and validates bond lengths. Retries for up to 60 seconds.

        Args:
            target_composition: list of atom counts parallel to self.elements.

        Returns:
            Tuple of (list_of_atoms, chemical_formula_string).
        """
        seeds_atoms = []
        chemical_formula = "".join([f"{ele}{num}" for ele, num in zip(self.elements, target_composition)])
        time0 = time.time()
        while len(seeds_atoms) < 1:
            if time.time() - time0 > 60:
                return seeds_atoms, chemical_formula

            # Bulk
            if self.dimension == 3 or 2:
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
                        # uid = str(uuid.uuid4())[:16]
                        # self.seeds_db.write(c1, data={"formula": chemical_formula}, key_value_pairs={"uid": uid})
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
                        # uid = str(uuid.uuid4())[:16]
                        # self.seeds_db.write(c1, data={"formula": chemical_formula}, key_value_pairs={"uid": uid})
                except:
                    pass
        return seeds_atoms, chemical_formula


    def gen_seeds(self):
        """Generate random seed structures in parallel and save to database.

        Distributes generation across up to 16 processes. Each composition is
        attempted *seeds_num* times. Successfully generated structures are
        written to self.seeds_db.
        """
        start = time.time()
        futures = []
        seeds_num = int(self.seeds_num)
        seeds_collect = connect(self.seeds_db)
        with concurrent.futures.ProcessPoolExecutor(max_workers=16) as executor:
            for i in range(seeds_num):
                for c in self.target_composition:
                    futures.append(executor.submit(self.gen_random_seed, c))
            success_count = 0
            for future in tqdm(concurrent.futures.as_completed(futures),
                               total=len(futures),
                               disable=False,
                               desc='Running···',
                               colour='green',
                               ncols=90):
                seed_list, chemical_formula = future.result()
                if len(seed_list) > 0:
                    for seed in seed_list:
                        seeds_collect.write(seed, data={"formula": chemical_formula})
                    success_count += 1
                else:
                    pass
        print(f"Gen Random Seeds Success count: {success_count}")
        time_cost = time.time() - start
        print(f"Time cost: {time_cost:.2f}s")


if __name__ == '__main__':
    from ase.io import read
    from tools.mace_optimizer import seeds_optimizer

    dimension = 2
    substrate_pwd = "<YOUR_SUBSTRATE_PATH>"
    seeds_db_path = "<YOUR_SEEDS_DB_PATH>"
    gpu_ids = [4, 5, 6, 7]

    test = GenSeeds(elements=["O", "Cu", "Y"],
                    target_composition=[[7, 34, 23], [9, 55, 0], [23, 36, 5], [0, 49, 15],
                                        [12, 44, 8], [17, 32, 15], [19, 45, 0], [7, 41, 16]],
                    dimension=3,
                    seeds_db=seeds_db_path,
                    seeds_num=2)
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

    # Submit seed structure optimization
    Rand_seeds_opt = seeds_optimizer(seeds_db_path=seeds_db_path, gpus=gpu_ids)