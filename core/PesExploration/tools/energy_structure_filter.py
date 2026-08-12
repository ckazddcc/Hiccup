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
from ase.atoms import Atoms


def fp_mbtr(atoms, dimension, constraint_z):
    """Compute the MBTR fingerprint of a structure.

    For 2D systems, atoms below *constraint_z* (substrate) are excluded
    from the descriptor so that only the active region is represented.

    Args:
        atoms: ASE Atoms object.
        dimension: 2 for slab/surface systems, otherwise full structure.
        constraint_z: z-coordinate threshold below which atoms are excluded.

    Returns:
        1-D list concatenating k2 and k3 MBTR components.
    """
    # atoms_c = atoms.copy()
    periodic = True
    normalization = 'l2_each'
    if dimension == 2:
        fix_index = [atom.index for atom in atoms if atom.position[2] < constraint_z]
        atoms_c = Atoms([atom for atom in atoms.copy() if atom.index not in fix_index])
        atoms_c.set_pbc(atoms.pbc)
        atoms_c.set_cell(atoms.cell)
    else:
        atoms_c = atoms.copy()

    mbtr = MBTR(
        species=list(set(atoms_c.get_atomic_numbers())),
        k2={
            "geometry": {"function": "distance"},
            "grid": {"min": 0.0, "max": 5.0, "sigma": 0.1, "n": 100},
            "weighting": {"function": "inverse_square", "r_cut": 5.0, "scale": 0.5, "threshold": 1e-3},
        },
        k3={
            "geometry": {"function": "cosine"},
            "grid": {"min": -1.0, "max": 1.0, "sigma": 0.1, "n": 100},
            "weighting": {"function": "exp", "r_cut": 4.0, "scale": 0.3, "threshold": 1e-3},
        },
        periodic=periodic,
        sparse=False,
        flatten=False,
        normalization=normalization
    )
    with warnings.catch_warnings():
        warnings.filterwarnings(action="ignore", message=".*invalid value encountered in true_divide.*",
                                category=RuntimeWarning)
        mbtr_fp = mbtr.create(system=atoms_c, n_jobs=1, only_physical_cores=False)
    k2 = np.sum(mbtr_fp["k2"], axis=(0, 1))
    k3 = np.sum(mbtr_fp["k3"], axis=(0, 1, 2))
    # Set all NaN values in k2, k3 to 0
    k2[np.isnan(k2)] = 0
    k3[np.isnan(k3)] = 0
    return np.concatenate((k2, k3)).tolist()


def split_db(db_path, fold_name):
    """Split a database into multiple files grouped by chemical formula.

    Args:
        db_path: path to the source ASE database.
        fold_name: output directory for the split database files.
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
                            dimension=0,
                            constraint_z=3.0,
                            max_filter_ratio=0.8,
                            max_filter_num=100000,
                            similarity_threshold=0.95,
                            output_mode="delete"  # "delete" or "split"
                            ):
    """Filter structures by energy and structural similarity.

    Structures are split by chemical formula. For each group, the top 10%
    highest-energy structures are removed, then farthest-point sampling based
    on MBTR fingerprints removes redundant structures until the target count
    is reached.

    Args:
        db_path: path to the source ASE database.
        dimension: 2 for slab/surface systems (substrate excluded from
            fingerprint), 0 for bulk.
        constraint_z: z-coordinate threshold for substrate exclusion.
        max_filter_ratio: maximum fraction of structures to retain per group.
        max_filter_num: global cap on retained structures, distributed
            proportionally across groups.
        similarity_threshold: initial MBTR similarity threshold for
            deduplication (decreased iteratively if too few survive).
        output_mode: "delete" to replace the source database, "split" to
            keep both retained and removed sets.
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

    # Compute energies and fingerprints
    filenames = os.listdir(tmp_dir)
    for filename in filenames:
        energy_info = {}
        fp_dict = {}
        db_path_i = os.path.join(tmp_dir, filename)
        db_i = connect(db_path_i)
        _similarity_threshold = similarity_threshold
        for row in db_i.select():
            atoms = row.toatoms()
            data = row.data
            if "fitness" not in data.keys():
                if "energy" in data.keys():
                    energy = row.data['energy']
                else:
                    try:
                        energy = atoms.get_potential_energy()
                    except:
                        energy = 0
                        warnings.warn(
                            f"Failed to calculate energy for structure {row.id}; set energy=0.0",
                            category=RuntimeWarning,
                            stacklevel=2
                        )
            else:
                energy = data["fitness"]
            energy_info[row.id] = energy
            fp = fp_mbtr(atoms, dimension, constraint_z)
            fp_dict[row.id] = fp
        # Initial screening by similarity; remove top 10% highest-energy structures
        sorted_energy_info = sorted(energy_info.items(), key=lambda x: x[1])
        remove_num = math.floor(len(sorted_energy_info) * 0.1)
        remove_ids = [i[0] for i in sorted_energy_info[-remove_num:]]
        # Determine number of structures to keep
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

    # Ensure both groups have at least one structure
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

    # Remove temporary directory
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
    import os
    from pathlib import Path

    def find_db_files(root_dir):
        root = Path(root_dir)
        return [root for root in root.iterdir() if root.is_dir()]

    energy_structure_filter(db_path="<YOUR_CANDIDATES_DB_PATH>",
                            max_filter_ratio=0.80,
                            max_filter_num=60,
                            similarity_threshold=0.95,
                            output_mode="split")

    # dirs = find_db_files("<YOUR_GA_DIR_PATH>")
    # to_md_db = connect("<YOUR_TO_MD_DB_PATH>")
    # for dir in dirs:
    #     db = connect(os.path.join(dir, "gathered.db"))
    #     print(f"{dir} Total number of structures: {db.count()}")
    #     _infos = []
    #     to_md_atoms = []
    #     for row in db.select():
    #         fitness = row.data['fitness']
    #         _infos.append((row.id, fitness))
    #     _infos.sort(key=lambda x: x[1])
    #     stable_row = db.get(_infos[0][0])
    #     stable_atoms = stable_row.toatoms()
    #     to_md_atoms.append(stable_atoms)
    #     to_md_db.write(stable_atoms, data=stable_row.data, key_value_pairs=stable_row.key_value_pairs)

    if os.path.exists(os.path.join(cwd, 'warnings.log')):
        os.remove(os.path.join(cwd, 'warnings.log'))
