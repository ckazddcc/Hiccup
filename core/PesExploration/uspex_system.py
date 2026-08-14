import json
import os
import sys
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
    """Manage USPEX structure search jobs with NN-based local optimization.

    Generates USPEX input files (INPUT.txt, POSCARS, POTCAR, INCAR) from
    random seed structures, monitors job status, and collects results.
    Supports 2D slab and 3D bulk searches with MACE or DP calculators.
    """

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
                 ediffg=0.1,
                 nsw=200,
                 gpu=0
                 ):
        self.work_dir = workdir
        if not os.path.exists(workdir):
            os.makedirs(workdir)
        # Sort elements and compositions by atomic number
        elem_dict = {elem: target_composition[i] for i, elem in enumerate(elements)}
        elem_sort = sorted(elem_dict.items(), key=lambda x: atomic_numbers[x[0]])
        elements = [ele for ele, _ in elem_sort]
        target_composition = [elem_dict[ele] for ele, _ in elem_sort]

        self.elements = elements
        self.target_composition = target_composition
        self.dimension = dimension
        chemformula = "".join(f"{ele}{num}" for ele, num in zip(elements, target_composition))
        unique_mark = f"{self.dimension}_{chemformula}"
        # Distinguish substrates when running multiple in parallel
        if multi_substrates:
            substrate_index = os.path.basename(substrate_path).split("_")[-1]
            unique_mark += f"_{substrate_index}"
        self.unique_mark = unique_mark
        # Create composition working directory
        self.compos_work_dir = os.path.join(workdir, f"{unique_mark}")
        self.template_dir = uspex_templates_dir
        self.substrate_path = substrate_path
        self.model_path = model_path
        self.gpu = gpu
        # GA parameters
        self.generation_num = generation_num
        self.pop_size = pop_size
        self.ini_pop_size = ini_pop_size
        self.calculator = calculator
        self.opt_method = opt_method
        self.ediffg = ediffg
        self.nsw = nsw
        # Seed structure settings
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

        potcar_dir = os.environ.get("POTCAR_DIR")
        if not potcar_dir:
            raise EnvironmentError(
                "Environment variable POTCAR_DIR is not set. "
                "Please set it to the path of your POTCAR library directory, "
                "e.g.: export POTCAR_DIR=/path/to/POTCAR_library"
            )
        self.potcar_dir = potcar_dir

    @staticmethod
    def write_poscars(atoms_list, output_file):
        """Write a list of Atoms objects into a combined POSCARS file.

        Each structure is sorted by atomic number, wrapped into the cell,
        and labeled with an EA header for USPEX compatibility.

        Args:
            atoms_list: list of ASE Atoms objects.
            output_file: path to the output POSCARS file.
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
            # Sort by atomic number
            atoms = sort(atoms, tags=atoms.get_atomic_numbers())
            tmp_file = os.path.join(os.path.dirname(output_file), f"POSCAR_{i}")
            # Clear atom constraints
            atoms_clean = atoms.copy()
            atoms_clean.set_constraint()
            # Write POSCAR format via ASE and label EA in the first line
            write(tmp_file, atoms_clean, direct=True, vasp5=True)
            with open(tmp_file, "r") as f:
                content = f.readlines()
            content[0] = f"EA{i + 1}\n"
            gathered_poscars.extend(content)
            os.remove(tmp_file)

        with open(output_file, "w") as f:
            f.writelines(gathered_poscars)

    def random_seeds_selection(self, num):
        """Select the most stable seed structures for GA initialization.

        Picks structures matching the target composition, preferring those
        with low max force and low selection count. Selection history is
        persisted to a JSON file to avoid reusing the same seeds.

        Args:
            num: number of structures to select.

        Returns:
            List of selected ASE Atoms objects.
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
            # Filter by selection count
            info.sort(key=lambda x: x[2])
            _num = min(math.ceil(num * 2), len(info))
            selected_times = info[:_num]
            # Filter by energy
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
        """Generate the USPEX INPUT.txt file.

        Compatible with USPEX 10.5 and earlier. Newer versions use
        input.uspex instead.
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
        """Renumber EA labels in a POSCARS file sequentially (EA1, EA2, ...).

        Args:
            good_poscars_path: path to the POSCARS file to renumber.

        Returns:
            List of updated file lines.
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

        # Find line indices of all EA entries
        config_indices = [i for i, line in enumerate(good_poscars) if line.startswith("EA")]
        # Renumber EA labels sequentially
        for idx, line_idx in enumerate(config_indices, start=1):
            line = good_poscars[line_idx].split()
            line[0] = f'EA{idx}'
            good_poscars[line_idx] = f"{line[0]}  {'  '.join(line[1:-1])}    {line[-1]}\n"
        return good_poscars

    def generate_potcar(self, output_file='POTCAR_1'):
        """Concatenate per-element POTCAR files into a single POTCAR.

        Args:
            output_file: path to the output POTCAR file.
        """
        with open(output_file, 'w') as potcar:
            for element in self.elements:
                potcar_path = os.path.join(self.potcar_dir, f'POTCAR_{element}')
                if not os.path.exists(potcar_path):
                    raise FileNotFoundError(f"{potcar_path} not found!")
                with open(potcar_path, 'r') as potcar_part:
                    potcar.write(potcar_part.read())

    def create_uspex_input(self):
        """Generate all USPEX input files and resources for a structure search.

        Creates the working directory, selects seed structures (from random
        seeds or a specified database), writes POSCARS for each generation,
        copies POTCAR/INCAR files, generates INPUT.txt, and patches the
        local optimization script with correct model paths and constraints.
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

        # For 2D systems, copy substrate file
        if self.dimension == 2:
            shutil.copy(self.substrate_path, os.path.join(self.compos_work_dir, "POSCAR_SUBSTRATE"))

        # Generate POSCARS_1: use specified seeds if available, otherwise select from random seeds
        selected_atoms = []
        if not self.init_seeds_path:
            if self.random_seeds_path is None:
                pass
            else:
                self.init_seeds_path = self.random_seeds_path
                selected_atoms = self.random_seeds_selection(self.ini_pop_size * 0.2)
                POSCARS_1 = os.path.join(seeds_dir, 'POSCARS_1')
                self.write_poscars(selected_atoms, POSCARS_1)

        elif "POSCARS" in os.path.basename(self.init_seeds_path):
            # Renumber EA labels
            good_poscars = self.renumber_EA(self.init_seeds_path)
            # Write updated content to POSCARS_1
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
            if len(selected_atoms) > self.ini_pop_size * 0.5:
                num = int(math.ceil(self.ini_pop_size * 0.5))
                selected_atoms = random.sample(selected_atoms, num)
            POSCARS_1 = os.path.join(seeds_dir, 'POSCARS_1')
            self.write_poscars(selected_atoms, POSCARS_1)

        # Generate POSCARS for subsequent generations
        for i in range(2, self.generation_num + 1):
            seeds_num = math.ceil(0.2 * self.pop_size)
            if self.random_seeds_path:
                selected_atoms = self.random_seeds_selection(seeds_num)
            else:
                selected_atoms = []
            POSCARS_i = os.path.join(seeds_dir, f'POSCARS_{i}')
            self.write_poscars(selected_atoms, POSCARS_i)

        for elem in self.elements:
            potcar_path = os.path.join(self.potcar_dir, f"POTCAR_{elem}")
            shutil.copy(potcar_path, os.path.join(specific_dir, f"POTCAR_{elem}"))
        shutil.copy(os.path.join(self.template_dir, "INCAR_1"), os.path.join(specific_dir, "INCAR_1"))

        # Generate INPUT.txt
        calc_tag = "mace" if self.calculator == "MACE" else "dp"
        bash_path = os.path.join(self.template_dir, f"run_{calc_tag}.sh")
        self.generate_input_txt(bash_path)

        # Patch .py path and python interpreter in run_dp/mace.sh
        nn_inf_path = os.path.join(self.template_dir, f"{calc_tag}_opt.py")
        python_path = sys.executable
        with open(bash_path, "r") as f:
            content = f.readlines()
            for i, line in enumerate(content):
                if "cp" in line:
                    content[i] = f"  cp {nn_inf_path} .\n"
                if "<YOUR_PYTHON_PATH>" in line:
                    content[i] = line.replace("<YOUR_PYTHON_PATH>", python_path)
        with open(bash_path, "w") as f:
            f.writelines(content)

        # Patch paths in dp/mace_opt.py
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
                # Patch model path in dp_opt.py
                if calc_tag == "dp":
                    if "model_path =" in line and "#" not in line:
                        content[i] = f"model_path = \"{self.model_path}\"\n"
                # Patch model path in mace_opt.py
                elif calc_tag == "mace":
                    if "model_path =" in line and "#" not in line:
                        script_directory = Path(__file__).parent
                        model_path = os.path.join(script_directory, f"tools/mace-mpa-0-medium-float32.model")
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
        """Gracefully stop a running USPEX job.

        Renames the work directory, waits for active calculations to finish,
        removes temporary calculation folders, then restores the directory.
        """
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
        """Monitor USPEX job status.

        Checks for done/failed marker files. If the log file has not been
        updated for 30 minutes, marks the job as failed and kills it.

        Returns:
            "RUNNING", "DONE", or "FAILED".
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
            # Get last modified time of the log file
            last_modified_time = os.path.getmtime(output_file_path)
            # Convert to datetime
            last_modified_datetime = datetime.fromtimestamp(last_modified_time)
            # Get current time
            current_time = datetime.now()
            # Check if log has been stale for over 30 minutes
            time_difference = current_time - last_modified_datetime
            if time_difference > timedelta(minutes=30):
                with open(fail_falg, "w") as f:
                    f.write("USPEX IS FAILED")
                STATE = "FAILED"
                self.kill_uspex()
                return STATE
            else:
                return STATE


# Process Individuals file: deduplicate structures and write to gathered.db
def pick_individuals(individuals_path):
    """Extract deduplicated structures from a USPEX Individuals file.

    Parses the Individuals file, deduplicates by volume/density/fitness,
    and returns unique structure IDs with their fitness values.

    Args:
        individuals_path: path to the USPEX Individuals file.

    Returns:
        List of (structure_id, fitness) tuples.
    """
    ids = []
    fits = []
    with open(individuals_path) as fp:
        # Skip first two header lines
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
    # Deduplicate
    unique_fits = set(fits)
    # Map unique fits back to original indices
    unique_ids = [(ids[fits.index(u)], float(u.split("_")[2])) for u in unique_fits]
    # Convert to integers
    unique_ids = [(int(ii[0]), ii[1]) for ii in unique_ids]
    return unique_ids


def write_to_db(work_dir, constraint_z=0):
    """Collect USPEX results into a gathered.db database.

    Reads the latest results directory, deduplicates structures, converts
    POSCARS to ASE Atoms, applies substrate constraints if needed, and
    writes them to gathered.db. Structures with abnormal fixed-atom counts
    are filtered out.

    Args:
        work_dir: USPEX working directory containing results* subdirectories.
        constraint_z: z-coordinate threshold for fixing substrate atoms;
            0 means no constraint.

    Returns:
        Path to the gathered database, or False if no results found.
    """
    subdirectories = []
    fix_num = []
    for root, dirs, files in os.walk(work_dir):
        # Get subdirectory names (non-recursive)
        subdirectories.extend(dirs)
        break
    subdirectories = [int(i[7:]) for i in subdirectories if i.startswith("results")]
    if not subdirectories:
        print("No results directory found.")
        return False
    results_id = max(subdirectories)
    results_path = os.path.join(work_dir, f"results{results_id}")
    alls_db_path = os.path.join(work_dir, "gathered.db")
    # Remove existing gathered.db or create empty one
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
        for (i, fit) in unique_ids:
            try:
                poscar_dir = os.path.join(results_path, "POSCAR")
                poscar_content = contents[(i-1) * length_poscar:i * length_poscar]
                with open(poscar_dir, 'w') as poscar_file:
                    poscar_file.writelines(poscar_content)
                atoms = read(poscar_dir)
                # Apply atom constraints
                if constraint_z > 0:
                    fix_indexs = [atom.index for atom in atoms if atom.position[2] < constraint_z]
                    fix_num.append(len(fix_indexs))
                    c = FixAtoms(indices=[atom.index for atom in atoms if atom.index in fix_indexs])
                    atoms.set_constraint(c)
                all_db.write(atoms, data={'fitness': fit, "Individual": i})
            except:
                print(f"{i} Error")
                continue

    # Filter out unreasonable structures
    if constraint_z != 0:
        counter = Counter(fix_num)
        most_common_fix_num = counter.most_common(1)[0][0]
        for row in all_db.select():
            atoms = row.toatoms()
            fix_indexs = [atom.index for atom in atoms if atom.position[2] < constraint_z]
            if abs(len(fix_indexs) - most_common_fix_num) > 2:
                all_db.delete([row.id])
    print(f"{work_dir}/gathered.db has been created successfully.")
    return alls_db_path


if __name__ == '__main__':
    uspex = UspexSystem(elements=["O", "Cu"],
                        target_composition=[10, 10],  
                        dimension=0,
                        workdir="<YOUR_WORKDIR_PATH>",
                        uspex_templates_dir="<YOUR_USPEX_TEMPLATES_DIR>",
                        generation_num=2,
                        pop_size=10,
                        ini_pop_size=5,
                        model_path="<YOUR_MODEL_PATH>",
                        calculator="MACE"
                        )
    uspex.create_uspex_input()

    if os.path.exists(os.path.join(cwd, 'warnings.log')):
        os.remove(os.path.join(cwd, 'warnings.log'))
