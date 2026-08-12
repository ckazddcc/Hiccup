from ase.io import read, write
from ase.io.lammpsdata import write_lammps_data
import os
from ase.db import connect
from ase.data import atomic_numbers, atomic_masses
from string import Template
import shutil


class LammpsSystem:
    """Generate LAMMPS MD sampling jobs from a structure database.

    Creates per-structure subdirectories with LAMMPS input files, data files,
    model files, and GPU-assigned run scripts. Also provides a utility to
    collect MD trajectory results.
    """

    def __init__(self,
                 lammps_in_template,
                 structures_db,
                 workdir,
                 model_path,
                 gpus,
                 ):
        self.lammpsin_template = lammps_in_template
        self.structures_db = structures_db
        self.workdir = workdir
        self.model_path = model_path
        self.gpus = gpus

    def gen_lammps_jobs(self):
        """Generate LAMMPS input files and run scripts for all structures.

        For each structure in the database, creates a working subdirectory
        containing: lammps.data, lammps.in (with mass block), the model file,
        a cell-record database (a.db), and a GPU-assigned run script.
        """
        db = connect(self.structures_db)
        for row in db.select():
            kvp = row.key_value_pairs
            uid = kvp.get("uid", row.id)
            workdir_i = f"{self.workdir}/{uid}"
            if not os.path.exists(workdir_i):
                os.makedirs(workdir_i)
            struct_ = row.toatoms()
            # Generate LAMMPS input files
            numbers = struct_.numbers
            struct = struct_[numbers.argsort()]
            symbols = set(struct.get_chemical_symbols())
            sorted_symbols = sorted(symbols, key=lambda x: atomic_numbers[x])
            # print(f"Processing structure with uid: {uid}, symbols: {sorted_symbols}")
            write_lammps_data(f"{workdir_i}/lammps.data", struct, atom_style="atomic",
                              specorder=sorted_symbols)

            # Create a.db to record cell parameters
            db_i = connect(f"{workdir_i}/a.db")
            data = row.data
            kvp = row.key_value_pairs
            db_i.write(struct, data=data, key_value_pairs=kvp)

            # Copy LAMMPS input template to working directory
            mass_block = ""
            for i, symbol in enumerate(sorted_symbols, start=1):
                mass = atomic_masses[atomic_numbers[symbol]]
                mass_block += f"mass            {i}   {mass:.2f}\n"
            with open(os.path.join(self.lammpsin_template, "lammps.in"), 'r') as f:
                tpl = Template(f.read())
            filled = tpl.substitute(mass_block=mass_block.strip())
            lammps_in = f"{workdir_i}/lammps.in"
            with open(lammps_in, 'w') as f:
                f.write(filled)

            # Copy model file to working directory, keeping original filename
            model_dst = os.path.join(workdir_i, os.path.basename(self.model_path))
            if not os.path.exists(model_dst):
                shutil.copyfile(self.model_path, model_dst)

            # Generate submission script
            gpu_i = row.id % len(self.gpus)
            gpu_id = str(self.gpus[gpu_i])
            submit_sh_template = os.path.join(self.lammpsin_template, "run_lammps_mpi.sh")
            with open(submit_sh_template, 'r') as f:
                for line in f:
                    if "export CUDA_VISIBLE_DEVICES" in line:
                        line = f"export CUDA_VISIBLE_DEVICES={gpu_id}\n"
                    with open(f"{workdir_i}/run_lammps_mpi.sh", 'a') as f_out:
                        f_out.write(line)


def collect_results(workdir_i, elements):
    """Collect LAMMPS MD trajectory frames into a gathered database.

    Reads the output XYZ dump, converts numeric type labels to element
    symbols, applies the original cell, and writes each frame to
    gathered.db.

    Args:
        workdir_i: working directory of a single LAMMPS job.
        elements: list of element symbols in LAMMPS type order.
    """
    cell = None
    db = connect(f"{workdir_i}/a.db")
    for row in db.select():
        atoms = row.toatoms()
        cell = atoms.get_cell()
        break
    gather_db = connect(workdir_i + "/gathered.db")
    output_xyz = os.path.join(workdir_i, "xyz")
    if not os.path.exists(output_xyz):
        os.makedirs(output_xyz)

    with open(os.path.join(workdir_i, "out_new.xyz"), "r") as f:
        contents = f.readlines()
        num_poscar = sum([1 for _ in contents if 'Atoms' in _])
        length_poscar = len(contents) // num_poscar
        for i in range(num_poscar):
            xyz = []
            xyz_old = contents[i * length_poscar:(i + 1) * length_poscar]
            elem_num = [str(i) for i in range(1, len(elements) + 1)]
            for row in range(length_poscar):
                c = xyz_old[row]
                if c.split()[0] in elem_num:
                    elem_index = elem_num.index(c.split()[0])
                    xyz.append(c.replace(c.split()[0], elements[elem_index], 1))
                else:
                    xyz.append(c)

            with open(os.path.join(output_xyz, "%d.xyz" % i), "w") as f:
                f.writelines(xyz)

            atoms = read(os.path.join(output_xyz, "%d.xyz" % i))
            atoms.set_cell(cell)
            gather_db.write(atoms)


if __name__ == "__main__":
    # Example usage
    lammps_in_template = "<YOUR_LAMMPS_TEMPLATE_PATH>"
    structures_db = "<YOUR_STRUCTURES_DB_PATH>"
    workdir = "<YOUR_WORKDIR_PATH>"
    model_path = "<YOUR_MODEL_PATH>"
    gpus = [0]  # Example GPU IDs
    lammps_system = LammpsSystem(lammps_in_template, structures_db, workdir, model_path, gpus)
    lammps_system.gen_lammps_jobs()
    # collect_results("<YOUR_LMP_RESULTS_DIR>", ["O", "Cu"])
