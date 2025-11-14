from ase.io import read, write
from ase.io.lammpsdata import write_lammps_data
import os
from ase.db import connect
from ase.data import atomic_numbers, atomic_masses
from string import Template
import shutil


class LammpsSystem:
    """
    Lammps采样，自动生成输入文件提交作业，回收采样结构。
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
        """
        获取Lammps输入文件内容
        """
        db = connect(self.structures_db)
        for row in db.select():
            kvp = row.key_value_pairs
            uid = kvp.get("uid", row.id)
            workdir_i = f"{self.workdir}/{uid}"
            if not os.path.exists(workdir_i):
                os.makedirs(workdir_i)
            struct_ = row.toatoms()
            # 生成lammps输入文件
            numbers = struct_.numbers
            struct = struct_[numbers.argsort()]
            symbols = set(struct.get_chemical_symbols())
            sorted_symbols = sorted(symbols, key=lambda x: atomic_numbers[x])
            # print(f"Processing structure with uid: {uid}, symbols: {sorted_symbols}")
            write_lammps_data(f"{workdir_i}/lammps.data", struct, atom_style="atomic",
                              specorder=sorted_symbols)

            # 在子目录下生成一个a.db文件用来记录晶胞参数
            db_i = connect(f"{workdir_i}/a.db")
            data = row.data
            kvp = row.key_value_pairs
            db_i.write(struct, data=data, key_value_pairs=kvp)

            # 将lammps输入文件模板复制到工作目录
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

            # 复制模型文件到工作目录
            model_dst = f"{workdir_i}/frozen_model.pb"
            if not os.path.exists(model_dst):
                shutil.copyfile(self.model_path, model_dst)

            # 生成提交脚本
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
    """
    收集Lammps计算结果
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
    lammps_in_template = "/home/cchen/Hiccup/template/lammps"
    structures_db = "/home/cchen/tmp/lmp/cata.db"
    workdir = "/home/cchen/tmp/lmp"
    model_path = "/home/cchen/tmp/lmp/frozen_model.pb"
    gpus = [0]  # Example GPU IDs
    # lammps_system = LammpsSystem(lammps_in_template, structures_db, workdir, model_path, gpus)
    collect_results("/home/cchen/tmp/lmp/1", ["O", "Cu", "Y"])
