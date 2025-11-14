"""
处理二维基底结构，处理真空层，将元素按照原子序数进行排列，生成新的 POSCAR_SUBSTRATE 文件
"""
import os
from ase.io import read, write
from ase import Atoms
from ase.data import atomic_numbers


def process_2d_substrate(substrate_path):
    """
    处理二维基底结构，并生成 POSCAR_SUBSTRATE 文件
    """
    # 获取基底索引和路径
    dir_path = os.path.dirname(substrate_path)
    substrate = read(substrate_path)
    cell = substrate.get_cell()
    # cell[2][2] = cell[2][2] / 3 * 2
    substrate_updata = Atoms([atom for atom in sorted(substrate, key=lambda atom: atom.number)])
    elements = list(set(substrate_updata.get_chemical_symbols()))
    elements_sort = sorted(elements, key=lambda x: atomic_numbers[x])

    sub_ele_num_str = [str(substrate_updata.get_chemical_symbols().count(elem)) for elem in elements_sort]
    substrate_updata.set_cell(cell)
    _new_substrate = os.path.join(dir_path, "POSCAR_SUBSTRATE_tmp")
    write(_new_substrate, substrate_updata, direct=True, vasp5=False)
    with open(_new_substrate, "r") as f:
        sub_content = f.readlines()
    sub_content[0] = "POSCAR_SUBSTRATE\n"
    sub_content[5] = "  ".join(sub_ele_num_str) + "\n"
    sub_content.insert(5, "  ".join(elements) + "\n")
    new_substrate = os.path.join(dir_path, "POSCAR_SUBSTRATE_new")

    with open(new_substrate, 'w') as f:
        f.writelines(sub_content)
    os.remove(_new_substrate)
    return


if __name__ == '__main__':
    # cwd = os.getcwd()
    # substrate_path = os.path.join(cwd, 'POSCAR_SUBSTRATE')
    # process_2d_substrate(substrate_path)
    process_2d_substrate("/home/ubuntu/Documents/test/test/NN/0107/POSCAR_BiVO4")
