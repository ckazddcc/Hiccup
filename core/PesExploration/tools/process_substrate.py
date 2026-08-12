"""
Process 2D substrate structures.

Sorts atoms by atomic number, adjusts the vacuum layer, and writes a
POSCAR_SUBSTRATE file for downstream use.
"""
import os
from ase.io import read, write
from ase import Atoms
from ase.data import atomic_numbers


def process_2d_substrate(substrate_path):
    """Process a 2D substrate and write a POSCAR_SUBSTRATE file.

    Sorts atoms by atomic number, rewrites the element-order and count lines
    in VASP POSCAR format, and saves the result as POSCAR_SUBSTRATE_new.

    Args:
        substrate_path: path to the input substrate structure file.
    """
    # Get substrate index and path
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
    process_2d_substrate("<YOUR_SUBSTRATE_PATH>")
