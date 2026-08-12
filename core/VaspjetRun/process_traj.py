import os
from ase.io import read
from ase.db import connect
import sys


def gather_traj(root_dir, db_path, process="filter", max_force_threshold=5.0):
    """Collect trajectory files into a database.

    Walks *root_dir* for .traj files, processes them according to *process*
    mode, and writes energies and forces to a database.

    Args:
        root_dir: directory to search for .traj files.
        db_path: path to the output database.
        process: processing mode - "filter" (select by force/energy
            criteria), "all" (keep all frames), or "last_image" (last frame only).
        max_force_threshold: maximum force magnitude for filtering (eV/Å).

    Returns:
        Path to the trajectory database.
    """
    if os.path.exists(db_path):
        os.remove(db_path)
    db_traj = connect(db_path)

    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.traj'):
                traj_file_path = os.path.join(root, file)  # Get full path
                try:
                    atomss = read(traj_file_path, index=':')
                except:
                    print("Error in reading traj file: ", traj_file_path)
                    atomss = []
                    for i in range(10000):
                        try:
                            atomss.append(read(traj_file_path, index=i))
                        except:
                            continue

                atoms = []
                if process == "filter":
                    atoms = process_traj_filter(atomss, max_force_threshold)
                elif process == "all":
                    atoms = atomss
                elif process == "last_image":
                    atoms = [atomss[-1]]

                if len(atoms) == 0:
                    atoms.append(atomss[-1])
                for a in atoms:
                    data = a.info.get("data", {})
                    data["energy"] = a.get_potential_energy()
                    data["forces"] = a.get_forces(apply_constraint=False)
                    key_value_pairs = a.info.get("key_value_pairs", {})
                    if type(data) == dict and type(key_value_pairs) == dict:
                        db_traj.write(a,
                                      data=data,
                                      key_value_pairs=key_value_pairs)

    count = db_traj.count()
    with open("./TRAJ_DONE", 'w') as f:
        f.write(f"Total {count} atoms have been written to traj.db !")
    print(f"Total {count} atoms have been written to traj.db !")
    return db_path


def process_traj_filter(traj_atoms, max_force_threshold, delta_energy_threshold=0.1):
    """Filter trajectory frames by force and energy criteria.

    Keeps frames with max force between 0.5 and *max_force_threshold*, and
    low-force frames only if their energy differs from the last kept frame
    by at least *delta_energy_threshold*. The final frame is always included.

    Args:
        traj_atoms: list of ASE Atoms objects from a trajectory.
        max_force_threshold: upper force magnitude cutoff (eV/Å).
        delta_energy_threshold: minimum energy difference for deduplication (eV).

    Returns:
        List of selected Atoms objects.
    """
    selected_traj_atoms = []
    energy = 0
    for a in traj_atoms:
        data = a.info.get("data", {})
        forces = a.get_forces(apply_constraint=False)
        max_force = (forces ** 2).sum(axis=1).max() ** 0.5
        if max_force > max_force_threshold:
            continue
        elif 0.5 <= max_force <= max_force_threshold:
            key_value_pairs = a.info.get("key_value_pairs", {})
            if type(data) == dict and type(key_value_pairs) == dict:
                selected_traj_atoms.append(a)
        else:
            _energy = a.get_potential_energy()
            if abs(energy - _energy) < delta_energy_threshold:
                continue
            else:
                energy = _energy
                key_value_pairs = a.info.get("key_value_pairs", {})
                if type(data) == dict and type(key_value_pairs) == dict:
                    selected_traj_atoms.append(a)

    if traj_atoms[-1] not in selected_traj_atoms:
        selected_traj_atoms.append(traj_atoms[-1])
    return selected_traj_atoms


if __name__ == '__main__':
    # First argument is the max force threshold
    max_force = float(sys.argv[1])
    process_mode = str(sys.argv[2])
    # Usage: python process_traj.py 20 filter  (20 = max force threshold, filter = process mode)
    gather_traj('./', './traj.db', process=process_mode, max_force_threshold=max_force)
    # gather_traj('<YOUR_GA_DIR_PATH>',
    #             '<YOUR_TRAJ_DB_PATH>',
    #             process="filter")

