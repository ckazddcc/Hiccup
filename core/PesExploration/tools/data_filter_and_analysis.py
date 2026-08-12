import os
import time
import json
import numpy as np
from ase.db import connect
from glob import glob
from deepmd.calculator import DP
from concurrent.futures import ProcessPoolExecutor
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import multiprocessing
import matplotlib.pyplot as plt
import matplotlib
import logging
import shutil
import re


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


def count_natom(s):
    """Parse a chemical formula string and return the total number of atoms."""
    pattern = r"([A-Z][a-z]?)(\d*)"
    matches = re.findall(pattern, s)
    element_dict = {}
    for element, count in matches:
        element_dict[element] = int(count) if count else 1
    return sum(element_dict.values())


def data_filter_and_analysis(workdir,
                             model_path,
                             gpu_ids=None,
                             energy_filter=0.1,
                             force_filter=2):
    """Filter structures by prediction error and compute accuracy metrics.

    Structures are split by chemical formula, then evaluated in parallel across
    GPUs using a DP model. Structures whose per-atom energy error exceeds
    *energy_filter* or whose max force error exceeds *force_filter* are moved
    to a residue database; the rest are kept in an iteration database.

    Args:
        workdir: working directory containing per-formula .db files.
        model_path: path to the frozen DP model.
        gpu_ids: list of GPU device IDs for parallel inference.
        energy_filter: per-atom energy error threshold (eV/atom).
        force_filter: max force component error threshold (eV/Å).

    Returns:
        Path to the iteration database containing retained structures.
    """
    if gpu_ids is None:
        raise ValueError("gpu_ids must be provided")
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        # Ignore error if start_method is already set
        pass

    # Main process does not use GPU
    os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    gpu_num = len(gpu_ids)
    dbs = glob(os.path.abspath(os.path.join(workdir, "*.db")))
    remove_db_path = os.path.join(workdir, "residue.db")
    iter_db_path = os.path.join(workdir, "iter.db")
    composition_error_dict = {}
    outdir = os.path.join(workdir, "out")
    os.makedirs(outdir, exist_ok=True)

    # Allocate GPU resources
    db_groups = [[] for _ in gpu_ids]
    # Distribute database files evenly across GPUs
    for i, d in enumerate(dbs):
        if os.path.basename(d) in ["iter.db", "init.db", "traj.db", "new_iter.db", "residue.db"]:
            continue
        db_groups[i % gpu_num].append(d)

    # Process in parallel using multiprocessing
    with ProcessPoolExecutor(max_workers=gpu_num) as executor:
        futures = []
        for i in range(gpu_num):
            if db_groups[i]:
                future = executor.submit(
                    process_gpu_group,
                    db_groups[i],
                    model_path,
                    gpu_ids[i],
                    outdir,
                    energy_filter,
                    force_filter,
                    remove_db_path,
                    iter_db_path
                )
                futures.append(future)

        # Collect results
        for future in futures:
            formula_errors = future.result()
            if formula_errors:
                composition_error_dict.update(formula_errors)

    # Save results
    model_name = os.path.basename(os.path.dirname(model_path))
    timestamp = time.strftime('_%Y_%m_%d_%H_%M_%S')
    output_path = os.path.join(workdir, f"out/Composition_Error_of_{model_name}{timestamp}.json")
    with open(output_path, "w") as f:
        json.dump(composition_error_dict, f, indent=4)
    return os.path.join(workdir, "iter.db")


def process_gpu_group(db_list, model_path, gpu_id, outdir, energy_filter, force_filter, remove_db_path, iter_db_path):
    """Process a group of databases on a single GPU.

    Computes DP predictions for each structure, writes prediction files,
    partitions structures into residue/iteration databases based on error
    thresholds, and returns per-formula error statistics.

    Args:
        db_list: list of database file paths to process.
        model_path: path to the frozen DP model.
        gpu_id: GPU device index assigned to this process.
        outdir: directory for prediction output files.
        energy_filter: per-atom energy error threshold (eV/atom).
        force_filter: max force component error threshold (eV/Å).
        remove_db_path: path to the residue database (rejected structures).
        iter_db_path: path to the iteration database (retained structures).

    Returns:
        Dict mapping chemical formula to energy/force RMSE and MAE.
    """
    # Each process occupies one GPU exclusively
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    calc = DP(model=model_path)
    remove_db = connect(remove_db_path)
    iter_db = connect(iter_db_path)
    composition_error_dict = {}
    for d in db_list:
        try:
            db = connect(d)
            first_atoms = db.get(1).toatoms()
        except:
            continue
        formula = first_atoms.get_chemical_formula()
        natom = len(first_atoms)
        true_energies, pre_energies, true_forces, pre_forces = [], [], [], []
        entries = []

        # Collect data
        for i in range(db.count()):
            ati = db.get(i + 1)
            atoms = db.get_atoms(i + 1)
            data = ati.data
            entries.append(ati)
            try:
                true_energy = data["energy"]
                true_force = data["forces"]
            except KeyError:
                continue
            atoms.calc = calc
            pre_energy = atoms.get_potential_energy()
            pre_force = atoms.get_forces()
            true_energies.append(true_energy)
            pre_energies.append(pre_energy)
            true_forces.append(true_force)
            pre_forces.append(pre_force)

        # Convert to NumPy arrays
        true_energies = np.array(true_energies)
        pre_energies = np.array(pre_energies)
        true_forces = np.array(true_forces)
        pre_forces = np.array(pre_forces)

        # Compute errors
        energy_errors = np.abs((true_energies - pre_energies) / natom)
        force_errors = np.max(np.abs(true_forces - pre_forces), axis=(1, 2))
        remove_mask = (energy_errors >= energy_filter) | (force_errors >= force_filter)

        # Write to files
        e_file = os.path.join(outdir, f"{formula}.out.e.out")
        f_file = os.path.join(outdir, f"{formula}.out.f.out")

        np.savetxt(e_file, np.vstack([true_energies, pre_energies]).T,
                   header="# data_e pred_e", fmt="%f  %f", comments='')

        with open(f_file, "w") as ff:
            ff.write("# data_fx data_fy data_fz pred_fx pred_fy pred_fz\n")
            for tf, pf in zip(true_forces, pre_forces):
                np.savetxt(ff, np.hstack([tf, pf]), fmt="%f  %f  %f  %f  %f  %f")

        # Write to database
        remove_entries = [row for idx, row in enumerate(entries) if remove_mask[idx]]
        iter_entries = [row for idx, row in enumerate(entries) if not remove_mask[idx]]

        write_to_db(remove_db, remove_entries)
        write_to_db(iter_db, iter_entries)

        # Compute statistics
        rmse_force = np.sqrt(np.mean((true_forces - pre_forces) ** 2))
        mae_force = np.mean(np.abs(true_forces - pre_forces))
        rmse_energy = np.sqrt(np.mean(((true_energies - pre_energies) / natom) ** 2))
        mae_energy = np.mean(np.abs((true_energies - pre_energies) / natom))

        composition_error_dict[formula] = {
            "Energy(meV/atom)": {"RMSE": rmse_energy * 1000, "MAE": mae_energy * 1000},
            "Force(eV/A)": {"RMSE": rmse_force, "MAE": mae_force}
        }
    return composition_error_dict


def write_to_db(db, entries):
    """Write a list of ASE database rows into the given database."""
    for row in entries:
        db.write(row.toatoms(), data=row.data, **row.key_value_pairs)


def plt_out(outdir, model_name):
    """Generate parity plots and error statistics for energy and forces.

    Reads prediction output files from *outdir*, plots DFT vs. predicted
    values, and saves figures plus a JSON summary of RMSE/MAE.

    Args:
        outdir: directory containing .out.e.out and .out.f.out files.
        model_name: name label used in plot titles and output filenames.
    """
    matplotlib.use('Agg')
    out_f = glob(os.path.join(outdir, "*.out.f.out"))
    out_e = glob(os.path.join(outdir, "*.out.e.out"))
    all_true_forces = []
    all_pre_forces = []
    all_true_energies = []
    all_pre_energies = []
    for out in out_f:
        with open(out, 'r') as f:
            fp = f.readlines()
        for i in range(1, len(fp)):
            line = fp[i]
            fx, fy, fz, pfx, pfy, pfz = line.strip().split()
            true_forces = [float(fx), float(fy), float(fz)]
            pre_forces = [float(pfx), float(pfy), float(pfz)]
            all_true_forces.extend(true_forces)
            all_pre_forces.extend(pre_forces)

    for out in out_e:
        composition = os.path.basename(out).split(".")[0]
        natom = count_natom(composition)
        with open(out, 'r') as f:
            fp = f.readlines()
        for i in range(1, len(fp)):
            line = fp[i]
            e, pe = line.strip().split()
            all_true_energies.append(float(e) / natom)
            all_pre_energies.append(float(pe) / natom)

    force_fig = os.path.abspath(os.path.join(outdir, 'Force_Prediction_of_Model_%s' % (model_name) + time.strftime(
        '_%Y_%m_%d_%H_%M_%S.png', time.localtime(time.time()))))
    energy_fig = os.path.abspath(os.path.join(outdir, 'Energy_Prediction_of_Model_%s' % (model_name) + time.strftime(
        '_%Y_%m_%d_%H_%M_%S.png', time.localtime(time.time()))))
    output_file = 'Error_of_%s' % (model_name) + time.strftime('_%Y_%m_%d_%H_%M_%S.json', time.localtime(time.time()))
    output_file = os.path.join(outdir, output_file)
    max_force = (int(max(all_true_forces) * 2) + 1) / 2
    min_force = (int(min(all_true_forces) * 2) - 1) / 2
    max_energy = (int(max(all_true_energies) * 2) + 1) / 2
    min_energy = (int(min(all_true_energies) * 2) - 1) / 2
    ff = np.linspace(min_force, max_force, 100)
    ee = np.linspace(min_energy, max_energy, 100)
    plt.plot(ff, ff)
    plt.plot(all_true_forces, all_pre_forces, 'o', markersize=1)
    all_true_forces = np.array(all_true_forces)
    all_pre_forces = np.array(all_pre_forces)
    all_true_energies = np.array(all_true_energies)
    all_pre_energies = np.array(all_pre_energies)
    rmse_force = ((all_true_forces - all_pre_forces) ** 2).mean() ** 0.5
    mae_force = (np.abs(all_true_forces - all_pre_forces)).mean()
    rmse_energy = ((all_true_energies - all_pre_energies) ** 2).mean() ** 0.5
    mae_energy = (np.abs(all_true_energies - all_pre_energies)).mean()

    error_dict = {"Energy(meV/atom)": {"RMSE": rmse_energy * 1000, "MAE": mae_energy * 1000},
                  "Force(eV/A)": {"RMSE": rmse_force, "MAE": mae_force}}
    for k, v in error_dict.items():
        print(k)
        for kk, vv in v.items():
            print(kk, vv)
    with open(output_file, 'w') as f:
        json.dump(error_dict, f)

    plt.grid(True)
    plt.xlabel('DFT')
    plt.ylabel('NN')
    plt.title('Forces predicted by NN %s' % model_name)
    plt.text((max_force + min_force) * 0.5, (max_force + min_force + 1) * 0.5, 'RMSE:%.4f' % rmse_force)
    plt.text((max_force + min_force) * 0.5, (max_force + min_force) * 0.5, 'MAE:%.4f' % mae_force)
    plt.savefig(force_fig, dpi=400)
    plt.figure()

    plt.plot(ee, ee)
    plt.plot(all_true_energies, all_pre_energies, 'o', markersize=1)
    plt.grid(True)
    plt.xlabel('DFT')
    plt.ylabel('NN')
    plt.title('Energy predicted by NN %s' % model_name)
    plt.text((max_energy + min_energy) * 0.5, (max_energy + min_energy + 1) * 0.5, 'RMSE:%.5f' % rmse_energy)
    plt.text((max_energy + min_energy) * 0.5, (max_energy + min_energy) * 0.5, 'MAE:%.5f' % mae_energy)
    plt.savefig(energy_fig, dpi=400)
    plt.figure()


def analysis(db_path,
             model_path,
             gpu_ids,
             energy_filter=0.1,
             force_filter=2,
             model_name="model"):
    """Run the full filter-and-analysis pipeline on a database.

    Splits the database by chemical formula, filters structures by prediction
    error, and generates parity plots and error statistics.

    Args:
        db_path: path to the source ASE database.
        model_path: path to the frozen DP model.
        gpu_ids: list of GPU device IDs for parallel inference.
        energy_filter: per-atom energy error threshold (eV/atom).
        force_filter: max force component error threshold (eV/Å).
        model_name: name label used in plot titles and output filenames.
    """
    cwd = os.getcwd()
    logging.basicConfig(filename=os.path.join(cwd, 'warnings.log'),
                        level=logging.DEBUG,
                        format='%(asctime)s - %(levelname)s - %(message)s')
    logging.captureWarnings(True)
    workdir = os.path.join(os.path.dirname(db_path), "split")
    if os.path.exists(workdir):
        shutil.rmtree(workdir)
    split_db(db_path, workdir)
    data_filter_and_analysis(workdir=workdir,
                             model_path=model_path,
                             gpu_ids=gpu_ids,
                             energy_filter=energy_filter,
                             force_filter=force_filter)
    plt_out(outdir=os.path.join(workdir, "out"),
            model_name=model_name)
    if os.path.exists(os.path.join(cwd, 'warnings.log')):
        os.remove(os.path.join(cwd, 'warnings.log'))


def error_eval(db_path, model_path, gpu):
    """Evaluate DP model prediction errors against DFT reference data.

    Computes per-atom energy and force RMSE/MAE for all structures in the
    database and prints the results.

    Args:
        db_path: path to the ASE database with DFT reference data.
        model_path: path to the frozen DP model.
        gpu: GPU device index for inference.

    Returns:
        Tuple of (true_energy, true_forces, pred_energy, pred_forces).
    """
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
    dp_calculate = DP(model=model_path)
    true_energy = []
    true_forces = []
    pred_energy = []
    pred_forces = []
    db = connect(db_path)
    for row in db.select():
        atoms = row.toatoms()
        # energy = row.data['energy']
        # forces = row.data['forces']
        energy = atoms.get_potential_energy()
        forces = atoms.get_forces(apply_constraint=False)
        true_energy_i = energy / len(atoms)
        true_forces_i = forces
        atoms.calc = dp_calculate
        pred_energy_i = atoms.get_potential_energy() / len(atoms)
        pred_forces_i = atoms.get_forces(apply_constraint=False)
        true_energy.append(true_energy_i)
        true_forces.extend(true_forces_i.reshape(-1))
        pred_energy.append(pred_energy_i)
        pred_forces.extend(pred_forces_i.reshape(-1))

    true_energy = np.array(true_energy)
    true_forces = np.array(true_forces)
    pred_energy = np.array(pred_energy)
    pred_forces = np.array(pred_forces)

    energy_rmse = np.sqrt(np.mean((true_energy - pred_energy) ** 2))
    force_rmse = np.sqrt(np.mean((true_forces - pred_forces) ** 2))

    energy_mae = np.mean(np.abs(true_energy - pred_energy))
    force_mae = np.mean(np.abs(true_forces - pred_forces))

    print(f"Energy RMSE: {energy_rmse:.4f} eV / atom")
    print(f"Force RMSE: {force_rmse:.4f} eV / Å")
    print(f"Energy MAE: {energy_mae:.4f} eV / atom")
    print(f"Force MAE: {force_mae:.4f} eV / Å")

    return true_energy, true_forces, pred_energy, pred_forces

if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
    start_time_1 = time.perf_counter()
    split_db(db_path="<YOUR_DB_PATH>", fold_name="<YOUR_FOLD_NAME>")
    data_filter_and_analysis(workdir="<YOUR_WORKDIR_PATH>",
                             model_path="<YOUR_MODEL_PATH>",
                             gpu_ids=[0,1,2,3],
                             energy_filter=0.2,
                             force_filter=1.0)
    # # Record end time of the first function
    end_time_1 = time.perf_counter()
    duration_1 = end_time_1 - start_time_1
    print(f"data_filter_and_analysis elapsed time: {duration_1:.6f} s")

    # db_path = "<YOUR_DB_PATH>"
    # model_path = "<YOUR_MODEL_PATH>"
    # true_energy, true_forces, pred_energy, pred_forces = error_eval(db_path, model_path, gpu=0)
