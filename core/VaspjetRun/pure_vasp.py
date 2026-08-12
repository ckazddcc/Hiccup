import yaml
import os
import re
import time
import shutil
import numpy as np
import pandas as pd
import submitit
import tabulate
import random
import xml.etree.ElementTree as ET
from art import text2art
from copy import deepcopy
from typing import NamedTuple, List, Dict
from submitit.core.core import Job
from ase.db import connect
from ase import Atoms
from ase.io import Trajectory
from ase.calculators.vasp import Vasp
from ase.calculators.singlepoint import SinglePointCalculator as sp

VaspJet1 = """
██╗   ██╗ █████╗ ███████╗██████╗      ██╗███████╗████████╗
██║   ██║██╔══██╗██╔════╝██╔══██╗     ██║██╔════╝╚══██╔══╝
██║   ██║███████║███████╗██████╔╝     ██║█████╗     ██║   
╚██╗ ██╔╝██╔══██║╚════██║██╔═══╝ ██   ██║██╔══╝     ██║   
 ╚████╔╝ ██║  ██║███████║██║     ╚█████╔╝███████╗   ██║   
  ╚═══╝  ╚═╝  ╚═╝╚══════╝╚═╝      ╚════╝ ╚══════╝   ╚═╝   
Welcome to VaspJet, a tool to accelerate VASP relaxation!
Written by: Yinkaai. \n
"""

VaspJet2 = """
 __     __                               _____            __                 ,:
|  \   |  \                             |     \          |  \              ,' |
| ▓▓   | ▓▓ ______   _______  ______     \▓▓▓▓▓ ______  _| ▓▓_            /   :
| ▓▓   | ▓▓|      \ /       \/      \      | ▓▓/      \|   ▓▓ \        --'   /
 \▓▓\ /  ▓▓ \▓▓▓▓▓▓\  ▓▓▓▓▓▓▓  ▓▓▓▓▓▓\__   | ▓▓  ▓▓▓▓▓▓|\▓▓▓▓▓▓        \/ /:/
  \▓▓\  ▓▓ /      ▓▓\▓▓    \| ▓▓  | ▓▓  \  | ▓▓ ▓▓    ▓▓ | ▓▓ __     __/ ://_\ 
   \▓▓ ▓▓ |  ▓▓▓▓▓▓▓_\▓▓▓▓▓▓\ ▓▓__/ ▓▓ ▓▓__| ▓▓ ▓▓▓▓▓▓▓▓ | ▓▓|  \    )-.  / 
    \▓▓▓   \▓▓    ▓▓       ▓▓ ▓▓    ▓▓\▓▓    ▓▓\▓▓     \  \▓▓  ▓▓   ./  : \ 
     \▓     \▓▓▓▓▓▓▓\▓▓▓▓▓▓▓| ▓▓▓▓▓▓▓  \▓▓▓▓▓▓  \▓▓▓▓▓▓▓   \▓▓▓▓      ///""
                            | ▓▓                                     */*
                            | ▓▓                                  .-"*
                             \▓▓                                 ( *   *) 
Thanks for using VaspJet, Bye!
"""


def print_pretty_list(long_list, elements_per_row=15, column_width=5, prefix="") -> str:
    res_str = []
    for i in range(0, len(long_list), elements_per_row):
        column = long_list[i:i + elements_per_row]
        formatted_columns = [f"{item:<.3f}" if isinstance(item, (float,)) else str(item) for item in column]
        formatted_columns = [f"{item:<{column_width}}" for item in formatted_columns]
        res_str.append(prefix + ' '.join(formatted_columns))
    return '\n'.join(res_str)


# Named tuple representing a submitted job
class Job_tuple(NamedTuple):
    """Container for a submitted VASP job.

    Attributes:
        conf_id: configuration index.
        job: submitit Job object.
    """
    conf_id: int
    job: Job


class PureVaspRelax(object):
    def __init__(self, yml_path: str):
        """
        Vasp Relaxation.
        Args:
            yml_path (str): The path to the yml file.
        """
        with open(yml_path, 'r') as yml:
            yml_config = yaml.safe_load(yml)
            db_path = os.path.abspath(os.path.expanduser(yml_config['db_path']))
            work_dir = os.path.abspath(os.path.expanduser(yml_config['work_path']))
            job_settings = yml_config['job_settings']

        if not os.path.exists(work_dir):
            os.makedirs(work_dir)
            self.work_dir = os.path.abspath(os.path.expanduser(work_dir))
        else:
            if os.listdir(work_dir):
                raise Exception(f"Work dir {work_dir} is not empty, please backup or delete it")
            else:
                self.work_dir = os.path.abspath(os.path.expanduser(work_dir))

        self.final_conf = self._get_init_conf_data(db_path)
        self.vasp_version = job_settings['slurm'].pop('vasp_version', 'vasp_std')
        self.slurm_params = self._check_slurm_params(job_settings['slurm'])
        self.vasp_params = self._check_vasp_params(job_settings['vasp'])
        self.fmax = np.abs(job_settings['vasp'].get('ediffg', 0.05))
        self.save_log = job_settings.get('save_log', True)
        self.print_log = job_settings.get('print_log', False)
        self.log_file = open(os.path.join(self.work_dir, 'Jet-log.txt'), 'w')
        # Store results to disk in real time
        self.intime_db = connect(os.path.join(self.work_dir, 'intime_results.db'))
        self._job_health_code = ['COMPLETED', 'PENDING', 'RUNNING', 'REQUEUED', 'RESIZING', 'UNKNOWN']
        self._job_unfinished_code = ['PENDING', 'RUNNING', 'REQUEUED', 'RESIZING', 'UNKNOWN']
        self.job_monitor = pd.DataFrame(
            columns=['conf_id', 'slurm_state', 'converged', 'fmax', 'energy', 'ionic_steps', 'mins', 'err', 'err_msg']
        )
        self.job_monitor = self.job_monitor.astype(
            {'conf_id': int, 'slurm_state': str, 'converged': bool, 'fmax': float, 'energy': float,
             'ionic_steps': int, 'mins': float, 'err': bool, 'err_msg': str})
        for i in range(len(self.final_conf)):
            self.job_monitor.loc[i] = [i, 'PD', False, np.nan, np.nan, 0, 0.0, False, 'NA']

        self.logger(VaspJet1)
        self.logger(
            f'>> JOB SUMMARY:\n'
            f' - VaspJet PureVasp Version is starting at {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}\n'
            f' - VASP version is {self.vasp_version}\n'
            f' - There are {len(self.final_conf)} initial configurations waiting to be relaxed.\n\n'
        )

    def logger(self, log: str, prefix: str = ''):
        if self.save_log:
            self.log_file.write(prefix + log + '\n')
            self.log_file.flush()
        if self.print_log:
            print(prefix + log)

    @staticmethod
    def _get_init_conf_data(init_db_path: str) -> List[Atoms]:
        """
        Get data from db file, and assign the id of each data to atoms.info['id'] to ensure that the calculation results
        of each initial structure are tracked correctly
        :return: The initial configuration data in List.
        """
        init_db = connect(init_db_path)
        init_data = []
        if init_db.count() == 0:
            raise ValueError("The initial database is empty")
        for row in init_db.select():
            image = row.toatoms(add_additional_information=True)
            image.info['id'] = row.id - 1
            init_data.append(image)
        return init_data

    @staticmethod
    def _check_slurm_params(settings: Dict) -> Dict:
        param_item = ['slurm_partition', 'nodes', 'tasks_per_node', 'cpus_per_task']
        for item in settings.keys():
            if item in param_item:
                param_item.remove(item)
        if len(param_item) != 0:
            raise ValueError(f"Missing slurm parameters: {param_item}")
        if settings.get('tasks_per_node', -1) != 1:
            raise ValueError("The slurm parameters tasks_per_node should be 1")
        if settings.get('nodes', -1) != 1:
            raise ValueError("The slurm parameters nodes should be 1")
        return deepcopy(settings)

    def _check_vasp_params(self, settings: Dict) -> Dict:
        if settings.get('lwave', True):
            raise ValueError("The VASP parameter 'LWAVE' should set to False")
        if settings.get('lcharg', True):
            raise ValueError("The VASP parameter 'LCHARG' should set to False")
        settings['command'] = f"mpirun -np {self.slurm_params['cpus_per_task']} {self.vasp_version}"
        return deepcopy(settings)

    @staticmethod
    def _do_vasp_relax(params: Dict, atoms: Atoms):
        st = time.time()
        err_msg = 'NA'
        err = False
        params_copy = deepcopy(params)
        atoms_copy = atoms.copy()
        if 0 in params['kpts']:
            cell_length = np.linalg.norm(atoms.get_cell(), axis=1, ord=np.inf)
            for i in range(3):
                if params['kpts'][i] == 0:
                    params_copy['kpts'][i] = max(1, int(round(15 / cell_length[i])))
                    # params_copy['kpts'][i] = max(1, int(round(40 / cell_length[i])))
        calc = Vasp(**params_copy)
        atoms.calc = calc
        # Must try because calc.resort reorders atoms and force arrays
        try:
            atoms.get_potential_energy()
        except Exception as e:
            err_msg = str(e)
            err = True
        # Check SCF convergence
        file_line = []
        osz_path = os.path.join(params_copy['directory'], 'OSZICAR')
        with open(osz_path, 'r') as file:
            for line in file.readlines():
                file_line.append(line)
        scf_converaged = []
        for idx, item in enumerate(file_line):
            if 'F=' in item and 'E0' in item:
                columns = file_line[idx - 1].split()
                dE, d_eps = columns[3], columns[4]
                if (abs(float(dE)) < params_copy.get('ediff', 1.0e-4) and
                        abs(float(d_eps)) < params_copy.get('ediff', 1.0e-4)):
                    scf_converaged.append(True)
                else:
                    scf_converaged.append(False)
        # calculator vasprun.xml
        xml_path = os.path.join(params_copy['directory'], 'vasprun.xml')
        with open(xml_path, 'r', encoding='ISO-8859-1') as _file:
            xml_string = _file.read()
        pattern = re.compile(re.escape('<calculation>') + r'(.*?)' + re.escape('</calculation>'), re.DOTALL)
        # Find all matching sections
        matches = pattern.findall(xml_string)
        calc_lst = ['<calculation>' + match + '</calculation>' for match in matches] if matches is not None else []
        if len(calc_lst) == 0:
            return np.nan, np.nan, atoms, 0, round((time.time() - st) / 60, 1), err, err_msg
        # find the converaged atoms
        converged_atom = []
        for scf, calc_str in zip(scf_converaged, calc_lst):
            if scf:
                calcroot = ET.fromstring(calc_str)
                # position
                posroot = calcroot.find(".//varray[@name='positions']")
                positions = []
                for v in posroot.findall('v'):
                    # Extract values from each line, stripping whitespace
                    values = v.text.strip().split()
                    # Convert to floats and append to positions list
                    positions.append([float(value) for value in values])
                positions = atoms.get_cell().cartesian_positions(np.array(positions))
                # force
                forcesroot = calcroot.find(".//varray[@name='forces']")
                _forces = []
                for v in forcesroot.findall('v'):
                    # Extract values from each line, stripping whitespace
                    values = v.text.strip().split()
                    # Convert to floats and append to forces list
                    _forces.append([float(value) for value in values])
                # energy
                _energy = float(calcroot.find("./energy/i[@name='e_fr_energy']").text)

                image = Atoms(
                    numbers=atoms.get_atomic_numbers(),
                    positions=positions[calc.resort],
                    cell=atoms.get_cell(),
                    pbc=atoms.get_pbc()
                )
                image.set_tags(atoms.get_tags())
                image.constraints = atoms.constraints
                image.info = atoms.info
                sp_calc = sp(
                    atoms=image,
                    energy=_energy,
                    forces=np.array(_forces)[calc.resort]
                )
                sp_calc.implemented_properties = ["energy", "forces"]
                image.set_calculator(sp_calc)
                converged_atom.append(image)
        # write trajectory
        if len(converged_atom) == 0:
            err = True
            err_msg = 'No converged atoms'
            return np.nan, np.nan, atoms_copy, 0, round((time.time() - st) / 60, 1), err, err_msg
        trajfile = os.path.join(params_copy['directory'], 'relaxation.traj')
        trajwrite = Trajectory(trajfile, 'w')
        for item in converged_atom:
            trajwrite.write(item)
        trajwrite.close()
        energy = converged_atom[-1].get_potential_energy()
        fmax = np.max(np.linalg.norm(converged_atom[-1].get_forces(), axis=1))
        return energy, fmax, converged_atom[-1], len(calc_lst), round((time.time() - st) / 60, 1), err, err_msg

    def _submit_var_slurm(self, part: float, unrelax_conf: List[Atoms]) -> List[Job_tuple]:
        """
        Submit the VASP calculation jobs to the slurm system, and return the job list
        Args:
            part (int): the part of the calculation, 1 for the unaccelerate part, 2 for the accelerate part
            unrelax_conf (List[Atoms]): The unrelaxed configuration list

        Returns:
            List[Job]: The job list
        """
        if not unrelax_conf:
            return []

        vasp_work_dir = os.path.join(self.work_dir, f'part-{part:<3.1f}')

        if not os.path.exists(vasp_work_dir):
            os.makedirs(vasp_work_dir)
        else:
            shutil.rmtree(vasp_work_dir)
            os.makedirs(vasp_work_dir)

        # init slurm parameters
        executor = submitit.AutoExecutor(folder=os.path.join(vasp_work_dir, 'submitit_log'))
        executor.update_parameters(**self.slurm_params)
        jobs_list = []
        job_info_record = ["Slurm shell script and log files are in the folder 'slurm_log'. Job info is as follows:"]
        for i in range(len(unrelax_conf)):
            # parepare parameters
            conf_id = unrelax_conf[i].info['id']
            vasp_params_copy = deepcopy(self.vasp_params)
            vasp_params_copy['directory'] = os.path.join(vasp_work_dir, str(conf_id))
            fun_kwargs = dict(
                params=vasp_params_copy,
                atoms=unrelax_conf[i],
            )

            # submit job
            executor.update_parameters(slurm_job_name=f"{part:<3.1f}-{conf_id}")
            job = executor.submit(self._do_vasp_relax, **fun_kwargs)

            job_item = Job_tuple(conf_id=conf_id, job=job)
            jobs_list.append(job_item)
            job_info_record.append(
                f"Slurm Job ID {int(job.job_id)}: Part {part:<3.1f}, Conf_id {conf_id:<5.0f}, Batch_id {i:<5.0f}")
            time.sleep(5)

        with open(os.path.join(vasp_work_dir, 'slurm_job_info.txt'), 'w') as file:
            file.write('\n'.join(job_info_record))

        return jobs_list

    def _check_job_state(self, jobs_list: List[Job_tuple]):
        """
        Check the job state, and return the state, error information, conf_id, trajectory
        Args:
            jobs_list: The job list
        Returns:
            Tuple[str, str, int, List[Atoms]]: The state, error information, job_id, conf_id, trajectory
        """
        if not jobs_list:
            return []

        jobs_state = ['---'] * len(self.final_conf)
        error_info = []

        PD_list = [item.conf_id for item in jobs_list]
        cache_list = []

        cd_num = 0
        failed_num = 0
        converged_num = 0

        self.logger(log=
                    f"{time.strftime('%m-%d %H:%M:%S', time.localtime())}: "
                    f"Pending+Running: {len(PD_list):<4.0f}, "
                    f"Completed: {cd_num:<4.0f}, "
                    f"Converged: {converged_num:<4.0f}, "
                    f"Failed: {failed_num:<4.0f}",
                    prefix='   '
                    )

        time.sleep(10)
        while True:
            for item in jobs_list:
                time.sleep(5)
                conf_id = item.conf_id
                job = item.job
                state = job.get_info()['State']

                if (conf_id not in PD_list) or (conf_id in cache_list):
                    continue

                if state not in self._job_health_code:
                    self.job_monitor.loc[conf_id, 'slurm_state'] = state
                    self.job_monitor.loc[conf_id, 'err'] = True
                    self.job_monitor.loc[conf_id, 'err_msg'] = state
                    failed_num += 1
                    cd_num += 1
                    cache_list.append(conf_id)
                    PD_list.remove(conf_id)

                elif state == 'COMPLETED':
                    energy, fmax, last_image, ionic_num, ust, err, err_msg = job.result()
                    ust += self.job_monitor.loc[conf_id, 'mins']
                    ionic_num += self.job_monitor.loc[conf_id, 'ionic_steps']
                    if err:
                        failed_num += 1
                        self.job_monitor.loc[conf_id] = [conf_id, state, False, fmax,
                                                         energy, ionic_num, ust, True, err_msg]
                    else:
                        converged = True if fmax < self.fmax else False
                        converged_num += 1 if converged else 0
                        self.job_monitor.loc[conf_id] = [conf_id, state, converged, fmax,
                                                         energy, ionic_num, ust, False, err_msg]
                        _data = last_image.info.get("data", {})
                        _data['energy'] = last_image.get_potential_energy()
                        _data['forces'] = last_image.get_forces(apply_constraint=False)
                        _kvp = last_image.info.get("key_value_pairs", {})
                        self.intime_db.write(atoms=last_image,
                                             data=_data,
                                             key_value_pairs=_kvp)
                    self.final_conf[conf_id] = last_image
                    cd_num += 1
                    cache_list.append(conf_id)
                    PD_list.remove(conf_id)

                elif state in self._job_unfinished_code:
                    self.job_monitor.loc[conf_id, 'slurm_state'] = state

            if (len(cache_list) >= len(jobs_list) // 15) and (len(cache_list) != 0):
                self.logger(log=
                            f"{time.strftime('%m-%d %H:%M:%S', time.localtime())}: "
                            f"Pending+Running: {len(PD_list):<4.0f}, "
                            f"Completed: {cd_num:<4.0f}, "
                            f"Converged: {converged_num:<4.0f}, "
                            f"Failed: {failed_num:<4.0f}",
                            prefix='   '
                            )
                cache_list.clear()
            if len(PD_list) == 0:
                self.logger(log=
                            f"{time.strftime('%m-%d %H:%M:%S', time.localtime())}: "
                            f"Pending+Running: {len(PD_list):<4.0f}, "
                            f"Completed: {cd_num:<4.0f}, "
                            f"Converged: {converged_num:<4.0f}, "
                            f"Failed: {failed_num:<4.0f}",
                            prefix='   '
                            )
                break

        for i in range(len(self.job_monitor)):
            if self.job_monitor.loc[i, 'err']:
                jobs_state[i] = "Error"
                error_info.append(f"conf_id-{i:<4}: {self.job_monitor.loc[i, 'err_msg']}")
            else:
                if self.job_monitor.loc[i, 'fmax'] <= self.fmax:
                    jobs_state[i] = self.job_monitor.loc[i, 'fmax']
                else:
                    jobs_state[i] = self.job_monitor.loc[i, 'fmax']

        self.logger(log=
                    f"Finished! Fmax list is as follows:\n" +
                    print_pretty_list(long_list=jobs_state, prefix='   '),
                    prefix='   '
                    )
        if len(error_info) != 0:
            self.logger(log=f"Error info is here, you should check it:\n   " + '\n   '.join(error_info),
                        prefix='   '
                        )
        self.logger(log='\n')
        return

    def _check_all_confs_converged(self, end: bool = False) -> bool:
        """
        Check whether all the configurations are converged
        Returns:
            bool: Whether all the configurations are converged
        """
        if all(self.job_monitor['converged'].tolist()) or end:
            table = tabulate.tabulate(
                self.job_monitor.loc[:, ['conf_id', 'converged', 'fmax', 'energy', 'ionic_steps', 'mins']],
                headers='keys',
                tablefmt='rst',
                showindex=False
            )

            unacc_id = list(range(len(self.final_conf)))
            unacc_ionic_num = self.job_monitor.loc[unacc_id, 'ionic_steps'].tolist()
            unacc_mins = self.job_monitor.loc[unacc_id, 'mins'].tolist()
            unacc_avg_ionic = sum(unacc_ionic_num) / len(unacc_ionic_num)
            unacc_avg_mins = sum(unacc_mins) / len(unacc_mins)

            if all(self.job_monitor['converged'].tolist()):
                self.logger(
                    log=
                    f"*** BRAVO!!! ALL THE CONFS ARE CONVERGED AT "
                    f"{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())} ***\n\n" +
                    table + "\n" +
                    f"Mean Ionic Steps: {unacc_avg_ionic:.2f}; " +
                    f"Mean Times: {unacc_avg_mins:.2f} mins" +
                    "\n\n" + VaspJet2
                )
                self.log_file.close()
            else:
                self.logger(
                    log=
                    f"*** OH GOSH!!! SOME CONFS ARE NOT CONVERGED AT "
                    f"{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())} ***\n\n" +
                    table + "\n" +
                    f"Mean Ionic Steps: {unacc_avg_ionic:.2f}; " +
                    f"Mean Times: {unacc_avg_mins:.2f} mins" +
                    "\n\n" + VaspJet2
                )
                self.log_file.close()
            return True
        else:
            return False

    def run(self, max_round: int = 2):
        """
        Run the VASP relaxation.
        """
        p = 1
        while not self._check_all_confs_converged(end=True if p > max_round else False):
            unconverage_conf = []
            for i in range(len(self.final_conf)):
                if not self.job_monitor.loc[i, 'converged']:
                    unconverage_conf.append(self.final_conf[i].copy())

            jobs = self._submit_var_slurm(part=p, unrelax_conf=unconverage_conf)
            self.logger(log=text2art(text=f'... Part-{p} ...', font='standard'))
            self.logger(f">> START TO CHECK THE PART-{p} JOBS:")
            self._check_job_state(jobs_list=jobs)
            p += 1

        return self.final_conf


if "__main__" == __name__:
    import argparse

    parser = argparse.ArgumentParser(description="Here is VaspJet Pure-Vasp warp!")
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Create subparser for the 'run' command
    parser_run = subparsers.add_parser(
        "run",
        help="Run the VASP calculation with specified configurations. 'vaspjet run --help' for more information.")
    parser_run.add_argument(
        "-yml",
        dest="yml",
        type=str,
        help="Path to the .yml config. Typing: str, Required: True",
        required=True
    )
    parser_run.add_argument(
        "-r",
        dest="round",
        type=int,
        help="The round of the calculation, only work for --vasponly. Typing: int, Required: False"
    )

    args = parser.parse_args()
    if args.command == 'run':
        pv = PureVaspRelax(yml_path=args.yml)
        res = pv.run(max_round=args.round if args.round is not None else 2)
        finaldb = connect(os.path.join(pv.work_dir, 'final.db'))
        for finalatoms in res:
            kvp = finalatoms.info.get('key_value_pairs', {})
            data = finalatoms.info.get('data', {})
            try:
                data["energy"] = finalatoms.get_potential_energy()
                data["forces"] = finalatoms.get_forces(apply_constraint=False)
                finaldb.write(
                    atoms=finalatoms,
                    key_value_pairs=kvp,
                    data=data
                )
            except Exception as e:
                pass
    else:
        parser.print_help()
    print("All Calculation Finished!")

