import os
import logging
import json
from ase.db import connect
from dpdata import MultiSystems, LabeledSystem
from glob import glob
import numpy as np
import sys
import random
from pathlib import Path
import subprocess
import time
from deepmd.calculator import DP
from deepmd.entrypoints.main import main as dpmain
import shutil
import matplotlib.pyplot as plt
import torch.multiprocessing as mp
import multiprocessing
from datetime import datetime, timedelta


class DeepmdSystem:
    """Manage DeepMD model training pipelines.

    Handles dataset preparation (splitting, format conversion), input file
    generation, training, freezing, and best-model selection across multiple
    model replicas with different random seeds.
    """

    # Mapping from backend name to CLI alias
    _BACKEND_FLAGS = {
        "tensorflow": "--tf",
        "pytorch": "--pt",
        "paddle": "--pd",
        "jax": "--jax",
    }

    def __init__(self,
                 elements,
                 db_path,
                 train_ratio,
                 workdir,  # e.g. pes/dp/nn0
                 gpu,
                 dp_input_template,
                 init_model_path=None,
                 models_num=4,
                 backend="tensorflow"
                 ):

        self.elements = elements
        self.db_path = db_path
        self.db = connect(db_path)
        self.train_ratio = train_ratio
        self.workdir = workdir
        if not os.path.exists(self.workdir):
            os.makedirs(self.workdir)
        self.gpu = gpu
        self.models_num = models_num
        self.dp_input_template = dp_input_template
        self.sub_dirs = [os.path.join(self.workdir, f"{i:03d}") for i in range(self.models_num)]
        self.init_model_path = init_model_path
        self.best_model = init_model_path
        self.backend = backend

    @staticmethod
    def run_dp(cmd, backend="tensorflow"):
        """Execute a dp CLI command via the DeepMD Python entry point.

        Args:
            cmd: dp command string (e.g. "dp train input.json").
            backend: DeepMD backend ("tensorflow", "pytorch", "paddle", "jax").
        """
        cmds = cmd.split()
        if cmds[0] == "dp":
            cmds = cmds[1:]
        else:
            raise RuntimeError("The command is not dp")
        flag = DeepmdSystem._BACKEND_FLAGS.get(backend, "--tf")
        cmds = [flag] + cmds
        dpmain(cmds)

    @staticmethod
    def _split_db(db, fold_name):
        """Split a database into multiple files grouped by chemical formula.

        Args:
            db: open ASE database connection.
            fold_name: output directory for the split database files.
        """
        N2db = {}
        for row in db.select():
            atoms = row.toatoms()
            n = atoms.get_chemical_formula()
            if N2db.get(n, None) is None:
                N2db[n] = connect('{}/{}.db'.format(fold_name, n))
                N2db[n].write(atoms, data=row.data, key_value_pairs=row.key_value_pairs)
            else:
                N2db[n].write(atoms, data=row.data, key_value_pairs=row.key_value_pairs)
        return

    @staticmethod
    def _db2deepmd(db, output_fold):
        """Convert an ASE database to DeepMD npy format.

        Args:
            db: open ASE database connection with energy/forces data.
            output_fold: output directory for DeepMD data files.
        """
        ms = MultiSystems()
        for row in db.select():
            atoms = row.toatoms()
            ls = LabeledSystem(atoms, fmt='ase/structure')
            ls.data["energies"] = np.array([row.data["energy"]])
            ls.data["forces"] = np.array([row.data["forces"]])
            if 'virials' in ls.data:
                del ls.data['virials']
            ms.append(ls)

        ms.to_deepmd_npy(output_fold, 1000000)
        # ms.to_deepmd_raw(output_fold)

    def _split_db_to_train_valid(self):
        """Split the dataset into training and validation sets.

        Structures are grouped by chemical formula, randomly shuffled, and
        divided according to train_ratio. The resulting sets are converted
        to DeepMD npy format.
        """
        work_dir = self.workdir
        train_ratio = self.train_ratio

        dbs_path = os.path.join(work_dir, 'dbs')
        self._split_db(self.db, dbs_path)
        train_dir = os.path.join(work_dir, 'dbs/train')
        test_dir = os.path.join(work_dir, 'dbs/test')
        if not os.path.exists(train_dir):
            os.makedirs(train_dir)
        if not os.path.exists(test_dir):
            os.makedirs(test_dir)

        train_db_path = os.path.join(train_dir, 'train.db')
        test_db_path = os.path.join(test_dir, 'test.db')

        db_s = glob(dbs_path + '/*.db')
        for db_i_path in db_s:
            if os.path.basename(db_i_path) in ["init.db"]:
                continue
            db_i = connect(db_i_path)
            entries = list(range(1, db_i.count() + 1))

            # Shuffle entries randomly
            random.shuffle(entries)
            # Split into training and validation sets
            train_size = int(len(entries) * train_ratio)
            train_entries = entries[:train_size]
            test_entries = entries[train_size:]

            with connect(train_db_path) as train_db:
                for entry in train_entries:
                    row = db_i.get(entry)
                    atoms = row.toatoms()
                    data = row.data
                    if "energy" not in data.keys() or "forces" not in data.keys():
                        data['energy'] = atoms.get_potential_energy()
                        data['forces'] = atoms.get_forces(apply_constraint=False)
                    else:
                        pass
                    train_db.write(atoms, data=data, key_value_pairs=row.key_value_pairs)

            with connect(test_db_path) as test_db:
                for entry in test_entries:
                    row = db_i.get(entry)
                    atoms = row.toatoms()
                    data = row.data
                    if "energy" not in data.keys() or "forces" not in data.keys():
                        data['energy'] = atoms.get_potential_energy()
                        data['forces'] = atoms.get_forces(apply_constraint=False)
                    else:
                        pass
                    test_db.write(atoms, data=data, key_value_pairs=row.key_value_pairs)

        self._db2deepmd(train_db, os.path.join(dbs_path, 'train'))
        self._db2deepmd(test_db, os.path.join(dbs_path, 'test'))

    def _setup_directory(self, num_splits=4):
        """Create numbered subdirectories (000, 001, ...) under the workdir.

        Args:
            num_splits: number of subdirectories to create.
        """
        # Create workdir if it does not exist
        if not os.path.exists(self.workdir):
            os.makedirs(self.workdir)
        # Create subdirectories: 000, 001, 002, ...
        sub_dirs = [os.path.join(self.workdir, f"{i:03d}") for i in range(num_splits)]
        for sub_dir in sub_dirs:
            os.makedirs(sub_dir, exist_ok=True)
        return

    @staticmethod
    def _resolve_train_command(run_config_path, init_model_path=None, backend="tensorflow"):
        """Build the dp train command string.

        Args:
            run_config_path: path to the input.json file.
            init_model_path: optional path to a frozen model (.pb) or
                checkpoint (.ckpt) for transfer learning.
            backend: DeepMD backend ("tensorflow", "pytorch", "paddle", "jax").

        Returns:
            Command string for dp train.
        """
        output_dir = Path(run_config_path).parent
        flag = DeepmdSystem._BACKEND_FLAGS.get(backend, "--tf")
        command = f'dp {flag} train {run_config_path}'
        if init_model_path is not None:
            init_model_path = Path(init_model_path)
            if init_model_path.name.endswith(".pb"):
                command += " --init-frz-model {}".format(str(init_model_path))
            elif init_model_path.name.endswith("model.ckpt"):
                command += " --init-model {}".format(str(init_model_path))
            else:
                raise RuntimeError(f"Unknown init_model {str(init_model_path)}.")
        command += f' 2>&1 > {output_dir}/err.out'
        return command

    @staticmethod
    def _resolve_freeze_command(frozen_name, backend="tensorflow"):
        """Build the dp freeze command string.

        Args:
            frozen_name: output filename for the frozen model.
            backend: DeepMD backend ("tensorflow", "pytorch", "paddle", "jax").

        Returns:
            Command string for dp freeze.
        """
        flag = DeepmdSystem._BACKEND_FLAGS.get(backend, "--tf")
        command = "dp {0} freeze -o {1} 2>&1 >> ./frozen_err.out".format(flag, frozen_name)
        return command

    def _set_dp_calc(self, model_path):
        """Initialize a DP calculator from a model file.

        If the model format is incompatible, attempts automatic conversion.

        Args:
            model_path: path to the frozen DP model.

        Returns:
            DP calculator instance.
        """
        try:
            calc = DP(model=model_path)
        except RuntimeError:
            base_dir, file_name = os.path.split(model_path)
            new_file_name = f"{file_name}.old"
            model_path_old = os.path.join(base_dir, new_file_name)
            os.rename(model_path, model_path_old)
            print(model_path)
            # convert-from is only supported by the TensorFlow backend
            self.run_dp(f"dp convert-from -i {model_path_old} -o {model_path}", backend="tensorflow")
            calc = DP(model=model_path)
        return calc

    def creat_dp_input(self):
        """Generate DeepMD input.json files for all model replicas.

        Prepares the dataset (split, convert to npy), then writes an input.json
        with randomized seeds into each subdirectory.
        """
        dbs_path = os.path.join(self.workdir, 'dbs')
        if not os.path.exists(dbs_path):
            os.makedirs(dbs_path)
        if os.path.dirname(self.db_path) != os.path.join(self.workdir, "dbs"):
            shutil.copy(self.db_path, os.path.join(dbs_path, "init.db"))
            self.db_path = os.path.join(dbs_path, "init.db")
            self.db = connect(self.db_path)

        self._split_db_to_train_valid()
        workdir = self.workdir
        elements = self.elements
        dp_input = json.load(open(self.dp_input_template))
        self._setup_directory(self.models_num)
        for i, sub_dir in enumerate(self.sub_dirs):
            dp_input['model']['type_map'] = elements
            dp_input['model']['descriptor']["sel"] = [200 for _ in elements]
            dp_input['model']['descriptor']['seed'] = random.randrange(sys.maxsize) % (2 ** 32)
            if "type_embedding" in dp_input["model"].keys():
                dp_input['model']["type_embedding"]['seed'] = random.randrange(sys.maxsize) % (2 ** 32)
            dp_input['model']['fitting_net']['seed'] = random.randrange(sys.maxsize) % (2 ** 32)
            dp_input['training']['seed'] = random.randrange(sys.maxsize) % (2 ** 32)
            # Convert relative paths to absolute paths
            deeps_train = [os.path.abspath(p) for p in glob(os.path.join(workdir, f'dbs/train/*')) if os.path.isdir(p)]
            deeps_test = [os.path.abspath(p) for p in glob(os.path.join(workdir, f'dbs/test/*')) if os.path.isdir(p)]
            dp_input['training']['training_data']['systems'] = deeps_train
            dp_input['training']['validation_data']['systems'] = deeps_test
            min_train_num = min(
                [len(np.load(os.path.join(d, 'set.000/energy.npy'))) for d in
                 glob(os.path.join(workdir, f'dbs/train/*')) if os.path.isdir(d)])
            min_valid_num = min(
                [len(np.load(os.path.join(d, 'set.000/energy.npy'))) for d in
                 glob(os.path.join(workdir, f'dbs/test/*')) if os.path.isdir(d)])
            numb_btch = min([int(min_train_num * self.train_ratio), min_valid_num])
            if numb_btch > 4:
                dp_input["training"]["validation_data"]["numb_btch"] = numb_btch
            dp_input['training']["training_data"]['batch_size'] = [1 for _ in deeps_train]
            dp_input['training']["validation_data"]['batch_size'] = [1 for _ in deeps_test]
            with open(os.path.join(sub_dir, 'input.json'), 'w') as f:
                json.dump(dp_input, f, sort_keys=False, indent=4, separators=(",", ":"))
            print(f"Input data of model {i:03d} has been successfully processed！")

    def train_single_model(self, sub_dir, gpu_i):
        """Train a single model replica.

        Args:
            sub_dir: directory containing input.json.
            gpu_i: GPU device index for this training run.
        """
        run_config_path = os.path.join(sub_dir, 'input.json')
        init_model_path_i = None
        if self.init_model_path is not None:
            init_model_path_i = os.path.join(sub_dir, f"init_{os.path.basename(self.init_model_path)}")
            shutil.copy(self.init_model_path, init_model_path_i)
        command = self._resolve_train_command(run_config_path, init_model_path_i, backend=self.backend)
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_i)
        command = f'nohup ' + command + ' &'
        subprocess.run(command, cwd=sub_dir, shell=True, env=env)
        print(command)

    def train_models(self):
        """Train all model replicas sequentially with a 60s gap between launches."""
        for i in range(self.models_num):
            name = f"{i:03d}"
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.gpu[i])
            sub_dir = os.path.join(self.workdir, name)
            self.train_single_model(sub_dir, self.gpu[i])
            time.sleep(60)
        # processes = []
        # for i, gpu in enumerate(self.gpu):
        #     p = multiprocessing.Process(target=self.train_single_model, args=(self.sub_dirs[i], gpu))
        #     processes.append(p)
        #     p.start()
        # for p in processes:
        #     p.join()

    def monitor_training(self):
        """Monitor training progress for all model replicas.

        Checks the learning curve file to determine finished steps. Marks a
        model as done if training is complete, or as "Error" if the log file
        has been stale for over 30 minutes.

        Returns:
            List of status values (True, False, or "Error") per model.
        """
        train_state = [False for _ in self.sub_dirs]
        for i, sub in enumerate(self.sub_dirs):
            done_path = os.path.join(sub, 'TRAINDONE')
            if train_state[i] is False:
                all_steps = json.load(open(os.path.join(sub, 'input.json')))['training']['numb_steps']
                all_steps = int(all_steps)
                lcurve_file = os.path.join(sub, 'lcurve.out')
                while not os.path.exists(lcurve_file):
                    time.sleep(30)
                out = open(lcurve_file, 'r')
                lines = out.readlines()
                try:
                    finished_steps = int(lines[-1][:10])
                except:
                    finished_steps = 0
                if finished_steps >= all_steps:
                    with open(done_path, "w") as f:
                        f.write(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
                        f.write("Training Done!")
                        pass
                    train_state[i] = True
                else:
                    output_file_path = os.path.join(sub, 'lcurve.out')
                    # Get last modified time of the log file
                    last_modified_time = os.path.getmtime(output_file_path)
                    # Convert to datetime
                    last_modified_datetime = datetime.fromtimestamp(last_modified_time)
                    # Get current time
                    current_time = datetime.now()
                    # Check if log has been stale for over 30 minutes
                    time_difference = current_time - last_modified_datetime
                    if time_difference > timedelta(minutes=30):
                        train_state[i] = "Error"
            else:
                train_state[i] = True
        return train_state

    def freeze_models(self):
        """Freeze all trained models into frozen_model.pb files.

        Runs dp freeze in each subdirectory. Retries once after 30 seconds
        if the frozen model is not generated on the first attempt.

        Returns:
            List of boolean status values per model.
        """
        frozen_state = [False for _ in self.sub_dirs]
        for i, path in enumerate(self.sub_dirs):
            command = self._resolve_freeze_command("frozen_model.pb", backend=self.backend)
            command = f'CUDA_VISIBLE_DEVICES={self.gpu[i]} nohup ' + command + ' &'
            subprocess.run(command, cwd=path, shell=True)
            # If frozen_model.pb was not generated, retry after 30 seconds
            if not os.path.exists(os.path.join(path, "frozen_model.pb")):
                time.sleep(30)
                subprocess.run(command, cwd=path, shell=True)
                # Check again whether frozen_model.pb was generated
                if os.path.exists(os.path.join(path, "frozen_model.pb")):
                    frozen_state[i] = True
            else:
                frozen_state[i] = True
        return frozen_state

    def get_best_model(self):
        """Select the best model based on validation loss.

        Compares the L2 validation loss across all trained models and returns
        the one with the lowest weighted loss (10% validation + 90% training).

        Returns:
            Tuple of (best_model_path, best_model_id, models_info_dict).
        """
        models = [m for m in os.listdir(self.workdir) if '00' in m and os.path.isdir(os.path.join(self.workdir, m))]
        best_model_path = ''
        best_loss = float('inf')
        models_info = {}
        best_model_id = ''
        for m in models:
            lcurve = os.path.join(self.workdir, m, "lcurve.out")
            if os.path.exists(lcurve):
                with open(lcurve, "r") as f:
                    lines = f.readlines()
                    for i in range(len(lines) - 1, 1, -1):
                        line = lines[i].strip()
                        if "nan" not in line:
                            result = line.split()
                            l2_val = round(float(result[1]), 4)
                            l2_trn = round(float(result[2]), 4)
                            l2_e_val = round(float(result[3]), 4)
                            l2_e_trn = round(float(result[4]), 4)
                            l2_f_val = round(float(result[5]), 4)
                            l2_f_trn = round(float(result[6]), 4)
                            model_info = [l2_val, l2_trn, l2_e_val, l2_e_trn, l2_f_val, l2_f_trn]
                            models_info[m] = model_info
                            loss = l2_val * 0.1 + l2_trn * 0.9
                            break
                        else:
                            loss = float('inf')
                if loss < best_loss:
                    best_loss = loss
                    best_model_path = os.path.join(self.workdir, m, "frozen_model.pb")
                    self.best_model = best_model_path
                    best_model_id = m
        return best_model_path, best_model_id, models_info

if __name__ == '__main__':
    dp = DeepmdSystem(elements=["O", "Cu"],
                      db_path="<YOUR_DB_PATH>",
                      train_ratio=0.9,
                      workdir="<YOUR_WORKDIR_PATH>",
                      gpu=[1],
                      dp_input_template="<YOUR_DP_INPUT_TEMPLATE>",
                      init_model_path="<YOUR_INIT_MODEL_PATH>",
                      #init_model_path=None,
                      models_num=1)
    dp.creat_dp_input()
    dp.train_models()
    train_state = dp.monitor_training()
    while not all(train_state):
        train_state = dp.monitor_training()
        print("train state:", train_state)
        time.sleep(60)
    frozen_state = dp.freeze_models()
    best_model_path, best_model_id, models_info = dp.get_best_model()
    print(best_model_path)
    print(best_model_id)
