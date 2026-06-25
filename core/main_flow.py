import logging
import os

cwd = os.getcwd()
logging.basicConfig(filename=os.path.join(cwd, 'warnings.log'),
                    level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logging.captureWarnings(True)
import yaml
import shutil
import time
from ase.db import connect
from ase import Atom
import subprocess
from PesExploration.gen_seeds import GenSeeds
from PesExploration.uspex_system import UspexSystem
from PesExploration.uspex_system import write_to_db
from VaspjetRun.dft import dft
from VaspjetRun.vaspjet_run import VaspjetRun, vaspjet_monitor
from Train.deepmd_module import DeepmdSystem
from PesExploration.nn_deviation import NNDeviation
from PesExploration.tools.energy_structure_filter import energy_structure_filter
from PesExploration.tools.calculator_select import calculator_select
from PesExploration.tools.mace_optimizer import seeds_optimizer
from PesExploration.tools.md_sample import run_md_parallel, gather_md_traj

from ase.io import read
import torch.multiprocessing as mp


class Hiccup:
    def __init__(self, config):
        self.config = config
        self.base_config = config["BASE"]
        self.cpu_config = config["CPU"]
        self.gpu = self.base_config["Gpu"]
        self.elements, self.compositions = (
            self.update_composition(self.base_config["Elements"],
                                    self.base_config["Compositions"]))
        self.templates = self.base_config["Templates"]
        self.workdir = self.base_config["Workdir"]
        if not os.path.exists(self.workdir):
            os.makedirs(self.workdir)
        # 创建pes目录
        self.pes_dir = os.path.join(self.workdir, "pes")
        if not os.path.exists(self.pes_dir):
            os.makedirs(self.pes_dir)

        # 训练器配置接口，创建dp目录
        self.trainer_config = config["TRAINER"]["Deepmd"]
        self.dp_dir = os.path.join(self.workdir, "dp")
        if not os.path.exists(self.dp_dir):
            os.makedirs(self.dp_dir)
        self.best_model = self.trainer_config.get("Initial Model", None)
        self.db_path = self.trainer_config.get("Data Path", None)
        if self.trainer_config.get("Initial Model", None):
            self.best_model = self.trainer_config["Initial Model"]

        # 采样器配置接口，创建seeds目录，ga目录
        self.sampler_ga = config["SAMPLER"]["GA"]
        self.seeds_config = self.sampler_ga["RANDOMSEEDS"]
        self.uspex_config = self.sampler_ga["USPEX"]
        self.dimension = self.uspex_config["Dimension"]
        self.constraint_z = self.uspex_config.get("Constraint z", 0)
        self.ga_dir = os.path.join(self.pes_dir, "ga")
        if not os.path.exists(self.ga_dir):
            os.makedirs(self.ga_dir)
        self.seeds_dir = os.path.join(self.pes_dir, "seeds")
        if not os.path.exists(self.seeds_dir):
            os.makedirs(self.seeds_dir)
        if not self.seeds_config.get("Activate", False):
            if (self.seeds_config.get("Random Seeds Path", None) is not None
                    and self.seeds_config.get("Random Seeds Path", None) != os.path.join(self.seeds_dir, "random_seeds.db")):
                shutil.copy(self.seeds_config["Random Seeds Path"], os.path.join(self.seeds_dir, "random_seeds.db"))
        self.random_seeds_db = os.path.join(self.seeds_dir, "random_seeds.db")
        self.nnmd_config = config["SAMPLER"]["NNMD"]

        # 初始种子
        self.init_seeds_db = self.seeds_config.get("Init Seeds Path", None)

        # postprocess
        self.postprocess_config = self.config["POSTPROCESSING"]

        # tag_flag
        self.model_tag = self.uspex_config.get("Calculator", "MACE")  # 优化模型默认MACE

        # 收敛标准
        self.accuracy_threshold = self.base_config.get("Accuracy Threshold", 0.95)
        self.stall_iterations = self.base_config.get("Stall Iterations", 3)
        self.accurate_ratio = 0
        self.failed_ratio = 1
        self.best_model_info = None
        self.converge_count = 0

        # 日志文件
        self.log = os.path.join(self.workdir, "hiccup-log.txt")
        # if os.path.exists(self.log):
        #     os.remove(self.log)

    @staticmethod
    def update_composition(elements, target_composition):
        """
        根据原子序数更新目标组分的元素顺序
        """
        ele_c = {}
        for i, ele in enumerate(elements):
            tmp = []
            for c in target_composition:
                tmp.append(c[i])
            ele_c[ele] = tmp
        new_elements = sorted(elements, key=lambda x: Atom(x).number)
        new_target_composition = []
        for i in range(len(target_composition)):
            tmp = []
            for ele in new_elements:
                tmp.append(ele_c[ele][i])
            new_target_composition.append(tmp)
        return new_elements, new_target_composition

    @staticmethod
    def create_separator_line(text, total_length=100, separator='='):
        """
        创建一个分割线，确保整个长度一致，中间是动态文字。
        """
        text_length = len(text)
        # 计算两边等号的数量
        separator_length = (total_length - text_length) // 2
        left_separator = separator * separator_length
        right_separator = separator * (total_length - text_length - separator_length)
        # 创建分割线
        separator_line = f"{left_separator}{text}{right_separator}\n"
        return separator_line

    def randomseeds_generator(self):
        """
        启动随机种子结构生成器，并对生成的随机种子结构进行优化
        """
        start = time.time()
        # 生成随机种子结构db文件
        random_seeds_gen_db = os.path.join(self.seeds_dir, "random_seeds_gen.db")
        if os.path.exists(random_seeds_gen_db):
            os.remove(random_seeds_gen_db)
        Rand_seeds_gen = GenSeeds(elements=self.elements,
                                  target_composition=self.compositions,
                                  dimension=self.seeds_config["Dimension"],
                                  seeds_db=random_seeds_gen_db,
                                  seeds_num=self.seeds_config.get("Random Seeds Num", 100),
                                  vacuum_layer_thickness=self.seeds_config.get("Vacuum Layer Thickness", 10))

        Rand_seeds_gen.gen_seeds()
        # 删除cell大于substrate的1.4*cell的结构
        if self.uspex_config["Dimension"] == 2:
            sub = read(self.uspex_config["Substrate"])
            cell = sub.get_cell()
            cell_x = cell[0][0] * 1.4
            cell_y = cell[1][1] * 1.4
            cell_z = cell[2][2] + 13
            for row in connect(random_seeds_gen_db).select():
                atoms = row.toatoms()
                seeds_cell = atoms.get_cell()
                if seeds_cell[0][0] > cell_x or seeds_cell[1][1] > cell_y or seeds_cell[2][2] > cell_z:
                    connect(random_seeds_gen_db).delete([row.id])

        # 提交随机种子结构优化任务
        Rand_seeds_opt = seeds_optimizer(seeds_db_path=random_seeds_gen_db, gpus=self.gpu)
        self.random_seeds_db = Rand_seeds_opt

        # 写入日志 Random Seeds Generator
        composition = []
        for c in self.compositions:
            chemical_formula = "".join([f"{ele}{num}" for ele, num in zip(self.elements, c)])
            composition.append(chemical_formula)
        with open(self.log, "a") as f:
            f.write(self.create_separator_line(" Random Seeds Generator ", total_length=100, separator='-'))
            composition = ", ".join(composition)
            f.write(f"Target Compositions: {composition}\n")
            f.write(f"Dimension: {self.seeds_config['Dimension']}\n")
            num = len(self.compositions) * self.seeds_config.get("Random Seeds Num", 100)
            f.write(f"Random Seeds Num: {num}\n")
            f.write(f"Valid Seeds Num: {connect(random_seeds_gen_db).count()}\n")
            f.write(f"Vacuum Layer Thickness: {self.seeds_config.get('Vacuum Layer Thickness', 10)}\n")
            f.write(f"Time Cost: {round(time.time() - start, 2)} s\n")
            f.write("The random seed structures have been successfully generated and optimized !!! \n")
        return

    def dp_train_nn(self, nn_id, gpu_set, precision="normal"):
        """
        提交训练NN
        """
        # 创建workdir/dp/nn0目录
        nn_dir = os.path.join(self.dp_dir, f"nn{nn_id}")
        if os.path.exists(nn_dir):
            shutil.rmtree(nn_dir)
        os.makedirs(nn_dir)
        gpu_set = [str(i) for i in gpu_set]
        # 训练初始模型，更新best_model
        dp_input_temp = os.path.join(self.templates, "trainer/deepmd_input.json")
        if precision == "accurate":
            dp_input_temp = os.path.join(self.templates, "trainer/deepmd_input_accurate.json")
        DpTrainer = DeepmdSystem(elements=self.elements,
                                 db_path=self.db_path,
                                 workdir=nn_dir,
                                 gpu=gpu_set,
                                 train_ratio=self.trainer_config.get("Train Ratio", 0.8),
                                 dp_input_template=dp_input_temp,
                                 init_model_path=self.best_model)
        DpTrainer.creat_dp_input()
        DpTrainer.train_models()
        with open(self.log, "a") as f:
            gpu_set = "  ".join(gpu_set)
            f.write(f"GPU: {gpu_set}\n")
            f.write(f"Init model: {self.best_model}\n")
            start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            if nn_id == self.base_config["Iterations"] + 1:
                f.write(f"{start_time} NN_Ultimate is training ...\n")
            else:
                f.write(f"{start_time} NN_{nn_id} is training ...\n")
        return DpTrainer

    def dp_monitor(self, trainer, iter_id):
        """
        监控NN训练状态
        若训练完成，冻结模型，更新best_model
        """
        train_state = trainer.monitor_training()
        while not all(train_state):
            time.sleep(30)
            train_state = trainer.monitor_training()
            print("NN training state: ", train_state)

        frozen_state = trainer.freeze_models()
        fail = [str(i) for i, j in enumerate(frozen_state) if not j]
        best_model_path, best_model_id, models_info = trainer.get_best_model()
        self.best_model = best_model_path
        self.best_model_info = models_info[best_model_id]

        # 写入日志 NN_1 Training Results
        with open(self.log, "a") as f:
            f.write(self.create_separator_line(f" NN_{iter_id} Training Results ", total_length=100, separator='-'))
            f.write("model_id  rmse_val  rmse_trn  rmse_e_val  rmse_e_trn  rmse_f_val  rmse_f_trn  state\n")
            for ii, k in enumerate(["000", "001", "002", "003"]):
                v = models_info.get(k, ["Nan", "Nan", "Nan", "Nan", "Nan", "Nan"])
                v = "     ".join([f"{i:.4f}" for i in v])
                v += "      " + str(train_state[ii])
                f.write(f"  {k}      {v}\n")
            f.write(f"Best Model: {best_model_path}\n")
            if fail:
                fail_str = " ".join(fail)
                f.write(f"Warning: Model {fail_str} failed to freeze !!!\n")
            else:
                f.write("All models have been successfully frozen !!!\n")

    def run_ga(self, ga_id, gpu, composition, result_queue):
        """
        启动GA
        """
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
        # 创建workdir/pes/ga/1目录
        workdir = os.path.join(self.ga_dir, f"ga{ga_id}")
        if not os.path.exists(workdir):
            os.makedirs(workdir)

        # 获取unique_mark
        chemformula = "".join(f"{ele}{num}" for ele, num in zip(self.elements, composition))
        dim = self.uspex_config["Dimension"]
        unique_mark = f"{dim}_{chemformula}"
        multi_substrates = self.uspex_config.get("Multi Substrates", False)
        if multi_substrates:
            substrate_index = os.path.basename(self.uspex_config["Substrate"]).split("_")[-1]
            unique_mark += f"_{substrate_index}"

        # 更新init_seeds_db
        if ga_id > 1:
            for i in range(ga_id, 2, -1):
                compos_dir = os.path.join(self.ga_dir, f"ga{i - 1}", unique_mark)
                success_flag = os.path.join(compos_dir, "USPEX_IS_DONE")
                if os.path.exists(success_flag):
                    subdirectories = []
                    for root, dirs, files in os.walk(compos_dir):
                        # 获取子目录名，只返回当前目录的子目录，避免递归深度遍历
                        subdirectories.extend(dirs)
                        break
                    subdirectories = [int(i[7:]) for i in subdirectories if i.startswith("results")]
                    results_id = max(subdirectories)
                    results_path = os.path.join(compos_dir, f"results{results_id}")
                    good_structures = os.path.join(results_path, "goodStructures_POSCARS")
                    self.init_seeds_db = good_structures
                    print("init_seeds_db: ", self.init_seeds_db)
                    break

        # 生成GA输入文件
        ga_system = UspexSystem(elements=self.elements,
                                target_composition=composition,
                                dimension=self.uspex_config["Dimension"],
                                workdir=workdir,
                                uspex_templates_dir=os.path.join(self.templates, "uspex"),
                                model_path=self.best_model,
                                generation_num=self.uspex_config["Generation Num"],
                                pop_size=self.uspex_config["Pop Size"],
                                ini_pop_size=self.uspex_config["Init Pop Size"],
                                substrate_path=self.uspex_config.get("Substrate", None),
                                random_seeds_path=self.random_seeds_db,
                                init_seeds_path=self.init_seeds_db,
                                multi_substrates=self.uspex_config.get("Multi Substrates", False),
                                calculator=self.model_tag,
                                gpu=gpu
                                )
        ga_system.create_uspex_input()

        # 启动GA
        compos_dir = os.path.join(workdir, ga_system.unique_mark)
        uspex_env = self.uspex_config["USPEX Env"]
        uspex_env_path = os.path.expanduser(uspex_env)
        env = os.environ.copy()
        env["PATH"] = f"{uspex_env_path}:{env['PATH']}"
        cmd = "nohup USPEX -r > uspex.log 2>&1 & echo $! > uid"
        subprocess.run(cmd, shell=True, cwd=compos_dir, env=env, check=True)

        # 写入日志 GA
        with open(self.log, "a") as f:
            f.write(f"Gpu: {gpu}\n")
            f.write(f"Model: {self.model_tag}\n")
            f.write(f"Composition: {ga_system.unique_mark}\n")
            start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            f.write(f"{start_time} GA_{ga_id} {ga_system.unique_mark} is running ...\n")
            f.write("\n")
        result_queue.put(ga_system)
        return ga_system

    def run_ga_0(self):
        """
        在没有初始数据的情况下进行GA搜索
        """
        with open(self.log, "a") as f:
            f.write(self.create_separator_line("GA_0", total_length=100, separator='-'))

        gpus = []
        n = len(self.gpu) * 3
        compos_batch = [self.compositions[i:i+n] for i in range(0, len(self.compositions), n)]
        for compos in compos_batch:
            for i, c in enumerate(compos):
                ii = i % len(self.gpu)
                gpus.append(self.gpu[ii])

            ga0_systems = []
            processes = []
            result_queue = mp.Queue()
            for i, c in enumerate(compos):
                p = mp.Process(target=self.run_ga, args=(0, gpus[i], c, result_queue))
                p.start()
                processes.append(p)
            for p in processes:
                p.join()
            while not result_queue.empty():
                ga0_systems.append(result_queue.get())

            time.sleep(180)
            GA0_IS_FINISH = False
            ga0_states = []
            while not GA0_IS_FINISH:
                ga0_states = []
                for ga in ga0_systems:
                    ga0_states.append(ga.uspex_monitor())
                if not "RUNNING" in ga0_states:
                    GA0_IS_FINISH = True
                else:
                    time.sleep(180)

        # 提取GA0的结果
        with open(self.log, "a") as f:
            f.write(self.create_separator_line(f"GA_0 Results", total_length=100, separator='-'))
            start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            f.write(f"{start_time} All GA_0 tasks have been completed successfully !!!\n")
            compos_str = " ".join(
                ["".join([f"{ele}{num}" for ele, num in zip(self.elements, c)]) for c in self.compositions])
            f.write(f"Compositions: {compos_str}\n")
            state_str = "  ".join(ga0_states)
            f.write(f"GA state: {state_str}\n")

        # 提取GA0的结果
        gathered_db_path = self.postprocess(0)

        # 提交 sp + opt 作业， 在初始化阶段充分生成数据
        # 提交 sp 作业
        dft_sp = VaspjetRun(db_path=gathered_db_path,
                            cpu_config=self.cpu_config,
                            cpu_workdir=os.path.join(self.cpu_config["CPU Working Directory"], "iter0/sp"),
                            vaspjet_yml=os.path.join(self.templates, "vaspjet/config_sp.yml"))
        # ！！！没有开启 vasp ! ！！!
        dft_sp.run_vaspjet()

        # 筛选出一批最稳定的结构进行 opt -> alls_1.db
        energy_structure_filter(db_path=gathered_db_path,
                                dimension=self.dimension,
                                max_filter_ratio=self.postprocess_config["Max Filter Ratio"],
                                max_filter_num=self.postprocess_config["Max Filter Num"],
                                similarity_threshold=0.95,
                                output_mode="split",
                                constraint_z=self.constraint_z)

        with open(self.log, "a") as f:
            f.write(f"GA_0 results have been submitted to the cpu for sp-dft calculation !!!\n")
        time.sleep(120)

        alls_1 = gathered_db_path.replace(".db", "_1.db")
        dft_opt = VaspjetRun(db_path=alls_1,
                            cpu_config=self.cpu_config,
                            cpu_workdir=os.path.join(self.cpu_config["CPU Working Directory"], "iter0/opt"),
                            vaspjet_yml=os.path.join(self.templates, "vaspjet/config_opt.yml"))
        dft_opt.run_vaspjet()

        with open(self.log, "a") as f:
            f.write(f"GA_0 results have been submitted to the cpu for opt-dft calculation !!!\n")
        time.sleep(120)

        # 监控dft计算状态
        ga_dir_iter = os.path.join(self.ga_dir, "ga0")
        sp_results_db = os.path.join(ga_dir_iter, "sp.db")
        opt_results_db = os.path.join(ga_dir_iter, "opt.db")
        next_iter_db = os.path.join(ga_dir_iter, "next_iter.db")
        sp_opt_results = False
        while not sp_opt_results:
            time.sleep(180)
            sp_state, sp_download = vaspjet_monitor(cpu_config=self.cpu_config,
                                                    cpu_workdir=os.path.join(self.cpu_config["CPU Working Directory"],
                                                                             "iter0/sp"),
                                                    download_results=True,
                                                    local_path=sp_results_db,
                                                    traj_process_mode=None
                                                    )
            opt_states, opt_download = vaspjet_monitor(cpu_config=self.cpu_config,
                                                       cpu_workdir=os.path.join(self.cpu_config["CPU Working Directory"],
                                                                                "iter0/opt"),
                                                       download_results=False,
                                                       local_path=opt_results_db,
                                                       traj_process_mode=None
                                                       )
            if sp_state == "DONE" and sp_download:
                if opt_states == "DONE":
                    sp_opt_results = True
            print("sp_state: ", sp_state)
            print("opt_states: ", opt_states)

        opt_states, opt_download = vaspjet_monitor(cpu_config=self.cpu_config,
                                                   cpu_workdir=os.path.join(self.cpu_config["CPU Working Directory"],
                                                                            "iter0/opt"),
                                                   download_results=True,
                                                   local_path=opt_results_db,
                                                   traj_process_mode="filter"
                                                   )
        if opt_download:
            next_iter = connect(next_iter_db)
            for row in connect(sp_results_db):
                atoms = row.toatoms()
                next_iter.write(atoms, data=row.data, key_value_pairs=row.key_value_pairs)
            for row in connect(opt_results_db):
                atoms = row.toatoms()
                next_iter.write(atoms, data=row.data, key_value_pairs=row.key_value_pairs)
            with open(self.log, "a") as f:
                f.write(f"GA_0 dft calculation has been completed successfully !!!\n")
        # 提取dft计算结果，更新database
        self.db_path = next_iter_db
        return

    def initialize(self):
        """
        初始化：
        若无初始模型，则训练初始模型NN0，更新best_model
        若无随机种子结构文件，则启动随机种子结构生成器
        """
        with open(self.log, "a") as f:
            f.write(self.create_separator_line(" Initialization ", total_length=100, separator='='))

        # 处理初始种子结构
        if self.init_seeds_db is not None and self.uspex_config["Dimension"] == 2:
            # 删除cell大于substrate的1.4*cell的结构
            sub = read(self.uspex_config["Substrate"])
            cell = sub.get_cell()
            cell_x = cell[0][0] * 1.4
            cell_y = cell[1][1] * 1.4
            cell_z = cell[2][2] + 20
            for row in connect(self.init_seeds_db).select():
                atoms = row.toatoms()
                seeds_cell = atoms.get_cell()
                if seeds_cell[0][0] > cell_x or seeds_cell[1][1] > cell_y or seeds_cell[2][2] > cell_z:
                    connect(self.init_seeds_db).delete([row.id])
            print("Valid init seeds: ", connect(self.init_seeds_db).count())

        # 开启随机种子生成器，提交种子优化任务，更新self.random_seeds_db
        if self.seeds_config.get("Activate", False):
            self.randomseeds_generator()

        # 处理初始给定的随机种子结构
        if connect(self.random_seeds_db).count() > 0 and self.uspex_config["Dimension"] == 2:
            # 删除cell大于substrate的1.4*cell的结构
            sub = read(self.uspex_config["Substrate"])
            cell = sub.get_cell()
            cell_x = cell[0][0] * 1.4
            cell_y = cell[1][1] * 1.4
            cell_z = cell[2][2] + 20
            for row in connect(self.random_seeds_db).select():
                atoms = row.toatoms()
                seeds_cell = atoms.get_cell()
                if seeds_cell[0][0] > cell_x or seeds_cell[1][1] > cell_y or seeds_cell[2][2] > cell_z:
                    connect(self.random_seeds_db).delete([row.id])
            print("Valid random seeds: ", connect(self.random_seeds_db).count())

        # 若无初始数据集，则进行GA0搜索，并提交进行DFT计算获取初始数据集
        if not self.db_path or connect(self.db_path).count() == 0:
            self.run_ga_0()

        with open(self.log, "a") as f:
            f.write("Initialization is complete\n")
            f.write(f"Model: {self.best_model}\n")
            f.write(f"Data: {self.db_path}")
            f.write("\n")
        print("database_path: ", self.db_path)
        print("best_model: ", self.best_model)
        print("random_seeds: ", self.random_seeds_db)
        return

    def postprocess(self, iter_id):
        """
        回收USPEX运行结果 -> 对回收结果进行去冗余 -> 用4个模型综合评估结构是否学会
        """
        with open(self.log, "a") as f:
            f.write(self.create_separator_line(f"Postprocessing", total_length=100, separator='-'))
        ga_dir_iter = os.path.join(self.ga_dir, f"ga{iter_id}")
        subdirectories = []
        for root, dirs, files in os.walk(ga_dir_iter):
            # 获取子目录名，只返回当前目录的子目录，避免递归深度遍历
            subdirectories.extend(dirs)
            break
        alls_db_path = os.path.join(ga_dir_iter, "alls.db")
        alls_db = connect(alls_db_path)
        if os.path.exists(alls_db_path):
            os.remove(alls_db_path)

        with open(self.log, "a") as f:
            f.write(f"Started screening structures based on energy and similarity...\n")

        for subdir in subdirectories:
            # # 简单去重得到ga/ga1/2_O2Cu18/gathered.db
            gathered_db_path = write_to_db(os.path.join(ga_dir_iter, subdir), self.constraint_z)

            # 若uspex失败，无gathered.db，则跳过
            if not os.path.exists(gathered_db_path):
                print(f"Warning: {gathered_db_path} does not exist.")
                continue
            # 能量和力筛选，得到去重后的 ga/1/O2Cu18_2/gathered.db
            gathered_count = connect(gathered_db_path).count()
            energy_structure_filter(db_path=gathered_db_path,
                                    dimension=self.dimension,
                                    max_filter_ratio=0.9,
                                    similarity_threshold=0.95,
                                    output_mode="delete",
                                    constraint_z=self.constraint_z)
            filtered_db_path = gathered_db_path
            filtered_count = connect(filtered_db_path).count()
            with open(self.log, "a") as f:
                f.write(f"Composition: {os.path.basename(subdir)}\n")
                f.write(f"Gathered: {gathered_count}\n")
                f.write(f"Filtered: {filtered_count}\n")
                f.write("\n")
            filtered_db = connect(filtered_db_path)
            for row in filtered_db.select():
                alls_db.write(row.toatoms(), data=row.data, key_value_pairs=row.key_value_pairs)

        if iter_id == 0:
            print(f"GA_0 searched {alls_db.count()} structures in total.")
            return alls_db_path

        # 用训练好的模型对GA搜索结果进行评估
        with open(self.log, "a") as f:
            f.write(f"Start wrong atom identification of the structure based on NN_{iter_id} evaluation...\n")

        force_deviation_lower = self.postprocess_config.get("Force Deviation Lower", "Auto")
        force_deviation_upper = self.postprocess_config.get("Force Deviation Upper", "Auto")
        if force_deviation_lower == "Auto":
            force_deviation_lower = float(self.best_model_info[5]) # validation RMSEf
        if force_deviation_upper == "Auto":
            force_deviation_upper = force_deviation_lower + 0.15

        nn_deviation = NNDeviation(model_dir=os.path.join(self.dp_dir, f"nn{iter_id}"),
                                   ga_db_path=alls_db_path,
                                   force_err_lower=force_deviation_lower,
                                   force_err_upper=force_deviation_upper,
                                   type=self.postprocess_config.get("Type", "slab"),
                                   lcs_radius=self.postprocess_config.get("LCS Radius", 5.0),
                                   lcs_layers_num=self.postprocess_config.get("LCS Layers Num", 3)
                                   )
        nn_deviation.get_deviation()
        nn_dev_result = nn_deviation.dev_results
        with open(self.log, "a") as f:
            f.write(f"Accurate:  {nn_dev_result['Accurate'][0]}  {nn_dev_result['Accurate'][1]}\n")
            f.write(f"Candidate: {nn_dev_result['Candidate'][0]}  {nn_dev_result['Candidate'][1]}\n")
            f.write(f"Failed:    {nn_dev_result['Failed'][0]}  {nn_dev_result['Failed'][1]}\n")
            f.write("\n")
        self.accurate_ratio = float(nn_dev_result['Accurate'][1])
        self.failed_ratio = float(nn_dev_result['Failed'][1])

        # 判断是否需要进行lcs处理
        lcs_flag = self.postprocess_config.get("LCS Process", False)
        if lcs_flag:
            with open(self.log, "a") as f:
                f.write(f"Start local coordination structure processing...\n")
            lcs_count = nn_deviation.lcs_process(os.path.join(ga_dir_iter, "candidates.db"))
            with open(self.log, "a") as f:
                f.write(f"Local Coordination Structure Processed: {lcs_count}\n")

        print("candidates.db has been successfully postprocessed !!!")
        return os.path.join(ga_dir_iter, "candidates.db")

    def update(self):
        with open(self.log, "a") as f:
            f.write(self.create_separator_line("Update", total_length=100, separator='='))
        nn_trainer = self.dp_train_nn(self.base_config["Iterations"] + 1, self.gpu[:4], precision="accurate")
        self.dp_monitor(nn_trainer, self.base_config["Iterations"] + 1)
        end_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        with open(self.log, "a") as f:
            f.write(f"The final data set path is {self.db_path} containing {connect(self.db_path).count()} data.\n")
            f.write(f"End Time: {end_time}\n")
            f.write("All tasks have been completed successfully !!!\n")
        if os.path.exists(os.path.join(cwd, "warnings.log")):
            os.remove(os.path.join(cwd, "warnings.log"))

    def run_dp_GA(self):
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(i) for i in self.gpu])
        mp.set_start_method('spawn', force=True)
        start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        with open(self.log, "w") as f:
            f.write(f"Start Time: {start_time}\n")

        # 初始化
        self.initialize()
        # ==================================Main Loop==================================
        for iter_id in range(1, self.base_config["Iterations"] + 1):
            # 分batch
            seeds_compositions = self.compositions
            b1 = (len(self.gpu) - 4) * 3
            batch1 = seeds_compositions[:b1]
            batch2 = seeds_compositions[b1:]

            batch1_str = ", ".join(["".join([f"{ele}{num}" for ele, num in zip(self.elements, c)]) for c in batch1])
            if batch2:
                batch2_str = ", ".join(["".join([f"{ele}{num}" for ele, num in zip(self.elements, c)]) for c in batch2])
            else:
                batch2_str = "None"

            # -----------------------------------batch1-----------------------------------
            with open(self.log, "a") as f:
                f.write(self.create_separator_line(f" Iteration {iter_id} "))
                f.write(f"Batch1: {batch1_str}\n")
                f.write(f"Batch2: {batch2_str}\n")

            # 启动NN训练
            with open(self.log, "a") as f:
                f.write(self.create_separator_line(f" NN_{iter_id} Training ", total_length=100, separator='-'))
            nn_trainer = self.dp_train_nn(iter_id, self.gpu[:4])

            # 启动GA搜索
            # gpu不够的情况，等待nn训练完进行GA搜索
            if len(self.gpu) <= 4:
                dp_state = [False, False, False, False]
                while False in dp_state:
                    time.sleep(60)
                    dp_state = nn_trainer.monitor_training()

            if len(self.gpu) <= 4:
                calculator_gpu = 0
                ga_gpu1 = self.gpu
            else:
                calculator_gpu = 4
                ga_gpu1 = self.gpu[4:]

            # GA引擎选择
            if self.model_tag == "MACE" and self.best_model is not None and self.db_path is not None:
                self.model_tag = calculator_select(workdir=os.path.join(self.ga_dir, "tmp"),
                                                   db_path=self.db_path,
                                                   dp_model_path=self.best_model,
                                                   gpu_id=calculator_gpu)

            # 启动BATCH1 GA搜索
            with open(self.log, "a") as f:
                f.write(self.create_separator_line(f" GA_{iter_id} Searching ", total_length=100, separator='-'))
            ga1_systems = []
            processes = []
            result_queue = mp.Queue()
            ga_gpus_1 = []
            for i in range(len(batch1)):
                ga_gpus_1.append(ga_gpu1[i % len(ga_gpu1)])
            for i, c in enumerate(batch1):
                p = mp.Process(target=self.run_ga, args=(iter_id, ga_gpus_1[i], c, result_queue))
                p.start()
                processes.append(p)
            for p in processes:
                p.join()
            while not result_queue.empty():
                ga1_systems.append(result_queue.get())

            # 监控运行状态
            # ga_state："DONE" "RUNNING" "FAILED"
            BATCH1_IS_FINISH = False
            ga1_states = []
            while not BATCH1_IS_FINISH:
                ga1_states = []
                for ga in ga1_systems:
                    ga1_states.append(ga.uspex_monitor())
                # dp_state: [False/"Error", True, True, True]
                dp_state = nn_trainer.monitor_training()
                if all(dp_state) and not "RUNNING" in ga1_states:
                    BATCH1_IS_FINISH = True
                else:
                    time.sleep(60)
            self.dp_monitor(nn_trainer, iter_id)
            state = ga1_states

            # -----------------------------------batch2-----------------------------------
            if batch2:
                # 启动BATCH2 GA搜索
                with open(self.log, "a") as f:
                    f.write(self.create_separator_line(f"GA_{iter_id} Searching", total_length=100, separator='-'))
                ga2_states = []
                ga_gpu2 = self.gpu
                ga2_systems = []
                processes = []
                ga_gpus_2 = []
                for i in range(len(batch2)):
                    ga_gpus_2.append(ga_gpu2[i % len(ga_gpu2)])
                result_queue = mp.Queue()
                for i, c in enumerate(batch2):
                    p = mp.Process(target=self.run_ga, args=(iter_id, ga_gpus_2[i], c, result_queue))
                    p.start()
                    processes.append(p)
                for p in processes:
                    p.join()
                while not result_queue.empty():
                    ga2_systems.append(result_queue.get())

                # 监控运行状态
                BATCH2_IS_FINISH = False
                while not BATCH2_IS_FINISH:
                    ga2_states = []
                    for ga in ga2_systems:
                        ga2_states.append(ga.uspex_monitor())
                    if not "RUNNING" in ga2_states:
                        BATCH2_IS_FINISH = True
                    else:
                        time.sleep(60)
                state.extend(ga2_states)

            with open(self.log, "a") as f:
                f.write(self.create_separator_line(f"GA_{iter_id} Results", total_length=100, separator='-'))
                start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                f.write(f"{start_time} All GA tasks have been completed successfully !!!\n")
                compos_str = " ".join(
                 ["".join([f"{ele}{num}" for ele, num in zip(self.elements, c)]) for c in self.compositions])
                f.write(f"Compositions: {compos_str}\n")
                state_str = "  ".join(state)
                f.write(f"GA state: {state_str}\n")

            # ----------------------------------- MD Sampling -----------------------------------
            # force 在训练集和验证集上的表现都达标后才能进行 MD 采样
            ga_dir_iter = os.path.join(self.ga_dir, f"ga{iter_id}")
            alls_db_path = os.path.join(ga_dir_iter, "alls.db")
            alls_db = connect(alls_db_path)
            if self.best_model_info is not None:
                if (self.best_model_info[4] <= self.nnmd_config.get('NN Force Accuracy', 0.15)
                        and self.best_model_info[5] <= self.nnmd_config.get('NN Force Accuracy', 0.15)):
                    # 最终要汇入 alls_db_path
                    md_dir = os.path.join(ga_dir_iter, "MD")
                    if not os.path.exists(md_dir):
                        os.makedirs(md_dir)
                    to_md_db_path = os.path.join(md_dir, "to_md.db")
                    to_md_db = connect(to_md_db_path)
                    to_md_atoms = []
                    ga_compositions_dir = [os.path.join(ga_dir_iter, d) for d in os.listdir(ga_dir_iter)
                                if os.path.isdir(os.path.join(ga_dir_iter, d)) and not d.startswith("new") and not d.startswith("MD")]
                    for ga_compos in ga_compositions_dir:
                        gathered_db = connect(os.path.join(ga_compos, "gathered.db"))
                        if gathered_db.count() == 0:
                            continue
                        _infos = []
                        for row in gathered_db.select():
                            fitness = row.data['fitness']
                            _infos.append((row.id, fitness))
                        _infos.sort(key=lambda x: x[1])
                        stable_row = gathered_db.get(_infos[0][0])
                        stable_atoms = stable_row.toatoms()
                        to_md_atoms.append(stable_atoms)
                        to_md_db.write(stable_atoms, data=stable_row.data, key_value_pairs=stable_row.key_value_pairs)

                    # 生成 MD 任务
                    with open(self.log, "a") as f:
                        start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                        f.write(self.create_separator_line(f"MD_{iter_id} Sampling", total_length=100, separator='-'))
                        f.write(f"Number of structures to be sampled: {len(to_md_atoms)}\n")
                        f.write(f"{start_time} Start sampling...\n")
                    run_md_parallel(
                        atoms_list=to_md_atoms,
                        type_map=self.elements,
                        dp_model_path=self.best_model,
                        base_workdir=md_dir,
                        nproc=len(self.gpu)*2,
                        nsteps=self.nnmd_config.get('MD Steps', 20000),
                        timestep_fs=self.nnmd_config.get('MD Timestep', 0.5),
                        dump_interval=self.nnmd_config.get('MD Dump Interval', 100),
                        cpu_only_inference=False,
                        temperature_K=self.nnmd_config.get('MD Temperature K', 500),
                        gpu_ids=self.gpu,
                    )
                    with open(self.log, "a") as f:
                        end_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                        f.write(f"{end_time} MD sampling completed.\n")

                    # 回收 md 轨迹，并写入 alls.db
                    gathered_md_traj_path = gather_md_traj(md_dir, self.best_model, self.gpu[0])
                    gathered_md_traj = connect(gathered_md_traj_path)
                    for row in gathered_md_traj.select():
                        atoms = row.toatoms()
                        alls_db.write(atoms, data=row.data, key_value_pairs=row.key_value_pairs)
                    with open(self.log, "a") as f:
                        f.write(f"A total of {gathered_md_traj.count()} MD trajectories have been written to {alls_db_path} !\n")

            # -----------------------------------PostProcessing-----------------------------------
            candidates_db_path = self.postprocess(iter_id)
            candidates_count = connect(candidates_db_path).count()
            with open(self.log, "a") as f:
                f.write(self.create_separator_line(f"DFT Labeling", total_length=100, separator='-'))

            if candidates_count > 0:
                if iter_id == self.base_config["Iterations"]:
                    IS_LAST_ITER = True
                else:
                    IS_LAST_ITER = False
                new_database = dft(iter_id=iter_id,
                                   config=self.config,
                                   model_path=self.best_model,
                                   log_path=self.log,
                                   is_last_iter=IS_LAST_ITER)
                self.db_path = new_database

            if self.accurate_ratio > self.accuracy_threshold:
                self.converge_count += 1
            else:
                self.converge_count = 0

            if self.converge_count >= self.stall_iterations:
                with open(self.log, "a") as f:
                    f.write(f"Accurate ratio: {self.accurate_ratio}\n")
                    f.write(f"Failed ratio: {self.failed_ratio}\n")
                    f.write(f"Convergence criteria met. Iteration: {iter_id}\n")
                break
        # ==================================Main Loop Finish==================================
        # 最后一次训练，更新best_model
        self.update()

        return

if __name__ == '__main__':
    with open("/home/yliu/cchen/CuClO/config.yml") as file:
        dict_value = yaml.load(file.read(), Loader=yaml.FullLoader)
    config = dict_value
    hiccup = Hiccup(config)
    # hiccup.run_dp_GA()
    hiccup.initialize()
    hiccup.postprocess(0)
    if os.path.exists(os.path.join(cwd, 'warnings.log')):
        os.remove(os.path.join(cwd, 'warnings.log'))
