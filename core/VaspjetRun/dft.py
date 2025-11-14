"""
DFT计算
1.将candidates.db划分为candidates_1.db, candidates_2.db
2.提交candidates_1.db到cpu进行sp计算，等待计算完成得到sp_1.db，同时进行数据清洗
3.提交candidates_2.db到cpu进行sp计算
4.将sp_2.db提交进行优化计算
5.筛选出需要进行md计算的结构，提交md计算
"""
import os
import logging
import shutil

cwd = os.getcwd()
logging.basicConfig(filename=os.path.join(cwd, 'warnings.log'),
                    level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logging.captureWarnings(True)

import time
from ase.db import connect
from VaspjetRun.vaspjet_run import VaspjetRun, vaspjet_monitor, kill_vaspjet
from PesExploration.tools.data_filter_and_analysis import data_filter_and_analysis, plt_out
from PesExploration.tools.energy_structure_filter import energy_structure_filter, split_db


def merge_dbs(out_db_path, db_list):
    """
    合并多个db文件
    合并的筛选前提：文件存在，文件不为空
    """
    if os.path.exists(out_db_path):
        os.remove(out_db_path)
    out_db = connect(out_db_path)
    for db_path in db_list:
        if os.path.exists(db_path):
            db = connect(db_path)
            if db.count() > 0:
                try:
                    for row in db.select():
                        atoms = row.toatoms()
                        out_db.write(atoms, data=row.data, key_value_pairs=row.key_value_pairs)
                except Exception as e:
                    print(f"Error DB: {db_path}")
                    print(f"Error: {e}")
                    continue
            else:
                continue
        else:
            continue
    return


def dft(iter_id,
        config,
        model_path,
        log_path,
        is_last_iter=False):
    ga_dir = os.path.join(config["BASE"]["Workdir"], "pes/ga")
    dp_dir = os.path.join(config["BASE"]["Workdir"], "dp")
    cpu_config = config["CPU"]
    postprocess_config = config["SAMPLING"]["GA"]["POSTPROCESSING"]
    templates_path = config["BASE"]["Templates"]
    gpu = config["BASE"]["Gpu"]
    ga_dir_iter = os.path.join(ga_dir, f"ga{iter_id}")
    MD_FLAG = postprocess_config.get("AIMD", False)

    # 1.将candidates.db划分为candidates_1.db, candidates_2.db
    to_be_sp_db = os.path.join(ga_dir_iter, "candidates.db")
    remote_dft_dir = os.path.join(cpu_config["CPU Working Directory"], f"iter{iter_id}")
    # 对candidates.db中的结构进行筛选，得到candidates_1.db, candidates_2.db
    to_be_sp1 = os.path.join(ga_dir_iter, f"{os.path.basename(to_be_sp_db).split('.')[0]}_1.db")
    to_be_sp2 = os.path.join(ga_dir_iter, f"{os.path.basename(to_be_sp_db).split('.')[0]}_2.db")
    if os.path.exists(to_be_sp1):
        os.remove(to_be_sp1)
    if os.path.exists(to_be_sp2):
        os.remove(to_be_sp2)
    energy_structure_filter(db_path=to_be_sp_db,
                            best_model_path=model_path,
                            max_filter_ratio=postprocess_config.get("Max Filter Ratio", 0.8),
                            max_filter_num=postprocess_config.get("Max Filter Num", 100),
                            similarity_threshold=0.9,
                            output_mode="split")
    count1 = connect(to_be_sp1).count()
    count2 = connect(to_be_sp2).count()
    with open(log_path, "a") as f:
        f.write("Energy and structure filtering results:\n")
        f.write(f"SP_1: {count1}\n")
        f.write(f"SP_2: {count2}\n")
        f.write("\n")

    # 2.1 检查sp_2, opt_1, md作业的状态，并杀死未完成的作业
    _iter_id = int(iter_id) - 1
    _ga_dir_iter = os.path.join(ga_dir, f"ga{_iter_id}")
    _candidates_1_db = os.path.join(_ga_dir_iter, "candidates_1.db")
    _candidates_2_db = os.path.join(_ga_dir_iter, "candidates_2.db")
    _md_db = os.path.join(_ga_dir_iter, "md.db")
    _remote_dft_dir = os.path.join(cpu_config["CPU Working Directory"], f"iter{_iter_id}")
    # sp_2, opt_1, md作业结果下载的路径
    _sp_2_db = os.path.join(_ga_dir_iter, "sp_2.db")
    _opt_db = os.path.join(_ga_dir_iter, "opt.db")
    _opt_lm_db = os.path.join(_ga_dir_iter, "opt_lm.db")
    _md_db = os.path.join(_ga_dir_iter, "md.db")
    _md_lm_db = os.path.join(_ga_dir_iter, "md_lm.db")

    # ===========检查上一代作业的完成情况==========
    if os.path.exists(_ga_dir_iter):
        # SP_2
        if os.path.exists(_candidates_2_db):
            with open(log_path, "a") as f:
                f.write("Checking the status of the SP_2, OPT_1, and MD jobs:\n")
            sp2_state = "RUNNING"
            sp2_download = False
            while sp2_state == "RUNNING":
                sp2_state, sp2_download = vaspjet_monitor(cpu_config=cpu_config,
                                                          cpu_workdir=os.path.join(_remote_dft_dir, "sp2"),
                                                          download_results=True,
                                                          local_path=_sp_2_db,
                                                          traj_process_mode=None)
                time.sleep(180)
            with open(log_path, "a") as f:
                f.write(f"SP_2: {sp2_state}, download: {sp2_download}\n")

        # OPT
        if os.path.exists(_candidates_2_db):
            opt1_state = "RUNNING"
            opt1_download = False
            while opt1_state == "RUNNING":
                opt1_state, opt1_download = vaspjet_monitor(cpu_config=cpu_config,
                                                            cpu_workdir=os.path.join(_remote_dft_dir, "opt"),
                                                            download_results=False)
                time.sleep(180)
            with open(log_path, "a") as f:
                f.write(f"OPT: {opt1_state}, download: {opt1_download}\n")
                f.write("\n")

            # 待作业完成后，下载结果
            opt1_state, opt1_download = vaspjet_monitor(cpu_config=cpu_config,
                                                        cpu_workdir=os.path.join(_remote_dft_dir, "opt"),
                                                        download_results=True,
                                                        local_path=_opt_db,
                                                        traj_process_mode="filter",
                                                        max_force_threshold=20)
            with open(log_path, "a") as f:
                f.write(f"OPT: {opt1_state}, download: {opt1_download}\n")
            opt1_lm_state, opt1_lm_download = vaspjet_monitor(cpu_config=cpu_config,
                                                              cpu_workdir=os.path.join(_remote_dft_dir, "opt"),
                                                              download_results=True,
                                                              local_path=_opt_lm_db,
                                                              traj_process_mode=None)
            with open(log_path, "a") as f:
                f.write(f"OPT_LM: {opt1_lm_state}, download: {opt1_lm_download}\n")
                f.write("\n")

        # MD
        if MD_FLAG == True and os.path.exists(os.path.join(_ga_dir_iter, "to_run_md.db")):
            md_state, md_download = vaspjet_monitor(cpu_config=cpu_config,
                                                    cpu_workdir=os.path.join(_remote_dft_dir, "md"),
                                                    download_results=True,
                                                    local_path=_md_db,
                                                    traj_process_mode="filter",
                                                    max_force_threshold=50)

            if md_state == "RUNNING":
                kill_vaspjet(cpu_config=cpu_config,
                             cpu_workdir=os.path.join(_remote_dft_dir, "md"))
                time.sleep(180)
            with open(log_path, "a") as f:
                f.write(f"MD: {md_state}, download: {md_download}\n")
            md_state_lm, md_download_lm = vaspjet_monitor(cpu_config=cpu_config,
                                                          cpu_workdir=os.path.join(_remote_dft_dir, "md"),
                                                          download_results=True,
                                                          local_path=_md_lm_db,
                                                          traj_process_mode="last_image")
            with open(log_path, "a") as f:
                f.write(f"MD_LM: {md_state_lm}, download: {md_download_lm}\n")
                f.write("\n")
    else:
        pass

    # ===========提交这一代作业==========
    # 2.2 提交candidates_1.db到cpu进行sp计算
    dft_sp1 = VaspjetRun(db_path=to_be_sp1,
                         cpu_config=cpu_config,
                         cpu_workdir=os.path.join(remote_dft_dir, "sp1"),
                         vaspjet_yml=os.path.join(templates_path, "vaspjet/pure_vasp_sp.yml"))
    dft_sp1.run_vaspjet()

    with open(log_path, "a") as f:
        start = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        f.write(f"-Submit SP1-\n")
        f.write(f"Submission Time: {start}\n")
        f.write(f"{to_be_sp1} has been submitted to the cpu for DFT calculation.\n")
        f.write(f"Waiting for the SP calculation results...\n")
        f.write("\n")

    # 2.3 数据回收&数据清洗
    with open(log_path, "a") as f:
        f.write("-Clean Data-\n")
        f.write("Cleaning the initial database while waiting for the DFT calculation...\n")
    # 清洗初始数据集init.db
    iter_db = data_filter_and_analysis(workdir=os.path.join(dp_dir, f'nn{iter_id}/dbs'),
                                       model_path=model_path,
                                       gpu_ids=gpu,
                                       energy_filter=postprocess_config.get("Energy Filter", 0.1),
                                       force_filter=postprocess_config.get("Force Filter", 2)
                                       )
    # 清洗后更新db_path
    new_db_path = iter_db
    outdir = os.path.join(dp_dir, f'nn{iter_id}/dbs/out')
    model_name = model_path.split("/")[-2]
    if "00" not in model_name:
        model_name = "init_model"
    plt_out(outdir=outdir, model_name=model_name)

    with open(log_path, "a") as f:
        f.write("Initial data set cleaning is complete !!!\n")
        f.write(
            f"Data cleaning and analysis results can be found in {os.path.join(dp_dir, f'nn{iter_id}/dbs')}.\n")
        f.write("\n")

    # 2.4 下载sp1计算结果
    sp1_results_db = os.path.join(ga_dir_iter, "sp_1.db")
    sp1_results = False
    while not sp1_results:
        time.sleep(60)
        sp1_state, sp1_download = vaspjet_monitor(cpu_config=cpu_config,
                                                  cpu_workdir=os.path.join(remote_dft_dir, "sp1"),
                                                  download_results=True,
                                                  local_path=sp1_results_db,
                                                  traj_process_mode=None
                                                  )
        if sp1_state == "DONE" and sp1_download:
            sp1_results = True
        print("sp1_state: ", sp1_state)

    with open(log_path, "a") as f:
        end = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        f.write("-Get SP1 Results-\n")
        f.write(f"Finish Time: {end}\n")
        f.write(f"SP1 calculation results have been successfully downloaded to {sp1_results_db} !!!\n")
        f.write("\n")

    # 2.5 将sp_1.db, _sp_2.db, _opt.db, _md.db合并清洗
    new_data_dbs = [sp1_results_db, _sp_2_db, _opt_db, _md_db]
    new_data_db = os.path.join(ga_dir_iter, "new_opt_data.db")
    merge_dbs(new_data_db, new_data_dbs)
    # energy_structure_filter(db_path=new_data_db,
    #                         best_model_path=model_path,
    #                         max_filter_ratio=0.9,
    #                         similarity_threshold=0.95,
    #                         output_mode="delete")
    new_data_split_dir = os.path.join(ga_dir_iter, "new_data_split")
    os.mkdir(new_data_split_dir)
    split_db(new_data_db, new_data_split_dir)

    new_iter = data_filter_and_analysis(workdir=new_data_split_dir,
                                        model_path=model_path,
                                        gpu_ids=gpu,
                                        energy_filter=postprocess_config.get("Energy Filter", 0.1),
                                        force_filter=postprocess_config.get("Force Filter", 2)
                                        )
    new_iter_db = os.path.join(ga_dir_iter, "next_iter.db")
    merge_dbs(new_iter_db, [new_iter, new_db_path])

    # 更新db_path
    new_db_path = new_iter_db
    os.remove(new_data_db)

    if not is_last_iter:
        # 3.提交candidates_2.db到cpu进行sp计算
        dft_sp2 = VaspjetRun(db_path=to_be_sp2,
                             cpu_config=cpu_config,
                             cpu_workdir=os.path.join(remote_dft_dir, "sp2"),
                             vaspjet_yml=os.path.join(templates_path, "vaspjet/pure_vasp_sp.yml"))
        dft_sp2.run_vaspjet()
        with open(log_path, "a") as f:
            f.write("-Submit SP2-\n")
            start = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            f.write(f"Submission Time: {start}\n")
            f.write(f"{to_be_sp2} has been submitted to the cpu for DFT calculation.\n")
            f.write(f"Waiting for the SP calculation results...\n")
            f.write("\n")

        # 提交sp_1.db到cpu进行opt计算
        dft_opt = VaspjetRun(db_path=to_be_sp1,
                             cpu_config=cpu_config,
                             cpu_workdir=os.path.join(remote_dft_dir, "opt"),
                             vaspjet_yml=os.path.join(templates_path, "vaspjet/pure_vasp_opt.yml"))
        dft_opt.run_vaspjet()
        with open(log_path, "a") as f:
            f.write("-Submit OPT-\n")
            start = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            f.write(f"Submission Time: {start}\n")
            f.write(f"{to_be_sp1} has been submitted to the cpu for DFT calculation.\n")
            f.write(f"Waiting for the SP calculation results...\n")
            f.write("\n")

        # 筛选需要进行md计算的结构
        if MD_FLAG:
            _opt_lm_db = os.path.join(_ga_dir_iter, "opt_lm.db")
            _md_lm_db = os.path.join(_ga_dir_iter, "md_lm.db")
            to_run_md = os.path.join(ga_dir_iter, "to_run_md.db")
            if os.path.exists(to_run_md):
                os.remove(to_run_md)

            if not os.path.exists(_md_db):
                if postprocess_config.get("MD Init Data", None):
                    shutil.copy(postprocess_config.get("MD Init Data"), to_run_md)
            else:
                if os.path.exists(_opt_lm_db):
                    if os.path.exists(_md_lm_db):
                        merge_dbs(to_run_md, [_opt_lm_db, _md_lm_db])
                    else:
                        shutil.copy(_opt_lm_db, to_run_md)
                    energy_structure_filter(db_path=to_run_md,
                                            best_model_path=model_path,
                                            max_filter_ratio=postprocess_config.get("MD Max Filter Ratio", 0.5),
                                            max_filter_num=postprocess_config.get("MD Max Filter Num", 100),
                                            similarity_threshold=0.9,
                                            output_mode="delete")
            # 提交md计算
            print("to_run_md: ", to_run_md)
            if os.path.exists(to_run_md) and connect(to_run_md).count() > 0:
                dft_md = VaspjetRun(db_path=to_run_md,
                                    cpu_config=cpu_config,
                                    cpu_workdir=os.path.join(remote_dft_dir, "md"),
                                    vaspjet_yml=os.path.join(templates_path, "vaspjet/pure_vasp_md.yml"))
                dft_md.run_vaspjet()
                with open(log_path, "a") as f:
                    f.write("-Submit MD-\n")
                    start = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                    f.write(f"Submission Time: {start}\n")
                    f.write(f"{to_run_md} has been submitted to the cpu for DFT calculation.\n")
                    f.write(f"Waiting for the MD calculation results...\n")
                    f.write("\n")
    print(new_db_path)
    with open(log_path, "a") as f:
        f.write(f"New data set: {new_db_path}\n")
        f.write(f"Data volume of new database: {connect(new_db_path).count()}\n")
        f.write("\n")
    return new_db_path


if __name__ == '__main__':
    import yaml
    log = "/home/cchen/Train_NN/example/cluster/hiccup-log.txt"
    if os.path.exists(log):
        os.remove(log)
    with open("/home/cchen/Train_NN/example/cluster/config.yml") as file:
        dict_value = yaml.load(file.read(), Loader=yaml.FullLoader)
    config = dict_value
    dft(iter_id=0,
        config=config,
        model_path="/home/cchen/Train_NN/example/cluster/cluster_init_model.pb",
        log_path=log,
        is_last_iter=False)
    if os.path.exists(os.path.join(cwd, 'warnings.log')):
        os.remove(os.path.join(cwd, 'warnings.log'))
