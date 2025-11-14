import paramiko
import os
import time
from ase.db import connect


class VaspjetRun:
    def __init__(self,
                 db_path,
                 cpu_config,
                 cpu_workdir,
                 vaspjet_yml):
        self.db_path = db_path
        self.cpu_config = cpu_config
        self.vaspjet_yml = vaspjet_yml
        self.cpu_workdir = cpu_workdir

    @staticmethod
    def update_db(_db_path):
        db = connect(_db_path)
        new_db = os.path.dirname(_db_path) + "/new_" + os.path.basename(_db_path)
        new = connect(new_db)
        for row in db.select():
            atoms = row.toatoms()
            atoms.set_tags([1] * len(atoms))
            data = row.data
            kvp = row.key_value_pairs
            new.write(atoms, data=data, key_value_pairs=kvp)
        os.remove(_db_path)
        os.rename(new_db, _db_path)

    def run_vaspjet(self):
        self.update_db(self.db_path)
        # 与cpu服务器建立远程连接
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(hostname=self.cpu_config["CPU IP"],
                    port=self.cpu_config["CPU Port"],
                    username=self.cpu_config["CPU Username"],
                    password=self.cpu_config["CPU Password"])
        sftp = ssh.open_sftp()
        # 确保远程目录存在，如果不存在则创建
        try:
            sftp.mkdir(os.path.dirname(self.cpu_workdir))
        except IOError:
            pass
        try:
            sftp.mkdir(self.cpu_workdir)
        except IOError:
            pass
        # 上传模型文件，数据文件，配置文件
        script_dir = os.path.dirname(os.path.abspath(__file__))
        # local_pt = os.path.join(script_dir, "mlp_direct_h512_all.pt")
        local_py = os.path.join(script_dir, "pure_vasp.py")
        local_db = self.db_path
        local_yml = self.vaspjet_yml
        db_name = os.path.basename(self.db_path)
        with open(local_yml, "r") as f:
            content = f.readlines()
        for i, line in enumerate(content):
            if "db_path" in line and "#" not in line:
                content[i] = f"db_path: './{db_name}'\n"
                break
        with open(local_yml, "w") as f:
            f.writelines(content)
        # sftp.put(local_pt, os.path.join(self.cpu_workdir, "mlp_direct_h512_all.pt"))
        sftp.put(local_py, os.path.join(self.cpu_workdir, "pure_vasp.py"))
        sftp.put(local_db, os.path.join(self.cpu_workdir, os.path.basename(self.db_path)))
        sftp.put(local_yml, os.path.join(self.cpu_workdir, os.path.basename(self.vaspjet_yml)))
        command = """
        source ~/.zshrc && conda activate vaspjet && cd {0} && nohup python pure_vasp.py run -yml *.yml -r 1 1>./out.log 2>./err.log & echo $!
        """.format(self.cpu_workdir)
        ssh.exec_command(command)
        sftp.close()
        ssh.close()
        return


def vaspjet_monitor(cpu_config,
                    cpu_workdir,
                    download_results=False,
                    local_path=None,
                    traj_process_mode=None,  # "filter"/"last_image"/None
                    max_force_threshold=50):
    """
    监控vaspjet任务的运行情况
    :return: 任务状态: "RUNNING", "DONE", "KILLED"
    """
    state = "RUNNING"
    # 与cpu服务器建立远程连接
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(hostname=cpu_config["CPU IP"],
                port=cpu_config["CPU Port"],
                username=cpu_config["CPU Username"],
                password=cpu_config["CPU Password"])
    sftp = ssh.open_sftp()
    command = f"ls {os.path.join(cpu_workdir, 'results')}"
    stdin, stdout, stderr = ssh.exec_command(command)
    files = stdout.read().decode().split("\n")
    if "KILLED" in files:
        state = "KILLED"
    else:
        if "final.db" in files:
            state = "DONE"

    # 下载结果文件
    IS_DOWNLOADED = False
    if download_results:
        if traj_process_mode:
            current_file_path = os.path.abspath(__file__)
            process_traj_path = os.path.join(os.path.dirname(current_file_path), "process_traj.py")
            sftp.put(process_traj_path,
                     os.path.join(cpu_workdir, "results/process_traj.py"))
            command = (f"source ~/.zshrc && conda activate vaspjet && cd {cpu_workdir}/results && "
                       f"python process_traj.py {max_force_threshold} {traj_process_mode}")
            ssh.exec_command(command)
            remote_dir = os.path.join(cpu_workdir, "results/traj.db")
            # 确保轨迹处理完毕
            traj_done_tag = os.path.join(cpu_workdir, "results/TRAJ_DONE")
            TRAJ_DONE = False
            while not TRAJ_DONE:
                try:
                    sftp.get(traj_done_tag, os.path.join(os.path.dirname(local_path), "TRAJ_DONE"))
                    TRAJ_DONE = True
                    os.remove(os.path.join(os.path.dirname(local_path), "TRAJ_DONE"))
                    break
                except:
                    time.sleep(60)
        else:
            if state == "RUNNING" or state == "KILLED":
                remote_dir = os.path.join(cpu_workdir, "results/intime_results.db")
            else:
                remote_dir = os.path.join(cpu_workdir, "results/final.db")
        for i in range(10):
            print(f"Downloading {remote_dir} to {local_path}")
            try:
                sftp.get(remote_dir, local_path)
                IS_DOWNLOADED = True
                break
            except:
                time.sleep(60)
    sftp.close()
    ssh.close()
    return state, IS_DOWNLOADED


def kill_vaspjet(cpu_config, cpu_workdir):
    # 与cpu服务器建立远程连接
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(hostname=cpu_config["CPU IP"],
                port=cpu_config["CPU Port"],
                username=cpu_config["CPU Username"],
                password=cpu_config["CPU Password"])
    # 将kill_vaspjet.py上传到cpu服务器
    sftp = ssh.open_sftp()
    current_file_path = os.path.abspath(__file__)
    kill_vaspjet_path = os.path.join(os.path.dirname(current_file_path), "kill_vaspjet.py")
    sftp.put(kill_vaspjet_path,
             os.path.join(cpu_workdir, "kill_vaspjet.py"))
    # 杀死进程并创建KILLED文件作为标记
    command = (f"source ~/.zshrc && conda activate vaspjet && cd {cpu_workdir} && python kill_vaspjet.py && "
               f"touch {cpu_workdir}/results/KILLED")
    ssh.exec_command(command)
    ssh.close()
    return


if __name__ == '__main__':
    # db_path = "/home/cchen/Train_NN/slab/seeds_opt.db"
    cpu_config = {
        "CPU IP": "202.120.101.188",
        "CPU Username": "materdesign",
        "CPU Port": 22,
        "CPU Password": "md188"
    }
    # test = VaspjetRun(db_path,
    #                   cpu_config,
    #                   "/home/materdesign/cc/slab/iter1/sp",
    #                   "/home/cchen/Train_NN/slab/template/vaspjet/pure_vasp.yml"
    #                   )
    # test.run_vaspjet()
    # time.sleep(300)
    vaspjet_monitor(cpu_config=cpu_config,
                    cpu_workdir="/home/materdesign/cc/test-s1/iter2/opt",
                    download_results=True,
                    local_path="/home/cchen/test/0317/traj.db",
                    traj_process_mode="filter",
                    max_force_threshold=0.5)
