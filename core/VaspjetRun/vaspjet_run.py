import paramiko
import os
import time
from ase.db import connect


class VaspjetRun:
    """Submit VASP calculations to a remote CPU server via SSH.

    Uploads database and config files, then launches VaspJet remotely.
    """

    def __init__(self,
                 db_path,
                 cpu_config,
                 cpu_workdir,
                 vaspjet_yml):
        self.db_path = db_path
        self.cpu_config = cpu_config
        self.vaspjet_yml = vaspjet_yml
        self.cpu_workdir = cpu_workdir

    def run_vaspjet(self):
        """Upload files and launch VaspJet on the remote CPU server."""
        # Establish SSH connection to the CPU server
        key_env = self.cpu_config.get("CPU SSH Key Env", "HICCUP_CPU_SSH_KEY")
        ssh_key_path = os.environ.get(key_env)
        if not ssh_key_path:
            raise RuntimeError(f"Environment variable {key_env} is not set.")

        ssh = paramiko.SSHClient()
        ssh.load_host_keys(os.path.expanduser("~/.ssh/known_hosts"))
        ssh.connect(hostname=self.cpu_config["CPU IP"],
                    port=self.cpu_config["CPU Port"],
                    username=self.cpu_config["CPU Username"],
                    key_filename=os.path.expanduser(ssh_key_path),
                    look_for_keys=False,
                    allow_agent=False)
        sftp = ssh.open_sftp()
        # Ensure remote directories exist, create if missing
        try:
            sftp.mkdir(os.path.dirname(self.cpu_workdir))
        except IOError:
            pass
        try:
            sftp.mkdir(self.cpu_workdir)
        except IOError:
            pass

        # Upload model, data, and config files
        script_dir = os.path.dirname(os.path.abspath(__file__))
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
        sftp.put(local_db, os.path.join(self.cpu_workdir, os.path.basename(self.db_path)))
        sftp.put(local_yml, os.path.join(self.cpu_workdir, os.path.basename(self.vaspjet_yml)))
        command = """
        source ~/.bashrc && conda activate vaspjet && cd {0} && nohup vaspjet run -yml *.yml 1>out.log 2>err.log & & echo $!
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
    Monitor VaspJet job status on the remote CPU server.

    Args:
        cpu_config: dict with CPU connection parameters.
        cpu_workdir: remote working directory path.
        download_results: if True, download result files when the job is done.
        local_path: local path to save downloaded results.
        traj_process_mode: trajectory processing mode - "filter",
            "last_image", or None.
        max_force_threshold: max force cutoff for trajectory filtering.

    Returns:
        Tuple of (state, is_downloaded) where state is "RUNNING",
        "DONE", or "KILLED".
    """
    state = "RUNNING"
    # Establish SSH connection to the CPU server
    key_env = cpu_config.get("CPU SSH Key Env", "HICCUP_CPU_SSH_KEY")
    ssh_key_path = os.environ.get(key_env)
    if not ssh_key_path:
        raise RuntimeError(f"Environment variable {key_env} is not set.")

    ssh = paramiko.SSHClient()
    ssh.load_host_keys(os.path.expanduser("~/.ssh/known_hosts"))
    ssh.connect(hostname=cpu_config["CPU IP"],
                port=cpu_config["CPU Port"],
                username=cpu_config["CPU Username"],
                key_filename=os.path.expanduser(ssh_key_path),
                look_for_keys=False,
                allow_agent=False)
    sftp = ssh.open_sftp()
    command = f"ls {os.path.join(cpu_workdir, 'workdir')}"
    stdin, stdout, stderr = ssh.exec_command(command)
    files = stdout.read().decode().split("\n")
    if "KILLED" in files:
        state = "KILLED"
    else:
        if "final.db" in files:
            state = "DONE"

    # Download result files
    IS_DOWNLOADED = False
    if download_results:
        if traj_process_mode:
            current_file_path = os.path.abspath(__file__)
            process_traj_path = os.path.join(os.path.dirname(current_file_path), "process_traj.py")
            sftp.put(process_traj_path,
                     os.path.join(cpu_workdir, "workdir/process_traj.py"))
            # command = (f"source ~/.zshrc && conda activate vaspjet && cd {cpu_workdir}/workdir && "
            #            f"python process_traj.py {max_force_threshold} {traj_process_mode}")
            command = (f"conda activate vaspjet && cd {cpu_workdir}/workdir && "
                       f"python process_traj.py {max_force_threshold} {traj_process_mode}")
            ssh.exec_command(command)
            remote_dir = os.path.join(cpu_workdir, "workdir/traj.db")
            # Wait for trajectory processing to complete
            traj_done_tag = os.path.join(cpu_workdir, "workdir/TRAJ_DONE")
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
            if state == "KILLED" or state == "DONE":
                remote_dir = os.path.join(cpu_workdir, "workdir/final.db")
            else:
                remote_dir = None
                print("DFT is not done yet, wait for 60s...")

        for i in range(10):
            try:
                if remote_dir is not None:
                    sftp.get(remote_dir, local_path)
                    IS_DOWNLOADED = True
                    break
                else:
                    time.sleep(10)
            except:
                time.sleep(60)
    sftp.close()
    ssh.close()
    return state, IS_DOWNLOADED


def kill_vaspjet(cpu_config, cpu_workdir):
    """Kill running VaspJet jobs on the remote CPU server.

    Uploads kill_vaspjet.py, executes it remotely, and creates a KILLED
    marker file.
    """
    # Establish SSH connection to the CPU server
    key_env = cpu_config.get("CPU SSH Key Env", "HICCUP_CPU_SSH_KEY")
    ssh_key_path = os.environ.get(key_env)
    if not ssh_key_path:
        raise RuntimeError(f"Environment variable {key_env} is not set.")

    ssh = paramiko.SSHClient()
    ssh.load_host_keys(os.path.expanduser("~/.ssh/known_hosts"))
    ssh.connect(hostname=cpu_config["CPU IP"],
                port=cpu_config["CPU Port"],
                username=cpu_config["CPU Username"],
                key_filename=os.path.expanduser(ssh_key_path),
                look_for_keys=False,
                allow_agent=False)

    # Upload kill_vaspjet.py to the CPU server
    sftp = ssh.open_sftp()
    current_file_path = os.path.abspath(__file__)
    kill_vaspjet_path = os.path.join(os.path.dirname(current_file_path), "kill_vaspjet.py")
    sftp.put(kill_vaspjet_path,
             os.path.join(cpu_workdir, "kill_vaspjet.py"))
    # Kill processes and create KILLED marker file
    command = (f"conda activate vaspjet && cd {cpu_workdir} && python kill_vaspjet.py && "
               f"touch {cpu_workdir}/workdir/KILLED")
    ssh.exec_command(command)
    ssh.close()
    return


if __name__ == '__main__':
    templates = "<YOUR_TEMPLATES_PATH>"
    cpu_config = {
        "CPU IP": "<YOUR_CPU_IP>",
        "CPU Username": "<YOUR_CPU_USERNAME>",
        "CPU Port": "<YOUR_CPU_PORT>",
        "CPU SSH Key Env": "HICCUP_CPU_SSH_KEY"
    }

    # dft_sp = VaspjetRun(db_path="<YOUR_DB_PATH>",
    #                     cpu_config=cpu_config,
    #                     cpu_workdir=os.path.join("<YOUR_CPU_WORKDIR>", "0"),
    #                     vaspjet_yml=os.path.join(templates, "vaspjet/config_opt.yml"))
    # dft_sp.run_vaspjet()

    state, IS_DOWNLOADED = vaspjet_monitor(cpu_config,
                    cpu_workdir="<YOUR_CPU_WORKDIR>",
                    download_results=False)
    print(state, IS_DOWNLOADED)

