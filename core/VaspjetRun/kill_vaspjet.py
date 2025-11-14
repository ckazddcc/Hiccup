import os


def get_jobids(directory):
    jobids = []
    part_files = []
    for root, dirs, files in os.walk(directory):  # 遍历指定目录及其子目录
        for dir in dirs:
            if dir.startswith('part'):  # 判断文件是否以'part'开头
                part_files.append(os.path.join(root, dir))  # 获取文件的完整路径

    for file in part_files:
        slum_job_info_path = os.path.join(str(file), 'slurm_job_info.txt')
        if os.path.exists(slum_job_info_path):
            with open(slum_job_info_path, 'r') as f:
                for index, line in enumerate(f.readlines()):
                    if line.startswith('Slurm Job ID'):
                        jobid = line.split(" ")[3][:-1]
                        jobids.append(jobid)
    return jobids


def kill_job(directory):
    jobids = get_jobids(directory)
    for jobid in jobids:
        os.system(f'scancel {jobid}')
    return


if __name__ == "__main__":
    kill_job('./')
