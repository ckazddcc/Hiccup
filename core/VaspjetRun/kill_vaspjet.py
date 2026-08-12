import os


def get_jobids(directory):
    """Collect Slurm job IDs from vaspdir subdirectories."""
    jobids = []
    part_files = []
    for root, dirs, files in os.walk(directory):  # Walk directory and subdirectories
        for dir in dirs:
            if dir.startswith('vaspdir'):  # Check if directory starts with 'vaspdir'
                part_files.append(os.path.join(root, dir))  # Get full path

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
    """Cancel all Slurm VASP jobs found under the given directory."""
    jobids = get_jobids(directory)
    for jobid in jobids:
        os.system(f'scancel {jobid}')
    return


if __name__ == "__main__":
    kill_job('./')
