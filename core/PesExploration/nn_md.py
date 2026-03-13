from tools.md_sample import run_many_atoms_parallel
from ase.db import connect

class NN_md:
    def __init__(self,
                 to_md_db,
                 nn_model,
                 gpus,
                 md_workdir,
                 nn_md_config
                 ):
        self.to_md_db = to_md_db
        self.nn_model = nn_model
        self.gpus = gpus
        self.md_workdir = md_workdir
        self.nn_md_config = nn_md_config

    def run(self):
        # 进行批量 MD 采样
        to_md_atoms = [row.toatoms() for row in connect(self.to_md_db).select()]

        md_results = run_many_atoms_parallel(
            atoms_list=to_md_atoms,
            dp_model_path=self.nn_model,
            base_workdir=self.md_workdir,
            nproc=len(self.gpus),  # one process per atoms seed, up to you
            nsteps=self.nn_md_config.get('MD Steps', 10000),
            timestep_fs=self.nn_md_config.get('MD Timestep', 0.5),
            dump_interval=self.nn_md_config.get('MD Dump Interval', 50),
            temperature_K=self.nn_md_config.get('MD Temperature K', 500),
            cpu_only_inference=False,
            gpu_ids=self.gpus
        )
        print(type(md_results))
        print(md_results)

if __name__ == '__main__':
    import time
    start = time.time()
    nn_md = NN_md(
        to_md_db='/home/cchen/slab/hiccup/pes/ga/ga2/MD/to_md.db',
        nn_model='/home/cchen/CuY/hiccup/hiccup3/dp/nn4/002/frozen_model.pb',
        gpus=[4,5,6,7],
        md_workdir='/home/cchen/slab/hiccup/pes/ga/ga2/MD',
        nn_md_config={
            'MD Steps': 5000,
            'MD Timestep': 0.5,
            'MD Dump Interval': 50,
            'MD Temperature K': 500,
        }
    )
    nn_md.run()
    end = time.time()
    print(end - start)
