# from tools.md_sample import run_many_atoms_parallel
from tools.md_sample import run_md_parallel
from ase.db import connect

class NN_md:
    """Run MD sampling using a DP model in parallel.

    Reads structures from a database, launches Langevin MD for each structure
    using run_md_parallel, and collects trajectory results.
    """

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
        """Run parallel MD sampling on all structures in the database.

        Reads structures from to_md_db, launches MD with parameters from
        nn_md_config, and prints summary results.
        """
        # Run batch MD sampling
        to_md_atoms = [row.toatoms() for row in connect(self.to_md_db).select()]

        md_results = run_md_parallel(
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
        to_md_db='<YOUR_TO_MD_DB_PATH>',
        nn_model='<YOUR_NN_MODEL_PATH>',
        gpus=[1,2,3,4,5,6],
        md_workdir='<YOUR_MD_WORKDIR>',
        nn_md_config={
            'MD Steps': 10000,
            'MD Timestep': 0.5,
            'MD Dump Interval': 50,
            'MD Temperature K': 500,
        }
    )
    nn_md.run()
    end = time.time()
    print(end - start)
