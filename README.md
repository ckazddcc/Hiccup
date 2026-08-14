# Hiccup

Version 1.0

Hiccup is an automated platform for training high-performance neural network potentials. The name “Hiccup” is inspired by the protagonist of the movie How to Train Your Dragon. It reflects our vision of transforming the expert-dependent and challenging task of training neural network potentials—analogous to “training a dragon”—into an automated and high-performance workflow.

## Introduction

### 1. Workflow

![workflow](./workflow.jpg)

## Quickstart

Hiccup runs across two machines: a **GPU server** (NN training, GA search) and a **CPU server** (VASP DFT calculations via VaspJet). Both must be configured before use.

### Step 1: Install Hiccup (GPU Server)

```bash
conda create -n hiccup python=3.11 -y
conda activate hiccup
git clone https://github.com/ckazddcc/Hiccup.git
cd Hiccup
pip install -r requirements.txt
pip install .
```

To verify the installation and detect dependency or environment incompatibilities, run the basic test suite from the repository root:

```bash
pytest test/test_basic.py
```

The tests check core Python dependencies, CUDA/GPU availability, DeepMD-kit and LAMMPS integration, configuration files, templates, composition generation, and CLI argument parsing. Because the TensorFlow, PyTorch, DeepMD-kit, and NumPy stack can be sensitive to version combinations, rerun this suite after changing any pinned dependency.

Hiccup uses USPEX (v9.4.4) for genetic algorithm structure search, which requires a **Python 2** environment. Install USPEX following its official documentation and ensure it is runnable. Record the Python 2 environment path (e.g., `/path/to/uspex-env/bin`) — you will need it in `config.yml` under `USPEX Env`.

### Step 2: Configure SSH Key (GPU → CPU)

Hiccup connects to the CPU server via SSH key authentication (no password). On the GPU server:

```bash
ssh-keygen -t ed25519
ssh-copy-id -i ~/.ssh/id_ed25519.pub your_username@cpu_server_ip
chmod 600 ~/.ssh/id_ed25519
export HICCUP_CPU_SSH_KEY=~/.ssh/id_ed25519
```

> The environment variable name is read from `CPU SSH Key Env` in `config.yml` (default: `HICCUP_CPU_SSH_KEY`). The private key must have `600` permissions.

### Step 3: Set Up VaspJet (CPU Server)

Hiccup uploads `.db` files and `pure_vasp.py` to the CPU server, then remotely invokes `python pure_vasp.py run -yml config.yml` via the `vaspjet` conda environment. Set up this environment:

```bash
conda create -n vaspjet python=3.11 -y
pip install -r vaspjet_requirements.txt
```

**Configure VASP**: VaspJet submits each structure as a SLURM job, running VASP via `mpirun -np {cpus} {vasp_version}`. The VASP executable, MPI, and POTCAR library must be available on the **compute nodes**. Edit the `slurm_setup` commands in `template/vaspjet/config_*.yml` to load them:

```yaml
slurm:
  slurm_partition: '<YOUR_PARTITION>'     # [REQUIRED] SLURM partition
  cpus_per_task: 24                        # [REQUIRED] CPU cores per job
  vasp_version: 'vasp_gam'
  slurm_setup:
    - 'source /path/to/vasp/oneapi/setvars.sh --force'
    - 'export PATH=/path/to/vasp/bin:$PATH'
    - 'export VASP_PP_PATH=/path/to/POTCAR/library'   # [REQUIRED]
```

### Step 4: Configure and Run

Copy the example config and update the required fields:

```bash
cp example/cluster/config.yml ./config.yml
```

Key fields to set in `config.yml`:

```yaml
BASE:
  Templates: /path/to/Hiccup/template     # [REQUIRED] Template directory
  Workdir: /path/to/workdir               # [REQUIRED] Working directory
  Gpu: [0,1,2,3]                          # [REQUIRED] GPU IDs (4 recommended)
CPU:
  CPU IP: '<YOUR_CPU_IP>'                 # [REQUIRED]
  CPU Username: '<YOUR_USERNAME>'         # [REQUIRED]
  CPU SSH Key Env: HICCUP_CPU_SSH_KEY
  CPU Working Directory: /path/on/cpu     # [REQUIRED]
```

```bash
conda activate hiccup
hiccup run -yml config.yml
```

## Usage

### 1. `template` Directory

- **`trainer/`**: DeePMD training input files. `deepmd_input.json` for iterative training, `deepmd_input_accurate.json` for final high-accuracy training.
- **`uspex/`**: VASP input files (`INCAR_1`, `KPOINTS`, `POTCAR`) required by USPEX (parameters do not affect Hiccup's workflow). `dp_opt.py` / `mace_opt.py` are NN/MACE calculator scripts; `run_dp.sh` / `run_mace.sh` are the corresponding runner scripts called by USPEX. `TEMP_INPUT_0/2/3.txt` are USPEX templates for cluster/surface/bulk searches.
- **`vaspjet/`**: VaspJet YAML configs for single-point (`config_sp.yml`), optimization (`config_opt.yml`), and molecular dynamics (`config_md.yml`) calculations.

### 2. `config.yml` Configuration File

The complete configuration file contains the following sections. Fields marked `[REQUIRED]` must be set by the user.

```yaml
BASE:
  Compositions: [[17, 40],[18,40]]         # Target compositions
  Elements: [O, Cu]                         # Element list
  Gpu: [0,1,2,3]                            # [REQUIRED] Available GPU IDs (4 recommended)
  Iterations: 3                             # Number of iterations
  Stall Iterations: 3                       # Stall generations before early stop
  Accuracy Threshold: 0.95                  # Convergence threshold
  Templates: /path/to/Hiccup/template       # [REQUIRED] Template directory
  Workdir: /path/to/workdir                 # [REQUIRED] Working directory

CPU:
  CPU IP: '<YOUR_CPU_IP>'                   # [REQUIRED]
  CPU Port: '<YOUR_CPU_PORT>'                              # [REQUIRED] SSH port
  CPU Username: '<YOUR_USERNAME>'           # [REQUIRED]
  CPU SSH Key Env: HICCUP_CPU_SSH_KEY       # Env var for SSH key path
  CPU Working Directory: /path/on/cpu       # [REQUIRED]

SAMPLER:
  GA:
    RANDOMSEEDS:
      Activate: True                        # Enable random seed generator
      Dimension: 3                          # 0: cluster, 3: bulk
      Random Seeds Num: 100                 # Seeds per composition
    USPEX:
      Dimension: 2                          # 1: cluster, 2: surface, 3: bulk
      Generation Num: 3                     # GA generations
      Init Pop Size: 10                     # First generation population
      Pop Size: 10                          # Subsequent generation population
      Calculator: DP                        # DP / MACE
      USPEX Env: /path/to/uspex-env/bin     # USPEX Python 2 env
  NNMD:
    NN Force Accuracy: 0.15                 # Threshold for launching NNMD
    MD Timestep: 1                          # fs
    MD Steps: 10000
    MD Temperature K: 500

TRAINER:
  Deepmd:
    Data Path: /path/to/init_database.db    # Initial dataset (ASE .db)
    Initial Model: /path/to/init_model.pb   # Initial model (optional)
    Train Ratio: 0.9

POSTPROCESSING:
  Force Deviation Lower: 0.05               # Auto: best model validation error
  Force Deviation Upper: 0.2                # Auto: above + 1.5
  Max Filter Ratio: 0.8
  Max Filter Num: 100                       # Recommended: compositions x 10
  Energy Filter: 0.1
  Force Filter: 2
```

**Parameter Details:**

- **BASE**: `Compositions` — list of `[atom_count, total_atoms]` pairs. `Elements` — element list, order must match compositions. `Gpu` — GPU IDs, 4 recommended. `Iterations` — total iterations excluding the final one. `Stall Iterations` — early stop after N generations without improvement. `Accuracy Threshold` — ratio of accurate structures for convergence.
- **CPU**: SSH connection to the CPU server. `CPU Working Directory` should be an empty directory.
- **SAMPLER > RANDOMSEEDS**: `Dimension` — 0 for cluster, 3 for bulk. `Random Seeds Num` — structures per composition (not total). If `Activate: False`, provide `Random Seeds Path` to a pre-generated ASE `.db` file with unique `uid` in `key_value_pairs`.
- **SAMPLER > USPEX**: `Dimension` — 1/2/3 for cluster/surface/bulk. `Substrate` — required for surface searches. `Calculator` — `DP` or `MACE`.
- **SAMPLER > NNMD**: `NN Force Accuracy` — threshold based on current best model's validation error. Defaults: Timestep 1 fs, Steps 10000, Dump 100, Temperature 500 K.
- **TRAINER**: `Data Path` — ASE `.db` file with energy/force in `row.data`. `Train Ratio` — train/test split ratio.
- **POSTPROCESSING**: Structures are classified as `accurate`/`candidate`/`failed` by force deviation. `Force Deviation Lower/Upper` — bounds for classification (`Auto` uses best model's validation error). `Max Filter Ratio/Num` — limits for DFT labeling batch selection. `Energy/Force Filter` — thresholds for cleaning bad data points.

### 3. Command-line Usage

```bash
# Run a Hiccup workflow
hiccup run -yml config.yml

# Evaluate model performance
hiccup eva -db database.db -m model.pb -g gpu_id -e 0.1 -f 2 -n model_name

# Generate target compositions with FPS
hiccup compos -yml fps_config.yml
```
