# Hiccup

Version 2.0

Genetic Algorithm-Driven Neural Network Potential Trainer.

## Introduction

### 1. Workflow

![workflow](./workflow.jpg)

## Installation

```bash
git clone https://gitee.com/ccccissy/Hiccup.git
pip install .
```

## Usage

### 1. Input Files

#### 1.1 `template` Directory

##### (1) Directory Structure

```text
-trainer
  --deepmd_input.json
  --deepmd_input_accurate.json
-uspex
  --INCAR_1
  --KPOINTS
  --POTCAR_A
  --POTCAR_B
  --run_dp.sh
  --run_mace.sh
  --dp_opt.py
  --mace_opt.py
  --TEMP_INPUT_0.txt
  --TEMP_INPUT_2.txt
  --TEMP_INPUT_3.txt
-vaspjet
  --pure_vasp_sp.yml
  --pure_vasp_opt.yml
  --pure_vasp_md.yml
```

##### (2) Description

**`trainer` directory:**

This directory contains two DeePMD input files.

- `deepmd_input.json`: input file for DeePMD training during the iterative workflow.
- `deepmd_input_accurate.json`: input file for the final high-accuracy DeePMD training.

**`uspex` directory:**

This directory contains the basic input files required for VASP calculations, such as `INCAR_1`, `KPOINTS`, and `POTCAR`. Although USPEX is not driven by VASP in this workflow, these basic input files are still required. Otherwise, USPEX may report an error. The parameters in these files do not affect the calculations in this workflow.

- `dp_opt.py`: uses the trained neural network potential as the calculator and outputs energy and force information. It can be modified as needed.
- `mace_opt.py`: uses the general-purpose MACE potential as the calculator and outputs energy and force information. It can be modified as needed.
- `run_dp.sh`: script used by USPEX to call the neural network potential as the calculator. A Python 3 environment is required to execute `dp_opt.py`.
- `run_mace.sh`: script used by USPEX to call MACE as the calculator. A Python 3 environment is required to execute `mace_opt.py`.
- `TEMP_INPUT_0.txt`: USPEX input template for `dimension = 0`, corresponding to cluster structure search.
- `TEMP_INPUT_2.txt`: USPEX input template for `dimension = 2`, corresponding to surface structure search.
- `TEMP_INPUT_3.txt`: USPEX input template for `dimension = 3`, corresponding to bulk structure search.

**`vaspjet` directory:**

- `pure_vasp_sp.yml`: VapsJet configuration file for single-point energy calculations.
- `pure_vasp_opt.yml`: VapsJet configuration file for structure optimization calculations.
- `pure_vasp_md.yml`: VapsJet configuration file for molecular dynamics calculations.

#### 1.2 `config.yml` Configuration File

##### (1) File Structure

```yaml
# Basic configuration
BASE:
  Compositions: [[17, 40],[18,40],[19,40],[20,40],[21,40]] # Target compositions
  Elements: [O, Cu] # Element list
  Gpu: [0,1,2,3,4,5,6,7] # Available GPU IDs
  Iterations: 3 # Number of iterations
  Stall Iterations: 3 # Number of stall generations
  Accuracy Threshold: 0.95 # Convergence threshold; ratio of accurate structures
  Templates: /home/cchen/Train_NN/template # Template directory path
  Workdir: /home/cchen/Train_NN/example/slab/hiccup # Working directory path

# CPU configuration
CPU:
  CPU IP: 202.120.101.188
  CPU Password: xxx
  CPU Port: 22
  CPU Username: materdesign
  CPU Working Directory: /home/materdesign/cc/test1

# Sampler configuration
SAMPLER:
  GA:
    RANDOMSEEDS:
      Activate: True # Whether to enable the random seed generator. If disabled, a random seed file must be provided.
      Dimension: 3 # 0: cluster, 3: bulk
      Random Seeds Num: 100 # Number of random seeds
      #Random Seeds Path: /home/cchen/CuY/hiccup2/random_seeds.db # Random seed file path
      #Init Seeds Path: /home/cchen/CuY/hiccup2/workdir/pes/ga/ga6/good.db # Initial seed file path
    USPEX:
      Dimension: 2 # 1: cluster, 2: surface, 3: bulk
      Generation Num: 3 # Number of GA generations
      Init Pop Size: 10 # Population size of the first generation
      Pop Size: 10 # Population size of each generation
      Calculator: DP # Calculator: DP/MACE
      Constraint z: 2.0
      Substrate: /home/cchen/CuY/hiccup2/template/POSCAR_SUBSTRATE # Substrate structure
      USPEX Env: /home/cchen/.conda/envs/uspex/bin # Environment path
  NNMD:
    NN Force Accuracy: 0.15 # Force-accuracy threshold for launching NNMD
    MD Timestep: 1 # fs
    MD Steps: 10000
    MD Dump Interval: 100
    MD Temperature K: 500

# Trainer configuration
TRAINER:
  Deepmd:
    Data Path: /home/cchen/Train_NN/example/slab/init_database.db # Initial dataset path
    Initial Model: /home/cchen/Train_NN/slab/init_model.pb # Initial model path; default is None
    Train Ratio: 0.9 # Training set ratio

POSTPROCESSING:
  # NNP evaluator: accurate, candidate, failed
  Force Deviation Lower: 0.05 # Default: Auto
  Force Deviation Upper: 0.2 # Default: Auto

  # Energy and force filter: SP -> DFT optimization
  Max Filter Ratio: 0.8
  Max Filter Num: 100

  # Use the best model to clean bad data points
  Energy Filter: 0.1
  Force Filter: 2

TRAINER:
  Deepmd:
    Data Path: /home/cchen/Train_NN/slab/init_database.db # Initial dataset path
    Initial Model: /home/cchen/CuY/hiccup2/workdir/dp/nn6/002/frozen_model.pb # Initial model path; default is None
    Train Ratio: 0.9 # Training set ratio
```

##### (2) Detailed Description

##### Basic Configuration

- `Compositions`: list of target compositions.
- `Elements`: list of elements. The element order must be consistent with the order in the composition list.
- `Gpu`: list of available GPU IDs. At least four GPUs are required.
- `Iterations`: integer. Total number of iterations, excluding the final iteration.
- `Stall Iterations`: number of stall generations.
- `Accuracy Threshold`: convergence threshold, defined as the ratio of accurate structures.
- `Templates`: string. Path to the template directory.
- `Workdir`: string. Working directory.

##### CPU Configuration

- `CPU IP`: string. IP address of the CPU server.
- `CPU Password`: string. Password of the CPU server.
- `CPU Port`: string. Port number.
- `CPU Username`: string. Username.
- `CPU Working Directory`: string. Working directory on the CPU server. It is recommended to use an empty directory.

##### Sampler Configuration

**Random seed generator configuration:**

- `Activate`: boolean. Whether to enable the random seed generator.
- `Dimension`: integer. The random seed generator uses PyXtal to generate random structures. `0` corresponds to cluster systems, and `3` corresponds to bulk systems.
- `Random Seeds Num`: integer. Number of random structures generated for each composition. This is not the total number of random seed structures. The default value is `100`.
- `Random Seeds Path`: string. Path to the random seed structure file. If the random seed generator is disabled, this file must be provided. The file should be an ASE database file. Each `atoms` object must contain a unique `uid` in `key_value_pairs`, for example `key_value_pairs={'uid': 'unique_uid'}`.
- `Init Seeds Path`: string. Path to the initial seed structure file.

**USPEX configuration:**

- `Dimension`: integer. Search type in USPEX. `1`: cluster, `2`: surface, `3`: bulk.
- `Generation Num`: integer. Number of generations in the genetic algorithm.
- `Init Pop Size`: integer. Population size of the first generation. By default, it is the same as `Pop Size`.
- `Pop Size`: integer. Population size of each generation.
- `Calculator`: string. Calculator selection. Supported values are `DP` and `MACE`. The default is `MACE`.
- `USPEX Env`: string. Path to the USPEX Python 2 environment.
- `Substrate`: string. Path to the substrate file. This is required for surface structure searches.

**NNMD configuration:**

- `NN Force Accuracy`: force-accuracy threshold for starting NNMD. It is determined based on the force error of the current best NNP model on the validation set. The default value is `0.15`.
- `MD Timestep`: float. MD time step. The default value is `1 fs`.
- `MD Steps`: integer. Number of MD steps. The default value is `10000`.
- `MD Dump Interval`: integer. MD output interval. The default value is `100`.
- `MD Temperature K`: float. MD temperature. The default value is `500 K`.

##### Post-processing Configuration

Four models are used to evaluate structures. According to the per-atom force evaluation results, structures generated by GA search are divided into three categories: `accurate`, `candidate`, and `failed`.

- `Force Deviation Lower`: float. Lower bound of the maximum force deviation. In `Auto` mode, it is set to the force error of the current best NNP model on the validation set.
- `Force Deviation Upper`: float. Upper bound of the maximum force deviation. In `Auto` mode, it is set to the force error of the current best NNP model on the validation set plus `1.5`.

**Structure similarity evaluation and filtering parameters:**

These parameters are used to evaluate whether structures are representative. All structures awaiting DFT labeling are divided into high-priority and low-priority structures, and calculations are then performed in batches.

- `Max Filter Ratio`: float. Upper limit of the ratio of selected structures to all structures. The default value is `0.8`.
- `Max Filter Num`: integer. Upper limit of the total number of selected structures. The recommended value is `number of target compositions × 10`.

**Data cleaning parameters:**

These parameters are used to clean bad data points with the best model.

- `Energy Filter`: float. Upper limit of the energy deviation between predicted values and DFT-calculated values. This value can be increased appropriately when the model quality is poor. The default value is `0.1`.
- `Force Filter`: float. Upper limit of the force deviation between predicted values and DFT-calculated values. This value can be increased appropriately when the model quality is poor. The default value is `2`.

**LCS processing parameters:**

- `LCS Process`: boolean. Whether to perform LCS processing. The default value is `False`.
- `Type`: string. Type of structures to be processed, such as `slab` or `cluster`.
- `LCS Layers Num`: integer. Number of substrate layers divided by LCS for slab systems.
- `LCS Radius`: float. Cutting radius used in LCS processing for cluster systems.

##### Trainer Configuration

Currently, only DeePMD is supported.

- `Data Path`: string. Path to the initial dataset. The initial dataset should be provided as an ASE database file. The `row.data` field must contain energy and force information.
- `Initial Model`: string. Path to the initial model. The default value is `None`.
- `Train Ratio`: float. Ratio of the training set to the test set. The default value is `0.8`.

### 2. Command-line Usage

##### (1) Run a Hiccup Task

```bash
hiccup run -yml config.yml
# -yml: path to the *.yml configuration file
```

##### (2) Evaluate Model Performance

```bash
hiccup eva -db database.db -m model.pb -g gpu_id -e 0.1 -f 2 -n model_name
# -db: path to the *.db database file
# -m: path to the model file to be evaluated
# -g: GPU ID
# -e: upper limit of the energy error; default value is 0.1
# -f: upper limit of the force error; default value is 2
# -n: model name; default name is "Model"
```

##### (3) Automatically Generate Target Compositions with FPS

```bash
hiccup compos -yml fps_config.yml
# -yml: path to the fps_config.yml configuration file
```
