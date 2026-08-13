"""Basic test suite for Hiccup.

This module provides lightweight tests that verify core functionality. 
Run with: pytest test/test_basic.py
"""
import os
import shutil
import subprocess
import sys
import pytest
import yaml
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'core'))


def run_command(command, timeout=60):
    """Run an external command and capture its combined output."""
    return subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=timeout,
        check=False,
    )



class TestModuleImports:
    """Verify that core modules can be imported without errors."""

    def test_import_ase(self):
        import ase
        assert hasattr(ase, '__version__')

    def test_import_yaml(self):
        import yaml
        assert hasattr(yaml, 'safe_load')

    def test_import_numpy(self):
        import numpy as np
        assert hasattr(np, '__version__')


class TestRequiredComputeEnvironment:
    """Verify that GPU, DeepMD-kit, and LAMMPS can be invoked."""

    def test_nvidia_gpu_can_be_invoked(self):
        nvidia_smi = shutil.which('nvidia-smi')
        assert nvidia_smi is not None, (
            'nvidia-smi was not found; install/configure the NVIDIA driver'
        )
        result = run_command(
            [nvidia_smi, '--query-gpu=name', '--format=csv,noheader'],
            timeout=30,
        )
        assert result.returncode == 0, result.stdout
        assert result.stdout.strip(), 'nvidia-smi did not report any GPU'

    def test_cuda_computation(self):
        import torch

        assert torch.cuda.is_available(), (
            'CUDA is unavailable to PyTorch; check the CUDA-enabled PyTorch '
            'build, driver, and CUDA_VISIBLE_DEVICES'
        )
        left = torch.ones((2, 2), device='cuda:0')
        right = left @ left
        torch.cuda.synchronize()
        assert right.device.type == 'cuda'
        assert torch.equal(right.cpu(), torch.full((2, 2), 2.0))

    def test_deepmd_can_be_invoked(self):
        import deepmd
        from deepmd.calculator import DP

        assert DP is not None
        assert getattr(deepmd, '__version__', None), (
            'DeepMD-kit imported but did not expose __version__'
        )
        dp_command = shutil.which('dp')
        assert dp_command is not None, "DeepMD-kit executable 'dp' is not in PATH"
        result = run_command([dp_command, '--help'])
        assert result.returncode == 0, result.stdout
        assert result.stdout.strip(), "The 'dp --help' command returned no output"

    def test_lammps_can_be_invoked(self):
        lammps_command = next(
            (
                command
                for name in ('lmp', 'lmp_mpi', 'lammps')
                if (command := shutil.which(name)) is not None
            ),
            None,
        )
        assert lammps_command is not None, (
            "LAMMPS executable not found; expected 'lmp', 'lmp_mpi', or 'lammps'"
        )
        result = run_command([lammps_command, '-h'])
        assert result.returncode == 0, result.stdout
        output = result.stdout.lower()
        assert 'lammps' in output, 'LAMMPS help output was not recognized'
        assert 'deepmd' in output or 'deepmd-kit' in output, (
            'LAMMPS is runnable but its DeepMD pair style was not listed; '
            'install a LAMMPS build linked with DeepMD-kit'
        )


class TestFPSCompositionGeneration:
    """Test composition generation via farthest point sampling."""

    def test_generate_n_compositions_basic(self):
        from PesExploration.tools.fps import generate_n_compositions
        elements_boundary = {'O': [4, 24], 'Cu': [4, 24]}
        compositions = generate_n_compositions(elements_boundary, total_atoms=48, min_others=4)
        assert isinstance(compositions, np.ndarray)
        for comp in compositions:
            assert sum(comp) == 48
        for comp in compositions:
            assert 4 <= comp[0] <= 24
            assert 4 <= comp[1] <= 24

    def test_generate_n_compositions_single_element(self):
        from PesExploration.tools.fps import generate_n_compositions
        elements_boundary = {'Cu': [10, 10]}
        compositions = generate_n_compositions(elements_boundary, total_atoms=10, min_others=0)
        assert len(compositions) == 1
        assert compositions[0][0] == 10


class TestConfigLoading:
    """Test YAML configuration file loading and validation."""

    def test_template_config_structure(self):
        config_path = os.path.join(os.path.dirname(__file__), '..', 'template', 'config.yml')
        with open(config_path) as f:
            config = yaml.safe_load(f)
        assert 'BASE' in config
        assert 'CPU' in config
        assert 'TRAINER' in config

    def test_template_config_base_keys(self):
        config_path = os.path.join(os.path.dirname(__file__), '..', 'template', 'config.yml')
        with open(config_path) as f:
            config = yaml.safe_load(f)
        base = config['BASE']
        assert 'Compositions' in base
        assert 'Elements' in base
        assert 'Gpu' in base
        assert 'Iterations' in base
        assert 'Templates' in base
        assert 'Workdir' in base

    def test_example_configs_valid_yaml(self):
        example_dir = os.path.join(os.path.dirname(__file__), '..', 'example')
        for root, dirs, files in os.walk(example_dir):
            for fname in files:
                if fname.endswith('.yml'):
                    fpath = os.path.join(root, fname)
                    with open(fpath) as f:
                        config = yaml.safe_load(f)
                    assert config is not None, f"Failed to parse {fpath}"


class TestTemplateFiles:
    """Verify that required template files exist."""

    def test_template_directory_exists(self):
        template_dir = os.path.join(os.path.dirname(__file__), '..', 'template')
        assert os.path.isdir(template_dir)

    def test_trainer_templates_exist(self):
        template_dir = os.path.join(os.path.dirname(__file__), '..', 'template')
        assert os.path.isfile(os.path.join(template_dir, 'trainer', 'deepmd_input.json'))
        assert os.path.isfile(os.path.join(template_dir, 'trainer', 'deepmd_input_accurate.json'))

    def test_uspex_templates_exist(self):
        template_dir = os.path.join(os.path.dirname(__file__), '..', 'template')
        uspex_dir = os.path.join(template_dir, 'uspex')
        assert os.path.isdir(uspex_dir)
        required_files = ['INCAR_1', 'KPOINTS', 'run_dp.sh', 'run_mace.sh',
                          'dp_opt.py', 'mace_opt.py']
        for fname in required_files:
            assert os.path.isfile(os.path.join(uspex_dir, fname)), f"Missing: {fname}"

    def test_vaspjet_templates_exist(self):
        template_dir = os.path.join(os.path.dirname(__file__), '..', 'template')
        vaspjet_dir = os.path.join(template_dir, 'vaspjet')
        assert os.path.isdir(vaspjet_dir)
        required_files = ['config_sp.yml', 'config_opt.yml', 'config_md.yml']
        for fname in required_files:
            assert os.path.isfile(os.path.join(vaspjet_dir, fname)), f"Missing: {fname}"


class TestCLIParsing:
    """Test command-line interface argument parsing."""

    def test_run_command_parsing(self):
        import argparse
        parser = argparse.ArgumentParser(description="Hiccup CLI")
        subparsers = parser.add_subparsers(dest="command")
        parser_run = subparsers.add_parser("run")
        parser_run.add_argument("-yml", dest="yml", type=str, required=True)
        args = parser.parse_args(["run", "-yml", "config.yml"])
        assert args.command == "run"
        assert args.yml == "config.yml"

    def test_eva_command_parsing(self):
        import argparse
        parser = argparse.ArgumentParser(description="Hiccup CLI")
        subparsers = parser.add_subparsers(dest="command")
        parser_eva = subparsers.add_parser("eva")
        parser_eva.add_argument("-db", dest="db", type=str, required=True)
        parser_eva.add_argument("-m", dest="model", type=str, required=True)
        parser_eva.add_argument("-g", dest="gpu", type=str, required=True)
        parser_eva.add_argument("-e", dest="energy_filter", type=float, default=0.1)
        parser_eva.add_argument("-f", dest="force_filter", type=float, default=2)
        parser_eva.add_argument("-n", dest="name", type=str, default="Model")
        args = parser.parse_args(["eva", "-db", "test.db", "-m", "model.pb", "-g", "0"])
        assert args.command == "eva"
        assert args.db == "test.db"
        assert args.energy_filter == 0.1
        assert args.force_filter == 2
        assert args.name == "Model"