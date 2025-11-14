#!/usr/bin/env python3
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import argparse
from main_flow import Hiccup
from PesExploration.tools.data_filter_and_analysis import analysis
import yaml


def main():
    parser = argparse.ArgumentParser(description="Here is Hiccup!")
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    #
    # 为 run 命令创建子解析器
    parser_run = subparsers.add_parser(
        "run",
        help="Run the Hiccup fow with specified configurations.")
    parser_run.add_argument(
        "-yml",
        dest="yml",
        type=str,
        help="Path to the .yml config. Typing: str, Required: True",
        required=True
    )

    # 为 eva 命令创建子解析器
    parser_eva = subparsers.add_parser(
        "eva",
        help="Evaluate the dp model with specified db file.")
    parser_eva.add_argument(
        "-db",
        dest="db",
        type=str,
        help="Path to the .db file. Typing: str, Required: True",
        required=True
    )
    parser_eva.add_argument(
        "-m",
        dest="model",
        type=str,
        help="Path to the dp model. Typing: str, Required: True",
        required=True
    )
    parser_eva.add_argument(
        "-g",
        dest="gpu",
        type=str,
        help="Path to the output file. Typing: str, Required: True",
        required=True
    )
    parser_eva.add_argument(
        "-e",
        dest="energy_filter",
        type=float,
        help="Energy filter. Typing: float, Required: False, Default: 0.1",
        default=0.1
    )
    parser_eva.add_argument(
        "-f",
        dest="force_filter",
        type=float,
        help="Force filter. Typing: float, Required: False, Default: 2",
        default=2
    )
    parser_eva.add_argument(
        "-n",
        dest="name",
        type=str,
        help="Model name. Typing: str, Required: False, Default: Model",
        default="Model"
    )

    args = parser.parse_args()
    if args.command == "run":
        with open(args.yml) as file:
            dict_value = yaml.load(file.read(), Loader=yaml.FullLoader)
        config = dict_value
        print("Running Hiccup with configurations in", args.yml)
        hiccup = Hiccup(config)
        hiccup.run_dp_GA()
    elif args.command == "eva":
        analysis(db_path=args.db,
                 model_path=args.model,
                 gpu_ids=args.gpu,
                 energy_filter=args.energy_filter,
                 force_filter=args.force_filter,
                 model_name=args.name)

    else:
        print("Please specify a command to run Hiccup.")

