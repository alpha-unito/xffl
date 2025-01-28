"""Guided configuration files creation for xFFL

This script guides the user in the creation of the StreamFlow and CWL configuration files
necessary to run xFFL workloads across different HPCs
"""

import argparse
import json
import os
import shutil
from logging import Logger, getLogger
from pathlib import Path

import yaml

from xffl.custom.types import PathLike
from xffl.utils.constants import DEFAULT_xFFL_DIR
from xffl.utils.utils import check_input, resolve_path
from xffl.workflow.templates.cwl import (
    get_aggregate_step,
    get_config,
    get_main_cwl,
    get_round_cwl,
    get_training_step,
    get_workflow_step,
)
from xffl.workflow.templates.sh import get_aggregate
from xffl.workflow.templates.streamflow import get_streamflow_config

logger: Logger = getLogger(__name__)
"""Default xFFL logger"""


def config(args: argparse.Namespace):
    """Gathers from the user all the necessary parameters to generate a valid StreamFlow file for xFFL

    :param args: Command line arguments
    :type args: argparse.Namespace
    :raises FileNotFoundError: If the command-line provided workdir does not exists
    :raises FileExistsError: If the command-line provided project name already exists
    """

    # Project folder and path checks
    workdir: PathLike = resolve_path(args.workdir)
    if not Path(workdir).exists():
        raise FileNotFoundError(
            f"The provided working directory path {workdir} does not exists"
        )
    workdir = os.path.join(workdir, args.project)
    if Path(workdir).exists():
        raise FileExistsError(
            f"Impossible creating project {args.project} inside {workdir}: directory already exists"
        )
    os.makedirs(workdir)

    # StreamFlow guided configuration
    insert = True
    facilities = set()
    streamflow_config = get_streamflow_config()
    main_cwl = get_main_cwl()
    round_cwl = get_round_cwl()
    cwl_config = get_config()

    # todo: add user interaction
    cwl_config |= {
        "model": {"class": "Directory", "path": "llama3.1-8b"},
        "model_basename": "llama3",
        "max_rounds": 2,
        "epochs": 1,
        "script_aggregation": {
            "class": "File",
            "path": os.path.join(workdir, "cwl", "scripts", "aggregation.py"),
        },
    }

    while insert:

        name = check_input(
            "Type facility's logic name: ",
            "Facility name {} already used.",
            lambda name: name not in facilities,
        )
        # facilities.add(name)

        # address = input(f"Type {name}'s frontend node address [IP:port]: ")
        # username = input(f"{name}'s username: ")

        # key = check_input(
        #     f"Path to {name}'s SSH key file: ",
        #     "{} does not exists.",
        #     lambda key: os.path.exists(key),
        #     is_path=True,
        # )

        # workdir = input("Path to the facility's working directory: ")

        # # todo: list the needed pragmas
        # slurm_template = check_input(
        #     f"Path to {name}'s SLURM template with the required directives: ",
        #     "{} does not exists.",
        #     lambda path: os.path.exists(path),
        #     is_path=True,
        # )

        facilities.add(name)
        address = "login.leonardo.cineca.it"
        username = "amulone1"
        key = "/home/ubuntu/.ssh/cineca-certificates/amulone1_ecdsa"
        step_workdir = "/leonardo_scratch/fast/uToID_bench/tmp/streamflow/ssh"
        slurm_template = "/home/ubuntu/xffl/examples/llama/client/slurm_templates/leonardo.slurm"  # todo: copy the template in the project dir?

        # todo: query to user
        code_path = "/leonardo/home/userexternal/amulone1/xffl"
        dataset_path = "/leonardo_scratch/fast/uToID_bench/23_llama_sc24/datasets"
        image_path = "/leonardo_scratch/fast/uToID_bench/23_llama_sc24/worker/workspace/worker.sif"
        num_test_sample = 100
        num_train_sample = 1000
        gpu_per_nodes = 4
        model_path = "/leonardo_scratch/fast/uToID_bench/23_llama_sc24/worker/workspace/llama3.1-8b"  # fixme: remote it

        step_config = [
            {
                "step": f"/iteration/training_on_{name}",
                "target": [
                    {"deployment": name, "service": "pragma"},
                ],
            },
            {
                "port": f"/repository_{name}",
                "target": {
                    "deployment": f"{name}",
                    "workdir": os.path.dirname(code_path),
                },
            },
            {
                "port": f"/dataset_{name}",
                "target": {
                    "deployment": f"{name}",
                    "workdir": os.path.dirname(dataset_path),
                },
            },
            {
                "port": f"/image_{name}",
                "target": {
                    "deployment": f"{name}",
                    "workdir": os.path.dirname(image_path),
                },
            },
            {
                # fixme: remote it
                "port": f"/model",
                "target": {
                    "deployment": f"{name}",
                    "workdir": os.path.dirname(model_path),
                },
            },
        ]

        main_cwl["inputs"] |= {
            f"facility_{name}": "string",
            f"repository_{name}": "Directory",
            f"test_samples_{name}": "int",
            f"train_samples_{name}": "int",
            f"repository_{name}": "Directory",
            f"image_{name}": "File",
            f"gpu_per_nodes_{name}": "int",
            f"dataset_{name}": "Directory",
        }
        main_cwl["steps"]["iteration"]["in"] |= {
            f"facility_{name}": f"facility_{name}",
            f"repository_{name}": f"repository_{name}",
            f"test_samples_{name}": f"test_samples_{name}",
            f"train_samples_{name}": f"train_samples_{name}",
            f"repository_{name}": f"repository_{name}",
            f"image_{name}": f"image_{name}",
            f"gpu_per_nodes_{name}": f"gpu_per_nodes_{name}",
            f"dataset_{name}": f"dataset_{name}",
        }

        round_cwl["inputs"] |= {
            f"facility_{name}": "string",
            f"repository_{name}": "Directory",
            f"test_samples_{name}": "int",
            f"train_samples_{name}": "int",
            f"repository_{name}": "Directory",
            f"image_{name}": "File",
            f"gpu_per_nodes_{name}": "int",
            f"dataset_{name}": "Directory",
        }
        round_cwl["steps"] |= get_workflow_step(name)

        round_cwl["steps"]["merge"]["in"] |= {name: f"training_on_{name}/output_model"}
        round_cwl["steps"]["merge"]["run"]["inputs"] |= {name: "Directory"}
        round_cwl["steps"]["merge"]["run"]["expression"].append(f"inputs.{name}")

        # fixme: insert correct values
        cwl_config |= {
            f"facility_{name}": name,
            f"repository_{name}": {
                "class": "Directory",
                "path": os.path.basename(code_path),
            },
            f"image_{name}": {
                "class": "File",
                "path": os.path.basename(image_path),
            },
            f"dataset_{name}": {
                "class": "Directory",
                "path": os.path.basename(dataset_path),
            },
            f"test_samples_{name}": num_test_sample,
            f"train_samples_{name}": num_train_sample,
            f"gpu_per_nodes_{name}": gpu_per_nodes,
        }

        if name == "local":
            deployment_config = {
                f"{name}-ssh": {
                    "type": "ssh",
                    "config": {
                        "nodes": [address],
                        "username": username,
                        "sshKey": key,
                    },
                    "workdir": step_workdir,
                },
                name: {
                    "type": "slurm",
                    "config": {"services": {"pragma": {"file": slurm_template}}},
                    "wraps": f"{name}-ssh",
                    "workdir": step_workdir,
                },
            }
        else:
            deployment_config = {
                name: {
                    "type": "local",
                    "config": {},
                    "workdir": step_workdir,
                },
            }

        streamflow_config["workflows"]["xffl"]["bindings"].extend(step_config)
        streamflow_config["deployments"] |= deployment_config

        logger.debug(
            "\n".join(
                [
                    f"Inserted the following record for {name} in the StreamFlow file:",
                    json.dumps(step_config, indent=2),
                    json.dumps(deployment_config, indent=2),
                ]
            )
        )

        insert = check_input(
            "Insert another facility? [y/n]: ",
            "Answer {} not accepted.",
            lambda answer: answer.lower() in ["y", "yes", "n", "no"],
        ) in ["y", "yes"]

    # YAML exportation
    ## StreamFlow file
    with open(os.path.join(workdir, "streamflow.yml"), "w") as outfile:
        yaml.dump(streamflow_config, outfile, default_flow_style=False, sort_keys=False)

    inputs_list = ",".join(round_cwl["steps"]["merge"]["run"]["expression"])
    round_cwl["steps"]["merge"]["run"]["expression"] = (
        "$({'models': [" + inputs_list + "] })"
    )
    ## CWL files
    os.makedirs(os.path.join(workdir, "cwl", "clt"))
    with open(os.path.join(workdir, "cwl", "main.cwl"), "w") as outfile:
        outfile.write("#!/usr/bin/env cwl-runner\n")
        yaml.dump(main_cwl, outfile, default_flow_style=False, sort_keys=False)
    with open(os.path.join(workdir, "cwl", "round.cwl"), "w") as outfile:
        outfile.write("#!/usr/bin/env cwl-runner\n")
        yaml.dump(round_cwl, outfile, default_flow_style=False, sort_keys=False)
    with open(os.path.join(workdir, "cwl", "clt", "aggregate.cwl"), "w") as outfile:
        outfile.write("#!/usr/bin/env cwl-runner\n")
        yaml.dump(
            get_aggregate_step(), outfile, default_flow_style=False, sort_keys=False
        )
    with open(os.path.join(workdir, "cwl", "clt", "training.cwl"), "w") as outfile:
        outfile.write("#!/usr/bin/env cwl-runner\n")
        yaml.dump(
            get_training_step(), outfile, default_flow_style=False, sort_keys=False
        )
    ## CWL config file
    with open(os.path.join(workdir, "cwl", "config.yml"), "w") as outfile:
        yaml.dump(cwl_config, outfile, default_flow_style=False, sort_keys=False)
    ## Scripts
    shutil.copytree(
        os.path.join(DEFAULT_xFFL_DIR, "workflow", "scripts"),
        os.path.join(workdir, "cwl", "scripts"),
    )
    # fixme: remove aggregation_script
    os.makedirs(os.path.join(workdir, "cwl", "py_scripts"))
    with open(os.path.join(workdir, "cwl", "py_scripts", "aggregation.py"), "w") as fd:
        fd.write(get_aggregate())


def main(args: argparse.Namespace) -> int:
    """xFFL project's guided configuration entrypoint

    :param args: Command line arguments
    :type args: argparse.Namespace
    :return: Exit code
    :rtype: int
    """
    logger.info(
        "*** Cross-Facility Federated Learning (xFFL) - Guided configuration ***"
    )
    exit_code = 0
    try:
        config(args)
    except (FileNotFoundError, FileExistsError) as e:
        logger.exception(e.strerror)
        exit_code = 1
    finally:
        logger.info(
            "*** Cross-Facility Federated Learning (xFFL) - Guided configuration ***"
        )
        return exit_code


if __name__ == "__main__":
    from xffl.cli.parser import config_parser

    try:
        main(config_parser.parse_args())
    except KeyboardInterrupt:
        logger.exception("Unexpected keyboard interrupt")
