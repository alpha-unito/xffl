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
    AggregateStep,
    CWLConfig,
    MainWorkflow,
    RoundWorkflow,
    TrainingStep,
)
from xffl.workflow.templates.streamflow import StreamFlowFile

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

    aggregate_cwl = AggregateStep()
    cwl_config = CWLConfig()
    main_cwl = MainWorkflow()
    round_cwl = RoundWorkflow()
    streamflow_config = StreamFlowFile()
    training_cwl = TrainingStep()

    # todo: add user interaction
    cwl_config.content |= {
        "model": {"class": "Directory", "path": "llama3.1-8b"},
        "model_basename": "llama3",
        "max_rounds": 2,
        "epochs": 1,
        "script_aggregation": {
            "class": "File",
            "path": os.path.join("py_scripts", "aggregation.py"),
        },
        "executable": "examples/llama/client/src/training.py",
    }
    while insert:

        # name = check_input(
        #     "Type facility's logic name: ",
        #     "Facility name {} already used.",
        #     lambda name: name not in facilities,
        # )
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

        name = "leonardo"
        facilities.add(name)
        address = "login.leonardo.cineca.it"
        username = "amulone1"
        key = "/home/ubuntu/.ssh/cineca-certificates/amulone1_ecdsa"
        step_workdir = "/leonardo_scratch/fast/uToID_bench/tmp/streamflow/ssh"
        slurm_template = "/home/ubuntu/xffl/examples/llama/client/slurm_templates/leonardo.slurm"  # todo: copy the template in the project dir?

        # todo: query to user
        code_path = "/leonardo/home/userexternal/amulone1/xffl"
        dataset_path = "/leonardo_scratch/fast/uToID_bench/23_llama_sc24/datasets"
        image_path = "/leonardo_scratch/fast/EUHPC_B18_066/client.sif"
        val_batch_size = 1
        train_batch_size = 4
        subsampling = 16
        model_path = "/leonardo_scratch/fast/uToID_bench/23_llama_sc24/worker/workspace/llama3.1-8b"  # fixme: remote it

        main_cwl.add_inputs(name)
        round_cwl.add_inputs(name)

        streamflow_config.add_deployment(
            name, address, username, key, step_workdir, slurm_template
        )
        streamflow_config.add_step_binding(
            name, code_path, dataset_path, model_path, image_path
        )
        streamflow_config.add_inputs(name)

        cwl_config.add_input_values(
            name,
            code_path,
            image_path,
            dataset_path,
            val_batch_size,
            train_batch_size,
            subsampling,
        )
        cwl_config.add_inputs(name)

        logger.debug(
            "\n".join(
                [
                    f"Inserted the following record for {name} in the StreamFlow file:",
                    json.dumps(streamflow_config.step_bindings[name], indent=2),
                    json.dumps(streamflow_config.deployments[name], indent=2),
                ]
            )
        )

        insert = check_input(
            "Insert another facility? [y/n]: ",
            "Answer {} not accepted.",
            lambda answer: answer.lower() in ["y", "yes", "n", "no"],
        ) in ["y", "yes"]
    round_cwl.update_merge_step()

    # YAML exportation
    ## StreamFlow file
    with open(os.path.join(workdir, "streamflow.yml"), "w") as outfile:
        yaml.dump(
            streamflow_config.save(), outfile, default_flow_style=False, sort_keys=False
        )

    ## CWL files
    os.makedirs(os.path.join(workdir, "cwl", "clt"))
    with open(os.path.join(workdir, "cwl", "main.cwl"), "w") as outfile:
        outfile.write("#!/usr/bin/env cwl-runner\n")
        yaml.dump(main_cwl.save(), outfile, default_flow_style=False, sort_keys=False)
    with open(os.path.join(workdir, "cwl", "round.cwl"), "w") as outfile:
        outfile.write("#!/usr/bin/env cwl-runner\n")
        yaml.dump(round_cwl.save(), outfile, default_flow_style=False, sort_keys=False)
    with open(os.path.join(workdir, "cwl", "clt", "aggregate.cwl"), "w") as outfile:
        outfile.write("#!/usr/bin/env cwl-runner\n")
        yaml.dump(
            aggregate_cwl.save(), outfile, default_flow_style=False, sort_keys=False
        )
    with open(os.path.join(workdir, "cwl", "clt", "training.cwl"), "w") as outfile:
        outfile.write("#!/usr/bin/env cwl-runner\n")
        yaml.dump(
            training_cwl.save(), outfile, default_flow_style=False, sort_keys=False
        )
    ## CWL config file
    with open(os.path.join(workdir, "cwl", "config.yml"), "w") as outfile:
        yaml.dump(cwl_config.save(), outfile, default_flow_style=False, sort_keys=False)
    ## Scripts
    shutil.copytree(
        os.path.join(DEFAULT_xFFL_DIR, "workflow", "scripts"),
        os.path.join(workdir, "cwl", "scripts"),
    )
    # fixme: remove aggregation_script
    os.makedirs(os.path.join(workdir, "cwl", "py_scripts"))
    shutil.copy(
        os.path.join(
            DEFAULT_xFFL_DIR, "workflow", "templates", "aggregation_application.py"
        ),
        os.path.join(workdir, "cwl", "py_scripts"),
    )


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
    try:
        config(args)
    except Exception as e:
        logger.exception(e)
        raise
    finally:
        logger.info(
            "*** Cross-Facility Federated Learning (xFFL) - Guided configuration ***"
        )


if __name__ == "__main__":
    from xffl.cli.parser import config_parser

    try:
        main(config_parser.parse_args())
    except KeyboardInterrupt as e:
        logger.exception(e)
