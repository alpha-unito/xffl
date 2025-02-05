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

from xffl.custom.types import FolderLike
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
from xffl.workflow.utils import from_args_to_cwl

logger: Logger = getLogger(__name__)
"""Default xFFL logger"""

from argparse import ArgumentParser

parser = ArgumentParser(
    prog="Cross-Facility Federated Learning (xFFL) - LLaMA example",
    description="This xFFL example pre-trains a LLaMA-3.1 8B model on multiple HPC infrastructures.",
)

parser.add_argument(
    "-attn",
    "--attention",
    help="Type of attention implementation to use",
    type=str,
    default="flash_attention_2",
    choices=["sdpa", "eager", "flash_attention_2"],
)

parser.add_argument("-on", "--online", help="Online mode", action="store_true")

parser.add_argument(
    "-s",
    "--seed",
    help="Random execution seed (for reproducibility purposes)",
    type=int,
    default=None,
)

parser.add_argument("-wb", "--wandb", help="Enable WandB", action="store_true")

parser.add_argument(
    "-name",
    "--wandb-name",
    help="WandB group name",
    type=str,
    default="LLaMA-3.1 8B",
)

parser.add_argument(
    "-mode",
    "--wandb-mode",
    help="WandB mode",
    type=str,
    default="online",
    choices=["online", "offline", "disabled"],
)

parser.add_argument(
    "-sub",
    "--subsampling",
    help="Quantity of data samples to load (for each dataset)",
    type=int,
    default=0,
)

parser.add_argument(
    "-t",
    "--train-batch-size",
    help="Training batch size",
    type=int,
    default=4,
)

parser.add_argument(
    "-v",
    "--val-batch-size",
    help="Validation batch size",
    type=int,
    default=1,
)

parser.add_argument(
    "-ws",
    "--workers",
    help="Number of data loaders workers",
    type=int,
    default=2,
)

parser.add_argument(
    "-lr",
    "--learning-rate",
    help="Learning rate",
    type=float,
    default=1e-4,
)

parser.add_argument(
    "-wd",
    "--weight-decay",
    help="Weight decay",
    type=float,
    default=0,
)

parser.add_argument(
    "-sz",
    "--step-size",
    help="Learning rate scheduler step size",
    type=int,
    default=1,
)

parser.add_argument(
    "-g",
    "--gamma",
    help="Learning rate scheduler gamma",
    type=float,
    default=0.85,
)

parser.add_argument(
    "-om",
    "--output-model",
    help="Saved model name",
    type=str,
    default=None,
)


def config(args: argparse.Namespace):
    """Gathers from the user all the necessary parameters to generate a valid StreamFlow file for xFFL

    :param args: Command line arguments
    :type args: argparse.Namespace
    :raises FileNotFoundError: If the command-line provided workdir does not exists
    :raises FileExistsError: If the command-line provided project folder already exists
    """

    # Project folder and path checks
    logger.debug(f"Verifying working directory {args.workdir}")
    workdir: str = resolve_path(path=args.workdir)
    if Path(workdir).exists():
        workdir: FolderLike = os.path.join(workdir, args.project)
        logger.debug(f"Verifying project directory {workdir}")
        try:
            os.makedirs(workdir)
        except FileExistsError as e:
            raise e
    else:
        raise FileNotFoundError(
            f"The provided working directory path {workdir} does not exists"
        )
    logger.info(f"Project directory successfully created at {workdir}")

    # Guided StreamFlow configuration
    logger.debug(f"Creating the StreamFlow and CWL templates")
    facilities = set()

    aggregate_cwl = AggregateStep()
    cwl_config = CWLConfig()
    main_cwl = MainWorkflow()
    round_cwl = RoundWorkflow()
    streamflow_config = StreamFlowFile()
    training_cwl = TrainingStep()
    logger.debug(f"StreamFlow and CWL templates created")

    # todo: add user interaction
    cwl_config.content |= {
        "model": {"class": "Directory", "path": "llama3.1-8b"},
        "model_basename": "llama3",
        "max_rounds": 2,
        "script_aggregation": {
            "class": "File",
            "path": os.path.join("py_scripts", "aggregation.py"),
        },
        "executable": {
            "class": "File",
            "path": "/home/ubuntu/xffl/examples/llama/client/src/training.py",  # TODO: user interation
        },
    }

    insert = True
    while insert:

        # facility = check_input(
        #     "Type facility's logic facility: ",
        #     "Facility facility {} already used.",
        #     lambda facility: facility not in facilities,
        # )
        # facilities.add(facility)

        # address = input(f"Type {facility}'s frontend node address [IP:port]: ")
        # username = input(f"{facility}'s username: ")

        # key = check_input(
        #     f"Path to {facility}'s SSH key file: ",
        #     "{} does not exists.",
        #     lambda key: os.path.exists(key),
        #     is_path=True,
        # )

        # workdir = input("Path to the facility's working directory: ")

        # # todo: list the needed pragmas
        # slurm_template = check_input(
        #     f"Path to {facility}'s SLURM template with the required directives: ",
        #     "{} does not exists.",
        #     lambda path: os.path.exists(path),
        #     is_path=True,
        # )

        facility = "leonardo"
        facilities.add(facility)

        address = "login.leonardo.cineca.it"
        username = "amulone1"
        key = "/home/ubuntu/.ssh/cineca-certificates/amulone1_ecdsa"
        step_workdir = "/leonardo_scratch/fast/uToID_bench/tmp/streamflow/ssh"
        slurm_template = "/home/ubuntu/xffl/examples/llama/client/slurm_templates/leonardo.slurm"  # todo: copy the template in the project dir?

        # todo: query to user
        code_path = (
            "/leonardo/home/userexternal/amulone1/xffl"  # TODO: potrebbe sparire
        )
        dataset_path = "/leonardo_scratch/fast/uToID_bench/23_llama_sc24/datasets"
        image_path = "/leonardo_scratch/fast/EUHPC_B18_066/client.sif"
        model_path = "/leonardo_scratch/fast/uToID_bench/23_llama_sc24/worker/workspace/llama3.1-8b"  # TODO: sparirà
        # TODO: passare output-dir
        # TODO: far scegliere all'utente il numero di round

        try:
            # from examples.llama.client.src.training import (
            #    parser as parser,  # TODO: questo sarà dinamico
            # )

            training_step_args, main_cwl_args, round_cwl_inputs, config_cwl_args = (
                from_args_to_cwl(parser=parser, arguments=args.arguments)
            )
        except Exception as e:
            raise e

        main_cwl.add_inputs(facility_name=facility, extra_inputs=main_cwl_args)
        round_cwl.add_inputs(facility_name=facility, extra_inputs=round_cwl_inputs)
        cwl_config.add_inputs(
            facility_name=facility,
            extra_inputs={
                "code_path": os.path.basename(code_path),
                "image_path": os.path.basename(image_path),
                "dataset_path": os.path.basename(dataset_path),
            }
            | config_cwl_args,
        )
        training_cwl.add_inputs(facility_name=facility, extra_inputs=training_step_args)

        streamflow_config.add_deployment(
            facility_name=facility,
            address=address,
            username=username,
            ssh_key=key,
            step_workdir=step_workdir,
            slurm_template=slurm_template,
        )

        # TODO: questi vanno chiesti interattivamente all'utente o inclusi in un file di configurazione
        # TODO: va aggiunta anche la cartella di output?
        streamflow_config.add_step_binding(
            facility_name=facility,
            mapping={
                f"repository_{facility}": os.path.dirname(code_path),
                f"dataset_{facility}": os.path.dirname(dataset_path),
                f"image_{facility}": os.path.dirname(image_path),
                "model": os.path.dirname(model_path),  # TODO: Verrà tolto
            },
        )
        streamflow_config.add_inputs(facility_name=facility)

        logger.debug(
            "\n".join(
                [
                    f"Inserted the following record for {facility} in the StreamFlow file:",
                    json.dumps(streamflow_config.step_bindings[facility], indent=2),
                    json.dumps(streamflow_config.deployments[facility], indent=2),
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
    exit_code = 0
    try:
        config(args=args)
    except Exception as e:
        logger.exception(e)
        exit_code = 1
    finally:
        logger.info(
            "*** Cross-Facility Federated Learning (xFFL) - Guided configuration ***"
        )
        return exit_code


if __name__ == "__main__":
    from xffl.cli.parser import config_parser

    try:
        main(args=config_parser.parse_args())
    except KeyboardInterrupt as e:
        logger.exception(e)
