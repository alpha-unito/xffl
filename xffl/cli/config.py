"""Guided configuration files creation for xFFL

This script guides the user in the creation of the StreamFlow and CWL configuration files
necessary to run xFFL workloads across different HPCs
"""

import argparse
import json
import os
import shutil
from logging import Logger, getLogger

import yaml

from xffl.cli.parser import config_parser
from xffl.cli.utils import check_and_create_workdir, check_cli_arguments
from xffl.utils.constants import DEFAULT_xFFL_DIR
from xffl.utils.utils import check_input
from xffl.workflow.templates.cwl import (
    AggregateStep,
    CWLConfig,
    MainWorkflow,
    RoundWorkflow,
    TrainingStep,
)
from xffl.workflow.templates.streamflow import StreamFlowFile
from xffl.workflow.utils import from_args_to_cwl, import_from_path

logger: Logger = getLogger(__name__)
"""Default xFFL logger"""


def config(args: argparse.Namespace) -> None:  # TODO: check exceptions raised
    """Gathers from the user all the necessary parameters to generate a valid StreamFlow file for xFFL

    :param args: Command line arguments
    :type args: argparse.Namespace
    :raises FileNotFoundError: If the command-line provided workdir does not exists
    :raises FileExistsError: If the command-line provided project folder already exists
    """

    # Check the CLI arguments
    check_cli_arguments(args=args, parser=config_parser)

    # Project folder and path checks
    logger.debug(
        f"Verifying working directory {args.workdir} and project name {args.project}"
    )
    try:
        workdir = check_and_create_workdir(
            workdir_path=args.workdir, project_name=args.project
        )
    except (FileExistsError, FileNotFoundError) as e:
        raise e

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

    # TODO: add user interaction
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
        slurm_template = "/home/ubuntu/xffl/examples/llama/client/slurm_templates/leonardo.slurm"  # TODO: copy the template in the project dir?
        code_path = (
            "/leonardo/home/userexternal/amulone1/xffl"  # TODO: potrebbe sparire
        )
        dataset_path = "/leonardo_scratch/fast/uToID_bench/23_llama_sc24/datasets"
        image_path = "/leonardo_scratch/fast/EUHPC_B18_066/client.sif"
        model_path = "/leonardo_scratch/fast/uToID_bench/23_llama_sc24/worker/workspace/llama3.1-8b"  # TODO: sparirà
        parser_path = "/leonardo_scratch/large/userexternal/gmittone/xffl/examples/llama/client/src/parser.py"
        parser_name = "parser"
        # TODO: passare output-dir
        # TODO: far scegliere all'utente il numero di round

        # Command line arguments extraction from user's parser arguments
        logger.debug(
            f"Dinamically loading ArgumentParser '{parser_name}' from file '{parser_path}'"
        )
        try:
            executable_parser_module = import_from_path(
                module_name=parser_name, file_path=parser_path
            )
            logger.info(
                f"Command line argument parser '{parser_name}' from file '{parser_path}' correctly imported"
            )

            arg_to_bidding, arg_to_type, arg_to_value = from_args_to_cwl(
                parser=executable_parser_module.parser,  # TODO: the name "parser" is hardcoded
                arguments=args.arguments,
                start_position=training_cwl.position,
            )
        except Exception as e:
            raise e

        # Creating CWL configuration
        logger.debug(f"CWL configuration population...")
        main_cwl.add_inputs(facility_name=facility, extra_inputs=arg_to_type)
        round_cwl.add_inputs(facility_name=facility, extra_inputs=arg_to_type)
        cwl_config.add_inputs(
            facility_name=facility,
            extra_inputs={
                "repository": {
                    "class": "Directory",
                    "path": os.path.basename(code_path),
                },
                "image": {"class": "File", "path": os.path.basename(image_path)},
                "dataset": {
                    "class": "Directory",
                    "path": os.path.basename(dataset_path),
                },
            }
            | arg_to_value,
        )
        training_cwl.add_inputs(facility_name=facility, extra_inputs=arg_to_bidding)

        # Creating StreamFlow configuration
        logger.debug(f"StreamFlow configuration population...")
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

    return 0


def main(args: argparse.Namespace) -> int:
    """xFFL project's guided configuration entrypoint

    :param args: Command line arguments
    :type args: argparse.Namespace
    :raises e: Exception raised during the configuration
    :return: Exit code
    :rtype: int
    """
    logger.info(
        "*** Cross-Facility Federated Learning (xFFL) - Guided configuration ***"
    )
    try:
        config(args=args)
    except Exception as e:
        logger.exception(e)
        raise e
    finally:
        logger.info(
            "*** Cross-Facility Federated Learning (xFFL) - Guided configuration ***"
        )
    return 0


if __name__ == "__main__":
    try:
        main(args=config_parser.parse_args())
    except KeyboardInterrupt as e:
        logger.exception(e)
