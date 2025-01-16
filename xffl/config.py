"""Guided configuration files creation for xFFL

This script guides the user in the creation of the StreamFlow and CWL configuration files
necessary to run xFFL workloads across different HPCs
"""

import argparse
import json
import os
import stat
from pathlib import Path

import yaml

from xffl.templates.cwl import (
    get_aggregate_step,
    get_config,
    get_main_cwl,
    get_round_cwl,
    get_training_step,
    get_workflow_step,
)
from xffl.templates.sh import get_aggregate, get_run_sh
from xffl.templates.streamflow import get_streamflow_config
from xffl.utils import check_input, resolve_path


def create_deployment(args: argparse.Namespace):
    """Gathers from the user all the necessary parameters to generate a valid StreamFlow file for xFFL

    :param args: Command line arguments
    :type args: argparse.Namespace
    :raises FileNotFoundError: If the command-line provided workdir does not exists
    :raises FileExistsError: If the command-line provided project name already exists
    """

    # Command-line parameters check
    workdir = resolve_path(args.workdir)
    if not Path(workdir).exists():
        raise FileNotFoundError(
            f"The provided working directory path {workdir} does not exists"
        )
    if Path(workdir, args.project).exists():
        raise FileExistsError(
            f"Impossible create project {args.project} inside {workdir}: directory already exists"
        )
    workdir = os.path.join(workdir, args.project)
    os.makedirs(workdir)

    # StreamFlow guided configuration
    insert = True
    facilities = set()
    streamflow_config = get_streamflow_config()
    main_cwl = get_main_cwl()
    round_cwl = get_round_cwl()
    cwl_config = get_config()

    # todo: add user interaction
    # fixme: remove aggregation_script
    os.makedirs(os.path.join(workdir, "cwl", "data", "model_placeholder"))
    with open(
        os.path.join(workdir, "cwl", "data", "model_placeholder", "file.txt"), "w"
    ) as fd:
        fd.write("Hello")
    os.makedirs(os.path.join(workdir, "cwl", "data", "tokenizer_placeholder"))
    os.makedirs(os.path.join(workdir, "cwl", "scripts"))
    with open(os.path.join(workdir, "cwl", "scripts", "aggregation.py"), "w") as fd:
        fd.write(get_aggregate())
    cwl_config |= {
        "model": {"class": "Directory", "path": "data/model_placeholder"},
        "tokenizer": {"class": "Directory", "path": "data/tokenizer_placeholder"},
        "model_basename": "ciao",
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
        facilities.add(name)

        address = input(f"Type {name}'s frontend node address [IP:port]: ")
        username = input(f"{name}'s username: ")

        key = check_input(
            f"Path to {name}'s SSH key file: ",
            "{} does not exists.",
            lambda key: os.path.exists(key),
            is_path=True,
        )

        remote_workdir = input("Path to the facility's working directory: ")

        # todo: list the needed pragmas
        slurm_template = check_input(
            f"Path to {name}'s SLURM template with the required directives: ",
            "{} does not exists.",
            lambda path: os.path.exists(path),
            is_path=True,
        )

        step_config = {
            "step": f"/iteration/training_on_{name}",
            "target": [
                {"deployment": name, "service": "pragma"},
                # {
                #     "port": f"/repository_{name}",
                #     "target": {
                #         "deployment": f"{name}-ssh",
                #         "workdir": code_path,
                #     },
                # },
            ],
        }

        main_cwl["inputs"] |= {
            f"facility_{name}": "string",
            f"repository_{name}": "Directory",
            f"test_samples_{name}": "int",
            f"train_samples_{name}": "int",
            f"gpus_per_node_{name}": "int",
        }
        main_cwl["steps"]["iteration"]["in"] |= {
            f"facility_{name}": f"facility_{name}",
            f"repository_{name}": f"repository_{name}",
            f"test_samples_{name}": f"test_samples_{name}",
            f"train_samples_{name}": f"train_samples_{name}",
            f"gpus_per_node_{name}": f"gpus_per_node_{name}",
        }

        round_cwl["inputs"] |= {
            f"facility_{name}": "string",
            f"repository_{name}": "Directory",
            f"test_samples_{name}": "int",
            f"train_samples_{name}": "int",
            f"gpus_per_node_{name}": "int",
        }
        round_cwl["steps"] |= get_workflow_step(name)

        round_cwl["steps"]["merge"]["in"] |= {name: f"training_on_{name}/output_model"}
        round_cwl["steps"]["merge"]["run"]["inputs"] |= {name: "Directory"}
        round_cwl["steps"]["merge"]["run"]["expression"].append(f"inputs.{name}")

        # fixme: insert correct values
        os.makedirs(
            os.path.join(workdir, "cwl", "data", f"repository_placeholder_{name}")
        )
        cwl_config |= {
            f"facility_{name}": name,
            f"repository_{name}": {
                "class": "Directory",
                "path": f"data/repository_placeholder_{name}",
            },
            f"test_samples_{name}": 100,
            f"train_samples_{name}": 100,
            f"gpus_per_node_{name}": 4,
        }

        deployment_config = {
            f"{name}-ssh": {
                "type": "ssh",
                "config": {
                    "nodes": [address],
                    "username": username,
                    "sshKey": key,
                },
                "workdir": remote_workdir,
            },
            name: {
                "type": "slurm",
                "config": {"services": {"pragma": {"file": slurm_template}}},
                "wraps": f"{name}-ssh",
                "workdir": remote_workdir,
            },
        }

        streamflow_config["workflows"]["xffl"]["bindings"].append(step_config)
        streamflow_config["deployments"] |= deployment_config

        if args.verbose:
            print(
                f"Inserted the following record for {name} in the StreamFlow file:",
                json.dumps(step_config),
                json.dumps(deployment_config),
                sep="\n",
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
    os.makedirs(os.path.join(workdir, "cwl", "scripts"), exist_ok=True)
    with open(os.path.join(workdir, "cwl", "scripts", "run.sh"), "w") as outfile:
        outfile.write(get_run_sh())
    usr_permissions = stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR
    grp_permissions = stat.S_IRGRP | stat.S_IXGRP
    os.chmod(
        os.path.join(workdir, "cwl", "scripts", "run.sh"),
        usr_permissions | grp_permissions,
    )


def main(args: argparse.Namespace):
    print("*** Cross-Facility Federated Learning (xFFL) - Guided configuration ***\n")
    create_deployment(args)
    print("\n*** Cross-Facility Federated Learning (xFFL) - Guided configuration ***")


if __name__ == "__main__":
    from xffl.cli.parser import config_parser

    try:
        main(config_parser.parse_args())
    except KeyboardInterrupt:
        print("Unexpected keyboard interrupt")
