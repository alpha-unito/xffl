"""Custom xFFL argument parser

The custom xFFL argument parser is essentially a Python standard ArgumentParser, but with a few standard mandatory arguments necessary for the workflow handling
"""

import argparse

from xffl.custom import DATASETS, MODELS
from xffl.custom.types import FolderLike


class ArgumentParser(argparse.ArgumentParser):
    """xFFL argument parser"""

    def __init__(
        self,
        prog=None,
        usage=None,
        description=None,
        epilog=None,
        parents=[],
        formatter_class=argparse.HelpFormatter,
        prefix_chars="-",
        fromfile_prefix_chars=None,
        argument_default=None,
        conflict_handler="error",
        add_help=True,
        allow_abbrev=True,
        exit_on_error=True,
    ):
        """Creates a new instance of the xFFL argument parser

        The returned parser is an instance of the standard argparse.ArgumentParser with a few already added arguments, necessary for the xFFL inner workings:
            -mn/--model-name      Model's name
            -mp/--model-path      Path to the model's folder
            -dn/--dataset-name    Dataset's name
            -dp/--dataset-path    Path to the dataset's folder
            -w/--workspace        Path to the folder containing the necessary Python scripts to run the training
            -o/--output           Path to the model saving folder
        """
        super().__init__(
            prog,
            usage,
            description,
            epilog,
            parents,
            formatter_class,
            prefix_chars,
            fromfile_prefix_chars,
            argument_default,
            conflict_handler,
            add_help,
            allow_abbrev,
            exit_on_error,
        )

        self.add_argument(
            "-m",
            "--model",
            help="Path to the model's configuration class",
            type=str,
            required=True,
            choices=list(MODELS.keys()),
        )

        self.add_argument(
            "-d",
            "--dataset",
            help="Dataset's name",
            type=str,
            required=True,
            choices=list(DATASETS.keys()),
        )

        self.add_argument(
            "-w",
            "--workspace",
            help="Path to the folder containing the necessary Python scripts to run the training",
            type=FolderLike,
            default=None,
        )

        self.add_argument(
            "-o",
            "--output",
            help="Path to the model saving folder",
            type=FolderLike,
            default=None,
        )
