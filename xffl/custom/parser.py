"""Custom xFFL argument parser

The custom xFFL argument parser is essentially a Python standard ArgumentParser, but with a few standard mandatory arguments necessary for the workflow handling
"""

import argparse

from xffl.custom.types import FolderLike


class ArgumentParser(argparse.ArgumentParser):

    def __init__(
        self,
        prog=None,
        usage=None,
        description=None,
        epilog=None,
        parents=...,
        formatter_class=...,
        prefix_chars="-",
        fromfile_prefix_chars=None,
        argument_default=None,
        conflict_handler="error",
        add_help=True,
        allow_abbrev=True,
        exit_on_error=True,
    ):
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
            help="Path to the model's folder",
            type=FolderLike,
            deafult=None,
        )

        self.add_argument(
            "-d",
            "--dataset",
            help="Path to the dataset's folder",
            type=FolderLike,
            deafult=None,
        )

        self.add_argument(
            "-w",
            "--workspace",
            help="Path to the folder containing the necessary Python scripts to run the training",
            type=FolderLike,
            deafult=None,
        )

        self.add_argument(
            "-o",
            "--output",
            help="Path to the model saving folder",
            type=FolderLike,
            deafult=None,
        )
