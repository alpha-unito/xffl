"""CLI utility methods"""

from argparse import ArgumentParser
from logging import Logger, getLogger
from typing import Any, Dict

logger: Logger = getLogger(__name__)
"""Default xFFL logger"""


def check_default_value(
    argument_name: str, argument_value: Any, parser: ArgumentParser
) -> Any:
    """Checks if an arguments equals its default value

    :param argument_name: Variable name of the argument
    :type argument_name: str
    :param argument_value: Actual value of the argument
    :type argument_value: Any
    :param parser: Parser from which the argument originated
    :type parser: ArgumentParser
    :return: Actual value of the argument
    :rtype: Any
    """
    if argument_value == parser.get_default(dest=argument_name):
        logger.warning(f'CLI argument "{argument_name}" has got default value')
    return argument_value


def setup_env(
    args: Dict[str, Any], mapping: Dict[str, str], parser: ArgumentParser
) -> Dict[str, str]:
    """Creates a mapping between the CLI arguments and new enviromental variables

    :param args: CLI arguments
    :type args: Dict[str, Any]
    :param mapping: Mapping between environmental variables and CLI arguments names
    :type mapping: Dict[str, str]
    :param parser: Parser from which the arguments originated
    :type parser: ArgumentParser
    :return: New environment variables dictionary
    :rtype: Dict[str, str]
    """
    env = {}
    for env_var, parse_var in mapping.items():
        env[env_var] = str(check_default_value(parse_var, args[parse_var], parser))

    return env
