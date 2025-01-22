"""Collection of StreamFlow-related template code
"""

from collections.abc import MutableMapping
from typing import Any


def get_streamflow_config() -> MutableMapping[str, Any]:
    """Returns a basic StreamFlow configuration to be updated and saved into a .yml

    :return: a basic StreamFlow configuration for xFFL
    :rtype: MutableMapping[str, Any]
    """

    return {
        "version": "v1.0",
        "workflows": {
            "xffl": {
                "type": "cwl",
                "config": {
                    "file": "cwl/main.cwl",
                    "settings": "cwl/config.yml",
                },
                "bindings": [],
            }
        },
        "deployments": {},
    }
