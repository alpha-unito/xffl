"""Collection of CWL-related template code
"""

from collections.abc import MutableMapping
from typing import Any


def get_aggregate_step() -> MutableMapping[str, Any]:
    """Get the CWL file standard content for a xFFL application aggregation step

    :return: Dict template of a CWL xFFL aggregation step
    :rtype: MutableMapping[str, Any]
    """
    return {
        "cwlVersion": "v1.2",
        "class": "CommandLineTool",
        "requirements": {
            "InlineJavascriptRequirement": {},
            # todo: create the image llm-aggregator on alphaunito DockerHub
            # "DockerRequirement": {"dockerPull": "alphaunito/llm-aggregator"}
        },
        "baseCommand": ["python"],
        "arguments": [
            {
                "position": 5,
                "valueFrom": "$(inputs.model_basename)-merged-round$(inputs.round)",
                "prefix": "--outname",
            }
        ],
        "inputs": {
            "script": {"type": "File", "inputBinding": {"position": 1}},
            "models": {
                "type": {
                    "type": "array",
                    "items": "Directory",
                    "inputBinding": {"position": 2, "prefix": "--model"},
                }
            },
            "model_basename": {"type": "string"},
            "round": {"type": "int"},
        },
        "outputs": {
            "output_model": {
                "type": "Directory",
                "outputBinding": {
                    "glob": "$(inputs.model_basename)-merged-round$(inputs.round)"
                },
            }
        },
    }


def get_config() -> MutableMapping[str, Any]:
    return {
        "script_train": {"class": "Directory", "path": "scripts"},
    }


def get_main_cwl() -> MutableMapping[str, Any]:
    """Get the CWL main file standard content for a xFFL application

    :return: main.cwl template
    :rtype: MutableMapping[str, Any]
    """
    return {
        "cwlVersion": "v1.2",
        "class": "Workflow",
        "$namespaces": {"cwltool": "http://commonwl.org/cwltool#"},
        "requirements": {
            "SubworkflowFeatureRequirement": {},
            "InlineJavascriptRequirement": {},
            "StepInputExpressionRequirement": {},
        },
        "inputs": {
            "script_train": "Directory",
            "script_aggregation": "File",
            "model": "Directory",
            "epochs": "int",
            "model_basename": "string",
            "max_rounds": "int",
        },
        "outputs": {
            "output_model": {
                "type": "Directory",
                "outputSource": "iteration/output_model",
            }
        },
        "steps": {
            "iteration": {
                "run": "round.cwl",
                "in": {
                    "script_train": "script_train",
                    "script_aggregation": "script_aggregation",
                    "model": "model",
                    "epochs": "epochs",
                    "model_basename": "model_basename",
                    "round": {"default": 0},
                    "max_rounds": "max_rounds",
                },
                "out": ["output_model"],
                "requirements": {
                    "cwltool:Loop": {
                        "loopWhen": "$(inputs.round < inputs.max_rounds)",
                        "loop": {
                            "round": {"valueFrom": "$(inputs.round + 1)"},
                            "model": "output_model",
                        },
                        "outputMethod": "last",
                    }
                },
            }
        },
    }


def get_round_cwl() -> MutableMapping[str, Any]:
    """Get the CWL round file standard content for a xFFL application

    :return: round.cwl template
    :rtype: MutableMapping[str, Any]
    """
    return {
        "cwlVersion": "v1.2",
        "class": "Workflow",
        "inputs": {
            "script_train": "Directory",
            "script_aggregation": "File",
            "model": "Directory",
            "epochs": "int",
            "model_basename": "string",
            "max_rounds": "int",
            "round": "int",
        },
        "outputs": {
            "output_model": {
                "type": "Directory",
                "outputSource": "aggregate/output_model",
            }
        },
        "steps": {
            "merge": {
                "run": {
                    "class": "ExpressionTool",
                    "inputs": {},
                    "outputs": {"models": "Directory[]"},
                    "expression": [],
                },
                "in": {},
                "out": ["models"],
            },
            "aggregate": {
                "run": "clt/aggregate.cwl",
                "in": {
                    "model_basename": "model_basename",
                    "round": "round",
                    "script": "script_aggregation",
                    "models": "merge/models",
                },
                "out": ["output_model"],
            },
        },
    }


def get_training_step() -> MutableMapping[str, Any]:
    """Get the CWL file standard content for a xFFL application training step

    :return: Dict template of a CWL xFFL training step
    :rtype: MutableMapping[str, Any]
    """
    return {
        "cwlVersion": "v1.2",
        "class": "CommandLineTool",
        "requirements": {
            "InlineJavascriptRequirement": {},
            "EnvVarRequirement": {
                "envDef": {
                    "CODE_FOLDER": "$(inputs.repository.path)",
                    "MODEL_FOLDER": "$(inputs.model.path)",
                    "DATASET_FOLDER": "$(inputs.dataset.path)",
                    "IMAGE": "$(inputs.image.path)",
                    "FACILITY": "$(inputs.facility)",
                }
            },
        },
        "arguments": [
            {
                "position": 1,
                "valueFrom": "$(inputs.script.path)/facilitator.sh",
            },
            {
                "position": 5,
                "valueFrom": "$(inputs.model_basename)-round$(inputs.round)",
                "prefix": "--output",
            },
        ],
        "inputs": {
            "script": {
                "type": "Directory",
            },
            "image": {
                "type": "File",
            },
            "facility": {
                "type": "string",
            },
            "model": {
                "type": "Directory",
            },
            "dataset": {
                "type": "Directory",
            },
            "repository": {
                "type": "Directory",
            },
            "train_samples": {
                "type": "int",
                "inputBinding": {"position": 3, "prefix": "--train"},
            },
            "test_samples": {
                "type": "int",
                "inputBinding": {"position": 3, "prefix": "--validation"},
            },
            "epochs": {
                "type": "int",
                "inputBinding": {"position": 4, "prefix": "--epochs"},
            },
            "seed": {
                "type": "int",
                "inputBinding": {"position": 4, "prefix": "--seed"},
                "default": 42,
            },
            "wandb": {
                "type": "string",
                "inputBinding": {"position": 4, "prefix": "--wandb"},
                "default": "offline",
            },
            "model_basename": {"type": "string"},
            "round": {"type": "int"},
        },
        "outputs": {
            "output_model": {
                "type": "Directory",
                "outputBinding": {
                    "glob": "$(inputs.model_basename)-round$(inputs.round)"
                },
            }
        },
    }


def get_workflow_step(name: str) -> MutableMapping[str, Any]:
    """Get the CWL file standard content for a xFFL application step

    :param name: Name of the facility on which the step will be executed
    :type name: str
    :return: Dict template of a CWL xFFL step
    :rtype: MutableMapping[str, Any]
    """
    return {
        f"training_on_{name}": {
            "run": "clt/training.cwl",
            "in": {
                # todo: add `wandb` and `seed` inputs
                "script": "script_train",
                "facility": f"facility_{name}",
                "train_samples": f"train_samples_{name}",
                "test_samples": f"test_samples_{name}",
                "repository": f"repository_{name}",
                "image": f"image_{name}",
                "dataset": f"dataset_{name}",
                "model": "model",
                "epochs": "epochs",
                "model_basename": "model_basename",
                "round": "round",
            },
            "out": ["output_model"],
        }
    }
