"""Collection of CWL-related template code
"""

from collections.abc import MutableMapping
from typing import Any

import cwl_utils.parser.cwl_v1_2 as cwl


def get_aggregate_step() -> MutableMapping[str, Any]:
    """Get the CWL file standard content for a xFFL application aggregation step

    :return: Dict template of a CWL xFFL aggregation step
    :rtype: MutableMapping[str, Any]
    """
    return cwl.CommandLineTool(
        arguments=[
            cwl.CommandLineBinding(
                position=5,
                prefix="--outname",
                valueFrom="$(inputs.model_basename)-merged-round$(inputs.round)",
            )
        ],
        baseCommand=["python"],
        cwlVersion="1.2",
        inputs=[
            cwl.CommandInputParameter(
                id="script",
                type_="File",
                inputBinding=cwl.CommandLineBinding(position=1),
            ),
            cwl.CommandInputParameter(
                id="models",
                type_=cwl.CommandInputArraySchema(
                    type_="array",
                    items="Directory",
                    inputBinding=cwl.CommandLineBinding(prefix="--model", position=2),
                ),
                inputBinding=cwl.CommandLineBinding(position=1),
            ),
            cwl.CommandInputParameter(
                id="model_basename",
                type_="string",
            ),
            cwl.CommandInputParameter(
                id="round",
                type_="int",
            ),
        ],
        outputs=[
            cwl.CommandOutputParameter(
                id="output_model",
                type_="Directory",
                outputBinding=cwl.CommandOutputBinding(
                    glob="$(inputs.model_basename)-merged-round$(inputs.round)"
                ),
            )
        ],
        requirements=[
            cwl.InlineJavascriptRequirement(),
        ],
    ).save()


def get_config() -> MutableMapping[str, Any]:
    return {
        "script_train": {"class": "Directory", "path": "scripts"},
    }


def get_main_cwl() -> MutableMapping[str, Any]:
    """Get the CWL main file standard content for a xFFL application

    :return: main.cwl template
    :rtype: MutableMapping[str, Any]
    """
    loadingOptions = cwl.LoadingOptions(
        namespaces={"cwltool": "http://commonwl.org/cwltool#"}
    )
    return cwl.Workflow(
        cwlVersion="v1.2",
        inputs=[
            cwl.WorkflowInputParameter(id="script_train", type_="Directory"),
            cwl.WorkflowInputParameter(id="script_aggregation", type_="File"),
            cwl.WorkflowInputParameter(id="model", type_="Directory"),
            cwl.WorkflowInputParameter(id="epochs", type_="int"),
            cwl.WorkflowInputParameter(id="model_basename", type_="string"),
            cwl.WorkflowInputParameter(id="max_rounds", type_="int"),
        ],
        loadingOptions=loadingOptions,
        outputs=[
            cwl.WorkflowOutputParameter(
                id="output_model",
                type_="Directory",
                outputSource="iteration/output_model",
            )
        ],
        steps=[
            cwl.WorkflowStep(
                id="iteration",
                in_=[
                    cwl.WorkflowStepInput(
                        id="script_train",
                        source="script_train",
                    ),
                    cwl.WorkflowStepInput(
                        id="script_aggregation",
                        source="script_aggregation",
                    ),
                    cwl.WorkflowStepInput(
                        id="model",
                        source="model",
                    ),
                    cwl.WorkflowStepInput(
                        id="epochs",
                        source="epochs",
                    ),
                    cwl.WorkflowStepInput(
                        id="model_basename",
                        source="model_basename",
                    ),
                    cwl.WorkflowStepInput(
                        default=0,
                        id="round",
                    ),
                    cwl.WorkflowStepInput(
                        id="max_rounds",
                        source="max_rounds",
                    ),
                ],
                out=[cwl.WorkflowStepOutput(id="output_model")],
                requirements=[
                    cwl.Loop(
                        loadingOptions=loadingOptions,
                        loop=[
                            cwl.LoopInput(id="model", loopSource="output_model"),
                            cwl.LoopInput(
                                id="round",
                                loopSource="round",
                                valueFrom="$(inputs.round + 1)",
                            ),
                        ],
                        loopWhen="$(inputs.round < inputs.max_rounds)",
                        outputMethod="last",
                    )
                ],
                run="round.cwl",
            ),
        ],
        requirements=[
            cwl.InlineJavascriptRequirement(),
            cwl.StepInputExpressionRequirement(),
            cwl.SubworkflowFeatureRequirement(),
        ],
    ).save(top=True)


def get_round_cwl() -> MutableMapping[str, Any]:
    """Get the CWL round file standard content for a xFFL application

    :return: round.cwl template
    :rtype: MutableMapping[str, Any]
    """
    return cwl.Workflow(
        cwlVersion="v1.2",
        inputs=[
            cwl.WorkflowInputParameter(id="script_train", type_="Directory"),
            cwl.WorkflowInputParameter(id="script_aggregation", type_="File"),
            cwl.WorkflowInputParameter(id="model", type_="Directory"),
            cwl.WorkflowInputParameter(id="epochs", type_="int"),
            cwl.WorkflowInputParameter(id="model_basename", type_="string"),
            cwl.WorkflowInputParameter(id="max_rounds", type_="int"),
            cwl.WorkflowInputParameter(id="round", type_="int"),
        ],
        outputs=[
            cwl.WorkflowOutputParameter(
                id="output_model",
                type_="Directory",
                outputSource="aggregate/output_model",
            )
        ],
        steps=[
            cwl.WorkflowStep(
                id="merge",
                in_=[],
                out=[cwl.WorkflowStepOutput(id="models")],
                run=cwl.ExpressionTool(
                    inputs=[],
                    outputs=[
                        cwl.ExpressionToolOutputParameter(
                            id="models",
                            type_=cwl.OutputArraySchema(
                                type_="array",
                                items="Directory",
                            ),
                        )
                    ],
                    expression=[],
                ),
            ),
            cwl.WorkflowStep(
                id="aggregate",
                in_=[
                    cwl.WorkflowStepInput(
                        id="model_basename",
                        source="model_basename",
                    ),
                    cwl.WorkflowStepInput(
                        id="round",
                        source="round",
                    ),
                    cwl.WorkflowStepInput(
                        id="script",
                        source="script_aggregation",
                    ),
                    cwl.WorkflowStepInput(
                        id="models",
                        source="merge/models",
                    ),
                ],
                out=[cwl.WorkflowStepOutput(id="output_model")],
                run="clt/aggregate.cwl",
            ),
        ],
    ).save()


def get_training_step() -> MutableMapping[str, Any]:
    """Get the CWL file standard content for a xFFL application training step

    :return: Dict template of a CWL xFFL training step
    :rtype: MutableMapping[str, Any]
    """
    return cwl.CommandLineTool(
        arguments=[
            cwl.CommandLineBinding(
                position=1,
                valueFrom="$(inputs.script.path)/facilitator.sh",
            ),
            cwl.CommandLineBinding(
                position=5,
                prefix="--output",
                valueFrom="$(inputs.model_basename)-round$(inputs.round)",
            ),
        ],
        cwlVersion="v1.2",
        inputs=[
            cwl.CommandInputParameter(
                id="script",
                type_="Directory",
            ),
            cwl.CommandInputParameter(
                id="image",
                type_="File",
            ),
            cwl.CommandInputParameter(
                id="facility",
                type_="string",
            ),
            cwl.CommandInputParameter(
                id="model",
                type_="Directory",
            ),
            cwl.CommandInputParameter(
                id="dataset",
                type_="Directory",
            ),
            cwl.CommandInputParameter(
                id="repository",
                type_="Directory",
            ),
            cwl.CommandInputParameter(
                id="train_samples",
                type_="int",
                inputBinding=cwl.CommandLineBinding(
                    position=3,
                    prefix="--train",
                ),
            ),
            cwl.CommandInputParameter(
                id="test_samples",
                type_="int",
                inputBinding=cwl.CommandLineBinding(
                    position=3,
                    prefix="--validation",
                ),
            ),
            cwl.CommandInputParameter(
                id="epochs",
                type_="int",
                inputBinding=cwl.CommandLineBinding(
                    position=3,
                    prefix="--epochs",
                ),
            ),
            cwl.CommandInputParameter(
                default=42,
                id="seed",
                type_="int",
                inputBinding=cwl.CommandLineBinding(
                    position=4,
                    prefix="--seed",
                ),
            ),
            cwl.CommandInputParameter(
                default="offline",
                id="wandb",
                type_="string",
                inputBinding=cwl.CommandLineBinding(
                    position=4,
                    prefix="--wandb",
                ),
            ),
            cwl.CommandInputParameter(
                id="gpu_per_nodes",
                type_="int",
            ),
            cwl.CommandInputParameter(
                id="model_basename",
                type_="string",
            ),
            cwl.CommandInputParameter(
                id="round",
                type_="int",
            ),
        ],
        outputs=[
            cwl.CommandOutputParameter(
                id="output_model",
                type_="Directory",
                outputBinding=cwl.CommandOutputBinding(
                    glob="$(inputs.model_basename)-round$(inputs.round)"
                ),
            )
        ],
        requirements=[
            cwl.InlineJavascriptRequirement(),
            cwl.EnvVarRequirement(
                envDef=[
                    cwl.EnvironmentDef(
                        envName="CODE_FOLDER",
                        envValue="$(inputs.repository.path)",
                    ),
                    cwl.EnvironmentDef(
                        envName="MODEL_FOLDER",
                        envValue="$(inputs.model.path)",
                    ),
                    cwl.EnvironmentDef(
                        envName="DATASET_FOLDER",
                        envValue="$(inputs.dataset.path)",
                    ),
                    cwl.EnvironmentDef(
                        envName="IMAGE",
                        envValue="$(inputs.image.path)",
                    ),
                    cwl.EnvironmentDef(
                        envName="FACILITY",
                        envValue="$(inputs.facility)",
                    ),
                    cwl.EnvironmentDef(
                        envName="INSTANCES",
                        envValue="$(inputs.gpu_per_nodes)",
                    ),
                ]
            ),
        ],
    ).save()


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
                "gpu_per_nodes": f"gpu_per_nodes_{name}",
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
