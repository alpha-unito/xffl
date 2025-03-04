"""Collection of CWL-related template code"""

from abc import ABC, abstractmethod
from collections.abc import MutableMapping
from logging import Logger, getLogger
from typing import Any

import cwl_utils.parser.cwl_v1_2 as cwl

from xffl.workflow.config import YamlConfig

logger: Logger = getLogger(__name__)
"""Default xFFL logger"""


class CWLConfig(YamlConfig):
    """Class modelling a CWL configuration file as a YamlConfig object"""

    def __init__(self):
        """Creates a new CWLConfig instance with empty facilities data"""
        super().__init__()

    @classmethod
    def get_default_content(cls) -> MutableMapping[str, Any]:
        """Returns a basic CWL configuration to be updated and saved into a .yml

        :return: Basic CWL configuration for xFFL
        :rtype: MutableMapping[str, Any]
        """
        return {
            "script_train": {"class": "Directory", "path": "scripts"},
        }

    def add_inputs(
        self, facility_name: str, extra_inputs: MutableMapping[str, Any]
    ) -> None:
        """Adds the CWL inputs to the YAML content

        :param facility_name: Facility's name
        :param extra_inputs: Extra inputs
        :type facility_name: str
        :type extra_inputs: MutableMapping[str, Any]
        """
        self.content |= {f"facility_{facility_name}": facility_name} | {
            f"{name}_{facility_name}": value for name, value in extra_inputs.items()
        }


class Workflow(ABC):
    """Abstract base class describing workflow-like objects"""

    def __init__(self):
        """Creates a new Workflow instance with empty cwl data"""
        self.cwl: cwl.Process = type(self).get_default_definition()

    @classmethod
    def get_default_definition(cls) -> cwl.Process:
        """Returns the default workflow definition

        :return: Default workflow definition
        :rtype: cwl.Process()
        """
        return cwl.Process()

    def save(self) -> MutableMapping[str, Any]:
        """Returns the current CWL Workflow definition

        :return: Current CWL Workflow definition
        :rtype: MutableMapping[str, Any]
        """
        return self.cwl.save(top=True)

    @abstractmethod
    def add_inputs(
        self, facility_name: str, extra_inputs: MutableMapping[str, Any]
    ) -> None:
        """Add the given extra inputs to the Workflow definition

        :param facility_name: Facility's name
        :type facility_name: str
        :param extra_inputs: Command line argument required by the executable script
        :type extra_inputs: MutableMapping[str, str]
        """
        ...


class AggregateStep(Workflow):
    """Workflow describing the Federated Learning aggregation step"""

    @classmethod
    def get_default_definition(cls) -> cwl.Process:
        """Get the CWL file standard content for a xFFL application aggregation step

        :return: CWL object instance of the xFFL aggregation step
        :rtype: cwl.Process()
        """
        return cwl.CommandLineTool(
            arguments=[
                cwl.CommandLineBinding(
                    position=3,
                    prefix="--outname",
                    valueFrom="$(inputs.model_basename)-merged-round$(inputs.round)",
                )
            ],
            baseCommand=["python"],
            cwlVersion="v1.2",
            id="aggregate",
            inputs=[
                cwl.CommandInputParameter(
                    id="script",
                    type_="File",
                    inputBinding=cwl.CommandLineBinding(position=1),
                ),
                cwl.CommandInputParameter(
                    id="models",
                    type_=cwl.CommandInputArraySchema(
                        name="models",
                        type_="array",
                        items="Directory",
                        inputBinding=cwl.CommandLineBinding(
                            prefix="--model", position=2
                        ),
                    ),
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
                    id="new_model",
                    type_="Directory",
                    outputBinding=cwl.CommandOutputBinding(
                        glob="$(inputs.model_basename)-merged-round$(inputs.round)"
                    ),
                )
            ],
            requirements=[
                cwl.InlineJavascriptRequirement(),
            ],
        )

    def add_inputs(
        self, facility_name: str, extra_inputs: MutableMapping[str, Any]
    ) -> None:
        """Add the given extra inputs to the AggregateStep definition

        :param facility_name: Facility's name
        :type facility_name: str
        :param extra_inputs: Command line argument required by the executable script
        :type extra_inputs: MutableMapping[str, str]
        """
        pass


class MainWorkflow(Workflow):
    """Main description of a Federated Learning workflow"""

    @classmethod
    def get_default_definition(cls) -> cwl.Process:
        """Get the CWL main file standard content for a xFFL application

        :return: CWL object instance of the xFFL main workflow
        :rtype: cwl.Process
        """

        loadingOptions = cwl.LoadingOptions(
            namespaces={"cwltool": "http://commonwl.org/cwltool#"}
        )
        return cwl.Workflow(
            cwlVersion="v1.2",
            id="main",
            inputs=[
                cwl.WorkflowInputParameter(id="script_train", type_="Directory"),
                cwl.WorkflowInputParameter(id="script_aggregation", type_="File"),
                cwl.WorkflowInputParameter(id="model", type_="Directory"),
                cwl.WorkflowInputParameter(id="executable", type_="File"),
                cwl.WorkflowInputParameter(id="model_basename", type_="string"),
                cwl.WorkflowInputParameter(id="max_rounds", type_="int"),
            ],
            loadingOptions=loadingOptions,
            outputs=[
                cwl.WorkflowOutputParameter(
                    id="new_model",
                    type_="Directory",
                    outputSource="iteration/new_model",
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
                            id="executable",
                            source="executable",
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
                    out=[cwl.WorkflowStepOutput(id="new_model")],
                    requirements=[
                        cwl.Loop(
                            loadingOptions=loadingOptions,
                            loop=[
                                cwl.LoopInput(id="model", loopSource="new_model"),
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
        )

    def add_inputs(
        self, facility_name: str, extra_inputs: MutableMapping[str, Any]
    ) -> None:
        """Add the given extra inputs to the MainWorkflow definition

        :param facility_name: Facility's name
        :type facility_name: str
        :param extra_inputs: Command line argument required by the executable script [name: cwl type]
        :type extra_inputs: MutableMapping[str, str]
        """
        mandatory_inputs = {
            "image": "File",
            "facility": "string",
            "dataset": "Directory",
        }
        for key in mandatory_inputs.keys():
            if key in extra_inputs.keys():
                logger.warning(
                    f"{type(self).__name__}: The {key} input will be override"
                )
        inputs = dict(extra_inputs) | mandatory_inputs

        self.cwl.inputs.extend(
            [
                cwl.WorkflowInputParameter(id=f"{name}_{facility_name}", type_=_type)
                for name, _type in inputs.items()
            ]
        )
        iteration_step = next(elem for elem in self.cwl.steps if elem.id == "iteration")
        iteration_step.in_.extend(
            cwl.WorkflowStepInput(
                id=f"{name}_{facility_name}",
                source=f"{name}_{facility_name}",
            )
            for name, _type in inputs.items()
        )


class RoundWorkflow(Workflow):
    """Round workflow CWL description"""

    @classmethod
    def get_default_definition(cls) -> cwl.Process:
        """Get the CWL round file standard content for a xFFL application

        :return: round.cwl template
        :rtype: cwl.Process
        """
        return cwl.Workflow(
            cwlVersion="v1.2",
            id="round",
            inputs=[
                cwl.WorkflowInputParameter(id="script_train", type_="Directory"),
                cwl.WorkflowInputParameter(id="script_aggregation", type_="File"),
                cwl.WorkflowInputParameter(id="model", type_="Directory"),
                cwl.WorkflowInputParameter(id="executable", type_="File"),
                cwl.WorkflowInputParameter(id="model_basename", type_="string"),
                cwl.WorkflowInputParameter(id="max_rounds", type_="int"),
                cwl.WorkflowInputParameter(id="round", type_="int"),
            ],
            outputs=[
                cwl.WorkflowOutputParameter(
                    id="new_model",
                    type_="Directory",
                    outputSource="aggregate/new_model",
                )
            ],
            steps=[
                cwl.WorkflowStep(
                    id="merge",
                    in_=[],
                    out=[cwl.WorkflowStepOutput(id="models")],
                    run=cwl.ExpressionTool(
                        id="merge",
                        inputs=[],
                        outputs=[
                            cwl.ExpressionToolOutputParameter(
                                id="models",
                                type_=cwl.OutputArraySchema(
                                    name="models",
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
                    out=[cwl.WorkflowStepOutput(id="new_model")],
                    run="clt/aggregate.cwl",
                ),
            ],
        )

    @classmethod
    def get_training_step(cls, name: str) -> cwl.Process:
        """Get the CWL file standard content for a xFFL application step

        :param name: Name of the facility on which the step will be executed
        :type name: str
        :return: Dict template of a CWL xFFL step
        :rtype: cwl.Process
        """
        return cwl.WorkflowStep(
            id=f"training_on_{name}",
            in_=[
                cwl.WorkflowStepInput(
                    id="executable",
                    source="executable",
                ),
                cwl.WorkflowStepInput(
                    id="model",
                    source="model",
                ),
                cwl.WorkflowStepInput(
                    id="script",
                    source="script_train",
                ),
                cwl.WorkflowStepInput(
                    id="facility",
                    source=f"facility_{name}",
                ),
                cwl.WorkflowStepInput(
                    id="image",
                    source=f"image_{name}",
                ),
                cwl.WorkflowStepInput(
                    id="dataset",
                    source=f"dataset_{name}",
                ),
                cwl.WorkflowStepInput(
                    id="model_basename",
                    source="model_basename",
                ),
                cwl.WorkflowStepInput(
                    id="round",
                    source="round",
                ),
            ],
            out=[cwl.WorkflowStepOutput(id="new_model")],
            run="clt/training.cwl",
        )

    def add_inputs(
        self, facility_name: str, extra_inputs: MutableMapping[str, Any]
    ) -> None:
        """Add the given extra inputs to the RoundWorkflow definition

        :param facility_name: Facility's name
        :type facility_name: str
        :param extra_inputs: Command line argument required by the executable script [name: cwl type]
        :type extra_inputs: MutableMapping[str, str]
        """
        mandatory_inputs = {
            "image": "File",
            "facility": "string",
            "dataset": "Directory",
        }
        for key in mandatory_inputs.keys():
            if key in extra_inputs.keys():
                logger.warning(
                    f"{type(self).__name__}: The {key} input will be override"
                )
        inputs = dict(extra_inputs) | mandatory_inputs

        self.cwl.inputs.extend(
            [
                cwl.WorkflowInputParameter(id=f"{name}_{facility_name}", type_=_type)
                for name, _type in inputs.items()
            ]
        )

        training_step = RoundWorkflow.get_training_step(facility_name)
        training_step.in_.extend(
            [
                cwl.WorkflowStepInput(id=name, source=f"{name}_{facility_name}")
                for name in extra_inputs.keys()
            ]
        )
        self.cwl.steps.append(training_step)
        merge_step = next(elem for elem in self.cwl.steps if elem.id == "merge")
        merge_step.in_.append(
            cwl.WorkflowStepInput(
                id=facility_name, source=f"training_on_{facility_name}/new_model"
            )
        )
        merge_step.run.inputs.append(
            cwl.WorkflowInputParameter(id=facility_name, type_="Directory")
        )
        merge_step.run.expression.append(f"inputs.{facility_name}")

    def update_merge_step(self) -> None:
        """Updates the merge step"""
        merge_step = next(elem for elem in self.cwl.steps if elem.id == "merge")
        merge_step.run.expression = (
            "$({'models': [" + ",".join(merge_step.run.expression) + "] })"
        )


class TrainingStep(Workflow):
    """Workflow modelling a training step"""

    def __init__(self):
        super().__init__()

    @classmethod
    def get_default_definition(cls) -> cwl.Process:
        """Get the CWL file standard content for a xFFL application training step

        :return: CWL object instance of a xFFL training step
        :rtype: cwl.Process
        """
        return cwl.CommandLineTool(
            arguments=[
                cwl.CommandLineBinding(
                    position=1,
                    valueFrom="$(inputs.script.path)/facilitator.sh",
                ),
                cwl.CommandLineBinding(
                    position=5,
                    prefix="--output-model",
                    valueFrom="$(inputs.model_basename)",
                ),
                cwl.CommandLineBinding(
                    position=6,
                    prefix="--output",
                    valueFrom="$(runtime.outdir)",
                ),
                cwl.CommandLineBinding(
                    position=7,
                    prefix="--workspace",
                    valueFrom="$XFFL_TMPDIR_FOLDER",
                ),
            ],
            cwlVersion="v1.2",
            id="training",
            inputs=[
                cwl.CommandInputParameter(
                    id="script",
                    type_="Directory",
                ),
                cwl.CommandInputParameter(
                    id="executable",
                    type_="File",
                    inputBinding=cwl.CommandLineBinding(
                        position=2,
                    ),
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
                    inputBinding=cwl.CommandLineBinding(position=3, prefix="--model"),
                ),
                cwl.CommandInputParameter(
                    id="dataset",
                    type_="Directory",
                    inputBinding=cwl.CommandLineBinding(position=4, prefix="--dataset"),
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
                    id="new_model",
                    type_="Directory",
                    outputBinding=cwl.CommandOutputBinding(
                        glob="$(inputs.model_basename)"
                    ),
                )
            ],
            requirements=[
                cwl.InlineJavascriptRequirement(),
                cwl.EnvVarRequirement(
                    envDef=[
                        cwl.EnvironmentDef(
                            envName="XFFL_MODEL_FOLDER",
                            envValue="$(inputs.model.path)",
                        ),
                        cwl.EnvironmentDef(
                            envName="XFFL_DATASET_FOLDER",
                            envValue="$(inputs.dataset.path)",
                        ),
                        cwl.EnvironmentDef(
                            envName="XFFL_IMAGE",
                            envValue="$(inputs.image.path)",
                        ),
                        cwl.EnvironmentDef(
                            envName="XFFL_FACILITY",
                            envValue="$(inputs.facility)",
                        ),
                        cwl.EnvironmentDef(
                            envName="XFFL_OUTPUT_FOLDER",
                            envValue="$(runtime.outdir)",
                        ),
                        cwl.EnvironmentDef(
                            envName="XFFL_TMPDIR_FOLDER",
                            envValue="/tmp",
                            # "$(inputs.workspace == null ? runtime.tmpdir : inputs.workspace)",
                            # TODO: an error is raised from torch when the tmp path is too long
                        ),
                    ]
                ),
            ],
        )

    def add_inputs(
        self, facility_name: str | None, extra_inputs: MutableMapping[str, Any]
    ) -> None:
        """Add the given extra inputs to the TrainingStep definition

        :param facility_name: Facility's name
        :type facility_name: str
        :param extra_inputs: Command line argument required by the executable script [name: cwl type]
        :type extra_inputs: MutableMapping[str, str]
        """
        i = self.get_available_position()
        for name, values in extra_inputs.items():
            input_binding = None
            if "prefix" in values.keys():
                input_binding = cwl.CommandLineBinding(
                    position=i,
                    prefix=values["prefix"],
                )
                i += 1
            self.cwl.inputs.append(
                cwl.WorkflowInputParameter(
                    id=name,
                    type_=values["type"],
                    inputBinding=input_binding,
                    default=values.get("default", None),
                )
            )

    def get_available_position(self) -> int:
        position = 0
        for _input in self.cwl.inputs:
            if _input.inputBinding and position < (
                curr_pos := _input.inputBinding.position
            ):
                position = curr_pos
        for _input in self.cwl.arguments:
            if position < (curr_pos := _input.position):
                position = curr_pos
        return position + 1
