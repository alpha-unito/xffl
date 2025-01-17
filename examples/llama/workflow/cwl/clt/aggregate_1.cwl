#!/usr/bin/env cwl-runner

cwlVersion: v1.2
class: CommandLineTool

$namespaces:
  cwltool: "http://commonwl.org/cwltool#"
  s: https://schema.org/

$schemas:
 - https://schema.org/version/latest/schemaorg-current-http.rdf

requirements:
  InlineJavascriptRequirement: {}

baseCommand: ["python"]
arguments:
  - position: 5
    valueFrom: "$(inputs.model_basename)-merged-round$(inputs.round)"
    prefix: -o

inputs:
  script:
    type: File
    inputBinding:
      position: 1
  input_model_a: 
    type: Directory
    inputBinding:
      position: 2
      prefix: -m
  model_basename: 
    type: string
  round: 
    type: int

outputs:
  output_model:
    type: Directory
    outputBinding:
      glob: "$(inputs.model_basename)-merged-round$(inputs.round)"