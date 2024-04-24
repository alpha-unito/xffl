#!/usr/bin/env cwl-runner

cwlVersion: v1.2
class: Workflow

$namespaces:
  cwltool: "http://commonwl.org/cwltool#"
  s: https://schema.org/

$schemas:
 - https://schema.org/version/latest/schemaorg-current-http.rdf

s:author:
  - class: s:Person
    s:identifier: https://orcid.org/0000-0001-9290-2017
    s:email: mailto:iacopo.colonnelli@unito.it
    s:name: Iacopo Colonnelli
  - class: s:Person
    s:identifier: https://orcid.org/0000-0003-1144-3707
    s:email: mailto:robert.birke@unito.it
    s:name: Robert Birke
  - class: s:Person
    s:identifier: https://orcid.org/0009-0006-4862-7429
    s:email: mailto:giulio.malenza@unito.it
    s:name: Giulio Malenza
  - class: s:Person
    s:identifier: https://orcid.org/0000-0002-1887-6911
    s:email: mailto:gianluca.mittone@unito.it
    s:name: Gianluca Mittone
  - class: s:Person
    s:identifier: https://orcid.org/0009-0009-2600-613X
    s:email: mailto:alberto.mulone@unito.it
    s:name: Alberto Mulone
  - class: s:Person
    s:identifier: https://orcid.org/0000-0001-8788-0829
    s:email: mailto:marco.aldinucci@unito.it
    s:name: Marco Aldinucci


s:codeRepository: https://github.com/alpha-unito/xffl
s:dateCreated: "2024-04-20"
s:license: https://spdx.org/licenses/LGPL-3.0-only
s:programmingLanguage: Python

requirements:
  InlineJavascriptRequirement: {}
  SubworkflowFeatureRequirement: {}
  StepInputExpressionRequirement: {}


inputs:
  train_script_a: File
  train_script_b: File
  aggregate_script: File
  dataset_a: string
  dataset_b: string
  input_model: Directory
  tokenizer_a: Directory
  tokenizer_b: Directory
  dataset_path_a: Directory
  dataset_path_b: Directory
  rounds: int


outputs:
  output_model:
    type: Directory
    outputSource: loop/output_model


steps:
  loop:
    in:
      train_script_a: train_script_a
      train_script_b: train_script_b
      aggregate_script: aggregate_script
      input_model: input_model
      dataset_a: dataset_a
      dataset_b: dataset_b
      tokenizer_a: tokenizer_a
      tokenizer_b: tokenizer_b
      dataset_path_a: dataset_path_a
      dataset_path_b: dataset_path_b
      round:
        default: 0
      rounds: rounds
    out: 
      [ output_model ]
    requirements:
      cwltool:Loop:
        loopWhen: $(inputs.round < inputs.rounds)
        loop:
          round:
            valueFrom: $(inputs.round + 1)
          input_model: output_model
        outputMethod: last
    run: round.cwl