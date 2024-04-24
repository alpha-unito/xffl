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
  SubworkflowFeatureRequirement: {}

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
  round: int
  rounds: int

outputs:
  output_model:
    type: Directory
    outputSource: aggregate/output_model

steps:
  train_dataset_a:
    run: clt/train.cwl
    in:
      train_script: train_script_a
      dataset: dataset_a
      dataset_path: dataset_path_a
      input_model: input_model
      tokenizer: tokenizer_a
    out:
      [ output_model ]


  train_dataset_b:
    run: clt/train.cwl    
    in:
      train_script: train_script_b
      dataset: dataset_b
      dataset_path: dataset_path_b
      input_model: input_model
      tokenizer: tokenizer_b
    out:
      [ output_model ]


  aggregate:
    run: clt/aggregate.cwl
    in:
      aggregate_script: aggregate_script
      input_models:
        source:
          - train_dataset_a/output_model
          - train_dataset_b/output_model
    out: 
      [ output_model ]
