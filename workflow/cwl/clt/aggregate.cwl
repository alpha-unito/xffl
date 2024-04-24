#!/usr/bin/env cwl-runner

cwlVersion: v1.2
class: CommandLineTool

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
  MultipleInputFeatureRequirement: {}
  ShellCommandRequirement: {}


arguments:
  - position: 1
    valueFrom: "python "
    shellQuote: false
  - position: 4
    valueFrom: "llama-2-7b-merged"
    prefix: -o
    shellQuote: false

inputs:
  aggregate_script:
    type: File
    inputBinding:
      position: 2
  input_models:
    type:
      type: array
      items: Directory
      inputBinding:
        prefix: "-m"
    inputBinding:
      position: 3
      shellQuote: false 
      



outputs:
  output_model:
    type: Directory
    outputBinding:
      glob: "llama-2-7b-merged"