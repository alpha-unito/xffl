#!/bin/bash -ex

# KAROLINA specific environment variables
export CONTAINER="apptainer"
export XFFL_LOCAL_TMPDIR="/tmp/"
export APPTAINER_BINDPATH=$SINGULARITY_BINDPATH
export APPTAINERENV_LD_PRELOAD=$SINGULARITYENV_LD_PRELOAD