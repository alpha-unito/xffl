#!/bin/bash

. ${HOME}/.bashrc
if [ ! command -v module > /dev/null ]; then
    echo "module not found"
    exit 1
fi

module load Python/3.11.10-GCCcore-13.3.0 Apptainer/1.3.6-GCCcore-13.3.0
