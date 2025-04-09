#!/bin/bash

module load cuda/12.1
# LEONARDO specific environment variables
export XFFL_LOCAL_TMPDIR=${TMPDIR}