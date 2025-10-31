#!/bin/bash -ex

# C3S specific environment variables
export XFFL_LOCAL_TMPDIR=${TMPDIR}

killall python > /dev/null 2>&1
source /etc/profile > /dev/null 2>&1
spack load python@3.12.1 cuda@12.3.2 cudnn@8.9.5.30-12
