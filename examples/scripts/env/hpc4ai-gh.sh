#!/bin/bash

cd /beegfs/home/gmittone/xffl
source .venv-gh/bin/activate
spack load python@3.12.1 cuda@12.3.2 cudnn@8.9.5.30-12

export PYTHONUNBUFFERED=1

# killall python > /dev/null 2>&1
GLOO_SOCKET_IFNAME=$(ip --brief link | awk '{print $1}' | grep -E 'enx.*' | tr -d '\n')
source /etc/profile > /dev/null 2>&1
spack load /w4pv4j6
