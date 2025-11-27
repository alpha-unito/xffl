#!/bin/bash -ex

# C3S specific environment variables
export XFFL_LOCAL_TMPDIR=${TMPDIR}
export PYTHONUNBUFFERED=1

# killall python > /dev/null 2>&1
GLOO_SOCKET_IFNAME=$(ip --brief link | awk '{print $1}' | grep -E 'enx.*' | tr -d '\n')
source /etc/profile > /dev/null 2>&1
spack load /w4pv4j6
