#!/bin/bash

# LUMI specific environment variables
export XFFL_LOCAL_TMPDIR="/tmp/"
module load cray-python/3.11.7

# MIOPEN needs some initialisation for the cache as the default location
# does not work on LUMI as Lustre does not provide the necessary features.
MIOPEN_USER_DB_PATH="/tmp/$(whoami)-miopen-cache-$SLURM_NODEID"
export MIOPEN_USER_DB_PATH
export MIOPEN_CUSTOM_CACHE_DIR=$MIOPEN_USER_DB_PATH
if [ "$SLURM_LOCALID" -eq 0 ] ; then
	rm -rf "$MIOPEN_USER_DB_PATH"
	mkdir -p "$MIOPEN_USER_DB_PATH"
fi

# Set interfaces to be used by RCCL.
# This is needed as otherwise RCCL tries to use a network interface it has
# no access to on LUMI.
export NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3
export NCCL_NET_GDR_LEVEL=3
