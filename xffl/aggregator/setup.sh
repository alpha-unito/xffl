#!/bin/bash

echo "*** xFFL - Aggregator setup ***"
echo "Assuming the local directory as working directory"

echo "Singularity container creation and start..."
singularity build aggregator.sif ../aggregator.def

echo "*** xFFL - Aggregator setup - Done! ***"
