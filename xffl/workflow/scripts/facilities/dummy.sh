#!/bin/bash -ex

if [ ${XFFL_FACILITY} = "dummy" ]; then
  cp -r ${XFFL_MODEL_FOLDER} output # FIXME: hardcoded
  EXIT_CODE=$?
  exit $EXIT_CODE
fi