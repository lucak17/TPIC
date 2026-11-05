#!/bin/bash -l
# Weak-scaling entry point. Shared logic lives in ../../runScalingCommon.sh;
# the Slurm resources of this system come from ../system.sh.

set -euo pipefail

LAUNCHER_PATH=${BASH_SOURCE[0]}
if [[ ${1:-} == --worker ]]; then
  if [[ -z ${TPIC_SCALING_LAUNCHER_PATH:-} ]]; then
    echo "Internal error: TPIC_SCALING_LAUNCHER_PATH is not set" >&2
    exit 2
  fi
  LAUNCHER_PATH=$TPIC_SCALING_LAUNCHER_PATH
fi

SCRIPT_DIR=$(cd -- "$(dirname -- "$LAUNCHER_PATH")" && pwd -P)
exec bash "${SCRIPT_DIR}/../../runScalingCommon.sh" \
  weak "${SCRIPT_DIR}/$(basename -- "$LAUNCHER_PATH")" "$@"
