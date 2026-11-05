#!/bin/bash

set -euo pipefail

readonly tpic_scaling_local_id=${SLURM_LOCALID:-}
if [[ ! $tpic_scaling_local_id =~ ^[0-7]$ ]]; then
  echo "SLURM_LOCALID must be an integer from 0 through 7, got: ${tpic_scaling_local_id:-<unset>}" >&2
  exit 2
fi

export ROCR_VISIBLE_DEVICES=$tpic_scaling_local_id
exec "$@"
