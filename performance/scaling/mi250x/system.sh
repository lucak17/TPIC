# LUMI-G: eight AMD MI250X GCDs (64 GB each) per node, one MPI rank per GCD.
# Sourced by ../runScalingCommon.sh at submission and inside the job.

TASKS_PER_NODE=8
GPU_LABEL='GPUs/GCDs'
SYSTEM_SBATCH_ARGS=(--partition=standard-g --gpus-per-node=8)

if [[ ! -x ${SYSTEM_DIR}/selectGPU.sh ]]; then
  echo "GPU binding wrapper is not executable: ${SYSTEM_DIR}/selectGPU.sh" >&2
  exit 1
fi

# Runs the given TPIC command line on every rank of the allocation. The masks
# give local rank i six cores of the NUMA domain attached to GCD i, the GCD
# that selectGPU.sh exposes to that rank.
system_srun() {
  export MPICH_GPU_SUPPORT_ENABLED=1
  export MPICH_MAX_THREAD_SAFETY=multiple
  # Holds every exchange buffer mapping of a run; the default of 50 evicts them and a
  # peer GCD then faults on an unmapped address (docs/user/support.md).
  export MPICH_GPU_IPC_CACHE_MAX_SIZE=1024

  local cpu_bind="mask_cpu:7e000000000000,7e00000000000000"
  cpu_bind="${cpu_bind},7e0000,7e000000"
  cpu_bind="${cpu_bind},7e,7e00"
  cpu_bind="${cpu_bind},7e00000000,7e0000000000"

  srun --kill-on-bad-exit=1 --cpu-bind="$cpu_bind" \
    "${SYSTEM_DIR}/selectGPU.sh" "$@"
}
