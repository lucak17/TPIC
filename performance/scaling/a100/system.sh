# MeluXina GPU nodes: four NVIDIA A100 (40 GB each) per node, one MPI rank per
# GPU. Sourced by ../runScalingCommon.sh at submission and inside the job.

TASKS_PER_NODE=4
GPU_LABEL='GPUs'
# A node has 128 hardware threads (64 cores); each rank gets a quarter.
CPUS_PER_TASK=32
SYSTEM_QOS=default
SYSTEM_SBATCH_ARGS=(
  --partition=gpu
  --gpus-per-node=4
  --cpus-per-task="$CPUS_PER_TASK"
)

# Runs the given TPIC command line on every rank of the allocation. TPIC takes
# the GPU of each rank from its node-local rank, or the only GPU the launcher
# exposes to it, so no device wrapper is needed.
system_srun() {
  srun --kill-on-bad-exit=1 --cpus-per-task="$CPUS_PER_TASK" "$@"
}
