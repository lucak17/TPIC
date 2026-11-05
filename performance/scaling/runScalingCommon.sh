#!/bin/bash

# Shared TPIC strong/weak scaling launcher and scheduled-job worker. Called by
# <system>/<kind>/runTPIC*.slurm; <system>/system.sh supplies the ranks per
# node, the Slurm resources and the srun command of that system.

set -euo pipefail

if (( $# < 2 )); then
  echo "Internal error: scaling kind and launcher path are required" >&2
  exit 2
fi

SCALING_KIND=$1
LAUNCHER_PATH=$2
shift 2

case $SCALING_KIND in
strong)
  INPUT_PREFIX=GEMstrong
  DEFAULT_TIME_LIMIT=00:30:00
  ;;
weak)
  INPUT_PREFIX=GEMweak
  DEFAULT_TIME_LIMIT=00:20:00
  ;;
*)
  echo "Internal error: unknown scaling kind: ${SCALING_KIND}" >&2
  exit 2
  ;;
esac

INPUT_DIR=$(cd -- "$(dirname -- "$LAUNCHER_PATH")" && pwd -P)
SYSTEM_DIR=$(cd -- "${INPUT_DIR}/.." && pwd -P)
SCALING_DIR=$(cd -- "${SYSTEM_DIR}/.." && pwd -P)
SYSTEM=$(basename -- "$SYSTEM_DIR")
# Defines TASKS_PER_NODE, GPU_LABEL, SYSTEM_SBATCH_ARGS and system_srun, and
# SYSTEM_QOS where the system needs a Slurm QOS.
source "${SYSTEM_DIR}/system.sh"
SOLVER=ecsim
PARTICLE_ORDER=partial
PRECISION=double
MASS_MATRIX=matrix-core
MASS_APPLY=split

usage() {
  cat <<EOF
Usage: ${LAUNCHER_PATH} [OPTIONS] (--all | MIN_NODES [MAX_NODES])

Required environment:
  TPIC_SLURM_ACCOUNT  Slurm billing account
  TPIC_PROFILE        Readable runtime profile sourced on each compute node
  TPIC_EXECUTABLE     TPIC executable named tpic

Options:
  --all                 Select every ${INPUT_PREFIX}N*.inp file
  --dry-run             Validate and print commands without creating/submitting
  --exclude, -x LIST    Slurm nodes to exclude
  --qos NAME            Slurm QOS (default: ${SYSTEM_QOS:-none})
  --time LIMIT          Override the default Slurm time limit (${DEFAULT_TIME_LIMIT})
  -h, --help            Show this help

A single node count selects exactly that input. Two counts select the
available inputs in the inclusive range. No job is submitted without an
explicit node selection or --all.

Each job runs TPIC ECSIM with ${TASKS_PER_NODE} MPI ranks per node, partial particle
ordering, double precision, the matrix-core mass-matrix kernel, and split
Maxwell mass application.
EOF
}

input_value() {
  local input_file=$1
  local wanted_key=$2
  awk -F= -v wanted="$wanted_key" '
    function trim(value) {
      sub(/^[[:space:]]+/, "", value)
      sub(/[[:space:]]+$/, "", value)
      return value
    }
    {
      line = $0
      sub(/#.*/, "", line)
      delimiter = index(line, "=")
      if (delimiter == 0) next
      key = trim(substr(line, 1, delimiter - 1))
      if (key == wanted) {
        value = trim(substr(line, delimiter + 1))
        found = 1
      }
    }
    END {
      if (!found) exit 1
      print value
    }
  ' "$input_file"
}

normalized_invariants() {
  local input_file=$1
  awk -F= -v kind="$SCALING_KIND" '
    function trim(value) {
      sub(/^[[:space:]]+/, "", value)
      sub(/[[:space:]]+$/, "", value)
      return value
    }
    {
      line = $0
      sub(/#.*/, "", line)
      delimiter = index(line, "=")
      if (delimiter == 0) next
      key = trim(substr(line, 1, delimiter - 1))
      value = trim(substr(line, delimiter + 1))
      gsub(/[[:space:]]+/, " ", value)

      if (key == "XLEN" || key == "YLEN" || key == "ZLEN") next
      if (kind == "weak" &&
          (key == "Lx" || key == "Ly" || key == "Lz" ||
           key == "nxc" || key == "nyc" || key == "nzc")) next
      print key "=" value
    }
  ' "$input_file"
}

require_nonnegative_integer() {
  local value=$1
  local label=$2
  if [[ ! $value =~ ^[0-9]+$ ]]; then
    echo "${label} must be a non-negative integer, got: ${value}" >&2
    return 1
  fi
}

require_finite_number() {
  local value=$1
  local label=$2
  if ! awk -v value="$value" '
      BEGIN {
        number = "^[+-]?([0-9]+([.][0-9]*)?|[.][0-9]+)([eE][+-]?[0-9]+)?$"
        if (value !~ number) exit 1
        rendered = tolower(sprintf("%.17g", value + 0.0))
        if (rendered ~ /nan|inf/) exit 1
      }
    '; then
    echo "${label} must be a finite number, got: ${value}" >&2
    return 1
  fi
}

validate_input() {
  local input_file=$1
  local nodes=$2
  local case_name b0x harris_b0x smoothing smoothing_operator
  local field_cycle particle_cycle restart_cycle diagnostics_cycle
  local ncycles nxc nyc nzc xlen ylen zlen lx ly lz

  case_name=$(input_value "$input_file" Case) || {
    echo "$(basename -- "$input_file"): missing Case" >&2
    return 1
  }
  b0x=$(input_value "$input_file" B0x) || {
    echo "$(basename -- "$input_file"): missing B0x" >&2
    return 1
  }
  harris_b0x=$(input_value "$input_file" HarrisB0x) || {
    echo "$(basename -- "$input_file"): missing HarrisB0x" >&2
    return 1
  }
  smoothing=$(input_value "$input_file" ECSimSmoothing) || {
    echo "$(basename -- "$input_file"): missing ECSimSmoothing" >&2
    return 1
  }
  smoothing_operator=$(input_value "$input_file" ECSimSmoothingOperator) || {
    echo "$(basename -- "$input_file"): missing ECSimSmoothingOperator" >&2
    return 1
  }

  if [[ $case_name != GEM ]]; then
    echo "$(basename -- "$input_file"): expected Case=GEM, got ${case_name}" >&2
    return 1
  fi
  require_finite_number "$b0x" "$(basename -- "$input_file"): B0x" || return 1
  require_finite_number "$harris_b0x" "$(basename -- "$input_file"): HarrisB0x" || return 1
  if ! awk -v background="$b0x" -v sheet="$harris_b0x" \
      'BEGIN { exit !((background + 0.0) == 0.0 && (sheet + 0.0) != 0.0) }'; then
    echo "$(basename -- "$input_file"): expected B0x=0 and nonzero HarrisB0x" >&2
    return 1
  fi
  case ${smoothing} in
  none|energy_conserving) ;;
  *)
    echo "$(basename -- "$input_file"): invalid ECSimSmoothing value: ${smoothing}" >&2
    return 1
    ;;
  esac
  case ${smoothing_operator} in
  face7|binomial27) ;;
  *)
    echo "$(basename -- "$input_file"): invalid ECSimSmoothingOperator value: ${smoothing_operator}" >&2
    return 1
    ;;
  esac

  ncycles=$(input_value "$input_file" ncycles) || return 1
  nxc=$(input_value "$input_file" nxc) || return 1
  nyc=$(input_value "$input_file" nyc) || return 1
  nzc=$(input_value "$input_file" nzc) || return 1
  xlen=$(input_value "$input_file" XLEN) || return 1
  ylen=$(input_value "$input_file" YLEN) || return 1
  zlen=$(input_value "$input_file" ZLEN) || return 1
  lx=$(input_value "$input_file" Lx) || return 1
  ly=$(input_value "$input_file" Ly) || return 1
  lz=$(input_value "$input_file" Lz) || return 1

  local name
  for name in ncycles nxc nyc nzc xlen ylen zlen; do
    require_nonnegative_integer "${!name}" "$(basename -- "$input_file"): ${name}" || return 1
  done
  for name in lx ly lz; do
    require_finite_number "${!name}" "$(basename -- "$input_file"): ${name}" || return 1
    if ! awk -v value="${!name}" 'BEGIN { exit !((value + 0.0) > 0.0) }'; then
      echo "$(basename -- "$input_file"): ${name} must be positive" >&2
      return 1
    fi
  done
  ncycles=$((10#$ncycles))
  nxc=$((10#$nxc))
  nyc=$((10#$nyc))
  nzc=$((10#$nzc))
  xlen=$((10#$xlen))
  ylen=$((10#$ylen))
  zlen=$((10#$zlen))

  if (( ncycles < 2 || xlen < 1 || ylen < 1 || zlen < 1 )); then
    echo "$(basename -- "$input_file"): invalid cycles or MPI topology" >&2
    return 1
  fi
  if (( xlen * ylen * zlen != nodes * TASKS_PER_NODE )); then
    echo "$(basename -- "$input_file"): topology has $((xlen * ylen * zlen)) ranks; expected $((nodes * TASKS_PER_NODE))" >&2
    return 1
  fi
  if (( nxc % xlen != 0 || nyc % ylen != 0 || nzc % zlen != 0 )); then
    echo "$(basename -- "$input_file"): grid must divide evenly over the MPI topology for scaling" >&2
    return 1
  fi
  if (( nxc / xlen < 2 || nyc / ylen < 2 || nzc / zlen < 2 )); then
    echo "$(basename -- "$input_file"): fewer than two local cells in a dimension" >&2
    return 1
  fi

  field_cycle=$(input_value "$input_file" FieldOutputCycle) || return 1
  particle_cycle=$(input_value "$input_file" ParticlesOutputCycle) || return 1
  restart_cycle=$(input_value "$input_file" RestartOutputCycle) || return 1
  diagnostics_cycle=$(input_value "$input_file" DiagnosticsOutputCycle) || return 1
  for name in field_cycle particle_cycle restart_cycle diagnostics_cycle; do
    require_nonnegative_integer "${!name}" "$(basename -- "$input_file"): ${name}" || return 1
    if (( 10#${!name} != 0 )); then
      echo "$(basename -- "$input_file"): ${name} must be zero for scaling" >&2
      return 1
    fi
  done
}

campaign_grid_signature() {
  local input_file=$1
  local nxc nyc nzc xlen ylen zlen lx ly lz

  nxc=$(input_value "$input_file" nxc) || return 1
  nyc=$(input_value "$input_file" nyc) || return 1
  nzc=$(input_value "$input_file" nzc) || return 1
  xlen=$(input_value "$input_file" XLEN) || return 1
  ylen=$(input_value "$input_file" YLEN) || return 1
  zlen=$(input_value "$input_file" ZLEN) || return 1
  lx=$(input_value "$input_file" Lx) || return 1
  ly=$(input_value "$input_file" Ly) || return 1
  lz=$(input_value "$input_file" Lz) || return 1

  if [[ $SCALING_KIND == strong ]]; then
    printf '%sx%sx%s:%sx%sx%s\n' "$nxc" "$nyc" "$nzc" "$lx" "$ly" "$lz"
  else
    awk -v nx="$nxc" -v ny="$nyc" -v nz="$nzc" \
            -v px="$xlen" -v py="$ylen" -v pz="$zlen" \
            -v x="$lx" -v y="$ly" -v z="$lz" \
      'BEGIN { printf "%d:%d:%d:%.17g:%.17g:%.17g\n", nx/px, ny/py, nz/pz, x/px, y/py, z/pz }'
  fi
}

validate_campaign_input() {
  local input_file=$1
  local nodes=$2
  local campaign_invariants=$3
  local campaign_grid=$4
  local filename
  filename=$(basename -- "$input_file")

  validate_input "$input_file" "$nodes" || return 1
  if [[ $(normalized_invariants "$input_file") != "$campaign_invariants" ]]; then
    echo "${filename}: non-scaling parameters differ from the campaign baseline" >&2
    return 1
  fi

  local grid
  grid=$(campaign_grid_signature "$input_file") || return 1
  if [[ $grid != "$campaign_grid" ]]; then
    if [[ $SCALING_KIND == strong ]]; then
      echo "${filename}: strong scaling requires a fixed global grid/domain" >&2
    else
      echo "${filename}: weak scaling requires fixed work and domain size per rank" >&2
    fi
    return 1
  fi
}

absolute_path() {
  local path=$1
  if [[ $path != /* ]]; then
    path=${PWD}/${path}
  fi
  local directory basename
  directory=$(cd -- "$(dirname -- "$path")" && pwd -P) || return 1
  basename=$(basename -- "$path")
  printf '%s/%s\n' "$directory" "$basename"
}

write_manifest_header() {
  local manifest=$1
  local source_executable=$2
  local snapshot_executable=$3
  local profile=$4
  local solver=$5
  local planned_nodes=$6
  local snapshot_common=$7
  local snapshot_system=$8
  local snapshot_launcher=$9
  local profile_sha256=${10}
  local cmake_cache cmake_home_directory
  cmake_cache=$(dirname -- "$source_executable")/CMakeCache.txt
  cmake_home_directory=""
  if [[ -r $cmake_cache ]]; then
    cmake_home_directory=$(awk -F= '
      $1 == "CMAKE_HOME_DIRECTORY:INTERNAL" {
        print substr($0, index($0, "=") + 1)
        exit
      }
    ' "$cmake_cache")
  fi

  {
    echo "format=TPIC-scaling-v1"
    echo "kind=${SCALING_KIND}"
    echo "system=${SYSTEM}"
    echo "solver=${solver}"
    echo "particle_order=${PARTICLE_ORDER}"
    echo "precision=${PRECISION}"
    echo "mass_matrix=${MASS_MATRIX}"
    echo "mass_apply=${MASS_APPLY}"
    echo "created_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "tasks_per_node=${TASKS_PER_NODE}"
    echo "gpu_label=${GPU_LABEL}"
    echo "source_executable=${source_executable}"
    echo "snapshot_executable=${snapshot_executable}"
    echo "profile=${profile}"
    echo "planned_nodes=${planned_nodes}"
    echo "submission_complete=false"
    if command -v sha256sum >/dev/null 2>&1; then
      echo "executable_sha256=$(sha256sum "$snapshot_executable" | awk '{print $1}')"
      echo "profile_sha256=${profile_sha256}"
      echo "run_scaling_common_sha256=$(sha256sum "$snapshot_common" | awk '{print $1}')"
      echo "system_sha256=$(cat -- "$snapshot_system"/*.sh | sha256sum | awk '{print $1}')"
      echo "launcher_sha256=$(sha256sum "$snapshot_launcher" | awk '{print $1}')"
    else
      echo "executable_sha256=unavailable"
      echo "profile_sha256=unavailable"
      echo "run_scaling_common_sha256=unavailable"
      echo "system_sha256=unavailable"
      echo "launcher_sha256=unavailable"
    fi
    if [[ -r $cmake_cache ]]; then
      echo "cmake_cache=${cmake_cache}"
    else
      echo "cmake_cache=unavailable"
    fi
    if [[ -n $cmake_home_directory ]]; then
      echo "cmake_home_directory=${cmake_home_directory}"
    else
      echo "cmake_home_directory=unavailable"
    fi
    if [[ -n $cmake_home_directory ]] && command -v git >/dev/null 2>&1 &&
        git -C "$cmake_home_directory" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
      echo "git_provenance=available"
      echo "git_commit=$(git -C "$cmake_home_directory" rev-parse HEAD)"
      if [[ -n $(git -C "$cmake_home_directory" status --porcelain) ]]; then
        echo "git_worktree_dirty=true"
      else
        echo "git_worktree_dirty=false"
      fi
    else
      echo "git_provenance=unavailable"
      echo "git_commit=unavailable"
      echo "git_worktree_dirty=unavailable"
    fi
    if [[ -r $cmake_cache ]]; then
      grep -E '^(CMAKE_BUILD_TYPE|TPIC_[A-Z0-9_]+|LPFX_[A-Z0-9_]+):' "$cmake_cache" || true
    fi
  } > "$manifest"
}

run_worker() {
  if (( $# != 1 )); then
    echo "Internal worker usage: --worker INPUT" >&2
    return 2
  fi
  if [[ -z ${SLURM_JOB_ID:-} ]]; then
    echo "The internal worker may only run inside a Slurm job" >&2
    return 2
  fi

  local -r tpic_scaling_worker_input=$1
  local -r tpic_scaling_worker_executable=${TPIC_SCALING_EXECUTABLE:-}
  local -r tpic_scaling_worker_profile=${TPIC_SCALING_PROFILE:-}
  local -r tpic_scaling_worker_profile_sha256=${TPIC_SCALING_PROFILE_SHA256:-}
  local -r tpic_scaling_worker_run_directory=$PWD
  local -r tpic_scaling_worker_restart_count=${SLURM_RESTART_COUNT:-0}
  local -r tpic_scaling_worker_attempt="TPIC_SCALING_ATTEMPT solver=${SOLVER} restart_count=${tpic_scaling_worker_restart_count}"
  echo "$tpic_scaling_worker_attempt"
  echo "$tpic_scaling_worker_attempt" >&2

  if [[ ! -f $tpic_scaling_worker_input ]]; then
    echo "Input file not found: ${tpic_scaling_worker_input}" >&2
    return 1
  fi
  if [[ ! -r $tpic_scaling_worker_profile ]]; then
    echo "Runtime profile is not readable: ${tpic_scaling_worker_profile:-<unset>}" >&2
    return 1
  fi
  if [[ ! -x $tpic_scaling_worker_executable ]]; then
    echo "Scaling executable is not executable: ${tpic_scaling_worker_executable:-<unset>}" >&2
    return 1
  fi

  require_nonnegative_integer "${SLURM_JOB_NUM_NODES:-}" "SLURM_JOB_NUM_NODES" || return 1
  require_nonnegative_integer "${SLURM_NTASKS:-}" "SLURM_NTASKS" || return 1
  local -r tpic_scaling_worker_nodes=$((10#${SLURM_JOB_NUM_NODES}))
  local -r tpic_scaling_worker_ranks=$((10#${SLURM_NTASKS}))
  if (( tpic_scaling_worker_nodes < 1 ||
        tpic_scaling_worker_ranks != tpic_scaling_worker_nodes * TASKS_PER_NODE )); then
    echo "Slurm allocation has ${tpic_scaling_worker_nodes} nodes and ${tpic_scaling_worker_ranks} ranks; expected $((tpic_scaling_worker_nodes * TASKS_PER_NODE)) ranks" >&2
    return 1
  fi
  validate_input "$tpic_scaling_worker_input" "$tpic_scaling_worker_nodes" || return 1

  if [[ -n $tpic_scaling_worker_profile_sha256 ]]; then
    if ! command -v sha256sum >/dev/null 2>&1; then
      echo "sha256sum is required to verify the submitted runtime profile" >&2
      return 1
    fi
    local tpic_scaling_worker_actual_profile_sha256
    tpic_scaling_worker_actual_profile_sha256=$(sha256sum "$tpic_scaling_worker_profile" | awk '{print $1}')
    readonly tpic_scaling_worker_actual_profile_sha256
    if [[ $tpic_scaling_worker_actual_profile_sha256 != "$tpic_scaling_worker_profile_sha256" ]]; then
      echo "Runtime profile changed after submission: ${tpic_scaling_worker_profile}" >&2
      return 1
    fi
  fi

  local tpic_scaling_worker_profile_status
  set +e +u
  if source "$tpic_scaling_worker_profile"; then
    tpic_scaling_worker_profile_status=0
  else
    tpic_scaling_worker_profile_status=$?
  fi
  readonly tpic_scaling_worker_profile_status
  set -euo pipefail
  if (( tpic_scaling_worker_profile_status != 0 )); then
    echo "Runtime profile failed with status ${tpic_scaling_worker_profile_status}: ${tpic_scaling_worker_profile}" >&2
    return "$tpic_scaling_worker_profile_status"
  fi
  cd -- "$tpic_scaling_worker_run_directory"

  echo "Scaling application: TPIC ECSIM"
  echo "Scaling executable: ${tpic_scaling_worker_executable}"
  echo "Scaling input: ${tpic_scaling_worker_input}"
  echo "Running on ${tpic_scaling_worker_nodes} nodes with ${tpic_scaling_worker_ranks} MPI ranks"

  system_srun "$tpic_scaling_worker_executable" \
    --run "$tpic_scaling_worker_input" --order "$PARTICLE_ORDER" \
    --precision "$PRECISION" --mass-matrix "$MASS_MATRIX" \
    --mass-apply "$MASS_APPLY" || return $?

  echo "TPIC_SCALING_COMPLETE solver=${SOLVER}"
}

if [[ ${1:-} == --worker ]]; then
  shift
  run_worker "$@"
  exit $?
fi

dry_run=0
select_all=0
exclude_nodes=""
qos=""
time_limit=""
solver=$SOLVER
declare -a range_args=()

while (( $# > 0 )); do
  case $1 in
  --all)
    select_all=1
    ;;
  --dry-run)
    dry_run=1
    ;;
  --exclude|-x)
    if (( $# < 2 )) || [[ -z $2 ]]; then
      echo "$1 requires a Slurm node list" >&2
      exit 2
    fi
    exclude_nodes=$2
    shift
    ;;
  --exclude=*)
    exclude_nodes=${1#*=}
    if [[ -z $exclude_nodes ]]; then
      echo "--exclude requires a Slurm node list" >&2
      exit 2
    fi
    ;;
  --qos)
    if (( $# < 2 )) || [[ -z $2 ]]; then
      echo "--qos requires a Slurm QOS name" >&2
      exit 2
    fi
    qos=$2
    shift
    ;;
  --qos=*)
    qos=${1#*=}
    if [[ -z $qos ]]; then
      echo "--qos requires a Slurm QOS name" >&2
      exit 2
    fi
    ;;
  --time)
    if (( $# < 2 )) || [[ -z $2 ]]; then
      echo "--time requires a Slurm time limit" >&2
      exit 2
    fi
    time_limit=$2
    shift
    ;;
  --time=*)
    time_limit=${1#*=}
    if [[ -z $time_limit ]]; then
      echo "--time requires a Slurm time limit" >&2
      exit 2
    fi
    ;;
  -h|--help)
    usage
    exit 0
    ;;
  --)
    shift
    range_args+=("$@")
    break
    ;;
  -*)
    echo "Unknown option: $1" >&2
    usage >&2
    exit 2
    ;;
  *)
    range_args+=("$1")
    ;;
  esac
  shift
done
set -- "${range_args[@]}"

if (( select_all )) && (( $# != 0 )); then
  echo "--all cannot be combined with a node range" >&2
  exit 2
fi
if (( ! select_all )) && (( $# < 1 || $# > 2 )); then
  echo "Select a node count/range or pass --all" >&2
  usage >&2
  exit 2
fi

min_nodes=1
max_nodes=2147483647
if (( ! select_all )); then
  for value in "$@"; do
    if [[ ! $value =~ ^[0-9]+$ ]]; then
      echo "Node counts must be positive integers: ${value}" >&2
      exit 2
    fi
  done
  min_nodes=$((10#$1))
  max_nodes=$min_nodes
  if (( $# == 2 )); then
    max_nodes=$((10#$2))
  fi
  if (( min_nodes < 1 || max_nodes < min_nodes )); then
    echo "Invalid node range: ${min_nodes}-${max_nodes}" >&2
    exit 2
  fi
fi

if [[ -z ${TPIC_SLURM_ACCOUNT:-} ]]; then
  echo "TPIC_SLURM_ACCOUNT is not set" >&2
  exit 2
fi
if [[ -z ${TPIC_PROFILE:-} ]]; then
  echo "TPIC_PROFILE is not set" >&2
  exit 2
fi
if [[ -z ${TPIC_EXECUTABLE:-} ]]; then
  echo "TPIC_EXECUTABLE is not set" >&2
  exit 2
fi

profile=$(absolute_path "$TPIC_PROFILE") || exit 1
source_executable=$(absolute_path "$TPIC_EXECUTABLE") || exit 1
expected_executable=tpic

if [[ ! -r $profile ]]; then
  echo "Runtime profile is not readable: ${profile}" >&2
  exit 1
fi
if [[ ! -x $source_executable ]]; then
  echo "TPIC executable is not executable: ${source_executable}" >&2
  exit 1
fi
if [[ $(basename -- "$source_executable") != "$expected_executable" ]]; then
  echo "TPIC_EXECUTABLE must name an executable called ${expected_executable}" >&2
  exit 1
fi

shopt -s nullglob
candidate_inputs=("${INPUT_DIR}/${INPUT_PREFIX}"N*.inp)
shopt -u nullglob
if (( ${#candidate_inputs[@]} == 0 )); then
  echo "No ${INPUT_PREFIX}N*.inp inputs found in ${INPUT_DIR}" >&2
  exit 1
fi
mapfile -t candidate_inputs < <(printf '%s\n' "${candidate_inputs[@]}" | sort -V)

declare -a selected_inputs=()
declare -a selected_nodes=()
declare -A selected_node_seen=()
filename_regex="^${INPUT_PREFIX}N([0-9]+)\\.inp$"
for input_file in "${candidate_inputs[@]}"; do
  filename=$(basename -- "$input_file")
  [[ $filename =~ $filename_regex ]] || continue
  nodes=$((10#${BASH_REMATCH[1]}))
  (( nodes >= min_nodes && nodes <= max_nodes )) || continue
  if [[ -n ${selected_node_seen[$nodes]+present} ]]; then
    echo "Duplicate ${nodes}-node input: ${filename}" >&2
    exit 1
  fi
  validate_input "$input_file" "$nodes"
  selected_inputs+=("$input_file")
  selected_nodes+=("$nodes")
  selected_node_seen[$nodes]=1
done

if (( ${#selected_inputs[@]} == 0 )); then
  echo "No ${INPUT_PREFIX}N*.inp inputs selected" >&2
  exit 1
fi

campaign_invariants=$(normalized_invariants "${selected_inputs[0]}")
campaign_grid=$(campaign_grid_signature "${selected_inputs[0]}")

for index in "${!selected_inputs[@]}"; do
  validate_campaign_input "${selected_inputs[$index]}" "${selected_nodes[$index]}" \
    "$campaign_invariants" "$campaign_grid"
done

timestamp=$(date +%Y%m%d_%H%M%S)
campaign_dir=${INPUT_DIR}/results_tpic_${timestamp}
snapshot_executable=${campaign_dir}/bin/${expected_executable}
snapshot_runtime=${campaign_dir}/runtime
snapshot_common=${snapshot_runtime}/runScalingCommon.sh
snapshot_system=${snapshot_runtime}/${SYSTEM}
snapshot_launcher=${snapshot_system}/${SCALING_KIND}/$(basename -- "$LAUNCHER_PATH")
manifest=${campaign_dir}/manifest.txt
effective_time_limit=${time_limit:-$DEFAULT_TIME_LIMIT}
effective_qos=${qos:-${SYSTEM_QOS:-}}
planned_nodes=$(IFS=,; printf '%s' "${selected_nodes[*]}")
profile_sha256=""
if command -v sha256sum >/dev/null 2>&1; then
  profile_sha256=$(sha256sum "$profile" | awk '{print $1}')
fi

if (( dry_run )); then
  echo "Validated ${#selected_inputs[@]} ${SCALING_KIND} input(s) for TPIC ECSIM on ${SYSTEM}."
  echo "Planned campaign directory: ${campaign_dir}"
  echo "Planned executable snapshot: ${snapshot_executable}"
  echo "Planned launcher snapshot: ${snapshot_launcher}"
else
  if ! command -v sbatch >/dev/null 2>&1; then
    echo "sbatch is not available; no campaign was created" >&2
    exit 1
  fi
  if ! mkdir "$campaign_dir"; then
    echo "Campaign directory already exists; retry in one second: ${campaign_dir}" >&2
    exit 1
  fi
  mkdir "${campaign_dir}/bin" "$snapshot_runtime" "$snapshot_system" \
    "${snapshot_system}/${SCALING_KIND}"
  cp -p -- "$source_executable" "$snapshot_executable"
  cp -p -- "$SCALING_DIR/runScalingCommon.sh" "$snapshot_common"
  cp -p -- "$SYSTEM_DIR"/*.sh "$snapshot_system/"
  cp -p -- "$LAUNCHER_PATH" "$snapshot_launcher"
  write_manifest_header "$manifest" "$source_executable" "$snapshot_executable" \
    "$profile" "$solver" "$planned_nodes" "$snapshot_common" \
    "$snapshot_system" "$snapshot_launcher" "$profile_sha256"
fi

export TPIC_SCALING_EXECUTABLE=$snapshot_executable
export TPIC_SCALING_PROFILE=$profile
export TPIC_SCALING_PROFILE_SHA256=$profile_sha256
export TPIC_SCALING_LAUNCHER_PATH=$snapshot_launcher

declare -a staged_inputs=()
declare -a run_directories=()

for index in "${!selected_inputs[@]}"; do
  input_file=${selected_inputs[$index]}
  nodes=${selected_nodes[$index]}
  run_dir=${campaign_dir}/N${nodes}
  staged_input=${run_dir}/$(basename -- "$input_file")
  run_directories+=("$run_dir")
  staged_inputs+=("$staged_input")

  if (( ! dry_run )); then
    mkdir "$run_dir"
    cp -p -- "$input_file" "$staged_input"
    validate_campaign_input "$staged_input" "$nodes" \
      "$campaign_invariants" "$campaign_grid"
    if command -v sha256sum >/dev/null 2>&1; then
      echo "input_N${nodes}_sha256=$(sha256sum "$staged_input" | awk '{print $1}')" >> "$manifest"
    else
      echo "input_N${nodes}_sha256=unavailable" >> "$manifest"
    fi
  fi
done

# Avoid a user login environment silently adding conflicting resource requests.
unset SBATCH_CPUS_PER_TASK SBATCH_GPUS_PER_TASK SBATCH_GPUS SBATCH_GRES

for index in "${!selected_inputs[@]}"; do
  nodes=${selected_nodes[$index]}
  tasks=$((nodes * TASKS_PER_NODE))
  run_dir=${run_directories[$index]}
  staged_input=${staged_inputs[$index]}
  sbatch_command=(
    sbatch
    --parsable
    --account="$TPIC_SLURM_ACCOUNT"
    --export=ALL
    "${SYSTEM_SBATCH_ARGS[@]}"
    --nodes="$nodes"
    --ntasks="$tasks"
    --ntasks-per-node="$TASKS_PER_NODE"
    --time="$effective_time_limit"
    --job-name="${SCALING_KIND}-tpic-N${nodes}"
    --chdir="$run_dir"
    --output="${campaign_dir}/stdout_N${nodes}_%j"
    --error="${campaign_dir}/stderr_N${nodes}_%j"
  )
  if [[ -n $effective_qos ]]; then
    sbatch_command+=(--qos="$effective_qos")
  fi
  if [[ -n $exclude_nodes ]]; then
    sbatch_command+=(--exclude="$exclude_nodes")
  fi
  sbatch_command+=("$snapshot_launcher" --worker "$staged_input")

  if (( dry_run )); then
    printf 'DRY RUN:'
    printf ' %q' "${sbatch_command[@]}"
    printf '\n'
  else
    job_id=$("${sbatch_command[@]}")
    echo "job_N${nodes}=${job_id}" >> "$manifest"
    echo "Submitted ${SCALING_KIND} TPIC N${nodes}: ${job_id}"
  fi
done

if (( dry_run )); then
  echo "Dry run complete; no files or jobs were created."
else
  sed -i 's/^submission_complete=false$/submission_complete=true/' "$manifest"
  echo "Campaign directory: ${campaign_dir}"
fi
