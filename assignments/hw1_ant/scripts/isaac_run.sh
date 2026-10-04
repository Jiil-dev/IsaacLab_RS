#!/usr/bin/env bash
# Run an Isaac Lab python script inside the course conda environment and log its output.
#
# Usage (from anywhere):
#   [GPU=<index>] bash assignments/hw1_ant/scripts/isaac_run.sh <log_file> <script.py> [script args...]
#
# Notes:
#   - GPU=<index> makes only that GPU visible to CUDA; the script then runs on ``--device cuda:0`` (the default).
#     On this machine the physics diverges (NaN within a few steps) when the second GPU is used while both GPUs are
#     visible (PCIe x1 link with IOMMU, peer-to-peer errors). Isolated, both GPUs give identical results.
#     Omniverse then prints harmless warnings about CUDA and Vulkan device enumeration.
#   - Python output is unbuffered, otherwise prints are lost when Isaac Sim closes the process.
#   - Every Isaac Sim process needs about 4-5 GB of RAM and this machine has 31 GB. A launch waits until fewer than
#     MAX_ISAAC_PROCS (default 4) Isaac scripts run and at least MIN_FREE_GB (default 6) GB are available.
#     Set NO_SLOT=1 for light scripts that do not start the simulator.
set -o pipefail
if [[ -n "${GPU:-}" ]]; then
    export CUDA_VISIBLE_DEVICES="${GPU}"
fi
MAX_ISAAC_PROCS="${MAX_ISAAC_PROCS:-4}"
MIN_FREE_GB="${MIN_FREE_GB:-6}"
ISAAC_SCRIPTS='lerobot-arena/bin/python .*(train|play|play_one_episode|eval|collect_rollouts|test_equivalence|check_setup)\.py'

if [[ $# -lt 2 ]]; then
    echo "Usage: bash isaac_run.sh <log_file> <script.py> [args...]" >&2
    exit 2
fi
LOG_FILE="$1"
shift

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
# CONDA_EXE is set by conda's shell hook; `conda info --base` is avoided because it can print plugin errors.
if [[ -n "${CONDA_EXE:-}" ]]; then
    CONDA_BASE="$(dirname "$(dirname "${CONDA_EXE}")")"
else
    CONDA_BASE="${HOME}/miniconda3"
fi
# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate lerobot-arena || { echo "conda env 'lerobot-arena' not found" >&2; exit 2; }

cd "${REPO_ROOT}" || exit 2
mkdir -p "$(dirname "${LOG_FILE}")"
export PYTHONUNBUFFERED=1

if [[ -n "${NO_SLOT:-}" ]]; then
    ./isaaclab.sh -p "$@" > "${LOG_FILE}" 2>&1
    exit $?
fi

# One launcher at a time checks the resources; the lock is held until the new process has allocated its memory.
# PRIORITY=high launches leave a marker while waiting; PRIORITY=low launches yield as long as such a marker exists.
LOCK_DIR="${REPO_ROOT}/logs/.isaac_slots"
mkdir -p "${LOCK_DIR}"
marker="${LOCK_DIR}/high_$$"
if [[ "${PRIORITY:-}" == "high" ]]; then
    touch "${marker}"
fi
exec 9> "${LOCK_DIR}/launch.lock"
while true; do
    flock 9
    running=$(pgrep -fc "${ISAAC_SCRIPTS}" || true)
    free_gb=$(awk '/MemAvailable/ {print int($2 / 1048576)}' /proc/meminfo)
    high_waiting=$(find "${LOCK_DIR}" -name 'high_*' | wc -l)
    if (( running < MAX_ISAAC_PROCS && free_gb >= MIN_FREE_GB )) \
        && [[ "${PRIORITY:-}" != "low" || ${high_waiting} -eq 0 ]]; then
        break
    fi
    flock -u 9
    sleep 15
done
rm -f "${marker}"
./isaaclab.sh -p "$@" > "${LOG_FILE}" 2>&1 9>&- &
pid=$!
sleep 40
flock -u 9
wait "${pid}"
