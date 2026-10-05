#!/usr/bin/env bash
# Team self-evaluation of the course announcement: run the official play_one_episode.py (seed 24, 100 envs) on the
# team's own test terrains and collect the printed mean and std of the episode reward (original Isaac-Ant-v0 reward,
# which none of the HW1 tasks changes).
#
# Usage (from the repository root):
#   [GPU=<index>] bash assignments/hw1_ant/scripts/official_eval.sh <label> <task prefix> <checkpoint> [envs...]
# Example:
#   bash assignments/hw1_ant/scripts/official_eval.sh E2_seed42 Isaac-Ant-WideScan \
#       assignments/hw1_ant/checkpoints/E2_seed42/model.pt
#
# Envs default to "Flat T1 T2 T3 T4 T5 T6"; Flat is the original scene (<prefix>-v0), the others <prefix>-<env>-v0.
# Logs go to assignments/hw1_ant/results/official/<label>_<env>.log and the table to .../<label>.md.
set -o pipefail
if [[ $# -lt 3 ]]; then
    echo "Usage: bash official_eval.sh <label> <task prefix> <checkpoint> [envs...]" >&2
    exit 2
fi
LABEL="$1"
PREFIX="$2"
CHECKPOINT="$3"
shift 3
ENVS=("$@")
if [[ ${#ENVS[@]} -eq 0 ]]; then
    ENVS=(Flat T1 T2 T3 T4 T5 T6)
fi
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_DIR="$HERE/../results/official"
mkdir -p "$OUT_DIR"
TABLE="$OUT_DIR/$LABEL.md"
{
    echo "# Official play_one_episode.py: $LABEL"
    echo
    echo "Checkpoint \`$CHECKPOINT\`, \`--seed 24 --num_envs 100\`. Mean and std over the 100 environments."
    echo
    echo "| env | task | reward mean | reward std | steps mean |"
    echo "|---|---|---|---|---|"
} > "$TABLE"
for env in "${ENVS[@]}"; do
    if [[ "$env" == "Flat" ]]; then task="$PREFIX-v0"; else task="$PREFIX-$env-v0"; fi
    log="$OUT_DIR/${LABEL}_${env}.log"
    NO_SLOT=1 bash "$HERE/isaac_run.sh" "$log" scripts/reinforcement_learning/rsl_rl/play_one_episode.py \
        --task "$task" --seed 24 --num_envs 100 --checkpoint "$CHECKPOINT" --headless
    reward=$(grep -a "\[RESULT\] Episode reward total" "$log" | tail -1)
    steps=$(grep -a "\[RESULT\] Episode steps" "$log" | tail -1)
    mean=$(sed -n 's/.*mean=\([-0-9.]*\).*/\1/p' <<< "$reward")
    std=$(sed -n 's/.*std=\([-0-9.]*\).*/\1/p' <<< "$reward")
    smean=$(sed -n 's/.*mean=\([-0-9.]*\).*/\1/p' <<< "$steps")
    echo "| $env | \`$task\` | ${mean:-failed} | ${std:-failed} | ${smean:-failed} |" >> "$TABLE"
    echo "[official_eval] $LABEL $env: mean=${mean:-failed} std=${std:-failed}"
done
echo "[official_eval] wrote $TABLE"
