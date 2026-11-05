#!/bin/bash
# Blokus training on Tufts. Resumes by itself after preemption or the 2-day limit.
#
# Usage, from the repo root on a login node (partition comes from env.sh):
#   source slurm/tufts/env.sh
#   mkdir -p "$BLOKUS_SCRATCH/logs"
#   RUN=tufts-v1 CONFIG=configs/tufts-v1.yaml TARGET=300 INIT=/path/to/imitation.pt \
#     sbatch --partition="$BLOKUS_PARTITION" --output="$BLOKUS_SCRATCH/logs/%x-%j.out" \
#     slurm/tufts/train.sbatch
#
# The log directory must exist before sbatch, or the job fails without a message.
#
#SBATCH --job-name=blokus-train
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=28
#SBATCH --mem=96G
#SBATCH --time=2-00:00:00
#SBATCH --open-mode=append
#SBATCH --requeue
# USR1 five minutes before the limit: finish the iteration, exit, resubmit.
#SBATCH --signal=B:USR1@300

set -uo pipefail
source "${SLURM_SUBMIT_DIR}/slurm/tufts/env.sh"
source "$BLOKUS_VENV/bin/activate"
cd "$BLOKUS_REPO"

RUN_DIR="$BLOKUS_SCRATCH/${RUN:?set RUN}"
mkdir -p "$RUN_DIR"

echo "Job $SLURM_JOB_ID restart ${SLURM_RESTART_COUNT:-0} on $(hostname) at $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
echo "Commit: $(git rev-parse --short HEAD)"

ARGS=(--config "${CONFIG:?set CONFIG}" --save-dir "$RUN_DIR" --target-iteration "${TARGET:?set TARGET}")
if [ -f "$RUN_DIR/checkpoint_latest.pt" ]; then
    ARGS+=(--resume latest)     # The replay buffer in RUN_DIR comes along
elif [ -n "${INIT:-}" ]; then
    ARGS+=(--init-weights "$INIT")
fi

# In the background so bash can act on the signal while Python runs.
python3 -u scripts/train.py "${ARGS[@]}" &
TRAIN_PID=$!
trap 'kill -USR1 "$TRAIN_PID" 2>/dev/null' USR1
trap 'kill -TERM "$TRAIN_PID" 2>/dev/null' TERM

# wait returns early when a trapped signal arrives, so loop until Python is gone
while kill -0 "$TRAIN_PID" 2>/dev/null; do
    wait "$TRAIN_PID"
    STATUS=$?
done
echo "train.py exited with status $STATUS at $(date)"

if [ "$STATUS" -eq 85 ]; then
    scontrol requeue "$SLURM_JOB_ID"
fi
exit "$STATUS"
