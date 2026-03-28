#!/bin/bash
#SBATCH --job-name=blokus-az
#SBATCH --partition=gpu-preempt
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=blokus-az-%j.out

# ---- Blokus AlphaZero Training on VACC ----
# Usage:
#   sbatch slurm/train_alphazero.sh                              # full.yaml defaults
#   TRAIN_CONFIG=configs/small.yaml sbatch slurm/train_alphazero.sh  # small network
#   TRAIN_ARGS="--sims 200" sbatch slurm/train_alphazero.sh      # extra CLI overrides
#   RESUME=1 sbatch slurm/train_alphazero.sh                     # resume from latest

set -euo pipefail

PROJECT_DIR="$HOME/projects/blokus_rl"
cd "$PROJECT_DIR"
source venv/bin/activate

echo "================================================"
echo "Blokus AlphaZero Training"
echo "================================================"
echo "Job ID:    $SLURM_JOB_ID"
echo "Node:      $(hostname)"
echo "CPUs:      $SLURM_CPUS_PER_TASK"
echo "Started:   $(date)"
echo "Python:    $(python3 --version)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "GPU: N/A"
echo "================================================"

# Config file (default: configs/scale-v1.yaml)
CONFIG="${TRAIN_CONFIG:-configs/scale-v1.yaml}"

# Extra CLI args
ARGS="${TRAIN_ARGS:-}"

# Resume flag
if [ "${RESUME:-0}" = "1" ]; then
    ARGS="$ARGS --resume"
fi

echo "Config: $CONFIG"
echo "Extra args: $ARGS"
echo ""

python3 -u scripts/train.py --config "$CONFIG" $ARGS

echo ""
echo "Finished: $(date)"
