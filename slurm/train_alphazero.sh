#!/bin/bash
#SBATCH --job-name=blokus-az
#SBATCH --partition=gpu-preempt
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=blokus-az-%j.out

# ---- Blokus AlphaZero Training on VACC ----
# Usage:
#   sbatch slurm/train_alphazero.sh                    # defaults
#   sbatch --time=12:00:00 slurm/train_alphazero.sh    # longer run
#
# Override training args via TRAIN_ARGS env var:
#   TRAIN_ARGS="--iterations 200 --sims 200" sbatch slurm/train_alphazero.sh

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

# Default training arguments
DEFAULT_ARGS="--iterations 100 --games-per-iter 10 --sims 100 --num-workers 4 --wandb"

# Allow override via environment variable
ARGS="${TRAIN_ARGS:-$DEFAULT_ARGS}"

echo "Training args: $ARGS"
echo ""

python3 -u scripts/train.py $ARGS

echo ""
echo "Finished: $(date)"
