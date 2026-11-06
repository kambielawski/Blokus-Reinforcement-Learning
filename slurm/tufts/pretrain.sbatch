#!/bin/bash
# Imitation pretraining: game generation on the CPUs, network training on the GPU.
#   source slurm/tufts/env.sh
#   sbatch --partition="$BLOKUS_PARTITION" --output="$BLOKUS_SCRATCH/logs/%x-%j.out" \
#       --export=ALL,OUT=imitation-5b.pt,PRETRAIN_ARGS="--games 20000 --epochs 4" \
#       slurm/tufts/pretrain.sbatch
#
#SBATCH --job-name=blokus-pretrain
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=28
#SBATCH --mem=128G
#SBATCH --time=06:00:00

set -uo pipefail
source "${SLURM_SUBMIT_DIR}/slurm/tufts/env.sh"
blokus_modules
source "$BLOKUS_VENV/bin/activate"
cd "$BLOKUS_REPO"

echo "Job $SLURM_JOB_ID on $(hostname), commit $(git rev-parse --short HEAD)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
mkdir -p "$BLOKUS_SCRATCH/pretrain"
python -u scripts/pretrain_imitation.py --device cuda --workers "$SLURM_CPUS_PER_TASK" \
    --out "$BLOKUS_SCRATCH/pretrain/${OUT:?set OUT}" ${PRETRAIN_ARGS:-}
