#!/bin/bash
# CPU strength ladder. Pass ladder.py arguments in LADDER_ARGS and an output dir name in NAME.
#   sbatch --partition=batch --output="$BLOKUS_SCRATCH/logs/%x-%j.out" \
#       --export=ALL,NAME=base,LADDER_ARGS="--pairs-per-matchup 40 --player ..." slurm/tufts/ladder.sbatch
#
#SBATCH --job-name=blokus-ladder
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=12:00:00

set -uo pipefail
source "${SLURM_SUBMIT_DIR}/slurm/tufts/env.sh"
blokus_modules
source "$BLOKUS_VENV/bin/activate"
cd "$BLOKUS_REPO"

echo "Job $SLURM_JOB_ID on $(hostname), commit $(git rev-parse --short HEAD)"
python -u scripts/ladder.py --workers "$SLURM_CPUS_PER_TASK" \
    --out "$BLOKUS_SCRATCH/ladder/${NAME:?set NAME}" $LADDER_ARGS
