#!/bin/bash
# Kam's pass mark for one model: 800 sims and raw policy against every bot, 200 games each.
#   sbatch --partition=batch --output="$BLOKUS_SCRATCH/logs/%x-%j.out" \
#       --export=ALL,NAME=i50,CKPT=/path/to/checkpoint.pt slurm/tufts/gauntlet.sbatch
#
#SBATCH --job-name=blokus-gauntlet
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=12:00:00

set -uo pipefail
source "${SLURM_SUBMIT_DIR}/slurm/tufts/env.sh"
blokus_modules
source "$BLOKUS_VENV/bin/activate"
cd "$BLOKUS_REPO"

ARGS=(--pairs-per-matchup 100 --workers "$SLURM_CPUS_PER_TASK"
      --out "$BLOKUS_SCRATCH/gauntlet/${NAME:?set NAME}"
      --player "model_s800=mcts:ckpt=${CKPT:?set CKPT},sims=800"
      --player "model_raw=raw:ckpt=$CKPT")
for bot in random size space1 space2 mobility lookahead; do
    ARGS+=(--player "$bot=$bot" --matchup "model_s800,$bot" --matchup "model_raw,$bot")
done
# Strongest fixed reference: the bots alone can't separate good from very good.
ARGS+=(--player "imitation_s800=mcts:ckpt=$BLOKUS_SCRATCH/pretrain/imitation-5b.pt,sims=800"
       --matchup "model_s800,imitation_s800")
echo "Job $SLURM_JOB_ID on $(hostname), commit $(git rev-parse --short HEAD), model $CKPT"
python -u scripts/ladder.py "${ARGS[@]}"
