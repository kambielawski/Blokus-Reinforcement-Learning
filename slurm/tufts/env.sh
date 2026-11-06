# Settings shared by the Tufts scripts. Source this; do not run it.

# Checked with sinfo on 2026-11-05. Use "preempt" only for work that may be killed
# at any time; the job script resumes either way.
export BLOKUS_PARTITION="gpu"

export BLOKUS_REPO="$HOME/dev/Blokus-Reinforcement-Learning"
export BLOKUS_SCRATCH="/cluster/scratch/$USER/blokus"
export BLOKUS_VENV="$BLOKUS_SCRATCH/venv"      # Home is nearly full; setup_env.sh rebuilds it
export UV_CACHE_DIR="/cluster/scratch/$USER/uv-cache"
export WANDB_DIR="$BLOKUS_SCRATCH/wandb"

blokus_modules() {
    module load uv/0.9.18 python/3.13.2
}
