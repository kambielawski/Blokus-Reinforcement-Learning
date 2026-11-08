# Settings shared by the Tufts scripts. Source this; do not run it.

# Checked with sinfo on 2026-11-05. Use "preempt" only for work that may be killed
# at any time; the job script resumes either way.
export BLOKUS_PARTITION="gpu"

export BLOKUS_REPO="$HOME/dev/Blokus-Reinforcement-Learning"
export BLOKUS_SCRATCH="/cluster/scratch/$USER/blokus"
export BLOKUS_VENV="$BLOKUS_SCRATCH/venv"      # Home is nearly full; setup_env.sh rebuilds it
export UV_CACHE_DIR="/cluster/scratch/$USER/uv-cache"
export WANDB_DIR="$BLOKUS_SCRATCH/wandb"
# Offline until `wandb login` has been run on the cluster; `wandb sync` uploads later.
export WANDB_MODE="${WANDB_MODE:-offline}"

blokus_modules() {
    module load uv/0.9.18 python/3.13.2
}

# The user quota on scratch is shared with other data (1 TB, met at 992 GB of
# du usage on 2026-11-08). train.py skips a save that would not fit, using the
# usage that usage_probe.sh records.
export BLOKUS_USER_SCRATCH="/cluster/scratch/$USER"
export BLOKUS_QUOTA_BYTES="1000000000000"
export BLOKUS_USAGE_FILE="$BLOKUS_SCRATCH/.usage.json"
