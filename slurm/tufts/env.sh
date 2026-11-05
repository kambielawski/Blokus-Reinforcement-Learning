# Settings shared by the Tufts scripts. Source this; do not run it.

# >>> CONFIRM BEFORE FIRST USE <<<
# Partition. "gpu" comes from notes, not from a check of this account. Use "preempt"
# only with a run you are happy to have killed; the job script resumes either way.
export BLOKUS_PARTITION="gpu"

export BLOKUS_REPO="$HOME/dev/Blokus-Reinforcement-Learning"
export BLOKUS_VENV="$HOME/venvs/blokus"              # Home: scratch purges unaccessed files
export BLOKUS_SCRATCH="/cluster/scratch/$USER/blokus"
export BLOKUS_PYTHON_MODULE="anaconda/2024.06"       # Only used to create the venv
