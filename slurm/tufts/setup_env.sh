#!/bin/bash
# Setup on a Tufts login node; safe to rerun, for example after a scratch purge.
#   cd ~/dev/Blokus-Reinforcement-Learning && bash slurm/tufts/setup_env.sh
set -euo pipefail
source "$(dirname "$0")/env.sh"
blokus_modules

mkdir -p "$BLOKUS_SCRATCH/logs" "$WANDB_DIR"
[ -d "$BLOKUS_VENV" ] || uv venv --python 3.13 "$BLOKUS_VENV"
source "$BLOKUS_VENV/bin/activate"

# GPU nodes run driver 575, which the default (cu130) torch build cannot use.
uv pip install "torch==2.7.1+cu126" --index-url https://download.pytorch.org/whl/cu126 \
    --extra-index-url https://pypi.org/simple --index-strategy unsafe-best-match
uv pip install numpy pyyaml pillow wandb cython setuptools pytest
python setup_cython.py build_ext --inplace

python -c "import torch; from blokus.engine import game_state as g; print('torch', torch.__version__, 'cuda build', torch.version.cuda, 'cython engine', g._USE_CYTHON)"
echo "Done. venv: $BLOKUS_VENV  data: $BLOKUS_SCRATCH"
