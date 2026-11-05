#!/bin/bash
# One-time setup on a Tufts login node. Run from the repo root:
#   bash slurm/tufts/setup_env.sh
# Afterward run `wandb login` once so compute nodes can log.
set -euo pipefail
source "$(dirname "$0")/env.sh"

module load "$BLOKUS_PYTHON_MODULE"
mkdir -p "$(dirname "$BLOKUS_VENV")" "$BLOKUS_SCRATCH"
python -m venv "$BLOKUS_VENV"
source "$BLOKUS_VENV/bin/activate"
pip install --upgrade pip
pip install torch numpy pyyaml pillow wandb cython setuptools pytest
pip install -e .
python setup_cython.py build_ext --inplace

python - <<'PY'
import torch
from blokus.engine import game_state
print('torch', torch.__version__, '| cuda build', torch.version.cuda,
      '| cython engine', game_state._USE_CYTHON)
PY
python -m pytest tests/test_play.py -q
echo "Done. venv: $BLOKUS_VENV  data: $BLOKUS_SCRATCH"
