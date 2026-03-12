#!/bin/bash
#SBATCH --job-name=blokus-bench
#SBATCH --partition=gpu-debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=%x-%j.out

module load python3.11-anaconda/2024.02-1 cuda/12.4.1

cd ~/projects/blokus_rl
source venv/bin/activate

echo "Job started: $(date)"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

python3 scripts/benchmark_vacc.py

echo "Job finished: $(date)"
