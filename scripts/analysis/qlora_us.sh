#!/bin/bash
#SBATCH --job-name=q06B
#SBATCH --output=../../logs/%j.out
#SBATCH --error=../../logs/%j.err
#SBATCH --time=01:00:00
#SBATCH --partition=nmes_gpu,gpu
#SBATCH --mem=20GB
#SBATCH --gres=gpu:1

set -e
nvidia-smi
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

uv run qlora_us.py --lang "en"