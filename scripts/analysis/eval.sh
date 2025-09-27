#!/bin/bash
#SBATCH --job-name=eval
#SBATCH --output=../../logs/%j.out
#SBATCH --error=../../logs/%j.err
#SBATCH --time=01:00:00
#SBATCH --partition=gpu,nmes_gpu,interruptible_gpu
#SBATCH --mem=20GB
#SBATCH --gres=gpu:1

nvidia-smi

uv run eval.py