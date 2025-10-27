#!/bin/bash
#SBATCH --job-name=desc
#SBATCH --output=../../logs/%j.out
#SBATCH --error=../../logs/%j.err
#SBATCH --time=01:00:00
# SBATCH --partition=nmes_gpu,gpu,interruptible_gpu
#SBATCH --partition=nmes_cpu,cpu,interruptible_gpu
#SBATCH --mem=20GB
#sSBATCH --gres=gpu:1

nvidia-smi

uv run descriptives.py

