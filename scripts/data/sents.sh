#!/bin/bash
#SBATCH --job-name=ru-id
#SBATCH --output=../../logs/%j.out
#SBATCH --error=../../logs/%j.err
#SBATCH --time=10:00:00
# SBATCH --partition=cpu,nmes_cpu
#SBATCH --partition=nmes_gpu,gpu
#SBATCH --mem=20GB
#SBATCH --gres=gpu:1

nvidia-smi

uv run sents2.py