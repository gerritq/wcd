#!/bin/bash
#SBATCH --job-name=translation-eval
#SBATCH --output=../../logs/%j.out
#SBATCH --error=../../logs/%j.err
#SBATCH --time=00:30:00
#SBATCH --partition=gpu,nmes_gpu,interruptible_gpu
#SBATCH --mem=10GB
#SBATCH --gres=gpu:1
#SBATCH --exclude=erc-hpc-comp050 

uv run trans_eval.py
