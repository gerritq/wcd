#!/bin/bash
#SBATCH --job-name=q-l8-c
#SBATCH --output=../../logs/%j.out
#SBATCH --error=../../logs/%j.err
#SBATCH --time=05:30:00
#SBATCH --partition=nmes_gpu,gpu
#SBATCH --mem=20GB
#SBATCH --gres=gpu:1

nvidia-smi

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

LANGUAGES=("en")
MODELS=("qwen3_06b")
TRIALS=1
for LANGUAGE in "${LANGUAGES[@]}"; do
  for MODEL in "${MODELS[@]}"; do

    uv run qlora2_hp.py \
      --lang "$LANGUAGE" \
      --model "$MODEL" \
      --trials $TRIALS
  done
done