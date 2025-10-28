#!/bin/bash
#SBATCH --job-name=slm
#SBATCH --output=../../logs/%j.out
#SBATCH --error=../../logs/%j.err
#SBATCH --time=01:30:00
#SBATCH --partition=nmes_gpu,gpu
#SBATCH --mem=20GB
#SBATCH --gres=gpu:1

nvidia-smi

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# TIME: llama8b takes about 45 mins

# VARs
#LANGUAGES=("en" "nl" "no" "it" "pt" "ro" "ru" "uk" "bg" "id" "vi" "tr")
LANGUAGES=("en")
MODELS=("llama3_8b")

# HPs
SHOTS=(0 1)

for LANGUAGE in "${LANGUAGES[@]}"; do
  for MODEL in "${MODELS[@]}"; do
    for SHOT in "${SHOTS[@]}"; do
        uv run slm.py \
        --lang "$LANGUAGE" \
        --model "$MODEL" \
        --shots "$SHOT"
    done
  done
done